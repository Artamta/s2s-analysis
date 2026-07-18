#!/usr/bin/env python3
"""Run one memory-bounded 42-day NeuralGCM pilot and standardize it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import pickle
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import jax
import neuralgcm
import numpy as np
import xarray as xr


COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))

import neuralgcm_run_contract as run_contract  # noqa: E402
import pilot_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=run_contract.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for package, expected in config["runtime"]["expected_versions"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(
                f"runtime version mismatch for {package}: expected {expected}, found {actual}"
            )
        result[package] = actual
    return result


def verify_file(path: Path, size: int | None, digest: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"size mismatch for {path}")
    if pilot_contract.sha256_file(path) != digest:
        raise ValueError(f"SHA256 mismatch for {path}")


def existing_output_is_valid(output: Path, manifest: Path) -> bool:
    if not output.is_file() or not manifest.is_file():
        return False
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        record.get("status") == "passed"
        and record.get("output_sha256") == pilot_contract.sha256_file(output)
    )


def pressure_level_indices(model: Any, levels: list[int]) -> list[int]:
    available = np.asarray(model.data_coords.vertical.centers)
    indices: list[int] = []
    for level in levels:
        matches = np.flatnonzero(available == level)
        if matches.size != 1:
            raise ValueError(f"checkpoint does not have exactly one {level} hPa level")
        indices.append(int(matches[0]))
    return indices


def make_post_processor(
    model: Any, config: dict[str, Any]
) -> tuple[Callable[..., Any], list[str]]:
    product = config["model"]["product"]
    decoded_field = config["model"]["decoded_field"]
    if product == "daily_tp":
        def select_tp(state: Any, forcings: Any) -> Any:
            return model.decode(state, forcings)[decoded_field]

        return select_tp, ["tp"]

    levels = [int(value) for value in config["model"]["pressure_levels_hpa"]]
    indices = pressure_level_indices(model, levels)

    def select_temperature_levels(state: Any, forcings: Any) -> tuple[Any, ...]:
        temperature = model.decode(state, forcings)[decoded_field]
        return tuple(temperature[index] for index in indices)

    return select_temperature_levels, [f"t{level}" for level in levels]


def as_time_lon_lat(value: Any, frames: int, nlon: int, nlat: int) -> np.ndarray:
    array = np.asarray(jax.device_get(value), dtype=np.float32)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    expected = (frames, nlon, nlat)
    if array.shape != expected:
        raise ValueError(f"expected decoded shape {expected}, found {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("decoded trajectory contains non-finite values")
    return array


def aggregate_member(
    predictions: Any,
    product: str,
    fields: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    forecast = config["forecast"]
    frames = int(forecast["unroll_steps"])
    nlon = int(config["model"]["native_longitude_count"])
    nlat = int(config["model"]["native_latitude_count"])
    lead_days = int(forecast["lead_days"])
    frames_per_day = 24 // int(forecast["output_interval_hours"])
    diagnostics: dict[str, float] = {}

    if product == "daily_tp":
        cumulative = as_time_lon_lat(predictions, frames, nlon, nlat)
        increments = np.diff(cumulative.astype(np.float64), axis=0)
        minimum_increment = float(increments.min())
        tolerance = float(forecast["negative_increment_tolerance_m"])
        if minimum_increment < -tolerance:
            raise ValueError(
                f"cumulative TP decreased by {minimum_increment} m"
            )
        boundaries = cumulative[::frames_per_day]
        if boundaries.shape[0] != lead_days + 1:
            raise ValueError("daily TP boundary count is incorrect")
        daily = np.diff(boundaries.astype(np.float64), axis=0) * 1000.0
        if float(daily.min()) < -tolerance * 1000.0:
            raise ValueError("daily TP is negative beyond tolerance")
        daily = np.maximum(daily, 0.0).astype(np.float32)
        diagnostics["minimum_six_hour_increment_m"] = minimum_increment
        return {"tp": daily.transpose(0, 2, 1)}, diagnostics

    if not isinstance(predictions, (tuple, list)) or len(predictions) != len(fields):
        raise ValueError("pressure-temperature post-processor returned wrong structure")
    result: dict[str, np.ndarray] = {}
    for field, prediction in zip(fields, predictions, strict=True):
        snapshots = as_time_lon_lat(prediction, frames, nlon, nlat)
        endpoints = snapshots[1:]
        expected_endpoints = lead_days * frames_per_day
        if endpoints.shape[0] != expected_endpoints:
            raise ValueError("daily temperature endpoint count is incorrect")
        daily = endpoints.reshape(
            lead_days, frames_per_day, nlon, nlat
        ).mean(axis=1)
        result[field] = daily.transpose(0, 2, 1).astype(np.float32)
    return result, diagnostics


def target_grid() -> spherical_harmonic.Grid:
    return spherical_harmonic.Grid(
        longitude_nodes=240,
        latitude_nodes=121,
        latitude_spacing="equiangular_with_poles",
        longitude_offset=0.0,
    )


def nearest_indices(values: np.ndarray, expected: np.ndarray) -> list[int]:
    indices: list[int] = []
    for target in expected:
        index = int(np.argmin(np.abs(values - target)))
        if abs(float(values[index]) - float(target)) > 1e-6:
            raise ValueError(f"target coordinate {target} is absent")
        indices.append(index)
    return indices


def regrid_to_common(
    native: xr.Dataset, model: Any, product: str
) -> xr.Dataset:
    destination = target_grid()
    if product == "daily_tp":
        regridder = horizontal_interpolation.ConservativeRegridder(
            model.data_coords.horizontal, destination, skipna=False
        )
    else:
        regridder = horizontal_interpolation.BilinearRegridder(
            model.data_coords.horizontal, destination
        )
    global_data = xarray_utils.regrid(native, regridder)
    expected_lat = np.arange(39.0, -0.1, -1.5, dtype=np.float64)
    expected_lon = np.arange(60.0, 99.1, 1.5, dtype=np.float64)
    lat_indices = nearest_indices(global_data.latitude.values, expected_lat)
    lon_indices = nearest_indices(global_data.longitude.values, expected_lon)
    common = global_data.isel(latitude=lat_indices, longitude=lon_indices)
    common = common.assign_coords(latitude=expected_lat, longitude=expected_lon)
    if common.sizes["latitude"] != 27 or common.sizes["longitude"] != 27:
        raise ValueError("common output is not 27x27")
    return common


def period_coordinates(init_time: np.datetime64, lead_days: int) -> dict[str, Any]:
    lead = np.arange(1, lead_days + 1, dtype=np.int16)
    period_start = init_time + np.arange(lead_days).astype("timedelta64[D]")
    period_end = init_time + np.arange(1, lead_days + 1).astype("timedelta64[D]")
    return {
        "lead_day": lead,
        "valid_time": ("lead_day", period_end),
        "forecast_period_start": ("lead_day", period_start),
        "forecast_period_end": ("lead_day", period_end),
        "forecast_period_bounds": (
            ("lead_day", "bounds"), np.stack([period_start, period_end], axis=1)
        ),
        "bounds": np.asarray([0, 1], dtype=np.int8),
        "init_time": init_time,
        "forecast_reference_time": init_time,
        "information_cutoff_time": init_time,
    }


def build_output(
    member_values: list[dict[str, np.ndarray]],
    model: Any,
    config: dict[str, Any],
) -> xr.Dataset:
    product = config["model"]["product"]
    fields = config["model"]["retained_fields"]
    nlon = int(config["model"]["native_longitude_count"])
    nlat = int(config["model"]["native_latitude_count"])
    lead_days = int(config["forecast"]["lead_days"])
    native = xr.Dataset(
        data_vars={
            field: (
                ("member", "lead_day", "latitude", "longitude"),
                np.stack([values[field] for values in member_values]),
            )
            for field in fields
        },
        coords={
            "member": np.arange(len(member_values), dtype=np.int16),
            "lead_day": np.arange(1, lead_days + 1, dtype=np.int16),
            "latitude": np.rad2deg(model.data_coords.horizontal.latitudes),
            "longitude": np.rad2deg(model.data_coords.horizontal.longitudes),
        },
    )
    if native.sizes["longitude"] != nlon or native.sizes["latitude"] != nlat:
        raise ValueError("native output grid differs from checkpoint contract")
    common = regrid_to_common(native, model, product)
    init_time = np.datetime64(config["case"]["init_time"])
    common = common.assign_coords(**period_coordinates(init_time, lead_days))
    run_mode = config.get("run_mode", "pilot")
    common.attrs.update(
        {
            "model": config["model"]["name"],
            "run_label": config["run_label"],
            "init_date": str(init_time.astype("datetime64[D]")),
            "benchmark_mode": "strict_information_matched_00utc",
            "strict_operational": "true",
            "information_cutoff_matches_issue_time": "true",
            "valid_time_role": "period_end",
            "input_source": config["source"]["name"],
            "forcing_policy": config["initial_conditions"]["forcing_policy"],
            "ensemble": f"{config['forecast']['member_count']} stochastic NeuralGCM member(s)",
            "domain": "exact physics grid: 39-0 N, 60-99 E, 1.5 degrees",
            "checkpoint_sha256": config["model"]["checkpoint_sha256"],
            "paper_status": (
                "five-year production forecast"
                if run_mode == "production"
                else "42-day pilot only; not for skill scores"
            ),
        }
    )
    common["valid_time"].attrs.update(
        {
            "long_name": "forecast valid time",
            "bounds": "forecast_period_bounds",
            "representation": "period_end",
        }
    )
    if product == "daily_tp":
        common["tp"].attrs.update(
            {
                "long_name": "NeuralGCM total precipitation",
                "units": "mm day-1",
                "cell_methods": "time: sum (24-hour period)",
                "source_diagnostic": "precipitation_cumulative_mean",
                "conversion": "1000 times consecutive 24-hour cumulative differences",
                "horizontal_regrid": "Dinosaur conservative",
            }
        )
    else:
        for field, level in zip(fields, config["model"]["pressure_levels_hpa"], strict=True):
            common[field].attrs.update(
                {
                    "long_name": f"air temperature at {level} hPa",
                    "units": "K",
                    "cell_methods": "time: mean (four 6-hour endpoint snapshots)",
                    "pressure_level_hpa": int(level),
                    "horizontal_regrid": "Dinosaur bilinear",
                    "t2m_equivalence": "none; this variable must not be labeled or scored as T2M",
                }
            )
    return common.transpose("member", "lead_day", "latitude", "longitude", ...)


def run(config: dict[str, Any], force: bool) -> tuple[Path, Path]:
    output_path = run_contract.storage_path(config, "output")
    manifest_path = run_contract.storage_path(config, "output_manifest")
    run_mode = config.get("run_mode", "pilot")
    report_path = (
        run_contract.storage_path(config, "pilot_report")
        if run_mode == "pilot"
        else None
    )
    if not force and existing_output_is_valid(output_path, manifest_path):
        print(f"validated existing 42-day output: {output_path}", flush=True)
        return output_path, manifest_path
    if (output_path.exists() or manifest_path.exists()) and not force:
        raise RuntimeError("partial or invalid output exists; inspect before --force")

    versions = package_versions(config)
    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"42-day pilot requires a JAX GPU, found {devices}")

    checkpoint = run_contract.storage_path(config, "checkpoint")
    verify_file(
        checkpoint,
        int(config["model"]["checkpoint_size_bytes"]),
        config["model"]["checkpoint_sha256"],
    )
    input_path = run_contract.storage_path(config, "input")
    input_manifest_path = run_contract.storage_path(config, "input_manifest")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    verify_file(input_path, None, input_manifest["input_sha256"])
    if not input_manifest.get("forcing_shift_verified_against_direct_D_minus_1"):
        raise ValueError("staged IC lacks forcing-shift proof")

    with checkpoint.open("rb") as handle:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(handle))
    staged = xr.load_dataset(input_path, engine="netcdf4")
    try:
        inputs = model.inputs_from_xarray(staged.isel(time=0))
        input_forcings = model.forcings_from_xarray(staged.isel(time=0))
        persisted_forcings = model.forcings_from_xarray(staged)
        post_process_fn, fields = make_post_processor(model, config)
        product = config["model"]["product"]
        member_values: list[dict[str, np.ndarray]] = []
        diagnostic_records: list[dict[str, float]] = []
        member_runtimes: list[float] = []
        inference_started = time.perf_counter()
        for member, seed in enumerate(config["forecast"]["member_seeds"]):
            print(f"running member={member} seed={seed}", flush=True)
            member_started = time.perf_counter()
            initial_state = model.encode(
                inputs, input_forcings, jax.random.key(np.uint32(seed))
            )
            final_state, predictions = model.unroll(
                initial_state,
                persisted_forcings,
                steps=int(config["forecast"]["unroll_steps"]),
                timedelta=np.timedelta64(
                    int(config["forecast"]["output_interval_hours"]), "h"
                ),
                start_with_input=True,
                post_process_fn=post_process_fn,
            )
            del final_state
            values, diagnostics = aggregate_member(
                predictions, product, fields, config
            )
            member_values.append(values)
            member_runtime = time.perf_counter() - member_started
            cumulative_runtime = time.perf_counter() - inference_started
            member_runtimes.append(member_runtime)
            diagnostic_records.append(
                {
                    **diagnostics,
                    "member_runtime_seconds": member_runtime,
                    "cumulative_inference_seconds": cumulative_runtime,
                }
            )
            print(
                f"finished member={member} member_seconds={member_runtime:.3f} "
                f"cumulative_seconds={cumulative_runtime:.3f}",
                flush=True,
            )
            del predictions
        inference_runtime = time.perf_counter() - inference_started
        standardized = build_output(member_values, model, config)
    finally:
        staged.close()

    stats: dict[str, dict[str, float]] = {}
    for field in config["model"]["retained_fields"]:
        values = np.asarray(standardized[field].values)
        if not np.isfinite(values).all():
            raise ValueError(f"standardized {field} contains non-finite values")
        stats[field] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
        }
    if config["model"]["product"] == "daily_tp":
        if stats["tp"]["minimum"] < 0.0 or stats["tp"]["maximum"] > 2000.0:
            raise ValueError(f"implausible TP range: {stats['tp']}")
    else:
        for field in config["model"]["retained_fields"]:
            if stats[field]["minimum"] < 150.0 or stats[field]["maximum"] > 350.0:
                raise ValueError(f"implausible pressure temperature: {field} {stats[field]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    encoding = {
        field: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "chunksizes": (1, 7, 27, 27),
        }
        for field in config["model"]["retained_fields"]
    }
    time_encoding = {
        "units": "hours since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    for name in (
        "valid_time",
        "forecast_period_start",
        "forecast_period_end",
        "forecast_period_bounds",
        "init_time",
        "forecast_reference_time",
        "information_cutoff_time",
    ):
        encoding[name] = time_encoding.copy()
    standardized.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    with xr.open_dataset(temporary) as check:
        expected_sizes = {
            "member": int(config["forecast"]["member_count"]),
            "lead_day": 42,
            "latitude": 27,
            "longitude": 27,
            "bounds": 2,
        }
        if dict(check.sizes) != expected_sizes:
            raise ValueError(f"written output dimensions are wrong: {dict(check.sizes)}")
        check.load()
    os.replace(temporary, output_path)

    output_hash = pilot_contract.sha256_file(output_path)
    result = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_label": config["run_label"],
        "purpose": config["purpose"],
        "repository_commit": repository_commit(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "jax_devices": [str(device) for device in devices],
        "package_versions": versions,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": config["model"]["checkpoint_sha256"],
        "input_path": str(input_path),
        "input_sha256": input_manifest["input_sha256"],
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": output_hash,
        "product": config["model"]["product"],
        "retained_fields": config["model"]["retained_fields"],
        "lead_days": 42,
        "member_seeds": config["forecast"]["member_seeds"],
        "member_runtime_seconds": member_runtimes,
        "inference_runtime_seconds": inference_runtime,
        "field_stats": stats,
        "member_diagnostics": diagnostic_records,
        "t2m_written": "t2m" in standardized.data_vars,
        "run_mode": run_mode,
        "promoted_to_production": run_mode == "production",
    }
    if result["t2m_written"]:
        raise ValueError("NeuralGCM precipitation run must not write a false T2M variable")
    write_json_atomic(result, manifest_path)
    if report_path is not None:
        write_json_atomic(result, report_path)
    print(f"42-day NeuralGCM {run_mode} passed: {output_path}", flush=True)
    print(f"output SHA256: {output_hash}", flush=True)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return output_path, manifest_path


def main() -> None:
    args = parse_args()
    config = run_contract.load_config(args.config)
    run(config, args.force)


if __name__ == "__main__":
    main()
