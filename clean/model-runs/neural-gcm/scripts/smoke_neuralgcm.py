#!/usr/bin/env python3
"""Run a one-member NeuralGCM +0/+6/+12-hour GPU smoke forecast."""

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
from pathlib import Path
from typing import Any

import jax
import numpy as np
import xarray as xr
import neuralgcm


COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))

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
        cwd=pilot_contract.REPO_ROOT,
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
                f"runtime version mismatch for {package}: expected {expected}, "
                f"found {actual}"
            )
        result[package] = actual
    return result


def verify_file(path: Path, size: int | None, digest: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise ValueError(
            f"size mismatch for {path}: expected {size}, found {path.stat().st_size}"
        )
    actual_hash = pilot_contract.sha256_file(path)
    if actual_hash != digest:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {digest}, found {actual_hash}"
        )


def existing_output_is_valid(output: Path, manifest: Path) -> bool:
    if not output.is_file() or not manifest.is_file():
        return False
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        record.get("status") == "passed"
        and record.get("output_sha256") == pilot_contract.sha256_file(output)
    )


def run(config: dict[str, Any], force: bool) -> tuple[Path, Path]:
    output_path = pilot_contract.storage_path(config, "output")
    manifest_path = pilot_contract.storage_path(config, "output_manifest")
    report_path = pilot_contract.storage_path(config, "pilot_report")
    if not force and existing_output_is_valid(output_path, manifest_path):
        print(f"validated existing NeuralGCM smoke output: {output_path}", flush=True)
        return output_path, manifest_path
    if (output_path.exists() or manifest_path.exists()) and not force:
        raise RuntimeError(
            "partial or invalid smoke output exists; inspect it and rerun with --force "
            "only if replacement is intentional"
        )

    versions = package_versions(config)
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError(f"NeuralGCM smoke requires a JAX GPU, found {devices}")

    checkpoint = pilot_contract.storage_path(config, "checkpoint")
    verify_file(
        checkpoint,
        int(config["model"]["checkpoint_size_bytes"]),
        config["model"]["checkpoint_sha256"],
    )
    input_path = pilot_contract.storage_path(config, "input")
    input_manifest_path = pilot_contract.storage_path(config, "input_manifest")
    if not input_manifest_path.is_file():
        raise FileNotFoundError(input_manifest_path)
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    verify_file(input_path, None, input_manifest["input_sha256"])
    if not input_manifest.get("forcing_shift_verified_against_direct_D_minus_1"):
        raise ValueError("staged IC does not prove the D-1 forcing shift")

    with checkpoint.open("rb") as handle:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(handle))
    staged = xr.load_dataset(input_path, engine="netcdf4")
    try:
        missing = sorted(
            set(model.input_variables + model.forcing_variables).difference(
                staged.data_vars
            )
        )
        if missing:
            raise ValueError(f"staged IC is missing checkpoint variables: {missing}")
        if staged.sizes.get("time") != 1:
            raise ValueError("staged IC must contain exactly one labeled time")

        initial_slice = staged.isel(time=0)
        inputs = model.inputs_from_xarray(initial_slice)
        input_forcings = model.forcings_from_xarray(initial_slice)
        rng_key = jax.random.key(int(config["smoke"]["member_seed"]))
        initial_state = model.encode(inputs, input_forcings, rng_key)

        persisted_forcings = model.forcings_from_xarray(staged)
        frame_hours = np.asarray(
            config["smoke"]["expected_frame_hours"], dtype=np.int32
        )
        final_state, predictions = model.unroll(
            initial_state,
            persisted_forcings,
            steps=int(config["smoke"]["unroll_steps"]),
            timedelta=np.timedelta64(
                int(config["smoke"]["output_interval_hours"]), "h"
            ),
            start_with_input=bool(config["smoke"]["start_with_input"]),
        )
        del final_state
        predictions_ds = model.data_to_xarray(predictions, times=frame_hours)
    finally:
        staged.close()

    field = config["smoke"]["output_fields"][0]
    if field not in predictions_ds:
        raise KeyError(
            f"checkpoint output lacks {field}; found {sorted(predictions_ds.data_vars)}"
        )
    forbidden = {"2m_temperature", "t2m"}.intersection(predictions_ds.data_vars)
    if forbidden:
        raise ValueError(f"unexpected T2M claim in precipitation checkpoint: {forbidden}")

    cumulative = predictions_ds[field]
    if "surface" in cumulative.dims:
        if cumulative.sizes["surface"] != 1:
            raise ValueError(
                "expected one precipitation surface, found "
                f"{cumulative.sizes['surface']}"
            )
        cumulative = cumulative.isel(surface=0, drop=True)
    cumulative = cumulative.transpose("time", "latitude", "longitude")
    values = np.asarray(cumulative.values, dtype=np.float32)
    expected_shape = (len(frame_hours), 64, 128)
    if values.shape != expected_shape:
        raise ValueError(f"expected cumulative TP shape {expected_shape}, found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("NeuralGCM cumulative precipitation contains non-finite values")
    increments = np.diff(values.astype(np.float64), axis=0)
    minimum_increment = float(increments.min())
    tolerance = float(config["smoke"]["negative_increment_tolerance_m"])
    if minimum_increment < -tolerance:
        raise ValueError(
            f"cumulative precipitation decreased by {minimum_increment} m, "
            f"exceeding tolerance {-tolerance} m"
        )

    init_time = np.datetime64(config["case"]["init_time"])
    valid_time = init_time + frame_hours.astype("timedelta64[h]")
    smoke_output = xr.Dataset(
        data_vars={
            field: (
                ("member", "frame", "latitude", "longitude"),
                values[np.newaxis, ...],
            )
        },
        coords={
            "member": np.asarray([0], dtype=np.int16),
            "frame": np.arange(len(frame_hours), dtype=np.int16),
            "lead_time_hours": ("frame", frame_hours),
            "valid_time": ("frame", valid_time),
            "latitude": cumulative.latitude.values,
            "longitude": cumulative.longitude.values,
            "init_time": init_time,
            "forecast_reference_time": init_time,
            "information_cutoff_time": np.datetime64(
                config["case"]["information_cutoff_time"]
            ),
        },
        attrs={
            "run_label": config["run_label"],
            "purpose": config["purpose"],
            "model": config["model"]["name"],
            "checkpoint_url": config["model"]["checkpoint_url"],
            "checkpoint_sha256": config["model"]["checkpoint_sha256"],
            "input_sha256": input_manifest["input_sha256"],
            "forcing_policy": config["initial_conditions"]["forcing_policy"],
            "member_seed": int(config["smoke"]["member_seed"]),
            "field_availability": "tp available; t2m unavailable in checkpoint",
            "paper_status": "smoke output only; not for skill scores",
        },
    )
    smoke_output[field].attrs.update(
        {
            "long_name": "precipitation accumulated from forecast initialization",
            "units": "m",
            "temporal_statistic": "cumulative from initialization",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    smoke_output.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={field: {"zlib": True, "complevel": 1, "shuffle": True}},
    )
    with xr.open_dataset(temporary) as check:
        if check.sizes.get("frame") != len(frame_hours):
            raise ValueError("written smoke file has the wrong frame count")
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
        "frame_hours": frame_hours.tolist(),
        "member_seed": int(config["smoke"]["member_seed"]),
        "cumulative_min_m": float(values.min()),
        "cumulative_max_m": float(values.max()),
        "minimum_six_hour_increment_m": minimum_increment,
        "negative_increment_tolerance_m": tolerance,
        "t2m_written": False,
        "promoted_to_full_pilot": False,
    }
    write_json_atomic(result, manifest_path)
    write_json_atomic(result, report_path)
    print(f"NeuralGCM smoke test passed: {output_path}", flush=True)
    print(f"output SHA256: {output_hash}", flush=True)
    print(f"minimum 6-hour increment: {minimum_increment} m", flush=True)
    return output_path, manifest_path


def main() -> None:
    args = parse_args()
    config = pilot_contract.load_config(args.config)
    run(config, args.force)


if __name__ == "__main__":
    main()
