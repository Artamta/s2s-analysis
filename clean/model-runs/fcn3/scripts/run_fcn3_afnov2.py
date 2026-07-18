#!/usr/bin/env python3
"""Run streaming FCN3 native T2M plus AFNOv2 TP inference."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from fcn3_common import (
    conservative_weights,
    existing_pair_is_valid,
    load_config,
    period_coordinates,
    product_paths,
    select_date,
    sha256_file,
    target_coordinates,
    write_json_atomic,
)


PLEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def select_field(x: torch.Tensor, coords: OrderedDict, name: str) -> torch.Tensor:
    variables = list(coords["variable"])
    index = variables.index(name)
    dimension = list(coords).index("variable")
    field = x.select(dimension, index).squeeze()
    if field.ndim != 2:
        raise ValueError(f"{name} did not reduce to a 2-D field: {field.shape}")
    return field


def target_remappers(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    device: torch.device,
):
    lat_index = np.rint((source_lat[0] - target_lat) / 0.25).astype(int)
    lon_index = np.rint((target_lon - source_lon[0]) / 0.25).astype(int)
    if not np.allclose(source_lat[lat_index], target_lat) or not np.allclose(
        source_lon[lon_index], target_lon
    ):
        raise ValueError("canonical T2M nodes are absent from the FCN3 grid")

    lat_w, lon_w, weight_hash = conservative_weights(
        source_lat, source_lon, target_lat, target_lon
    )
    lat_nonzero = np.flatnonzero(np.any(lat_w > 0, axis=0))
    lon_nonzero = np.flatnonzero(np.any(lon_w > 0, axis=0))
    lat_slice = slice(lat_nonzero[0], lat_nonzero[-1] + 1)
    lon_slice = slice(lon_nonzero[0], lon_nonzero[-1] + 1)
    lat_tensor = torch.as_tensor(lat_w[:, lat_slice], dtype=torch.float32, device=device)
    lon_tensor = torch.as_tensor(lon_w[:, lon_slice], dtype=torch.float32, device=device)
    return lat_index, lon_index, lat_slice, lon_slice, lat_tensor, lon_tensor, weight_hash


def run_member(
    wrapped,
    base_model,
    initial: torch.Tensor,
    coords: OrderedDict,
    seed: int,
    steps: int,
    remappers,
) -> tuple[np.ndarray, np.ndarray, dict]:
    (
        lat_index,
        lon_index,
        lat_slice,
        lon_slice,
        lat_weights,
        lon_weights,
        _,
    ) = remappers
    base_model.set_rng(seed=seed, reset=True)
    iterator = wrapped.create_iterator(initial.clone(), coords.copy())
    boundaries: list[torch.Tensor] = []
    daily_t2m: list[np.ndarray] = []
    daily_tp: list[np.ndarray] = []
    tp_accumulator = torch.zeros((27, 27), dtype=torch.float32, device=initial.device)
    initial_tp = None
    native_tp_min = float("inf")
    native_tp_max = float("-inf")

    for step in range(steps + 1):
        state, out_coords = next(iterator)
        lead_hours = int(out_coords["lead_time"][0] / np.timedelta64(1, "h"))
        if lead_hours != step * 6:
            raise ValueError(f"unexpected FCN3 lead at step {step}: {lead_hours} h")
        t2m = select_field(state, out_coords, "t2m")
        tp06 = select_field(state, out_coords, "tp06")
        if not torch.isfinite(t2m).all() or not torch.isfinite(tp06).all():
            raise ValueError(f"non-finite FCN3 output at step {step}")
        native_tp_min = min(native_tp_min, float(tp06.min()))
        native_tp_max = max(native_tp_max, float(tp06.max()))
        if float(tp06.min()) < 0:
            raise ValueError("AFNOv2 returned negative precipitation after its transform")

        target_t2m = t2m[lat_index][:, lon_index]
        boundaries.append(target_t2m)
        if step == 0:
            initial_tp = float(tp06.mean())
        else:
            cropped = tp06[lat_slice, lon_slice]
            target_tp = lat_weights @ cropped @ lon_weights.T
            tp_accumulator += target_tp * 1000.0

        if step > 0 and step % 4 == 0:
            if len(boundaries) != 5:
                raise ValueError("daily T2M boundary count is invalid")
            mean = (
                0.5 * boundaries[0]
                + boundaries[1]
                + boundaries[2]
                + boundaries[3]
                + 0.5 * boundaries[4]
            ) / 4.0
            daily_t2m.append((mean - 273.15).float().cpu().numpy())
            daily_tp.append(tp_accumulator.float().cpu().numpy())
            boundaries = [boundaries[-1]]
            tp_accumulator.zero_()
            print(
                f"seed={seed} completed day={step // 4}/{steps // 4} "
                f"t2m=[{daily_t2m[-1].min():.2f},{daily_t2m[-1].max():.2f}] "
                f"tp=[{daily_tp[-1].min():.3f},{daily_tp[-1].max():.3f}]",
                flush=True,
            )

    return np.stack(daily_t2m), np.stack(daily_tp), {
        "seed": seed,
        "initial_diagnostic_tp06_mean_m": initial_tp,
        "native_tp06_minimum_m": native_tp_min,
        "native_tp06_maximum_m": native_tp_max,
    }


def build_dataset(
    t2m: np.ndarray,
    tp: np.ndarray,
    config: dict,
    init_date: str,
    seeds: list[int],
    lead_days: int,
    weight_hash: str,
) -> xr.Dataset:
    latitude, longitude = target_coordinates(config)
    dataset = xr.Dataset(
        {
            "t2m": (("member", "lead_day", "latitude", "longitude"), t2m),
            "tp": (("member", "lead_day", "latitude", "longitude"), tp),
        },
        coords={
            "member": np.arange(len(seeds), dtype=np.int16),
            "seed": ("member", np.asarray(seeds, dtype=np.int64)),
            "latitude": latitude,
            "longitude": longitude,
            **period_coordinates(init_date, lead_days),
        },
        attrs={
            "run_label": config["model"]["run_label"],
            "model_display_name": config["model"]["display_name"],
            "model_package_uri": config["model"]["package_uri"],
            "precipitation_package_uri": config["model"]["precipitation_package_uri"],
            "native_grid": "global 0.25 degree latitude-longitude",
            "native_timestep": "6 hours",
            "calendar_sha256": config["calendar"]["sha256"],
            "precipitation_regrid_weight_sha256": weight_hash,
            "ensemble_definition": "native stochastic FCN3 samples from fixed seeds and a common unperturbed ERA5 IC",
        },
    )
    dataset["valid_time"].attrs.update(
        {"bounds": "forecast_period_bounds", "representation": "period_end"}
    )
    dataset["t2m"].attrs.update(
        {
            "long_name": "2 metre temperature",
            "units": "degC",
            "source": "native FCN3 prognostic t2m",
            "cell_methods": "time: mean (trapezoidal integration of 0/6/12/18/24 h boundaries)",
            "horizontal_regrid": "exact selection of common 1.5 degree nodes from native 0.25 degree grid",
        }
    )
    dataset["tp"].attrs.update(
        {
            "long_name": "total precipitation",
            "units": "mm day-1",
            "source": "PrecipitationAFNOv2 tp06 diagnostic driven by FCN3 plus derived surface pressure",
            "cell_methods": "time: sum (four 6-hour accumulations)",
            "horizontal_regrid": "spherical conservative cell-overlap area mean from 0.25 to 1.5 degrees",
        }
    )
    return dataset


def resolve_output_mode(
    production_output: Path,
    steps: int,
    members: int,
    configured_members: int,
    test_output: Path | None,
) -> tuple[bool, Path]:
    if test_output is not None:
        return False, test_output
    if steps == 168 and members == configured_members:
        return True, production_output
    raise ValueError("non-production runs require --test-output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--init-date")
    parser.add_argument("--steps", type=int, default=168)
    parser.add_argument("--members", type=int)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    calendar_index, init_date = select_date(config, args.index, args.init_date)
    paths = product_paths(config, init_date)
    members = args.members or config["model"]["members"]
    if args.steps <= 0 or args.steps % 4 != 0:
        raise ValueError("--steps must be a positive multiple of four")
    if not 1 <= members <= config["model"]["members"]:
        raise ValueError("invalid member count")
    production, output = resolve_output_mode(
        paths["output"],
        args.steps,
        members,
        config["model"]["members"],
        args.test_output,
    )
    manifest_path = paths["manifest"] if production else output.with_suffix(".json")
    if not args.force and existing_pair_is_valid(output, manifest_path, "output_sha256"):
        print(f"validated existing forecast: {output}", flush=True)
        return
    if (output.exists() or manifest_path.exists()) and not args.force:
        raise RuntimeError("partial or invalid output exists; inspect before --force")
    if not existing_pair_is_valid(paths["stage"], paths["stage_manifest"], "input_sha256"):
        raise RuntimeError("staged FCN3 initial condition is absent or invalid")
    if not paths["inventory"].is_file():
        raise RuntimeError("checkpoint inventory is absent")
    if not torch.cuda.is_available():
        raise RuntimeError("FCN3 inference requires a CUDA GPU")

    from earth2studio.models.dx import DerivedSurfacePressure, PrecipitationAFNOv2
    from earth2studio.models.px import DiagnosticWrapper, FCN3

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    with xr.open_dataset(paths["stage"]) as source:
        source.load()
        initial = torch.from_numpy(source["state"].values.astype(np.float32)).to(device)
        coords = OrderedDict(
            (dimension, np.asarray(source.coords[dimension].values))
            for dimension in source["state"].dims
        )

    model_package = FCN3.load_default_package()
    base_model = FCN3.load_model(model_package).to(device)
    orography = np.asarray(
        xr.open_dataset(model_package.resolve("orography.nc"))["Z"].values
    ).squeeze()
    if orography.shape != (721, 1440):
        raise ValueError(f"unexpected FCN3 orography shape {orography.shape}")
    sp_model = DerivedSurfacePressure(
        p_levels=PLEVELS,
        surface_geopotential=torch.as_tensor(orography, dtype=torch.float32),
        surface_geopotential_coords=OrderedDict(
            {"lat": coords["lat"], "lon": coords["lon"]}
        ),
    ).to(device)
    with_sp = DiagnosticWrapper(px_model=base_model, dx_model=sp_model).to(device)
    precip = PrecipitationAFNOv2.load_model(
        PrecipitationAFNOv2.load_default_package()
    ).to(device)
    wrapped = DiagnosticWrapper(px_model=with_sp, dx_model=precip).to(device)

    source_lat = precip.input_coords()["lat"]
    source_lon = precip.input_coords()["lon"]
    remappers = target_remappers(
        source_lat, source_lon, *target_coordinates(config), device
    )
    seeds = config["model"]["seeds"][:members]
    t2m_members, tp_members, diagnostics, runtimes = [], [], [], []
    for member, seed in enumerate(seeds):
        member_started = time.perf_counter()
        print(f"running member={member}/{members - 1} seed={seed}", flush=True)
        t2m, tp, diagnostic = run_member(
            wrapped, base_model, initial, coords, seed, args.steps, remappers
        )
        t2m_members.append(t2m)
        tp_members.append(tp)
        diagnostics.append(diagnostic)
        runtimes.append(time.perf_counter() - member_started)
        gc.collect()
        torch.cuda.empty_cache()

    fields = {"t2m": np.stack(t2m_members), "tp": np.stack(tp_members)}
    expected = (members, args.steps // 4, 27, 27)
    stats = {}
    for name, values in fields.items():
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"{name} has invalid shape or non-finite values")
        stats[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
        }
    if not (-100 < stats["t2m"]["minimum"] < stats["t2m"]["maximum"] < 70):
        raise ValueError(f"implausible T2M range {stats['t2m']}")
    if not (0 <= stats["tp"]["minimum"] <= stats["tp"]["maximum"] < 2000):
        raise ValueError(f"implausible TP range {stats['tp']}")
    if members > 1 and max(
        float(np.max(np.std(fields[name], axis=0))) for name in fields
    ) <= 1e-6:
        raise ValueError("FCN3 stochastic members have no detectable spread")

    dataset = build_dataset(
        fields["t2m"], fields["tp"], config, init_date, seeds, args.steps // 4, remappers[-1]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (1, min(7, args.steps // 4), 27, 27),
        }
        for name in fields
    }
    dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    with xr.open_dataset(temporary) as check:
        check.load()
        if check["t2m"].shape != expected or check["tp"].shape != expected:
            raise ValueError("written FCN3 output shape is invalid")
    os.replace(temporary, output)
    record = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_contract": production,
        "calendar_index": calendar_index,
        "init_date": init_date,
        "calendar_sha256": config["calendar"]["sha256"],
        "model_package_uri": config["model"]["package_uri"],
        "precipitation_package_uri": config["model"]["precipitation_package_uri"],
        "checkpoint_inventory_sha256": sha256_file(paths["inventory"]),
        "input_sha256": json.loads(paths["stage_manifest"].read_text())["input_sha256"],
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "members": members,
        "seeds": seeds,
        "steps": args.steps,
        "lead_days": args.steps // 4,
        "statistics": stats,
        "diagnostics": diagnostics,
        "member_runtime_seconds": runtimes,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "precipitation_regrid_weight_sha256": remappers[-1],
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    }
    write_json_atomic(record, manifest_path)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
