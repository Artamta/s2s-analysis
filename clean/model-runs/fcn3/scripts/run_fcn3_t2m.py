#!/usr/bin/env python3
"""Run streaming FCN3 native T2M inference for the frozen S2S calendar."""

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
    existing_pair_is_valid,
    load_config,
    period_coordinates,
    product_paths,
    select_date,
    sha256_file,
    target_coordinates,
    write_json_atomic,
)


def select_field(x: torch.Tensor, coords: OrderedDict, name: str) -> torch.Tensor:
    variables = list(coords["variable"])
    index = variables.index(name)
    dimension = list(coords).index("variable")
    field = x.select(dimension, index).squeeze()
    if field.ndim != 2:
        raise ValueError(f"{name} did not reduce to a 2-D field: {field.shape}")
    return field


def target_indices(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat_index = np.rint((source_lat[0] - target_lat) / 0.25).astype(int)
    lon_index = np.rint((target_lon - source_lon[0]) / 0.25).astype(int)
    if not np.allclose(source_lat[lat_index], target_lat) or not np.allclose(
        source_lon[lon_index], target_lon
    ):
        raise ValueError("canonical T2M nodes are absent from the FCN3 grid")
    return lat_index, lon_index


def daily_trapezoid(boundaries: list[torch.Tensor]) -> torch.Tensor:
    if len(boundaries) != 5:
        raise ValueError("daily T2M requires boundaries at 0, 6, 12, 18, and 24 h")
    return (
        0.5 * boundaries[0]
        + boundaries[1]
        + boundaries[2]
        + boundaries[3]
        + 0.5 * boundaries[4]
    ) / 4.0


@torch.inference_mode()
def run_member(
    model,
    initial: torch.Tensor,
    coords: OrderedDict,
    seed: int,
    steps: int,
    lat_index: np.ndarray,
    lon_index: np.ndarray,
) -> np.ndarray:
    model.set_rng(seed=seed, reset=True)
    iterator = model.create_iterator(initial.clone(), coords.copy())
    boundaries: list[torch.Tensor] = []
    daily_t2m: list[np.ndarray] = []

    for step in range(steps + 1):
        state, out_coords = next(iterator)
        lead_hours = int(out_coords["lead_time"][0] / np.timedelta64(1, "h"))
        if lead_hours != step * 6:
            raise ValueError(f"unexpected FCN3 lead at step {step}: {lead_hours} h")
        t2m = select_field(state, out_coords, "t2m")
        if not torch.isfinite(t2m).all():
            raise ValueError(f"non-finite FCN3 T2M at step {step}")
        boundaries.append(t2m[lat_index][:, lon_index])

        if step > 0 and step % 4 == 0:
            mean_celsius = daily_trapezoid(boundaries) - 273.15
            values = mean_celsius.float().cpu().numpy()
            daily_t2m.append(values)
            boundaries = [boundaries[-1]]
            print(
                f"seed={seed} completed day={step // 4}/{steps // 4} "
                f"t2m=[{values.min():.2f},{values.max():.2f}]",
                flush=True,
            )

    return np.stack(daily_t2m)


def build_dataset(
    t2m: np.ndarray,
    config: dict,
    init_date: str,
    seeds: list[int],
    lead_days: int,
) -> xr.Dataset:
    latitude, longitude = target_coordinates(config)
    dataset = xr.Dataset(
        {"t2m": (("member", "lead_day", "latitude", "longitude"), t2m)},
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
            "native_grid": "global 0.25 degree latitude-longitude",
            "native_timestep": "6 hours",
            "calendar_sha256": config["calendar"]["sha256"],
            "ensemble_definition": "native stochastic FCN3 samples from fixed seeds and a common unperturbed ERA5 IC",
            "retained_fields": "t2m only; FCN3 has no native precipitation output",
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
    if config["model"]["fields"] != ["t2m"]:
        raise ValueError("T2M runner requires a config with fields=[\"t2m\"]")
    calendar_index, init_date = select_date(config, args.index, args.init_date)
    paths = product_paths(config, init_date)
    members = args.members or config["model"]["members"]
    if args.steps <= 0 or args.steps % 4 != 0:
        raise ValueError("--steps must be a positive multiple of four")
    if not 1 <= members <= config["model"]["members"]:
        raise ValueError("invalid member count")
    production, output = resolve_output_mode(
        paths["output"], args.steps, members, config["model"]["members"], args.test_output
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

    from earth2studio.models.px import FCN3

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

    package = FCN3.load_default_package()
    model = FCN3.load_model(package).to(device).eval()
    lat_index, lon_index = target_indices(
        np.asarray(coords["lat"]), np.asarray(coords["lon"]), *target_coordinates(config)
    )
    seeds = config["model"]["seeds"][:members]
    member_fields, runtimes = [], []
    for member, seed in enumerate(seeds):
        member_started = time.perf_counter()
        print(f"running member={member}/{members - 1} seed={seed}", flush=True)
        member_fields.append(
            run_member(model, initial, coords, seed, args.steps, lat_index, lon_index)
        )
        runtimes.append(time.perf_counter() - member_started)
        gc.collect()
        torch.cuda.empty_cache()

    values = np.stack(member_fields)
    expected = (members, args.steps // 4, 27, 27)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("T2M has invalid shape or non-finite values")
    stats = {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }
    if not (-100 < stats["minimum"] < stats["maximum"] < 70):
        raise ValueError(f"implausible T2M range {stats}")
    if members > 1 and float(np.max(np.std(values, axis=0))) <= 1e-6:
        raise ValueError("FCN3 stochastic members have no detectable spread")

    dataset = build_dataset(values, config, init_date, seeds, args.steps // 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={
            "t2m": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "chunksizes": (1, min(7, args.steps // 4), 27, 27),
            }
        },
    )
    with xr.open_dataset(temporary) as check:
        check.load()
        if check["t2m"].shape != expected or set(check.data_vars) != {"t2m"}:
            raise ValueError("written FCN3 T2M output contract is invalid")
        if "forecast_period_bounds" not in check.coords:
            raise ValueError("written FCN3 T2M output lacks period bounds")
    os.replace(temporary, output)
    record = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_contract": production,
        "calendar_index": calendar_index,
        "init_date": init_date,
        "calendar_sha256": config["calendar"]["sha256"],
        "model_package_uri": config["model"]["package_uri"],
        "checkpoint_inventory_sha256": sha256_file(paths["inventory"]),
        "input_sha256": json.loads(paths["stage_manifest"].read_text())["input_sha256"],
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "members": members,
        "seeds": seeds,
        "steps": args.steps,
        "lead_days": args.steps // 4,
        "statistics": {"t2m": stats},
        "member_runtime_seconds": runtimes,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    }
    write_json_atomic(record, manifest_path)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
