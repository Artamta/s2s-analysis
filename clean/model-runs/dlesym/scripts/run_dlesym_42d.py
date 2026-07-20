#!/usr/bin/env python3
"""Run one verified 42-day DLESyM forecast from a staged HEALPix state."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import earth2grid
import numpy as np
import torch
import xarray as xr

from dlesym_common import (
    conservative_weights,
    coords_from_dataset,
    existing_pair_is_valid,
    load_config,
    period_coordinates,
    product_paths,
    repository_commit,
    select_date,
    sha256_file,
    target_coordinates,
    write_json_atomic,
)


def hours(values: np.ndarray) -> np.ndarray:
    return np.asarray(values / np.timedelta64(1, "h"), dtype=np.int64)


def make_regridder(device: str):
    hpx = earth2grid.healpix.Grid(
        level=6, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
    )
    latlon = earth2grid.latlon.equiangular_lat_lon_grid(721, 1440)
    return earth2grid.get_regridder(hpx, latlon).to(device=device, dtype=torch.float32)


def to_latlon(fields: torch.Tensor, regridder: Any) -> torch.Tensor:
    leading = fields.shape[:-3]
    flat = fields.reshape(-1, int(np.prod(fields.shape[-3:])))
    return regridder(flat).reshape(*leading, 721, 1440)


def t2m_to_common(
    daily_hpx: torch.Tensor,
    regridder: Any,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    latlon = to_latlon(daily_hpx, regridder)
    source_lat = np.linspace(90.0, -90.0, 721)
    source_lon = np.linspace(0.0, 359.75, 1440)
    lat_index = [int(np.argmin(np.abs(source_lat - value))) for value in target_lat]
    lon_index = [int(np.argmin(np.abs(source_lon - value))) for value in target_lon]
    if not np.allclose(source_lat[lat_index], target_lat) or not np.allclose(
        source_lon[lon_index], target_lon
    ):
        raise ValueError("canonical T2M coordinates are not exact 0.25-degree nodes")
    common = latlon.index_select(
        -2, torch.tensor(lat_index, device=latlon.device)
    ).index_select(-1, torch.tensor(lon_index, device=latlon.device))
    return (common - 273.15).cpu().numpy().astype(np.float32)


def tp_to_common(
    daily_hpx: torch.Tensor,
    regridder: Any,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[np.ndarray, str]:
    latlon = to_latlon(daily_hpx, regridder)
    source_lat = np.linspace(90.0, -90.0, 721)
    source_lon = np.linspace(0.0, 359.75, 1440)
    lat_w, lon_w, weight_hash = conservative_weights(
        source_lat, source_lon, target_lat, target_lon
    )
    lat_weight = torch.as_tensor(lat_w, device=latlon.device, dtype=torch.float64)
    lon_weight = torch.as_tensor(lon_w, device=latlon.device, dtype=torch.float64)
    common = torch.einsum(
        "ai,...ij,bj->...ab", lat_weight, latlon.double(), lon_weight
    )
    return common.cpu().numpy().astype(np.float32), weight_hash


def aggregate_t2m(states: list[torch.Tensor]) -> torch.Tensor:
    values = torch.stack(states)
    if values.shape[0] != 169:
        raise ValueError(f"expected 169 T2M boundaries, got {values.shape[0]}")
    return (
        0.5 * values[:-1:4]
        + values[1::4]
        + values[2::4]
        + values[3::4]
        + 0.5 * values[4::4]
    ) / 4.0


def load_native_model(product: str, package: Any, pair: tuple[int, int] = (0, 0)):
    if product == "v1_t2m":
        from earth2studio.models.px import DLESyM

        return DLESyM.load_model(package, pair[0], pair[1])
    from earth2studio.models.px import DLESyMv0_ISCCP_ERA5

    return DLESyMv0_ISCCP_ERA5.load_model(package, use_ttr=False)


def rollout_v1_member(
    model: Any, initial: torch.Tensor, initial_coords: OrderedDict[str, np.ndarray]
) -> tuple[torch.Tensor, list[int]]:
    coords = initial_coords.copy()
    state = initial
    variable_index = list(coords["variable"]).index("t2m")
    zero_index = list(hours(coords["lead_time"])).index(0)
    t2m = [state[0, 0, zero_index, variable_index].detach().clone()]
    leads = [0]
    for _ in range(11):
        output = model._forward(state, coords)
        output_coords = model.output_coords(coords)
        for index, lead in enumerate(hours(output_coords["lead_time"])):
            if 0 < lead <= 1008:
                t2m.append(output[0, 0, index, variable_index].detach().clone())
                leads.append(int(lead))
        state, coords = model._next_step_inputs(output, output_coords)
    if leads != list(range(0, 1009, 6)):
        raise ValueError(f"V1 T2M lead sequence is invalid: {leads[:5]}...{leads[-5:]}")
    return aggregate_t2m(t2m), leads


def combined_state(
    atmos: torch.Tensor,
    sst: torch.Tensor,
    atmos_variables: list[str],
    precip_variables: list[str],
) -> torch.Tensor:
    values = {name: atmos[..., index, :, :, :] for index, name in enumerate(atmos_variables)}
    values["sst"] = sst
    return torch.stack([values[name] for name in precip_variables], dim=-4)


def rollout_v0(
    model: Any,
    precip: Any,
    initial: torch.Tensor,
    initial_coords: OrderedDict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, list[int], float]:
    coords = initial_coords.copy()
    state = initial
    all_variables = list(coords["variable"])
    t2m_index = all_variables.index("t2m")
    zero_index = list(hours(coords["lead_time"])).index(0)
    t2m = [state[0, 0, zero_index, t2m_index].detach().clone()]
    precip_variables = list(precip.input_coords()["variable"])
    initial_order = [all_variables.index(name) for name in precip_variables]
    previous = state[0, 0, zero_index, initial_order].detach().clone()
    tp06: list[torch.Tensor] = []
    leads = [0]
    minimum_tp06 = float("inf")

    for _ in range(11):
        output = model._forward(state, coords)
        output_coords = model.output_coords(coords)
        atmos, atmos_coords = model.retrieve_valid_atmos_outputs(output, output_coords)
        ocean, ocean_coords = model.retrieve_valid_ocean_outputs(output, output_coords)
        block_leads = hours(atmos_coords["lead_time"])
        ocean_leads = hours(ocean_coords["lead_time"])
        if len(block_leads) != 16 or len(ocean_leads) != 2:
            raise ValueError("unexpected DLESyM coupled output cadence")
        block_start = int(block_leads[0] - 6)
        if ocean_leads.tolist() != [block_start + 48, block_start + 96]:
            raise ValueError("ocean output times do not bracket the atmosphere block")

        start_sst = previous[precip_variables.index("sst")]
        middle_sst = ocean[0, 0, 0, 0]
        end_sst = ocean[0, 0, 1, 0]
        interpolated = []
        for step in range(1, 17):
            if step <= 8:
                alpha = step / 8.0
                interpolated.append((1.0 - alpha) * start_sst + alpha * middle_sst)
            else:
                alpha = (step - 8) / 8.0
                interpolated.append((1.0 - alpha) * middle_sst + alpha * end_sst)
        sst = torch.stack(interpolated)
        current = combined_state(
            atmos[0, 0], sst, list(atmos_coords["variable"]), precip_variables
        )
        previous_states = torch.cat([previous.unsqueeze(0), current[:-1]], dim=0)
        pairs = torch.stack([previous_states, current], dim=1).unsqueeze(0)
        valid = np.array(
            [initial_coords["time"][0] + np.timedelta64(int(lead), "h") for lead in block_leads]
        )
        precip_coords = OrderedDict(
            {
                "batch": np.array([0]),
                "time": valid,
                "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
                "variable": np.array(precip_variables),
                "face": np.arange(12),
                "height": np.arange(64),
                "width": np.arange(64),
            }
        )
        diagnosed, _ = precip(pairs, precip_coords)
        block_tp = diagnosed[0, :, 0, 0]
        minimum_tp06 = min(minimum_tp06, float(block_tp.min()))
        if not torch.isfinite(block_tp).all() or float(block_tp.min()) < -1.1e-8:
            raise ValueError("DLESyM tp06 is non-finite or materially negative")

        for index, lead in enumerate(block_leads):
            if 0 < lead <= 1008:
                t2m.append(atmos[0, 0, index, t2m_index].detach().clone())
                tp06.append(torch.clamp_min(block_tp[index].detach().clone(), 0.0))
                leads.append(int(lead))
        previous = current[-1].detach().clone()
        state, coords = model._next_step_inputs(output, output_coords)

    if leads != list(range(0, 1009, 6)) or len(tp06) != 168:
        raise ValueError("V0 forecast lead sequence is invalid")
    daily_t2m = aggregate_t2m(t2m)
    daily_tp = torch.stack(tp06).reshape(42, 4, 12, 64, 64).sum(dim=1) * 1000.0
    return daily_t2m, daily_tp, leads, minimum_tp06


def build_output(
    fields: dict[str, np.ndarray],
    config: dict[str, Any],
    product: str,
    init_date: str,
    weight_hash: str | None,
) -> xr.Dataset:
    target_lat, target_lon = target_coordinates(config)
    members = config["products"][product]["members"]
    lead_days = config["forecast"]["lead_days"]
    dataset = xr.Dataset(
        {
            name: (
                ("member", "lead_day", "latitude", "longitude"),
                values,
            )
            for name, values in fields.items()
        },
        coords={
            "member": np.arange(members, dtype=np.int16),
            "latitude": target_lat,
            "longitude": target_lon,
            **period_coordinates(init_date, lead_days),
        },
        attrs={
            "run_label": config["products"][product]["run_label"],
            "model_display_name": config["products"][product]["display_name"],
            "package_uri": config["products"][product]["package_uri"],
            "native_grid": "HEALPix nside=64 (12 x 64 x 64)",
            "native_atmosphere_timestep": "6 hours",
            "forecast_length": "42 complete UTC days",
            "calendar_sha256": config["calendar"]["sha256"],
            "earth2studio_commit": config["software"]["earth2studio_commit"],
            "earth2grid_commit": config["software"]["earth2grid_commit"],
            "repository_commit": repository_commit() or "unknown",
            "precipitation_regrid_weight_sha256": weight_hash or "not applicable",
            "ensemble_definition": config["products"][product].get(
                "ensemble_definition", "single deterministic checkpoint"
            ),
        },
    )
    dataset["valid_time"].attrs.update(
        {"bounds": "forecast_period_bounds", "representation": "period_end"}
    )
    if "t2m" in dataset:
        dataset["t2m"].attrs.update(
            {
                "long_name": "2 metre temperature",
                "units": "degC",
                "cell_methods": "time: mean (trapezoidal integration of 0/6/12/18/24 h boundaries)",
                "source": "native DLESyM prognostic t2m",
                "horizontal_regrid": "Earth2Grid HEALPix-to-0.25-degree bilinear; exact canonical node selection",
            }
        )
    if "tp" in dataset:
        dataset["tp"].attrs.update(
            {
                "long_name": "total precipitation",
                "units": "mm day-1",
                "cell_methods": "time: sum (four 6-hour accumulations)",
                "source": "DLESyMv0_ISCCP_ERA5Precip tp06 diagnostic",
                "horizontal_regrid": "Earth2Grid HEALPix-to-0.25-degree then spherical conservative cell-overlap average",
            }
        )
    return dataset.transpose("member", "lead_day", "latitude", "longitude", ...)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--product", required=True, choices=["v1_t2m", "v0_tp_t2m"])
    parser.add_argument("--index", type=int)
    parser.add_argument("--init-date")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    calendar_index, init_date = select_date(config, args.index, args.init_date)
    paths = product_paths(config, args.product, init_date)
    if not args.force and existing_pair_is_valid(
        paths["output"], paths["manifest"], "output_sha256"
    ):
        print(f"validated existing forecast: {paths['output']}", flush=True)
        return
    if (paths["output"].exists() or paths["manifest"].exists()) and not args.force:
        raise RuntimeError("partial or invalid forecast exists; inspect before --force")
    if not existing_pair_is_valid(
        paths["stage"], paths["stage_manifest"], "input_sha256"
    ):
        raise RuntimeError("staged initial condition is absent or invalid")
    if not paths["inventory"].is_file():
        raise RuntimeError("checkpoint inventory is absent; run prefetch_dlesym.py first")
    if not torch.cuda.is_available():
        raise RuntimeError("DLESyM forecast requires a CUDA GPU")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"
    with xr.open_dataset(paths["stage"]) as source:
        source.load()
        coords = coords_from_dataset(source)
        initial_cpu = torch.from_numpy(source["state"].values.astype(np.float32))
    target_lat, target_lon = target_coordinates(config)
    regridder = make_regridder(device)

    if args.product == "v1_t2m":
        from earth2studio.models.px import DLESyM

        package = DLESyM.load_default_package()
        members = []
        runtimes = []
        for member, pair_values in enumerate(config["products"][args.product]["checkpoint_pairs"]):
            member_started = time.perf_counter()
            pair = (int(pair_values[0]), int(pair_values[1]))
            print(f"running V1 member={member} atmosphere={pair[0]} ocean={pair[1]}", flush=True)
            model = load_native_model(args.product, package, pair).to(device)
            daily, _ = rollout_v1_member(model, initial_cpu.to(device), coords)
            members.append(t2m_to_common(daily, regridder, target_lat, target_lon))
            runtimes.append(time.perf_counter() - member_started)
            del model, daily
            gc.collect()
            torch.cuda.empty_cache()
        fields = {"t2m": np.stack(members)}
        if float(np.max(np.std(fields["t2m"], axis=0))) <= 1e-6:
            raise ValueError("V1 checkpoint ensemble has no detectable member spread")
        weight_hash = None
        minimum_tp06 = None
    else:
        from earth2studio.models.dx import DLESyMv0_ISCCP_ERA5Precip
        from earth2studio.models.px import DLESyMv0_ISCCP_ERA5

        package = DLESyMv0_ISCCP_ERA5.load_default_package()
        model = load_native_model(args.product, package).to(device)
        precip = DLESyMv0_ISCCP_ERA5Precip.load_model(
            package, use_ttr=False
        ).to(device)
        forecast_started = time.perf_counter()
        daily_t2m, daily_tp, _, minimum_tp06 = rollout_v0(
            model, precip, initial_cpu.to(device), coords
        )
        t2m = t2m_to_common(daily_t2m, regridder, target_lat, target_lon)
        tp, weight_hash = tp_to_common(daily_tp, regridder, target_lat, target_lon)
        fields = {"t2m": t2m[None], "tp": tp[None]}
        runtimes = [time.perf_counter() - forecast_started]

    expected_shape = (
        config["products"][args.product]["members"],
        42,
        27,
        27,
    )
    stats = {}
    for name, values in fields.items():
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"{name} has invalid shape or non-finite values")
        stats[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
        }
    if "t2m" in stats and not (
        stats["t2m"]["minimum"] > -100.0
        and stats["t2m"]["maximum"] < 70.0
    ):
        raise ValueError(f"implausible T2M range: {stats['t2m']}")
    if "tp" in stats and not (stats["tp"]["minimum"] >= 0.0 and stats["tp"]["maximum"] < 2000.0):
        raise ValueError(f"implausible TP range: {stats['tp']}")

    dataset = build_output(fields, config, args.product, init_date, weight_hash)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["output"].with_name(f".{paths['output'].name}.{os.getpid()}.tmp")
    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (1, 7, 27, 27),
        }
        for name in fields
    }
    dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    with xr.open_dataset(temporary) as check:
        if dict(check.sizes) != {
            "member": expected_shape[0],
            "lead_day": 42,
            "latitude": 27,
            "longitude": 27,
            "bounds": 2,
        }:
            raise ValueError(f"written dimensions are invalid: {dict(check.sizes)}")
        check.load()
    os.replace(temporary, paths["output"])
    output_hash = sha256_file(paths["output"])
    elapsed = time.perf_counter() - started
    stage_record = json.loads(paths["stage_manifest"].read_text(encoding="utf-8"))
    inventory_hash = sha256_file(paths["inventory"])
    manifest = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "product": args.product,
        "run_label": config["products"][args.product]["run_label"],
        "calendar_index": calendar_index,
        "init_date": init_date,
        "calendar_sha256": config["calendar"]["sha256"],
        "package_uri": config["products"][args.product]["package_uri"],
        "package_revision": config["products"][args.product]["package_revision"],
        "checkpoint_inventory_path": str(paths["inventory"]),
        "checkpoint_inventory_sha256": inventory_hash,
        "input_path": str(paths["stage"]),
        "input_sha256": stage_record["input_sha256"],
        "output_path": str(paths["output"]),
        "output_size_bytes": paths["output"].stat().st_size,
        "output_sha256": output_hash,
        "fields": list(fields),
        "member_count": expected_shape[0],
        "ensemble_definition": config["products"][args.product].get(
            "ensemble_definition", "single deterministic checkpoint"
        ),
        "checkpoint_pairs": config["products"][args.product].get("checkpoint_pairs"),
        "lead_days": 42,
        "daily_t2m_method": "five-boundary trapezoidal integration with weights 0.5/1/1/1/0.5 divided by 4",
        "daily_tp_method": "sum of diagnostics ending +6/+12/+18/+24 h; metres multiplied by 1000",
        "minimum_native_tp06_m": minimum_tp06,
        "precipitation_regrid_weight_sha256": weight_hash,
        "statistics": stats,
        "member_runtime_seconds": runtimes,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "repository_commit": repository_commit(),
        "earth2studio_commit": config["software"]["earth2studio_commit"],
        "earth2grid_commit": config["software"]["earth2grid_commit"],
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    }
    write_json_atomic(manifest, paths["manifest"])
    if paths["failed"].exists():
        paths["failed"].unlink()
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"DLESyM run failed: {type(error).__name__}: {error}", flush=True)
        raise
