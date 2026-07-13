#!/usr/bin/env python3
"""Download and plot a one-init, one-day S2S physics-model smoke test."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2s-smoke")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/s2s-smoke-cache")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DATASET = "s2s-forecasts"
AREA = [40.0, 60.0, 0.0, 100.0]
FTYPES = {"cf": "control_forecast", "pf": "perturbed_forecast"}
DEFAULT_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/smoke_tests/physics_models"
)
PROVIDERS = ("ecmwf", "ukmo", "ncep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-date", type=dt.date.fromisoformat, default=dt.date(2024, 6, 2))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tasks(init_date: dt.date, root: Path) -> list[dict[str, Any]]:
    common = {
        "year": str(init_date.year),
        "month": f"{init_date.month:02d}",
        "day": f"{init_date.day:02d}",
        "time": "00:00",
        "level_type": "single_level",
        "area": AREA,
        "data_format": "grib",
    }
    result = []
    for provider in PROVIDERS:
        if provider in ("ecmwf", "ukmo"):
            fields = (
                ("tp", "total_precipitation", ["24"]),
                ("t2m", "2_m_temperature", ["0_24"]),
            )
        else:
            fields = (
                ("tp", "total_precipitation", ["24"]),
                ("mx2t6", "maximum_2_m_temperature_in_the_last_6_hours", ["6", "12", "18", "24"]),
                ("mn2t6", "minimum_2_m_temperature_in_the_last_6_hours", ["6", "12", "18", "24"]),
            )
        for label, variable, leadtime in fields:
            for short_type, api_type in FTYPES.items():
                request = {
                    **common,
                    "origin": provider,
                    "forecast_type": api_type,
                    "variable": variable,
                    "leadtime_hour": leadtime,
                }
                result.append(
                    {
                        "provider": provider,
                        "field": label,
                        "forecast_type": short_type,
                        "request": request,
                        "path": root / f"{init_date:%Y%m%d}" / provider / f"{label}_{short_type}.grib",
                    }
                )
    return result


def request_hash(request: dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def find_variable(path: Path, short_name: str) -> xr.DataArray:
    import cfgrib

    datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    try:
        for dataset in datasets:
            if short_name in dataset.data_vars:
                return dataset[short_name].load()
        found = sorted({name for dataset in datasets for name in dataset.data_vars})
        raise ValueError(f"{path}: expected {short_name}, found {found}")
    finally:
        for dataset in datasets:
            dataset.close()


def spatial_names(array: xr.DataArray) -> tuple[str, str]:
    latitude = "latitude" if "latitude" in array.dims else "lat"
    longitude = "longitude" if "longitude" in array.dims else "lon"
    if latitude not in array.dims or longitude not in array.dims:
        raise ValueError(f"missing spatial dimensions in {array.dims}")
    return latitude, longitude


def member_field(path: Path, short_name: str, average_steps: bool = False) -> xr.DataArray:
    array = find_variable(path, short_name)
    latitude, longitude = spatial_names(array)
    if "step" in array.dims:
        array = array.mean("step") if average_steps else array.squeeze("step", drop=True)
    for dim in tuple(array.dims):
        if dim not in (latitude, longitude, "number"):
            if array.sizes[dim] != 1:
                raise ValueError(f"unexpected non-singleton dimension {dim}={array.sizes[dim]}")
            array = array.squeeze(dim, drop=True)
    if "number" in array.dims:
        array = array.rename(number="member")
    else:
        array = array.expand_dims(member=[-1])
    return array


def ensemble_field(root: Path, provider: str, field: str, average_steps: bool = False) -> xr.DataArray:
    control = member_field(root / provider / f"{field}_cf.grib", field, average_steps)
    perturbed = member_field(root / provider / f"{field}_pf.grib", field, average_steps)
    return xr.concat((control, perturbed), dim="member", join="exact").mean("member")


def qc_file(task: dict[str, Any]) -> dict[str, Any]:
    array = find_variable(task["path"], task["field"])
    latitude, longitude = spatial_names(array)
    return {
        "variable": task["field"],
        "units": array.attrs.get("units"),
        "members": int(array.sizes.get("number", 1)),
        "steps": int(array.sizes.get("step", 1)),
        "latitude_points": int(array.sizes[latitude]),
        "longitude_points": int(array.sizes[longitude]),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def download(client: Any, task: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    path = task["path"]
    record = {
        "provider": task["provider"],
        "field": task["field"],
        "forecast_type": task["forecast_type"],
        "path": str(path),
        "request_hash": request_hash(task["request"]),
        "request": task["request"],
    }
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        record["status"] = "existing_valid"
        record.update(qc_file(task))
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    client.retrieve(DATASET, task["request"], str(temporary))
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError(f"empty download: {temporary}")
    temporary.replace(path)
    record["status"] = "downloaded_valid"
    record["size_bytes"] = path.stat().st_size
    record.update(qc_file(task))
    return record


def plot_comparison(run_root: Path, init_date: dt.date) -> tuple[Path, dict[str, Any]]:
    precipitation = {provider: ensemble_field(run_root, provider, "tp") for provider in PROVIDERS}
    temperature = {
        "ecmwf": ensemble_field(run_root, "ecmwf", "t2m") - 273.15,
        "ukmo": ensemble_field(run_root, "ukmo", "t2m") - 273.15,
    }
    ncep_max = ensemble_field(run_root, "ncep", "mx2t6", average_steps=True)
    ncep_min = ensemble_field(run_root, "ncep", "mn2t6", average_steps=True)
    temperature["ncep"] = (ncep_max + ncep_min) / 2.0 - 273.15

    tp_max = max(float(field.max()) for field in precipitation.values())
    t_min = min(float(field.min()) for field in temperature.values())
    t_max = max(float(field.max()) for field in temperature.values())
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    tp_image = temp_image = None
    labels = {"ecmwf": "ECMWF", "ukmo": "UKMO", "ncep": "NCEP"}
    for column, provider in enumerate(PROVIDERS):
        tp = precipitation[provider]
        latitude, longitude = spatial_names(tp)
        tp_image = axes[0, column].pcolormesh(
            tp[longitude], tp[latitude], tp, shading="auto", cmap="YlGnBu", vmin=0, vmax=tp_max
        )
        axes[0, column].set_title(labels[provider])
        temp = temperature[provider]
        latitude, longitude = spatial_names(temp)
        temp_image = axes[1, column].pcolormesh(
            temp[longitude], temp[latitude], temp, shading="auto", cmap="coolwarm", vmin=t_min, vmax=t_max
        )
        axes[1, column].set_title("Daily mean" if provider != "ncep" else "6-hour extrema proxy")
        for row in (0, 1):
            axes[row, column].set_xlabel("Longitude")
            axes[row, column].set_ylabel("Latitude")
            axes[row, column].set_xlim(60, 100)
            axes[row, column].set_ylim(0, 40)
    figure.colorbar(tp_image, ax=axes[0, :], label="Day-1 precipitation (mm)", shrink=0.9)
    figure.colorbar(temp_image, ax=axes[1, :], label="Day-1 temperature (degC)", shrink=0.9)
    figure.suptitle(f"Physics-model S2S smoke test: {init_date:%Y-%m-%d} 00Z, ensemble mean")
    output = run_root / "physics_models_day1_comparison.png"
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    summary = {
        provider: {
            "tp_domain_mean_mm": float(precipitation[provider].mean()),
            "tp_domain_max_mm": float(precipitation[provider].max()),
            "temperature_domain_mean_degc": float(temperature[provider].mean()),
            "temperature_domain_min_degc": float(temperature[provider].min()),
            "temperature_domain_max_degc": float(temperature[provider].max()),
        }
        for provider in PROVIDERS
    }
    return output, summary


def main() -> int:
    args = parse_args()
    run_root = args.output_root / f"{args.init_date:%Y%m%d}"
    all_tasks = tasks(args.init_date, args.output_root)
    records = []
    if not args.skip_download:
        import cdsapi

        client = cdsapi.Client(quiet=True)
        for index, task in enumerate(all_tasks, 1):
            print(f"[{index:02d}/{len(all_tasks)}] {task['provider']} {task['field']} {task['forecast_type']}", flush=True)
            records.append(download(client, task, args.overwrite))
        manifest = run_root / "manifest.json"
        manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing = [str(task["path"]) for task in all_tasks if not task["path"].exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} smoke-test files; first: {missing[0]}")
    figure, summary = plot_comparison(run_root, args.init_date)
    summary_path = run_root / "field_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(figure), "summary": str(summary_path), "files": len(all_tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
