#!/usr/bin/env python3
"""Assemble one FuXi-S2S input from staged ERA5 daily statistics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "clean/config/fuxi_operational_2020_2025.json"
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PRESSURE_FIELDS = [
    ("z", "geopotential"),
    ("t", "temperature"),
    ("u", "u_component_of_wind"),
    ("v", "v_component_of_wind"),
    ("q", "specific_humidity"),
]
SURFACE_FIELDS = [
    ("t2m", "2m_temperature"),
    ("d2m", "2m_dewpoint_temperature"),
    ("sst", "sea_surface_temperature"),
    ("ttr", "top_net_thermal_radiation"),
    ("10u", "10m_u_component_of_wind"),
    ("10v", "10m_v_component_of_wind"),
    ("100u", "100m_u_component_of_wind"),
    ("100v", "100m_v_component_of_wind"),
    ("msl", "mean_sea_level_pressure"),
    ("tcwv", "total_column_water_vapour"),
    ("tp", "total_precipitation"),
]


def expected_channels() -> list[str]:
    channels: list[str] = []
    for prefix, _ in PRESSURE_FIELDS:
        channels.extend(f"{prefix}{level}" for level in LEVELS)
    channels.extend(short_name for short_name, _ in SURFACE_FIELDS)
    return channels


def expected_latitudes() -> np.ndarray:
    return np.linspace(90.0, -90.0, 121, dtype=np.float32)


def expected_longitudes() -> np.ndarray:
    return np.arange(0.0, 360.0, 1.5, dtype=np.float32)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def staging_paths(config: dict[str, Any], day: pd.Timestamp) -> dict[str, Path]:
    root = (
        Path(config["storage_root"])
        / config["input"]["staging_subdirectory"]
        / "monthly"
        / day.strftime("%Y%m")
    )
    return {
        component: root / f"{component}.nc" for component in ("pressure", "surface")
    }


def canonicalize_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    rename = {}
    for source, target in (
        ("valid_time", "time"),
        ("date", "time"),
        ("pressure_level", "level"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ):
        if source in dataset.dims or source in dataset.coords:
            if target not in dataset.dims and target not in dataset.coords:
                rename[source] = target
    dataset = dataset.rename(rename)
    for dimension in ("number", "expver"):
        if dimension in dataset.dims and dataset.sizes[dimension] == 1:
            dataset = dataset.isel({dimension: 0}, drop=True)
    if "lon" in dataset.coords:
        dataset = dataset.assign_coords(lon=np.mod(dataset.lon, 360.0)).sortby("lon")
    if "lat" in dataset.coords and dataset.lat.values[0] < dataset.lat.values[-1]:
        dataset = dataset.sortby("lat", ascending=False)
    return dataset


def find_variable(dataset: xr.Dataset, short_name: str, long_name: str) -> xr.DataArray:
    aliases = {
        "10u": ("u10",),
        "10v": ("v10",),
        "100u": ("u100",),
        "100v": ("v100",),
    }
    for candidate in (short_name, long_name, *aliases.get(short_name, ())):
        if candidate in dataset.data_vars:
            return dataset[candidate]
    raise KeyError(
        f"missing {short_name}/{long_name}; available variables: "
        f"{sorted(dataset.data_vars)}"
    )


def select_days(data: xr.DataArray, days: pd.DatetimeIndex) -> xr.DataArray:
    if "time" not in data.dims:
        raise ValueError(f"{data.name} has no time dimension: {data.dims}")
    normalized = pd.DatetimeIndex(data.time.values).normalize()
    if normalized.duplicated().any():
        raise ValueError(f"{data.name} staging data has duplicate daily timestamps")
    data = data.assign_coords(time=normalized.values)
    missing = days.difference(normalized)
    if len(missing):
        raise ValueError(f"{data.name} staging data is missing {missing.tolist()}")
    return data.sel(time=days.values)


def convert_tp(values: np.ndarray, units: str) -> np.ndarray:
    normalized = units.lower().replace(" ", "")
    if normalized in {"m", "mofwaterequivalent", "metres"} or not normalized:
        return np.clip(values * 1000.0, 0.0, 1000.0)
    if "mm" in normalized:
        return np.clip(values, 0.0, 1000.0)
    raise ValueError(f"unsupported ERA5 TP units: {units!r}")


def convert_ttr(values: np.ndarray, units: str) -> np.ndarray:
    normalized = units.lower().replace(" ", "")
    if "j" in normalized or not normalized:
        return values / 3600.0
    if "w" in normalized:
        return values
    raise ValueError(f"unsupported ERA5 TTR units: {units!r}")


def build_input(
    init_date: pd.Timestamp,
    config: dict[str, Any],
) -> xr.DataArray:
    days = pd.DatetimeIndex([init_date - pd.Timedelta(days=1), init_date])
    paths_by_day = {day: staging_paths(config, day) for day in days}
    all_paths = sorted(
        {path for paths in paths_by_day.values() for path in paths.values()}
    )
    missing_files = [str(path) for path in all_paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"ERA5 daily staging is incomplete: {missing_files}")

    opened = {
        path: canonicalize_coordinates(xr.open_dataset(path)) for path in all_paths
    }
    try:
        arrays: list[np.ndarray] = []
        channels: list[str] = []
        for prefix, long_name in PRESSURE_FIELDS:
            daily_values = []
            for day in days:
                path = paths_by_day[day]["pressure"]
                field = find_variable(opened[path], prefix, long_name)
                field = select_days(field, pd.DatetimeIndex([day])).sel(level=LEVELS)
                field = field.transpose("time", "level", "lat", "lon")
                daily_values.append(field.values.astype(np.float32, copy=False))
            arrays.append(np.concatenate(daily_values, axis=0))
            channels.extend(f"{prefix}{level}" for level in LEVELS)

        for short_name, long_name in SURFACE_FIELDS:
            daily_values = []
            for day in days:
                path = paths_by_day[day]["surface"]
                field = find_variable(opened[path], short_name, long_name)
                field = select_days(field, pd.DatetimeIndex([day]))
                field = field.transpose("time", "lat", "lon")
                values = field.values.astype(np.float32, copy=False)
                units = str(field.attrs.get("units", ""))
                if short_name == "tp":
                    values = convert_tp(values, units)
                elif short_name == "ttr":
                    values = convert_ttr(values, units)
                daily_values.append(values)
            arrays.append(np.concatenate(daily_values, axis=0)[:, None])
            channels.append(short_name)

        values = np.concatenate(arrays, axis=1)
        return xr.DataArray(
            values,
            dims=("time", "channel", "lat", "lon"),
            coords={
                "time": days.values,
                "channel": channels,
                "lat": expected_latitudes(),
                "lon": expected_longitudes(),
            },
            name="data",
            attrs={
                "source": config["input"]["source"],
                "init_date": init_date.strftime("%Y-%m-%d"),
                "temporal_statistic": "UTC daily mean derived from 1-hourly ERA5",
                "source_days": f"{days[0]:%Y-%m-%d},{days[1]:%Y-%m-%d}",
                "tp_preprocessing": "daily mean of hourly ERA5 accumulations; m to mm h-1",
                "ttr_preprocessing": "daily mean of hourly ERA5 accumulations; J m-2 divided by 3600 to W m-2",
                "staging_files": ",".join(str(path) for path in all_paths),
                "staging_request_sets": ",".join(
                    sorted(
                        str(opened[path].attrs.get("request_set_sha256", ""))
                        for path in all_paths
                    )
                ),
            },
        )
    finally:
        for dataset in opened.values():
            dataset.close()


def validate_input(path: Path, init_date: pd.Timestamp) -> dict[str, object]:
    data = xr.open_dataarray(path)
    try:
        if data.dims != ("time", "channel", "lat", "lon"):
            raise ValueError(f"unexpected input dimensions: {data.dims}")
        if data.shape != (2, 76, 121, 240):
            raise ValueError(f"unexpected input shape: {data.shape}")
        channels = [str(value) for value in data.channel.values.tolist()]
        if channels != expected_channels():
            raise ValueError("FuXi input channel order does not match the checkpoint")
        if not np.allclose(data.lat.values, expected_latitudes()):
            raise ValueError(
                "FuXi input latitude grid is not the expected 1.5 degree grid"
            )
        if not np.allclose(data.lon.values, expected_longitudes()):
            raise ValueError(
                "FuXi input longitude grid is not the expected 1.5 degree grid"
            )
        expected_times = pd.DatetimeIndex(
            [init_date - pd.Timedelta(days=1), init_date]
        ).values
        if not np.array_equal(
            data.time.values.astype("datetime64[ns]"), expected_times
        ):
            raise ValueError(f"unexpected input times: {data.time.values}")
        if (
            data.attrs.get("temporal_statistic")
            != "UTC daily mean derived from 1-hourly ERA5"
        ):
            raise ValueError(
                "input does not declare the required ERA5 daily-mean statistic"
            )
        for channel in channels:
            values = data.sel(channel=channel).values
            if channel == "sst":
                if not np.isfinite(values).any():
                    raise ValueError("SST has no finite ocean values")
            elif not np.isfinite(values).all():
                raise ValueError(f"{channel} contains missing or infinite values")
        return {
            "shape": list(data.shape),
            "time_start": str(pd.Timestamp(data.time.values[0])),
            "time_end": str(pd.Timestamp(data.time.values[-1])),
            "temporal_statistic": data.attrs["temporal_statistic"],
            "size_bytes": path.stat().st_size,
        }
    finally:
        data.close()


def quarantine(path: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.invalid.{timestamp}")
    path.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="initialization date YYYYMMDD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    init_date = pd.Timestamp(args.date)
    if init_date.strftime("%Y%m%d") != args.date:
        raise SystemExit("date must use YYYYMMDD format")
    config = load_config(args.config)

    if args.output.exists():
        try:
            details = validate_input(args.output, init_date)
            print(f"existing input valid: {args.output} {details}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            moved = quarantine(args.output)
            print(f"quarantined invalid input at {moved}: {exc}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    data = build_input(init_date, config)
    data.to_netcdf(temporary)
    details = validate_input(temporary, init_date)
    temporary.replace(args.output)
    print(f"wrote valid input: {args.output} {details}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
