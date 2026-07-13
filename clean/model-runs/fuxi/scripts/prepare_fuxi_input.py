#!/usr/bin/env python3
"""Build and validate one two-day, 76-channel FuXi-S2S ERA5 input."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from earth2studio.data import ARCO


LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PRESSURE_PREFIXES = ["z", "t", "u", "v", "q"]
SURFACE_MAP = [
    ("t2m", "t2m"),
    ("d2m", "d2m"),
    ("sst", "sst"),
    ("ttr", "ttr"),
    ("u10m", "10u"),
    ("v10m", "10v"),
    ("u100m", "100u"),
    ("v100m", "100v"),
    ("msl", "msl"),
    ("tcwv", "tcwv"),
    ("tp", "tp"),
]


def expected_channels() -> list[str]:
    channels = []
    for prefix in PRESSURE_PREFIXES:
        channels.extend(f"{prefix}{level}" for level in LEVELS)
    channels.extend(output_name for _, output_name in SURFACE_MAP)
    return channels


def expected_latitudes() -> np.ndarray:
    return np.linspace(90.0, -90.0, 121, dtype=np.float32)


def expected_longitudes() -> np.ndarray:
    return np.arange(0.0, 360.0, 1.5, dtype=np.float32)


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
            raise ValueError("FuXi input latitude grid is not the expected 1.5 degree grid")
        if not np.allclose(data.lon.values, expected_longitudes()):
            raise ValueError("FuXi input longitude grid is not the expected 1.5 degree grid")
        expected_times = pd.DatetimeIndex(
            [init_date - pd.Timedelta(days=1), init_date]
        ).values
        if not np.array_equal(data.time.values.astype("datetime64[ns]"), expected_times):
            raise ValueError(f"unexpected input times: {data.time.values}")
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
            "size_bytes": path.stat().st_size,
        }
    finally:
        data.close()


def fetch_input(init_date: pd.Timestamp, timeout: int, cache: bool) -> xr.DataArray:
    times = [
        (init_date - pd.Timedelta(days=1)).to_pydatetime(),
        init_date.to_pydatetime(),
    ]
    source = ARCO(cache=cache, verbose=False, async_timeout=timeout)
    arrays: list[np.ndarray] = []
    channels: list[str] = []
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None

    for prefix in PRESSURE_PREFIXES:
        variables = [f"{prefix}{level}" for level in LEVELS]
        print(f"fetch {prefix}: {variables[0]}..{variables[-1]}", flush=True)
        data = source(times, variables)
        data = data.isel(lat=slice(None, None, 6), lon=slice(None, None, 6))
        arrays.append(data.values.astype(np.float32, copy=False))
        channels.extend(variables)
        latitudes = data.lat.values.astype(np.float32)
        longitudes = data.lon.values.astype(np.float32)

    surface_variables = [source_name for source_name, _ in SURFACE_MAP]
    surface_channels = [output_name for _, output_name in SURFACE_MAP]
    print(f"fetch surface: {surface_variables[0]}..{surface_variables[-1]}", flush=True)
    surface = source(times, surface_variables)
    surface = surface.isel(lat=slice(None, None, 6), lon=slice(None, None, 6))
    arrays.append(surface.values.astype(np.float32, copy=False))
    channels.extend(surface_channels)
    latitudes = surface.lat.values.astype(np.float32)
    longitudes = surface.lon.values.astype(np.float32)

    values = np.concatenate(arrays, axis=1)
    channel_index = {name: index for index, name in enumerate(channels)}
    values[:, channel_index["tp"]] = np.clip(
        values[:, channel_index["tp"]] * 1000.0, 0.0, 1000.0
    )
    values[:, channel_index["ttr"]] /= 3600.0

    return xr.DataArray(
        values,
        dims=("time", "channel", "lat", "lon"),
        coords={
            "time": pd.to_datetime(times),
            "channel": channels,
            "lat": latitudes,
            "lon": longitudes,
        },
        name="data",
        attrs={
            "source": "ARCO ERA5 via earth2studio.data.ARCO",
            "init_date": init_date.strftime("%Y-%m-%d"),
            "tp_preprocessing": "m to mm, clipped to [0, 1000]",
            "ttr_preprocessing": "divided by 3600",
        },
    )


def quarantine(path: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.invalid.{timestamp}")
    path.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="initialization date YYYYMMDD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--persistent-cache", action="store_true")
    args = parser.parse_args()

    init_date = pd.Timestamp(args.date)
    if init_date.strftime("%Y%m%d") != args.date:
        raise SystemExit("date must use YYYYMMDD format")

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
    last_error: Exception | None = None
    for attempt in range(1, args.attempts + 1):
        print(f"input attempt {attempt}/{args.attempts}: {args.date}", flush=True)
        try:
            data = fetch_input(init_date, args.timeout, args.persistent_cache)
            data.to_netcdf(temporary)
            validate_input(temporary, init_date)
            temporary.replace(args.output)
            print(f"wrote valid input: {args.output}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            temporary.unlink(missing_ok=True)
            print(f"input attempt failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt < args.attempts:
                time.sleep(60 * attempt)

    raise SystemExit(f"failed to build FuXi input for {args.date}: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())
