#!/usr/bin/env python3
"""
Download ERA5 FuXi-S2S inputs for JJAS 2025 from ARCO ERA5.

Output:
  /storage/raj.ayush/All_Model_Data/fuxi/jjas2025/inputs/{YYYYMMDD}/input.nc

Each input uses the previous day and init day at 00 UTC on the global 1.5 deg
grid expected by FuXi-S2S. Existing input.nc files are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent / "FuXi-S2S"))
from data_util import make_input


ARCO_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_INPUT_DIR = Path("/storage/raj.ayush/All_Model_Data/fuxi/jjas2025/inputs")
LOG_DIR = Path(__file__).parent / "logs"

DATE_START = "2025-06-01"
DATE_END = "2025-09-30"

PL_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
SFC_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "sea_surface_temperature",
    "top_net_thermal_radiation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
    "total_precipitation",
]
PL_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def ymd(date: pd.Timestamp) -> str:
    return f"{date:%Y%m%d}"


def input_done(input_dir: Path, date: pd.Timestamp) -> bool:
    path = input_dir / ymd(date) / "input.nc"
    return path.exists() and path.stat().st_size > 0


def parse_dates(args) -> list[pd.Timestamp]:
    if args.date:
        return [pd.Timestamp(args.date)]
    return list(pd.date_range(args.start, args.end, freq="D"))


def save_dataarray(da: xr.DataArray, path: Path) -> None:
    da = da.astype("float32")
    da.name = "data"
    da.to_netcdf(str(path))


def download_one(ds_arco: xr.Dataset, input_dir: Path, init_date: pd.Timestamp, log: logging.Logger) -> bool:
    date_str = ymd(init_date)
    date_dir = input_dir / date_str
    out_file = date_dir / "input.nc"
    tmp_file = date_dir / "input.nc.tmp"

    if input_done(input_dir, init_date):
        log.info("SKIP  %s input.nc exists", date_str)
        return True

    date_dir.mkdir(parents=True, exist_ok=True)
    prev_day = init_date - pd.Timedelta(days=1)
    times = [
        prev_day.strftime("%Y-%m-%dT00:00:00"),
        init_date.strftime("%Y-%m-%dT00:00:00"),
    ]

    log.info("START %s ARCO times=%s,%s", date_str, times[0], times[1])
    for name in PL_VARS:
        path = date_dir / f"{name}.nc"
        if path.exists() and path.stat().st_size > 0:
            continue
        log.info("  pressure %s", name)
        da = ds_arco[name].sel(
            time=times,
            level=PL_LEVELS,
            latitude=slice(None, None, 6),
            longitude=slice(None, None, 6),
        ).compute()
        save_dataarray(da, path)

    for name in SFC_VARS:
        path = date_dir / f"{name}.nc"
        if path.exists() and path.stat().st_size > 0:
            continue
        log.info("  surface %s", name)
        da = ds_arco[name].sel(
            time=times,
            latitude=slice(None, None, 6),
            longitude=slice(None, None, 6),
        ).compute()
        if "level" in da.dims:
            da = da.isel(level=0)
        da = da.assign_coords(level=1000).expand_dims("level")
        save_dataarray(da, path)

    log.info("  combine -> input.nc")
    if tmp_file.exists():
        tmp_file.unlink()
    inp = make_input(str(date_dir))
    if "latitude" in inp.dims:
        inp = inp.rename({"latitude": "lat", "longitude": "lon"})
    clean_channels = [c.replace(".0", "") if str(c).endswith(".0") else str(c) for c in inp.channel.values.tolist()]
    inp = inp.assign_coords(channel=clean_channels)
    inp.to_netcdf(str(tmp_file))
    tmp_file.replace(out_file)

    for path in date_dir.iterdir():
        if path.suffix == ".nc" and path.name != "input.nc":
            path.unlink()

    mb = out_file.stat().st_size / 1024**2
    log.info("DONE  %s %.0f MB -> %s", date_str, mb, out_file)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download FuXi JJAS 2025 inputs from ARCO ERA5")
    parser.add_argument("--date", type=str, default=None, help="Single init date YYYYMMDD")
    parser.add_argument("--start", type=str, default=DATE_START)
    parser.add_argument("--end", type=str, default=DATE_END)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--check", action="store_true", help="Only report missing inputs")
    args = parser.parse_args()

    dates = parse_dates(args)
    pending = [date for date in dates if not input_done(args.input_dir, date)]

    log_file = LOG_DIR / f"fuxi_inputs_arco_jjas2025_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)
    log.info("=" * 65)
    log.info("FuXi-S2S ERA5 Input Download via ARCO - JJAS 2025")
    log.info("  ARCO store : %s", ARCO_ZARR)
    log.info("  Output dir : %s", args.input_dir)
    log.info("  Dates      : %d (%s -> %s)", len(dates), dates[0].date(), dates[-1].date())
    log.info("  Done       : %d", len(dates) - len(pending))
    log.info("  Pending    : %d", len(pending))
    log.info("=" * 65)

    if args.check:
        for date in pending:
            log.info("MISSING %s", ymd(date))
        return 0

    if not pending:
        log.info("All input.nc files exist - nothing to do.")
        return 0

    args.input_dir.mkdir(parents=True, exist_ok=True)
    log.info("Opening ARCO ERA5 on GCS...")
    ds_arco = xr.open_zarr(ARCO_ZARR, storage_options={"token": "anon"})

    failed = []
    for i, date in enumerate(pending, 1):
        try:
            download_one(ds_arco, args.input_dir, date, log)
        except Exception as exc:  # keep long runs resumable
            log.exception("FAIL  %s: %s", ymd(date), exc)
            failed.append(ymd(date))
        log.info("Progress %d/%d failed=%d", i, len(pending), len(failed))

    log.info("=" * 65)
    log.info("Done. Success=%d Failed=%d", len(pending) - len(failed), len(failed))
    for date_str in failed:
        log.info("  FAILED: %s", date_str)
    log.info("=" * 65)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
