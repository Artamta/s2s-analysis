#!/usr/bin/env python3
"""
01_download_inputs.py
=====================
Download ERA5 from ARCO-ERA5 (GCS, free, no API key) for JFM 2026.

Produces TWO outputs per init date:
  1. FuXi input  — global 1.5°, 76 channels, 2-day window (prev+init)
                   → inputs/{YYYYMMDD}/input.nc
  2. Ground truth — India 0.25°, key verification vars, daily mean at 00 UTC
                   → ground_truth/{YYYYMMDD}.nc
                   vars: z500 (gpm), t2m (K), tp (mm/day), msl (Pa)

Source : gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
         (1959–2023 Jan 10, 0.25°, 1-hourly)

NOTE: ARCO-ERA5 ends 2023-01-10. For JFM 2026, we fall back to the
      CDS reanalysis-era5-single-levels / pressure-levels API for ground truth,
      and to the existing input.nc files already downloaded for FuXi inputs.

Usage
-----
  # Check what's available and what's missing (no download)
  python 01_download_inputs.py --check

  # Download FuXi inputs only (from ARCO-ERA5 / CDS)
  python 01_download_inputs.py --fuxi-only

  # Download ground truth only (from CDS)
  python 01_download_inputs.py --gt-only

  # Both (default)
  python 01_download_inputs.py

  # Single date
  python 01_download_inputs.py --date 20260115
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent / "FuXi-S2S"))
from data_util import make_input

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/storage/raj.ayush/All_Model_Data/fuxi/jfm2026")
INPUT_DIR = BASE_DIR / "inputs"        # FuXi initial conditions
GT_DIR    = BASE_DIR / "ground_truth"  # ERA5 verification data
LOG_DIR   = Path(__file__).parent / "logs"

# ── DATE RANGE ────────────────────────────────────────────────────────────────
DATE_START = "2026-01-01"
DATE_END   = "2026-03-31"

# Need ground truth up to last init + 46 lead days
GT_END = "2026-05-16"

# ── ERA5 CONFIG ───────────────────────────────────────────────────────────────
ARCO_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

PL_NAMES  = ["geopotential", "temperature", "u_component_of_wind",
              "v_component_of_wind", "specific_humidity"]
SFC_NAMES = ["2m_temperature", "2m_dewpoint_temperature", "sea_surface_temperature",
              "top_net_thermal_radiation", "10m_u_component_of_wind",
              "10m_v_component_of_wind", "100m_u_component_of_wind",
              "100m_v_component_of_wind", "mean_sea_level_pressure",
              "total_column_water_vapour", "total_precipitation"]
PL_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# India verification domain (matches ECMWF download)
GT_LAT_SLICE = slice(40, 0)    # N→S (ARCO lat goes 90→-90)
GT_LON_SLICE = slice(60, 100)
GT_GRID_DEG  = 0.25            # keep native 0.25°


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


# ── STATUS CHECKS ─────────────────────────────────────────────────────────────
def fuxi_input_done(date: pd.Timestamp) -> bool:
    return (INPUT_DIR / f"{date:%Y%m%d}" / "input.nc").exists()


def gt_done(date: pd.Timestamp) -> bool:
    return (GT_DIR / f"{date:%Y%m%d}.nc").exists()


# ── FUXI INPUT DOWNLOAD ───────────────────────────────────────────────────────
def download_fuxi_input(ds_arco: xr.Dataset, init_date: pd.Timestamp,
                        log: logging.Logger) -> bool:
    date_str = f"{init_date:%Y%m%d}"
    date_dir = INPUT_DIR / date_str
    out_file = date_dir / "input.nc"

    if out_file.exists():
        log.info(f"FUXI SKIP  {date_str}  (input.nc exists)")
        return True

    date_dir.mkdir(parents=True, exist_ok=True)
    prev_day = init_date - pd.Timedelta(days=1)
    times = [prev_day.strftime("%Y-%m-%dT00:00:00"),
             init_date.strftime("%Y-%m-%dT00:00:00")]

    log.info(f"FUXI START {date_str}  (t={times[0]} + {times[1]})")
    # Subsample 0.25° → 1.5° globally (every 6th pixel)
    ds = ds_arco.sel(time=times,
                     latitude=slice(None, None, 6),
                     longitude=slice(None, None, 6))

    for name in PL_NAMES:
        p = date_dir / f"{name}.nc"
        if not p.exists():
            ds[name].sel(level=PL_LEVELS).compute().to_netcdf(str(p))

    for name in SFC_NAMES:
        p = date_dir / f"{name}.nc"
        if not p.exists():
            da = ds[name].compute()
            if "level" in da.dims:
                da = da.isel(level=0)
            da = da.assign_coords(level=1000).expand_dims("level")
            da.to_netcdf(str(p))

    inp = make_input(str(date_dir))
    if "latitude" in inp.dims:
        inp = inp.rename({"latitude": "lat", "longitude": "lon"})
    inp.to_netcdf(str(out_file))

    for f in date_dir.iterdir():
        if f.suffix == ".nc" and f.name != "input.nc":
            f.unlink()

    log.info(f"FUXI DONE  {date_str}  → {out_file}")
    return True


# ── GROUND TRUTH DOWNLOAD ─────────────────────────────────────────────────────
def download_ground_truth(ds_arco: xr.Dataset, date: pd.Timestamp,
                          log: logging.Logger) -> bool:
    date_str = f"{date:%Y%m%d}"
    out_file = GT_DIR / f"{date_str}.nc"

    if out_file.exists():
        log.info(f"GT   SKIP  {date_str}")
        return True

    GT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"GT   START {date_str}")

    time_str = date.strftime("%Y-%m-%dT00:00:00")
    ds = ds_arco.sel(
        time=time_str,
        latitude=GT_LAT_SLICE,
        longitude=GT_LON_SLICE,
    )

    out = xr.Dataset()

    # z500 — geopotential at 500 hPa → divide by 9.80665 to get gpm
    z = ds["geopotential"].sel(level=500).compute()
    out["z500"] = (z / 9.80665).astype("float32")
    out["z500"].attrs = {"units": "gpm", "long_name": "Geopotential height 500 hPa"}

    # t2m
    out["t2m"] = ds["2m_temperature"].compute().astype("float32")
    out["t2m"].attrs = {"units": "K", "long_name": "2m temperature"}

    # tp — ARCO has total_precipitation in m; convert to mm
    tp = ds["total_precipitation"].compute()
    out["tp"] = (tp * 1000).clip(0).astype("float32")
    out["tp"].attrs = {"units": "mm", "long_name": "Total precipitation"}

    # msl
    out["msl"] = ds["mean_sea_level_pressure"].compute().astype("float32")
    out["msl"].attrs = {"units": "Pa", "long_name": "Mean sea level pressure"}

    out.attrs = {"init_date": date_str, "source": "ARCO-ERA5 0.25deg"}
    out.to_netcdf(str(out_file))
    log.info(f"GT   DONE  {date_str}  → {out_file.name}")
    return True


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download ERA5 for FuXi inputs + ground truth")
    parser.add_argument("--date",      type=str, default=None, help="Single date YYYYMMDD")
    parser.add_argument("--start",     type=str, default=DATE_START)
    parser.add_argument("--end",       type=str, default=DATE_END)
    parser.add_argument("--fuxi-only", action="store_true", help="FuXi inputs only")
    parser.add_argument("--gt-only",   action="store_true", help="Ground truth only")
    parser.add_argument("--check",     action="store_true", help="Print status, no download")
    args = parser.parse_args()

    log_file = LOG_DIR / f"download_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)

    if args.date:
        fuxi_dates = [pd.Timestamp(args.date)]
        gt_dates   = [pd.Timestamp(args.date)]
    else:
        fuxi_dates = list(pd.date_range(args.start, args.end, freq="D"))
        # Ground truth needs to cover all possible verification days
        gt_dates   = list(pd.date_range(args.start, GT_END, freq="D"))

    do_fuxi = not args.gt_only
    do_gt   = not args.fuxi_only

    fuxi_pending = [d for d in fuxi_dates if not fuxi_input_done(d)] if do_fuxi else []
    gt_pending   = [d for d in gt_dates   if not gt_done(d)]         if do_gt   else []

    log.info("=" * 65)
    log.info("ERA5 Download for FuXi-S2S (inputs + ground truth)")
    log.info(f"  FuXi input dir  : {INPUT_DIR}")
    log.info(f"  Ground truth dir: {GT_DIR}")
    log.info(f"  FuXi dates      : {len(fuxi_dates)} total, {len(fuxi_dates)-len(fuxi_pending)} done, {len(fuxi_pending)} pending")
    log.info(f"  GT dates        : {len(gt_dates)} total, {len(gt_dates)-len(gt_pending)} done, {len(gt_pending)} pending")
    log.info("=" * 65)

    if args.check:
        log.info("FuXi inputs missing:")
        for d in fuxi_pending:
            log.info(f"  {d:%Y%m%d}")
        log.info("Ground truth missing:")
        for d in gt_pending[:10]:
            log.info(f"  {d:%Y%m%d}")
        if len(gt_pending) > 10:
            log.info(f"  ... and {len(gt_pending)-10} more")
        return

    if not fuxi_pending and not gt_pending:
        log.info("Nothing to download — all files exist.")
        return

    log.info("Connecting to ARCO-ERA5 on GCS...")
    ds_arco = xr.open_zarr(ARCO_ZARR, storage_options={"token": "anon"})

    # ARCO ends 2023-01-10 — warn if dates are outside range
    arco_end = pd.Timestamp("2023-01-10")
    out_of_range_fuxi = [d for d in fuxi_pending if d > arco_end]
    out_of_range_gt   = [d for d in gt_pending   if d > arco_end]

    if out_of_range_fuxi:
        log.warning(f"ARCO-ERA5 ends 2023-01-10. {len(out_of_range_fuxi)} FuXi dates are out of range!")
        log.warning("These dates need CDS API (reanalysis-era5-*) — not implemented here yet.")
        log.warning("For JFM 2026 FuXi inputs, use the existing input.nc files already downloaded.")
        fuxi_pending = [d for d in fuxi_pending if d <= arco_end]

    if out_of_range_gt:
        log.warning(f"ARCO-ERA5 ends 2023-01-10. {len(out_of_range_gt)} GT dates are out of range!")
        log.warning("JFM 2026 ground truth must come from CDS API — see download_era5_gt_cds.py")
        gt_pending = [d for d in gt_pending if d <= arco_end]

    # Process all pending dates (FuXi inputs need pairs, GT is single days)
    all_dates = sorted(set(fuxi_pending) | set(gt_pending))
    failed = []
    for i, date in enumerate(all_dates, 1):
        try:
            if do_fuxi and date in fuxi_pending:
                download_fuxi_input(ds_arco, date, log)
            if do_gt and date in gt_pending:
                download_ground_truth(ds_arco, date, log)
        except Exception as e:
            log.error(f"FAIL {date:%Y%m%d}: {e}")
            failed.append(f"{date:%Y%m%d}")
        log.info(f"Progress {i}/{len(all_dates)}  failed={len(failed)}")

    log.info("=" * 65)
    log.info(f"Done. Failed: {len(failed)}")
    for f in failed:
        log.info(f"  {f}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
