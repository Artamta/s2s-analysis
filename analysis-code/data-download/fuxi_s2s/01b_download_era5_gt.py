#!/usr/bin/env python3
"""
01b_download_era5_gt.py
=======================
Download ERA5 ground truth for JFM 2026 verification from CDS API.

Covers Jan 1 2026 → May 16 2026 (90 init dates + 46-day lead padding).
One file per day, daily mean at 00 UTC, India domain, 0.25° resolution.

Variables:
  z500  — geopotential height at 500 hPa (gpm)
  t2m   — 2m temperature (K)
  tp    — total precipitation (mm/day)
  msl   — mean sea level pressure (Pa)

Output : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/ground_truth/{YYYYMMDD}.nc

Usage
-----
  python 01b_download_era5_gt.py           # all missing dates
  python 01b_download_era5_gt.py --date 20260101
  # via SLURM:
  sbatch slurm/fuxi_01b_era5_gt.sbatch
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
import tempfile

# ── PATHS ─────────────────────────────────────────────────────────────────────
GT_DIR  = Path("/storage/raj.ayush/All_Model_Data/fuxi/jfm2026/ground_truth")
LOG_DIR = Path(__file__).parent / "logs"

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATE_START = "2026-01-01"
DATE_END   = "2026-05-16"   # last init (Mar 31) + 46 lead days

# India domain — matches ECMWF download [N, W, S, E]
AREA = [40, 60, 0, 100]
GRID = [0.25, 0.25]

MAX_RETRIES   = 5
RETRY_BACKOFF = [60, 120, 300, 600, 900]


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def is_done(date: pd.Timestamp) -> bool:
    return (GT_DIR / f"{date:%Y%m%d}.nc").exists()


def download_one(client: cdsapi.Client, date: pd.Timestamp,
                 log: logging.Logger) -> bool:
    date_str = f"{date:%Y%m%d}"
    out_file = GT_DIR / f"{date_str}.nc"
    GT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"START {date_str}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Surface variables ──────────────────────────────────────────────
        sfc_req = {
            "product_type": "reanalysis",
            "variable": ["2m_temperature", "total_precipitation",
                         "mean_sea_level_pressure"],
            "year":  str(date.year),
            "month": f"{date.month:02d}",
            "day":   f"{date.day:02d}",
            "time":  "00:00",
            "area":  AREA,
            "grid":  GRID,
            "data_format": "netcdf",
        }

        # ── Pressure level z500 ───────────────────────────────────────────
        pl_req = {
            "product_type": "reanalysis",
            "variable": ["geopotential"],
            "pressure_level": ["500"],
            "year":  str(date.year),
            "month": f"{date.month:02d}",
            "day":   f"{date.day:02d}",
            "time":  "00:00",
            "area":  AREA,
            "grid":  GRID,
            "data_format": "netcdf",
        }

        sfc_file = tmp / "sfc.nc"
        pl_file  = tmp / "pl.nc"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client.retrieve("reanalysis-era5-single-levels",   sfc_req, str(sfc_file))
                client.retrieve("reanalysis-era5-pressure-levels",  pl_req,  str(pl_file))
                break
            except Exception as e:
                log.warning(f"  attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF[attempt - 1]
                    log.info(f"  waiting {wait}s...")
                    time.sleep(wait)
                else:
                    log.error(f"FAIL {date_str} after {MAX_RETRIES} attempts")
                    return False

        # ── Combine into one file ─────────────────────────────────────────
        sfc = xr.open_dataset(str(sfc_file))
        pl  = xr.open_dataset(str(pl_file))

        out = xr.Dataset()

        # z500: geopotential (m²/s²) → geopotential height (gpm)
        z_var = [v for v in pl.data_vars if "z" in v.lower() or "geopotential" in v.lower()][0]
        z500  = pl[z_var].squeeze()
        if "pressure_level" in z500.dims:
            z500 = z500.sel(pressure_level=500)
        elif "level" in z500.dims:
            z500 = z500.isel(level=0)
        out["z500"] = (z500 / 9.80665).astype("float32")
        out["z500"].attrs = {"units": "gpm", "long_name": "Geopotential height 500 hPa"}

        # t2m
        t2m_var = [v for v in sfc.data_vars if "t2m" in v or "2m_temperature" in v.lower() or v == "t2m"][0]
        out["t2m"] = sfc[t2m_var].squeeze().astype("float32")
        out["t2m"].attrs = {"units": "K", "long_name": "2m temperature"}

        # tp — CDS returns m; convert to mm
        tp_var = [v for v in sfc.data_vars if "tp" in v or "precipitation" in v.lower()][0]
        tp = sfc[tp_var].squeeze()
        out["tp"] = (tp * 1000).clip(0).astype("float32")
        out["tp"].attrs = {"units": "mm", "long_name": "Total precipitation (accumulated)"}

        # msl
        msl_var = [v for v in sfc.data_vars if "msl" in v or "sea_level" in v.lower()][0]
        out["msl"] = sfc[msl_var].squeeze().astype("float32")
        out["msl"].attrs = {"units": "Pa", "long_name": "Mean sea level pressure"}

        out.attrs = {
            "date": date_str,
            "source": "ERA5 reanalysis via CDS API",
            "domain": "India 0-40N 60-100E 0.25deg",
        }
        out.to_netcdf(str(out_file))

    mb = out_file.stat().st_size / 1024**2
    log.info(f"DONE  {date_str}  {mb:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 ground truth for JFM 2026")
    parser.add_argument("--date",  type=str, default=None, help="Single date YYYYMMDD")
    parser.add_argument("--start", type=str, default=DATE_START)
    parser.add_argument("--end",   type=str, default=DATE_END)
    args = parser.parse_args()

    log_file = LOG_DIR / f"era5_gt_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        dates = list(pd.date_range(args.start, args.end, freq="D"))

    pending = [d for d in dates if not is_done(d)]

    log.info("=" * 60)
    log.info("ERA5 Ground Truth Download — JFM 2026")
    log.info(f"  Output dir : {GT_DIR}")
    log.info(f"  Period     : {dates[0].date()} → {dates[-1].date()}  ({len(dates)} days)")
    log.info(f"  Done       : {len(dates) - len(pending)}")
    log.info(f"  Pending    : {len(pending)}")
    log.info(f"  Variables  : z500 (gpm), t2m (K), tp (mm), msl (Pa)")
    log.info(f"  Domain     : India 0-40N 60-100E @ 0.25°")
    log.info("=" * 60)

    if not pending:
        log.info("All files exist — nothing to download.")
        return

    client = cdsapi.Client(quiet=True)
    failed = []
    for i, date in enumerate(pending, 1):
        ok = download_one(client, date, log)
        if not ok:
            failed.append(f"{date:%Y%m%d}")
        log.info(f"Progress {i}/{len(pending)}  failed={len(failed)}")

    log.info("=" * 60)
    log.info(f"Done. Success: {len(pending)-len(failed)}  Failed: {len(failed)}")
    for f in failed:
        log.info(f"  FAILED: {f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
