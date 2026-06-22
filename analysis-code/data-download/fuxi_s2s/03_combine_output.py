#!/usr/bin/env python3
"""
03_combine_output.py
====================
Combine per-member per-step FuXi raw output into a single NetCDF per init date.

Raw layout : raw/{YYYYMMDD}/member/{MM}/{SS}.nc
             dims: (time=1, lead_time=1, channel=76, lat=121, lon=240)

Output     : combined/{YYYYMMDD}.nc
             dims: (member=11, lead_time=42, channel=76, lat=121, lon=240)
             ~350 MB per date (float32)

Usage
-----
  # All dates that have complete raw output
  python 03_combine_output.py

  # Single date
  python 03_combine_output.py --date 20260101

  # Keep raw files after combining (default: delete to save space)
  python 03_combine_output.py --keep-raw
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/storage/raj.ayush/All_Model_Data/fuxi/jfm2026")
RAW_DIR      = BASE_DIR / "raw"
COMBINED_DIR = BASE_DIR / "combined"
LOG_DIR      = Path(__file__).parent / "logs"

TOTAL_STEPS   = 42
TOTAL_MEMBERS = 11

DATE_START = "2026-01-01"
DATE_END   = "2026-03-31"


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def raw_is_complete(date: pd.Timestamp) -> bool:
    final = RAW_DIR / f"{date:%Y%m%d}" / "member" / f"{TOTAL_MEMBERS-1:02d}" / f"{TOTAL_STEPS:02d}.nc"
    return final.exists()


def combined_exists(date: pd.Timestamp) -> bool:
    return (COMBINED_DIR / f"{date:%Y%m%d}.nc").exists()


def combine_one(date: pd.Timestamp, keep_raw: bool, log: logging.Logger) -> bool:
    date_str = f"{date:%Y%m%d}"
    raw_path = RAW_DIR / date_str
    out_file = COMBINED_DIR / f"{date_str}.nc"

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"START  {date_str}")

    member_arrays = []
    for m in range(TOTAL_MEMBERS):
        step_arrays = []
        for s in range(1, TOTAL_STEPS + 1):
            f = raw_path / "member" / f"{m:02d}" / f"{s:02d}.nc"
            da = xr.open_dataarray(str(f))
            # shape: (time=1, lead_time=1, channel=76, lat=121, lon=240)
            step_arrays.append(da.squeeze(["time", "lead_time"]))
        # stack steps -> (lead_time=42, channel=76, lat, lon)
        member_da = xr.concat(step_arrays, dim=pd.Index(range(1, TOTAL_STEPS + 1), name="lead_time"))
        member_arrays.append(member_da)

    # stack members -> (member=11, lead_time=42, channel=76, lat, lon)
    combined = xr.concat(member_arrays, dim=pd.Index(range(TOTAL_MEMBERS), name="member"))
    combined = combined.assign_coords({"init_time": np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")})
    combined.name = "forecast"

    combined.to_netcdf(str(out_file))
    mb = out_file.stat().st_size / 1024**2
    log.info(f"DONE   {date_str}  {mb:.0f} MB  → {out_file.name}")

    if not keep_raw:
        shutil.rmtree(str(raw_path))
        log.info(f"  Deleted raw/{date_str}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Combine FuXi raw output into single NetCDF per date")
    parser.add_argument("--date",     type=str, default=None)
    parser.add_argument("--start",    type=str, default=DATE_START)
    parser.add_argument("--end",      type=str, default=DATE_END)
    parser.add_argument("--keep-raw", action="store_true", help="Don't delete raw files after combining")
    args = parser.parse_args()

    log_file = LOG_DIR / f"combine_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        dates = list(pd.date_range(args.start, args.end, freq="D"))

    # Only process dates that have complete raw output and aren't already combined
    ready   = [d for d in dates if raw_is_complete(d) and not combined_exists(d)]
    skipped = [d for d in dates if combined_exists(d)]
    missing = [d for d in dates if not raw_is_complete(d) and not combined_exists(d)]

    log.info("=" * 60)
    log.info("FuXi-S2S Output Combiner")
    log.info(f"  Combined dir : {COMBINED_DIR}")
    log.info(f"  Total dates  : {len(dates)}")
    log.info(f"  Already done : {len(skipped)}")
    log.info(f"  Ready to combine: {len(ready)}")
    log.info(f"  Missing raw  : {len(missing)}  (need 02_run_inference.py first)")
    log.info("=" * 60)

    failed = []
    for i, date in enumerate(ready, 1):
        try:
            combine_one(date, args.keep_raw, log)
        except Exception as e:
            log.error(f"FAIL   {date:%Y%m%d}: {e}")
            failed.append(f"{date:%Y%m%d}")
        log.info(f"Progress {i}/{len(ready)}")

    log.info("=" * 60)
    log.info(f"Done. Combined: {len(ready) - len(failed)}  Failed: {len(failed)}")
    if missing:
        log.info(f"Still need inference for {len(missing)} dates — run 02_run_inference.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
