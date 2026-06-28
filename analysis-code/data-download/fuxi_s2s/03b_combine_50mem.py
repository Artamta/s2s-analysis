#!/usr/bin/env python3
"""
03b_combine_50mem.py
====================
Combine 50-member FuXi raw output into single NetCDF per init date.

Raw    : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50/raw/{YYYYMMDD}/member/{MM}/{SS}.nc
Output : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50/combined/{YYYYMMDD}.nc
         dims: (member=50, lead_time=42, channel=76, lat=121, lon=240)  ~1.6 GB per date

Usage
-----
  python 03b_combine_50mem.py
  python 03b_combine_50mem.py --date 20260101
  python 03b_combine_50mem.py --keep-raw
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

BASE_DIR     = Path("/storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50")
RAW_DIR      = BASE_DIR / "raw"
COMBINED_DIR = BASE_DIR / "combined"
LOG_DIR      = Path(__file__).parent / "logs"

TOTAL_STEPS   = 42
TOTAL_MEMBERS = 50

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


def expected_raw_files(date: pd.Timestamp):
    root = RAW_DIR / f"{date:%Y%m%d}" / "member"
    for member in range(TOTAL_MEMBERS):
        for step in range(1, TOTAL_STEPS + 1):
            yield root / f"{member:02d}" / f"{step:02d}.nc"


def raw_status(date: pd.Timestamp):
    root = RAW_DIR / f"{date:%Y%m%d}" / "member"
    if not root.exists():
        return False, 0, [root]

    present = 0
    missing = []
    for path in expected_raw_files(date):
        if path.exists() and path.stat().st_size > 0:
            present += 1
        elif len(missing) < 5:
            missing.append(path)
    return present == TOTAL_MEMBERS * TOTAL_STEPS, present, missing


def raw_is_complete(date: pd.Timestamp) -> bool:
    return raw_status(date)[0]


def combined_exists(date: pd.Timestamp) -> bool:
    return (COMBINED_DIR / f"{date:%Y%m%d}.nc").exists()


def combine_one(date: pd.Timestamp, keep_raw: bool, log: logging.Logger) -> bool:
    date_str = f"{date:%Y%m%d}"
    raw_path = RAW_DIR / date_str
    out_file = COMBINED_DIR / f"{date_str}.nc"
    tmp_file = out_file.with_suffix(".nc.tmp")

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"START  {date_str}")
    complete, present, missing = raw_status(date)
    if not complete:
        miss = ", ".join(str(p.relative_to(raw_path)) for p in missing)
        raise RuntimeError(f"raw incomplete ({present}/{TOTAL_MEMBERS * TOTAL_STEPS}; "
                           f"missing examples: {miss})")

    member_arrays = []
    for m in range(TOTAL_MEMBERS):
        step_arrays = []
        for s in range(1, TOTAL_STEPS + 1):
            f = raw_path / "member" / f"{m:02d}" / f"{s:02d}.nc"
            da = xr.open_dataarray(str(f))
            step_arrays.append(da.squeeze(["time", "lead_time"]))
        member_da = xr.concat(step_arrays, dim=pd.Index(range(1, TOTAL_STEPS + 1), name="lead_time"))
        member_arrays.append(member_da)

    combined = xr.concat(member_arrays, dim=pd.Index(range(TOTAL_MEMBERS), name="member"))
    combined = combined.assign_coords({"init_time": np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")})
    combined.name = "forecast"

    enc = {"forecast": {"zlib": True, "complevel": 3, "dtype": "float32"}}
    if tmp_file.exists():
        tmp_file.unlink()
    combined.astype("float32").to_netcdf(str(tmp_file), encoding=enc)
    tmp_file.replace(out_file)
    mb = out_file.stat().st_size / 1024**2
    log.info(f"DONE   {date_str}  {mb:.0f} MB  → {out_file.name}")

    if not keep_raw:
        shutil.rmtree(str(raw_path))
        log.info(f"  Deleted raw/{date_str}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",     type=str, default=None)
    parser.add_argument("--start",    type=str, default=DATE_START)
    parser.add_argument("--end",      type=str, default=DATE_END)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    log_file = LOG_DIR / f"combine_ens50_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        dates = list(pd.date_range(args.start, args.end, freq="D"))

    log.info("=" * 60)
    log.info("FuXi-S2S Output Combiner — 50-member")
    log.info(f"  Combined dir    : {COMBINED_DIR}")
    log.info(f"  Total dates     : {len(dates)}")
    log.info("  Mode            : streaming resume")
    log.info("=" * 60)

    failed = []
    skipped = 0
    missing = 0
    combined_count = 0
    for i, date in enumerate(dates, 1):
        date_str = f"{date:%Y%m%d}"
        if combined_exists(date):
            skipped += 1
            log.info(f"SKIP   {date_str}  already combined")
            continue

        try:
            combine_one(date, args.keep_raw, log)
            combined_count += 1
        except Exception as e:
            msg = str(e)
            log.error(f"FAIL   {date_str}: {msg}")
            if "raw incomplete" in msg:
                missing += 1
            failed.append(date_str)
        log.info(f"Progress {i}/{len(dates)}  skipped={skipped} combined={combined_count} failed={len(failed)}")

    log.info("=" * 60)
    log.info(f"Done. Skipped: {skipped}  Combined: {combined_count}  Missing raw: {missing}  Failed: {len(failed)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
