#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_fuxi_from_extracted.py
=================================
Build compact JJAS FuXi-S2S NetCDFs from the persistent full extraction tree.

This script does not extract archives and does not delete raw data. It reads:

  /storage/raj.ayush/archive/All_Model_Data/models/fuxi/extracted_full/
    YYYY/MM/YYYYMMDD/member/NN/SS.nc

and writes analysis-ready compact files:

  /storage/raj.ayush/s2s_final_data/jjas/fuxi_combined/YYYYMMDD.nc

The compact files keep the same layout expected by adapters_fuxi.py:

  forecast(member, lead_time, channel, lat, lon), channel in {tp, z500}

By default member 00 is treated as control and dropped, leaving members 01..50.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/storage/raj.ayush/s2s_final_data/jjas")
RAW_ROOT = Path("/storage/raj.ayush/archive/All_Model_Data/models/fuxi/extracted_full")
OUT_DIR = DATA_ROOT / "fuxi_combined"
LOG_DIR = HERE / "logs"

TOTAL_STEPS = 42
ALL_MEMBERS = list(range(0, 51))
KEEP_CHANNELS = ["tp", "z500"]
RAW_VARNAME = "__xarray_dataarray_variable__"


def setup_logging(log_file):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("preprocess_fuxi_from_extracted")


def parse_months(value):
    if not value:
        return None
    months = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit() or not 1 <= int(item) <= 12:
            raise SystemExit(f"bad month: {item}")
        months.add(f"{int(item):02d}")
    return months


def date_dir(raw_root, init_str):
    return Path(raw_root) / init_str[:4] / init_str[4:6] / init_str


def compact_path(out_dir, init_str):
    return Path(out_dir) / f"{init_str}.nc"


def discover_from_raw(raw_root, args):
    months = parse_months(args.months)
    inits = []
    for path in sorted(Path(raw_root).glob("*/*/????????")):
        if not path.is_dir():
            continue
        date = path.name
        if len(date) != 8 or not date.isdigit():
            continue
        pd.Timestamp(date)
        if args.year and date[:4] != str(args.year):
            continue
        if args.start and date < args.start:
            continue
        if args.end and date > args.end:
            continue
        if months and date[4:6] not in months:
            continue
        if not args.allow_incomplete and not (path / ".complete").exists():
            continue
        inits.append(date)
    return inits


def resolve_inits(args):
    if args.init:
        inits = [s.strip() for s in args.init.split(",") if s.strip()]
    else:
        inits = discover_from_raw(args.raw_root, args)
    seen, out = set(), []
    for date in inits:
        if len(date) != 8 or not date.isdigit():
            raise SystemExit(f"bad init date: {date}")
        pd.Timestamp(date)
        if date not in seen:
            seen.add(date)
            out.append(date)
    if args.array_index is not None:
        if args.array_index < 0 or args.array_index >= len(out):
            raise SystemExit(
                f"array index {args.array_index} outside init list length {len(out)}"
            )
        out = [out[args.array_index]]
    if args.max_count:
        out = out[:args.max_count]
    return out


def read_member_day(nc_path):
    with xr.open_dataset(nc_path) as ds:
        da = ds[RAW_VARNAME]
        da = da.squeeze(["time", "lead_time"], drop=True)
        da = da.sel(channel=KEEP_CHANNELS).load()
    return da


def build_combined(root, members, log):
    member_arrays = []
    for mi, member in enumerate(members, 1):
        step_arrays = []
        for step in range(1, TOTAL_STEPS + 1):
            nc_path = root / "member" / f"{member:02d}" / f"{step:02d}.nc"
            if not nc_path.exists():
                raise FileNotFoundError(f"missing member-day file: {nc_path}")
            step_arrays.append(read_member_day(nc_path))
        member_da = xr.concat(
            step_arrays, dim=pd.Index(range(1, TOTAL_STEPS + 1), name="lead_time")
        )
        member_arrays.append(member_da)
        if mi % 10 == 0 or mi == len(members):
            log.info("    read member %02d (%d/%d)", member, mi, len(members))

    combined = xr.concat(member_arrays, dim=pd.Index(range(len(members)), name="member"))
    combined = combined.transpose("member", "lead_time", "channel", "lat", "lon")
    combined.name = "forecast"
    return combined.astype("float32")


def write_compact(combined, init_str, out_file, log):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.assign_coords(
        init_time=np.datetime64(f"{init_str[:4]}-{init_str[4:6]}-{init_str[6:8]}")
    )
    enc = {"forecast": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    tmp = out_file.with_suffix(".nc.tmp")
    combined.to_netcdf(str(tmp), encoding=enc)
    os.replace(tmp, out_file)
    log.info("  wrote %s %.0f MB", out_file, out_file.stat().st_size / 1024**2)


def process_one(init_str, args, members, log):
    raw_dir = date_dir(args.raw_root, init_str)
    out_file = compact_path(args.out_dir, init_str)

    if out_file.exists() and not args.overwrite:
        log.info("SKIP   %s (compact exists: %s)", init_str, out_file)
        return True
    if not raw_dir.is_dir():
        log.warning("MISS   %s (raw dir missing: %s)", init_str, raw_dir)
        return False
    if not args.allow_incomplete and not (raw_dir / ".complete").exists():
        log.warning("WAIT   %s (no .complete marker in %s)", init_str, raw_dir)
        return False

    t0 = time.time()
    log.info(
        "START  %s raw=%s members=%02d..%02d n=%d",
        init_str, raw_dir, members[0], members[-1], len(members),
    )
    try:
        combined = build_combined(raw_dir, members, log)
        write_compact(combined, init_str, out_file, log)
        log.info("DONE   %s elapsed=%.1f min", init_str, (time.time() - t0) / 60.0)
        return True
    except Exception as exc:
        log.error("FAIL   %s: %s", init_str, exc)
        tmp = out_file.with_suffix(".nc.tmp")
        if tmp.exists():
            tmp.unlink()
        if out_file.exists() and args.overwrite:
            out_file.unlink()
        return False


def main():
    p = argparse.ArgumentParser(
        description="Compact persistent FuXi-S2S extracted data into JJAS analysis files."
    )
    p.add_argument("--raw-root", default=str(RAW_ROOT))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--init", default=None, help="single init YYYYMMDD or comma list")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--months", default="06,07,08,09")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--max-count", type=int, default=None)
    p.add_argument("--keep-control", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.raw_root = Path(args.raw_root)
    args.out_dir = Path(args.out_dir)
    members = ALL_MEMBERS if args.keep_control else ALL_MEMBERS[1:]
    inits = resolve_inits(args)
    if not inits:
        raise SystemExit("no matching complete FuXi raw dates found")

    log = setup_logging(LOG_DIR / f"preprocess_fuxi_from_extracted_{pd.Timestamp.now():%Y%m%d_%H%M%S}.log")
    log.info("=" * 64)
    log.info("FuXi-S2S persistent raw -> compact JJAS preprocessor")
    log.info("  raw root : %s", args.raw_root)
    log.info("  out dir  : %s", args.out_dir)
    log.info("  months   : %s", args.months)
    log.info("  channels : %s", KEEP_CHANNELS)
    log.info("  members  : %d (%s)", len(members), "incl control 00" if args.keep_control else "01..50")
    log.info("  inits    : %d %s", len(inits), inits[:8])
    log.info("=" * 64)

    ok = fail = 0
    for i, init_str in enumerate(inits, 1):
        if process_one(init_str, args, members, log):
            ok += 1
        else:
            fail += 1
        log.info("Progress %d/%d (ok=%d fail=%d)", i, len(inits), ok, fail)

    log.info("=" * 64)
    log.info("Done. ok=%d fail=%d out=%s", ok, fail, args.out_dir)
    log.info("=" * 64)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
