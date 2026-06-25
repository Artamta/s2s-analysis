#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jjas/preprocess_fuxi.py  —  FuXi-S2S reforecast .7z -> one compact NetCDF / init.
================================================================================
The multi-year FuXi-S2S hindcast ships as one solid .7z per init date:

    /storage/.../models/fuxi/data/<YYYYMMDD>.7z
        <YYYYMMDD>/member/<NN>/<SS>.nc          NN = 00..50 , SS = 01..42

Each member-day .nc holds dims (time=1, lead_time=1, channel=26, lat=121, lon=240)
in an UNNAMED variable ('__xarray_dataarray_variable__'). The 26 channels include
z500 (geopotential m^2/s^2), t2m (K) and tp (mm/h rate). Grid is global 1.5deg
(lat 90..-90 descending, lon 0..358.5 ascending) — same as the JFM FuXi data.

Members 00..50 = control (00) + 50 perturbed members (01..50). The FuXi-S2S
product is a 50-member ensemble, so by default we keep members 01..50 and drop
the control (--keep-control to retain member 00 as well).

This script, for one init YYYYMMDD:
  1. extracts that .7z to a scratch dir,
  2. reads the (members x 42 leads), selecting ONLY the tp + z500 channels,
  3. writes ONE compact NetCDF mirroring the JFM `combined/` layout:
        forecast(member, lead_time, channel, lat, lon)   channel in {tp, z500}
     into jjas/fuxi_combined/<YYYYMMDD>.nc,
  4. deletes the extracted raw tree (so disk never holds >1 init's ~6.5 GB raw).

Idempotent (skips if the compact file already exists) and CLI-driven (one init,
a comma list, or --start/--end over the .7z archive).

A `fuxi_reforecast` adapter (jjas/adapters_fuxi.py) then reads these compact files
exactly like the JFM `fuxi_combined` adapter reads its `combined/` files.

Usage
-----
  python preprocess_fuxi.py --init 20190620
  python preprocess_fuxi.py --init 20190620,20190623,20190627
  python preprocess_fuxi.py --start 20190601 --end 20190831      # JJAS 2019
  python preprocess_fuxi.py --init 20190620 --keep-control       # keep member 00
================================================================================
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Generated DATA lives on /storage (NOT in the home repo); only code stays here.
HERE         = Path(__file__).resolve().parent
DATA_ROOT    = Path("/storage/raj.ayush/s2s_final_data/jjas")
ARCHIVE_DIR  = Path("/storage/raj.ayush/archive/All_Model_Data/models/fuxi/data")
OUT_DIR      = DATA_ROOT / "fuxi_combined"
LOG_DIR      = HERE / "logs"
SCRATCH_ROOT = Path(
    os.environ.get("FUXI_SCRATCH")
    or os.environ.get("SLURM_TMPDIR")
    or f"/tmp/{os.environ.get('USER', 'raj.ayush')}/fuxi_extract"
)
SEVENZIP     = "/usr/bin/7z"

# ── ARCHIVE SHAPE ─────────────────────────────────────────────────────────────
TOTAL_STEPS = 42                       # lead days 01..42
ALL_MEMBERS = list(range(0, 51))       # 00 (control) + 01..50 (perturbed)
KEEP_CHANNELS = ["tp", "z500"]         # only what JJAS verification needs (small)
RAW_VARNAME = "__xarray_dataarray_variable__"


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("preprocess_fuxi")


def archive_path(init_str: str) -> Path:
    return ARCHIVE_DIR / f"{init_str}.7z"


def combined_path(init_str: str) -> Path:
    return OUT_DIR / f"{init_str}.nc"


def extract_archive(init_str: str, dest: Path, log: logging.Logger) -> Path:
    """Extract the whole .7z for one init into `dest`. Returns the date subdir."""
    arc = archive_path(init_str)
    if not arc.exists():
        raise FileNotFoundError(f"archive not found: {arc}")
    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"  extracting {arc.name} ({arc.stat().st_size / 1024**3:.1f} GB) -> {dest}")
    # Use py7zr (pure-Python, works on SLURM COMPUTE nodes) so extraction never
    # needs the login node or the /usr/bin/7z binary (absent on compute nodes).
    # Falls back to the 7z binary only if py7zr is unavailable.
    try:
        import py7zr
        with py7zr.SevenZipFile(str(arc), mode="r") as z:
            z.extractall(path=str(dest))
    except ImportError:
        subprocess.run([SEVENZIP, "x", str(arc), f"-o{dest}", "-y", "-bso0", "-bsp0"],
                       check=True)
    date_dir = dest / init_str
    if not date_dir.is_dir():
        raise RuntimeError(f"expected extracted dir missing: {date_dir}")
    return date_dir


def read_member_day(nc_path: Path) -> xr.DataArray:
    """Load one member-day .nc, squeeze (time, lead_time), keep only KEEP_CHANNELS."""
    da = xr.open_dataset(nc_path)[RAW_VARNAME]
    da = da.squeeze(["time", "lead_time"], drop=True)        # (channel, lat, lon)
    return da.sel(channel=KEEP_CHANNELS).load()


def build_combined(date_dir: Path, members, log: logging.Logger) -> xr.DataArray:
    """Assemble forecast(member, lead_time, channel, lat, lon) from the raw tree."""
    member_arrays = []
    for m in members:
        step_arrays = []
        for s in range(1, TOTAL_STEPS + 1):
            f = date_dir / "member" / f"{m:02d}" / f"{s:02d}.nc"
            if not f.exists():
                raise FileNotFoundError(f"missing member-day file: {f}")
            step_arrays.append(read_member_day(f))
        member_da = xr.concat(
            step_arrays, dim=pd.Index(range(1, TOTAL_STEPS + 1), name="lead_time")
        )                                                    # (lead_time, channel, lat, lon)
        member_arrays.append(member_da)
        if (m + 1) % 10 == 0 or m == members[-1]:
            log.info(f"    read member {m:02d} ({members.index(m) + 1}/{len(members)})")

    combined = xr.concat(
        member_arrays, dim=pd.Index(range(len(members)), name="member")
    )                                                        # (member, lead_time, channel, lat, lon)
    combined = combined.transpose("member", "lead_time", "channel", "lat", "lon")
    combined.name = "forecast"
    return combined.astype("float32")


def write_compact(combined: xr.DataArray, init_str: str, out_file: Path,
                  log: logging.Logger) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = combined.assign_coords(
        init_time=np.datetime64(f"{init_str[:4]}-{init_str[4:6]}-{init_str[6:8]}")
    )
    enc = {"forecast": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    tmp = out_file.with_suffix(".nc.tmp")
    combined.to_netcdf(str(tmp), encoding=enc)
    os.replace(tmp, out_file)
    log.info(f"  wrote {out_file.name}  {out_file.stat().st_size / 1024**2:.0f} MB")


def process_one(init_str: str, members, log: logging.Logger) -> bool:
    out_file = combined_path(init_str)
    if out_file.exists():
        log.info(f"SKIP   {init_str}  (compact file exists: {out_file})")
        return True

    arc = archive_path(init_str)
    if not arc.exists():
        log.warning(f"MISS   {init_str}  (no archive {arc.name})")
        return False

    t0 = time.time()
    log.info(f"START  {init_str}  members={members[0]:02d}..{members[-1]:02d} "
             f"(n={len(members)})")
    work = Path(tempfile.mkdtemp(prefix=f"{init_str}_", dir=str(SCRATCH_ROOT)))
    try:
        date_dir = extract_archive(init_str, work, log)
        combined = build_combined(date_dir, members, log)
        write_compact(combined, init_str, out_file, log)
        raw_gb = sum(f.stat().st_size for f in date_dir.rglob("*.nc")) / 1024**3
        cmp_mb = out_file.stat().st_size / 1024**2
        log.info(f"DONE   {init_str}  raw≈{raw_gb:.1f} GB  compact={cmp_mb:.0f} MB  "
                 f"({(time.time() - t0) / 60:.1f} min)")
        return True
    except Exception as e:
        log.error(f"FAIL   {init_str}: {e}")
        # don't leave a half-written compact file behind
        if out_file.exists():
            out_file.unlink()
        return False
    finally:
        # ALWAYS remove the extracted raw tree so disk never holds >1 init's raw.
        shutil.rmtree(work, ignore_errors=True)


def resolve_inits(args) -> list:
    """Build the ordered, de-duplicated list of init YYYYMMDD strings to process."""
    if args.init:
        inits = [s.strip() for s in args.init.split(",") if s.strip()]
    elif args.start and args.end:
        inits = []
        for arc in sorted(ARCHIVE_DIR.glob("*.7z")):
            d = arc.stem
            if len(d) == 8 and d.isdigit() and args.start <= d <= args.end:
                inits.append(d)
    else:
        raise SystemExit("provide --init YYYYMMDD[,YYYYMMDD...] or --start/--end")
    # validate calendar dates, de-dup, preserve order
    seen, out = set(), []
    for d in inits:
        pd.Timestamp(d)                      # raises on a bad date
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Combine one FuXi-S2S .7z reforecast init into a compact NetCDF "
                    "(tp + z500 only) mirroring the JFM combined/ layout.")
    p.add_argument("--init", type=str, default=None,
                   help="single init YYYYMMDD or comma list")
    p.add_argument("--start", type=str, default=None, help="range start YYYYMMDD")
    p.add_argument("--end",   type=str, default=None, help="range end YYYYMMDD (incl.)")
    p.add_argument("--keep-control", action="store_true",
                   help="keep member 00 (control) in addition to 01..50")
    args = p.parse_args()

    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOG_DIR / f"preprocess_fuxi_{pd.Timestamp.now():%Y%m%d_%H%M%S}.log")

    members = ALL_MEMBERS if args.keep_control else ALL_MEMBERS[1:]   # drop 00 by default
    inits = resolve_inits(args)

    log.info("=" * 64)
    log.info("FuXi-S2S reforecast preprocessor")
    log.info(f"  archive dir : {ARCHIVE_DIR}")
    log.info(f"  out dir     : {OUT_DIR}")
    log.info(f"  channels    : {KEEP_CHANNELS}")
    log.info(f"  members     : {len(members)}  "
             f"({'incl. control 00' if args.keep_control else 'perturbed 01..50, control dropped'})")
    log.info(f"  inits       : {len(inits)}  {inits[:6]}{'...' if len(inits) > 6 else ''}")
    log.info("=" * 64)

    ok = fail = 0
    for i, init_str in enumerate(inits, 1):
        if process_one(init_str, members, log):
            ok += 1
        else:
            fail += 1
        log.info(f"Progress {i}/{len(inits)}  (ok={ok} fail={fail})")

    log.info("=" * 64)
    log.info(f"Done. ok={ok}  fail={fail}  out={OUT_DIR}")
    log.info("=" * 64)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
