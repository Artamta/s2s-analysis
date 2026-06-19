#!/usr/bin/env python3
"""
download_ecmwf_reforecasts.py
==============================
Download ECMWF S2S reforecasts (hindcasts) from ECDS for all
available initialization dates (Mon + Thu, all calendar years
covered by the dataset).

Variables
---------
Surface (single_level):
  tp   - total precipitation
  2t   - 2m temperature
  mx2t - maximum 2m temperature in last 24 h
  mn2t - minimum 2m temperature in last 24 h
  msl  - mean sea level pressure

Pressure level (pressure_level):
  z, t, u, v, q  at 200 / 500 / 850 / 1000 hPa

Output structure
----------------
  <OUT_BASE>/
  ├── tp/   <init_date>_cf.nc  <init_date>_pf.nc  ...
  ├── 2t/   ...
  ├── mx2t/ ...
  ├── mn2t/ ...
  ├── msl/  ...
  ├── z/200/  z/500/  z/850/  z/1000/
  ├── t/200/  ...
  ├── u/200/  ...
  ├── v/200/  ...
  └── q/200/  ...

Each file: one init date, one variable (one pressure level for PL vars),
           one forecast type (cf / pf), NetCDF format.

Usage
-----
  # All init dates, all variables, all years
  python download_ecmwf_reforecasts.py

  # Dry-run: print what would be downloaded, do nothing
  python download_ecmwf_reforecasts.py --dry-run

  # Single init date test (must be Mon or Thu)
  python download_ecmwf_reforecasts.py --date 2020-01-02

  # Year range
  python download_ecmwf_reforecasts.py --year-start 2000 --year-end 2010

  # Parallel workers (default 4)
  python download_ecmwf_reforecasts.py --workers 6

  # Skip pf (perturbed) to save time / quota
  python download_ecmwf_reforecasts.py --no-pf

Notes
-----
- ECMWF reforecasts are initialized every Monday and Thursday.
- The real-time date window for ECDS reforecasts is typically 2000-2019
  (the 20-year hindcast window). Adjust YEAR_START / YEAR_END if needed.
- cdsapi >= 0.7.7 + ECDS endpoint required (url: https://ecds.ecmwf.int/api).
- Set ~/.cdsapirc with your ECDS key before running.
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cdsapi
import pandas as pd

# ── OUTPUT CONFIGURATION ──────────────────────────────────────────────────────
OUT_BASE = Path("/storage/raj.ayush/All_Model_Data/ecmwf/reforecasts")

# Hindcast year window (ECMWF typically covers 2000-2019 for current real-time dates)
YEAR_START = 2000
YEAR_END   = 2019

# ── CDS REQUEST PARAMETERS ───────────────────────────────────────────────────
# Lead times: day-1 through day-46 expressed in hours
STEPS = [str(h) for h in range(24, 24 * 46 + 1, 24)]   # ["24","48",...,"1104"]

# CDS long-name → output short-name for surface variables
SURFACE_VARS = {
    "total_precipitation":                        "tp",
    "2m_temperature":                             "2t",
    "maximum_2m_temperature_in_the_last_24_hours": "mx2t",
    "minimum_2m_temperature_in_the_last_24_hours": "mn2t",
    "mean_sea_level_pressure":                    "msl",
}

# CDS long-name → output short-name for pressure-level variables
PL_VARS = {
    "geopotential":     "z",
    "temperature":      "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity":   "q",
}

PRESSURE_LEVELS = ["200", "500", "850", "1000"]

# Forecast types
FTYPES = {
    "cf": "control_reforecast",
    "pf": "perturbed_reforecast",
}

# ── RETRY SETTINGS ───────────────────────────────────────────────────────────
MAX_RETRIES    = 5
RETRY_BACKOFF  = [30, 60, 120, 300, 600]   # seconds between retry attempts

# ── LOGGING ──────────────────────────────────────────────────────────────────
LOG_DIR = OUT_BASE / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"download_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Request log (JSON lines, one per completed download)
REQUEST_LOG = LOG_DIR / "requests.jsonl"


def _log_request(record: dict):
    """Append one JSON record to the request log."""
    with open(REQUEST_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def _make_dirs(short_name: str, level: str | None = None) -> Path:
    """Return (and create) the output directory for a given variable."""
    if level is not None:
        d = OUT_BASE / short_name / level
    else:
        d = OUT_BASE / short_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_path(short_name: str, date: pd.Timestamp, ftype_key: str,
                 level: str | None = None) -> Path:
    d = _make_dirs(short_name, level)
    return d / f"{date:%Y%m%d}_{ftype_key}.nc"


def _already_done(path: Path) -> bool:
    """Return True if file exists and is non-empty (treat 0-byte as failed)."""
    return path.exists() and path.stat().st_size > 0


def _build_base_request(date: pd.Timestamp, ftype_key: str) -> dict:
    return {
        "origin":        "ecmwf",
        "forecast_type": FTYPES[ftype_key],
        "year":          str(date.year),
        "month":         f"{date.month:02d}",
        "day":           f"{date.day:02d}",
        "time":          "00:00",
        "step":          STEPS,
        "data_format":   "netcdf",
    }


def download_one(client: cdsapi.Client, task: dict) -> dict:
    """
    Execute a single CDS retrieve call with retries.

    task keys: short_name, cds_name, level_type, cds_level (optional),
               date, ftype_key, out_path
    """
    out_path = Path(task["out_path"])

    if _already_done(out_path):
        log.info(f"SKIP  {out_path.relative_to(OUT_BASE)}")
        return {**task, "status": "skipped"}

    req = _build_base_request(task["date"], task["ftype_key"])
    req["level_type"] = task["level_type"]
    req["variable"]   = task["cds_name"]
    if task.get("cds_level"):
        req["level"] = task["cds_level"]

    label = out_path.relative_to(OUT_BASE)
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"START {label}  (attempt {attempt}/{MAX_RETRIES})")
            t0 = time.time()
            client.retrieve("s2s-reforecasts", req, str(out_path))
            elapsed = time.time() - t0
            size_mb = out_path.stat().st_size / 1024**2
            log.info(f"DONE  {label}  {size_mb:.1f} MB  {elapsed:.0f}s")
            record = {**task, "status": "success", "size_mb": round(size_mb, 2),
                      "elapsed_s": round(elapsed, 1), "attempt": attempt,
                      "ts": datetime.utcnow().isoformat()}
            record["date"] = record["date"].isoformat()
            record["out_path"] = str(out_path)
            _log_request(record)
            return record
        except Exception as exc:
            last_exc = exc
            log.warning(f"FAIL  {label}  attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                log.info(f"      retrying in {wait}s ...")
                time.sleep(wait)

    log.error(f"GIVE UP {label} after {MAX_RETRIES} attempts: {last_exc}")
    record = {**task, "status": "failed", "error": str(last_exc),
              "ts": datetime.utcnow().isoformat()}
    record["date"] = record["date"].isoformat()
    record["out_path"] = str(out_path)
    _log_request(record)
    return record


def build_task_list(dates: list[pd.Timestamp], ftypes: list[str]) -> list[dict]:
    """Build the full list of (date, ftype, variable, level) download tasks."""
    tasks = []

    for date in dates:
        for ftype_key in ftypes:
            # Surface variables — one task per variable
            for cds_name, short_name in SURFACE_VARS.items():
                out_path = _output_path(short_name, date, ftype_key)
                tasks.append({
                    "short_name": short_name,
                    "cds_name":   cds_name,
                    "level_type": "single_level",
                    "cds_level":  None,
                    "date":       date,
                    "ftype_key":  ftype_key,
                    "out_path":   str(out_path),
                })

            # Pressure-level variables — one task per variable × level
            for cds_name, short_name in PL_VARS.items():
                for level in PRESSURE_LEVELS:
                    out_path = _output_path(short_name, date, ftype_key, level)
                    tasks.append({
                        "short_name": short_name,
                        "cds_name":   cds_name,
                        "level_type": "pressure_level",
                        "cds_level":  level,
                        "date":       date,
                        "ftype_key":  ftype_key,
                        "out_path":   str(out_path),
                    })

    return tasks


def generate_init_dates(year_start: int, year_end: int) -> list[pd.Timestamp]:
    """Return all Mon + Thu dates in [year_start, year_end]."""
    dates = pd.date_range(
        f"{year_start}-01-01",
        f"{year_end}-12-31",
        freq="D",
    )
    return [d for d in dates if d.weekday() in (0, 3)]   # Mon=0, Thu=3


def main():
    parser = argparse.ArgumentParser(
        description="Download ECMWF S2S reforecasts from ECDS."
    )
    parser.add_argument("--year-start", type=int, default=YEAR_START,
                        help=f"First hindcast year (default {YEAR_START})")
    parser.add_argument("--year-end",   type=int, default=YEAR_END,
                        help=f"Last hindcast year (default {YEAR_END})")
    parser.add_argument("--date",       type=str, default=None,
                        help="Single init date YYYY-MM-DD (must be Mon or Thu)")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Parallel download threads (default 4)")
    parser.add_argument("--no-pf",      action="store_true",
                        help="Skip perturbed reforecast (pf); download cf only")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print tasks without downloading anything")
    args = parser.parse_args()

    # ── Build init date list ─────────────────────────────────────────────────
    if args.date:
        dates = [pd.Timestamp(args.date)]
        if dates[0].weekday() not in (0, 3):
            log.warning(f"{args.date} is not a Monday or Thursday — "
                        "ECMWF reforecasts only exist on Mon/Thu. Proceeding anyway.")
    else:
        dates = generate_init_dates(args.year_start, args.year_end)

    ftypes = ["cf"] if args.no_pf else ["cf", "pf"]

    log.info("=" * 68)
    log.info("ECMWF S2S Reforecast Downloader")
    log.info(f"  Output base : {OUT_BASE}")
    log.info(f"  Init dates  : {len(dates)}  ({dates[0].date()} → {dates[-1].date()})")
    log.info(f"  Forecast types: {ftypes}")
    log.info(f"  Surface vars: {list(SURFACE_VARS.values())}")
    log.info(f"  PL vars     : {list(PL_VARS.values())} @ {PRESSURE_LEVELS} hPa")
    log.info(f"  Lead steps  : day 1–46 (24 h intervals)")
    log.info(f"  Workers     : {args.workers}")
    log.info(f"  Log         : {log_file}")
    log.info("=" * 68)

    tasks = build_task_list(dates, ftypes)
    total = len(tasks)
    log.info(f"Total tasks: {total}")

    if args.dry_run:
        for t in tasks:
            status = "EXISTS" if _already_done(Path(t["out_path"])) else "PENDING"
            print(f"  [{status}] {Path(t['out_path']).relative_to(OUT_BASE)}")
        log.info("Dry run complete — nothing downloaded.")
        return

    # Count already done
    pending = [t for t in tasks if not _already_done(Path(t["out_path"]))]
    already = total - len(pending)
    log.info(f"  Already done : {already}")
    log.info(f"  To download  : {len(pending)}")

    if not pending:
        log.info("Nothing to download.")
        return

    # ── Execute downloads ────────────────────────────────────────────────────
    # One client per thread (cdsapi.Client is not thread-safe for shared state)
    def _worker(task):
        c = cdsapi.Client(quiet=True)
        return download_one(c, task)

    results = {"success": 0, "skipped": 0, "failed": 0}
    failed_tasks = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t for t in pending}
        done_count = already  # already-skipped counted in progress
        for fut in as_completed(futures):
            done_count += 1
            try:
                rec = fut.result()
                results[rec["status"]] += 1
                if rec["status"] == "failed":
                    failed_tasks.append(rec)
            except Exception as exc:
                results["failed"] += 1
                log.error(f"Unexpected worker error: {exc}")
            pct = 100 * done_count / total
            log.info(f"Progress: {done_count}/{total} ({pct:.1f}%)  "
                     f"ok={results['success']} skip={results['skipped']+already} "
                     f"fail={results['failed']}")

    log.info("=" * 68)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  Success  : {results['success']}")
    log.info(f"  Skipped  : {results['skipped'] + already}")
    log.info(f"  Failed   : {results['failed']}")
    if failed_tasks:
        log.info("  Failed tasks:")
        for t in failed_tasks:
            log.info(f"    {t['out_path']}")
    log.info(f"  Full log : {log_file}")
    log.info(f"  Req log  : {REQUEST_LOG}")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
