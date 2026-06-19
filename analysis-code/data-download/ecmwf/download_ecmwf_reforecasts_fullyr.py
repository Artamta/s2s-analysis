#!/usr/bin/env python3
"""
download_ecmwf_reforecasts.py
==============================
Download ECMWF S2S reforecasts (hindcasts) from ECDS for all
Mon+Thu initialization dates across a full calendar year (all MM-DD combos).

One GRIB per (variable, MM-DD, forecast-type): each file contains all
hindcast years (typically 2000-2019, 20 members for pf) stacked on the
`time` dimension — exactly the same layout as the existing tp / z500 files.

Output directory
----------------
  /storage/raj.ayush/All_Model_Data/models/ecmwf/90_day_data/
  ├── tp/
  │   ├── tp_cf_0102.grib   # init Jan-02, control, all hindcast years
  │   ├── tp_pf_0102.grib   # init Jan-02, perturbed (all members)
  │   └── ...
  ├── 2t/
  ├── mx2t/
  ├── mn2t/
  ├── msl/
  ├── z/
  │   ├── 200/   z_cf_0102.grib ...
  │   ├── 500/
  │   ├── 850/
  │   └── 1000/
  ├── t/200/ ...   t/500/ ...   t/850/ ...   t/1000/
  ├── u/200/ ...
  ├── v/200/ ...
  └── q/200/ ...

Logs are written to:
  /storage/raj.ayush/All_Model_Data/models/ecmwf/download_scr/logs/

Usage
-----
  # Dry-run: show what would be downloaded
  python download_ecmwf_reforecasts.py --dry-run

  # Single init date test (must be Mon or Thu)
  python download_ecmwf_reforecasts.py --mmdd 0102

  # Surface variables only
  python download_ecmwf_reforecasts.py --sfc-only

  # Pressure-level variables only
  python download_ecmwf_reforecasts.py --pl-only

  # cf only (skip perturbed to move faster)
  python download_ecmwf_reforecasts.py --no-pf

  # Control number of parallel API calls (default 3)
  python download_ecmwf_reforecasts.py --workers 5

  # Background run with nohup
  nohup python download_ecmwf_reforecasts.py --workers 3 \
        > logs/run_$(date +%Y%m%d).out 2>&1 &

Notes
-----
- ECMWF reforecasts initialize every Mon + Thu only.
- One API request returns all ~20 hindcast years in a single GRIB.
- Requires ~/.cdsapirc with ECDS key (url: https://ecds.ecmwf.int/api).
- cdsapi >= 0.7.7 required.
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cdsapi
import pandas as pd

# ── PATHS ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path("/storage/raj.ayush/All_Model_Data/models/ecmwf/download_scr")
OUT_BASE   = Path("/storage/raj.ayush/All_Model_Data/models/ecmwf/90_day_data")
LOG_DIR    = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── API PARAMETERS ─────────────────────────────────────────────────────────────
# One ECMWF hindcast year anchor we use to generate all Mon/Thu MM-DD combos.
# The actual hindcast years returned are controlled by ECDS (typically 2000-2019).
ANCHOR_YEAR = 2020   # a year whose Mon/Thu dates cover full calendar

# Lead times: 24 h increments, day-1 through day-46
STEPS = [str(h) for h in range(24, 24 * 46 + 1, 24)]

# Surface variables: CDS long-name → subdirectory short-name
SURFACE_VARS = {
    "total_precipitation":                         "tp",
    "2m_temperature":                              "2t",
    "maximum_2m_temperature_in_the_last_24_hours": "mx2t",
    "minimum_2m_temperature_in_the_last_24_hours": "mn2t",
    "mean_sea_level_pressure":                     "msl",
}

# Pressure-level variables: CDS long-name → subdirectory short-name
PL_VARS = {
    "geopotential":          "z",
    "temperature":           "t",
    "u_component_of_wind":   "u",
    "v_component_of_wind":   "v",
    "specific_humidity":     "q",
}

PRESSURE_LEVELS = ["200", "500", "850", "1000"]

FTYPES = {
    "cf": "control_reforecast",
    "pf": "perturbed_reforecast",
}

# ── RETRY ──────────────────────────────────────────────────────────────────────
MAX_RETRIES   = 5
RETRY_BACKOFF = [30, 60, 120, 300, 600]   # seconds

# ── LOGGING ───────────────────────────────────────────────────────────────────
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

REQUEST_LOG = LOG_DIR / "requests.jsonl"


def _log_request(record: dict):
    with open(REQUEST_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def _outdir(short_name: str, level: str | None = None) -> Path:
    d = OUT_BASE / short_name / level if level else OUT_BASE / short_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _outpath(short_name: str, mmdd: str, ftype_key: str,
             level: str | None = None) -> Path:
    return _outdir(short_name, level) / f"{short_name}_{ftype_key}_{mmdd}.grib"


def _done(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def _base_request(mmdd: str, ftype_key: str) -> dict:
    """Build the common part of a CDS request from a MM-DD string."""
    month = mmdd[:2]
    day   = mmdd[2:]
    return {
        "origin":        "ecmwf",
        "forecast_type": FTYPES[ftype_key],
        "year":          str(ANCHOR_YEAR),
        "month":         month,
        "day":           day,
        "time":          "00:00",
        "step":          STEPS,
        "data_format":   "grib",
    }


def download_one(task: dict) -> dict:
    """Single download task with retries. Creates its own cdsapi.Client."""
    out = Path(task["out"])
    if _done(out):
        log.info(f"SKIP  {out.relative_to(OUT_BASE)}")
        return {**task, "status": "skipped"}

    req = _base_request(task["mmdd"], task["ftype_key"])
    req["level_type"] = task["level_type"]
    req["variable"]   = task["cds_name"]
    if task.get("level"):
        req["level"] = task["level"]

    label   = out.relative_to(OUT_BASE)
    last_ex = None
    client  = cdsapi.Client(quiet=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"START {label}  (attempt {attempt})")
            t0 = time.time()
            client.retrieve("s2s-reforecasts", req, str(out))
            elapsed = time.time() - t0
            size_mb = out.stat().st_size / 1024 ** 2
            log.info(f"DONE  {label}  {size_mb:.1f} MB  {elapsed:.0f}s")
            rec = {**task, "status": "success", "size_mb": round(size_mb, 2),
                   "elapsed_s": round(elapsed, 1), "attempt": attempt,
                   "ts": datetime.utcnow().isoformat(), "out": str(out)}
            _log_request(rec)
            return rec
        except Exception as exc:
            last_ex = exc
            log.warning(f"FAIL  {label}  attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                log.info(f"      retry in {wait}s ...")
                time.sleep(wait)

    log.error(f"GIVE UP {label}: {last_ex}")
    rec = {**task, "status": "failed", "error": str(last_ex),
           "ts": datetime.utcnow().isoformat(), "out": str(out)}
    _log_request(rec)
    return rec


def all_mon_thu_mmdds() -> list[str]:
    """Return all Mon+Thu MM-DD strings for a full calendar year."""
    dates = pd.date_range(f"{ANCHOR_YEAR}-01-01", f"{ANCHOR_YEAR}-12-31", freq="D")
    return [d.strftime("%m%d") for d in dates if d.weekday() in (0, 3)]


def build_tasks(mmdds: list[str], ftypes: list[str],
                do_sfc: bool, do_pl: bool) -> list[dict]:
    tasks = []
    for mmdd in mmdds:
        for ftype_key in ftypes:
            if do_sfc:
                for cds_name, short_name in SURFACE_VARS.items():
                    out = _outpath(short_name, mmdd, ftype_key)
                    tasks.append(dict(mmdd=mmdd, ftype_key=ftype_key,
                                      cds_name=cds_name, short_name=short_name,
                                      level_type="single_level", level=None,
                                      out=str(out)))
            if do_pl:
                for cds_name, short_name in PL_VARS.items():
                    for lev in PRESSURE_LEVELS:
                        out = _outpath(short_name, mmdd, ftype_key, lev)
                        tasks.append(dict(mmdd=mmdd, ftype_key=ftype_key,
                                          cds_name=cds_name, short_name=short_name,
                                          level_type="pressure_level", level=lev,
                                          out=str(out)))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Download ECMWF S2S reforecasts — all Mon+Thu inits, full year."
    )
    parser.add_argument("--mmdd",     type=str, default=None,
                        help="Single init MMDD (e.g. 0102) instead of all dates")
    parser.add_argument("--no-pf",   action="store_true",
                        help="Skip perturbed reforecast; download cf only")
    parser.add_argument("--sfc-only", action="store_true",
                        help="Surface variables only")
    parser.add_argument("--pl-only",  action="store_true",
                        help="Pressure-level variables only")
    parser.add_argument("--workers",  type=int, default=3,
                        help="Parallel download threads (default 3)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show pending tasks without downloading")
    args = parser.parse_args()

    mmdds   = [args.mmdd] if args.mmdd else all_mon_thu_mmdds()
    ftypes  = ["cf"] if args.no_pf else ["cf", "pf"]
    do_sfc  = not args.pl_only
    do_pl   = not args.sfc_only

    tasks   = build_tasks(mmdds, ftypes, do_sfc, do_pl)
    pending = [t for t in tasks if not _done(Path(t["out"]))]
    skipped = len(tasks) - len(pending)

    log.info("=" * 68)
    log.info("ECMWF S2S Reforecast Downloader — full-year edition")
    log.info(f"  Output       : {OUT_BASE}")
    log.info(f"  Init dates   : {len(mmdds)}  ({mmdds[0]} → {mmdds[-1]})")
    log.info(f"  Fcst types   : {ftypes}")
    log.info(f"  Surface vars : {list(SURFACE_VARS.values()) if do_sfc else 'skipped'}")
    log.info(f"  PL vars      : {list(PL_VARS.values())} @ {PRESSURE_LEVELS} hPa"
             if do_pl else "  PL vars      : skipped")
    log.info(f"  Total tasks  : {len(tasks)}  |  already done: {skipped}"
             f"  |  to download: {len(pending)}")
    log.info(f"  Workers      : {args.workers}")
    log.info(f"  Log          : {log_file}")
    log.info("=" * 68)

    if args.dry_run:
        for t in tasks:
            tag = "EXISTS " if _done(Path(t["out"])) else "PENDING"
            print(f"  [{tag}] {Path(t['out']).relative_to(OUT_BASE)}")
        log.info("Dry-run complete — nothing downloaded.")
        return

    if not pending:
        log.info("Nothing to download.")
        return

    counts = {"success": 0, "skipped": skipped, "failed": 0}
    failed = []
    done_n = skipped

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, t): t for t in pending}
        for fut in as_completed(futures):
            done_n += 1
            try:
                rec = fut.result()
                counts[rec["status"]] += 1
                if rec["status"] == "failed":
                    failed.append(rec["out"])
            except Exception as exc:
                counts["failed"] += 1
                log.error(f"Worker exception: {exc}")
            pct = 100 * done_n / len(tasks)
            log.info(f"Progress {done_n}/{len(tasks)} ({pct:.1f}%)  "
                     f"ok={counts['success']} skip={counts['skipped']} "
                     f"fail={counts['failed']}")

    log.info("=" * 68)
    log.info("DONE")
    log.info(f"  Success  : {counts['success']}")
    log.info(f"  Skipped  : {counts['skipped']}")
    log.info(f"  Failed   : {counts['failed']}")
    for f in failed:
        log.info(f"    FAILED: {f}")
    log.info(f"  Log      : {log_file}")
    log.info(f"  Req log  : {REQUEST_LOG}")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
