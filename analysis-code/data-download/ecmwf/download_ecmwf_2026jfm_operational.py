#!/usr/bin/env python3
"""
download_ecmwf_2026jfm_operational.py
=======================================
Download ECMWF S2S *operational* forecasts for JFM 2026
(Jan 01 – Mar 31 2026, all Mon+Thu init dates).

This is the actual real-time forecast data you compare against observations,
NOT the hindcasts used for climatology.

Variables
---------
Surface:
  tp    total_precipitation
  2t    2m_temperature
  mx2t  maximum_2m_temperature_in_the_last_24_hours
  mn2t  minimum_2m_temperature_in_the_last_24_hours
  msl   mean_sea_level_pressure

Pressure levels (200 / 500 / 850 / 1000 hPa):
  z     geopotential
  t     temperature
  u     u_component_of_wind
  v     v_component_of_wind
  q     specific_humidity

Output
------
  /storage/raj.ayush/All_Model_Data/models/ecmwf/jfm2026/
  ├── tp/    tp_cf_20260102.grib   tp_pf_20260102.grib  ...
  ├── 2t/
  ├── mx2t/  mn2t/  msl/
  ├── z/200/  z/500/  z/850/  z/1000/
  ├── t/200/  u/200/  v/200/  q/200/ ...

Each file: one init date (full YYYY-MM-DD in name), one variable,
           one forecast type (cf/pf), GRIB format.

Usage
-----
  python download_ecmwf_2026jfm_operational.py --dry-run
  python download_ecmwf_2026jfm_operational.py --date 2026-01-02   # single date test
  python download_ecmwf_2026jfm_operational.py --sfc-only
  python download_ecmwf_2026jfm_operational.py --pl-only
  python download_ecmwf_2026jfm_operational.py --no-pf
  python download_ecmwf_2026jfm_operational.py --workers 3
  nohup python download_ecmwf_2026jfm_operational.py --workers 3 \
        > logs/jfm2026_$(date +%Y%m%d).out 2>&1 &
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
OUT_BASE   = Path("/storage/raj.ayush/All_Model_Data/models/ecmwf/jfm2026")
LOG_DIR    = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── INIT DATES ────────────────────────────────────────────────────────────────
# All Mon + Thu in JFM 2026
_all_days = pd.date_range("2026-01-01", "2026-03-31", freq="D")
JFM2026_DATES = [d for d in _all_days if d.weekday() in (0, 3)]  # Mon=0, Thu=3

# ── API PARAMETERS ────────────────────────────────────────────────────────────
# Operational forecast dataset (not reforecasts)
DATASET = "s2s-forecasts"

# Lead times: 24 h increments, day-1 through day-46
STEPS = [str(h) for h in range(24, 24 * 46 + 1, 24)]

SURFACE_VARS = {
    "total_precipitation":                         "tp",
    "2m_temperature":                              "2t",
    "maximum_2m_temperature_in_the_last_24_hours": "mx2t",
    "minimum_2m_temperature_in_the_last_24_hours": "mn2t",
    "mean_sea_level_pressure":                     "msl",
}

PL_VARS = {
    "geopotential":        "z",
    "temperature":         "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity":   "q",
}

PRESSURE_LEVELS = ["200", "500", "850", "1000"]

FTYPES = {
    "cf": "control_forecast",
    "pf": "perturbed_forecast",
}

# ── RETRY ─────────────────────────────────────────────────────────────────────
MAX_RETRIES   = 5
RETRY_BACKOFF = [30, 60, 120, 300, 600]

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_file = LOG_DIR / f"jfm2026_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
REQUEST_LOG = LOG_DIR / "requests_jfm2026.jsonl"


def _log_request(record: dict):
    with open(REQUEST_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def _outdir(short_name: str, level: str | None = None) -> Path:
    d = OUT_BASE / short_name / level if level else OUT_BASE / short_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _outpath(short_name: str, date: pd.Timestamp, ftype_key: str,
             level: str | None = None) -> Path:
    fname = f"{short_name}_{ftype_key}_{date:%Y%m%d}.grib"
    return _outdir(short_name, level) / fname


def _done(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def download_one(task: dict) -> dict:
    out = Path(task["out"])
    if _done(out):
        log.info(f"SKIP  {out.relative_to(OUT_BASE)}")
        return {**task, "status": "skipped"}

    date = pd.Timestamp(task["date"])
    req = {
        "origin":        "ecmwf",
        "forecast_type": FTYPES[task["ftype_key"]],
        "level_type":    task["level_type"],
        "variable":      task["cds_name"],
        "year":          str(date.year),
        "month":         f"{date.month:02d}",
        "day":           f"{date.day:02d}",
        "time":          "00:00",
        "step":          STEPS,
        "data_format":   "grib",
    }
    if task.get("level"):
        req["level"] = task["level"]

    label   = out.relative_to(OUT_BASE)
    last_ex = None
    client  = cdsapi.Client(quiet=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"START {label}  (attempt {attempt})")
            t0 = time.time()
            client.retrieve(DATASET, req, str(out))
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


def build_tasks(dates: list[pd.Timestamp], ftypes: list[str],
                do_sfc: bool, do_pl: bool) -> list[dict]:
    tasks = []
    for date in dates:
        for ftype_key in ftypes:
            if do_sfc:
                for cds_name, short_name in SURFACE_VARS.items():
                    out = _outpath(short_name, date, ftype_key)
                    tasks.append(dict(date=str(date), ftype_key=ftype_key,
                                      cds_name=cds_name, short_name=short_name,
                                      level_type="single_level", level=None,
                                      out=str(out)))
            if do_pl:
                for cds_name, short_name in PL_VARS.items():
                    for lev in PRESSURE_LEVELS:
                        out = _outpath(short_name, date, ftype_key, lev)
                        tasks.append(dict(date=str(date), ftype_key=ftype_key,
                                          cds_name=cds_name, short_name=short_name,
                                          level_type="pressure_level", level=lev,
                                          out=str(out)))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Download ECMWF S2S operational forecasts for JFM 2026."
    )
    parser.add_argument("--date",     type=str, default=None,
                        help="Single init date YYYY-MM-DD instead of all JFM 2026")
    parser.add_argument("--no-pf",   action="store_true",
                        help="Skip perturbed forecast; cf only")
    parser.add_argument("--sfc-only", action="store_true",
                        help="Surface variables only")
    parser.add_argument("--pl-only",  action="store_true",
                        help="Pressure-level variables only")
    parser.add_argument("--workers",  type=int, default=3,
                        help="Parallel threads (default 3)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show pending tasks, do not download")
    args = parser.parse_args()

    dates  = [pd.Timestamp(args.date)] if args.date else JFM2026_DATES
    ftypes = ["cf"] if args.no_pf else ["cf", "pf"]
    do_sfc = not args.pl_only
    do_pl  = not args.sfc_only

    tasks   = build_tasks(dates, ftypes, do_sfc, do_pl)
    pending = [t for t in tasks if not _done(Path(t["out"]))]
    skipped = len(tasks) - len(pending)

    log.info("=" * 68)
    log.info("ECMWF S2S Operational Forecast Downloader — JFM 2026")
    log.info(f"  Output       : {OUT_BASE}")
    log.info(f"  Init dates   : {len(dates)}  "
             f"({dates[0].date()} → {dates[-1].date()})")
    log.info(f"  Fcst types   : {ftypes}")
    log.info(f"  Surface vars : {list(SURFACE_VARS.values()) if do_sfc else 'skipped'}")
    log.info(f"  PL vars/levs : {list(PL_VARS.values())} @ {PRESSURE_LEVELS} hPa"
             if do_pl else "  PL vars      : skipped")
    log.info(f"  Total tasks  : {len(tasks)}  |  done: {skipped}"
             f"  |  pending: {len(pending)}")
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
    log.info(f"  Success : {counts['success']}")
    log.info(f"  Skipped : {counts['skipped']}")
    log.info(f"  Failed  : {counts['failed']}")
    for f in failed:
        log.info(f"    FAILED: {f}")
    log.info(f"  Log     : {log_file}")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
