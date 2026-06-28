#!/usr/bin/env python3
"""
Download daily NCEP S2S forecasts into All_Model_Data.

Output layout:
  /storage/raj.ayush/All_Model_Data/ncep/<season>/
    surface/{cf,pf}/YYYYMMDD.grib   # 2t, mx2t6, mn2t6, tp
    z/500/{cf,pf}/YYYYMMDD.grib     # gh at 500 hPa

The downloader is intentionally resumable: existing non-empty files are skipped,
and new downloads are written to a temporary file before being moved into place.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path


OUT_ROOT = Path("/storage/raj.ayush/All_Model_Data/ncep/jfm2026")
DATASET = "s2s-forecasts"
AREA = [50, 55, 0, 105]  # North, West, South, East
GRID = [1.5, 1.5]
STEPS = [str(h) for h in range(24, 1057, 24)]  # daily leads: 1..44 days
SURFACE_VARIABLES = ["2t", "mx2t6", "mn2t6", "tp"]


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y%m%d").date()


def iter_dates(start: dt.date, end: dt.date):
    date = start
    while date <= end:
        yield date
        date += dt.timedelta(days=1)


def ymd(date: dt.date) -> str:
    return date.strftime("%Y%m%d")


def forecast_type(ftype: str) -> str:
    if ftype == "cf":
        return "control_forecast"
    if ftype == "pf":
        return "perturbed_forecast"
    raise ValueError(f"Unsupported forecast type: {ftype}")


def target_path(kind: str, ftype: str, date: dt.date) -> Path:
    if kind == "surface":
        return OUT_ROOT / "surface" / ftype / f"{ymd(date)}.grib"
    if kind == "z500":
        return OUT_ROOT / "z" / "500" / ftype / f"{ymd(date)}.grib"
    raise ValueError(f"Unsupported request kind: {kind}")


def build_request(kind: str, ftype: str, date: dt.date) -> dict:
    request = {
        "origin": "ncep",
        "forecast_type": forecast_type(ftype),
        "year": str(date.year),
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
        "time": "00:00",
        "step": STEPS,
        "area": AREA,
        "grid": GRID,
        "data_format": "grib",
    }

    if kind == "surface":
        request.update(
            {
                "level_type": "single_level",
                "variable": SURFACE_VARIABLES,
            }
        )
    elif kind == "z500":
        request.update(
            {
                "level_type": "pressure",
                "variable": "gh",
                "level": "500",
            }
        )
    else:
        raise ValueError(f"Unsupported request kind: {kind}")

    return request


def ensure_dirs() -> None:
    for kind in ("surface",):
        for ftype in ("cf", "pf"):
            (OUT_ROOT / kind / ftype).mkdir(parents=True, exist_ok=True)
    for ftype in ("cf", "pf"):
        (OUT_ROOT / "z" / "500" / ftype).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)


def is_complete(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def download_one(client, kind: str, ftype: str, date: dt.date, retries: int, sleep_between: float) -> str:
    target = target_path(kind, ftype, date)
    if is_complete(target):
        logging.info("SKIP  %s %s %s exists size=%d", ymd(date), kind, ftype, target.stat().st_size)
        return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    request = build_request(kind, ftype, date)
    for attempt in range(1, retries + 1):
        try:
            logging.info("START %s %s %s attempt=%d target=%s", ymd(date), kind, ftype, attempt, target)
            client.retrieve(DATASET, request, str(tmp))
            if not is_complete(tmp):
                raise RuntimeError(f"download produced empty file: {tmp}")
            tmp.replace(target)
            logging.info("DONE  %s %s %s size=%d", ymd(date), kind, ftype, target.stat().st_size)
            if sleep_between:
                time.sleep(sleep_between)
            return "downloaded"
        except Exception as exc:  # noqa: BLE001 - keep going through CDS/API failures
            logging.exception("FAIL  %s %s %s attempt=%d: %s", ymd(date), kind, ftype, attempt, exc)
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(min(300, 30 * attempt))

    return "failed"


def main() -> int:
    global OUT_ROOT

    parser = argparse.ArgumentParser(description="Download daily NCEP S2S forecasts")
    parser.add_argument(
        "--out-root",
        default=str(OUT_ROOT),
        help="Output root, e.g. /storage/raj.ayush/All_Model_Data/ncep/jfm2026",
    )
    parser.add_argument("--start", default="20260101", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20260331", help="End date YYYYMMDD")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep-between", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    OUT_ROOT = Path(args.out_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start = parse_date(args.start)
    end = parse_date(args.end)
    dates = list(iter_dates(start, end))
    requests = [(date, kind, ftype) for date in dates for kind in ("surface", "z500") for ftype in ("cf", "pf")]

    ensure_dirs()
    logging.info("NCEP daily downloader")
    logging.info("Output root : %s", OUT_ROOT)
    logging.info("Dates       : %s to %s (%d daily inits)", args.start, args.end, len(dates))
    logging.info("Requests    : %d (surface/z500 x cf/pf)", len(requests))
    logging.info("Variables   : surface=%s, z500=gh@500", ",".join(SURFACE_VARIABLES))
    logging.info("Steps       : %s..%s hours (%d daily leads)", STEPS[0], STEPS[-1], len(STEPS))

    if args.dry_run:
        missing = sum(not is_complete(target_path(kind, ftype, date)) for date, kind, ftype in requests)
        logging.info("Dry run: missing=%d existing=%d", missing, len(requests) - missing)
        return 0

    import cdsapi

    client = cdsapi.Client()
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for date, kind, ftype in requests:
        status = download_one(client, kind, ftype, date, args.retries, args.sleep_between)
        counts[status] += 1

    logging.info(
        "Summary downloaded=%d skipped=%d failed=%d",
        counts["downloaded"],
        counts["skipped"],
        counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
