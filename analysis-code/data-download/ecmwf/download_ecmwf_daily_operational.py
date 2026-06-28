#!/usr/bin/env python3
"""
Download ECMWF S2S real-time operational forecasts into All_Model_Data.

This uses CDS/ECDS `s2s-forecasts`, matching the successful JFM 2026
downloader, so requests are visible on the CDS/ECDS request page.

Output layout:
  /storage/raj.ayush/All_Model_Data/ecmwf/<season>/
    2t/YYYYMMDD_{cf,pf}.nc   # step=24 only in this stream
    msl/YYYYMMDD_{cf,pf}.nc
    tp/YYYYMMDD_{cf,pf}.nc
    z/{1000,850,500,200}/YYYYMMDD_{cf,pf}.nc

The downloader is resumable: existing non-empty files are skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path


DEFAULT_OUT_ROOT = Path("/storage/raj.ayush/All_Model_Data/ecmwf/jjas2019")
DATASET = "s2s-forecasts"
STEPS = [str(h) for h in range(24, 46 * 24 + 1, 24)]  # daily leads 1..46 days
STEPS_2T = ["24"]  # ECMWF operational S2S only stores 2t at step=24
AREA = [40, 60, 0, 100]
GRID = [1.5, 1.5]
SFC_VARS = {
    "tp": "tp",
    "167": "2t",
    "msl": "msl",
}
Z_LEVELS = ("1000", "850", "500", "200")
FTYPES = {
    "cf": "control_forecast",
    "pf": "perturbed_forecast",
}


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y%m%d").date()


def iter_dates(start: dt.date, end: dt.date):
    date = start
    while date <= end:
        yield date
        date += dt.timedelta(days=1)


def filter_dates(dates: list[dt.date], weekdays: str) -> list[dt.date]:
    if weekdays == "all":
        return dates
    if weekdays == "mon-thu":
        return [date for date in dates if date.weekday() in (0, 3)]
    raise ValueError(f"Unsupported weekdays mode: {weekdays}")


def ymd(date: dt.date) -> str:
    return date.strftime("%Y%m%d")


def target_path(out_root: Path, kind: str, ftype: str, date: dt.date, level: str | None = None) -> Path:
    if kind == "z":
        if level is None:
            raise ValueError("z requests require a pressure level")
        return out_root / "z" / level / f"{ymd(date)}_{ftype}.nc"
    return out_root / SFC_VARS[kind] / f"{ymd(date)}_{ftype}.nc"


def is_complete(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def ensure_dirs(out_root: Path) -> None:
    for short_name in SFC_VARS.values():
        (out_root / short_name).mkdir(parents=True, exist_ok=True)
    for level in Z_LEVELS:
        (out_root / "z" / level).mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)


def base_request(date: dt.date, ftype: str) -> dict:
    return {
        "origin": "ecmwf",
        "forecast_type": FTYPES[ftype],
        "year": str(date.year),
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
        "time": "00:00:00",
        "step": STEPS,
        "area": AREA,
        "grid": GRID,
        "data_format": "netcdf",
    }


def build_request(date: dt.date, ftype: str, kind: str, level: str | None = None) -> dict:
    req = base_request(date, ftype)
    if kind == "z":
        if level is None:
            raise ValueError("z requests require a pressure level")
        req.update(
            {
                "level_type": "pressure_level",
                "variable": "156",
                "level": level,
            }
        )
    else:
        steps = STEPS_2T if kind == "167" else STEPS
        req.update(
            {
                "level_type": "single_level",
                "variable": kind,
                "step": steps,
            }
        )
    return req


def request_label(kind: str, level: str | None) -> str:
    return f"z{level}" if kind == "z" else SFC_VARS[kind]


def download_one(
    client,
    out_root: Path,
    date: dt.date,
    ftype: str,
    kind: str,
    level: str | None,
    retries: int,
    sleep_between: float,
) -> str:
    target = target_path(out_root, kind, ftype, date, level)
    label = request_label(kind, level)
    if is_complete(target):
        logging.info("SKIP  %s %s %s size=%d", ymd(date), label, ftype, target.stat().st_size)
        return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    req = build_request(date, ftype, kind, level)
    for attempt in range(1, retries + 1):
        try:
            logging.info("START %s %s %s attempt=%d target=%s", ymd(date), label, ftype, attempt, target)
            client.retrieve(DATASET, req, str(tmp))
            if not is_complete(tmp):
                raise RuntimeError(f"download produced empty file: {tmp}")
            tmp.replace(target)
            logging.info("DONE  %s %s %s size=%d", ymd(date), label, ftype, target.stat().st_size)
            if sleep_between:
                time.sleep(sleep_between)
            return "downloaded"
        except Exception as exc:  # noqa: BLE001 - keep long downloads resumable
            logging.exception("FAIL  %s %s %s attempt=%d: %s", ymd(date), label, ftype, attempt, exc)
            if tmp.exists():
                tmp.unlink()
            if "MarsNoDataError" in str(exc) or "MARS returned no data" in str(exc):
                return "failed"
            if attempt < retries:
                time.sleep(min(300, 30 * attempt))
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ECMWF operational ENS forecasts")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-between", type=float, default=1.0)
    parser.add_argument("--weekdays", choices=("all", "mon-thu"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_root = Path(args.out_root)
    start = parse_date(args.start)
    end = parse_date(args.end)
    dates = filter_dates(list(iter_dates(start, end)), args.weekdays)
    requests = []
    for date in dates:
        for ftype in ("cf", "pf"):
            for kind in SFC_VARS:
                requests.append((date, ftype, kind, None))
            for level in Z_LEVELS:
                requests.append((date, ftype, "z", level))

    ensure_dirs(out_root)
    logging.info("ECMWF daily operational downloader")
    logging.info("Output root : %s", out_root)
    logging.info("Dates       : %s to %s (%d inits, weekdays=%s)", args.start, args.end, len(dates), args.weekdays)
    logging.info("Requests    : %d (7 vars/levels x cf/pf x dates)", len(requests))
    logging.info("Steps       : %s", STEPS)
    logging.info("Grid        : %s global", GRID)

    if args.dry_run:
        missing = sum(not is_complete(target_path(out_root, kind, ftype, date, level)) for date, ftype, kind, level in requests)
        logging.info("Dry run: missing=%d existing=%d", missing, len(requests) - missing)
        return 0

    import cdsapi

    client = cdsapi.Client(quiet=True)
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for date, ftype, kind, level in requests:
        status = download_one(client, out_root, date, ftype, kind, level, args.retries, args.sleep_between)
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
