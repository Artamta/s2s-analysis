#!/usr/bin/env python3
"""Download and validate one provider-year of all-season S2S forecasts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any


DATASET = "s2s-forecasts"
STORAGE_ROOT = Path("/storage/raj.ayush/s2s_final_data/final_iteration")
AREA = [40.0, 60.0, 0.0, 100.0]
FTYPES = {"cf": "control_forecast", "pf": "perturbed_forecast"}
PROVIDERS = ("ecmwf", "ukmo", "ncep", "cma", "cnrm")
DIRECT_T2M = {"ecmwf", "ukmo", "cma", "cnrm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=STORAGE_ROOT)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep-between", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-requests", type=int)
    return parser.parse_args()


def daily_endpoints() -> list[str]:
    return [str(hour) for hour in range(24, 42 * 24 + 1, 24)]


def daily_ranges() -> list[str]:
    return [f"{hour}_{hour + 24}" for hour in range(0, 42 * 24, 24)]


def six_hour_steps() -> list[str]:
    return [str(hour) for hour in range(6, 42 * 24 + 1, 6)]


def load_dates(path: Path, provider: str, year: int) -> list[dt.date]:
    dates = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["year"]) != year:
                continue
            if provider == "cnrm" and row["cnrm_available"] != "1":
                continue
            dates.append(dt.date.fromisoformat(row["init_date"]))
    if not dates:
        raise ValueError(f"no dates for {provider} {year} in {path}")
    if len(dates) != len(set(dates)):
        raise ValueError(f"duplicate dates for {provider} {year}")
    return dates


def request_base(provider: str, date: dt.date, ftype: str) -> dict[str, Any]:
    return {
        "origin": provider,
        "forecast_type": FTYPES[ftype],
        "year": str(date.year),
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
        "time": "00:00",
        "level_type": "single_level",
        "area": AREA,
        "data_format": "grib",
    }


def build_tasks(provider: str, dates: list[dt.date], root: Path) -> list[dict[str, Any]]:
    tasks = []
    for date in dates:
        base = request_base(provider, date, "cf")
        if provider in DIRECT_T2M:
            fields = (
                ("tp", "total_precipitation", daily_endpoints()),
                ("t2m", "2_m_temperature", daily_ranges()),
            )
        else:
            fields = (
                (
                    "surface",
                    [
                        "total_precipitation",
                        "maximum_2_m_temperature_in_the_last_6_hours",
                        "minimum_2_m_temperature_in_the_last_6_hours",
                    ],
                    six_hour_steps(),
                ),
            )
        for field, variable, leads in fields:
            for ftype, api_type in FTYPES.items():
                request = {
                    **base,
                    "forecast_type": api_type,
                    "variable": variable,
                    "leadtime_hour": leads,
                }
                target = (
                    root
                    / "raw"
                    / provider
                    / "forecast"
                    / f"annual{date.year}"
                    / field
                    / f"{date:%Y%m%d}_{ftype}.grib"
                )
                tasks.append(
                    {
                        "provider": provider,
                        "date": date,
                        "field": field,
                        "ftype": ftype,
                        "request": request,
                        "target": target,
                    }
                )
    return tasks


def request_hash(request: dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def open_grib_fields(path: Path) -> dict[str, Any]:
    import cfgrib

    datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    fields: dict[str, Any] = {}
    try:
        for dataset in datasets:
            for name, array in dataset.data_vars.items():
                fields[name] = array.load()
    finally:
        for dataset in datasets:
            dataset.close()
    return fields


def validate_grib(path: Path, provider: str, field: str, ftype: str) -> dict[str, Any]:
    fields = open_grib_fields(path)
    expected = {"tp", "t2m"} if field in ("tp", "t2m") else {"tp", "mx2t6", "mn2t6"}
    expected = {field} if field in ("tp", "t2m") else expected
    missing = expected - set(fields)
    if missing:
        raise ValueError(f"missing GRIB fields {sorted(missing)}; found {sorted(fields)}")
    details = {}
    member_counts = []
    for name in sorted(expected):
        array = fields[name]
        latitude = "latitude" if "latitude" in array.dims else "lat"
        longitude = "longitude" if "longitude" in array.dims else "lon"
        if array.sizes.get(latitude) != 27 or array.sizes.get(longitude) != 27:
            raise ValueError(f"{name}: expected 27x27 grid, found {dict(array.sizes)}")
        expected_steps = 168 if provider == "ncep" else 42
        if array.sizes.get("step", 1) != expected_steps:
            raise ValueError(f"{name}: expected {expected_steps} steps, found {dict(array.sizes)}")
        members = int(array.sizes.get("number", 1))
        if ftype == "cf" and members != 1:
            raise ValueError(f"control has {members} members")
        if ftype == "pf" and members < 2:
            raise ValueError(f"perturbed forecast has {members} members")
        if not bool(array.notnull().any()):
            raise ValueError(f"{name}: all values are missing")
        member_counts.append(members)
        details[name] = {
            "units": array.attrs.get("units"),
            "members": members,
            "steps": int(array.sizes.get("step", 1)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }
    return {"fields": details, "member_count": max(member_counts), "size_bytes": path.stat().st_size}


def download_one(
    client: Any,
    task: dict[str, Any],
    manifest: Path,
    retries: int,
    sleep_between: float,
    overwrite: bool,
) -> str:
    target: Path = task["target"]
    record = {
        "provider": task["provider"],
        "product": "operational_forecast",
        "init_date": task["date"].isoformat(),
        "field": task["field"],
        "forecast_type": task["ftype"],
        "lead_days": 42,
        "target": str(target),
        "request_hash": request_hash(task["request"]),
        "request": task["request"],
    }
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        try:
            record.update(validate_grib(target, task["provider"], task["field"], task["ftype"]))
            record.update(status="existing_valid", timestamp=dt.datetime.now(dt.timezone.utc).isoformat())
            append_jsonl(manifest, record)
            return "existing_valid"
        except Exception as exc:  # noqa: BLE001
            logging.error("existing invalid %s: %s", target, exc)
            return "failed"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    error = ""
    for attempt in range(1, retries + 1):
        try:
            client.retrieve(DATASET, task["request"], str(temporary))
            if not temporary.exists() or temporary.stat().st_size == 0:
                raise RuntimeError("download produced an empty file")
            qc = validate_grib(temporary, task["provider"], task["field"], task["ftype"])
            temporary.replace(target)
            record.update(qc)
            record.update(status="downloaded_valid", attempt=attempt, timestamp=dt.datetime.now(dt.timezone.utc).isoformat())
            append_jsonl(manifest, record)
            if sleep_between:
                time.sleep(sleep_between)
            return "downloaded_valid"
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            temporary.unlink(missing_ok=True)
            logging.exception("attempt %d/%d failed for %s", attempt, retries, target)
            if attempt < retries:
                time.sleep(min(600.0, 60.0 * attempt))
    record.update(status="failed", error=error, timestamp=dt.datetime.now(dt.timezone.utc).isoformat())
    append_jsonl(manifest, record)
    return "failed"


def main() -> int:
    args = parse_args()
    dates = load_dates(args.calendar, args.provider, args.year)
    tasks = build_tasks(args.provider, dates, args.output_root)
    if args.max_requests is not None:
        tasks = tasks[: args.max_requests]
    paths = [str(task["target"]) for task in tasks]
    hashes = [request_hash(task["request"]) for task in tasks]
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        raise ValueError("duplicate target path or request hash in task plan")
    manifest = (
        args.output_root / "manifests" / args.provider / "forecast" / f"annual{args.year}.jsonl"
    )
    summary = {
        "provider": args.provider,
        "year": args.year,
        "dates": len(dates),
        "requests": len(tasks),
        "manifest": str(manifest),
        "output_root": str(args.output_root),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return 0
    import cdsapi

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = cdsapi.Client(quiet=True)
    counts = {"downloaded_valid": 0, "existing_valid": 0, "failed": 0}
    for index, task in enumerate(tasks, 1):
        logging.info(
            "[%d/%d] %s %s %s %s",
            index, len(tasks), task["provider"], task["date"], task["field"], task["ftype"],
        )
        status = download_one(client, task, manifest, args.retries, args.sleep_between, args.overwrite)
        counts[status] += 1
    logging.info("summary=%s manifest=%s", counts, manifest)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
