#!/usr/bin/env python3
"""Download and validate ECMWF operational S2S data for one JJAS year."""

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
AREA = (40.0, 60.0, 0.0, 100.0)  # north, west, south, east
GRID = (1.5, 1.5)
FORECAST_TYPES = {
    "cf": "control_forecast",
    "pf": "perturbed_forecast",
}
VARIABLES: dict[str, dict[str, Any]] = {
    "tp": {
        "level_type": "single_level",
        "variable": "tp",
        "level": "",
        "directory": ("tp",),
        "netcdf_names": ("tp",),
        "step_encoding": "end_hour",
    },
    "t2m": {
        "level_type": "single_level",
        "variable": "167",
        "level": "",
        "directory": ("t2m",),
        "netcdf_names": ("t2m",),
        "step_encoding": "daily_average_range",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--start-mmdd", default="0601")
    parser.add_argument("--end-mmdd", default="0930")
    parser.add_argument("--cadence", choices=("mon-thu", "all"), default="mon-thu")
    parser.add_argument(
        "--dates-file",
        type=Path,
        default=None,
        help="Comparable-date CSV containing year, fuxi_init, and ecmwf_init columns",
    )
    parser.add_argument("--variables", default="tp,t2m")
    parser.add_argument("--forecast-types", default="cf,pf")
    parser.add_argument("--lead-days", type=int, default=42)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep-between", type=float, default=2.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_csv_choices(value: str, allowed: dict[str, Any], label: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    if not values:
        raise ValueError(f"at least one {label} is required")
    return list(dict.fromkeys(values))


def parse_mmdd(year: int, value: str) -> dt.date:
    return dt.datetime.strptime(f"{year}{value}", "%Y%m%d").date()


def initialization_dates(start: dt.date, end: dt.date, cadence: str) -> list[dt.date]:
    dates = []
    current = start
    while current <= end:
        if cadence == "all" or current.weekday() in (0, 3):
            dates.append(current)
        current += dt.timedelta(days=1)
    return dates


def load_date_pairs(path: Path, year: int) -> list[tuple[dt.date, dict[str, Any]]]:
    pairs: list[tuple[dt.date, dict[str, Any]]] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["year"]) != year:
                continue
            init = row["ecmwf_init"]
            if init in seen:
                raise ValueError(f"duplicate ECMWF initialization in {path}: {init}")
            seen.add(init)
            pairs.append(
                (
                    dt.date.fromisoformat(init),
                    {
                        "fuxi_init": row["fuxi_init"],
                        "ecmwf_init": init,
                        "init_offset_days": int(row["init_offset_days"]),
                        "pair_mode": row["pair_mode"],
                        "ecmwf_lead_start_day": int(row["ecmwf_lead_start_day"]),
                        "ecmwf_lead_end_day": int(row["ecmwf_lead_end_day"]),
                        "pair_policy": row["pair_policy"],
                        "fuxi_target_source": row["fuxi_target_source"],
                        "ecmwf_retrieval_status": row["ecmwf_retrieval_status"],
                    },
                )
            )
    if not pairs:
        raise ValueError(f"no comparable dates for {year} in {path}")
    return pairs


def target_path(out_root: Path, date: dt.date, variable: str, ftype: str) -> Path:
    spec = VARIABLES[variable]
    return out_root.joinpath(*spec["directory"], f"{date:%Y%m%d}_{ftype}.nc")


def request_steps(variable: str, lead_days: int) -> list[str]:
    if VARIABLES[variable]["step_encoding"] == "daily_average_range":
        return [f"{(day - 1) * 24}-{day * 24}" for day in range(1, lead_days + 1)]
    return [str(day * 24) for day in range(1, lead_days + 1)]


def build_request(date: dt.date, variable: str, ftype: str, lead_days: int) -> dict[str, Any]:
    spec = VARIABLES[variable]
    request: dict[str, Any] = {
        "origin": "ecmwf",
        "forecast_type": FORECAST_TYPES[ftype],
        "level_type": spec["level_type"],
        "variable": spec["variable"],
        "year": f"{date.year:04d}",
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
        "time": "00:00:00",
        "step": request_steps(variable, lead_days),
        "area": list(AREA),
        "grid": list(GRID),
        "data_format": "netcdf",
    }
    if spec["level"]:
        request["level"] = spec["level"]
    return request


def request_hash(request: dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def inspect_netcdf(path: Path, variable: str, ftype: str, lead_days: int) -> dict[str, Any]:
    import xarray as xr

    spec = VARIABLES[variable]
    open_errors = []
    dataset = None
    for kwargs in ({}, {"decode_timedelta": False}):
        try:
            dataset = xr.open_dataset(path, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001 - report both decoding attempts
            open_errors.append(str(exc))
    if dataset is None:
        raise ValueError(f"cannot open NetCDF: {'; '.join(open_errors)}")

    try:
        names = [name for name in spec["netcdf_names"] if name in dataset.data_vars]
        if not names:
            raise ValueError(
                f"expected one of {spec['netcdf_names']}, found {tuple(dataset.data_vars)}"
            )
        sizes = dict(dataset.sizes)
        step_count = int(sizes.get("step", sizes.get("lead_time", 0)))
        if step_count < lead_days:
            raise ValueError(f"expected at least {lead_days} lead steps, found {step_count}")

        lat_count = int(sizes.get("latitude", sizes.get("lat", 0)))
        lon_count = int(sizes.get("longitude", sizes.get("lon", 0)))
        if lat_count < 2 or lon_count < 2:
            raise ValueError(f"invalid spatial dimensions: lat={lat_count}, lon={lon_count}")

        member_dim = "number" if "number" in sizes else "member" if "member" in sizes else None
        member_count = int(sizes[member_dim]) if member_dim else 1
        if ftype == "pf" and member_count < 2:
            raise ValueError(f"perturbed forecast has only {member_count} member")

        return {
            "data_variable": names[0],
            "member_dimension": member_dim or "control_only",
            "member_count": member_count,
            "step_count": step_count,
            "latitude_count": lat_count,
            "longitude_count": lon_count,
        }
    finally:
        dataset.close()


def base_record(
    date: dt.date,
    variable: str,
    ftype: str,
    target: Path,
    request: dict[str, Any],
    lead_days: int,
    comparison_pair: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = VARIABLES[variable]
    record = {
        "provider": "ecmwf",
        "dataset": DATASET,
        "product": "operational_forecast",
        "season": f"jjas{date.year}",
        "init_date": date.isoformat(),
        "init_time_utc": "00:00:00",
        "forecast_type": ftype,
        "variable": variable,
        "level_hpa": int(spec["level"]) if spec["level"] else None,
        "lead_start_day": 1,
        "lead_end_day": lead_days,
        "area_north_west_south_east": list(AREA),
        "grid_degrees": list(GRID),
        "all_native_members_requested": True,
        "file_path": str(target),
        "request_hash": request_hash(request),
        "step_encoding": spec["step_encoding"],
    }
    if comparison_pair:
        record["comparison_pair"] = comparison_pair
    return record


def download_one(
    client: Any,
    date: dt.date,
    variable: str,
    ftype: str,
    target: Path,
    manifest: Path,
    lead_days: int,
    retries: int,
    sleep_between: float,
    overwrite: bool,
    comparison_pair: dict[str, Any] | None,
) -> str:
    request = build_request(date, variable, ftype, lead_days)
    record = base_record(
        date,
        variable,
        ftype,
        target,
        request,
        lead_days,
        comparison_pair,
    )
    record["started_utc"] = utc_now()

    if target.exists() and target.stat().st_size > 0 and not overwrite:
        try:
            qc = inspect_netcdf(target, variable, ftype, lead_days)
            record.update(qc)
            record.update(
                status="existing_valid",
                size_bytes=target.stat().st_size,
                completed_utc=utc_now(),
            )
            append_manifest(manifest, record)
            logging.info("SKIP  %s %s %s existing valid", date, variable, ftype)
            return "existing_valid"
        except Exception as exc:  # noqa: BLE001 - preserve invalid source for inspection
            record.update(status="existing_invalid", error=str(exc), completed_utc=utc_now())
            append_manifest(manifest, record)
            logging.error("INVALID existing %s: %s", target, exc)
            return "failed"

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            logging.info(
                "START %s %s %s attempt=%d target=%s",
                date,
                variable,
                ftype,
                attempt,
                target,
            )
            client.retrieve(DATASET, request, str(temporary))
            if not temporary.exists() or temporary.stat().st_size == 0:
                raise RuntimeError("download produced an empty file")
            qc = inspect_netcdf(temporary, variable, ftype, lead_days)
            temporary.replace(target)
            record.update(qc)
            record.update(
                status="downloaded_valid",
                attempt=attempt,
                size_bytes=target.stat().st_size,
                completed_utc=utc_now(),
            )
            append_manifest(manifest, record)
            logging.info(
                "DONE  %s %s %s members=%d steps=%d size=%d",
                date,
                variable,
                ftype,
                qc["member_count"],
                qc["step_count"],
                target.stat().st_size,
            )
            if sleep_between:
                time.sleep(sleep_between)
            return "downloaded_valid"
        except Exception as exc:  # noqa: BLE001 - long batch must record and continue
            last_error = str(exc)
            logging.exception("FAIL  %s %s %s attempt=%d", date, variable, ftype, attempt)
            if temporary.exists():
                temporary.unlink()
            no_data = "MarsNoDataError" in last_error or "MARS returned no data" in last_error
            if no_data or attempt == retries:
                break
            time.sleep(min(300.0, 30.0 * attempt))

    record.update(status="failed", error=last_error, completed_utc=utc_now())
    append_manifest(manifest, record)
    return "failed"


def main() -> int:
    args = parse_args()
    variables = parse_csv_choices(args.variables, VARIABLES, "variable")
    forecast_types = parse_csv_choices(args.forecast_types, FORECAST_TYPES, "forecast type")
    if args.lead_days < 1 or args.lead_days > 46:
        raise ValueError("lead-days must be between 1 and the ECMWF native maximum of 46")

    start = parse_mmdd(args.year, args.start_mmdd)
    end = parse_mmdd(args.year, args.end_mmdd)
    if end < start:
        raise ValueError("end date precedes start date")
    if args.dates_file:
        date_pairs = load_date_pairs(args.dates_file, args.year)
    else:
        date_pairs = [
            (date, {}) for date in initialization_dates(start, end, args.cadence)
        ]
    dates = [date for date, _ in date_pairs]

    out_root = args.out_root or (
        STORAGE_ROOT / "raw" / "ecmwf" / "forecast" / f"jjas{args.year}"
    )
    manifest = args.manifest or out_root / "manifests" / "requests.jsonl"
    tasks = [
        (
            date,
            variable,
            ftype,
            pair,
            int(pair.get("ecmwf_lead_end_day", args.lead_days)),
        )
        for date, pair in date_pairs
        for variable in variables
        for ftype in forecast_types
    ]
    if args.max_requests is not None:
        tasks = tasks[: args.max_requests]
    requested_lead_end_days = sorted({task[4] for task in tasks})
    if requested_lead_end_days and requested_lead_end_days[-1] > 46:
        raise ValueError("comparable-date plan exceeds ECMWF's 46-day native range")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("ECMWF operational S2S download plan")
    logging.info(
        "year=%d dates=%d source=%s",
        args.year,
        len(dates),
        args.dates_file or args.cadence,
    )
    logging.info(
        "variables=%s forecast_types=%s target_lead_days=%d request_lead_ends=%s",
        variables,
        forecast_types,
        args.lead_days,
        requested_lead_end_days,
    )
    logging.info("requests=%d output=%s", len(tasks), out_root)
    logging.info("all native members requested; no ECMWF number selector is used")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "year": args.year,
                    "initialization_count": len(dates),
                    "first_initialization": dates[0].isoformat() if dates else None,
                    "last_initialization": dates[-1].isoformat() if dates else None,
                    "cadence": "dates_file" if args.dates_file else args.cadence,
                    "dates_file": str(args.dates_file) if args.dates_file else None,
                    "variables": variables,
                    "forecast_types": forecast_types,
                    "lead_days": args.lead_days,
                    "request_lead_end_days": requested_lead_end_days,
                    "request_count": len(tasks),
                    "out_root": str(out_root),
                    "manifest": str(manifest),
                },
                indent=2,
            )
        )
        return 0

    import cdsapi

    client = cdsapi.Client(quiet=True)
    counts = {"downloaded_valid": 0, "existing_valid": 0, "failed": 0}
    for index, (date, variable, ftype, pair, request_lead_days) in enumerate(tasks, 1):
        status = download_one(
            client=client,
            date=date,
            variable=variable,
            ftype=ftype,
            target=target_path(out_root, date, variable, ftype),
            manifest=manifest,
            lead_days=request_lead_days,
            retries=args.retries,
            sleep_between=args.sleep_between,
            overwrite=args.overwrite,
            comparison_pair=pair,
        )
        counts[status] += 1
        logging.info("PROGRESS %d/%d counts=%s", index, len(tasks), counts)

    logging.info("SUMMARY %s manifest=%s", counts, manifest)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
