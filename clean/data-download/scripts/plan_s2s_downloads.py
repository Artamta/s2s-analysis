#!/usr/bin/env python3
"""Create duplicate-safe request manifests without contacting ECDS."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = REPO_ROOT / "clean/config/archive_policy.json"
DEFAULT_DATES = REPO_ROOT / "clean/config/comparable_dates_2019_2026.csv"
DEFAULT_OUTPUT = REPO_ROOT / "clean/data-download/manifests/download_plan"
STORAGE_ROOT = Path("/storage/raj.ayush/s2s_final_data/final_iteration")
FTYPES = ("cf", "pf")
ECDS_FTYPES = {"cf": "control_forecast", "pf": "perturbed_forecast"}
FUXI_MMDD = (
    "0602", "0606", "0609", "0613", "0616", "0620", "0623", "0627",
    "0630", "0704", "0707", "0711", "0714", "0718", "0721", "0725",
    "0728", "0801", "0804", "0808", "0811", "0815", "0818", "0822",
    "0825", "0829", "0901", "0905", "0908", "0912", "0915", "0919",
    "0922", "0926", "0929",
)
ECMWF_2020_MMDD = (
    "0601", "0604", "0608", "0611", "0615", "0618", "0622", "0625",
    "0629", "0702", "0706", "0709", "0713", "0716", "0720", "0723",
    "0727", "0730", "0803", "0806", "0810", "0813", "0817", "0820",
    "0824", "0827", "0831", "0903", "0907", "0910", "0914", "0917",
    "0921", "0924", "0928",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "reforecast", "all"), default="all")
    parser.add_argument("--providers", default="ecmwf,ukmo,ncep")
    parser.add_argument("--years", default="2020-2024")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--dates-file", type=Path, default=DEFAULT_DATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true", help="Write JSONL and CSV manifests")
    return parser.parse_args()


def parse_years(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(item) for item in value.split("-", 1))
        return list(range(start, end + 1))
    return [int(item) for item in value.split(",") if item.strip()]


def stable_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def forecast_pairs(path: Path, years: set[int]) -> dict[int, list[dict[str, str]]]:
    result = {year: [] for year in sorted(years)}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            year = int(row["year"])
            if year in years:
                result[year].append(row)
    for year, rows in result.items():
        if len(rows) != 35:
            raise ValueError(f"expected 35 target dates for {year}, found {len(rows)}")
    return result


def daily_ranges(days: int) -> list[str]:
    return [f"{24 * (day - 1)}_{24 * day}" for day in range(1, days + 1)]


def six_hour_steps(days: int) -> list[str]:
    return [str(hour) for hour in range(6, days * 24 + 1, 6)]


def output_path(provider: str, product: str, label: str, name: str) -> Path:
    return STORAGE_ROOT / "raw" / provider / product / label / name


def make_task(
    *, provider: str, phase: str, init: str, ftype: str, variable: str,
    target: Path, request: dict[str, Any], status: str = "planned",
    temperature_semantics: str = "not_applicable", native_members: int | str,
) -> dict[str, Any]:
    exists = target.exists() and target.stat().st_size > 0
    return {
        "provider": provider,
        "phase": phase,
        "initialization": init,
        "forecast_type": ftype,
        "variable": variable,
        "temperature_semantics": temperature_semantics,
        "native_members": native_members,
        "status": "existing_target" if exists and status == "planned" else status,
        "target": str(target),
        "request_hash": stable_hash(request),
        "request": request,
    }


def operational_tasks(
    providers: list[str], years: list[int], pairs: dict[int, list[dict[str, str]]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    tasks = []
    base = {
        "dataset": "s2s-forecasts", "area": policy["domain_north_west_south_east"],
        "level_type": "single_level", "time": "00:00", "data_format": "grib",
    }
    for provider in providers:
        members = policy["providers"][provider]["forecast_members"]
        for year in years:
            for pair in pairs[year]:
                date = pair["ecmwf_init"] if provider == "ecmwf" else pair["fuxi_init"]
                lead_days = int(pair["ecmwf_lead_end_day"]) if provider == "ecmwf" else 42
                parsed = dt.date.fromisoformat(date)
                common = {
                    **base, "origin": provider, "year": str(parsed.year),
                    "month": f"{parsed.month:02d}", "day": f"{parsed.day:02d}",
                }
                if provider in ("ecmwf", "ukmo"):
                    specs = (
                        ("tp", "total_precipitation", [str(24 * day) for day in range(1, lead_days + 1)], "not_applicable"),
                        ("t2m", "2_m_temperature", daily_ranges(lead_days), "daily_mean_t2m"),
                    )
                    for variable, api_variable, steps, semantics in specs:
                        for ftype in FTYPES:
                            request = {**common, "forecast_type": ECDS_FTYPES[ftype], "variable": api_variable, "leadtime_hour": steps}
                            target = output_path(provider, "forecast", f"jjas{year}", f"{variable}/{date.replace('-', '')}_{ftype}.grib")
                            tasks.append(make_task(provider=provider, phase="forecast", init=date, ftype=ftype, variable=variable, target=target, request=request, temperature_semantics=semantics, native_members=members))
                else:
                    for ftype in FTYPES:
                        request = {**common, "forecast_type": ECDS_FTYPES[ftype], "variable": ["total_precipitation", "maximum_2_m_temperature_in_the_last_6_hours", "minimum_2_m_temperature_in_the_last_6_hours"], "leadtime_hour": six_hour_steps(42)}
                        target = output_path(provider, "forecast", f"jjas{year}", f"surface/{ftype}/{date.replace('-', '')}.grib")
                        tasks.append(make_task(provider=provider, phase="forecast", init=date, ftype=ftype, variable="tp+t2m_proxy", target=target, request=request, temperature_semantics="proxy_from_four_6h_tmin_tmax_midranges", native_members=members))
    return tasks


def reforecast_tasks(providers: list[str], policy: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    p = policy["providers"]
    base = {"dataset": "s2s-reforecasts", "area": policy["domain_north_west_south_east"], "level_type": "single_level", "time": "00:00", "data_format": "grib"}
    for provider in providers:
        spec = p[provider]
        hyears = [str(y) for y in range(spec["reforecast_years"][0], spec["reforecast_years"][1] + 1)]
        if provider == "ecmwf":
            slots, version_year, variables = ECMWF_2020_MMDD, 2020, (("t2m", "2_m_temperature", daily_ranges(42), "daily_mean_t2m"),)
        elif provider == "ukmo":
            slots, version_year = tuple(spec["reforecast_jjas_mmdd"]), spec["reforecast_model_version_year"]
            variables = (("tp", "total_precipitation", six_hour_steps(42), "not_applicable"), ("t2m", "2_m_temperature", daily_ranges(42), "daily_mean_t2m"))
        else:
            slots, version_year = FUXI_MMDD, 2011
            variables = (("tp+t2m_proxy", ["total_precipitation", "maximum_2_m_temperature_in_the_last_6_hours", "minimum_2_m_temperature_in_the_last_6_hours"], six_hour_steps(42), "proxy_from_four_6h_tmin_tmax_midranges"),)
        for mmdd in slots:
            version = spec.get("reforecast_model_version_date", f"{version_year}-{mmdd[:2]}-{mmdd[2:]}")
            for variable, api_variable, steps, semantics in variables:
                for ftype in FTYPES:
                    version_date = dt.date.fromisoformat(version)
                    request = {
                        **base, "origin": provider, "year": str(version_date.year),
                        "month": f"{version_date.month:02d}", "day": f"{version_date.day:02d}",
                        "hyear": hyears, "hmonth": mmdd[:2], "hday": mmdd[2:],
                        "forecast_type": ECDS_FTYPES[ftype], "variable": api_variable,
                        "leadtime_hour": steps,
                    }
                    suffix = "grib"
                    target = output_path(provider, "reforecast", "jjas_native", f"{variable}/{mmdd}_{ftype}.{suffix}")
                    tasks.append(make_task(provider=provider, phase="reforecast", init=version, ftype=ftype, variable=variable, target=target, request=request, temperature_semantics=semantics, native_members=spec["reforecast_members"]))
        if provider == "ecmwf":
            for mmdd in slots:
                for ftype in FTYPES:
                    source = Path(spec["local_tp_reforecast"]) / f"tp_{ftype}_{mmdd}.grib"
                    tasks.append(make_task(provider=provider, phase="reforecast", init=f"2020-{mmdd[:2]}-{mmdd[2:]}", ftype=ftype, variable="tp", target=source, request={"reuse_local_file": str(source)}, status="reuse_existing", native_members=spec["reforecast_members"]))
    return tasks


def validate(tasks: list[dict[str, Any]]) -> None:
    paths = [task["target"] for task in tasks]
    hashes = [task["request_hash"] for task in tasks if task["status"] not in ("reuse_existing", "existing_target")]
    duplicate_paths = [item for item, count in Counter(paths).items() if count > 1]
    duplicate_requests = [item for item, count in Counter(hashes).items() if count > 1]
    if duplicate_paths or duplicate_requests:
        raise ValueError(f"duplicate plan entries: paths={len(duplicate_paths)} requests={len(duplicate_requests)}")


def write_outputs(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, sort_keys=True) + "\n")
    fields = ("provider", "phase", "initialization", "forecast_type", "variable", "temperature_semantics", "native_members", "status", "target", "request_hash")
    with (path / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: task[key] for key in fields} for task in tasks)


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    unknown = sorted(set(providers) - set(policy["providers"]))
    if unknown:
        raise ValueError(f"unknown providers: {', '.join(unknown)}")
    years = parse_years(args.years)
    tasks = []
    if args.phase in ("forecast", "all"):
        tasks.extend(operational_tasks(providers, years, forecast_pairs(args.dates_file, set(years)), policy))
    if args.phase in ("reforecast", "all"):
        tasks.extend(reforecast_tasks(providers, policy))
    validate(tasks)
    if args.write:
        write_outputs(args.output_dir, tasks)
    counts = Counter((task["phase"], task["provider"], task["status"]) for task in tasks)
    print(json.dumps({"dry_run": True, "production_downloads": 0, "tasks": len(tasks), "output_dir": str(args.output_dir) if args.write else None, "counts": [{"phase": key[0], "provider": key[1], "status": key[2], "tasks": value} for key, value in sorted(counts.items())]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
