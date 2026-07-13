#!/usr/bin/env python3
"""Build exact common all-season dates from live ECDS constraints."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2025
CORE_PROVIDERS = ("ecmwf", "ukmo", "ncep", "cma")
SECONDARY_PROVIDER = "cnrm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--constraints-file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to clean/config/all_season_dates_<start>_<end>.csv",
    )
    return parser.parse_args()


def load_constraints(path: Path | None) -> list[dict[str, Any]]:
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    import cdsapi
    from ecmwf.datastores import Client

    url, key, _ = cdsapi.api.get_url_key_verify(None, None, None)
    client = Client(url=url, key=key, progress=False)
    return client.get_collection("s2s-forecasts").constraints


def required_leads(provider: str, variable: str) -> set[str]:
    if provider == "ncep":
        return {str(hour) for hour in range(6, 42 * 24 + 1, 6)}
    if variable == "total_precipitation":
        return {str(hour) for hour in range(24, 42 * 24 + 1, 24)}
    return {f"{hour}_{hour + 24}" for hour in range(0, 42 * 24, 24)}


def variable_dates(
    constraints: list[dict[str, Any]], provider: str, year: int, variable: str
) -> set[dt.date]:
    dates: set[dt.date] = set()
    required = required_leads(provider, variable)
    for row in constraints:
        if provider not in row.get("origin", []):
            continue
        if str(year) not in row.get("year", []):
            continue
        if variable not in row.get("variable", []):
            continue
        if not required.issubset(set(row.get("leadtime_hour", []))):
            continue
        for month in row.get("month", []):
            for day in row.get("day", []):
                try:
                    dates.add(dt.date(year, int(month), int(day)))
                except ValueError:
                    pass
    return dates


def provider_dates(
    constraints: list[dict[str, Any]], provider: str, year: int
) -> set[dt.date]:
    variables = (
        (
            "total_precipitation",
            "maximum_2_m_temperature_in_the_last_6_hours",
            "minimum_2_m_temperature_in_the_last_6_hours",
        )
        if provider == "ncep"
        else ("total_precipitation", "2_m_temperature")
    )
    sets = [variable_dates(constraints, provider, year, variable) for variable in variables]
    return set.intersection(*sets)


def build_rows(
    constraints: list[dict[str, Any]], start_year: int, end_year: int
) -> list[dict[str, Any]]:
    rows = []
    for year in range(start_year, end_year + 1):
        available = {
            provider: provider_dates(constraints, provider, year)
            for provider in (*CORE_PROVIDERS, SECONDARY_PROVIDER)
        }
        core_dates = set.intersection(*(available[p] for p in CORE_PROVIDERS))
        for date in sorted(core_dates):
            rows.append(
                {
                    "year": year,
                    "init_date": date.isoformat(),
                    "core_providers": "+".join(CORE_PROVIDERS),
                    "cnrm_available": int(date in available[SECONDARY_PROVIDER]),
                    "lead_days": 42,
                    "date_policy": "exact_common_initialization",
                    "source": "ecds_s2s_forecasts_constraints",
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start year must not be later than end year")
    output = args.output or (
        REPO_ROOT
        / f"clean/config/all_season_dates_{args.start_year}_{args.end_year}.csv"
    )
    constraints = load_constraints(args.constraints_file)
    rows = build_rows(constraints, args.start_year, args.end_year)
    if not rows:
        raise ValueError("no exact common dates found for the requested years")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[int, dict[str, int]] = {}
    for year in range(args.start_year, args.end_year + 1):
        selected = [row for row in rows if row["year"] == year]
        counts[year] = {
            "core": len(selected),
            "cnrm": sum(int(row["cnrm_available"]) for row in selected),
        }
    print(json.dumps({"output": str(output), "rows": len(rows), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
