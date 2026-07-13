#!/usr/bin/env python3
"""Build reproducible ECMWF/FuXi JJAS initialization pairs for 2019-2026."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "clean" / "config" / "comparable_dates_2019_2026.csv"
DAILY_ECMWF_START = dt.date(2023, 6, 28)
ECMWF_DELAY = dt.timedelta(days=2)

# The fixed JJAS MMDD cadence present in the FuXi 2002-2021 archive.
FUXI_MMDD = (
    "0602",
    "0606",
    "0609",
    "0613",
    "0616",
    "0620",
    "0623",
    "0627",
    "0630",
    "0704",
    "0707",
    "0711",
    "0714",
    "0718",
    "0721",
    "0725",
    "0728",
    "0801",
    "0804",
    "0808",
    "0811",
    "0815",
    "0818",
    "0822",
    "0825",
    "0829",
    "0901",
    "0905",
    "0908",
    "0912",
    "0915",
    "0919",
    "0922",
    "0926",
    "0929",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=dt.date.today(),
        help="Availability date in YYYY-MM-DD form (default: today)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fuxi_dates(year: int) -> list[dt.date]:
    return [
        dt.datetime.strptime(f"{year}{mmdd}", "%Y%m%d").date()
        for mmdd in FUXI_MMDD
    ]


def ecmwf_dates(year: int) -> list[dt.date]:
    # A FuXi start in early June can need the preceding late-May ECMWF cycle.
    start = dt.date(year, 6, 1) - dt.timedelta(days=4)
    end = dt.date(year, 9, 30)
    dates = []
    current = start
    while current <= end:
        if current >= DAILY_ECMWF_START or current.weekday() in (0, 3):
            dates.append(current)
        current += dt.timedelta(days=1)
    return dates


def pair_dates(
    targets: list[dt.date],
    available: list[dt.date],
) -> list[dt.date]:
    """Find the minimum-lag, one-to-one ECMWF assignment for all FuXi starts."""

    @lru_cache(maxsize=None)
    def solve(
        target_index: int,
        first_available: int,
    ) -> tuple[int, tuple[int, ...]] | None:
        if target_index == len(targets):
            return 0, ()

        target = targets[target_index]
        best: tuple[int, tuple[int, ...]] | None = None
        for available_index in range(first_available, len(available)):
            lag = (target - available[available_index]).days
            if lag < 0:
                break
            if lag > 4:
                continue
            remainder = solve(target_index + 1, available_index + 1)
            if remainder is None:
                continue
            candidate = (lag + remainder[0], (available_index, *remainder[1]))
            if best is None or candidate < best:
                best = candidate
        return best

    solution = solve(0, 0)
    if solution is None:
        raise ValueError("no one-to-one ECMWF schedule can cover every FuXi window")
    return [available[index] for index in solution[1]]


def build_rows(
    start_year: int,
    end_year: int,
    as_of: dt.date,
) -> list[dict[str, object]]:
    rows = []
    retrieval_cutoff = as_of - ECMWF_DELAY
    for year in range(start_year, end_year + 1):
        targets = fuxi_dates(year)
        available = ecmwf_dates(year)
        assignments = pair_dates(targets, available)
        for fuxi_init, ecmwf_init in zip(targets, assignments, strict=True):
            offset = (ecmwf_init - fuxi_init).days
            lead_start = 1 - offset
            lead_end = 42 - offset
            exact = offset == 0
            rows.append(
                {
                    "year": year,
                    "fuxi_init": fuxi_init.isoformat(),
                    "ecmwf_init": ecmwf_init.isoformat(),
                    "init_offset_days": offset,
                    "pair_mode": "exact_init" if exact else "same_valid_window",
                    "ecmwf_lead_start_day": lead_start,
                    "ecmwf_lead_end_day": lead_end,
                    "ecmwf_frequency": (
                        "daily" if ecmwf_init >= DAILY_ECMWF_START else "mon_thu"
                    ),
                    "pair_policy": "one_to_one_minimum_lag",
                    "fuxi_target_source": (
                        "archive" if year <= 2021 else "model_run_required"
                    ),
                    "availability_as_of": as_of.isoformat(),
                    "ecmwf_retrieval_status": (
                        "mature" if ecmwf_init <= retrieval_cutoff else "future_or_embargoed"
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = build_rows(args.start_year, args.end_year, args.as_of)
    write_csv(args.output, rows)

    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[int(row["year"])][str(row["pair_mode"])] += 1
    print(f"output={args.output}")
    print(f"rows={len(rows)}")
    for year in sorted(counts):
        print(
            f"{year}: exact={counts[year]['exact_init']} "
            f"aligned={counts[year]['same_valid_window']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
