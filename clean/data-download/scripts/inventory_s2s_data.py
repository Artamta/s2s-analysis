#!/usr/bin/env python3
"""Inventory existing S2S forecast files for the clean 2019-2025 workspace.

The script does not download anything. It scans known storage roots, infers
provider/year/date/variable/level/forecast type from paths, and writes two CSVs:

  clean/data-download/manifests/existing_file_inventory.csv
  clean/data-download/manifests/coverage_summary.csv
  clean/data-download/manifests/reforecast_mmdd_summary.csv

It is intentionally lightweight: "present" means a non-empty file exists. A
later QC pass can open files to confirm member counts, lead coverage, and units.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "clean" / "data-download" / "manifests"

PROVIDERS = ("ecmwf", "ukmo", "ncep", "fuxi")
EXTENSIONS = {".nc", ".grib", ".grb", ".grib2", ".7z", ".zarr"}

DEFAULT_ROOTS = (
    Path("/storage/raj.ayush/All_Model_Data/ecmwf"),
    Path("/storage/raj.ayush/All_Model_Data/ukmo"),
    Path("/storage/raj.ayush/All_Model_Data/ncep"),
    Path("/storage/raj.ayush/All_Model_Data/fuxi"),
    Path("/storage/raj.ayush/archive/All_Model_Data/ecmwf"),
    Path("/storage/raj.ayush/archive/All_Model_Data/models/ecmwf"),
    Path("/storage/raj.ayush/archive/All_Model_Data/models/ukmo"),
    Path("/storage/raj.ayush/archive/All_Model_Data/models/ncep"),
    Path("/storage/raj.ayush/archive/All_Model_Data/models/fuxi"),
    Path("/storage/raj.ayush/s2s_final_data/jjas/fuxi_combined"),
)

DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
MMDD_RE = re.compile(r"(?<!\d)(0[1-9]|1[0-2])([0-3]\d)(?!\d)")
SEASON_RE = re.compile(r"(jjas|jfm|mam|son|djf)(20\d{2})", re.IGNORECASE)
FTYPE_RE = re.compile(r"(?:^|[_\-.])(cf|pf)(?:[_\-.]|$)", re.IGNORECASE)

VARIABLE_ALIASES = {
    "tp": "tp",
    "precip": "tp",
    "precipitation": "tp",
    "2t": "t2m",
    "t2m": "t2m",
    "mx2t": "tmax",
    "mx2t6": "tmax",
    "mn2t": "tmin",
    "mn2t6": "tmin",
    "msl": "msl",
    "z": "z",
    "z500": "z500",
    "gh": "z",
    "u": "u",
    "u850": "u850",
    "v": "v",
    "v850": "v850",
    "q": "q",
    "t": "t",
}

LEVELS = {"1000", "925", "850", "700", "600", "500", "400", "300", "250", "200", "150", "100", "50"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--months", default="6,7,8,9", help="Comma-separated months to check, default JJAS")
    parser.add_argument("--providers", default="ecmwf,ukmo,ncep,fuxi")
    parser.add_argument("--root", action="append", type=Path, default=None, help="Extra or replacement scan root")
    parser.add_argument("--only-default-roots", action="store_true", help="Ignore --root additions")
    return parser.parse_args()


def iter_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            yield path


def provider_from_path(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    for provider in PROVIDERS:
        if provider in parts or any(provider in p for p in parts):
            return provider
    return None


def season_from_path(path: Path) -> tuple[str, int | None]:
    text = str(path)
    m = SEASON_RE.search(text)
    if m:
        return m.group(1).lower(), int(m.group(2))
    return "", None


def infer_date(path: Path) -> tuple[str, int | None, str]:
    """Return (init_date, year, mmdd). init_date may be empty for MMDD files."""
    text = str(path)
    m = DATE_RE.search(text)
    if m:
        raw = m.group(1)
        try:
            d = dt.datetime.strptime(raw, "%Y%m%d").date()
        except ValueError:
            return "", None, ""
        return d.isoformat(), d.year, d.strftime("%m%d")

    # Reforecast files are often stored as variable_ftype_MMDD.grib.
    name = path.name
    candidates = list(MMDD_RE.finditer(name))
    if candidates:
        mmdd = "".join(candidates[-1].groups())
        return "", None, mmdd
    return "", None, ""


def infer_ftype(path: Path) -> str:
    m = FTYPE_RE.search(path.name)
    if m:
        return m.group(1).lower()
    for part in reversed([p.lower() for p in path.parts[:-1]]):
        if part in {"cf", "pf"}:
            return part
    return ""


def infer_level(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for part in reversed(parts):
        if part in LEVELS:
            return part
    m = re.search(r"(?<!\d)(1000|925|850|700|600|500|400|300|250|200|150|100|50)(?!\d)", path.name)
    return m.group(1) if m else ""


def infer_variable(path: Path, provider: str) -> str:
    parts = [p.lower() for p in path.parts]
    if "surface" in parts:
        return "surface_multi"
    # Prefer directory names immediately after provider/year-like folders.
    for part in reversed(parts[:-1]):
        clean = part.replace("-", "_")
        if clean in VARIABLE_ALIASES:
            var = VARIABLE_ALIASES[clean]
            level = infer_level(path)
            if var in {"z", "u", "v", "t", "q"} and level:
                return f"{var}{level}"
            return var
    stem_tokens = re.split(r"[_\-.]+", path.stem.lower())
    for tok in stem_tokens:
        if tok in VARIABLE_ALIASES:
            var = VARIABLE_ALIASES[tok]
            level = infer_level(path)
            if var in {"z", "u", "v", "t", "q"} and level:
                return f"{var}{level}"
            return var
    if provider == "fuxi" and path.suffix.lower() == ".7z":
        return "all"
    return ""


def expected_dates(year: int, months: set[int], mode: str) -> set[str]:
    start = dt.date(year, min(months), 1)
    end_month = max(months)
    next_month = dt.date(year + (end_month == 12), 1 if end_month == 12 else end_month + 1, 1)
    end = next_month - dt.timedelta(days=1)
    out = set()
    d = start
    while d <= end:
        if d.month in months:
            if mode == "all" or d.weekday() in (0, 3):
                out.add(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def scan(args: argparse.Namespace) -> list[dict[str, object]]:
    roots = list(DEFAULT_ROOTS)
    if args.root and not args.only_default_roots:
        roots.extend(args.root)
    wanted = {p.strip().lower() for p in args.providers.split(",") if p.strip()}
    roots = [
        root for root in roots
        if any(provider in str(root).lower() for provider in wanted)
    ]

    rows = []
    seen = set()
    for root in roots:
        for path in iter_files(root):
            if path in seen:
                continue
            seen.add(path)
            provider = provider_from_path(path)
            if not provider or provider not in wanted:
                continue
            init_date, year, mmdd = infer_date(path)
            season, season_year = season_from_path(path)
            if year is None:
                year = season_year
            variable = infer_variable(path, provider)
            level = infer_level(path)
            rows.append({
                "provider": provider,
                "product_hint": "reforecast" if not init_date and mmdd else "forecast",
                "season_hint": season,
                "year": year or "",
                "init_date": init_date,
                "mmdd": mmdd,
                "forecast_type": infer_ftype(path),
                "variable": variable,
                "level": level,
                "file_path": str(path),
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "nonempty": path.stat().st_size > 0,
            })
    return rows


def summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    months = {int(x) for x in args.months.split(",") if x.strip()}
    by_key: dict[tuple, set[str]] = defaultdict(set)
    file_counts: dict[tuple, int] = defaultdict(int)
    bytes_by_key: dict[tuple, int] = defaultdict(int)

    for row in rows:
        year = row["year"]
        init_date = row["init_date"]
        if not year or not init_date:
            continue
        year = int(year)
        d = dt.date.fromisoformat(str(init_date))
        if year < args.start_year or year > args.end_year or d.month not in months:
            continue
        key = (row["provider"], year, row["variable"], row["level"], row["forecast_type"])
        by_key[key].add(str(init_date))
        file_counts[key] += 1
        bytes_by_key[key] += int(row["size_bytes"])

    summary = []
    for key in sorted(by_key):
        provider, year, variable, level, ftype = key
        dates = by_key[key]
        exp_all = expected_dates(year, months, "all")
        exp_mon_thu = expected_dates(year, months, "mon-thu")
        mon_thu_hits = dates & exp_mon_thu
        all_hits = dates & exp_all
        summary.append({
            "provider": provider,
            "year": year,
            "variable": variable,
            "level": level,
            "forecast_type": ftype,
            "files": file_counts[key],
            "unique_init_dates": len(dates),
            "mon_thu_expected": len(exp_mon_thu),
            "mon_thu_present": len(mon_thu_hits),
            "mon_thu_missing": len(exp_mon_thu - dates),
            "all_days_expected": len(exp_all),
            "all_days_present": len(all_hits),
            "all_days_missing": len(exp_all - dates),
            "size_gb": round(bytes_by_key[key] / 1024**3, 3),
            "first_date": min(dates) if dates else "",
            "last_date": max(dates) if dates else "",
            "missing_mon_thu_dates": " ".join(sorted(exp_mon_thu - dates)[:40]),
        })
    return summary


def summarize_reforecast_mmdd(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    months = {int(x) for x in args.months.split(",") if x.strip()}
    by_key: dict[tuple, set[str]] = defaultdict(set)
    file_counts: dict[tuple, int] = defaultdict(int)
    bytes_by_key: dict[tuple, int] = defaultdict(int)

    for row in rows:
        if row["product_hint"] != "reforecast" or not row["mmdd"]:
            continue
        mmdd = str(row["mmdd"])
        if int(mmdd[:2]) not in months:
            continue
        key = (row["provider"], row["variable"], row["level"], row["forecast_type"])
        by_key[key].add(mmdd)
        file_counts[key] += 1
        bytes_by_key[key] += int(row["size_bytes"])

    summary = []
    for key in sorted(by_key):
        provider, variable, level, ftype = key
        mmdds = by_key[key]
        summary.append({
            "provider": provider,
            "variable": variable,
            "level": level,
            "forecast_type": ftype,
            "files": file_counts[key],
            "unique_mmdds": len(mmdds),
            "size_gb": round(bytes_by_key[key] / 1024**3, 3),
            "first_mmdd": min(mmdds) if mmdds else "",
            "last_mmdd": max(mmdds) if mmdds else "",
            "mmdds": " ".join(sorted(mmdds)),
        })
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = scan(args)
    summary = summarize(rows, args)
    reforecast_summary = summarize_reforecast_mmdd(rows, args)
    inv_path = OUT_DIR / "existing_file_inventory.csv"
    sum_path = OUT_DIR / "coverage_summary.csv"
    ref_path = OUT_DIR / "reforecast_mmdd_summary.csv"
    write_csv(inv_path, rows)
    write_csv(sum_path, summary)
    write_csv(ref_path, reforecast_summary)

    print(f"scanned_files={len(rows)}")
    print(f"inventory={inv_path}")
    print(f"summary={sum_path}")
    print(f"reforecast_summary={ref_path}")
    for row in summary[:30]:
        print(
            f"{row['provider']:6s} {row['year']} {row['variable'] or 'unknown':7s} "
            f"{row['forecast_type'] or '--':2s} dates={row['unique_init_dates']:3d} "
            f"mon_thu={row['mon_thu_present']:3d}/{row['mon_thu_expected']:3d} "
            f"all={row['all_days_present']:3d}/{row['all_days_expected']:3d}"
        )
    if len(summary) > 30:
        print(f"... {len(summary) - 30} more summary rows in CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
