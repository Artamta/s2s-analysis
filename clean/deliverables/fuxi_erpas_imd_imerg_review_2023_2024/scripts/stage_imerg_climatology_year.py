#!/usr/bin/env python3
"""Stage one validated year of IMERG Final V07B for the fixed climatology.

The requested calendar window is the exact union needed by the 31 paired
2023--2024 JJAS initializations: 06 June through 25 October (142 days).
Per-day NPZ caches make every array task safely resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
FETCH_SCRIPT = WORKSPACE / "deliverables/imd_study/scripts/fetch_imerg_final.py"
DATA_ROOT = HERE / "data/imerg_climatology"
CACHE_ROOT = DATA_ROOT / ".daily_cache"
YEAR_ROOT = DATA_ROOT / "year_chunks"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch = import_file("review_imerg_fetch", FETCH_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_dates(year: int) -> pd.DatetimeIndex:
    dates = pd.date_range(f"{year}-06-06", f"{year}-10-25", freq="D")
    if len(dates) != 142:
        raise ValueError(f"expected 142 climatology dates, got {len(dates)}")
    return dates


def output_path(year: int) -> Path:
    return YEAR_ROOT / f"imerg_final_v07b_daily_{year}_0606_1025.nc"


def validate_output(path: Path, year: int) -> dict:
    dates = expected_dates(year)
    with xr.open_dataset(path) as source:
        if source.attrs.get("product") != "GPM_3IMERGDF.07":
            raise ValueError(f"{path}: wrong product")
        if source.attrs.get("revision") != "V07B":
            raise ValueError(f"{path}: wrong revision")
        found = pd.DatetimeIndex(source.period_start.values)
        if not found.equals(dates) or not found.is_unique:
            raise ValueError(f"{path}: wrong/duplicate dates")
        if source.precipitation.shape != (142, 348, 333):
            raise ValueError(f"{path}: unexpected precipitation shape")
        if int(source.precipitation_cnt.min()) != 48 or int(source.precipitation_cnt.max()) != 48:
            raise ValueError(f"{path}: not every grid cell has 48 half-hour samples")
        if not np.isfinite(source.precipitation.values).all():
            raise ValueError(f"{path}: non-finite precipitation")
        if float(source.precipitation.min()) < 0:
            raise ValueError(f"{path}: negative precipitation")
        return {
            "year": year,
            "date_count": int(source.sizes["period_start"]),
            "shape": list(source.precipitation.shape),
            "minimum_half_hour_count": int(source.precipitation_cnt.min()),
            "maximum_half_hour_count": int(source.precipitation_cnt.max()),
            "first_date": str(found[0].date()),
            "last_date": str(found[-1].date()),
        }


def main() -> int:
    args = parse_args()
    if args.year < 2001 or args.year > 2022:
        raise ValueError("fixed baseline permits only years 2001--2022")
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be in 1..16")

    YEAR_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = CACHE_ROOT / str(args.year)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = output_path(args.year)
    manifest = output.with_suffix(".manifest.json")
    if output.is_file() and not args.overwrite:
        audit = validate_output(output, args.year)
        print(json.dumps({"status": "VALIDATED_EXISTING", **audit}, indent=2))
        return 0

    cookie = Path.home() / ".urs_cookies"
    netrc = Path.home() / ".netrc"
    if not cookie.is_file() or not netrc.is_file():
        raise FileNotFoundError("Earthdata .netrc and .urs_cookies are required")

    dates = expected_dates(args.year)
    results: dict[pd.Timestamp, tuple] = {}
    failures: dict[pd.Timestamp, str] = {}
    print(
        f"staging IMERG Final V07B {args.year}: {dates[0]:%Y-%m-%d}.."
        f"{dates[-1]:%Y-%m-%d} ({len(dates)} days)",
        flush=True,
    )
    pending = list(dates)
    for round_index in range(3):
        round_workers = max(1, args.workers // (2**round_index))
        round_failures: dict[pd.Timestamp, str] = {}
        print(
            f"retrieval round {round_index + 1}/3: {len(pending)} day(s), "
            f"workers={round_workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=round_workers) as pool:
            futures = {
                pool.submit(fetch.fetch_one, date, cookie, cache_dir): date
                for date in pending
            }
            for done, future in enumerate(as_completed(futures), start=1):
                date = futures[future]
                try:
                    item = future.result()
                    rain, count = item[1], item[2]
                    if rain.shape != (348, 333) or count.shape != (348, 333):
                        raise ValueError(
                            f"unexpected field shape rain={rain.shape}, count={count.shape}"
                        )
                    if not np.isfinite(rain).all() or float(np.min(rain)) < 0:
                        raise ValueError("non-finite or negative daily precipitation")
                    if int(np.min(count)) != 48 or int(np.max(count)) != 48:
                        raise ValueError(
                            f"half-hour count is {int(np.min(count))}.."
                            f"{int(np.max(count))}, expected 48"
                        )
                    results[date] = item
                except Exception as exc:
                    round_failures[date] = f"{type(exc).__name__}: {exc}"
                if done == 1 or done % 10 == 0 or done == len(pending):
                    print(
                        f"round {round_index + 1}: processed {done}/{len(pending)}",
                        flush=True,
                    )
        failures = round_failures
        if not failures:
            break
        pending = sorted(failures)
        if round_index < 2:
            delay = 10 * (round_index + 1)
            print(
                f"round {round_index + 1} left {len(pending)} failure(s); "
                f"retrying after {delay}s",
                flush=True,
            )
            time.sleep(delay)

    if failures:
        detail = "; ".join(
            f"{stamp:%Y-%m-%d}: {message}"
            for stamp, message in sorted(failures.items())
        )
        raise RuntimeError(
            f"{len(failures)} day(s) failed; successful day caches were retained. {detail}"
        )

    ordered = [results[date] for date in dates]
    latitude, longitude = ordered[0][3], ordered[0][4]
    for item in ordered[1:]:
        if not (np.array_equal(item[3], latitude) and np.array_equal(item[4], longitude)):
            raise ValueError("IMERG grid changed between dates")
    precipitation = np.stack([item[1] for item in ordered]).astype(np.float32)
    counts = np.stack([item[2] for item in ordered]).astype(np.int16)
    source_time = np.asarray([item[5] for item in ordered], dtype=np.float64)
    source_bounds = np.asarray([item[6] for item in ordered], dtype=np.float64)

    dataset = xr.Dataset(
        data_vars={
            "precipitation": (
                ("period_start", "latitude", "longitude"),
                precipitation,
                {
                    "units": "mm",
                    "long_name": "one-day precipitation accumulation",
                    "cell_methods": "time: sum (interval: 1 day)",
                },
            ),
            "precipitation_cnt": (
                ("period_start", "latitude", "longitude"),
                counts,
                {"units": "count", "long_name": "valid half-hour retrieval count"},
            ),
            "period_end": (
                ("period_start",),
                dates.values.astype("datetime64[ns]") + np.timedelta64(1, "D"),
            ),
            "source_time": (("period_start",), source_time),
            "source_time_bounds": (("period_start", "bounds"), source_bounds),
        },
        coords={
            "period_start": dates.values.astype("datetime64[ns]"),
            "latitude": latitude,
            "longitude": longitude,
            "bounds": np.arange(2, dtype=np.int8),
        },
        attrs={
            "title": "IMERG Final V07B daily India subset for fixed S2S climatology",
            "product": "GPM_3IMERGDF.07",
            "revision": "V07B",
            "doi": "10.5067/GPM/IMERGDF/DAY/07",
            "baseline_year": args.year,
            "calendar_window": "06-06 through 10-25 inclusive",
            "date_count": 142,
            "minimum_half_hour_count": 48,
            "maximum_half_hour_count": 48,
            "source_url": fetch.BASE_URL,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={
            "precipitation": {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            },
            "precipitation_cnt": {"zlib": True, "complevel": 4},
        },
    )
    os.replace(temporary, output)
    audit = validate_output(output, args.year)
    payload = {
        "status": "PASSED",
        **dataset.attrs,
        **audit,
        "output": str(output),
        "sha256": sha256_file(output),
    }
    temporary_manifest = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
