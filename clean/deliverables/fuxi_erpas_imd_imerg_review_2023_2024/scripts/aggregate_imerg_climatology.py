#!/usr/bin/env python3
"""Aggregate 2001--2022 IMERG year chunks onto the audited 1.5-degree grid."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
YEAR_ROOT = HERE / "data/imerg_climatology/year_chunks"
OUTPUT = HERE / "data/imerg_climatology/imerg_final_v07b_climatology_2001_2022_1p5_daily.nc"
BUILD_SCRIPT = (
    WORKSPACE
    / "deliverables/fuxi_erpas_acc_multiseason_2023_2024/scripts/build_acc_csv.py"
)


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = import_file("review_imerg_climo_build_support", BUILD_SCRIPT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def year_path(year: int) -> Path:
    return YEAR_ROOT / f"imerg_final_v07b_daily_{year}_0606_1025.nc"


def main() -> int:
    years = list(range(2001, 2023))
    paths = [year_path(year) for year in years]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "all 22 independently validated year chunks are required; missing: "
            + ", ".join(missing)
        )

    all_cases, excluded = build.build_cases()
    if len(all_cases) != 101:
        raise ValueError(f"expected 101 paired cycles, got {len(all_cases)}")
    with xr.open_dataset(build.SOURCE_DATA) as source:
        reference = source.load()
    target_lat, target_lon, _, original_weight, original_land_support = (
        build.accmod.load_land_support(reference)
    )
    _, _, india_fraction, weight, land_support, imd_audit = build.remap_imd(
        all_cases,
        target_lat,
        target_lon,
        original_land_support,
        original_weight > 0,
    )
    mask = weight > 0
    if int(mask.sum()) != 169:
        raise ValueError(f"expected fixed 169-cell support, got {int(mask.sum())}")

    total = None
    source_lat = source_lon = None
    month_days = None
    source_manifests: list[dict] = []
    for index, (year, path) in enumerate(zip(years, paths), start=1):
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "PASSED"
            or int(manifest.get("date_count", -1)) != 142
            or int(manifest.get("minimum_half_hour_count", -1)) != 48
            or int(manifest.get("maximum_half_hour_count", -1)) != 48
        ):
            raise ValueError(f"year manifest failed contract: {manifest_path}")
        actual_sha = sha256_file(path)
        if manifest.get("sha256") != actual_sha:
            raise ValueError(f"year chunk SHA-256 mismatch: {path}")

        with xr.open_dataset(path) as source:
            if (
                source.attrs.get("product") != "GPM_3IMERGDF.07"
                or source.attrs.get("revision") != "V07B"
                or int(source.attrs.get("baseline_year", -1)) != year
            ):
                raise ValueError(f"provenance failed: {path}")
            dates = pd.DatetimeIndex(source.period_start.values)
            expected = pd.date_range(f"{year}-06-06", f"{year}-10-25", freq="D")
            if not dates.equals(expected) or not dates.is_unique:
                raise ValueError(f"date coverage failed: {path}")
            if int(source.precipitation_cnt.min()) != 48 or int(source.precipitation_cnt.max()) != 48:
                raise ValueError(f"half-hour count failed: {path}")
            values = source.precipitation.load().values.astype(np.float64)
            lat = source.latitude.values.astype(float)
            lon = source.longitude.values.astype(float)
        if values.shape != (142, 348, 333) or not np.isfinite(values).all():
            raise ValueError(f"field shape/finite check failed: {path}")
        current_keys = np.asarray([stamp.strftime("%m-%d") for stamp in dates], dtype="U5")
        if source_lat is None:
            source_lat, source_lon, month_days = lat, lon, current_keys
            total = np.zeros_like(values, dtype=np.float64)
        elif not (
            np.array_equal(source_lat, lat)
            and np.array_equal(source_lon, lon)
            and np.array_equal(month_days, current_keys)
        ):
            raise ValueError(f"grid/calendar key changed in {path}")
        total += values
        source_manifests.append(
            {"year": year, "path": str(path), "sha256": actual_sha}
        )
        print(f"validated and accumulated year {index}/22: {year}", flush=True)

    source_mean = total / 22.0
    support, support_lat, support_lon = land_support
    source_support = build.core.remap_support_fraction(
        support, support_lat, support_lon, source_lat, source_lon
    )
    target_mean, denominator, target_area, remap_checks = build.core.remap_conservative(
        source_mean,
        source_lat,
        source_lon,
        target_lat,
        target_lon,
        support=source_support,
    )
    if target_mean.shape != (142, len(target_lat), len(target_lon)):
        raise ValueError(f"unexpected remapped shape {target_mean.shape}")
    if not np.isfinite(target_mean[:, mask]).all() or float(np.min(target_mean[:, mask])) < 0:
        raise ValueError("remapped climatology is non-finite or negative on India support")
    representation = build.core.support_representation_audit(
        denominator, target_area, india_fraction
    )
    support_difference = float(
        np.nansum(np.abs(denominator[0] / target_area - india_fraction) * target_area)
        / np.nansum(india_fraction * target_area)
    )
    if support_difference > 0.03:
        raise ValueError(f"IMERG support differs from verification support by {support_difference:.4f}")

    dataset = xr.Dataset(
        data_vars={
            "daily_precipitation_climatology": (
                ("calendar_day", "latitude", "longitude"),
                target_mean.astype(np.float32),
                {
                    "units": "mm day-1",
                    "long_name": "IMERG Final fixed-calendar-day precipitation climatology",
                    "cell_methods": "baseline_year: mean",
                },
            ),
            "daily_sample_count": (
                ("calendar_day",), np.full(142, 22, dtype=np.int8)
            ),
            "india_fraction": (
                ("latitude", "longitude"), india_fraction.astype(np.float32)
            ),
            "spatial_weight": (
                ("latitude", "longitude"), weight.astype(np.float64)
            ),
        },
        coords={
            "calendar_day": np.arange(1, 143, dtype=np.int16),
            "calendar_month_day": (("calendar_day",), month_days),
            "latitude": target_lat,
            "longitude": target_lon,
        },
        attrs={
            "title": "IMERG Final V07B 2001-2022 fixed daily climatology for India S2S verification",
            "product": "GPM_3IMERGDF.07",
            "revision": "V07B",
            "doi": "10.5067/GPM/IMERGDF/DAY/07",
            "baseline_years": "2001-2022",
            "baseline_year_count": 22,
            "verification_years_excluded": "2023-2024",
            "calendar_window": "06-06 through 10-25 inclusive (142 days)",
            "grid": "22x22 at 1.5 degrees; frozen 169-cell India support",
            "remapping": "conservative spherical overlap to fixed IMD-derived India support",
            "support_difference_fraction": support_difference,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={
            "daily_precipitation_climatology": {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            }
        },
    )
    os.replace(temporary, OUTPUT)
    output_sha = sha256_file(OUTPUT)
    checks = {
        "twenty_two_year_chunks": len(source_manifests) == 22,
        "exactly_142_calendar_days": dataset.sizes["calendar_day"] == 142,
        "exactly_22_samples_per_day": bool((dataset.daily_sample_count.values == 22).all()),
        "verification_years_excluded": dataset.attrs["baseline_years"] == "2001-2022",
        "fixed_support_169_cells": int(mask.sum()) == 169,
        "finite_nonnegative_on_support": bool(
            np.isfinite(target_mean[:, mask]).all() and np.min(target_mean[:, mask]) >= 0
        ),
        "support_difference_below_3_percent": support_difference <= 0.03,
    }
    audit = {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(OUTPUT),
        "output_sha256": output_sha,
        "source_year_chunks": source_manifests,
        "imd_fixed_support_audit": imd_audit,
        "remap_checks": remap_checks,
        "support_representation": representation,
        "checks": checks,
    }
    manifest_path = OUTPUT.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(audit, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, default=json_default), flush=True)
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
