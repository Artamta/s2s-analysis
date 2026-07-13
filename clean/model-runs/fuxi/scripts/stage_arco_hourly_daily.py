#!/usr/bin/env python3
"""Stage 2023-2025 FuXi daily inputs directly from public hourly ARCO ERA5."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

import arco_hourly
import stage_era5_daily as stage


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "clean/config/fuxi_operational_2020_2025.json"
FIELDS = {
    "pressure": arco_hourly.PRESSURE_FIELDS,
    "surface": arco_hourly.SURFACE_FIELDS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def remote_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["input"]["remote_hourly_arco"]


def validate_year(config: dict[str, Any], year: int) -> None:
    years = {int(value) for value in remote_config(config)["years"]}
    if year not in years:
        raise ValueError(f"remote ARCO year must be one of {sorted(years)}, found {year}")


def paths_for(
    config: dict[str, Any], year: int, month: int, component: str
) -> dict[str, Path]:
    root = (
        Path(config["storage_root"])
        / config["input"]["staging_subdirectory"]
        / "monthly"
        / f"{year}{month:02d}"
    )
    return {
        "output": root / f"{component}.nc",
        "shards": root / ".arco" / component,
    }


def field_digests(
    url: str,
    component: str,
    days: tuple[pd.Timestamp, ...],
) -> dict[str, str]:
    return {
        short_name: arco_hourly.field_contract_hash(
            url,
            short_name,
            source_name,
            days,
            component == "pressure",
        )
        for short_name, source_name in FIELDS[component].items()
    }


def component_digest(component: str, digests: dict[str, str]) -> str:
    return arco_hourly.canonical_hash(
        {"component": component, "field_contracts": digests}
    )


def netcdf_encoding(dataset: xr.Dataset, pressure: bool) -> dict[str, dict[str, Any]]:
    chunks = (
        (1, len(arco_hourly.LEVELS), 121, 240)
        if pressure
        else (1, 121, 240)
    )
    return {
        name: {
            "zlib": True,
            "complevel": 2,
            "dtype": "float32",
            "chunksizes": chunks,
        }
        for name in dataset.data_vars
    }


def combine_component(
    config: dict[str, Any],
    year: int,
    month: int,
    component: str,
    days: tuple[pd.Timestamp, ...],
    digests: dict[str, str],
) -> dict[str, Any]:
    paths = paths_for(config, year, month, component)
    digest = component_digest(component, digests)
    opened = [
        xr.open_dataset(paths["shards"] / f"{name}.nc")
        for name in FIELDS[component]
    ]
    try:
        combined = xr.merge(opened, compat="no_conflicts", join="exact")
        combined = combined.assign_attrs(
            source="ARCO-ERA5 public hourly Zarr on Google Cloud",
            source_path=remote_config(config)["zarr"],
            daily_statistic="daily_mean",
            hourly_sampling="1_hourly",
            time_zone="utc+00:00",
            component=component,
            request_set_sha256=digest,
            field_contract_sha256=json.dumps(digests, sort_keys=True),
        )
        paths["output"].parent.mkdir(parents=True, exist_ok=True)
        temporary = paths["output"].with_suffix(".nc.part")
        temporary.unlink(missing_ok=True)
        combined.to_netcdf(
            temporary,
            encoding=netcdf_encoding(combined, component == "pressure"),
        )
        stage.validate_monthly(temporary, component, days, digest)
        temporary.replace(paths["output"])
    finally:
        for dataset in opened:
            dataset.close()
    details = stage.validate_monthly(paths["output"], component, days, digest)
    if paths["shards"].exists():
        shutil.rmtree(paths["shards"])
    arco_root = paths["shards"].parent
    if arco_root.exists() and not any(arco_root.iterdir()):
        arco_root.rmdir()
    return details


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_year(config, args.year)
    if not 1 <= args.month <= 12:
        raise ValueError("month must be in 1..12")
    days = stage.required_month_dates(config, args.year, args.month)
    remote = remote_config(config)
    url = str(remote["zarr"])
    workers = int(args.workers or remote["workers"])
    summary = {
        "year": args.year,
        "month": args.month,
        "required_days": [day.strftime("%Y-%m-%d") for day in days],
        "source": url,
        "workers": workers,
        "pressure_hourly_chunks": len(days) * 24 * len(FIELDS["pressure"]),
        "surface_hourly_chunks": len(days) * 24 * len(FIELDS["surface"]),
        "cds_requests": 0,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return 0

    pending: dict[str, dict[str, str]] = {}
    for component in FIELDS:
        digests = field_digests(url, component, days)
        paths = paths_for(config, args.year, args.month, component)
        if paths["output"].is_file():
            try:
                details = stage.validate_monthly(
                    paths["output"],
                    component,
                    days,
                    component_digest(component, digests),
                )
            except Exception as exc:  # noqa: BLE001
                moved = stage.quarantine(paths["output"])
                print(f"quarantined incompatible {component} at {moved}: {exc}", flush=True)
            else:
                print(f"existing ARCO {component} valid: {details}", flush=True)
                continue
        pending[component] = digests
    if not pending:
        return 0

    dataset = arco_hourly.open_arco(url)
    try:
        arco_hourly.validate_availability(dataset, days)
        print(
            "ARCO coverage: "
            f"{dataset.attrs['valid_time_start']} through {dataset.attrs['valid_time_stop']}",
            flush=True,
        )
        for component, digests in pending.items():
            paths = paths_for(config, args.year, args.month, component)
            for short_name, source_name in FIELDS[component].items():
                _, details = arco_hourly.ensure_field_shard(
                    dataset,
                    url,
                    short_name,
                    source_name,
                    days,
                    paths["shards"] / f"{short_name}.nc",
                    pressure=component == "pressure",
                    workers=workers,
                )
                print(f"ARCO field valid {short_name}: {details}", flush=True)
            details = combine_component(
                config, args.year, args.month, component, days, digests
            )
            print(f"ARCO monthly {component} valid: {details}", flush=True)
    finally:
        dataset.close()
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
