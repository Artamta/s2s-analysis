#!/usr/bin/env python3
"""Stage FuXi-S2S ERA5 daily inputs in bounded, resumable CDS requests."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.datastores import Client


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "clean/config/fuxi_operational_2020_2025.json"
CDS_URL = "https://cds.climate.copernicus.eu/api"
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PRESSURE_FIELDS = [
    ("z", "geopotential"),
    ("t", "temperature"),
    ("u", "u_component_of_wind"),
    ("v", "v_component_of_wind"),
    ("q", "specific_humidity"),
]
SURFACE_FIELDS = [
    ("t2m", "2m_temperature"),
    ("d2m", "2m_dewpoint_temperature"),
    ("sst", "sea_surface_temperature"),
    ("ttr", "top_net_thermal_radiation"),
    ("10u", "10m_u_component_of_wind"),
    ("10v", "10m_v_component_of_wind"),
    ("100u", "100m_u_component_of_wind"),
    ("100v", "100m_v_component_of_wind"),
    ("msl", "mean_sea_level_pressure"),
    ("tcwv", "total_column_water_vapour"),
    ("tp", "total_precipitation"),
]
FIELDS = {"pressure": PRESSURE_FIELDS, "surface": SURFACE_FIELDS}
ALIASES = {
    "10u": ("u10",),
    "10v": ("v10",),
    "100u": ("u100",),
    "100v": ("v100",),
}


@dataclass(frozen=True)
class RequestSpec:
    name: str
    component: str
    dataset: str
    days: tuple[pd.Timestamp, ...]
    request: dict[str, Any]
    digest: str
    cost: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def absolute_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def request_hash(dataset: str, request: dict[str, Any]) -> str:
    payload = canonical_json({"dataset": dataset, "request": request}).encode()
    return hashlib.sha256(payload).hexdigest()


def request_set_hash(specs: list[RequestSpec]) -> str:
    return hashlib.sha256(
        canonical_json([spec.digest for spec in specs]).encode()
    ).hexdigest()


def required_dates(config: dict[str, Any]) -> pd.DatetimeIndex:
    calendar = pd.read_csv(absolute_repo_path(config["calendar"]))
    init_dates = pd.DatetimeIndex(pd.to_datetime(calendar["init_date"]))
    allowed_years = {int(year) for year in config["years"]}
    init_dates = init_dates[init_dates.year.astype(int).isin(allowed_years)]
    if len(init_dates) != int(config["date_count"]):
        raise ValueError(
            f"expected {config['date_count']} initialization dates, found {len(init_dates)}"
        )
    dates = init_dates.append(init_dates - pd.Timedelta(days=1)).unique().sort_values()
    if len(dates) != len(init_dates) * 2:
        raise ValueError("previous-day and initialization-day inputs are not unique")
    return dates


def required_month_dates(
    config: dict[str, Any], year: int, month: int
) -> tuple[pd.Timestamp, ...]:
    dates = required_dates(config)
    selected = dates[(dates.year == year) & (dates.month == month)]
    if not len(selected):
        raise ValueError(f"calendar needs no FuXi input dates in {year}-{month:02d}")
    return tuple(pd.Timestamp(value) for value in selected)


def build_request(
    config: dict[str, Any],
    component: str,
    days: tuple[pd.Timestamp, ...],
) -> tuple[str, dict[str, Any], int]:
    if not days:
        raise ValueError("cannot build an empty CDS request")
    if len({(day.year, day.month) for day in days}) != 1:
        raise ValueError("each CDS request must contain dates from one month")
    input_config = config["input"]
    fields = FIELDS[component]
    dataset = (
        input_config["pressure_dataset"]
        if component == "pressure"
        else input_config["surface_dataset"]
    )
    request: dict[str, Any] = {
        "product_type": "reanalysis",
        "variable": [long_name for _, long_name in fields],
        "year": str(days[0].year),
        "month": f"{days[0].month:02d}",
        "day": [f"{day.day:02d}" for day in days],
        "daily_statistic": input_config["daily_statistic"],
        "time_zone": input_config["time_zone"],
        "frequency": input_config["hourly_sampling"],
        "grid": f"{input_config['grid_degrees']}/{input_config['grid_degrees']}",
        "area": [90, 0, -90, 358.5],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if component == "pressure":
        request["pressure_level"] = [str(level) for level in LEVELS]
        cost = len(days) * len(fields) * len(LEVELS)
    else:
        cost = len(days) * len(fields)
    limit = int(input_config["request_cost_limit"])
    if cost > limit:
        raise ValueError(f"{component} request cost {cost} exceeds CDS limit {limit}")
    return dataset, request, cost


def build_requests(config: dict[str, Any], year: int, month: int) -> list[RequestSpec]:
    days = required_month_dates(config, year, month)
    batch_size = int(config["input"]["pressure_days_per_request"])
    specs: list[RequestSpec] = []
    for index, start in enumerate(range(0, len(days), batch_size)):
        selected = days[start : start + batch_size]
        dataset, request, cost = build_request(config, "pressure", selected)
        specs.append(
            RequestSpec(
                name=f"pressure_{index:02d}",
                component="pressure",
                dataset=dataset,
                days=selected,
                request=request,
                digest=request_hash(dataset, request),
                cost=cost,
            )
        )
    dataset, request, cost = build_request(config, "surface", days)
    specs.append(
        RequestSpec(
            name="surface_00",
            component="surface",
            dataset=dataset,
            days=days,
            request=request,
            digest=request_hash(dataset, request),
            cost=cost,
        )
    )
    return specs


def monthly_root(config: dict[str, Any], year: int, month: int) -> Path:
    return (
        Path(config["storage_root"])
        / config["input"]["staging_subdirectory"]
        / "monthly"
        / f"{year}{month:02d}"
    )


def paths_for(
    config: dict[str, Any], year: int, month: int, spec: RequestSpec
) -> dict[str, Path]:
    root = monthly_root(config, year, month)
    request_root = (
        Path(config["storage_root"])
        / config["input"]["staging_subdirectory"]
        / "requests"
        / f"{year}{month:02d}"
    )
    return {
        "output": root / f"{spec.component}.nc",
        "shard": root / ".shards" / f"{spec.name}.nc",
        "download": root / ".downloads" / f"{spec.name}.download",
        "extract": root / ".downloads" / f"{spec.name}.extract",
        "state": request_root / f"{spec.name}.json",
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cds_key() -> str:
    if os.environ.get("CDSAPI_KEY"):
        return os.environ["CDSAPI_KEY"]
    rc_path = Path.home() / ".cdsapirc"
    if not rc_path.is_file():
        raise FileNotFoundError(f"missing CDS credential: {rc_path}")
    for line in rc_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("key:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"no key entry in {rc_path}")


def client() -> Client:
    return Client(
        url=os.environ.get("CDS_ERA5_URL", CDS_URL),
        key=cds_key(),
        cleanup=False,
        progress=True,
        maximum_tries=500,
        retry_after=120,
        sleep_max=120,
    )


def canonicalize_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    rename = {}
    for source, target in (
        ("valid_time", "time"),
        ("date", "time"),
        ("pressure_level", "level"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ):
        if source in dataset.dims or source in dataset.coords:
            if target not in dataset.dims and target not in dataset.coords:
                rename[source] = target
    dataset = dataset.rename(rename)
    if "lon" in dataset.coords:
        dataset = dataset.assign_coords(lon=np.mod(dataset.lon, 360.0)).sortby("lon")
    if "lat" in dataset.coords and dataset.lat.values[0] < dataset.lat.values[-1]:
        dataset = dataset.sortby("lat", ascending=False)
    return dataset


def collapse_version_dimensions(data: xr.DataArray) -> xr.DataArray:
    for dimension in ("number", "expver"):
        if dimension not in data.dims:
            continue
        if data.sizes[dimension] == 1:
            data = data.isel({dimension: 0}, drop=True)
            continue
        combined = data.isel({dimension: 0}, drop=True)
        for index in range(1, data.sizes[dimension]):
            combined = combined.combine_first(data.isel({dimension: index}, drop=True))
        data = combined
    return data


def drop_singleton_dimensions(data: xr.DataArray, keep: set[str]) -> xr.DataArray:
    for dimension in tuple(data.dims):
        if dimension not in keep and data.sizes[dimension] == 1:
            data = data.isel({dimension: 0}, drop=True)
    return data


def find_variable(dataset: xr.Dataset, short_name: str, long_name: str) -> xr.DataArray:
    for candidate in (short_name, long_name, *ALIASES.get(short_name, ())):
        if candidate in dataset.data_vars:
            return dataset[candidate]
    raise KeyError(
        f"missing {short_name}/{long_name}; available variables: "
        f"{sorted(dataset.data_vars)}"
    )


def open_download(path: Path, extract_dir: Path) -> tuple[xr.Dataset, list[xr.Dataset]]:
    opened: list[xr.Dataset] = []
    if zipfile.is_zipfile(path):
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extract_dir)
        files = sorted(extract_dir.rglob("*.nc"))
        if not files:
            raise ValueError(f"CDS ZIP has no NetCDF files: {path}")
        opened = [canonicalize_coordinates(xr.open_dataset(file)) for file in files]
        return xr.merge(opened, compat="override", join="outer"), opened
    source = canonicalize_coordinates(xr.open_dataset(path))
    opened.append(source)
    return source, opened


def expected_variables(component: str) -> set[str]:
    return {short_name for short_name, _ in FIELDS[component]}


def normalize_download(spec: RequestSpec, paths: dict[str, Path]) -> None:
    source, opened = open_download(paths["download"], paths["extract"])
    try:
        variables: dict[str, xr.DataArray] = {}
        expected_time = pd.DatetimeIndex(spec.days)
        for short_name, long_name in FIELDS[spec.component]:
            data = collapse_version_dimensions(
                find_variable(source, short_name, long_name)
            )
            data = drop_singleton_dimensions(
                data,
                (
                    {"time", "level", "lat", "lon"}
                    if spec.component == "pressure"
                    else {"time", "lat", "lon"}
                ),
            )
            normalized_time = pd.DatetimeIndex(data.time.values).normalize()
            if normalized_time.duplicated().any():
                raise ValueError(f"{short_name} contains duplicate daily timestamps")
            data = data.assign_coords(time=normalized_time.values)
            missing = expected_time.difference(normalized_time)
            if len(missing):
                raise ValueError(
                    f"{short_name} is missing requested days {missing.tolist()}"
                )
            data = data.sel(time=expected_time.values)
            if spec.component == "pressure":
                data = data.sel(level=LEVELS).transpose("time", "level", "lat", "lon")
            else:
                data = data.transpose("time", "lat", "lon")
            attrs = dict(data.attrs)
            variables[short_name] = data.astype(np.float32).assign_attrs(attrs)

        normalized = xr.Dataset(variables).assign_attrs(
            source="CDS ERA5 daily statistics derived from 1-hourly reanalysis",
            daily_statistic="daily_mean",
            hourly_sampling="1_hourly",
            time_zone="utc+00:00",
            request_sha256=spec.digest,
            request_name=spec.name,
            request_cost=spec.cost,
            component=spec.component,
        )
        encoding = {}
        for name in normalized.data_vars:
            chunks = (
                (1, len(LEVELS), 121, 240)
                if spec.component == "pressure"
                else (1, 121, 240)
            )
            encoding[name] = {
                "zlib": True,
                "complevel": 2,
                "dtype": "float32",
                "chunksizes": chunks,
            }
        paths["shard"].parent.mkdir(parents=True, exist_ok=True)
        temporary = paths["shard"].with_suffix(".nc.part")
        temporary.unlink(missing_ok=True)
        normalized.to_netcdf(temporary, encoding=encoding)
        validate_shard(temporary, spec)
        temporary.replace(paths["shard"])
    finally:
        for dataset in opened:
            dataset.close()


def validate_grid(dataset: xr.Dataset, component: str) -> None:
    expected_sizes = {"lat": 121, "lon": 240}
    if component == "pressure":
        expected_sizes["level"] = len(LEVELS)
    for dimension, size in expected_sizes.items():
        if dataset.sizes.get(dimension) != size:
            raise ValueError(
                f"{component} {dimension}: expected {size}, found {dataset.sizes.get(dimension)}"
            )
    if not np.allclose(dataset.lat.values, np.linspace(90, -90, 121)):
        raise ValueError(f"{component} has the wrong latitude grid")
    if not np.allclose(dataset.lon.values, np.arange(0, 360, 1.5)):
        raise ValueError(f"{component} has the wrong longitude grid")
    if component == "pressure" and list(dataset.level.values) != LEVELS:
        raise ValueError(
            f"pressure levels are in the wrong order: {dataset.level.values}"
        )


def validate_values(dataset: xr.Dataset, component: str) -> None:
    indices = sorted({0, dataset.sizes["time"] // 2, dataset.sizes["time"] - 1})
    for name, data in dataset.data_vars.items():
        sample = data.isel(time=indices).values
        if name == "sst":
            if not np.isfinite(sample).any():
                raise ValueError("SST validation sample has no finite ocean values")
        elif not np.isfinite(sample).all():
            raise ValueError(
                f"{component}/{name} validation sample contains missing values"
            )


def validate_shard(path: Path, spec: RequestSpec) -> dict[str, Any]:
    dataset = xr.open_dataset(path)
    try:
        if set(dataset.data_vars) != expected_variables(spec.component):
            raise ValueError(
                f"{spec.name} variables: expected {sorted(expected_variables(spec.component))}, "
                f"found {sorted(dataset.data_vars)}"
            )
        actual_time = pd.DatetimeIndex(dataset.time.values).normalize()
        if not actual_time.equals(pd.DatetimeIndex(spec.days)):
            raise ValueError(f"{spec.name} does not contain its exact requested dates")
        validate_grid(dataset, spec.component)
        validate_values(dataset, spec.component)
        if dataset.attrs.get("request_sha256") != spec.digest:
            raise ValueError(f"{spec.name} request hash mismatch")
        return {"days": len(actual_time), "size_bytes": path.stat().st_size}
    finally:
        dataset.close()


def validate_monthly(
    path: Path,
    component: str,
    days: tuple[pd.Timestamp, ...],
    digest: str,
) -> dict[str, Any]:
    dataset = xr.open_dataset(path)
    try:
        if set(dataset.data_vars) != expected_variables(component):
            raise ValueError(
                f"{component} variables: expected {sorted(expected_variables(component))}, "
                f"found {sorted(dataset.data_vars)}"
            )
        actual_time = pd.DatetimeIndex(dataset.time.values).normalize()
        if not actual_time.equals(pd.DatetimeIndex(days)):
            raise ValueError(f"{component} monthly file does not match required dates")
        validate_grid(dataset, component)
        validate_values(dataset, component)
        if dataset.attrs.get("request_set_sha256") != digest:
            raise ValueError(f"{component} monthly request-set hash mismatch")
        if dataset.attrs.get("daily_statistic") != "daily_mean":
            raise ValueError(f"{component} monthly file is not marked as daily mean")
        return {"days": len(actual_time), "size_bytes": path.stat().st_size}
    finally:
        dataset.close()


def quarantine(path: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.invalid.{timestamp}")
    path.replace(destination)
    return destination


def ensure_request(
    api: Client,
    spec: RequestSpec,
    paths: dict[str, Path],
    poll_seconds: int,
    max_wait_seconds: int,
) -> None:
    if paths["shard"].is_file():
        try:
            details = validate_shard(paths["shard"], spec)
        except Exception as exc:  # noqa: BLE001
            moved = quarantine(paths["shard"])
            print(f"quarantined invalid shard at {moved}: {exc}", flush=True)
        else:
            print(f"existing request shard valid: {spec.name} {details}", flush=True)
            return

    state = read_state(paths["state"])
    if state and state.get("request_sha256") != spec.digest:
        raise ValueError(
            f"request state hash mismatch in {paths['state']}; refusing a duplicate"
        )
    if state and state.get("request_id"):
        remote = api.get_remote(state["request_id"])
        print(f"resuming CDS request {remote.request_id} for {spec.name}", flush=True)
    else:
        remote = api.submit(spec.dataset, spec.request)
        state = {
            "schema_version": 1,
            "name": spec.name,
            "component": spec.component,
            "dataset": spec.dataset,
            "days": [day.strftime("%Y-%m-%d") for day in spec.days],
            "request": spec.request,
            "request_cost": spec.cost,
            "request_sha256": spec.digest,
            "request_id": remote.request_id,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_state(paths["state"], state)
        print(f"submitted CDS request {remote.request_id} for {spec.name}", flush=True)

    started = time.monotonic()
    while True:
        status = remote.status
        state.update(
            status=status,
            checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        write_state(paths["state"], state)
        if status == "successful":
            break
        if status not in {"accepted", "running"}:
            raise RuntimeError(
                f"CDS request {remote.request_id} ended with status {status}"
            )
        if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
            raise TimeoutError(
                f"CDS request remains {status}; rerun to resume {remote.request_id}"
            )
        time.sleep(poll_seconds)

    paths["download"].parent.mkdir(parents=True, exist_ok=True)
    paths["download"].unlink(missing_ok=True)
    remote.download(str(paths["download"]))
    normalize_download(spec, paths)
    details = validate_shard(paths["shard"], spec)
    state.update(
        status="normalized_valid",
        shard=str(paths["shard"]),
        shard_size_bytes=details["size_bytes"],
        completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    write_state(paths["state"], state)
    paths["download"].unlink(missing_ok=True)
    if paths["extract"].exists():
        shutil.rmtree(paths["extract"])
    print(f"normalized valid CDS request: {spec.name} {details}", flush=True)


def combine_component(
    config: dict[str, Any],
    year: int,
    month: int,
    component: str,
    specs: list[RequestSpec],
) -> dict[str, Any]:
    days = tuple(day for spec in specs for day in spec.days)
    days = tuple(sorted(set(days)))
    digest = request_set_hash(specs)
    output = paths_for(config, year, month, specs[0])["output"]
    opened = [
        xr.open_dataset(paths_for(config, year, month, spec)["shard"]) for spec in specs
    ]
    try:
        combined = xr.concat(
            opened,
            dim="time",
            data_vars="all",
            coords="minimal",
            compat="override",
            join="exact",
        ).sortby("time")
        combined = combined.assign_attrs(
            source="CDS ERA5 daily statistics derived from 1-hourly reanalysis",
            daily_statistic="daily_mean",
            hourly_sampling="1_hourly",
            time_zone="utc+00:00",
            component=component,
            year=year,
            month=month,
            request_count=len(specs),
            request_set_sha256=digest,
            request_state_files=",".join(
                str(paths_for(config, year, month, spec)["state"]) for spec in specs
            ),
        )
        encoding = {}
        for name in combined.data_vars:
            chunks = (
                (1, len(LEVELS), 121, 240) if component == "pressure" else (1, 121, 240)
            )
            encoding[name] = {
                "zlib": True,
                "complevel": 2,
                "dtype": "float32",
                "chunksizes": chunks,
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".nc.part")
        temporary.unlink(missing_ok=True)
        combined.to_netcdf(temporary, encoding=encoding)
        validate_monthly(temporary, component, days, digest)
        temporary.replace(output)
    finally:
        for dataset in opened:
            dataset.close()
    details = validate_monthly(output, component, days, digest)
    for spec in specs:
        paths_for(config, year, month, spec)["shard"].unlink(missing_ok=True)
    for directory in (output.parent / ".shards", output.parent / ".downloads"):
        if directory.exists():
            shutil.rmtree(directory)
    return details


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    allowed_years = {int(year) for year in config["years"]}
    if args.year not in allowed_years:
        raise ValueError(f"year must be one of {sorted(allowed_years)}")
    if not 1 <= args.month <= 12:
        raise ValueError("month must be in 1..12")
    specs = build_requests(config, args.year, args.month)
    days = required_month_dates(config, args.year, args.month)
    summary = {
        "year": args.year,
        "month": args.month,
        "required_days": [day.strftime("%Y-%m-%d") for day in days],
        "request_count": len(specs),
        "requests": [
            {
                "name": spec.name,
                "component": spec.component,
                "days": [day.strftime("%Y-%m-%d") for day in spec.days],
                "cost": spec.cost,
                "limit": config["input"]["request_cost_limit"],
                "sha256": spec.digest,
            }
            for spec in specs
        ],
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return 0

    pending: dict[str, list[RequestSpec]] = {}
    for component in FIELDS:
        component_specs = [spec for spec in specs if spec.component == component]
        digest = request_set_hash(component_specs)
        output = paths_for(config, args.year, args.month, component_specs[0])["output"]
        if output.is_file():
            try:
                details = validate_monthly(output, component, days, digest)
            except Exception as exc:  # noqa: BLE001
                moved = quarantine(output)
                print(f"quarantined invalid monthly file at {moved}: {exc}", flush=True)
            else:
                print(f"existing monthly staging valid: {output} {details}", flush=True)
                continue
        pending[component] = component_specs

    if not pending:
        return 0
    api = client()
    for component, component_specs in pending.items():
        for spec in component_specs:
            ensure_request(
                api,
                spec,
                paths_for(config, args.year, args.month, spec),
                args.poll_seconds,
                args.max_wait_seconds,
            )
        details = combine_component(
            config, args.year, args.month, component, component_specs
        )
        output = paths_for(config, args.year, args.month, component_specs[0])["output"]
        print(
            f"staged valid monthly ERA5 daily statistics: {output} {details}",
            flush=True,
        )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
