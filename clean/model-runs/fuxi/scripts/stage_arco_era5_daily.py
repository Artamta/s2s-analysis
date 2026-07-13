#!/usr/bin/env python3
"""Stage 2020-2022 FuXi inputs from local daily and public hourly ARCO."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import get_codec

import arco_hourly
import stage_era5_daily as stage


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "clean/config/fuxi_operational_2020_2025.json"
LOCAL_PRESSURE_FIELDS = arco_hourly.PRESSURE_FIELDS
LOCAL_SURFACE_FIELDS = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sst": "sea_surface_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "tcwv": "total_column_water_vapour",
    "tp": "total_precipitation_24hr",
}
REMOTE_GAP_FIELDS = {
    name: arco_hourly.SURFACE_FIELDS[name] for name in ("ttr", "100u", "100v")
}


class ConsolidatedZarrReader:
    """Read one-full-grid-per-day Zarr v2 chunks without indexing overhead."""

    def __init__(self, root: Path) -> None:
        self.root = root
        metadata_path = root / ".zmetadata"
        self.metadata_bytes = metadata_path.read_bytes()
        self.metadata = json.loads(self.metadata_bytes)["metadata"]
        self.metadata_sha256 = hashlib.sha256(self.metadata_bytes).hexdigest()

    def array_metadata(self, name: str) -> dict[str, Any]:
        try:
            return self.metadata[f"{name}/.zarray"]
        except KeyError as exc:
            raise KeyError(f"{self.root} has no Zarr array {name!r}") from exc

    def array_attrs(self, name: str) -> dict[str, Any]:
        return dict(self.metadata.get(f"{name}/.zattrs", {}))

    def decode_chunk(self, name: str, key: str) -> np.ndarray:
        metadata = self.array_metadata(name)
        if metadata.get("filters"):
            raise ValueError(f"unsupported Zarr filters for {name}: {metadata['filters']}")
        decoded: Any = (self.root / name / key).read_bytes()
        if metadata.get("compressor"):
            decoded = get_codec(metadata["compressor"]).decode(decoded)
        shape = tuple(
            min(chunk, size)
            for chunk, size in zip(metadata["chunks"], metadata["shape"], strict=True)
        )
        values = np.frombuffer(decoded, dtype=np.dtype(metadata["dtype"]))
        if values.size != int(np.prod(shape)):
            raise ValueError(f"{name}/{key} decoded to an unexpected size")
        return values.reshape(shape, order=metadata.get("order", "C"))

    def coordinate(self, name: str) -> np.ndarray:
        metadata = self.array_metadata(name)
        if tuple(metadata["chunks"]) != tuple(metadata["shape"]):
            raise ValueError(f"coordinate {name} is unexpectedly split across chunks")
        return self.decode_chunk(name, "0")


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


def local_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["input"]["local_daily_arco"]


def remote_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["input"]["remote_hourly_arco"]


def validate_year(config: dict[str, Any], year: int) -> None:
    years = {int(value) for value in local_config(config)["years"]}
    if year not in years:
        raise ValueError(f"local ARCO year must be one of {sorted(years)}, found {year}")


def roots(config: dict[str, Any], year: int, month: int) -> dict[str, Path]:
    staging_root = Path(config["storage_root"]) / config["input"]["staging_subdirectory"]
    monthly = staging_root / "monthly" / f"{year}{month:02d}"
    return {
        "monthly": monthly,
        "pressure": monthly / "pressure.nc",
        "surface": monthly / "surface.nc",
        "surface_local": monthly / ".arco" / "surface_local.nc",
        "gap_root": monthly / ".arco" / "remote_gap",
    }


def source_time_index(reader: ConsolidatedZarrReader) -> dict[pd.Timestamp, int]:
    values = reader.coordinate("time").astype(np.int64)
    units = str(reader.array_attrs("time").get("units", ""))
    prefix = "days since "
    if not units.startswith(prefix):
        raise ValueError(f"unsupported local ARCO time units: {units!r}")
    origin = pd.Timestamp(units[len(prefix) :])
    times = pd.DatetimeIndex(origin + pd.to_timedelta(values, unit="D")).normalize()
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("local ARCO time coordinate must be unique and increasing")
    return {pd.Timestamp(value): index for index, value in enumerate(times)}


def validate_source_grid(
    reader: ConsolidatedZarrReader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitude = reader.coordinate("latitude").astype(np.float32)
    longitude = reader.coordinate("longitude").astype(np.float32)
    levels = reader.coordinate("level").astype(np.int64)
    if not np.allclose(latitude, np.linspace(90, -90, 721)):
        raise ValueError("local ARCO latitude grid is not 0.25 degree north-to-south")
    if not np.allclose(longitude, np.arange(0, 360, 0.25)):
        raise ValueError("local ARCO longitude grid is not 0.25 degree 0-to-360")
    missing = sorted(set(stage.LEVELS).difference(levels.tolist()))
    if missing:
        raise ValueError(f"local ARCO archive is missing levels {missing}")
    level_indices = np.asarray(
        [int(np.flatnonzero(levels == level)[0]) for level in stage.LEVELS]
    )
    return latitude[::6], longitude[::6], level_indices


def source_digest(
    reader: ConsolidatedZarrReader,
    component: str,
    days: tuple[pd.Timestamp, ...],
) -> str:
    fields = (
        LOCAL_PRESSURE_FIELDS if component == "pressure" else LOCAL_SURFACE_FIELDS
    )
    return arco_hourly.canonical_hash(
        {
            "schema_version": 1,
            "source_metadata_sha256": reader.metadata_sha256,
            "component": component,
            "fields": fields,
            "days": [day.strftime("%Y-%m-%d") for day in days],
            "levels": stage.LEVELS if component == "pressure" else None,
            "grid": "point-sampled 0.25 degree to 1.5 degree",
            "tp_conversion": "24-hour accumulation divided by 24",
        }
    )


def netcdf_encoding(dataset: xr.Dataset, pressure: bool) -> dict[str, dict[str, Any]]:
    chunks = (1, len(stage.LEVELS), 121, 240) if pressure else (1, 121, 240)
    return {
        name: {
            "zlib": True,
            "complevel": 2,
            "dtype": "float32",
            "chunksizes": chunks,
        }
        for name in dataset.data_vars
    }


def write_atomic(dataset: xr.Dataset, path: Path, pressure: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary, encoding=netcdf_encoding(dataset, pressure))
    temporary.replace(path)


def build_local_pressure(
    reader: ConsolidatedZarrReader,
    days: tuple[pd.Timestamp, ...],
    day_indices: dict[pd.Timestamp, int],
    lat: np.ndarray,
    lon: np.ndarray,
    level_indices: np.ndarray,
    digest: str,
) -> xr.Dataset:
    shape = (len(days), len(stage.LEVELS), len(lat), len(lon))
    variables: dict[str, xr.DataArray] = {}
    for short_name, source_name in LOCAL_PRESSURE_FIELDS.items():
        output = np.empty(shape, dtype=np.float32)
        for output_index, day in enumerate(days):
            chunk = reader.decode_chunk(source_name, f"{day_indices[day]}.0.0.0")
            output[output_index] = chunk[0, level_indices][:, ::6, ::6]
        attrs = reader.array_attrs(source_name)
        attrs.pop("_ARRAY_DIMENSIONS", None)
        variables[short_name] = xr.DataArray(
            output, dims=("time", "level", "lat", "lon"), attrs=attrs
        )
    return xr.Dataset(
        variables,
        coords={
            "time": pd.DatetimeIndex(days),
            "level": stage.LEVELS,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "source": "local WeatherBench2 daily ARCO ERA5 derived from 1-hourly data",
            "daily_statistic": "daily_mean",
            "hourly_sampling": "1_hourly",
            "time_zone": "utc+00:00",
            "component": "pressure",
            "request_set_sha256": digest,
            "source_metadata_sha256": reader.metadata_sha256,
            "spatial_sampling": "0.25 degree grid point-sampled every 6 cells",
        },
    )


def build_local_surface(
    reader: ConsolidatedZarrReader,
    days: tuple[pd.Timestamp, ...],
    day_indices: dict[pd.Timestamp, int],
    lat: np.ndarray,
    lon: np.ndarray,
    digest: str,
) -> xr.Dataset:
    shape = (len(days), len(lat), len(lon))
    variables: dict[str, xr.DataArray] = {}
    for short_name, source_name in LOCAL_SURFACE_FIELDS.items():
        output = np.empty(shape, dtype=np.float32)
        for output_index, day in enumerate(days):
            chunk = reader.decode_chunk(source_name, f"{day_indices[day]}.0.0")
            output[output_index] = chunk[0, ::6, ::6]
        attrs = reader.array_attrs(source_name)
        attrs.pop("_ARRAY_DIMENSIONS", None)
        if short_name == "tp":
            output /= np.float32(24.0)
            attrs["processing"] = "24-hour accumulation divided by 24"
        variables[short_name] = xr.DataArray(
            output, dims=("time", "lat", "lon"), attrs=attrs
        )
    return xr.Dataset(
        variables,
        coords={"time": pd.DatetimeIndex(days), "lat": lat, "lon": lon},
        attrs={
            "source": "local WeatherBench2 daily ARCO ERA5 derived from 1-hourly data",
            "daily_statistic": "daily_mean",
            "hourly_sampling": "1_hourly",
            "time_zone": "utc+00:00",
            "component": "surface_local",
            "request_set_sha256": digest,
            "source_metadata_sha256": reader.metadata_sha256,
            "spatial_sampling": "0.25 degree grid point-sampled every 6 cells",
        },
    )


def validate_local_surface(
    path: Path, days: tuple[pd.Timestamp, ...], digest: str
) -> dict[str, Any]:
    dataset = xr.open_dataset(path)
    try:
        if set(dataset.data_vars) != set(LOCAL_SURFACE_FIELDS):
            raise ValueError("local surface variables do not match the ARCO contract")
        if not pd.DatetimeIndex(dataset.time.values).normalize().equals(
            pd.DatetimeIndex(days)
        ):
            raise ValueError("local surface file does not contain the required days")
        stage.validate_grid(dataset, "surface")
        stage.validate_values(dataset, "surface")
        if dataset.attrs.get("request_set_sha256") != digest:
            raise ValueError("local surface provenance hash mismatch")
        return {"days": len(days), "size_bytes": path.stat().st_size}
    finally:
        dataset.close()


def ensure_local_pressure(
    reader: ConsolidatedZarrReader,
    days: tuple[pd.Timestamp, ...],
    paths: dict[str, Path],
    digest: str,
) -> dict[str, Any]:
    if paths["pressure"].is_file():
        try:
            return stage.validate_monthly(paths["pressure"], "pressure", days, digest)
        except Exception as exc:  # noqa: BLE001
            moved = stage.quarantine(paths["pressure"])
            print(f"quarantined incompatible pressure at {moved}: {exc}", flush=True)
    time_index = source_time_index(reader)
    missing = [day for day in days if day not in time_index]
    if missing:
        raise ValueError(f"local daily ARCO is missing {missing}")
    lat, lon, level_indices = validate_source_grid(reader)
    pressure = build_local_pressure(
        reader, days, time_index, lat, lon, level_indices, digest
    )
    write_atomic(pressure, paths["pressure"], pressure=True)
    return stage.validate_monthly(paths["pressure"], "pressure", days, digest)


def ensure_local_surface(
    reader: ConsolidatedZarrReader,
    days: tuple[pd.Timestamp, ...],
    paths: dict[str, Path],
    digest: str,
) -> dict[str, Any]:
    if paths["surface_local"].is_file():
        try:
            return validate_local_surface(paths["surface_local"], days, digest)
        except Exception as exc:  # noqa: BLE001
            moved = stage.quarantine(paths["surface_local"])
            print(f"quarantined incompatible surface base at {moved}: {exc}", flush=True)
    time_index = source_time_index(reader)
    lat, lon, _ = validate_source_grid(reader)
    surface = build_local_surface(reader, days, time_index, lat, lon, digest)
    write_atomic(surface, paths["surface_local"], pressure=False)
    return validate_local_surface(paths["surface_local"], days, digest)


def gap_digests(
    url: str, days: tuple[pd.Timestamp, ...]
) -> dict[str, str]:
    return {
        short_name: arco_hourly.field_contract_hash(
            url, short_name, source_name, days, False
        )
        for short_name, source_name in REMOTE_GAP_FIELDS.items()
    }


def combined_surface_digest(local_digest: str, remote_digests: dict[str, str]) -> str:
    return arco_hourly.canonical_hash(
        {"local_surface": local_digest, "remote_fields": remote_digests}
    )


def combine_surface(
    paths: dict[str, Path],
    days: tuple[pd.Timestamp, ...],
    local_digest: str,
    remote_digests: dict[str, str],
) -> dict[str, Any]:
    digest = combined_surface_digest(local_digest, remote_digests)
    local = xr.open_dataset(paths["surface_local"])
    opened = [xr.open_dataset(paths["gap_root"] / f"{name}.nc") for name in REMOTE_GAP_FIELDS]
    try:
        combined = xr.merge([local, *opened], compat="no_conflicts", join="exact")
        combined = combined.assign_attrs(
            source=(
                "local WeatherBench2 daily ARCO plus direct public hourly ARCO "
                "for ttr/100u/100v"
            ),
            daily_statistic="daily_mean",
            hourly_sampling="1_hourly",
            time_zone="utc+00:00",
            component="surface",
            request_set_sha256=digest,
            local_source_sha256=local_digest,
            remote_field_sha256=json.dumps(remote_digests, sort_keys=True),
        )
        write_atomic(combined, paths["surface"], pressure=False)
    finally:
        local.close()
        for dataset in opened:
            dataset.close()
    details = stage.validate_monthly(paths["surface"], "surface", days, digest)
    paths["surface_local"].unlink(missing_ok=True)
    if paths["gap_root"].exists():
        shutil.rmtree(paths["gap_root"])
    arco_root = paths["surface_local"].parent
    if arco_root.exists() and not any(arco_root.iterdir()):
        arco_root.rmdir()
    return details


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_year(config, args.year)
    if not 1 <= args.month <= 12:
        raise ValueError("month must be in 1..12")
    days = stage.required_month_dates(config, args.year, args.month)
    local_source = Path(local_config(config)["zarr"])
    remote = remote_config(config)
    remote_url = str(remote["zarr"])
    workers = int(args.workers or remote["workers"])
    paths = roots(config, args.year, args.month)
    summary = {
        "year": args.year,
        "month": args.month,
        "required_days": [day.strftime("%Y-%m-%d") for day in days],
        "local_daily_arco": str(local_source),
        "remote_hourly_arco": remote_url,
        "remote_gap_fields": REMOTE_GAP_FIELDS,
        "workers": workers,
        "cds_requests": 0,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return 0

    reader = ConsolidatedZarrReader(local_source)
    pressure_digest = source_digest(reader, "pressure", days)
    local_surface_digest = source_digest(reader, "surface", days)
    remote_digests = gap_digests(remote_url, days)
    final_surface_digest = combined_surface_digest(
        local_surface_digest, remote_digests
    )

    pressure_valid = False
    surface_valid = False
    if paths["pressure"].is_file():
        try:
            stage.validate_monthly(
                paths["pressure"], "pressure", days, pressure_digest
            )
            pressure_valid = True
        except Exception:  # noqa: BLE001
            pass
    if paths["surface"].is_file():
        try:
            stage.validate_monthly(
                paths["surface"], "surface", days, final_surface_digest
            )
            surface_valid = True
        except Exception:  # noqa: BLE001
            pass
    if pressure_valid and surface_valid:
        print("existing direct-ARCO monthly staging is valid", flush=True)
        return 0

    if not pressure_valid:
        details = ensure_local_pressure(
            reader, days, paths, pressure_digest
        )
        print(f"local ARCO pressure valid: {details}", flush=True)
    if surface_valid:
        return 0

    details = ensure_local_surface(reader, days, paths, local_surface_digest)
    print(f"local ARCO surface base valid: {details}", flush=True)
    remote_dataset = arco_hourly.open_arco(remote_url)
    try:
        for short_name, source_name in REMOTE_GAP_FIELDS.items():
            _, details = arco_hourly.ensure_field_shard(
                remote_dataset,
                remote_url,
                short_name,
                source_name,
                days,
                paths["gap_root"] / f"{short_name}.nc",
                pressure=False,
                workers=workers,
            )
            print(f"remote ARCO gap valid {short_name}: {details}", flush=True)
    finally:
        remote_dataset.close()
    details = combine_surface(paths, days, local_surface_digest, remote_digests)
    print(f"direct-ARCO surface valid: {details}", flush=True)
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
