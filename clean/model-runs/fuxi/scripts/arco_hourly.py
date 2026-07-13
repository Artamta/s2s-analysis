#!/usr/bin/env python3
"""Shared bounded-memory direct-ARCO reader for FuXi daily-mean inputs."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import get_codec


ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PRESSURE_FIELDS = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
}
SURFACE_FIELDS = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sst": "sea_surface_temperature",
    "ttr": "top_net_thermal_radiation",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "100u": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "tcwv": "total_column_water_vapour",
    "tp": "total_precipitation",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def expected_latitudes() -> np.ndarray:
    return np.linspace(90.0, -90.0, 121, dtype=np.float32)


def expected_longitudes() -> np.ndarray:
    return np.arange(0.0, 360.0, 1.5, dtype=np.float32)


class RemoteArcoReader:
    """Decode full-grid hourly Zarr chunks from anonymous GCS with bounded RAM."""

    def __init__(self, url: str = ARCO_URL) -> None:
        if not url.startswith("gs://"):
            raise ValueError(f"ARCO path must use gs://, found {url!r}")
        self.url = url.rstrip("/")
        self.root = self.url.removeprefix("gs://")
        self.fs = gcsfs.GCSFileSystem(token="anon")
        metadata_bytes = self.fs.cat_file(f"{self.root}/.zmetadata")
        document = json.loads(metadata_bytes)
        self.metadata = document["metadata"]
        self.attrs = dict(self.metadata[".zattrs"])
        self.valid_time_start = pd.Timestamp(self.attrs["valid_time_start"])
        self.valid_time_stop = pd.Timestamp(self.attrs["valid_time_stop"])
        time_attrs = self.array_attrs("time")
        units_prefix = "hours since "
        units = str(time_attrs.get("units", ""))
        if not units.startswith(units_prefix):
            raise ValueError(f"unsupported remote ARCO time units: {units!r}")
        epoch = pd.Timestamp(units[len(units_prefix) :])
        first_time_value = int(self._decode("time", "0").reshape(-1)[0])
        self.array_time_start = epoch + pd.Timedelta(hours=first_time_value)
        self._validate_coordinates()

    def array_metadata(self, name: str) -> dict[str, Any]:
        try:
            return self.metadata[f"{name}/.zarray"]
        except KeyError as exc:
            raise KeyError(f"remote ARCO store has no variable {name!r}") from exc

    def array_attrs(self, name: str) -> dict[str, Any]:
        attrs = dict(self.metadata.get(f"{name}/.zattrs", {}))
        attrs.pop("_ARRAY_DIMENSIONS", None)
        return attrs

    def _decode(self, name: str, key: str) -> np.ndarray:
        metadata = self.array_metadata(name)
        if metadata.get("filters"):
            raise ValueError(f"unsupported Zarr filters for {name}: {metadata['filters']}")
        payload = self.fs.cat_file(f"{self.root}/{name}/{key}")
        if metadata.get("compressor"):
            payload = get_codec(metadata["compressor"]).decode(payload)
        shape = tuple(metadata["chunks"])
        values = np.frombuffer(payload, dtype=np.dtype(metadata["dtype"]))
        if values.size != int(np.prod(shape)):
            raise ValueError(f"{name}/{key} decoded to an unexpected size")
        return values.reshape(shape, order=metadata.get("order", "C"))

    def _coordinate(self, name: str) -> np.ndarray:
        metadata = self.array_metadata(name)
        if tuple(metadata["chunks"]) != tuple(metadata["shape"]):
            raise ValueError(f"coordinate {name} is split across chunks")
        return self._decode(name, "0")

    def _validate_coordinates(self) -> None:
        latitude = self._coordinate("latitude")
        longitude = self._coordinate("longitude")
        levels = self._coordinate("level").astype(np.int64)
        if not np.allclose(latitude, np.linspace(90, -90, 721)):
            raise ValueError("remote ARCO latitude coordinate is unexpected")
        if not np.allclose(longitude, np.arange(0, 360, 0.25)):
            raise ValueError("remote ARCO longitude coordinate is unexpected")
        missing = sorted(set(LEVELS).difference(levels.tolist()))
        if missing:
            raise ValueError(f"remote ARCO is missing pressure levels {missing}")
        self.level_indices = np.asarray(
            [int(np.flatnonzero(levels == level)[0]) for level in LEVELS]
        )

    def validate_availability(self, days: tuple[pd.Timestamp, ...]) -> None:
        needed_start = min(days)
        needed_stop = max(days) + pd.Timedelta(hours=23)
        if needed_start < self.valid_time_start or needed_stop > self.valid_time_stop:
            raise ValueError(
                f"ARCO stable ERA5 covers {self.valid_time_start} through "
                f"{self.valid_time_stop}, but the request needs {needed_start} "
                f"through {needed_stop}"
            )

    def close(self) -> None:
        return None

    def _time_index(self, timestamp: pd.Timestamp) -> int:
        delta = timestamp - self.array_time_start
        index = int(delta / pd.Timedelta(hours=1))
        if self.array_time_start + pd.Timedelta(hours=index) != timestamp:
            raise ValueError(f"timestamp is not an exact ARCO hour: {timestamp}")
        if not 0 <= index < int(self.array_metadata("time")["shape"][0]):
            raise ValueError(f"timestamp is outside the ARCO array: {timestamp}")
        return index

    def _selected_hour(
        self,
        source_name: str,
        timestamp: pd.Timestamp,
        pressure: bool,
        retries: int,
        retry_seconds: int,
    ) -> np.ndarray:
        index = self._time_index(timestamp)
        key = f"{index}.0.0.0" if pressure else f"{index}.0.0"
        for attempt in range(1, retries + 1):
            try:
                chunk = self._decode(source_name, key)
                if pressure:
                    selected = chunk[0, self.level_indices][:, ::6, ::6]
                else:
                    selected = chunk[0, ::6, ::6]
                return np.asarray(selected, dtype=np.float32).copy()
            except Exception:  # noqa: BLE001
                if attempt == retries:
                    raise
                time.sleep(retry_seconds * attempt)
        raise AssertionError("unreachable")

    def daily_values(
        self,
        source_name: str,
        days: tuple[pd.Timestamp, ...],
        *,
        pressure: bool,
        workers: int,
        retries: int = 4,
        retry_seconds: int = 15,
    ) -> np.ndarray:
        self.validate_availability(days)
        field_shape = (
            (len(LEVELS), 121, 240) if pressure else (121, 240)
        )
        output = np.empty((len(days), *field_shape), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for day_index, day in enumerate(days):
                futures = [
                    executor.submit(
                        self._selected_hour,
                        source_name,
                        day + pd.Timedelta(hours=hour),
                        pressure,
                        retries,
                        retry_seconds,
                    )
                    for hour in range(24)
                ]
                total = np.zeros(field_shape, dtype=np.float64)
                for future in as_completed(futures):
                    total += future.result()
                output[day_index] = total / 24.0
                print(
                    f"ARCO {source_name}: day {day_index + 1}/{len(days)} "
                    f"{day:%Y-%m-%d}",
                    flush=True,
                )
        return output


def open_arco(url: str = ARCO_URL) -> RemoteArcoReader:
    return RemoteArcoReader(url)


def validate_availability(
    dataset: RemoteArcoReader, days: tuple[pd.Timestamp, ...]
) -> None:
    dataset.validate_availability(days)


def daily_field(
    dataset: RemoteArcoReader,
    short_name: str,
    source_name: str,
    days: tuple[pd.Timestamp, ...],
    *,
    pressure: bool,
    workers: int = 4,
    retries: int = 4,
    retry_seconds: int = 15,
) -> xr.DataArray:
    source_days = (
        tuple(day + pd.Timedelta(hours=1) for day in days)
        if short_name in {"ttr", "tp"}
        else days
    )
    values = dataset.daily_values(
        source_name,
        source_days,
        pressure=pressure,
        workers=workers,
        retries=retries,
        retry_seconds=retry_seconds,
    )
    attrs = dataset.array_attrs(source_name)
    if short_name == "ttr":
        values /= np.float32(3600.0)
        attrs["units"] = "W m**-2"
        attrs["processing"] = (
            "mean of hourly accumulations valid 01 UTC through next-day 00 UTC, "
            "divided by 3600"
        )
    elif short_name == "tp":
        attrs["processing"] = (
            "mean of hourly accumulations valid 01 UTC through next-day 00 UTC"
        )
    output_dims = (
        ("time", "level", "lat", "lon")
        if pressure
        else ("time", "lat", "lon")
    )
    coords: dict[str, Any] = {
        "time": pd.DatetimeIndex(days),
        "lat": expected_latitudes(),
        "lon": expected_longitudes(),
    }
    if pressure:
        coords["level"] = LEVELS
    attrs.update(
        source="ARCO-ERA5 public Zarr on Google Cloud",
        source_path=dataset.url,
        temporal_statistic="UTC daily mean from 24 hourly ERA5 values",
    )
    return xr.DataArray(
        values,
        dims=output_dims,
        coords=coords,
        name=short_name,
        attrs=attrs,
    )


def field_contract_hash(
    url: str,
    short_name: str,
    source_name: str,
    days: tuple[pd.Timestamp, ...],
    pressure: bool,
) -> str:
    return canonical_hash(
        {
            "schema_version": 3,
            "source": url,
            "short_name": short_name,
            "source_name": source_name,
            "days": [day.strftime("%Y-%m-%d") for day in days],
            "levels": LEVELS if pressure else None,
            "source_grid_degrees": 0.25,
            "target_grid_degrees": 1.5,
            "spatial_method": "point sample every sixth grid cell",
            "temporal_method": "mean of 24 hourly ERA5 values",
            "accumulation_window": "hours 01 UTC through next-day 00 UTC"
            if short_name in {"ttr", "tp"}
            else None,
            "ttr_conversion": "divide J m-2 hourly accumulations by 3600"
            if short_name == "ttr"
            else None,
        }
    )


def write_field_shard(data: xr.DataArray, path: Path, digest: str) -> None:
    dataset = data.to_dataset()
    dataset.attrs.update(
        field_contract_sha256=digest,
        daily_statistic="daily_mean",
        hourly_sampling="1_hourly",
        time_zone="utc+00:00",
    )
    pressure = "level" in data.dims
    chunks = (1, len(LEVELS), 121, 240) if pressure else (1, 121, 240)
    encoding = {
        data.name: {
            "zlib": True,
            "complevel": 2,
            "dtype": "float32",
            "chunksizes": chunks,
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary, encoding=encoding)
    temporary.replace(path)


def validate_field_shard(
    path: Path,
    short_name: str,
    days: tuple[pd.Timestamp, ...],
    digest: str,
    pressure: bool,
) -> dict[str, Any]:
    dataset = xr.open_dataset(path)
    try:
        if set(dataset.data_vars) != {short_name}:
            raise ValueError(f"{path} does not contain only {short_name}")
        data = dataset[short_name]
        expected_dims = (
            ("time", "level", "lat", "lon")
            if pressure
            else ("time", "lat", "lon")
        )
        if data.dims != expected_dims:
            raise ValueError(f"{short_name} has dimensions {data.dims}, expected {expected_dims}")
        if not pd.DatetimeIndex(data.time.values).normalize().equals(
            pd.DatetimeIndex(days)
        ):
            raise ValueError(f"{short_name} does not contain the required days")
        if pressure and data.level.values.tolist() != LEVELS:
            raise ValueError(f"{short_name} pressure levels are in the wrong order")
        if not np.allclose(data.lat.values, expected_latitudes()):
            raise ValueError(f"{short_name} has the wrong latitude grid")
        if not np.allclose(data.lon.values, expected_longitudes()):
            raise ValueError(f"{short_name} has the wrong longitude grid")
        if dataset.attrs.get("field_contract_sha256") != digest:
            raise ValueError(f"{short_name} field contract hash mismatch")
        sample = data.isel(time=sorted({0, len(days) - 1})).values
        if short_name == "sst":
            if not np.isfinite(sample).any():
                raise ValueError("SST shard has no finite ocean values")
        elif not np.isfinite(sample).all():
            raise ValueError(f"{short_name} shard contains missing values")
        return {"days": len(days), "size_bytes": path.stat().st_size}
    finally:
        dataset.close()


def ensure_field_shard(
    dataset: RemoteArcoReader,
    url: str,
    short_name: str,
    source_name: str,
    days: tuple[pd.Timestamp, ...],
    path: Path,
    *,
    pressure: bool,
    workers: int,
) -> tuple[str, dict[str, Any]]:
    digest = field_contract_hash(url, short_name, source_name, days, pressure)
    if path.is_file():
        try:
            return digest, validate_field_shard(
                path, short_name, days, digest, pressure
            )
        except Exception:  # noqa: BLE001
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            path.replace(path.with_name(f"{path.name}.invalid.{timestamp}"))
    started = time.monotonic()
    print(f"ARCO start {short_name}: {len(days) * 24} hourly chunks", flush=True)
    data = daily_field(
        dataset,
        short_name,
        source_name,
        days,
        pressure=pressure,
        workers=workers,
    )
    write_field_shard(data, path, digest)
    details = validate_field_shard(path, short_name, days, digest, pressure)
    details["elapsed_seconds"] = round(time.monotonic() - started, 1)
    print(f"ARCO done {short_name}: {details}", flush=True)
    return digest, details
