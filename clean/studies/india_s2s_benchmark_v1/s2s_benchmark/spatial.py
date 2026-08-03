from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Blosc

from .core import sha256_file
from .remap import centers_to_edges


EARTH_RADIUS_KM = 6371.0088


def spherical_cell_area_km2(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat_edges = centers_to_edges(latitude, clip_latitude=True)
    lon_edges = centers_to_edges(longitude)
    lat_band = np.abs(np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1])))
    lon_width = np.abs(np.deg2rad(lon_edges[1:] - lon_edges[:-1]))
    return (EARTH_RADIUS_KM**2 * lat_band[:, None] * lon_width[None, :]).astype(np.float64)


def support_fraction(
    source_support: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    """Map binary/fractional support by spherical overlap, treating outside as zero."""
    support = np.asarray(source_support, dtype=np.float64)
    if support.shape != (len(source_latitude), len(source_longitude)):
        raise ValueError("support shape does not match source coordinates")
    if not np.isfinite(support).all() or support.min() < 0 or support.max() > 1:
        raise ValueError("support must be finite and within [0, 1]")
    lat_order = np.argsort(source_latitude)
    lon_order = np.argsort(source_longitude)
    target_lat_order = np.argsort(target_latitude)
    source_lat = np.asarray(source_latitude)[lat_order]
    source_lon = np.asarray(source_longitude)[lon_order]
    target_lat = np.asarray(target_latitude)[target_lat_order]
    target_lon = np.asarray(target_longitude)
    support = support[lat_order, :][:, lon_order]
    sy = centers_to_edges(source_lat, clip_latitude=True)
    sx = centers_to_edges(source_lon)
    ty = centers_to_edges(target_lat, clip_latitude=True)
    tx = centers_to_edges(target_lon)
    lat_overlap = np.zeros((len(target_lat), len(source_lat)), dtype=np.float64)
    lon_overlap = np.zeros((len(target_lon), len(source_lon)), dtype=np.float64)
    for index in range(len(target_lat)):
        lo = np.maximum(sy[:-1], ty[index])
        hi = np.minimum(sy[1:], ty[index + 1])
        keep = hi > lo
        lat_overlap[index, keep] = np.sin(np.deg2rad(hi[keep])) - np.sin(np.deg2rad(lo[keep]))
    for index in range(len(target_lon)):
        lo = np.maximum(sx[:-1], tx[index])
        hi = np.minimum(sx[1:], tx[index + 1])
        keep = hi > lo
        lon_overlap[index, keep] = np.deg2rad(hi[keep] - lo[keep])
    target_area = (
        (np.sin(np.deg2rad(ty[1:])) - np.sin(np.deg2rad(ty[:-1])))[:, None]
        * np.deg2rad(tx[1:] - tx[:-1])[None, :]
    )
    represented_area = lat_overlap @ support @ lon_overlap.T
    fraction = np.divide(
        represented_area,
        target_area,
        out=np.zeros_like(represented_area),
        where=target_area > 0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    fraction[fraction < 1e-12] = 0.0
    return fraction[np.argsort(target_lat_order)].astype(np.float32)


def build_spatial_support(
    source: Path,
    destination: Path,
    regions: list[str],
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    common_grid_id: str,
) -> dict:
    if destination.exists():
        return validate_spatial_support(destination)
    with xr.open_dataset(source, engine="h5netcdf") as masks:
        missing = sorted(set(regions) - set(masks.data_vars))
        if missing:
            raise ValueError(f"region masks missing from {source}: {missing}")
        source_lat = masks["lat"].values
        source_lon = masks["lon"].values
        region_values = {name: np.asarray(masks[name].values, dtype=np.float64) for name in regions}
        union = np.any(np.stack([region_values[name] > 0 for name in regions]), axis=0).astype(np.float64)
        source_attrs = dict(masks.attrs)
    data_vars = {
        f"{name}_fraction": (
            ("latitude", "longitude"),
            support_fraction(region_values[name], source_lat, source_lon, target_latitude, target_longitude),
        )
        for name in regions
    }
    india_fraction = support_fraction(union, source_lat, source_lon, target_latitude, target_longitude)
    cell_area = spherical_cell_area_km2(target_latitude, target_longitude)
    data_vars.update(
        india_fraction=(("latitude", "longitude"), india_fraction),
        cell_area_km2=(("latitude", "longitude"), cell_area),
        india_area_weight_km2=(("latitude", "longitude"), cell_area * india_fraction),
    )
    ds = xr.Dataset(
        data_vars,
        coords={"latitude": target_latitude.astype(np.float64), "longitude": target_longitude.astype(np.float64)},
        attrs={
            "schema_version": 1,
            "archive_id": "india_s2s_benchmark_v1",
            "common_grid_id": common_grid_id,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "source_description": source_attrs.get("description", ""),
            "source_citation": source_attrs.get("citation", ""),
            "regions": json.dumps(regions),
            "remapping_method": "first-order spherical cell-overlap support fraction; outside source extent is zero",
            "weight_contract": "india_area_weight_km2 = cell_area_km2 * india_fraction",
        },
    )
    for name in regions:
        ds[f"{name}_fraction"].attrs.update(units="1", valid_min=0.0, valid_max=1.0)
    ds["india_fraction"].attrs.update(units="1", valid_min=0.0, valid_max=1.0)
    ds["cell_area_km2"].attrs.update(units="km2")
    ds["india_area_weight_km2"].attrs.update(units="km2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".incomplete-{uuid.uuid4().hex[:10]}")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    encoding = {name: {"compressor": compressor, "chunks": ds[name].shape} for name in ds.data_vars}
    try:
        ds.to_zarr(temporary, mode="w", consolidated=False, encoding=encoding)
        import zarr

        zarr.consolidate_metadata(str(temporary))
        validate_spatial_support(temporary)
        os.rename(temporary, destination)
    finally:
        ds.close()
    return validate_spatial_support(destination)


def validate_spatial_support(path: Path) -> dict:
    with xr.open_zarr(path, consolidated=True) as ds:
        required = {"india_fraction", "cell_area_km2", "india_area_weight_km2"}
        missing = required - set(ds.data_vars)
        if missing:
            raise ValueError(f"{path}: missing spatial fields {sorted(missing)}")
        fraction = ds["india_fraction"].values
        area = ds["cell_area_km2"].values
        weight = ds["india_area_weight_km2"].values
        if not np.isfinite(fraction).all() or fraction.min() < 0 or fraction.max() > 1:
            raise ValueError(f"{path}: invalid India support fractions")
        if not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError(f"{path}: invalid spherical cell areas")
        if not np.allclose(weight, area * fraction, rtol=1e-12, atol=1e-9):
            raise ValueError(f"{path}: India area weights do not match their contract")
        result = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "passed",
            "store": str(path),
            "common_grid_id": ds.attrs["common_grid_id"],
            "source_path": ds.attrs["source_path"],
            "source_sha256": ds.attrs["source_sha256"],
            "india_fraction_cell_count": int(np.sum(fraction > 0)),
            "india_fraction_sum": float(fraction.sum()),
            "india_weight_sum_km2": float(weight.sum()),
            "zmetadata_sha256": sha256_file(path / ".zmetadata"),
        }
    return result
