"""Grid, regridding, mask and area-weight helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .regions import REGION_KEYS, open_region_masks


@dataclass(frozen=True)
class GridSpec:
    """Regular India verification grid."""

    lat0: float = 38.0
    lat1: float = 5.0
    lon0: float = 65.0
    lon1: float = 100.0
    dgrid: float = 1.5


def make_grid(spec: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return descending latitude and ascending longitude arrays."""

    lat = np.arange(spec.lat0, spec.lat1, -spec.dgrid)
    lon = np.arange(spec.lon0, spec.lon1, spec.dgrid)
    return lat, lon


def normalize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """Rename common latitude/longitude coordinate names to lat/lon."""

    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"expected lat/lon dimensions, got {da.dims}")
    return da


def to_grid(da: xr.DataArray, spec: GridSpec) -> xr.DataArray:
    """Interpolate a field to the requested verification grid."""

    da = normalize_lat_lon(da)
    lat, lon = make_grid(spec)
    out = da.interp(lat=lat, lon=lon, method="linear").squeeze()
    return out.transpose(..., "lat", "lon")


def crop_india_box(
    da: xr.DataArray,
    lat_bounds: tuple[float, float] = (3.0, 40.0),
    lon_bounds: tuple[float, float] = (60.0, 102.0),
) -> xr.DataArray:
    """Crop a lat/lon field to a generous India box before regridding."""

    da = normalize_lat_lon(da)
    south, north = lat_bounds
    west, east = lon_bounds
    lat_slice = slice(north, south) if float(da.lat[0]) > float(da.lat[-1]) else slice(south, north)
    lon_slice = slice(west, east) if float(da.lon[0]) < float(da.lon[-1]) else slice(east, west)
    return da.sel(lat=lat_slice, lon=lon_slice)


def cosine_weights(lat: xr.DataArray | np.ndarray) -> xr.DataArray:
    """Cosine-latitude weights normalised to mean 1."""

    values = np.asarray(lat, dtype=float)
    weights = np.cos(np.deg2rad(values))
    weights = weights / np.nanmean(weights)
    coords = lat if isinstance(lat, xr.DataArray) else values
    return xr.DataArray(weights, dims=["lat"], coords={"lat": coords})


def region_mask(region: str, dgrid: float = 1.5) -> xr.DataArray:
    """Boolean mask for All India or one IMD homogeneous region."""

    ds = open_region_masks(dgrid)
    if region == "All India":
        arr = ds["All India"].astype(bool)
    elif region in REGION_KEYS:
        arr = ds[region].astype(bool)
    else:
        ds.close()
        raise ValueError(f"unknown region: {region!r}")
    arr = arr.load()
    ds.close()
    return arr


def apply_region(da: xr.DataArray, region: str = "All India", dgrid: float = 1.5) -> xr.DataArray:
    """Mask a gridded field to one verification region."""

    mask = region_mask(region, dgrid=dgrid)
    return normalize_lat_lon(da).where(mask)

