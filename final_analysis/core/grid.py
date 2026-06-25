#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/grid.py  —  The common verification grid, regridding, masks and weights.
================================================================================
Everything that puts two different datasets onto the SAME footing lives here:

  * make_grid()          build the regular lat/lon target grid from a GridSpec
  * land_mask()          boolean land mask on that grid (global_land_mask coast)
  * to_grid()            interpolate ANY field onto the target grid + mask ocean
  * load_region_masks()  read the prebuilt IMD-region NetCDF (4 homogeneous regions)
  * build_grid_context() bundle {lat, lon, land, region_masks, all_india} once
  * region_da()          boolean selector DataArray for one region
  * cos_weights()        cosine-latitude area weights (re-exported from metrics)

"Compare at the same resolution" is enforced HERE: every model and the truth are
pushed through `to_grid()` onto the one grid defined by the experiment's GridSpec,
so no metric ever sees mixed resolutions.

The grid context is a plain dict (picklable) so it can be built once per worker
process and passed around cheaply.
================================================================================
"""
import numpy as np
import xarray as xr
from global_land_mask import globe

from .metrics import cos_latitude_weights


# ============================================================== target grid ==
def make_grid(lat0, lat1, lon0, lon1, dgrid):
    """Regular descending-lat / ascending-lon grid.
       lat0 > lat1 (north -> south), lon0 < lon1 (west -> east)."""
    lat = np.arange(lat0, lat1, -dgrid)
    lon = np.arange(lon0, lon1, dgrid)
    return lat, lon


# ================================================================ land mask ==
def land_mask(lat, lon):
    """Boolean (lat, lon) DataArray, True over land, on the verification grid.
       Uses the offline global_land_mask coastline -> fully reproducible, no
       ERA5 land-sea-mask file needed."""
    LON, LAT = np.meshgrid(np.asarray(lon), np.asarray(lat))
    m = globe.is_land(LAT, LON)
    return xr.DataArray(m, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


def mask_land(da, land):
    """Keep land points only; ocean -> NaN (weighted metrics skip NaN)."""
    return da.where(land)


# =========================================================== region masks ===
def load_region_masks(path):
    """Read the prebuilt IMD 4-region boolean masks (NetCDF on the target grid).
       Returns {region_key: bool ndarray}. Build the file once via
       `core/regions.py` from the Survey-of-India shapefile."""
    ds = xr.open_dataset(path)
    return {k: ds[k].values.astype(bool) for k in ds.data_vars}


# ========================================================== grid context ====
def build_grid_context(grid_spec, region_mask_nc):
    """Bundle the grid + masks once per process.

    Parameters
    ----------
    grid_spec     : object with attrs lat0, lat1, lon0, lon1, dgrid
    region_mask_nc: path to the prebuilt IMD-region NetCDF

    Returns
    -------
    dict(lat, lon, land, region_masks, all_india, weights)
        all_india  = logical-OR of the 4 IMD region masks (land-and-region only)
        weights    = cosine-latitude weight DataArray (mean 1.0)
    """
    lat, lon = make_grid(grid_spec.lat0, grid_spec.lat1,
                         grid_spec.lon0, grid_spec.lon1, grid_spec.dgrid)
    land = land_mask(lat, lon)
    region_masks = load_region_masks(region_mask_nc)
    all_india = np.zeros((len(lat), len(lon)), dtype=bool)
    for m in region_masks.values():
        all_india |= m
    return dict(lat=lat, lon=lon, land=land,
                region_masks=region_masks, all_india=all_india,
                weights=cos_latitude_weights(lat, xr))


def region_da(rg, GC):
    """Boolean selector DataArray for region `rg` ('All India' or an IMD key)."""
    arr = GC["all_india"] if rg == "All India" else GC["region_masks"][rg]
    return xr.DataArray(arr, dims=["lat", "lon"],
                        coords={"lat": GC["lat"], "lon": GC["lon"]})


# ============================================================== regridding ===
def to_grid(da, GC):
    """Interpolate `da` onto the verification grid (bilinear) and mask ocean.
       Accepts latitude/longitude OR lat/lon names; squeezes singleton dims."""
    ren = {}
    if "latitude" in da.dims:
        ren["latitude"] = "lat"
    if "longitude" in da.dims:
        ren["longitude"] = "lon"
    if ren:
        da = da.rename(ren)
    out = da.interp(lat=GC["lat"], lon=GC["lon"], method="linear").squeeze()
    # normalise dim order to (..., lat, lon) — some sources (e.g. WeatherBench2)
    # carry (lon, lat) order, which otherwise breaks broadcasting against the
    # (lat, lon) masks/metrics with a (24,22) vs (22,24) shape mismatch.
    out = out.transpose(..., "lat", "lon")
    return mask_land(out, GC["land"])


def crop_box(da, lat_pad=(40.0, 3.0), lon_pad=(60.0, 102.0)):
    """Pre-crop a global field to a generous India box BEFORE interpolation
       (big speed-up: interp touches far fewer points). Handles either
       coordinate orientation (ascending or descending lat/lon)."""
    la = "latitude" if "latitude" in da.dims else "lat"
    lo = "longitude" if "longitude" in da.dims else "lon"
    n_pad, s_pad = lat_pad
    w_pad, e_pad = lon_pad
    lasl = slice(n_pad, s_pad) if float(da[la][0]) > float(da[la][-1]) else slice(s_pad, n_pad)
    losl = slice(w_pad, e_pad) if float(da[lo][0]) < float(da[lo][-1]) else slice(e_pad, w_pad)
    return da.sel({la: lasl, lo: losl})
