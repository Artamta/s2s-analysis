#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/regions.py  —  Build IMD homogeneous-region masks at ANY grid resolution.
================================================================================
The 4 IMD homogeneous rainfall regions (Pai et al., 2014) are built from the
official Survey-of-India STATE_BOUNDARY.shp (LCC projection, reprojected to
WGS84), unioning states into regions and running point-in-polygon on the target
grid. This is what lets us verify both on the COMMON grid (1.5°) and on a NATIVE
grid (e.g. SPIRE 0.5°) — same regions, different resolution.

Usage
-----
  # one-off: build a mask NetCDF for a given resolution
  python -m core.regions --dgrid 0.5 --out masks/imd_region_masks_0.5deg.nc

  # or programmatically
  from core.regions import build_masks
  masks = build_masks(lat, lon, shp_path)      # {region_key: bool (nlat,nlon)}

Region grouping (state names exactly as in the shapefile STATE attribute):
================================================================================
"""
import argparse
import os

import numpy as np

IMD_STATE_GROUPS = {
    "northwest_india": {
        "JAMMU AND KASHMIR", "LADAKH", "HIMACHAL PRADESH", "PUNJAB", "HARYANA",
        "DELHI", "CHANDIGARH", "UTTARAKHAND", "RAJASTHAN", "UTTAR PRADESH",
        "DISPUTED (MADHYA PRADESH & RAJASTHAN)", "DISPUTED (RAJATHAN & GUJARAT)",
    },
    "central_india": {
        "GUJARAT", "DADRA & NAGAR HAVELI & DAMAN & DIU", "MADHYA PRADESH",
        "CHHATTISGARH", "MAHARASHTRA", "GOA", "ODISHA",
        "DISPUTED (MADHYA PRADESH & GUJARAT)",
    },
    "south_peninsula": {
        "ANDHRA PRADESH", "TELANGANA", "TAMIL NADU", "PUDUCHERRY", "KARNATAKA",
        "KERALA", "LAKSHADWEEP",
    },
    "east_northeast_india": {
        "BIHAR", "WEST BENGAL", "SIKKIM", "JHARKHAND", "ASSAM", "MEGHALAYA",
        "NAGALAND", "MANIPUR", "MIZORAM", "TRIPURA", "ARUNACHAL PRADESH",
        "ANDAMAN & NICOBAR", "DISPUTED (WEST BENGAL , BIHAR & JHARKHAND)",
    },
}
REGION_DISPLAY = {
    "northwest_india": "Northwest India", "central_india": "Central India",
    "south_peninsula": "South Peninsula", "east_northeast_india": "East & NE India",
}

# SOI shapefile is in Lambert Conformal Conic; this proj4 -> WGS84 (no EPSG DB needed)
_LCC = ("+proj=lcc +lat_0=24 +lon_0=80 +lat_1=12.472944 +lat_2=35.172806"
        " +x_0=4000000 +y_0=4000000 +datum=WGS84 +units=m +no_defs")
_WGS84 = "+proj=longlat +datum=WGS84 +no_defs"


def _region_polygons(shp_path, verbose=True):
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union, transform as shp_transform
    from pyproj import Transformer
    tr = Transformer.from_crs(_LCC, _WGS84, always_xy=True)
    state_polys = {}
    for rec in shpreader.Reader(shp_path).records():
        state_polys[rec.attributes["STATE"]] = shp_transform(tr.transform, rec.geometry)
    if verbose:
        print(f"  {len(state_polys)} states loaded + reprojected to WGS84")
    out = {}
    for key, wanted in IMD_STATE_GROUPS.items():
        geoms = [state_polys[s] for s in wanted if s in state_polys]
        if geoms:
            out[key] = unary_union(geoms)
            if verbose:
                print(f"  {REGION_DISPLAY[key]:<18} <- {len(geoms)} states")
    return out


def build_masks(lat, lon, shp_path, verbose=True):
    """{region_key: bool ndarray (nlat,nlon)} via point-in-polygon on (lat,lon)."""
    from shapely.geometry import Point
    lat = np.asarray(lat, float)
    lon = np.where(np.asarray(lon, float) > 180, np.asarray(lon, float) - 360, np.asarray(lon, float))
    polys = _region_polygons(shp_path, verbose=verbose)
    LON, LAT = np.meshgrid(lon, lat)
    pts = list(zip(LON.ravel(), LAT.ravel()))
    masks = {}
    for key, poly in polys.items():
        masks[key] = np.array([poly.contains(Point(x, y)) for x, y in pts],
                              dtype=bool).reshape(len(lat), len(lon))
    if verbose:
        for key, m in masks.items():
            print(f"    {REGION_DISPLAY[key]:<18} {m.sum():>6} pts")
    return masks


def save_masks(masks, lat, lon, outpath):
    import xarray as xr
    ds = xr.Dataset({k: (["lat", "lon"], v.astype(np.int8)) for k, v in masks.items()},
                    coords={"lat": lat, "lon": lon})
    ds.attrs["description"] = ("IMD 4-region boolean masks (1=in region) from Survey "
                               "of India STATE_BOUNDARY.shp, LCC->WGS84.")
    ds.attrs["citation"] = "Pai et al. (2014); boundaries: Survey of India"
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    ds.to_netcdf(outpath)
    print(f"Saved -> {outpath}")


def _main():
    ap = argparse.ArgumentParser(description="Build IMD region masks at a given resolution")
    ap.add_argument("--dgrid", type=float, required=True, help="grid spacing in degrees")
    ap.add_argument("--lat0", type=float, default=38.0)
    ap.add_argument("--lat1", type=float, default=5.0)
    ap.add_argument("--lon0", type=float, default=65.0)
    ap.add_argument("--lon1", type=float, default=100.0)
    ap.add_argument("--shp", default="/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lat = np.arange(a.lat0, a.lat1, -a.dgrid)
    lon = np.arange(a.lon0, a.lon1, a.dgrid)
    print(f"Building IMD masks on {len(lat)}x{len(lon)} grid ({a.dgrid} deg) [SOI boundaries]")
    masks = build_masks(lat, lon, a.shp)
    save_masks(masks, lat, lon, a.out)


if __name__ == "__main__":
    _main()
