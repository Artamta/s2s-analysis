"""
The land mask underpins every land-only score. Check it on the verification grid
and at known land/ocean points over the India domain.
"""
import numpy as np
import pytest

pytest.importorskip("global_land_mask")
from utils.verification_extra import get_land_mask, mask_land
import xarray as xr

TARGET_LAT = np.arange(38, 5, -1.5)
TARGET_LON = np.arange(65, 100, 1.5)


def test_india_land_fraction_reasonable():
    m = get_land_mask(TARGET_LAT, TARGET_LON)
    frac = float(m.mean())
    # India box is a land/ocean mix; ~0.6 land. Guard against an all-land/all-sea bug.
    assert 0.4 < frac < 0.8, f"land fraction {frac:.2f} outside expected range"


def test_known_points():
    # unique grid coords so each (lat, lon) lookup is a single cell
    m = get_land_mask(np.array([28.5, 15.0]), np.array([64.0, 77.0, 88.0]))
    assert bool(m.sel(lat=28.5, lon=77.0))      # Delhi -> land
    assert not bool(m.sel(lat=15.0, lon=64.0))  # Arabian Sea -> ocean
    assert not bool(m.sel(lat=15.0, lon=88.0))  # Bay of Bengal -> ocean


def test_mask_sets_ocean_to_nan():
    m = get_land_mask(np.array([15.0]), np.array([64.0, 77.0]))  # ocean, land
    da = xr.DataArray([[1.0, 1.0]], coords={"lat": [15.0], "lon": [64.0, 77.0]},
                      dims=["lat", "lon"])
    out = mask_land(da, m)
    assert np.isnan(float(out.sel(lat=15.0, lon=64.0)))   # ocean -> NaN
    assert float(out.sel(lat=15.0, lon=77.0)) == 1.0       # land kept
