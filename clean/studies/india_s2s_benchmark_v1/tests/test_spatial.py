from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from s2s_benchmark.spatial import build_spatial_support, support_fraction


def test_support_fraction_preserves_aligned_binary_cells() -> None:
    latitude = np.array([1.5, 0.0])
    longitude = np.array([60.0, 61.5])
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    actual = support_fraction(source, latitude, longitude, latitude, longitude)
    np.testing.assert_allclose(actual, source, rtol=0, atol=1e-7)


def test_spatial_store_has_area_weight_contract(tmp_path: Path) -> None:
    source = tmp_path / "masks.nc"
    xr.Dataset(
        {
            "north": (("lat", "lon"), np.array([[1, 0], [0, 0]], dtype=np.int8)),
            "south": (("lat", "lon"), np.array([[0, 0], [0, 1]], dtype=np.int8)),
        },
        coords={"lat": [1.5, 0.0], "lon": [60.0, 61.5]},
        attrs={"description": "test", "citation": "test"},
    ).to_netcdf(source, engine="h5netcdf")
    result = build_spatial_support(
        source,
        tmp_path / "spatial.zarr",
        ["north", "south"],
        np.array([1.5, 0.0]),
        np.array([60.0, 61.5]),
        "test_grid",
    )
    assert result["status"] == "passed"
    with xr.open_zarr(result["store"], consolidated=True) as ds:
        np.testing.assert_allclose(
            ds["india_area_weight_km2"], ds["cell_area_km2"] * ds["india_fraction"]
        )
