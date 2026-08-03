from __future__ import annotations

import numpy as np
import xarray as xr

from s2s_benchmark.plotting import record_label, valid_times_intersection, weighted_spatial_mean


def test_record_label_distinguishes_erpas_sensitivity() -> None:
    base = {"model": "erpas", "experiment_id": "provider/erpas", "grid": "common_1p5"}
    sensitivity = {**base, "experiment_id": "provider/erpas/india_0p5_sensitivity"}
    assert record_label(base) == "ERPAS global"
    assert "sensitivity" in record_label(sensitivity)


def test_valid_time_intersection_matches_dates_not_lead_indices() -> None:
    first = np.array(["2023-06-29", "2023-06-30", "2023-07-01"], dtype="datetime64[D]")
    second = np.array(["2023-06-30", "2023-07-01", "2023-07-02"], dtype="datetime64[D]")
    np.testing.assert_array_equal(
        valid_times_intersection([first, second]),
        np.array(["2023-06-30", "2023-07-01"], dtype="datetime64[ns]"),
    )


def test_weighted_spatial_mean_renormalizes_missing_cells() -> None:
    field = xr.DataArray(
        [[[1.0, np.nan], [3.0, 5.0]]],
        dims=("lead_day", "latitude", "longitude"),
        coords={"lead_day": [1], "latitude": [1.5, 0.0], "longitude": [60.0, 61.5]},
    )
    weight = xr.DataArray(
        [[1.0, 10.0], [1.0, 2.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [1.5, 0.0], "longitude": [60.0, 61.5]},
    )
    assert weighted_spatial_mean(field, weight).item() == 3.5
