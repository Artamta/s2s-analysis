"""Scientific-contract tests for preprocessing."""

import numpy as np
import pandas as pd
import xarray as xr

from prepare_data import (
    _static_features,
    climatology_thresholds,
    ensemble_probabilities,
    observed_category,
)


def test_category_equality_moves_to_upper_nonempty_bin() -> None:
    observation = np.asarray([[1.0, 0.0, 5.0]], dtype=np.float32)
    thresholds = np.asarray(
        [
            [[0.0, 0.0, 5.0]],
            [[1.0, 0.0, 5.0]],
            [[2.0, 1.0, 5.0]],
            [[3.0, 2.0, 5.0]],
        ],
        dtype=np.float32,
    )

    result = observed_category(observation, thresholds)

    assert result.tolist() == [[2, 2, -1]]


def test_smoothed_member_probabilities_are_positive_and_normalized() -> None:
    members = np.arange(10, dtype=np.float32)[:, None, None]
    bounds = np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float32)[:, None, None]

    probabilities = ensemble_probabilities(members, bounds)

    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0)
    assert np.all(probabilities > 0.0)
    np.testing.assert_allclose(
        probabilities[:, 0, 0],
        (np.asarray([2, 2, 2, 2, 2]) + 0.5) / 12.5,
    )


def test_climatology_uses_only_previous_twenty_years_and_five_offsets() -> None:
    time = pd.date_range("1999-01-01", "2021-12-31", freq="D")
    values = np.asarray(time.year * 1000 + time.dayofyear, dtype=np.float32)[:, None, None]
    weekly = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [0.0], "longitude": [0.0]},
    )
    target = pd.Timestamp("2021-06-15")
    expected_dates = [
        target.replace(year=year) + pd.Timedelta(days=offset)
        for year in range(2001, 2021)
        for offset in (-4, -2, 0, 2, 4)
    ]
    expected = np.quantile(
        weekly.sel(time=expected_dates).values,
        [0.2, 0.4, 0.6, 0.8],
        axis=0,
    )

    actual = climatology_thresholds(weekly, target)

    np.testing.assert_allclose(actual, expected)


def test_feature_order_and_normalization() -> None:
    p0 = np.full((2, 5, 3, 4), 0.2, dtype=np.float32)
    quantiles = np.arange(10, dtype=np.float32).reshape(2, 5, 1, 1)
    quantiles = np.broadcast_to(quantiles, (2, 5, 3, 4)).copy()
    mean = np.arange(10, dtype=np.float32).reshape(2, 5)
    std = np.full((2, 5), 2.0, dtype=np.float32)
    latitude = np.asarray([90.0, 0.0, -90.0])
    longitude = np.asarray([0.0, 90.0, 180.0, 270.0])
    land = np.ones((3, 4), dtype=np.float32)

    features = _static_features(
        p0,
        quantiles,
        pd.Timestamp("2020-01-02"),
        latitude,
        longitude,
        land,
        mean,
        std,
    )

    assert features.shape == (2, 18, 3, 4)
    np.testing.assert_allclose(features[:, :5], np.log(0.2))
    np.testing.assert_allclose(features[:, 5:10], 0.0)
    np.testing.assert_allclose(features[0, 10, :, 0], [1.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(features[0, 12, 0], [0.0, 1.0, 0.0, -1.0], atol=1e-6)
    assert np.all(features[0, 16] == -1.0)
    assert np.all(features[1, 16] == 1.0)
    np.testing.assert_allclose(features[:, 17], 1.0)
