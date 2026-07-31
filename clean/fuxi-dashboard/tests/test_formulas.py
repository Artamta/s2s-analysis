"""Hand-calculated tests for every locked scientific formula."""

from __future__ import annotations

import numpy as np
import pytest

from science.formulas import (
    anomaly,
    area_weights,
    calendar_interpolation,
    climatology_spread,
    climatology_terciles,
    ensemble_mean,
    forecast_spread,
    geopotential_to_height_dam,
    kelvin_to_celsius,
    pascal_to_hectopascal,
    probability_above_normal,
    probability_below_normal,
    tp_mm_hour_to_mm_day,
    top_net_thermal_to_olr,
    verification_metrics,
    weekly_mean,
    weekly_mean_rainfall,
    weekly_total,
    weighted_spatial_correlation,
    wind_speed,
)


def test_forecast_unit_conversions_are_exact() -> None:
    np.testing.assert_array_equal(
        tp_mm_hour_to_mm_day([0.0, 0.5, 1.25]), [0.0, 12.0, 30.0]
    )
    np.testing.assert_allclose(
        kelvin_to_celsius([273.15, 300.0]), [0.0, 26.85], atol=1e-12
    )
    np.testing.assert_allclose(
        geopotential_to_height_dam([98.0665]), [1.0], atol=1e-12
    )
    np.testing.assert_array_equal(wind_speed([3.0], [4.0]), [5.0])
    np.testing.assert_array_equal(
        pascal_to_hectopascal([101_325.0]), [1013.25]
    )
    np.testing.assert_array_equal(top_net_thermal_to_olr([-240.0]), [240.0])


def test_seven_daily_fields_reproduce_weekly_total_and_rate() -> None:
    daily = np.arange(1.0, 15.0)
    np.testing.assert_array_equal(weekly_total(daily), [28.0, 77.0])
    np.testing.assert_array_equal(weekly_mean_rainfall(daily), [4.0, 11.0])
    np.testing.assert_array_equal(weekly_mean(daily), [4.0, 11.0])


def test_weekly_reduction_preserves_nonleading_axis_order() -> None:
    values = np.arange(2 * 14 * 3, dtype=float).reshape(2, 14, 3)
    result = weekly_total(values, day_axis=1)
    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result[:, 0], values[:, :7].sum(axis=1))
    np.testing.assert_array_equal(result[:, 1], values[:, 7:].sum(axis=1))


def test_weekly_reduction_rejects_partial_week() -> None:
    with pytest.raises(ValueError, match="divisible by seven"):
        weekly_total(np.ones(8))


def test_anomaly_sign_and_calendar_interpolation() -> None:
    np.testing.assert_array_equal(anomaly([5.0, 2.0], [3.0, 4.0]), [2.0, -2.0])
    np.testing.assert_allclose(
        calendar_interpolation([1.0, 4.0], [4.0, 10.0], 2.0 / 3.0),
        [3.0, 8.0],
    )
    with pytest.raises(ValueError, match="between zero and one"):
        calendar_interpolation([1.0], [2.0], 1.1)


def test_ensemble_statistics_use_locked_ddof_and_terciles() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    assert ensemble_mean(values) == 2.0
    assert forecast_spread(values) == pytest.approx(np.std(values, ddof=0))
    assert climatology_spread(values) == pytest.approx(np.std(values, ddof=1))
    yearly = np.arange(1.0, 21.0)
    lower, upper = climatology_terciles(yearly)
    np.testing.assert_allclose(
        [lower, upper], np.quantile(yearly, [1.0 / 3.0, 2.0 / 3.0])
    )


def test_tercile_probabilities_use_strict_member_counts() -> None:
    members = np.asarray([0.0, 1.0, 2.0, 3.0])
    assert probability_below_normal(members, 1.0) == 25.0
    assert probability_above_normal(members, 2.0) == 25.0


def test_area_weights_are_positive_and_normalized() -> None:
    weights = area_weights(
        np.asarray([0.0, 60.0]),
        np.asarray([[1.0, 0.0], [1.0, 1.0]]),
    )
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights[weights > 0.0] > 0.0)
    assert weights[0, 0] == pytest.approx(2.0 * weights[1, 0])


def test_verification_metrics_match_hand_calculation() -> None:
    forecast = np.asarray([2.0, 4.0, 6.0])
    observation = np.asarray([1.0, 5.0, 5.0])
    metrics = verification_metrics(
        forecast,
        observation,
        np.ones(3),
        forecast_anomaly=[-1.0, 0.0, 1.0],
        observation_anomaly=[-2.0, 0.0, 2.0],
    )
    assert metrics["bias"] == pytest.approx(1.0 / 3.0)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["acc"] == pytest.approx(1.0)


def test_acc_rejects_constant_anomaly_field() -> None:
    with pytest.raises(ValueError, match="constant"):
        weighted_spatial_correlation([1.0, 1.0], [0.0, 1.0], [1.0, 1.0])
