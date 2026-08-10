from __future__ import annotations

import numpy as np

from s2s_benchmark.core import StandardField, daily_to_weekly, ensemble_statistics, field_to_dataset


def sample_field(variable: str = "tp") -> StandardField:
    values = np.arange(2 * 7 * 2 * 3, dtype=np.float32).reshape(2, 7, 2, 3)
    return StandardField(
        model="test",
        experiment_id="test/ens2",
        variable=variable,
        initialization="2023-06-29",
        values=values,
        member=np.array([0, 1]),
        lead_day=np.arange(1, 8),
        latitude=np.array([1.5, 0.0]),
        longitude=np.array([60.0, 61.5, 63.0]),
        units="mm day-1" if variable == "tp" else "degC",
        temporal_statistic="daily_mean_rate" if variable == "tp" else "daily_mean",
        distribution_representation="members",
        source_paths=("/tmp/source.nc",),
    )


def test_weekly_precipitation_stores_mean_and_total() -> None:
    field = sample_field()
    weekly = daily_to_weekly(field.values, "tp")
    np.testing.assert_allclose(weekly["weekly_total"], weekly["weekly_mean"] * 7)


def test_incomplete_week_is_not_published() -> None:
    field = sample_field()
    six_days = field.values[:, :6]
    assert daily_to_weekly(six_days, "tp")["weekly_mean"].shape[1] == 0


def test_ensemble_statistics_use_population_std_and_valid_counts() -> None:
    values = np.array([[[[1.0]]], [[[3.0]]]], dtype=np.float32)
    mean, std, count = ensemble_statistics(values)
    assert mean.item() == 2.0
    assert std.item() == 1.0
    assert count.item() == 2


def test_dataset_keeps_members_and_derived_summary() -> None:
    ds = field_to_dataset(sample_field(), "common_1p5")
    assert ds.forecast.dims == ("init", "member", "lead_day", "latitude", "longitude")
    np.testing.assert_allclose(ds.ensemble_mean, ds.forecast.mean("member"))
    np.testing.assert_allclose(ds.forecast_weekly_total, 7 * ds.forecast_weekly_mean)
    assert ds.attrs["distribution_representation"] == "members"


def test_tsfc_semantics_remain_distinct_from_t2m() -> None:
    field = sample_field("t2m")
    tsfc = StandardField(**{**field.__dict__, "variable": "tsfc", "temporal_statistic": "instantaneous_daily_sample"})
    ds = field_to_dataset(tsfc, "common_1p5")
    assert ds.attrs["variable"] == "tsfc"
    assert ds.forecast.attrs["temporal_statistic"] == "instantaneous_daily_sample"


def test_provider_mean_records_source_ensemble_size_per_initialization() -> None:
    field = sample_field("t2m")
    provider_mean = StandardField(
        **{
            **field.__dict__,
            "distribution_representation": "mean_only",
            "source_ensemble_size": 20,
        }
    )
    ds = field_to_dataset(provider_mean, "common_1p5")
    assert ds.source_ensemble_size.dims == ("init",)
    assert ds.source_ensemble_size.item() == 20
