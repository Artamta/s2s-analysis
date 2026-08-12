"""Synthetic tests for case-wise spatial verification."""

import numpy as np
import pandas as pd
import pytest

from fuxi_adapter.metrics import (
    compute_case_metrics,
    paired_moving_block_bootstrap,
    summarize_metrics,
    weighted_bias,
    weighted_mae,
    weighted_negative_fraction,
    weighted_rmse,
    weighted_spatial_acc,
)


def test_weighted_scalar_metrics_have_known_values():
    truth = np.array([1.0, 2.0, 3.0])
    prediction = np.array([2.0, 2.0, 1.0])
    weights = np.array([1.0, 2.0, 1.0])

    assert weighted_rmse(truth, prediction, weights) == pytest.approx(np.sqrt(1.25))
    assert weighted_mae(truth, prediction, weights) == pytest.approx(0.75)
    assert weighted_bias(truth, prediction, weights) == pytest.approx(-0.25)


def test_spatial_acc_uses_supplied_anomalies_and_ignores_invalid_cells():
    truth_anomaly = np.array([[-1.0, 0.0], [1.0, np.nan]])
    prediction_anomaly = np.array([[-2.0, 0.0], [2.0, 999.0]])
    weights = np.array([[1.0, 4.0], [2.0, 1.0]])

    assert weighted_spatial_acc(truth_anomaly, prediction_anomaly, weights) == pytest.approx(
        1.0
    )
    assert weighted_spatial_acc(truth_anomaly, -prediction_anomaly, weights) == pytest.approx(
        -1.0
    )
    assert np.isnan(weighted_spatial_acc(np.ones(4), np.arange(4.0), np.ones(4)))


def test_negative_fraction_is_area_weighted_and_masked():
    prediction = np.array([-1.0, -2.0, 3.0, np.nan])
    weights = np.array([1.0, 3.0, 4.0, 100.0])

    assert weighted_negative_fraction(prediction, weights) == pytest.approx(0.5)
    assert weighted_negative_fraction(
        prediction, weights, valid_mask=np.array([True, False, True, True])
    ) == pytest.approx(0.2)


def test_compute_case_metrics_emits_case_lead_region_rows():
    # Two cases, two leads, and a 2 x 2 verification grid.
    truth = np.arange(16.0).reshape(2, 2, 2, 2) + 1.0
    prediction = truth + 1.0
    prediction[0, 0, 0, 0] = -1.0
    climatology = np.full_like(truth, 3.0)
    weights = np.array([[1.0, 2.0], [1.0, 2.0]])
    regions = {
        "india": np.ones((2, 2)),
        "north": np.array([[1.0, 1.0], [0.0, 0.0]]),
    }

    result = compute_case_metrics(
        truth,
        prediction,
        truth - climatology,
        prediction - climatology,
        weights,
        predictor="adapter",
        case_ids=pd.to_datetime(["2024-01-01", "2024-01-04"]),
        leads=[1, 2],
        seasons=np.array(["DJF", "DJF"]),
        region_weights=regions,
    )

    assert len(result) == 2 * 2 * 2
    assert set(result["region"]) == {"india", "north"}
    assert set(result["lead"]) == {1, 2}
    assert (result["valid_cells"] > 0).all()
    exact_rows = result.loc[
        ~((result["case_id"] == pd.Timestamp("2024-01-01")) & (result["lead"] == 1))
    ]
    assert np.allclose(exact_rows["rmse"], 1.0)
    damaged = result.loc[
        (result["case_id"] == pd.Timestamp("2024-01-01"))
        & (result["lead"] == 1)
        & (result["region"] == "india")
    ].iloc[0]
    assert damaged["negative_fraction"] == pytest.approx(1.0 / 6.0)


def test_summarize_metrics_averages_cases_not_grid_cells():
    frame = pd.DataFrame(
        {
            "predictor": ["adapter", "adapter"],
            "lead": [1, 1],
            "region": ["india", "india"],
            "season": ["JJA", "JJA"],
            "acc": [0.2, 0.8],
            "rmse": [1.0, 3.0],
            "mae": [0.5, 1.5],
            "bias": [-1.0, 1.0],
            "negative_fraction": [0.0, 0.2],
            "valid_cells": [10, 1000],
        }
    )

    summary = summarize_metrics(frame).iloc[0]
    assert summary["case_count"] == 2
    assert summary["acc_mean"] == pytest.approx(0.5)
    assert summary["rmse_mean"] == pytest.approx(2.0)
    assert summary["negative_fraction_mean"] == pytest.approx(0.1)


def _bootstrap_fixture():
    rows = []
    for case in range(20):
        for lead in (1, 2):
            baseline_acc = 0.01 * case + 0.05 * lead
            baseline_rmse = 3.0 + 0.02 * case + lead
            for predictor, acc, rmse in (
                ("raw", baseline_acc, baseline_rmse),
                ("adapter", baseline_acc + 0.2, baseline_rmse - 1.0),
            ):
                rows.append(
                    {
                        "case_id": case,
                        "predictor": predictor,
                        "lead": lead,
                        "region": "india",
                        "season": "ALL",
                        "acc": acc,
                        "rmse": rmse,
                        "mae": rmse / 2.0,
                        "bias": 0.0,
                        "negative_fraction": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_block_bootstrap_recovers_constant_differences_and_is_deterministic():
    frame = _bootstrap_fixture()
    first = paired_moving_block_bootstrap(
        frame,
        "adapter",
        "raw",
        metric_columns=("acc", "rmse"),
        block_length=13,
        n_resamples=250,
        seed=7,
    )
    second = paired_moving_block_bootstrap(
        frame,
        "adapter",
        "raw",
        metric_columns=("acc", "rmse"),
        block_length=13,
        n_resamples=250,
        seed=7,
    )
    pd.testing.assert_frame_equal(first, second)

    acc = first.loc[first["metric"] == "acc"]
    rmse = first.loc[first["metric"] == "rmse"]
    assert np.allclose(acc["mean_difference"], 0.2)
    assert np.allclose(acc["ci_lower"], 0.2)
    assert np.allclose(acc["ci_upper"], 0.2)
    assert np.allclose(rmse["mean_difference"], -1.0)
    assert np.all(acc["paired_case_count"] == 20)


def test_paired_block_bootstrap_uses_only_common_cases_per_lead():
    frame = _bootstrap_fixture()
    missing = (
        (frame["predictor"] == "raw")
        & (frame["lead"] == 2)
        & (frame["case_id"] == 5)
    )
    frame = frame.loc[~missing]

    result = paired_moving_block_bootstrap(
        frame,
        "adapter",
        "raw",
        metric_columns=("acc",),
        n_resamples=50,
    )
    counts = result.set_index("lead")["paired_case_count"].to_dict()
    assert counts == {1: 20, 2: 19}


def test_duplicate_case_rows_are_rejected_before_bootstrap():
    frame = _bootstrap_fixture()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate predictor rows"):
        paired_moving_block_bootstrap(frame, "adapter", "raw", n_resamples=10)
