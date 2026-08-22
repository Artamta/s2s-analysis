"""Integration contracts for the all-season categorical PBC driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fuxi_allseason_pbc_baseline as driver
import fuxi_allseason_ensemble_calibration as frozen
from fuxi_pbc_core import build_issue_time_lags


def _synthetic_dates() -> np.ndarray:
    groups = (
        np.datetime64("2002-01-03") + np.arange(12) * np.timedelta64(7, "D"),
        np.asarray(["2017-12-21", "2017-12-28"], dtype="datetime64[D]"),
        np.datetime64("2018-01-04") + np.arange(10) * np.timedelta64(7, "D"),
        np.asarray(["2019-12-19", "2019-12-26"], dtype="datetime64[D]"),
        np.datetime64("2020-01-02") + np.arange(10) * np.timedelta64(7, "D"),
    )
    return np.concatenate(groups)


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = _synthetic_dates()
    rng = np.random.default_rng(11)
    target_starts = dates[:, None] + (7 * np.arange(6)).astype("timedelta64[D]")[None]
    unique_starts, inverse = np.unique(target_starts.reshape(-1), return_inverse=True)
    unique_truth = rng.gamma(1.2, 2.0, size=(len(unique_starts), 2, 3)).astype(
        np.float32
    )
    unique_truth[rng.random(unique_truth.shape) < 0.65] = 0.0
    truth = unique_truth[inverse].reshape(len(dates), 6, 2, 3)
    members = np.maximum(
        0.0,
        truth[:, None] + 0.4 + rng.normal(0.0, 1.2, size=(len(dates), 7, 6, 2, 3)),
    ).astype(np.float32)
    return dates, truth, members


def test_parser_has_frozen_scientific_defaults() -> None:
    args = driver.build_parser().parse_args([])
    driver.validate_args(args)
    assert args.debias_spans == (14, 28, 35)
    assert args.calendar_window_days == 31
    assert args.ridge == pytest.approx(1.0e-3)
    assert args.bootstrap_samples == 2000
    smoke = driver.build_parser().parse_args(["--smoke"])
    driver.validate_args(smoke)
    assert smoke.bootstrap_samples == 100


def test_temporal_contract_proves_purge_and_exact_issue_time_lags() -> None:
    dates, truth, _ = _synthetic_data()
    splits = frozen.make_split_indices(dates)
    lags = build_issue_time_lags(dates, truth)
    evidence = driver.assert_temporal_contract(dates, splits, lags)
    assert evidence["training_latest_outcome_end"] < "2018-01-01"
    assert evidence["validation_latest_outcome_end"] < "2020-01-01"
    assert np.all(lags.available[np.concatenate((splits.validation, splits.test))])
    issue = np.broadcast_to(dates[:, None], lags.window_end.shape)
    available = lags.source_indices >= 0
    assert np.all(lags.window_end[available] < issue[available])

    bad_dates = dates.copy()
    bad_dates[-1] = np.datetime64("2025-01-02")
    with pytest.raises(driver.BaselineContractError, match="2020--2021|2025"):
        driver.assert_temporal_contract(
            bad_dates,
            frozen.SplitIndices(
                splits.train, splits.validation, splits.test, splits.embargo
            ),
        )


def test_atomic_main_publishes_once_and_retains_failure_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    successful = tmp_path / "published"

    def fake_run(args: argparse.Namespace, staging: Path):
        assert staging.name.startswith(".published.incomplete-")
        driver._atomic_json(
            staging / "manifest.json",
            {"experiment": driver.EXPERIMENT, "status": "complete"},
        )
        return {"status": "complete"}

    monkeypatch.setattr(driver, "run_experiment", fake_run)
    assert driver.main(["--output", str(successful), "--smoke"]) == 0
    assert (
        json.loads((successful / "manifest.json").read_text())["status"] == "complete"
    )
    assert not list(tmp_path.glob(".published.incomplete-*"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        driver.main(["--output", str(successful), "--smoke"])

    failed = tmp_path / "failed"

    def fail_run(args: argparse.Namespace, staging: Path):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(driver, "run_experiment", fail_run)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        driver.main(["--output", str(failed), "--smoke"])
    retained = list(tmp_path.glob(".failed.incomplete-*"))
    assert len(retained) == 1
    failure = json.loads((retained[0] / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["requested_output"] == str(failed.resolve())


def test_full_synthetic_smoke_writes_all_auditable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dates, truth, members = _synthetic_data()
    cache_file = tmp_path / "synthetic-members.npy"
    metadata_file = tmp_path / "synthetic-members.metadata.json"
    manifest_file = tmp_path / "synthetic-members.manifest.json"
    np.save(cache_file, members, allow_pickle=False)
    metadata_file.write_text("{}")
    manifest_file.write_text("{}")
    cache = frozen.MemberCache(
        members=np.load(cache_file, mmap_mode="r", allow_pickle=False),
        initializations=dates,
        latitude=np.asarray([1.0, 0.0]),
        longitude=np.asarray([70.0, 71.5, 73.0]),
        member_labels=np.arange(members.shape[1]),
        cache_root=tmp_path,
        members_path=cache_file,
        metadata_path=metadata_file,
        manifest_path=manifest_file,
    )
    weights = np.asarray([[1.0, 2.0, 1.0], [0.5, 1.5, 0.5]], dtype=np.float64)
    observations = frozen.ObservationBundle(
        weekly_truth=truth,
        weekly_climatology=np.zeros_like(truth),
        observation_fraction=np.ones((2, 3), dtype=np.float32),
        weights=weights,
        source_stores=("/synthetic/imd/2002.zarr", "/synthetic/imd/2020.zarr"),
    )
    cache_calls: list[bool] = []

    def load_cache(path: Path, *, allow_partial: bool = False):
        cache_calls.append(allow_partial)
        return cache

    monkeypatch.setattr(driver.frozen, "load_member_cache", load_cache)
    monkeypatch.setattr(
        driver.frozen, "load_imd_observations", lambda loaded: observations
    )
    daily_dates = np.arange(
        dates.min() - np.timedelta64(14, "D"),
        dates.max() + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    daily_values = np.zeros((daily_dates.size, 2, 3), dtype=np.float32)
    monkeypatch.setattr(
        driver,
        "load_daily_imd_for_lags",
        lambda loaded, stores: (daily_dates, daily_values),
    )
    monkeypatch.setattr(
        driver.frozen,
        "cache_provenance",
        lambda loaded: {
            "data_file": str(cache_file),
            "metadata_file": str(metadata_file),
            "manifest_file": str(manifest_file),
            "scope": "synthetic_test",
        },
    )
    output = tmp_path / "staging"
    output.mkdir()
    args = driver.build_parser().parse_args(
        [
            "--cache",
            str(cache_file),
            "--output",
            str(tmp_path / "final"),
            "--smoke",
            "--minimum-calendar-samples",
            "4",
            "--debias-spans",
            "14",
            "--cdf-chunk-size",
            "5",
            "--bootstrap-samples",
            "8",
        ]
    )
    driver.validate_args(args)
    result = driver.run_experiment(args, output)
    assert cache_calls == [False]
    assert result["status"] == "complete"
    assert result["contract"]["sealed_2025_target_opened"] is False
    assert result["methods"] == list(driver.METHODS)
    assert all(
        evidence["valid_probability_cdf"]
        for family in result["fitting"]["cdf_validity"].values()
        for evidence in family.values()
    )
    assert result["split_counts_selected"] == {
        "train": 12,
        "validation": 10,
        "test": 10,
    }
    expected = (
        "README.md",
        "manifest.json",
        "evaluation/scoring_support.npz",
        "models/pbc_fit.npz",
        "metrics/quintile_case_scores.csv",
        "metrics/weekwise_metrics.csv",
        "metrics/seasonal_weekwise_metrics.csv",
        "metrics/component_ablation_metrics.csv",
        "metrics/paired_block_bootstrap.csv",
        "metrics/semidecile_extreme_metrics.csv",
        "metrics/probability_bias.csv",
        "metrics/threshold_diagnostics.csv",
    )
    assert all((output / relative).is_file() for relative in expected)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mode"] == "smoke"
    assert manifest["fitting"]["quintile_selected_half_span_by_lead"] == [14] * 6
    assert set(manifest["artifact_sha256"]) >= set(expected) - {"manifest.json"}
    case_scores = pd.read_csv(output / "metrics/quintile_case_scores.csv")
    assert len(case_scores) == 10 * 6 * len(driver.METHODS)
    assert {
        "rpss_vs_nominal_climatology_case",
        "rpss_vs_training_empirical_climatology_case",
    }.issubset(case_scores)
    bootstrap = pd.read_csv(output / "metrics/paired_block_bootstrap.csv")
    assert set(bootstrap.lead_scope) == {"W1-W6", "W1", "W2", "W3", "W4", "W5", "W6"}
    assert len(bootstrap.loc[bootstrap.lead_scope == "W1-W6"]) == 5
    extremes = pd.read_csv(output / "metrics/semidecile_extreme_metrics.csv")
    assert "upper_5pct" in set(extremes.event)
    lower = extremes.loc[extremes.event == "lower_5pct"].iloc[0]
    assert lower.status.startswith("not_scored_partially_degenerate")
    threshold = pd.read_csv(output / "metrics/threshold_diagnostics.csv")
    assert {
        "zero_threshold_fraction",
        "duplicate_previous_threshold_fraction",
    }.issubset(threshold)
    assert not list(output.rglob("*.temporary"))
