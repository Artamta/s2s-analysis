"""Focused tests for the frozen neural-versus-PBC categorical evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fuxi_allseason_categorical_comparison as comparison
import fuxi_allseason_ensemble_calibration as frozen


def _sha(path: Path) -> str:
    return frozen.sha256_file(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def test_adjustment_loader_requires_hash_dates_seed_and_log_spread_identity(
    tmp_path: Path,
) -> None:
    dates = np.asarray(["2020-01-02", "2020-01-09"], dtype="datetime64[D]")
    shape = (2, 6, 27, 27)
    relative = "models/summary_only/seed_42/test_adjustments.npz"
    path = tmp_path / relative
    log_spread = np.full(shape, np.float32(0.2))
    _write_npz(
        path,
        initializations=dates,
        delta_log_location=np.zeros(shape, dtype=np.float32),
        log_spread=log_spread,
        spread_factor=np.exp(log_spread).astype(np.float32),
        seed=np.int64(42),
    )
    manifest = {"artifact_sha256": {relative: _sha(path)}}
    delta, spread, receipt = comparison.load_adjustment(
        tmp_path, manifest, "summary_only", 42, dates
    )
    assert delta.shape == shape
    np.testing.assert_allclose(spread, np.exp(np.float32(0.2)))
    assert receipt["sha256"] == _sha(path)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["spread_factor"] = np.ones(shape, dtype=np.float32)
    _write_npz(path, **payload)
    manifest["artifact_sha256"][relative] = _sha(path)
    with pytest.raises(comparison.ComparisonContractError, match="spread/log-spread"):
        comparison.load_adjustment(
            tmp_path, manifest, "summary_only", 42, dates
        )


def test_moment_reconstruction_uses_stored_fit_without_training(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    members = rng.gamma(1.2, 2.0, size=(2, 5, 6, 27, 27)).astype(np.float32)
    dates = np.asarray(["2020-01-02", "2020-07-02"], dtype="datetime64[D]")
    relative = "models/moment_calibration_fit.npz"
    path = tmp_path / relative
    _write_npz(
        path,
        delta_log_location=np.zeros((6, 12, 27, 27), dtype=np.float32),
        spread_factor=np.ones((6, 12), dtype=np.float32),
        shrinkage=np.float32(10.0),
    )
    manifest = {
        "artifact_sha256": {relative: _sha(path)},
        "moment_calibration": {"location_shrinkage": 10.0},
    }
    corrected, receipt = comparison.reconstruct_moment(
        tmp_path, manifest, members, dates
    )
    np.testing.assert_allclose(corrected, members, rtol=1.0e-6, atol=1.0e-6)
    assert receipt["sha256"] == _sha(path)


def test_categorical_scoring_reports_rps_upper_brier_and_bias() -> None:
    truth = np.asarray([[[[1.0]]], [[[3.0]]]], dtype=np.float32)
    members = np.repeat(truth[:, None], 5, axis=1)
    quintile_thresholds = np.broadcast_to(
        np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float32).reshape(1, 1, 4, 1, 1),
        (2, 1, 4, 1, 1),
    ).copy()
    semidecile_thresholds = np.broadcast_to(
        np.linspace(0.2, 4.0, 19, dtype=np.float32).reshape(1, 1, 19, 1, 1),
        (2, 1, 19, 1, 1),
    ).copy()
    from fuxi_pbc_core import observation_cdf

    quintile_observed = observation_cdf(truth, quintile_thresholds)
    semidecile_observed = observation_cdf(truth, semidecile_thresholds)
    scored = comparison.score_ensemble(
        members,
        quintile_thresholds,
        semidecile_thresholds,
        quintile_observed,
        semidecile_observed,
        np.ones((1, 1), dtype=np.float64),
        chunk_size=1,
    )
    np.testing.assert_allclose(scored.quintile_rps, 0.0)
    np.testing.assert_allclose(scored.upper_brier, 0.0)
    np.testing.assert_allclose(scored.quintile_probability_bias, 0.0)
    np.testing.assert_allclose(scored.semidecile_probability_bias, 0.0)


def _seed_case_frame() -> pd.DataFrame:
    records = []
    dates = ("2020-01-02", "2020-01-09")
    for method_index, method in enumerate(comparison.NEURAL_CONFIGURATIONS):
        for seed_index, seed in enumerate(comparison.SEEDS):
            for date in dates:
                for lead in (1, 2):
                    records.append(
                        {
                            "method": method,
                            "method_label": comparison.METHOD_LABELS[method],
                            "seed": seed,
                            "initialization": date,
                            "lead_week": lead,
                            "verification_start": date,
                            "verification_midpoint": date,
                            "verification_end": date,
                            "season": "DJF",
                            "rps": float(method_index + seed_index + lead),
                            "nominal_climatology_rps": 10.0,
                            "training_empirical_climatology_rps": 8.0,
                            "rpss_vs_nominal_climatology_case": 0.0,
                            "rpss_vs_training_empirical_climatology_case": 0.0,
                        }
                    )
    return pd.DataFrame.from_records(records)


def test_seed_aggregation_averages_scores_only_and_recomputes_skill() -> None:
    frame = _seed_case_frame()
    averaged = comparison.average_seed_case_scores(frame)
    assert len(averaged) == 2 * 2 * 2
    row = averaged.loc[
        (averaged.method == "summary_only") & (averaged.lead_week == 1)
    ].iloc[0]
    assert row.rps == pytest.approx(2.0)
    assert row.rpss_vs_nominal_climatology_case == pytest.approx(0.8)
    assert row.rpss_vs_training_empirical_climatology_case == pytest.approx(0.75)
    assert set(averaged.seed) == {"mean_of_seed_scores_42_43_44"}


def test_paired_bootstrap_has_pooled_and_weekwise_neural_vs_pbc_raw() -> None:
    records = []
    dates = pd.date_range("2020-01-02", periods=20, freq="7D").strftime("%Y-%m-%d")
    for method_index, method in enumerate(comparison.COMPARISON_METHODS):
        for date in dates:
            for lead in range(1, 7):
                records.append(
                    {
                        "method": method,
                        "initialization": date,
                        "lead_week": lead,
                        "rps": 1.0 + 0.05 * method_index + 0.01 * lead,
                    }
                )
    result = comparison.paired_block_bootstrap(
        pd.DataFrame.from_records(records), samples=20, block_length=5
    )
    assert len(result) == 7 * 7
    assert set(result.lead_scope) == {
        "W1-W6",
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "W6",
    }
    pooled = result.loc[result.lead_scope == "W1-W6"]
    assert set(zip(pooled.method, pooled.baseline)) >= {
        ("summary_only", "pbc_combined"),
        ("location_spread", "raw_fuxi_categorical"),
    }


def test_pbc_raw_identity_detects_any_case_score_drift() -> None:
    raw = pd.DataFrame(
        {
            "method": ["raw_fuxi_categorical"],
            "initialization": ["2020-01-02"],
            "lead_week": [1],
            "rps": [0.5],
            "nominal_climatology_rps": [0.8],
            "training_empirical_climatology_rps": [0.7],
        }
    )
    rows = []
    for method in comparison.PBC_METHODS:
        item = raw.iloc[0].to_dict()
        item["method"] = method
        rows.append(item)
    pbc = pd.DataFrame(rows)
    receipts = comparison.validate_pbc_case_scores(
        pbc,
        np.asarray(["2020-01-02"], dtype="datetime64[D]"),
        raw,
    )
    assert receipts["maximum_absolute_rps_difference"] == 0.0
    pbc.loc[pbc.method == "raw_fuxi_categorical", "rps"] += 0.01
    with pytest.raises(comparison.ComparisonContractError, match="raw identity"):
        comparison.validate_pbc_case_scores(
            pbc,
            np.asarray(["2020-01-02"], dtype="datetime64[D]"),
            raw,
        )


def test_atomic_main_publishes_fresh_output_and_retains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pbc_manifest = tmp_path / "pbc" / "manifest.json"
    pbc_manifest.parent.mkdir()
    pbc_manifest.write_text("{}")
    output = tmp_path / "published"

    def fake_run(args: argparse.Namespace, staging: Path):
        comparison._atomic_json(
            staging / "manifest.json",
            {"experiment": comparison.EXPERIMENT, "status": "complete"},
        )
        return {"status": "complete"}

    monkeypatch.setattr(comparison, "run_comparison", fake_run)
    assert (
        comparison.main(
            ["--pbc-manifest", str(pbc_manifest), "--output", str(output)]
        )
        == 0
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        comparison.main(
            ["--pbc-manifest", str(pbc_manifest), "--output", str(output)]
        )

    failed = tmp_path / "failed"

    def fail(args: argparse.Namespace, staging: Path):
        raise RuntimeError("synthetic comparison failure")

    monkeypatch.setattr(comparison, "run_comparison", fail)
    with pytest.raises(RuntimeError, match="synthetic comparison failure"):
        comparison.main(
            ["--pbc-manifest", str(pbc_manifest), "--output", str(failed)]
        )
    retained = list(tmp_path.glob(".failed.incomplete-*"))
    assert len(retained) == 1
    assert json.loads((retained[0] / "failure.json").read_text())["status"] == "failed"
