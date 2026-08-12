"""Focused contracts for validation-only v3 ensemble evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fuxi_adapter import v3_evaluate


def _prediction_dataset(model_values, *, longitude=None):
    model_values = np.asarray(model_values, dtype=np.float32)
    shape = model_values.shape
    assert shape == (2, 6, 2, 2)
    longitude = (
        np.asarray(longitude, dtype=np.float32)
        if longitude is not None
        else np.array([75.0, 76.0], dtype=np.float32)
    )
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 20.0
    candidate = model_values.copy()
    candidate[:, :2] = (base + 0.9)[:, :2]
    return xr.Dataset(
        {
            "truth_imerg": (("init", "lead_week", "latitude", "longitude"), base + 1.0),
            "imerg_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                base + 0.25,
            ),
            "raw_fuxi": (("init", "lead_week", "latitude", "longitude"), base + 0.75),
            "log_bias_correction": (
                ("init", "lead_week", "latitude", "longitude"),
                base + 0.9,
            ),
            "late_lead_temporal_unet": (
                ("init", "lead_week", "latitude", "longitude"),
                candidate,
            ),
        },
        coords={
            "init": np.array(["2023-01-02", "2023-01-05"], dtype="datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": np.array([20.0, 21.0], dtype=np.float32),
            "longitude": longitude,
        },
        attrs={
            "split": "validation",
            "scientific_output": True,
            "config_sha256": "synthetic-hash",
        },
    )


def _write_success(run, seed, **updates):
    run.mkdir(parents=True)
    payload = {
        "status": "success",
        "smoke": False,
        "seed": seed,
        "model": "late_lead_temporal_unet",
        "validation_only": True,
        "test_predictions_evaluated": False,
        "config_sha256": "synthetic-hash",
    }
    payload.update(updates)
    (run / "SUCCESS.json").write_text(json.dumps(payload))


def _patch_prediction_stores(monkeypatch, datasets):
    by_run = {str(Path(run).resolve()): dataset for run, dataset in datasets.items()}

    def fake_open_zarr(path, *, consolidated):
        assert consolidated is True
        run = str(Path(path).resolve().parents[1])
        return by_run[run]

    monkeypatch.setattr(v3_evaluate.xr, "open_zarr", fake_open_zarr)


def test_equal_seed_ensemble_is_equal_weighted_and_order_independent(
    tmp_path, monkeypatch
):
    values = {
        42: np.full((2, 6, 2, 2), 1.0, dtype=np.float32),
        43: np.full((2, 6, 2, 2), 4.0, dtype=np.float32),
        44: np.full((2, 6, 2, 2), 10.0, dtype=np.float32),
    }
    runs_by_seed = {}
    datasets = {}
    for seed in (42, 43, 44):
        run = tmp_path / f"seed{seed}"
        _write_success(run, seed)
        runs_by_seed[seed] = run
        datasets[run] = _prediction_dataset(values[seed])
    _patch_prediction_stores(monkeypatch, datasets)

    ensemble, reference, successes = v3_evaluate._load_equal_seed_ensemble(
        [runs_by_seed[44], runs_by_seed[42], runs_by_seed[43]],
        "late_lead_temporal_unet",
        [42, 43, 44],
    )

    expected = np.mean(
        np.stack(
            [
                datasets[runs_by_seed[seed]]["late_lead_temporal_unet"].values
                for seed in (42, 43, 44)
            ]
        ).astype(np.float64),
        axis=0,
    ).astype(np.float32)
    assert np.array_equal(ensemble, expected)
    assert reference is datasets[runs_by_seed[44]]
    assert [int(item["seed"]) for item in successes] == [44, 42, 43]


def test_equal_seed_ensemble_preserves_identical_float32_members_bit_exactly(
    tmp_path, monkeypatch
):
    identical = np.linspace(0.01, 7.0, 2 * 6 * 2 * 2, dtype=np.float32).reshape(
        2, 6, 2, 2
    )
    runs = []
    datasets = {}
    for seed in (42, 43, 44):
        run = tmp_path / f"seed{seed}"
        _write_success(run, seed)
        runs.append(run)
        datasets[run] = _prediction_dataset(identical.copy())
    _patch_prediction_stores(monkeypatch, datasets)

    ensemble, _, _ = v3_evaluate._load_equal_seed_ensemble(
        runs, "late_lead_temporal_unet", [42, 43, 44]
    )

    assert np.array_equal(
        ensemble, datasets[runs[0]]["late_lead_temporal_unet"].values
    )


def test_equal_seed_ensemble_rejects_wrong_seed_set(tmp_path, monkeypatch):
    datasets = {}
    runs = []
    for index, seed in enumerate((42, 43, 43)):
        run = tmp_path / f"run{index}"
        _write_success(run, seed)
        runs.append(run)
        datasets[run] = _prediction_dataset(np.zeros((2, 6, 2, 2)))
    _patch_prediction_stores(monkeypatch, datasets)

    with pytest.raises(ValueError, match="do not match predeclared"):
        v3_evaluate._load_equal_seed_ensemble(
            runs, "late_lead_temporal_unet", [42, 43, 44]
        )


def test_equal_seed_ensemble_rejects_nonvalidation_run_claim(tmp_path, monkeypatch):
    runs = []
    datasets = {}
    for seed in (42, 43, 44):
        run = tmp_path / f"seed{seed}"
        _write_success(run, seed, validation_only=seed != 43)
        runs.append(run)
        datasets[run] = _prediction_dataset(np.zeros((2, 6, 2, 2)))
    _patch_prediction_stores(monkeypatch, datasets)

    with pytest.raises(ValueError, match="not marked validation-only"):
        v3_evaluate._load_equal_seed_ensemble(
            runs, "late_lead_temporal_unet", [42, 43, 44]
        )


def test_equal_seed_ensemble_rejects_week_one_two_anchor_drift(tmp_path, monkeypatch):
    runs = []
    datasets = {}
    for seed in (42, 43, 44):
        run = tmp_path / f"seed{seed}"
        _write_success(run, seed)
        runs.append(run)
        datasets[run] = _prediction_dataset(np.zeros((2, 6, 2, 2)))
    datasets[runs[-1]]["late_lead_temporal_unet"].values[0, 0, 0, 0] += 0.01
    _patch_prediction_stores(monkeypatch, datasets)

    with pytest.raises(ValueError, match="Weeks 1-2 are not exact"):
        v3_evaluate._load_equal_seed_ensemble(
            runs, "late_lead_temporal_unet", [42, 43, 44]
        )


@pytest.mark.parametrize("drift", ["coordinate", "baseline"])
def test_equal_seed_ensemble_rejects_member_identity_drift(
    tmp_path, monkeypatch, drift
):
    runs = []
    datasets = {}
    for seed in (42, 43, 44):
        run = tmp_path / f"seed{seed}"
        _write_success(run, seed)
        runs.append(run)
        datasets[run] = _prediction_dataset(np.zeros((2, 6, 2, 2)))
    if drift == "coordinate":
        datasets[runs[-1]] = _prediction_dataset(
            np.zeros((2, 6, 2, 2)), longitude=[75.0, 77.0]
        )
        match = "coordinate differs"
    else:
        datasets[runs[-1]]["truth_imerg"].values[0, 0, 0, 0] += 1.0
        match = "baseline differs"
    _patch_prediction_stores(monkeypatch, datasets)

    with pytest.raises(ValueError, match=match):
        v3_evaluate._load_equal_seed_ensemble(
            runs, "late_lead_temporal_unet", [42, 43, 44]
        )


def _gate_case_metrics(candidate_acc=0.33):
    rows = []
    for region in ("india", "northwest_india"):
        for lead in (3, 4, 5, 6):
            rows.extend(
                [
                    {
                        "predictor": "log_bias_correction",
                        "case_id": "case0",
                        "region": region,
                        "lead": lead,
                        "season": "JJA",
                        "acc": 0.30,
                        "rmse": 2.0,
                        "mae": 1.0,
                        "bias": -0.50,
                    },
                    {
                        "predictor": "candidate",
                        "case_id": "case0",
                        "region": region,
                        "lead": lead,
                        "season": "JJA",
                        "acc": candidate_acc,
                        "rmse": 1.90,
                        "mae": 0.99,
                        "bias": -0.52,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _gate_bootstraps():
    by_week = pd.DataFrame(
        [
            {
                "region": "india",
                "lead": lead,
                "metric": "acc",
                "predictor": "candidate",
                "baseline": "log_bias_correction",
                "mean_difference": 0.03,
            }
            for lead in (3, 4, 5, 6)
        ]
    )
    late = pd.DataFrame(
        [
            {
                "region": "india",
                "metric": "acc",
                "predictor": "candidate",
                "baseline": "log_bias_correction",
                "mean_difference": 0.03,
                "ci_lower": 0.005,
                "ci_upper": 0.055,
            }
        ]
    )
    return by_week, late


def test_development_gate_computes_all_predeclared_checks_and_is_nonconfirmatory():
    by_week, late = _gate_bootstraps()

    gate = v3_evaluate._development_gate(
        _gate_case_metrics(), by_week, late, "candidate"
    )

    assert gate["status"] == "passes_all_development_gates"
    assert gate["confirmatory"] is False
    assert gate["selection_split"] == "2023_validation"
    assert gate["candidate_minus_log_bias"] == pytest.approx(
        {"acc": 0.03, "rmse": -0.10, "mae": -0.01, "bias": -0.02}
    )
    assert all(gate["checks"].values())


def test_development_gate_fails_when_late_acc_gain_is_below_threshold():
    by_week, late = _gate_bootstraps()

    gate = v3_evaluate._development_gate(
        _gate_case_metrics(candidate_acc=0.319), by_week, late, "candidate"
    )

    assert gate["status"] == "fails_one_or_more_development_gates"
    assert gate["checks"]["late_acc_gain_at_least_0p02"] is False


def test_development_gate_requires_one_bootstrap_row_for_every_late_week():
    by_week, late = _gate_bootstraps()
    by_week = by_week.loc[by_week["lead"].ne(6)]

    with pytest.raises(ValueError, match="exactly Weeks 3-6"):
        v3_evaluate._development_gate(
            _gate_case_metrics(), by_week, late, "candidate"
        )


def _raw_config(output_root):
    return {
        "experiment_name": "synthetic_v3",
        "archive_root": "/unused/archive",
        "output_root": str(output_root),
        "train_years": [2020, 2021, 2022],
        "validation_years": [2023],
        "test_years": [2024],
        "verification_start_offset_days": 0,
        "verification_day_count": 42,
        "verification_interval_convention": "start_inclusive_end_exclusive",
        "non_overlapping_split_targets": True,
        "models": ["late_lead_temporal_unet"],
        "seeds": [42, 43, 44],
        "bootstrap_block_length": 2,
        "bootstrap_samples": 10,
    }


def _write_config_and_resolved_runs(tmp_path, config, *, mismatch_last=False):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    runs = []
    for seed in config["seeds"]:
        run = tmp_path / f"resolved_seed{seed}"
        run.mkdir()
        resolved = dict(config)
        if mismatch_last and seed == config["seeds"][-1]:
            resolved["bootstrap_samples"] += 1
        (run / "resolved_config.json").write_text(json.dumps(resolved))
        runs.append(run)
    return config_path, runs


def test_evaluator_rejects_resolved_config_hash_mismatch(tmp_path, monkeypatch):
    config = _raw_config(tmp_path / "output")
    config_path, runs = _write_config_and_resolved_runs(
        tmp_path, config, mismatch_last=True
    )
    stored = _prediction_dataset(np.zeros((2, 6, 2, 2)))
    stored.attrs["config_sha256"] = v3_evaluate.config_sha256(config)
    successes = [{"seed": seed} for seed in config["seeds"]]
    monkeypatch.setattr(
        v3_evaluate,
        "_load_equal_seed_ensemble",
        lambda *args, **kwargs: (
            np.zeros((2, 6, 2, 2)),
            stored,
            successes,
        ),
    )

    with pytest.raises(ValueError, match="configuration differs"):
        v3_evaluate.evaluate_validation_ensemble(config_path, runs)


def test_validation_workflow_collapses_late_leads_per_case_before_bootstrap(
    tmp_path, monkeypatch
):
    config = _raw_config(tmp_path / "output")
    config_path, runs = _write_config_and_resolved_runs(tmp_path, config)
    shape = (2, 6, 2, 2)
    stored = _prediction_dataset(np.ones(shape, dtype=np.float32))
    stored.attrs["config_sha256"] = v3_evaluate.config_sha256(config)
    successes = [{"seed": seed} for seed in config["seeds"]]
    monkeypatch.setattr(
        v3_evaluate,
        "_load_equal_seed_ensemble",
        lambda *args, **kwargs: (np.ones(shape, dtype=np.float32), stored, successes),
    )
    validation = SimpleNamespace(
        initializations=stored["init"].values.copy(),
        imerg_truth=stored["truth_imerg"].values.copy(),
        imerg_climatology=stored["imerg_climatology"].values.copy(),
        fuxi_mean=stored["raw_fuxi"].values.copy(),
    )
    train = SimpleNamespace(
        fuxi_mean=np.zeros(shape, dtype=np.float32),
        imerg_truth=np.ones(shape, dtype=np.float32),
        initializations=stored["init"].values.copy(),
    )
    synthetic_data = SimpleNamespace(
        validation=validation,
        train=train,
        latitude=stored["latitude"].values.copy(),
        longitude=stored["longitude"].values.copy(),
        area_weight_km2=np.ones((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        v3_evaluate,
        "load_adapter_data",
        lambda *args, **kwargs: synthetic_data,
    )
    monkeypatch.setattr(v3_evaluate, "fit_log_bias_correction", lambda *args: object())
    monkeypatch.setattr(
        v3_evaluate,
        "apply_log_bias_correction",
        lambda *args: stored["log_bias_correction"].values.copy(),
    )

    rows = []
    for predictor, offset in (
        ("late_lead_temporal_unet", 10.0),
        ("log_bias_correction", 0.0),
    ):
        for case_id in ("2023-01-02", "2023-01-05"):
            for lead in (3, 4, 5, 6):
                value = offset + float(lead)
                rows.append(
                    {
                        "predictor": predictor,
                        "case_id": case_id,
                        "region": "india",
                        "lead": lead,
                        "season": "JJA",
                        "acc": value,
                        "rmse": value + 1.0,
                        "mae": value + 2.0,
                        "bias": value + 3.0,
                        "negative_fraction": value + 4.0,
                    }
                )
    case_metrics = pd.DataFrame(rows)
    monkeypatch.setattr(
        v3_evaluate,
        "evaluate_prediction_set",
        lambda *args, **kwargs: (case_metrics, pd.DataFrame(), pd.DataFrame()),
    )
    bootstrap_calls = []

    def capture_bootstrap(frame, predictor, baseline, **kwargs):
        bootstrap_calls.append((frame.copy(), predictor, baseline, kwargs))
        return pd.DataFrame()

    monkeypatch.setattr(v3_evaluate, "paired_moving_block_bootstrap", capture_bootstrap)
    monkeypatch.setattr(v3_evaluate, "_development_gate", lambda *args: {"ok": True})
    monkeypatch.setattr(v3_evaluate, "write_metric_tables", lambda *args: None)
    monkeypatch.setattr(v3_evaluate, "write_prediction_store", lambda *args: None)
    monkeypatch.setattr(v3_evaluate, "_plot_summary", lambda *args: None)

    output = v3_evaluate.evaluate_validation_ensemble(config_path, runs)

    assert len(bootstrap_calls) == 2
    late_cases = bootstrap_calls[1][0]
    assert len(late_cases) == 4
    selected = late_cases.loc[
        late_cases["predictor"].eq("late_lead_temporal_unet")
        & late_cases["case_id"].eq("2023-01-02")
    ].iloc[0]
    assert selected["acc"] == pytest.approx(np.mean([13.0, 14.0, 15.0, 16.0]))
    assert bootstrap_calls[1][3]["group_columns"] == ("region",)

    manifest = json.loads((output / "ensemble_manifest.json").read_text())
    assert manifest["status"] == "validation_only_development"
    assert manifest["confirmatory"] is False
    assert manifest["test_predictions_evaluated"] is False
    assert manifest["equal_weight_ensemble"] is True
    assert manifest["config_sha256"] == v3_evaluate.config_sha256(config)
