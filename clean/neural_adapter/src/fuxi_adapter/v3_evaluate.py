"""Equal-seed ensemble evaluation for validation-only v3 development runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import config_sha256, load_config, write_json
from .data import DataPaths, load_adapter_data
from .baselines import apply_log_bias_correction, fit_log_bias_correction
from .evaluation import (
    evaluate_prediction_set,
    write_metric_tables,
    write_prediction_store,
)
from .metrics import paired_moving_block_bootstrap
from .plotting import plot_metric_by_lead


def _load_success(run: Path) -> Mapping[str, object]:
    with (run / "SUCCESS.json").open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("status") != "success" or result.get("smoke"):
        raise ValueError(f"run is not a completed scientific run: {run}")
    return result


def _load_equal_seed_ensemble(
    runs: Sequence[Path],
    variable: str,
    expected_seeds: Sequence[int],
    *,
    require_validation_only: bool = True,
) -> Tuple[np.ndarray, xr.Dataset, List[Mapping[str, object]]]:
    if len(runs) != len(expected_seeds):
        raise ValueError("one completed run is required for every predeclared seed")
    datasets: List[xr.Dataset] = []
    successes: List[Mapping[str, object]] = []
    for run in runs:
        run = Path(run).resolve()
        success = _load_success(run)
        if success.get("model") != variable:
            raise ValueError(f"run model does not match {variable}: {run}")
        if require_validation_only and success.get("validation_only") is not True:
            raise ValueError(f"run is not marked validation-only: {run}")
        if require_validation_only:
            no_test_evaluation = (
                success.get("test_predictions_evaluated") is False
                or success.get("test_data_accessed") is False
            )
            if not no_test_evaluation:
                raise ValueError(f"run does not affirm validation-only evaluation: {run}")
        successes.append(success)
        dataset = xr.open_zarr(
            str(run / "predictions" / "validation.zarr"), consolidated=True
        ).load()
        if dataset.attrs.get("split") != "validation":
            raise ValueError(f"prediction store is not validation: {run}")
        if dataset.attrs.get("scientific_output") not in (True, "True"):
            raise ValueError(f"prediction store is not marked scientific: {run}")
        if variable not in dataset:
            raise ValueError(f"prediction store lacks {variable}: {run}")
        datasets.append(dataset)
    observed_seeds = sorted(int(values["seed"]) for values in successes)
    if observed_seeds != sorted(int(value) for value in expected_seeds):
        raise ValueError(
            f"run seeds {observed_seeds} do not match predeclared {sorted(expected_seeds)}"
        )
    reference = datasets[0]
    reference_config_hash = reference.attrs.get("config_sha256")
    if not reference_config_hash:
        raise ValueError("prediction store lacks a configuration hash")
    for dataset in datasets[1:]:
        if dataset.attrs.get("config_sha256") != reference_config_hash:
            raise ValueError("prediction stores have different configuration hashes")
        for coordinate in ("init", "lead_week", "latitude", "longitude"):
            if not np.array_equal(dataset[coordinate].values, reference[coordinate].values):
                raise ValueError(f"run coordinate differs: {coordinate}")
        for baseline in ("truth_imerg", "imerg_climatology", "raw_fuxi", "log_bias_correction"):
            if not np.array_equal(dataset[baseline].values, reference[baseline].values, equal_nan=True):
                raise ValueError(f"run baseline differs: {baseline}")
    for dataset in datasets:
        candidate = np.asarray(dataset[variable].values, dtype=np.float32)
        if not np.isfinite(candidate).all() or np.any(candidate < 0.0):
            raise ValueError(f"{variable} predictions must be finite and nonnegative")
        baseline = np.asarray(dataset["log_bias_correction"].values, dtype=np.float32)
        if require_validation_only:
            finite = np.isfinite(baseline[:, :2])
            if not np.array_equal(candidate[:, :2][finite], baseline[:, :2][finite]):
                raise ValueError("Weeks 1-2 are not exact training-only log-bias identity")
    members = [np.asarray(dataset[variable].values, dtype=np.float32) for dataset in datasets]
    ensemble = np.mean(np.stack(members).astype(np.float64), axis=0).astype(np.float32)
    return ensemble, reference, successes


def _plot_summary(summary: pd.DataFrame, output: Path) -> None:
    plotting = summary.copy()
    for metric in ("acc", "rmse", "mae", "bias"):
        plotting[metric] = plotting[f"{metric}_mean"]
        plot_metric_by_lead(
            plotting,
            metric,
            output / f"validation_{metric}_india.png",
            split="validation",
            region="india",
        )


def _metric_means(
    case_metrics: pd.DataFrame,
    predictor: str,
    *,
    region: str = "india",
    leads: Sequence[int] = (3, 4, 5, 6),
    season: Optional[str] = None,
) -> Dict[str, float]:
    selected = case_metrics.loc[
        case_metrics["predictor"].eq(predictor)
        & case_metrics["region"].eq(region)
        & case_metrics["lead"].isin(leads)
    ]
    if season is not None:
        selected = selected.loc[selected["season"].eq(season)]
    if selected.empty:
        raise ValueError("requested metric subset is empty")
    selected = selected.groupby(["predictor", "case_id", "region"], as_index=False)[
        ["acc", "rmse", "mae", "bias"]
    ].mean()
    return {
        metric: float(selected[metric].mean())
        for metric in ("acc", "rmse", "mae", "bias")
    }


def _development_gate(
    case_metrics: pd.DataFrame,
    by_week_bootstrap: pd.DataFrame,
    late_bootstrap: pd.DataFrame,
    predictor: str,
) -> Dict[str, object]:
    baseline = "log_bias_correction"
    candidate = _metric_means(case_metrics, predictor)
    bias = _metric_means(case_metrics, baseline)
    differences = {name: candidate[name] - bias[name] for name in candidate}
    interval_rows = late_bootstrap.loc[
        late_bootstrap["region"].eq("india")
        & late_bootstrap["metric"].eq("acc")
        & late_bootstrap["predictor"].eq(predictor)
        & late_bootstrap["baseline"].eq(baseline)
    ]
    if len(interval_rows) != 1:
        raise ValueError("late ACC bootstrap must contain exactly one India comparison")
    acc_interval = interval_rows.iloc[0]
    lead_acc = by_week_bootstrap.loc[
        by_week_bootstrap["region"].eq("india")
        & by_week_bootstrap["metric"].eq("acc")
        & by_week_bootstrap["lead"].isin([3, 4, 5, 6])
        & by_week_bootstrap["predictor"].eq(predictor)
        & by_week_bootstrap["baseline"].eq(baseline)
    ]
    if sorted(lead_acc["lead"].astype(int).tolist()) != [3, 4, 5, 6]:
        raise ValueError("weekly ACC bootstrap must contain exactly Weeks 3-6")
    jja_candidate = _metric_means(case_metrics, predictor, season="JJA")
    jja_bias = _metric_means(case_metrics, baseline, season="JJA")
    region_rows = []
    for region in sorted(set(case_metrics["region"]) - {"india"}):
        regional_candidate = _metric_means(case_metrics, predictor, region=region)
        regional_bias = _metric_means(case_metrics, baseline, region=region)
        region_rows.append(
            {
                "region": region,
                "delta_acc": regional_candidate["acc"] - regional_bias["acc"],
                "relative_mae": regional_candidate["mae"] / regional_bias["mae"],
            }
        )
    checks = {
        "late_acc_gain_at_least_0p02": differences["acc"] >= 0.02,
        "late_acc_block_ci_lower_positive": float(acc_interval["ci_lower"]) > 0.0,
        "no_late_lead_acc_drop_below_minus_0p01": bool(
            (lead_acc["mean_difference"] >= -0.01).all()
        ),
        "late_rmse_nonworse": differences["rmse"] <= 0.0,
        "late_mae_nonworse": differences["mae"] <= 0.0,
        "absolute_bias_not_worse_by_0p10": abs(candidate["bias"]) <= abs(bias["bias"]) + 0.10,
        "jja_acc_nonworse": jja_candidate["acc"] >= jja_bias["acc"],
        "jja_rmse_within_1_percent": jja_candidate["rmse"] <= 1.01 * jja_bias["rmse"],
        "jja_mae_within_1_percent": jja_candidate["mae"] <= 1.01 * jja_bias["mae"],
        "no_region_acc_drop_below_minus_0p02": all(
            row["delta_acc"] >= -0.02 for row in region_rows
        ),
        "no_region_mae_worse_than_2_percent": all(
            row["relative_mae"] <= 1.02 for row in region_rows
        ),
    }
    return {
        "status": "passes_all_development_gates" if all(checks.values()) else "fails_one_or_more_development_gates",
        "confirmatory": False,
        "selection_split": "2023_validation",
        "candidate_late_weeks_3_to_6": candidate,
        "log_bias_late_weeks_3_to_6": bias,
        "candidate_minus_log_bias": differences,
        "late_acc_block_interval": {
            "difference": float(acc_interval["mean_difference"]),
            "lower": float(acc_interval["ci_lower"]),
            "upper": float(acc_interval["ci_upper"]),
        },
        "jja_candidate": jja_candidate,
        "jja_log_bias": jja_bias,
        "regional_checks": region_rows,
        "checks": checks,
    }


def evaluate_validation_ensemble(
    config_path: Path,
    runs: Sequence[Path],
    *,
    reference_runs: Sequence[Path] = (),
) -> Path:
    config = load_config(config_path)
    model_name = str(config["models"][0])
    ensemble, stored, successes = _load_equal_seed_ensemble(
        runs, model_name, config["seeds"]
    )
    for run, success in zip(runs, successes):
        with (Path(run) / "resolved_config.json").open("r", encoding="utf-8") as stream:
            run_hash = config_sha256(json.load(stream))
            if run_hash != config_sha256(config):
                raise ValueError(f"run configuration differs from requested config: {run}")
            if stored.attrs.get("config_sha256") != run_hash:
                raise ValueError("prediction-store config hash differs from resolved config")
    predictions: Dict[str, np.ndarray] = {
        "raw_fuxi": np.asarray(stored["raw_fuxi"].values, dtype=np.float32),
        "imerg_climatology": np.asarray(
            stored["imerg_climatology"].values, dtype=np.float32
        ),
        "log_bias_correction": np.asarray(
            stored["log_bias_correction"].values, dtype=np.float32
        ),
        model_name: ensemble,
    }
    if reference_runs:
        reference, reference_store, _ = _load_equal_seed_ensemble(
            reference_runs,
            "residual_unet",
            (42, 43, 44),
            require_validation_only=False,
        )
        for run in reference_runs:
            with (Path(run) / "resolved_config.json").open(
                "r", encoding="utf-8"
            ) as stream:
                if reference_store.attrs.get("config_sha256") != config_sha256(
                    json.load(stream)
                ):
                    raise ValueError("v2 reference store hash differs from resolved config")
        for coordinate in ("init", "lead_week", "latitude", "longitude"):
            if not np.array_equal(reference_store[coordinate].values, stored[coordinate].values):
                raise ValueError(f"v2 reference coordinate differs: {coordinate}")
        for baseline in ("truth_imerg", "imerg_climatology", "raw_fuxi", "log_bias_correction"):
            if not np.array_equal(
                reference_store[baseline].values,
                stored[baseline].values,
                equal_nan=True,
            ):
                raise ValueError(f"v2 reference baseline differs: {baseline}")
        predictions["v2_residual_unet"] = reference

    data = load_adapter_data(DataPaths(Path(config["archive_root"])), strict=True)
    if not np.array_equal(data.validation.initializations.astype("datetime64[ns]"), stored["init"].values):
        raise ValueError("saved validation initializations differ from audited archive")
    if not np.array_equal(data.latitude, stored["latitude"].values):
        raise ValueError("saved latitude differs from audited archive")
    if not np.array_equal(data.longitude, stored["longitude"].values):
        raise ValueError("saved longitude differs from audited archive")
    if not np.array_equal(stored["lead_week"].values, np.arange(1, 7)):
        raise ValueError("saved lead weeks are not exactly 1-6")
    expected_baselines = {
        "truth_imerg": data.validation.imerg_truth,
        "imerg_climatology": data.validation.imerg_climatology,
        "raw_fuxi": data.validation.fuxi_mean,
    }
    fitted_bias = fit_log_bias_correction(
        data.train.fuxi_mean,
        data.train.imerg_truth,
        data.train.initializations,
        data.area_weight_km2 > 0.0,
    )
    expected_baselines["log_bias_correction"] = apply_log_bias_correction(
        data.validation.fuxi_mean,
        data.validation.initializations,
        fitted_bias,
    )
    for name, expected in expected_baselines.items():
        if not np.array_equal(stored[name].values, expected, equal_nan=True):
            raise ValueError(f"saved {name} differs from independently rebuilt archive value")
    case_metrics, summary, seasonal = evaluate_prediction_set(
        data.validation, data, predictions
    )
    block_length = int(config.get("bootstrap_block_length", 13))
    n_resamples = int(config.get("bootstrap_samples", 2000))
    by_week = paired_moving_block_bootstrap(
        case_metrics,
        model_name,
        "log_bias_correction",
        metric_columns=("acc", "rmse", "mae", "bias"),
        group_columns=("lead", "region"),
        block_length=block_length,
        n_resamples=n_resamples,
        seed=42,
    )
    late_cases = (
        case_metrics.loc[case_metrics["lead"].isin([3, 4, 5, 6])]
        .groupby(["predictor", "case_id", "region"], as_index=False)[
            ["acc", "rmse", "mae", "bias", "negative_fraction"]
        ]
        .mean()
    )
    late_bootstrap = paired_moving_block_bootstrap(
        late_cases,
        model_name,
        "log_bias_correction",
        metric_columns=("acc", "rmse", "mae", "bias"),
        group_columns=("region",),
        block_length=block_length,
        n_resamples=n_resamples,
        seed=42,
    )
    gate = _development_gate(case_metrics, by_week, late_bootstrap, model_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(config["output_root"])
        / "validation_results"
        / f"{config['experiment_name']}__{timestamp}"
    )
    (output / "metrics").mkdir(parents=True)
    (output / "figures").mkdir()
    write_metric_tables(output / "metrics", case_metrics, summary, seasonal)
    by_week.to_csv(output / "metrics" / "paired_block_bootstrap_by_week.csv", index=False)
    late_bootstrap.to_csv(
        output / "metrics" / "paired_block_bootstrap_late_weeks_3_to_6.csv",
        index=False,
    )
    write_json(output / "development_gate.json", gate)
    write_json(
        output / "ensemble_manifest.json",
        {
            "status": "validation_only_development",
            "confirmatory": False,
            "config_sha256": config_sha256(config),
            "model": model_name,
            "seeds": sorted(int(item["seed"]) for item in successes),
            "equal_weight_ensemble": True,
            "run_directories": [str(Path(run).resolve()) for run in runs],
            "reference_runs": [str(Path(run).resolve()) for run in reference_runs],
            "test_predictions_evaluated": False,
            "note": "2023 is a model-selection split; intervals are developmental, not confirmatory",
        },
    )
    write_prediction_store(
        output / "predictions.zarr",
        data.validation,
        data,
        predictions,
        {
            "split": "2023_validation",
            "status": "development_only",
            "test_predictions_evaluated": False,
            "config_sha256": config_sha256(config),
        },
    )
    _plot_summary(summary, output / "figures")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--reference-runs", nargs="*", default=())
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = evaluate_validation_ensemble(
        Path(args.config),
        [Path(value) for value in args.runs],
        reference_runs=[Path(value) for value in args.reference_runs],
    )
    print(f"V3 VALIDATION ENSEMBLE SUCCESS: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_validation_ensemble", "main"]
