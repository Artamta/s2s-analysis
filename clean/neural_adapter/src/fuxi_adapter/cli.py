"""Command-line workflow for auditing, developing, freezing, and testing."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .artifacts import (
    create_run_directory,
    freeze_development_run,
    initialize_run,
    load_unused_freeze,
    mark_failure,
    mark_success,
    mark_test_consumed,
)
from .baselines import apply_log_bias_correction, fit_log_bias_correction
from .config import config_sha256, load_config, write_json
from .data import (
    DataPaths,
    fit_normalization,
    load_adapter_data,
    make_model_arrays,
    reconstruct_precipitation,
)
from .evaluation import (
    evaluate_prediction_set,
    write_metric_tables,
    write_prediction_store,
)
from .metrics import paired_moving_block_bootstrap
from .models import build_model, count_parameters
from .plotting import plot_mean_maps, plot_metric_by_lead, plot_training_history
from .training import predict, set_deterministic_seed, train_model


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no assigned GPU is visible")
    return value


def _data(config: Mapping[str, Any]):
    return load_adapter_data(DataPaths(Path(config["archive_root"])), strict=True)


def _split_manifest(data) -> Dict[str, Any]:
    return {
        "audit": data.audit.to_dict(),
        "policy": (
            "half-open 42-day verification interval ends at or before next split issuance"
        ),
        "train_initializations": [
            np.datetime_as_string(value, unit="D") for value in data.train.initializations
        ],
        "validation_initializations": [
            np.datetime_as_string(value, unit="D")
            for value in data.validation.initializations
        ],
        "test_initializations_sha256_only_until_locked_evaluation": True,
    }


def _validation_predictions(data, neural_prediction: np.ndarray, model_name: str):
    fitted_bias = fit_log_bias_correction(
        data.train.fuxi_mean,
        data.train.imerg_truth,
        data.train.initializations,
        data.area_weight_km2 > 0,
    )
    return {
        "raw_fuxi": data.validation.fuxi_mean,
        "imerg_climatology": data.validation.imerg_climatology,
        "log_bias_correction": apply_log_bias_correction(
            data.validation.fuxi_mean,
            data.validation.initializations,
            fitted_bias,
        ),
        model_name: neural_prediction,
    }


def _plot_summaries(summary, figures: Path, split: str) -> None:
    plotting = summary.copy()
    for metric in ("acc", "rmse", "mae", "bias"):
        plotting[metric] = plotting[f"{metric}_mean"]
        plot_metric_by_lead(
            plotting,
            metric,
            figures / f"{split}_{metric}_india.png",
            split=split,
            region="india",
        )


def _build_seeded_model(
    model_name: str,
    *,
    seed: int,
    in_channels: int,
    base_channels: int,
    dropout: float,
):
    """Construct a model only after fixing every training random generator.

    ``train_model`` seeds the data order and stochastic layers again at the
    start of optimization.  This earlier call is separately required because
    PyTorch initializes model weights during construction.
    """

    set_deterministic_seed(seed)
    return build_model(
        model_name,
        in_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,
    )


def command_audit(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    print("Loading and auditing FuXi/IMERG archive...", flush=True)
    data = _data(config)
    payload = {
        "audit": data.audit.to_dict(),
        "sources": data.source_manifest,
        "config_sha256": config_sha256(config),
    }
    output = (
        Path(args.output)
        if args.output
        else Path(config["output_root"])
        / f"data_audit_{config['experiment_name']}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(json.dumps(payload, indent=2, default=str))
    print(f"Audit written to {output}", flush=True)
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    if args.model not in config["models"]:
        raise ValueError(f"model {args.model!r} is not enabled by the configuration")
    if args.seed not in config["seeds"] and not args.smoke:
        raise ValueError(f"seed {args.seed} is not frozen in the configuration")
    device = _device(args.device)
    print(f"Loading audited data for {args.model}, seed {args.seed}...", flush=True)
    data = _data(config)
    stats = fit_normalization(data.train, data.area_weight_km2)
    train_arrays = make_model_arrays(
        data.train, stats, data.latitude, data.longitude, data.area_weight_km2
    )
    validation_arrays = make_model_arrays(
        data.validation, stats, data.latitude, data.longitude, data.area_weight_km2
    )
    model_base_channels = 4 if args.smoke else int(config["base_channels"])
    run_model_name = args.model + ("_smoke" if args.smoke else "")
    run_directory = create_run_directory(
        Path(config["output_root"]), config["experiment_name"], run_model_name, args.seed
    )
    print(f"Run directory: {run_directory}", flush=True)
    initialize_run(run_directory, config, args.model, args.seed, _split_manifest(data))
    stats.save_json(run_directory / "normalization.json")
    write_json(run_directory / "data_audit.json", data.audit.to_dict())
    write_json(run_directory / "source_manifest.json", data.source_manifest)
    model = _build_seeded_model(
        args.model,
        seed=args.seed,
        in_channels=train_arrays.inputs.shape[2],
        base_channels=model_base_channels,
        dropout=float(config["dropout"]),
    )
    try:
        train_features = train_arrays.inputs[:8] if args.smoke else train_arrays.inputs
        train_target = train_arrays.target[:8] if args.smoke else train_arrays.target
        validation_features = (
            validation_arrays.inputs[:4] if args.smoke else validation_arrays.inputs
        )
        validation_target = (
            validation_arrays.target[:4] if args.smoke else validation_arrays.target
        )
        result = train_model(
            model,
            train_features,
            train_target,
            validation_features,
            validation_target,
            data.area_weight_km2,
            run_directory,
            seed=args.seed,
            device=device,
            batch_size=min(int(config["batch_size"]), len(train_features)),
            max_epochs=2 if args.smoke else int(config["max_epochs"]),
            patience=2 if args.smoke else int(config["early_stopping_patience"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            beta=float(config["smooth_l1_beta"]),
            num_workers=int(config["num_workers"]),
            use_amp=bool(config["use_amp"]),
        )
        validation_residual = predict(
            model,
            validation_features,
            device=device,
            batch_size=int(config["batch_size"]),
            use_amp=bool(config["use_amp"]),
        )
        validation_split = data.validation
        if args.smoke:
            # Smoke predictions are intentionally not scientific artifacts.
            validation_split = type(data.validation)(
                **{
                    field: (
                        getattr(data.validation, field)[:4]
                        if isinstance(getattr(data.validation, field), np.ndarray)
                        and getattr(data.validation, field).shape[:1] == (data.validation.sample_count,)
                        else getattr(data.validation, field)
                    )
                    for field in data.validation.__dataclass_fields__
                }
            )
        neural = reconstruct_precipitation(validation_split, validation_residual, stats)
        predictions = _validation_predictions(data, neural, args.model)
        if args.smoke:
            predictions = {
                key: value[:4] if value.shape[0] == data.validation.sample_count else value
                for key, value in predictions.items()
            }
        case_metrics, summary, seasonal = evaluate_prediction_set(
            validation_split, data, predictions
        )
        write_metric_tables(run_directory / "metrics", case_metrics, summary, seasonal)
        write_prediction_store(
            run_directory / "predictions" / "validation.zarr",
            validation_split,
            data,
            predictions,
            {
                "split": "validation",
                "model": args.model,
                "seed": args.seed,
                "config_sha256": config_sha256(config),
                "scientific_output": not args.smoke,
            },
        )
        plot_training_history(
            result.history, run_directory / "figures" / "training_history.png"
        )
        _plot_summaries(summary, run_directory / "figures", "validation")
        success = {
            "model": args.model,
            "seed": args.seed,
            "smoke": args.smoke,
            "device": device,
            "base_channels": model_base_channels,
            "dropout": float(config["dropout"]),
            "in_channels": int(train_arrays.inputs.shape[2]),
            "channel_names": list(train_arrays.channel_names),
            "parameter_count": count_parameters(model),
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "elapsed_seconds": result.elapsed_seconds,
        }
        mark_success(run_directory, success)
        print(json.dumps(success, indent=2), flush=True)
        print(f"SUCCESS: {run_directory}", flush=True)
        return 0
    except Exception as exc:
        mark_failure(run_directory, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        print(f"FAILED: {run_directory}", file=sys.stderr, flush=True)
        raise


def command_freeze(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = freeze_development_run(output, config, [Path(value) for value in args.runs])
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def _load_success(run_directory: Path) -> Dict[str, Any]:
    with (run_directory / "SUCCESS.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def command_test(args: argparse.Namespace) -> int:
    freeze_path = Path(args.frozen_manifest).resolve()
    manifest = load_unused_freeze(freeze_path)
    config = dict(manifest["config"])
    if manifest["config_sha256"] != config_sha256(config):
        raise RuntimeError("frozen configuration hash is invalid")
    device = _device(args.device)
    print("Loading locked 2024 test data...", flush=True)
    data = _data(config)
    fitted_bias = fit_log_bias_correction(
        data.train.fuxi_mean,
        data.train.imerg_truth,
        data.train.initializations,
        data.area_weight_km2 > 0,
    )
    predictions: Dict[str, np.ndarray] = {
        "raw_fuxi": data.test.fuxi_mean,
        "imerg_climatology": data.test.imerg_climatology,
        "log_bias_correction": apply_log_bias_correction(
            data.test.fuxi_mean, data.test.initializations, fitted_bias
        ),
    }
    by_model: Dict[str, List[np.ndarray]] = {}
    for frozen_run in manifest["runs"]:
        run_directory = Path(frozen_run["run_directory"])
        success = _load_success(run_directory)
        if success.get("smoke"):
            raise RuntimeError("smoke runs cannot consume the locked test set")
        with Path(frozen_run["source_manifest"]).open("r", encoding="utf-8") as stream:
            frozen_sources = json.load(stream)
        if frozen_sources != data.source_manifest:
            raise RuntimeError(
                f"live source metadata differs from the frozen run: {run_directory}"
            )
        with Path(frozen_run["normalization"]).open("r", encoding="utf-8") as stream:
            from .data import NormalizationStats

            stats = NormalizationStats.from_dict(json.load(stream))
        test_arrays = make_model_arrays(
            data.test, stats, data.latitude, data.longitude, data.area_weight_km2
        )
        model = _build_seeded_model(
            success["model"],
            seed=int(success["seed"]),
            in_channels=int(success["in_channels"]),
            base_channels=int(success["base_channels"]),
            dropout=float(success["dropout"]),
        )
        checkpoint = torch.load(frozen_run["checkpoint"], map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        residual = predict(
            model,
            test_arrays.inputs,
            device=device,
            batch_size=int(config["batch_size"]),
            use_amp=bool(config["use_amp"]),
        )
        corrected = reconstruct_precipitation(data.test, residual, stats)
        seed_name = f"{success['model']}_seed{success['seed']}"
        predictions[seed_name] = corrected
        by_model.setdefault(success["model"], []).append(corrected)
    for model_name, members in by_model.items():
        predictions[model_name] = np.mean(np.stack(members), axis=0).astype(np.float32)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_directory = (
        Path(config["output_root"])
        / "locked_tests"
        / f"{freeze_path.stem}__{timestamp}"
    )
    result_directory.mkdir(parents=True)
    (result_directory / "metrics").mkdir()
    (result_directory / "figures").mkdir()
    write_json(result_directory / "freeze_manifest_used.json", manifest)
    write_json(result_directory / "data_audit.json", data.audit.to_dict())
    case_metrics, summary, seasonal = evaluate_prediction_set(
        data.test, data, predictions
    )
    write_metric_tables(result_directory / "metrics", case_metrics, summary, seasonal)
    bootstrap_frames = []
    for predictor in predictions:
        if predictor in {"raw_fuxi", "imerg_climatology"}:
            continue
        bootstrap_frames.append(
            paired_moving_block_bootstrap(
                case_metrics,
                predictor,
                "raw_fuxi",
                metric_columns=("acc", "rmse", "mae", "bias"),
                block_length=int(config["bootstrap_block_length"]),
                n_resamples=int(config["bootstrap_samples"]),
                seed=42,
            )
        )
    if bootstrap_frames:
        import pandas as pd

        pd.concat(bootstrap_frames, ignore_index=True).to_csv(
            result_directory / "metrics" / "paired_block_bootstrap_by_season_vs_raw.csv",
            index=False,
        )
        by_week_frames = []
        lead_mean_frames = []
        lead_mean_cases = case_metrics.groupby(
            ["predictor", "case_id", "region"], as_index=False
        )[["acc", "rmse", "mae", "bias", "negative_fraction"]].mean()
        comparison_baselines = ("raw_fuxi", "log_bias_correction")
        adapter_predictors = [
            value
            for value in predictions
            if value not in {"raw_fuxi", "imerg_climatology", "log_bias_correction"}
        ]
        for predictor in adapter_predictors:
            for baseline in comparison_baselines:
                by_week_frames.append(
                    paired_moving_block_bootstrap(
                        case_metrics,
                        predictor,
                        baseline,
                        metric_columns=("acc", "rmse", "mae", "bias"),
                        group_columns=("lead", "region"),
                        block_length=int(config["bootstrap_block_length"]),
                        n_resamples=int(config["bootstrap_samples"]),
                        seed=42,
                    )
                )
                lead_mean_frames.append(
                    paired_moving_block_bootstrap(
                        lead_mean_cases,
                        predictor,
                        baseline,
                        metric_columns=("acc", "rmse", "mae", "bias"),
                        group_columns=("region",),
                        block_length=int(config["bootstrap_block_length"]),
                        n_resamples=int(config["bootstrap_samples"]),
                        seed=42,
                    )
                )
        pd.concat(by_week_frames, ignore_index=True).to_csv(
            result_directory / "metrics" / "paired_block_bootstrap_by_week.csv",
            index=False,
        )
        pd.concat(lead_mean_frames, ignore_index=True).to_csv(
            result_directory / "metrics" / "paired_block_bootstrap_lead_mean.csv",
            index=False,
        )
    write_prediction_store(
        result_directory / "predictions.zarr",
        data.test,
        data,
        predictions,
        {
            "split": "locked_test",
            "config_sha256": manifest["config_sha256"],
            "truth": "IMERG Final V07B",
            "climatology": "IMERG 2001-2019",
        },
    )
    _plot_summaries(summary, result_directory / "figures", "test")
    map_predictions = {"truth_imerg": data.test.imerg_truth, **predictions}
    plot_mean_maps(
        map_predictions,
        data.latitude,
        data.longitude,
        data.area_weight_km2 > 0,
        result_directory / "figures" / "mean_maps_all_leads.png",
    )
    for lead in range(6):
        plot_mean_maps(
            map_predictions,
            data.latitude,
            data.longitude,
            data.area_weight_km2 > 0,
            result_directory / "figures" / f"mean_maps_week_{lead + 1}.png",
            lead_index=lead,
        )
    mark_success(
        result_directory,
        {
            "config_sha256": manifest["config_sha256"],
            "predictors": list(predictions),
            "case_count": data.test.sample_count,
        },
    )
    mark_test_consumed(freeze_path, result_directory)
    print(f"LOCKED TEST SUCCESS: {result_directory}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fuxi-adapter",
        description="Deterministic FuXi-S2S to IMERG weekly precipitation adapter",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-data", help="validate all source and split contracts")
    audit.add_argument("--config", required=True)
    audit.add_argument("--output")
    audit.set_defaults(function=command_audit)

    train = commands.add_parser("train", help="train and evaluate on validation only")
    train.add_argument("--config", required=True)
    train.add_argument("--model", required=True, choices=("residual_unet", "temporal_attention_unet"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--smoke", action="store_true")
    train.set_defaults(function=command_train)

    freeze = commands.add_parser("freeze", help="hash development checkpoints before test")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--runs", nargs="+", required=True)
    freeze.set_defaults(function=command_freeze)

    test = commands.add_parser("evaluate-test", help="consume one frozen test manifest once")
    test.add_argument("--frozen-manifest", required=True)
    test.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    test.set_defaults(function=command_test)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
