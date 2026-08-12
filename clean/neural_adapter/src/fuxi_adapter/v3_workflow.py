"""Validation-only workflow for the climatology-conditioned late-lead v3 adapter.

The shared canonical loaders read and audit the complete archive, including
2024.  This workflow never uses 2024 for fitting, checkpoint selection,
prediction, metrics, or plots: candidates train on 2020--2022 and are selected
only on 2023 validation.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .anchored import (
    fit_anchored_target_scale,
    reconstruct_anchored_precipitation,
    standardize_anchored_target,
)
from .artifacts import (
    create_run_directory,
    initialize_run,
    mark_failure,
    mark_success,
)
from .baselines import apply_log_bias_correction, fit_log_bias_correction
from .config import config_sha256, load_config, write_json
from .data import (
    AdapterData,
    DataPaths,
    FuxiT2MData,
    FuxiTPDistributionData,
    ModelArrays,
    fit_normalization,
    load_adapter_data,
    load_fuxi_t2m_data,
    load_fuxi_tp_distribution_data,
    make_model_arrays,
)
from .evaluation import (
    evaluate_prediction_set,
    write_metric_tables,
    write_prediction_store,
)
from .models import build_model, count_parameters
from .plotting import plot_metric_by_lead, plot_training_history
from .training import predict, set_deterministic_seed
from .v3_training import AnchoredSequenceDataset, train_anchored_model


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no assigned GPU is visible")
    return value


def _use_tp_distribution_features(config: Mapping[str, Any]) -> bool:
    value = config.get("use_tp_distribution_features", False)
    if not isinstance(value, bool):
        raise ValueError("use_tp_distribution_features must be a boolean")
    return value


def _split_manifest(data: AdapterData) -> Dict[str, Any]:
    return {
        "audit": data.audit.to_dict(),
        "status": "development_only_2023_validation",
        "test_use": "forbidden_by_v3_workflow",
        "shared_loader_note": (
            "the canonical archive audit loads all splits, but v3 optimization, "
            "checkpoint selection, prediction, metrics, and plots use only train/validation"
        ),
        "train_initializations": [
            np.datetime_as_string(value, unit="D") for value in data.train.initializations
        ],
        "validation_initializations": [
            np.datetime_as_string(value, unit="D")
            for value in data.validation.initializations
        ],
        "test_initializations_sha256_only_until_new_outer_test": True,
    }


def _weighted_feature_moments(
    values: np.ndarray,
    valid_mask: np.ndarray,
    area_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-lead auxiliary-feature moments using training cases only."""

    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    area_weights = np.asarray(area_weights, dtype=np.float64)
    if values.shape != valid_mask.shape or values.ndim != 4:
        raise ValueError("auxiliary values/mask must match [case, lead, lat, lon]")
    if area_weights.shape != values.shape[-2:]:
        raise ValueError("auxiliary feature area weights do not match the grid")
    means = np.empty(values.shape[1], dtype=np.float32)
    standard_deviations = np.empty(values.shape[1], dtype=np.float32)
    for lead in range(values.shape[1]):
        mask = valid_mask[:, lead] & np.isfinite(values[:, lead])
        weights = np.where(mask, area_weights[None], 0.0)
        denominator = weights.sum(dtype=np.float64)
        if denominator <= 0.0:
            raise ValueError(f"auxiliary feature lead {lead + 1} has no valid support")
        mean = np.sum(np.where(mask, values[:, lead], 0.0) * weights) / denominator
        variance = (
            np.sum(np.where(mask, (values[:, lead] - mean) ** 2, 0.0) * weights)
            / denominator
        )
        means[lead] = np.float32(mean)
        standard_deviations[lead] = np.float32(max(np.sqrt(variance), 1.0e-6))
    return means, standard_deviations


def _append_normalized_feature(
    arrays: ModelArrays,
    values: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    channel_name: str,
    *,
    valid_mask: Optional[np.ndarray] = None,
) -> ModelArrays:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != arrays.target.shape:
        raise ValueError(f"{channel_name} shape does not match model arrays")
    mean = np.asarray(mean, dtype=np.float32)
    standard_deviation = np.asarray(standard_deviation, dtype=np.float32)
    if mean.shape != (values.shape[1],) or standard_deviation.shape != mean.shape:
        raise ValueError(f"{channel_name} normalization has the wrong lead shape")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(standard_deviation)):
        raise ValueError(f"{channel_name} normalization contains nonfinite values")
    if np.any(standard_deviation <= 0.0):
        raise ValueError(f"{channel_name} normalization scale must be positive")
    normalized = (
        values - mean[None, :, None, None]
    ) / standard_deviation[None, :, None, None]
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != values.shape:
            raise ValueError(f"{channel_name} valid mask has the wrong shape")
        normalized = np.where(valid_mask, normalized, np.float32(0.0))
    if not np.isfinite(normalized).all():
        raise ValueError(f"normalized {channel_name} contains nonfinite values")
    inputs = np.concatenate(
        (arrays.inputs, normalized[:, :, None].astype(np.float32)), axis=2
    )
    return ModelArrays(
        inputs=inputs,
        target=arrays.target,
        mask=arrays.mask,
        weight=arrays.weight,
        initializations=arrays.initializations,
        channel_names=arrays.channel_names + (channel_name,),
    )


def _append_t2m_mean(
    arrays: ModelArrays,
    values: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> ModelArrays:
    return _append_normalized_feature(
        arrays,
        values,
        mean,
        standard_deviation,
        "fuxi_t2m_mean_weekly",
    )


def _append_tp_distribution_features(
    train_arrays: ModelArrays,
    validation_arrays: ModelArrays,
    distribution: FuxiTPDistributionData,
    train_valid_mask: np.ndarray,
    validation_valid_mask: np.ndarray,
    area_weights: np.ndarray,
) -> Tuple[ModelArrays, ModelArrays, Dict[str, Any]]:
    """Append the frozen three-field TP distribution contract.

    Every moment is fitted independently by lead using only training cases and
    positive area-weight support.  Invalid/off-support normalized values are
    zero, the normalized training mean, rather than transformed fill values.
    """

    if not np.array_equal(
        distribution.train.initializations, train_arrays.initializations
    ):
        raise ValueError("TP distribution training initializations are misaligned")
    if not np.array_equal(
        distribution.validation.initializations,
        validation_arrays.initializations,
    ):
        raise ValueError("TP distribution validation initializations are misaligned")
    feature_names = (
        "member_log_median_anomaly",
        "member_log_iqr",
        "probability_exceeds_imerg_climatology",
    )
    normalization: Dict[str, Any] = {
        "feature_order": list(feature_names),
        "fitted_split": "train",
        "fit_support": "training valid_mask with positive area_weight_km2",
        "features": {},
    }
    for name in feature_names:
        train_values = np.asarray(getattr(distribution.train, name), dtype=np.float32)
        validation_values = np.asarray(
            getattr(distribution.validation, name), dtype=np.float32
        )
        mean, standard_deviation = _weighted_feature_moments(
            train_values,
            train_valid_mask,
            area_weights,
        )
        train_arrays = _append_normalized_feature(
            train_arrays,
            train_values,
            mean,
            standard_deviation,
            name,
            valid_mask=train_valid_mask,
        )
        validation_arrays = _append_normalized_feature(
            validation_arrays,
            validation_values,
            mean,
            standard_deviation,
            name,
            valid_mask=validation_valid_mask,
        )
        normalization["features"][name] = {
            "mean_by_lead": mean.tolist(),
            "std_by_lead": standard_deviation.tolist(),
            "fitted_split": "train",
        }
    return train_arrays, validation_arrays, normalization


def _make_features(
    data: AdapterData,
    t2m: FuxiT2MData,
    tp_distribution: Optional[FuxiTPDistributionData] = None,
) -> Tuple[ModelArrays, ModelArrays, Dict[str, Any]]:
    feature_fields = (
        "log_fuxi",
        "log_spread",
        "log_climatology",
        "fuxi_log_anomaly",
    )
    stats = fit_normalization(
        data.train,
        data.area_weight_km2,
        field_names=feature_fields,
    )
    train_arrays = make_model_arrays(
        data.train, stats, data.latitude, data.longitude, data.area_weight_km2
    )
    validation_arrays = make_model_arrays(
        data.validation, stats, data.latitude, data.longitude, data.area_weight_km2
    )
    t2m_mean, t2m_std = _weighted_feature_moments(
        t2m.train.ensemble_mean_weekly,
        data.train.valid_mask,
        data.area_weight_km2,
    )
    train_arrays = _append_t2m_mean(
        train_arrays, t2m.train.ensemble_mean_weekly, t2m_mean, t2m_std
    )
    validation_arrays = _append_t2m_mean(
        validation_arrays,
        t2m.validation.ensemble_mean_weekly,
        t2m_mean,
        t2m_std,
    )
    normalization = {
        "feature_normalization": stats.to_dict(),
        "t2m_mean_weekly": {
            "mean_by_lead": t2m_mean.tolist(),
            "std_by_lead": t2m_std.tolist(),
            "fitted_split": "train",
            "units": "degC",
        },
    }
    if tp_distribution is not None:
        (
            train_arrays,
            validation_arrays,
            distribution_normalization,
        ) = _append_tp_distribution_features(
            train_arrays,
            validation_arrays,
            tp_distribution,
            data.train.valid_mask,
            data.validation.valid_mask,
            data.area_weight_km2,
        )
        normalization["tp_distribution_features"] = distribution_normalization
    normalization["input_channel_names"] = list(train_arrays.channel_names)
    return train_arrays, validation_arrays, normalization


def _plot_summaries(summary, figures: Path) -> None:
    plotting = summary.copy()
    for metric in ("acc", "rmse", "mae", "bias"):
        plotting[metric] = plotting[f"{metric}_mean"]
        plot_metric_by_lead(
            plotting,
            metric,
            figures / f"validation_{metric}_india.png",
            split="validation",
            region="india",
        )


def train_validation_candidate(
    config: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    smoke: bool = False,
) -> Path:
    """Train one fixed seed and write only 2023 validation artifacts."""

    model_name = str(config["models"][0])
    if model_name != "late_lead_temporal_unet":
        raise ValueError("v3 workflow requires late_lead_temporal_unet")
    if seed not in config["seeds"] and not smoke:
        raise ValueError(f"seed {seed} is not predeclared in the configuration")
    lead_weights = np.asarray(config["lead_weights"], dtype=np.float32)
    if lead_weights.shape != (6,) or not np.array_equal(
        lead_weights[:2], np.zeros(2, dtype=np.float32)
    ) or np.any(lead_weights[2:] <= 0.0):
        raise ValueError("v3 lead_weights must be zero for Weeks 1-2 and positive for 3-6")
    lead_weights = lead_weights / lead_weights.sum()
    loss_coefficients = {
        key: float(config["loss_coefficients"][key])
        for key in ("smooth_l1", "acc", "bias")
    }
    use_tp_distribution_features = _use_tp_distribution_features(config)
    target_device = _device(device)
    paths = DataPaths(Path(config["archive_root"]))
    auxiliary_note = "T2M plus frozen TP distribution" if use_tp_distribution_features else "T2M"
    print(
        "Loading audited TP/IMERG and exactly aligned FuXi " + auxiliary_note + "...",
        flush=True,
    )
    data = load_adapter_data(paths, strict=True)
    t2m = load_fuxi_t2m_data(data, paths)
    tp_distribution = (
        load_fuxi_tp_distribution_data(data, paths)
        if use_tp_distribution_features
        else None
    )
    train_arrays, validation_arrays, normalization = _make_features(
        data, t2m, tp_distribution
    )

    fitted_bias = fit_log_bias_correction(
        data.train.fuxi_mean,
        data.train.imerg_truth,
        data.train.initializations,
        data.area_weight_km2 > 0.0,
    )
    train_bias = apply_log_bias_correction(
        data.train.fuxi_mean, data.train.initializations, fitted_bias
    )
    validation_bias = apply_log_bias_correction(
        data.validation.fuxi_mean,
        data.validation.initializations,
        fitted_bias,
    )
    target_scale = fit_anchored_target_scale(
        data.train.imerg_truth,
        train_bias,
        data.area_weight_km2,
        split_name="train",
        valid_mask=data.train.valid_mask,
    )
    train_target = standardize_anchored_target(
        data.train.imerg_truth,
        train_bias,
        target_scale,
        valid_mask=data.train.valid_mask,
    )
    validation_target = standardize_anchored_target(
        data.validation.imerg_truth,
        validation_bias,
        target_scale,
        valid_mask=data.validation.valid_mask,
    )
    normalization.update(
        {
            "anchored_target": {
                "definition": "log1p(IMERG)-log1p(training-only log-bias FuXi)",
                "rms_scale_by_lead": target_scale.tolist(),
                "active_leads": [3, 4, 5, 6],
                "fitted_split": "train",
            },
            "base_forecast": {
                "name": "training_only_log_bias_correction",
                "shrinkage": fitted_bias.shrinkage,
            },
        }
    )

    model_base_channels = 4 if smoke else int(config["base_channels"])
    run_model_name = model_name + ("_smoke" if smoke else "")
    run_directory = create_run_directory(
        Path(config["output_root"]),
        str(config["experiment_name"]),
        run_model_name,
        seed,
    )
    initialize_run(
        run_directory,
        dict(config),
        model_name,
        seed,
        _split_manifest(data),
    )
    write_json(run_directory / "normalization.json", normalization)
    write_json(run_directory / "data_audit.json", data.audit.to_dict())
    source_manifest = dict(data.source_manifest)
    source_manifest["fuxi_t2m"] = t2m.source_manifest
    if tp_distribution is not None:
        source_manifest["fuxi_tp_distribution"] = tp_distribution.source_manifest
    write_json(run_directory / "source_manifest.json", source_manifest)
    np.savez_compressed(
        run_directory / "bias_correction.npz",
        lead_month_residual=fitted_bias.lead_month_residual,
        shrinkage=np.float32(fitted_bias.shrinkage),
    )

    set_deterministic_seed(seed)
    model = build_model(
        model_name,
        in_channels=train_arrays.inputs.shape[2],
        base_channels=model_base_channels,
        dropout=float(config["dropout"]),
    )
    train_count = 8 if smoke else data.train.sample_count
    validation_count = 4 if smoke else data.validation.sample_count
    train_dataset = AnchoredSequenceDataset(
        train_arrays.inputs[:train_count],
        train_target[:train_count],
        train_bias[:train_count],
        data.train.imerg_truth[:train_count],
        data.train.imerg_climatology[:train_count],
        data.train.valid_mask[:train_count],
    )
    validation_dataset = AnchoredSequenceDataset(
        validation_arrays.inputs[:validation_count],
        validation_target[:validation_count],
        validation_bias[:validation_count],
        data.validation.imerg_truth[:validation_count],
        data.validation.imerg_climatology[:validation_count],
        data.validation.valid_mask[:validation_count],
    )
    try:
        result = train_anchored_model(
            model,
            train_dataset,
            validation_dataset,
            data.area_weight_km2,
            target_scale,
            lead_weights,
            loss_coefficients,
            run_directory,
            seed=seed,
            device=target_device,
            batch_size=min(int(config["batch_size"]), train_count),
            max_epochs=2 if smoke else int(config["max_epochs"]),
            patience=2 if smoke else int(config["early_stopping_patience"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            smooth_l1_beta=float(config["smooth_l1_beta"]),
            num_workers=int(config["num_workers"]),
            use_amp=bool(config["use_amp"]),
        )
        predicted_residual = predict(
            model,
            validation_arrays.inputs[:validation_count],
            device=target_device,
            batch_size=int(config["batch_size"]),
            use_amp=bool(config["use_amp"]),
        )
        corrected = reconstruct_anchored_precipitation(
            validation_bias[:validation_count],
            predicted_residual,
            target_scale,
            valid_mask=data.validation.valid_mask[:validation_count],
        )
        validation_split = data.validation
        if smoke:
            validation_split = type(data.validation)(
                **{
                    field: (
                        getattr(data.validation, field)[:validation_count]
                        if isinstance(getattr(data.validation, field), np.ndarray)
                        and getattr(data.validation, field).shape[:1]
                        == (data.validation.sample_count,)
                        else getattr(data.validation, field)
                    )
                    for field in data.validation.__dataclass_fields__
                }
            )
        predictions = {
            "raw_fuxi": data.validation.fuxi_mean[:validation_count],
            "imerg_climatology": data.validation.imerg_climatology[:validation_count],
            "log_bias_correction": validation_bias[:validation_count],
            model_name: corrected,
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
                "scientific_output": not smoke,
                "model": model_name,
                "seed": seed,
                "config_sha256": config_sha256(dict(config)),
                "test_predictions_evaluated": False,
                "selection_split": "2023_validation_only",
                "base_forecast": "training-only log-bias correction",
            },
        )
        plot_training_history(
            result.history, run_directory / "figures" / "training_history.png"
        )
        _plot_summaries(summary, run_directory / "figures")
        success = {
            "model": model_name,
            "seed": seed,
            "smoke": smoke,
            "device": target_device,
            "base_channels": model_base_channels,
            "dropout": float(config["dropout"]),
            "in_channels": int(train_arrays.inputs.shape[2]),
            "channel_names": list(train_arrays.channel_names),
            "use_tp_distribution_features": use_tp_distribution_features,
            "input_contract": config.get("input_contract"),
            "parameter_count": count_parameters(model),
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "elapsed_seconds": result.elapsed_seconds,
            "lead_weights": lead_weights.tolist(),
            "loss_coefficients": loss_coefficients,
            "validation_only": True,
            "test_predictions_evaluated": False,
            "selection_split": "2023_validation_only",
        }
        mark_success(run_directory, success)
        print(json.dumps(success, indent=2), flush=True)
        print(f"V3 VALIDATION SUCCESS: {run_directory}", flush=True)
        return run_directory
    except Exception as exc:
        mark_failure(run_directory, f"{type(exc).__name__}: {exc}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the validation-only FuXi/IMERG late-lead v3 adapter"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(Path(args.config))
        train_validation_candidate(
            config,
            seed=args.seed,
            device=args.device,
            smoke=args.smoke,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "train_validation_candidate"]
