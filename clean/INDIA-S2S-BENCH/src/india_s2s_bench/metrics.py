"""Area-weighted deterministic verification metrics."""

from __future__ import annotations

import numpy as np


def _valid(prediction: np.ndarray, truth: np.ndarray, weight: np.ndarray):
    mask = np.isfinite(prediction) & np.isfinite(truth) & np.isfinite(weight) & (weight > 0)
    if not np.any(mask):
        raise ValueError("metric has no finite positive-weight cells")
    return prediction[mask].astype(np.float64), truth[mask].astype(np.float64), weight[mask].astype(np.float64)


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weight) & (weight > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weight[mask]))


def spatial_acc(prediction_anomaly: np.ndarray, truth_anomaly: np.ndarray, weight: np.ndarray) -> float:
    p, o, w = _valid(prediction_anomaly, truth_anomaly, weight)
    p = p - np.average(p, weights=w)
    o = o - np.average(o, weights=w)
    denominator = np.sqrt(np.sum(w * p * p) * np.sum(w * o * o))
    return float(np.sum(w * p * o) / denominator) if denominator > 0 else float("nan")


def case_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    weight: np.ndarray,
    wet_threshold: float,
) -> dict[str, float]:
    p, o, w = _valid(prediction, truth, weight)
    c = climatology[np.isfinite(prediction) & np.isfinite(truth) & np.isfinite(weight) & (weight > 0)].astype(np.float64)
    error = p - o
    wet_prediction = np.average((p >= wet_threshold).astype(np.float64), weights=w)
    wet_truth = np.average((o >= wet_threshold).astype(np.float64), weights=w)
    return {
        "acc": spatial_acc(p - c, o - c, w),
        "rmse": float(np.sqrt(np.average(error * error, weights=w))),
        "mae": float(np.average(np.abs(error), weights=w)),
        "bias": float(np.average(error, weights=w)),
        "wet_fraction_error": float(wet_prediction - wet_truth),
        "negative_fraction": float(np.average((p < 0).astype(np.float64), weights=w)),
        "valid_area_weight": float(np.sum(w)),
        "valid_cells": int(len(p)),
    }


def aggregate_case_metrics(rows):
    """Aggregate already case-level metrics; ACC is never pooled over grid cells."""
    result = {"cases": int(len(rows))}
    for metric in ("acc", "rmse", "mae", "bias", "wet_fraction_error", "negative_fraction"):
        values = np.asarray(rows[metric], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[metric] = float(np.mean(finite)) if len(finite) else float("nan")
    return result
