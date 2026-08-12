"""Transparent deterministic baselines for the neural-adapter comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def verification_midpoint_months(initializations: np.ndarray, leads: int = 6) -> np.ndarray:
    """Month of the fourth observation period in each seven-day window.

    A FuXi forecast issued at 00 UTC begins its first daily interval at the
    initialization itself, so weekly observation-period starts use offsets
    0--6 and their fourth date has offset 3.
    """

    initializations = np.asarray(initializations, dtype="datetime64[D]")
    offsets = np.arange(leads, dtype="timedelta64[D]") * 7 + np.timedelta64(3, "D")
    dates = initializations[:, None] + offsets[None]
    return np.asarray(
        [[pd.Timestamp(value).month for value in row] for row in dates], dtype=np.int8
    )


@dataclass(frozen=True)
class LogBiasCorrection:
    """Training-only additive mean correction in log-rainfall space."""

    lead_month_residual: np.ndarray  # [lead, 12, latitude, longitude]
    shrinkage: float


def fit_log_bias_correction(
    fuxi_mean: np.ndarray,
    truth: np.ndarray,
    initializations: np.ndarray,
    valid_mask: np.ndarray,
    shrinkage: float = 10.0,
) -> LogBiasCorrection:
    """Fit a lead/month/grid correction, shrunk toward each lead's mean bias."""

    fuxi_mean = np.asarray(fuxi_mean, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if fuxi_mean.shape != truth.shape or fuxi_mean.ndim != 4:
        raise ValueError("fuxi_mean and truth must have shape [case, lead, lat, lon]")
    if shrinkage < 0:
        raise ValueError("shrinkage must be nonnegative")
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != fuxi_mean.shape[-2:]:
        raise ValueError("valid_mask does not match the spatial grid")

    residual = np.log1p(np.maximum(truth, 0.0)) - np.log1p(np.maximum(fuxi_mean, 0.0))
    residual[..., ~mask] = np.nan
    _, lead_count, height, width = residual.shape
    months = verification_midpoint_months(initializations, lead_count)
    correction = np.full((lead_count, 12, height, width), np.nan, dtype=np.float32)

    for lead in range(lead_count):
        lead_values = residual[:, lead]
        lead_valid = np.isfinite(lead_values)
        lead_count_grid = lead_valid.sum(axis=0)
        lead_sum = np.where(lead_valid, lead_values, 0.0).sum(axis=0)
        lead_mean = np.divide(
            lead_sum,
            lead_count_grid,
            out=np.zeros_like(lead_sum),
            where=lead_count_grid > 0,
        )
        for month in range(1, 13):
            selected = lead_values[months[:, lead] == month]
            finite = np.isfinite(selected)
            count = finite.sum(axis=0)
            total = np.where(finite, selected, 0.0).sum(axis=0)
            value = (total + shrinkage * lead_mean) / (count + shrinkage)
            value[~mask] = np.nan
            correction[lead, month - 1] = value.astype(np.float32)
    return LogBiasCorrection(correction, float(shrinkage))


def apply_log_bias_correction(
    fuxi_mean: np.ndarray,
    initializations: np.ndarray,
    correction: LogBiasCorrection,
) -> np.ndarray:
    """Apply a fitted correction and reconstruct nonnegative rainfall."""

    fuxi_mean = np.asarray(fuxi_mean, dtype=np.float64)
    if fuxi_mean.ndim != 4:
        raise ValueError("fuxi_mean must have shape [case, lead, lat, lon]")
    months = verification_midpoint_months(initializations, fuxi_mean.shape[1])
    adjusted = np.empty_like(fuxi_mean)
    for case in range(fuxi_mean.shape[0]):
        for lead in range(fuxi_mean.shape[1]):
            delta = correction.lead_month_residual[lead, months[case, lead] - 1]
            adjusted[case, lead] = np.expm1(
                np.maximum(0.0, np.log1p(np.maximum(fuxi_mean[case, lead], 0.0)) + delta)
            )
    return adjusted.astype(np.float32)
