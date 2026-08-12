"""Anchored residual targets and lead-aware physical verification losses.

The v3 adapter predicts a small correction around a training-only deterministic
bias-corrected forecast, rather than correcting raw FuXi directly.  Targets are
defined in log-rainfall space while the pattern and bias terms are evaluated in
physical ``mm day-1`` units.

This module is intentionally independent of the v2 training loop.  It contains
no data loading, checkpoint selection, or test-set access.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F


Array = np.ndarray
LossResult = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


def _numpy_valid_mask(
    values_shape: Tuple[int, ...], valid_mask: Optional[Array]
) -> Array:
    if valid_mask is None:
        return np.ones(values_shape, dtype=bool)
    mask = np.asarray(valid_mask, dtype=bool)
    try:
        return np.broadcast_to(mask, values_shape)
    except ValueError as exc:
        raise ValueError(
            "valid_mask with shape {} cannot be broadcast to {}".format(
                mask.shape, values_shape
            )
        ) from exc


def _validate_numpy_fields(
    truth: Array,
    baseline: Array,
    area_weights: Optional[Array] = None,
    valid_mask: Optional[Array] = None,
) -> Tuple[Array, Array, Optional[Array], Array]:
    truth_array = np.asarray(truth, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if truth_array.ndim != 4:
        raise ValueError("truth must have shape [case, lead, latitude, longitude]")
    if baseline_array.shape != truth_array.shape:
        raise ValueError(
            "baseline shape {} does not match truth shape {}".format(
                baseline_array.shape, truth_array.shape
            )
        )
    mask = _numpy_valid_mask(truth_array.shape, valid_mask)

    weight_array: Optional[Array] = None
    if area_weights is not None:
        weight_array = np.asarray(area_weights, dtype=np.float64)
        if weight_array.shape != truth_array.shape[-2:]:
            raise ValueError(
                "area_weights must have shape {}; got {}".format(
                    truth_array.shape[-2:], weight_array.shape
                )
            )
        if not np.isfinite(weight_array).all() or np.any(weight_array < 0.0):
            raise ValueError("area_weights must be finite and nonnegative")
        if not np.any(weight_array > 0.0):
            raise ValueError("area_weights contain no positive support")
        mask = mask & (weight_array[None, None, :, :] > 0.0)

    if not np.any(mask):
        raise ValueError("valid support is empty")
    if not np.isfinite(truth_array[mask]).all():
        raise ValueError("truth must be finite on valid support")
    if not np.isfinite(baseline_array[mask]).all():
        raise ValueError("baseline must be finite on valid support")
    if np.any(truth_array[mask] < 0.0) or np.any(baseline_array[mask] < 0.0):
        raise ValueError("truth and baseline precipitation must be nonnegative")
    return truth_array, baseline_array, weight_array, mask


def _validate_scale(scale: Array, lead_count: int, name: str) -> Array:
    value = np.asarray(scale, dtype=np.float64)
    if value.shape != (lead_count,):
        raise ValueError(
            "{} must have shape ({},); got {}".format(name, lead_count, value.shape)
        )
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError("{} must be finite and strictly positive".format(name))
    return value


def fit_anchored_target_scale(
    truth: Array,
    training_bias_baseline: Array,
    area_weights: Array,
    *,
    split_name: str,
    valid_mask: Optional[Array] = None,
    minimum_scale: float = 1.0e-6,
) -> Array:
    """Fit a per-lead RMS scale for the anchored log residual.

    The residual is

    ``log1p(truth) - log1p(training_bias_baseline)``.

    Every case receives the same spatial area weights.  ``split_name`` is
    deliberately required so validation or test targets cannot accidentally be
    used to fit preprocessing constants.
    """

    if split_name != "train":
        raise ValueError("anchored target scale must be fitted on the train split")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be finite and positive")
    truth_array, baseline_array, weights, mask = _validate_numpy_fields(
        truth, training_bias_baseline, area_weights, valid_mask
    )
    assert weights is not None

    residual = np.zeros_like(truth_array, dtype=np.float64)
    residual[mask] = np.log1p(truth_array[mask]) - np.log1p(baseline_array[mask])
    scales = np.empty(truth_array.shape[1], dtype=np.float32)
    base_weight = weights[None, :, :]
    for lead in range(truth_array.shape[1]):
        lead_mask = mask[:, lead]
        lead_weight = np.where(lead_mask, base_weight, 0.0)
        denominator = float(np.sum(lead_weight, dtype=np.float64))
        if denominator <= 0.0:
            raise ValueError("lead {} has no positive valid weight".format(lead + 1))
        mean_square = float(
            np.sum(lead_weight * residual[:, lead] ** 2, dtype=np.float64)
            / denominator
        )
        scales[lead] = np.float32(max(np.sqrt(max(mean_square, 0.0)), minimum_scale))
    return scales


def standardize_anchored_target(
    truth: Array,
    bias_baseline: Array,
    target_scale: Array,
    *,
    valid_mask: Optional[Array] = None,
) -> Array:
    """Return standardized anchored log residuals, with invalid cells set to zero."""

    truth_array, baseline_array, _, mask = _validate_numpy_fields(
        truth, bias_baseline, None, valid_mask
    )
    scale = _validate_scale(target_scale, truth_array.shape[1], "target_scale")
    target = np.zeros_like(truth_array, dtype=np.float64)
    target[mask] = (
        np.log1p(truth_array[mask]) - np.log1p(baseline_array[mask])
    )
    target /= scale[None, :, None, None]
    if not np.isfinite(target).all():
        raise ValueError("standardized anchored target is not finite")
    return target.astype(np.float32)


def reconstruct_anchored_precipitation(
    bias_baseline: Array,
    standardized_log_residual: Array,
    target_scale: Array,
    *,
    valid_mask: Optional[Array] = None,
    maximum_log_rain: float = 20.0,
) -> Array:
    """Reconstruct finite, nonnegative physical precipitation.

    ``maximum_log_rain`` is only a numerical overflow guard.  Its default
    corresponds to roughly 4.9e8 mm/day, many orders above a physical rainfall
    value, so it does not constrain plausible predictions.
    """

    baseline = np.asarray(bias_baseline, dtype=np.float64)
    residual = np.asarray(standardized_log_residual, dtype=np.float64)
    if baseline.ndim != 4:
        raise ValueError(
            "bias_baseline must have shape [case, lead, latitude, longitude]"
        )
    if residual.shape != baseline.shape:
        raise ValueError(
            "standardized_log_residual shape {} does not match baseline shape {}".format(
                residual.shape, baseline.shape
            )
        )
    mask = _numpy_valid_mask(baseline.shape, valid_mask)
    if not np.any(mask):
        raise ValueError("valid support is empty")
    if not np.isfinite(baseline[mask]).all() or np.any(baseline[mask] < 0.0):
        raise ValueError("bias_baseline must be finite and nonnegative on valid support")
    if not np.isfinite(residual[mask]).all():
        raise ValueError(
            "standardized_log_residual must be finite on valid support"
        )
    scale = _validate_scale(target_scale, baseline.shape[1], "target_scale")
    if not np.isfinite(maximum_log_rain) or maximum_log_rain <= 0.0:
        raise ValueError("maximum_log_rain must be finite and positive")

    safe_baseline = np.where(mask, baseline, 0.0)
    safe_residual = np.where(mask, residual, 0.0)
    log_rain = np.log1p(safe_baseline) + safe_residual * scale[None, :, None, None]
    log_rain = np.clip(log_rain, 0.0, maximum_log_rain)
    prediction = np.expm1(log_rain)
    # Preserve the adapter's identity contract bit-for-bit at zero residual.
    prediction = np.where(safe_residual == 0.0, safe_baseline, prediction)
    prediction[~mask] = 0.0
    if not np.isfinite(prediction).all():
        raise FloatingPointError("anchored reconstruction produced nonfinite rainfall")
    return prediction.astype(np.float32)


def _as_loss_vector(
    values: Union[Sequence[float], torch.Tensor],
    lead_count: int,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    strictly_positive: bool,
) -> torch.Tensor:
    vector = torch.as_tensor(values, device=device, dtype=dtype)
    if vector.shape != (lead_count,):
        raise ValueError(
            "{} must have shape ({},); got {}".format(name, lead_count, tuple(vector.shape))
        )
    if not torch.isfinite(vector).all():
        raise ValueError("{} must be finite".format(name))
    if strictly_positive:
        if torch.any(vector <= 0.0):
            raise ValueError("{} must be strictly positive".format(name))
    elif torch.any(vector < 0.0) or not torch.any(vector > 0.0):
        raise ValueError("{} must be nonnegative with at least one active lead".format(name))
    return vector


def _torch_broadcast_mask(
    valid_mask: Optional[torch.Tensor], shape: Tuple[int, ...], device: torch.device
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    try:
        return torch.broadcast_to(mask, shape)
    except RuntimeError as exc:
        raise ValueError(
            "valid_mask with shape {} cannot be broadcast to {}".format(
                tuple(mask.shape), shape
            )
        ) from exc


def anchored_composite_loss(
    predicted_standardized_residual: torch.Tensor,
    target_standardized_residual: torch.Tensor,
    bias_baseline: torch.Tensor,
    truth: torch.Tensor,
    climatology: torch.Tensor,
    target_scale: Union[Sequence[float], torch.Tensor],
    area_weights: torch.Tensor,
    lead_weights: Union[Sequence[float], torch.Tensor],
    *,
    valid_mask: Optional[torch.Tensor] = None,
    bias_scale: Optional[Union[Sequence[float], torch.Tensor]] = None,
    smooth_l1_coefficient: float = 0.60,
    acc_coefficient: float = 0.30,
    bias_coefficient: float = 0.10,
    smooth_l1_beta: float = 1.0,
    epsilon: float = 1.0e-8,
    maximum_log_rain: float = 20.0,
    return_components: bool = False,
) -> LossResult:
    """Return a lead-aware anchored residual loss.

    The three components are:

    1. area-weighted Smooth-L1 error in standardized anchored-residual space;
    2. one minus centred, India-area-weighted spatial anomaly correlation in
       reconstructed physical units relative to the fixed climatology;
    3. squared area-mean physical bias, normalized by either ``bias_scale`` or
       the detached per-case/lead RMS truth (floored at 1 mm/day).

    Zero lead weights fully deactivate a lead in every component.  Nonfinite
    values are allowed only outside the supplied valid support or on inactive
    leads, where they are sanitized before differentiable calculations.
    """

    prediction = predicted_standardized_residual
    if not isinstance(prediction, torch.Tensor) or prediction.ndim != 4:
        raise ValueError(
            "predicted_standardized_residual must be a tensor [batch, lead, lat, lon]"
        )
    if not prediction.is_floating_point():
        raise ValueError("predicted_standardized_residual must be floating point")
    shape = tuple(prediction.shape)
    batch_size, lead_count, height, width = shape
    if batch_size < 1 or lead_count < 1 or height < 1 or width < 1:
        raise ValueError("loss inputs must have non-empty batch, lead, and spatial axes")

    fields = {
        "target_standardized_residual": target_standardized_residual,
        "bias_baseline": bias_baseline,
        "truth": truth,
        "climatology": climatology,
    }
    converted: Dict[str, torch.Tensor] = {}
    for name, value in fields.items():
        tensor = torch.as_tensor(value, device=prediction.device, dtype=prediction.dtype)
        if tuple(tensor.shape) != shape:
            raise ValueError(
                "{} shape {} does not match prediction shape {}".format(
                    name, tuple(tensor.shape), shape
                )
            )
        converted[name] = tensor

    target = converted["target_standardized_residual"]
    baseline = converted["bias_baseline"]
    truth_tensor = converted["truth"]
    climate = converted["climatology"]
    scale = _as_loss_vector(
        target_scale,
        lead_count,
        name="target_scale",
        device=prediction.device,
        dtype=prediction.dtype,
        strictly_positive=True,
    )
    lead_weight = _as_loss_vector(
        lead_weights,
        lead_count,
        name="lead_weights",
        device=prediction.device,
        dtype=prediction.dtype,
        strictly_positive=False,
    )

    spatial_weight = torch.as_tensor(
        area_weights, device=prediction.device, dtype=prediction.dtype
    )
    if tuple(spatial_weight.shape) != (height, width):
        raise ValueError(
            "area_weights must have shape {}; got {}".format(
                (height, width), tuple(spatial_weight.shape)
            )
        )
    if not torch.isfinite(spatial_weight).all() or torch.any(spatial_weight < 0.0):
        raise ValueError("area_weights must be finite and nonnegative")
    if not torch.any(spatial_weight > 0.0):
        raise ValueError("area_weights contain no positive support")

    coefficients = (
        float(smooth_l1_coefficient),
        float(acc_coefficient),
        float(bias_coefficient),
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in coefficients):
        raise ValueError("loss coefficients must be finite and nonnegative")
    if sum(coefficients) <= 0.0:
        raise ValueError("at least one loss coefficient must be positive")
    if not np.isfinite(smooth_l1_beta) or smooth_l1_beta <= 0.0:
        raise ValueError("smooth_l1_beta must be finite and positive")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(maximum_log_rain) or maximum_log_rain <= 0.0:
        raise ValueError("maximum_log_rain must be finite and positive")

    support = _torch_broadcast_mask(valid_mask, shape, prediction.device)
    support = support & (spatial_weight[None, None, :, :] > 0.0)
    active = lead_weight[None, :, None, None] > 0.0
    evaluation_mask = support & active
    for name, tensor in {
        "predicted_standardized_residual": prediction,
        **converted,
    }.items():
        if not torch.isfinite(tensor[evaluation_mask]).all():
            raise ValueError("{} must be finite on active valid support".format(name))
    for name, tensor in {
        "bias_baseline": baseline,
        "truth": truth_tensor,
        "climatology": climate,
    }.items():
        if torch.any(tensor[evaluation_mask] < 0.0):
            raise ValueError("{} precipitation must be nonnegative".format(name))

    pair_spatial_weight = (
        spatial_weight[None, None, :, :]
        * support.to(dtype=prediction.dtype)
    )
    pair_weight_sum = pair_spatial_weight.sum(dim=(-2, -1))
    active_pairs = lead_weight[None, :].expand(batch_size, -1) > 0.0
    if torch.any(active_pairs & (pair_weight_sum <= 0.0)):
        raise ValueError("an active case/lead has no positive valid spatial weight")
    normalized_spatial_weight = pair_spatial_weight / pair_weight_sum.clamp_min(epsilon)[
        ..., None, None
    ]

    safe_prediction = torch.where(evaluation_mask, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(evaluation_mask, target, torch.zeros_like(target))
    safe_baseline = torch.where(evaluation_mask, baseline, torch.zeros_like(baseline))
    safe_truth = torch.where(evaluation_mask, truth_tensor, torch.zeros_like(truth_tensor))
    safe_climate = torch.where(evaluation_mask, climate, torch.zeros_like(climate))

    element_loss = F.smooth_l1_loss(
        safe_prediction,
        safe_target,
        reduction="none",
        beta=float(smooth_l1_beta),
    )
    smooth_by_pair = (element_loss * normalized_spatial_weight).sum(dim=(-2, -1))
    lead_pair_weight = lead_weight[None, :].expand(batch_size, -1)
    pair_denominator = lead_pair_weight.sum().clamp_min(epsilon)
    smooth_loss = (smooth_by_pair * lead_pair_weight).sum() / pair_denominator

    log_prediction = torch.log1p(safe_baseline) + safe_prediction * scale[
        None, :, None, None
    ]
    physical_prediction = torch.expm1(
        torch.clamp(log_prediction, min=0.0, max=float(maximum_log_rain))
    )

    prediction_anomaly = physical_prediction - safe_climate
    truth_anomaly = safe_truth - safe_climate
    prediction_mean = (prediction_anomaly * normalized_spatial_weight).sum(
        dim=(-2, -1), keepdim=True
    )
    truth_mean = (truth_anomaly * normalized_spatial_weight).sum(
        dim=(-2, -1), keepdim=True
    )
    prediction_centered = prediction_anomaly - prediction_mean
    truth_centered = truth_anomaly - truth_mean
    covariance = (
        prediction_centered * truth_centered * normalized_spatial_weight
    ).sum(dim=(-2, -1))
    prediction_variance = (
        prediction_centered.square() * normalized_spatial_weight
    ).sum(dim=(-2, -1))
    truth_variance = (truth_centered.square() * normalized_spatial_weight).sum(
        dim=(-2, -1)
    )
    correlation_denominator = torch.sqrt(
        prediction_variance.clamp_min(0.0) * truth_variance.clamp_min(0.0)
    )
    correlation_defined = (
        active_pairs
        & (prediction_variance > epsilon)
        & (truth_variance > epsilon)
    )
    correlation = covariance / correlation_denominator.clamp_min(epsilon)
    correlation = correlation.clamp(min=-1.0, max=1.0)
    correlation_pair_weight = lead_pair_weight * correlation_defined.to(prediction.dtype)
    correlation_weight_sum = correlation_pair_weight.sum()
    if bool(correlation_defined.any()):
        acc_loss = (
            (1.0 - correlation) * correlation_pair_weight
        ).sum() / correlation_weight_sum.clamp_min(epsilon)
        mean_acc = (
            correlation * correlation_pair_weight
        ).sum() / correlation_weight_sum.clamp_min(epsilon)
    else:
        # No spatially varying anomaly means there is no defined pattern term.
        # The zero is connected to prediction so backward remains valid.
        acc_loss = safe_prediction.sum() * 0.0
        mean_acc = safe_prediction.sum() * 0.0

    mean_error = (
        (physical_prediction - safe_truth) * normalized_spatial_weight
    ).sum(dim=(-2, -1))
    if bias_scale is None:
        normalization = torch.sqrt(
            (safe_truth.square() * normalized_spatial_weight).sum(dim=(-2, -1))
        ).detach().clamp_min(1.0)
    else:
        bias_scale_vector = _as_loss_vector(
            bias_scale,
            lead_count,
            name="bias_scale",
            device=prediction.device,
            dtype=prediction.dtype,
            strictly_positive=True,
        )
        normalization = bias_scale_vector[None, :].expand(batch_size, -1)
    normalized_bias = mean_error / normalization
    bias_loss = (
        normalized_bias.square() * lead_pair_weight
    ).sum() / pair_denominator

    total = (
        smooth_l1_coefficient * smooth_loss
        + acc_coefficient * acc_loss
        + bias_coefficient * bias_loss
    )
    if not torch.isfinite(total):
        raise FloatingPointError("anchored composite loss is nonfinite")
    if return_components:
        return total, {
            "smooth_l1": smooth_loss,
            "acc_loss": acc_loss,
            "mean_spatial_acc": mean_acc,
            "mean_bias_squared": bias_loss,
        }
    return total


# Descriptive alias retained for experiment configurations that use this name.
fit_anchored_residual_scale = fit_anchored_target_scale


__all__ = [
    "anchored_composite_loss",
    "fit_anchored_residual_scale",
    "fit_anchored_target_scale",
    "reconstruct_anchored_precipitation",
    "standardize_anchored_target",
]
