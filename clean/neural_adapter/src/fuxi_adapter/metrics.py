"""Transparent, case-wise verification for the FuXi neural adapter.

All spatial scores are calculated independently for each forecast case and lead.
This is important: grid cells are not treated as independent forecast samples.
The summary functions then give every forecast case equal weight.

The anomaly correlation coefficient (ACC) expects climatological anomalies as
inputs.  It never estimates a climatology from the verification period.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


METRIC_COLUMNS: Tuple[str, ...] = (
    "acc",
    "rmse",
    "mae",
    "bias",
    "negative_fraction",
)


def _broadcast_array(array: np.ndarray, shape: Tuple[int, ...], name: str) -> np.ndarray:
    """Broadcast ``array`` to ``shape`` or raise an informative error."""

    value = np.asarray(array)
    try:
        return np.broadcast_to(value, shape)
    except ValueError as exc:
        raise ValueError(
            "{} with shape {} cannot be broadcast to {}".format(name, value.shape, shape)
        ) from exc


def _valid_vectors(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened finite values on positive-weight verification support."""

    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if truth_array.shape != prediction_array.shape:
        raise ValueError(
            "truth and prediction must have identical shapes; got {} and {}".format(
                truth_array.shape, prediction_array.shape
            )
        )

    weight_array = _broadcast_array(
        np.asarray(weights, dtype=np.float64), truth_array.shape, "weights"
    )
    valid = (
        np.isfinite(truth_array)
        & np.isfinite(prediction_array)
        & np.isfinite(weight_array)
        & (weight_array > 0.0)
    )
    if valid_mask is not None:
        mask = _broadcast_array(np.asarray(valid_mask, dtype=bool), truth_array.shape, "valid_mask")
        valid &= mask

    return truth_array[valid], prediction_array[valid], weight_array[valid]


def weighted_spatial_acc(
    truth_anomaly: np.ndarray,
    prediction_anomaly: np.ndarray,
    weights: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Return weighted, centred spatial anomaly correlation.

    The supplied fields must already be anomalies relative to a training-period
    climatology.  Weighted spatial means are removed before correlation, matching
    the usual pattern-ACC definition.  A constant field has undefined ACC and
    therefore returns ``NaN``.
    """

    truth, prediction, weight = _valid_vectors(
        truth_anomaly, prediction_anomaly, weights, valid_mask
    )
    if truth.size < 2:
        return float("nan")

    weight_sum = float(np.sum(weight))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")

    truth_centred = truth - np.sum(weight * truth) / weight_sum
    prediction_centred = prediction - np.sum(weight * prediction) / weight_sum
    numerator = float(np.sum(weight * truth_centred * prediction_centred))
    denominator = float(
        np.sqrt(
            np.sum(weight * truth_centred**2)
            * np.sum(weight * prediction_centred**2)
        )
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def weighted_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Return area-weighted root-mean-square error."""

    truth_values, prediction_values, weight = _valid_vectors(
        truth, prediction, weights, valid_mask
    )
    if truth_values.size == 0:
        return float("nan")
    return float(
        np.sqrt(np.average((prediction_values - truth_values) ** 2, weights=weight))
    )


def weighted_mae(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Return area-weighted mean absolute error."""

    truth_values, prediction_values, weight = _valid_vectors(
        truth, prediction, weights, valid_mask
    )
    if truth_values.size == 0:
        return float("nan")
    return float(np.average(np.abs(prediction_values - truth_values), weights=weight))


def weighted_bias(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Return area-weighted additive bias (prediction minus truth)."""

    truth_values, prediction_values, weight = _valid_vectors(
        truth, prediction, weights, valid_mask
    )
    if truth_values.size == 0:
        return float("nan")
    return float(np.average(prediction_values - truth_values, weights=weight))


def weighted_negative_fraction(
    prediction: np.ndarray,
    weights: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Return the area-weighted fraction of finite predictions below zero."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    weight_array = _broadcast_array(
        np.asarray(weights, dtype=np.float64), prediction_array.shape, "weights"
    )
    valid = (
        np.isfinite(prediction_array)
        & np.isfinite(weight_array)
        & (weight_array > 0.0)
    )
    if valid_mask is not None:
        valid &= _broadcast_array(
            np.asarray(valid_mask, dtype=bool), prediction_array.shape, "valid_mask"
        )
    if not np.any(valid):
        return float("nan")
    return float(np.average(prediction_array[valid] < 0.0, weights=weight_array[valid]))


def _as_case_lead(array: np.ndarray, name: str) -> np.ndarray:
    """Normalize ``[case, spatial...]`` or ``[case, lead, spatial...]`` arrays."""

    value = np.asarray(array)
    if value.ndim < 3:
        raise ValueError(
            "{} must have shape [case, spatial...] or [case, lead, spatial...]".format(name)
        )
    if value.ndim == 3:
        value = value[:, np.newaxis, ...]
    return value


def _case_lead_labels(
    values: Optional[Sequence[object]],
    n_case: int,
    n_lead: int,
    name: str,
    default: object,
) -> np.ndarray:
    """Return labels broadcast to ``[case, lead]``."""

    if values is None:
        return np.full((n_case, n_lead), default, dtype=object)
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return np.full((n_case, n_lead), array.item(), dtype=object)
    if array.shape == (n_case,):
        return np.broadcast_to(array[:, np.newaxis], (n_case, n_lead))
    if array.shape == (n_lead,):
        return np.broadcast_to(array[np.newaxis, :], (n_case, n_lead))
    if array.shape == (n_case, n_lead):
        return array
    raise ValueError(
        "{} must be scalar or have shape ({},), ({},), or ({}, {}); got {}".format(
            name, n_case, n_lead, n_case, n_lead, array.shape
        )
    )


def compute_case_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    truth_anomaly: np.ndarray,
    prediction_anomaly: np.ndarray,
    weights: np.ndarray,
    *,
    predictor: str,
    case_ids: Optional[Sequence[object]] = None,
    leads: Optional[Sequence[object]] = None,
    seasons: Optional[Sequence[object]] = None,
    region_weights: Optional[Mapping[str, np.ndarray]] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Calculate one score record per case, lead, and region.

    Parameters
    ----------
    truth, prediction
        Absolute precipitation fields.  Accepted shapes are ``[case, y, x]`` or
        ``[case, lead, y, x]`` (and the analogous shape with extra spatial axes).
    truth_anomaly, prediction_anomaly
        Fields on the same shape after subtracting a training-only climatology.
    weights
        Area/coverage weights broadcastable to the input shape.
    region_weights
        Optional mapping of region names to fractional masks.  Each region mask is
        multiplied by ``weights``.  With no mapping, a single ``india`` region is
        evaluated.

    Notes
    -----
    ``negative_fraction`` is evaluated on the same finite truth/prediction support
    as the error metrics.  Forecasts should be passed before physical clipping when
    this diagnostic is intended to expose negative neural predictions.
    """

    truth_array = _as_case_lead(truth, "truth").astype(np.float64, copy=False)
    prediction_array = _as_case_lead(prediction, "prediction").astype(
        np.float64, copy=False
    )
    truth_anomaly_array = _as_case_lead(truth_anomaly, "truth_anomaly").astype(
        np.float64, copy=False
    )
    prediction_anomaly_array = _as_case_lead(
        prediction_anomaly, "prediction_anomaly"
    ).astype(np.float64, copy=False)

    expected_shape = truth_array.shape
    for name, value in (
        ("prediction", prediction_array),
        ("truth_anomaly", truth_anomaly_array),
        ("prediction_anomaly", prediction_anomaly_array),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                "{} has shape {}, expected {}".format(name, value.shape, expected_shape)
            )

    n_case, n_lead = expected_shape[:2]
    spatial_shape = expected_shape[2:]
    weight_array = _broadcast_array(
        np.asarray(weights, dtype=np.float64), expected_shape, "weights"
    )
    if valid_mask is None:
        mask_array = np.ones(expected_shape, dtype=bool)
    else:
        mask_array = _broadcast_array(
            np.asarray(valid_mask, dtype=bool), expected_shape, "valid_mask"
        )

    if case_ids is None:
        case_values = np.arange(n_case)
    else:
        case_values = np.asarray(case_ids, dtype=object)
        if case_values.shape != (n_case,):
            raise ValueError(
                "case_ids must have shape ({},); got {}".format(n_case, case_values.shape)
            )

    if leads is None:
        lead_values = np.arange(1, n_lead + 1)
    else:
        lead_values = np.asarray(leads, dtype=object)
        if lead_values.shape != (n_lead,):
            raise ValueError(
                "leads must have shape ({},); got {}".format(n_lead, lead_values.shape)
            )

    season_values = _case_lead_labels(seasons, n_case, n_lead, "seasons", "ALL")
    regions: Mapping[str, np.ndarray]
    if region_weights is None:
        regions = {"india": np.ones(spatial_shape, dtype=np.float64)}
    elif not region_weights:
        raise ValueError("region_weights must contain at least one region")
    else:
        regions = region_weights

    rows = []
    for case_index in range(n_case):
        for lead_index in range(n_lead):
            case_truth = truth_array[case_index, lead_index]
            case_prediction = prediction_array[case_index, lead_index]
            case_truth_anomaly = truth_anomaly_array[case_index, lead_index]
            case_prediction_anomaly = prediction_anomaly_array[case_index, lead_index]
            case_mask = mask_array[case_index, lead_index]
            base_weight = weight_array[case_index, lead_index]

            for region, region_weight in regions.items():
                region_factor = _broadcast_array(
                    np.asarray(region_weight, dtype=np.float64),
                    spatial_shape,
                    "region_weights[{}]".format(region),
                )
                combined_weight = base_weight * region_factor
                common_valid = (
                    case_mask
                    & np.isfinite(case_truth)
                    & np.isfinite(case_prediction)
                    & np.isfinite(combined_weight)
                    & (combined_weight > 0.0)
                )
                rows.append(
                    {
                        "predictor": str(predictor),
                        "case_id": case_values[case_index],
                        "lead": lead_values[lead_index],
                        "region": str(region),
                        "season": season_values[case_index, lead_index],
                        "valid_cells": int(np.count_nonzero(common_valid)),
                        "weight_sum": float(np.sum(combined_weight[common_valid])),
                        "acc": weighted_spatial_acc(
                            case_truth_anomaly,
                            case_prediction_anomaly,
                            combined_weight,
                            common_valid,
                        ),
                        "rmse": weighted_rmse(
                            case_truth,
                            case_prediction,
                            combined_weight,
                            common_valid,
                        ),
                        "mae": weighted_mae(
                            case_truth,
                            case_prediction,
                            combined_weight,
                            common_valid,
                        ),
                        "bias": weighted_bias(
                            case_truth,
                            case_prediction,
                            combined_weight,
                            common_valid,
                        ),
                        "negative_fraction": weighted_negative_fraction(
                            case_prediction, combined_weight, common_valid
                        ),
                    }
                )

    return pd.DataFrame.from_records(rows)


def summarize_metrics(
    case_metrics: pd.DataFrame,
    group_columns: Iterable[str] = ("predictor", "lead", "region", "season"),
    metric_columns: Iterable[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Summarize case-wise scores, giving every forecast case equal weight."""

    groups = list(group_columns)
    metrics = list(metric_columns)
    missing = [name for name in groups + metrics if name not in case_metrics.columns]
    if missing:
        raise ValueError("case_metrics is missing columns: {}".format(", ".join(missing)))

    rows = []
    grouped = case_metrics.groupby(groups, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["case_count"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row["{}_valid_cases".format(metric)] = int(finite.size)
            row["{}_mean".format(metric)] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            row["{}_median".format(metric)] = (
                float(np.median(finite)) if finite.size else float("nan")
            )
            row["{}_std".format(metric)] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _ordered_unique(values: pd.Series) -> np.ndarray:
    """Return deterministic chronological/numeric/string-sorted unique values."""

    unique = pd.Series(values.drop_duplicates().to_list())
    if unique.empty:
        return np.asarray([], dtype=object)
    try:
        return unique.sort_values(kind="mergesort").to_numpy()
    except (TypeError, ValueError):
        order = np.argsort(unique.astype(str).to_numpy(), kind="mergesort")
        return unique.iloc[order].to_numpy()


def _circular_block_indices(
    n_cases: int,
    n_resamples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Draw one shared set of circular moving-block case indices."""

    if n_cases <= 0:
        raise ValueError("at least one paired case is required")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")

    blocks_per_sample = int(np.ceil(float(n_cases) / float(block_length)))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_cases, size=(n_resamples, blocks_per_sample))
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., np.newaxis] + offsets) % n_cases
    return indices.reshape(n_resamples, -1)[:, :n_cases]


def paired_moving_block_bootstrap(
    case_metrics: pd.DataFrame,
    predictor: str,
    baseline: str,
    *,
    metric_columns: Iterable[str] = METRIC_COLUMNS,
    group_columns: Iterable[str] = ("lead", "region", "season"),
    case_column: str = "case_id",
    predictor_column: str = "predictor",
    block_length: int = 13,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Estimate paired model-minus-baseline differences with block bootstrap.

    Circular blocks are drawn along the ordered case axis.  Exactly the same
    bootstrap case indices are used for every lead, region, season, metric, and
    predictor pair, preserving cross-lead dependence.  Only rows present for both
    predictors enter an observed comparison.

    The returned difference is always ``predictor - baseline``.  Consequently,
    positive ACC differences are favourable, while negative RMSE/MAE differences
    are favourable.
    """

    groups = list(group_columns)
    metrics = list(metric_columns)
    required = [case_column, predictor_column] + groups + metrics
    missing = [name for name in required if name not in case_metrics.columns]
    if missing:
        raise ValueError("case_metrics is missing columns: {}".format(", ".join(missing)))
    if predictor == baseline:
        raise ValueError("predictor and baseline must be different")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    selected = case_metrics.loc[
        case_metrics[predictor_column].isin([predictor, baseline]), required
    ].copy()
    present = set(selected[predictor_column].astype(str).unique())
    if str(predictor) not in present or str(baseline) not in present:
        raise ValueError("both predictor and baseline must be present in case_metrics")

    pair_keys = [case_column] + groups
    duplicate = selected.duplicated(pair_keys + [predictor_column], keep=False)
    if duplicate.any():
        examples = selected.loc[duplicate, pair_keys + [predictor_column]].head(3)
        raise ValueError(
            "duplicate predictor rows for a case/group prevent pairing: {}".format(
                examples.to_dict(orient="records")
            )
        )

    model = selected.loc[selected[predictor_column] == predictor, pair_keys + metrics]
    reference = selected.loc[selected[predictor_column] == baseline, pair_keys + metrics]
    paired = model.merge(
        reference,
        how="inner",
        on=pair_keys,
        suffixes=("_model", "_baseline"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("predictor and baseline have no paired cases")

    ordered_cases = _ordered_unique(paired[case_column])
    case_to_index: Dict[object, int] = {
        case: index for index, case in enumerate(ordered_cases.tolist())
    }
    sampled_indices = _circular_block_indices(
        len(ordered_cases), n_resamples, block_length, seed
    )
    lower_percentile = 100.0 * (1.0 - confidence) / 2.0
    upper_percentile = 100.0 - lower_percentile

    rows = []
    grouped = paired.groupby(groups, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = dict(zip(groups, keys))
        case_positions = np.asarray(
            [case_to_index[value] for value in group[case_column].to_list()], dtype=int
        )

        for metric in metrics:
            model_values = pd.to_numeric(
                group["{}_model".format(metric)], errors="coerce"
            ).to_numpy(dtype=float)
            baseline_values = pd.to_numeric(
                group["{}_baseline".format(metric)], errors="coerce"
            ).to_numpy(dtype=float)
            finite = np.isfinite(model_values) & np.isfinite(baseline_values)

            difference_by_case = np.full(len(ordered_cases), np.nan, dtype=np.float64)
            difference_by_case[case_positions[finite]] = (
                model_values[finite] - baseline_values[finite]
            )

            draws = difference_by_case[sampled_indices]
            draw_finite = np.isfinite(draws)
            draw_counts = np.sum(draw_finite, axis=1)
            draw_sums = np.nansum(draws, axis=1)
            bootstrap_means = np.full(n_resamples, np.nan, dtype=np.float64)
            np.divide(
                draw_sums,
                draw_counts,
                out=bootstrap_means,
                where=draw_counts > 0,
            )
            finite_bootstrap = bootstrap_means[np.isfinite(bootstrap_means)]

            row = dict(group_values)
            row.update(
                {
                    "metric": metric,
                    "predictor": predictor,
                    "baseline": baseline,
                    "paired_case_count": int(np.count_nonzero(finite)),
                    "predictor_mean": (
                        float(np.mean(model_values[finite]))
                        if np.any(finite)
                        else float("nan")
                    ),
                    "baseline_mean": (
                        float(np.mean(baseline_values[finite]))
                        if np.any(finite)
                        else float("nan")
                    ),
                    "mean_difference": (
                        float(np.mean(model_values[finite] - baseline_values[finite]))
                        if np.any(finite)
                        else float("nan")
                    ),
                    "ci_lower": (
                        float(np.percentile(finite_bootstrap, lower_percentile))
                        if finite_bootstrap.size
                        else float("nan")
                    ),
                    "ci_upper": (
                        float(np.percentile(finite_bootstrap, upper_percentile))
                        if finite_bootstrap.size
                        else float("nan")
                    ),
                    "bootstrap_std": (
                        float(np.std(finite_bootstrap, ddof=1))
                        if finite_bootstrap.size > 1
                        else float("nan")
                    ),
                    "block_length": int(block_length),
                    "n_resamples": int(n_resamples),
                    "seed": int(seed),
                }
            )
            rows.append(row)

    columns = groups + [
        "metric",
        "predictor",
        "baseline",
        "paired_case_count",
        "predictor_mean",
        "baseline_mean",
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "bootstrap_std",
        "block_length",
        "n_resamples",
        "seed",
    ]
    return pd.DataFrame.from_records(rows, columns=columns)


# Explicit alias for readers who prefer the full statistical name.
paired_circular_moving_block_bootstrap = paired_moving_block_bootstrap


# Backwards-friendly short name used in a few exploratory notebooks.
weighted_acc = weighted_spatial_acc

