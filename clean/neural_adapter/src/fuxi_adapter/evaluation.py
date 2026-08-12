"""Prediction reconstruction, case-wise verification, and portable outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .data import AdapterData, SplitArrays
from .metrics import compute_case_metrics, summarize_metrics


def season_labels(valid_dates: np.ndarray) -> np.ndarray:
    """Season of the fourth (middle) day of every verification week."""

    midpoint = np.asarray(valid_dates, dtype="datetime64[D]")[:, :, 3]
    months = midpoint.astype("datetime64[M]").astype(int) % 12 + 1
    labels = np.full(months.shape, "SON", dtype="<U3")
    labels[np.isin(months, [12, 1, 2])] = "DJF"
    labels[np.isin(months, [3, 4, 5])] = "MAM"
    labels[np.isin(months, [6, 7, 8])] = "JJA"
    return labels


def region_factors(data: AdapterData) -> Dict[str, np.ndarray]:
    """Convert archived absolute regional weights to factors of India weights."""

    base = np.asarray(data.area_weight_km2, dtype=np.float64)
    factors: Dict[str, np.ndarray] = {}
    for name, absolute in data.region_weight_km2.items():
        factor = np.zeros_like(base)
        np.divide(
            np.asarray(absolute, dtype=np.float64),
            base,
            out=factor,
            where=base > 0,
        )
        factors[name] = factor
    return factors


def evaluate_prediction_set(
    split: SplitArrays,
    data: AdapterData,
    predictions: Mapping[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every predictor identically against fixed-climatology IMERG."""

    case_frames = []
    truth_anomaly = split.imerg_truth - split.imerg_climatology
    identifiers = [
        np.datetime_as_string(value, unit="D") for value in split.initializations
    ]
    seasons = season_labels(split.valid_dates)
    support = data.area_weight_km2 > 0
    regions = region_factors(data)
    for predictor, prediction in predictions.items():
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != split.imerg_truth.shape:
            raise ValueError(f"{predictor}: prediction shape does not match truth")
        frame = compute_case_metrics(
            split.imerg_truth,
            prediction,
            truth_anomaly,
            prediction - split.imerg_climatology,
            data.area_weight_km2,
            predictor=predictor,
            case_ids=identifiers,
            leads=np.arange(1, split.lead_count + 1),
            seasons=seasons,
            region_weights=regions,
            valid_mask=support,
        )
        frame.insert(0, "split", split.name)
        case_frames.append(frame)
    case_metrics = pd.concat(case_frames, ignore_index=True)
    summary = summarize_metrics(
        case_metrics,
        group_columns=("split", "predictor", "lead", "region"),
    )
    seasonal_summary = summarize_metrics(
        case_metrics,
        group_columns=("split", "predictor", "lead", "region", "season"),
    )
    return case_metrics, summary, seasonal_summary


def write_prediction_store(
    path: Path,
    split: SplitArrays,
    data: AdapterData,
    predictions: Mapping[str, np.ndarray],
    attrs: Mapping[str, object],
) -> None:
    """Write one self-describing Zarr containing truth, baseline, and predictions."""

    variables = {
        "truth_imerg": (
            ("init", "lead_week", "latitude", "longitude"),
            split.imerg_truth.astype(np.float32),
        ),
        "imerg_climatology": (
            ("init", "lead_week", "latitude", "longitude"),
            split.imerg_climatology.astype(np.float32),
        ),
    }
    for name, values in predictions.items():
        variables[name] = (
            ("init", "lead_week", "latitude", "longitude"),
            np.asarray(values, dtype=np.float32),
        )
    dataset = xr.Dataset(
        variables,
        coords={
            # Explicit nanosecond precision avoids xarray's noisy implicit
            # conversion warning while preserving the exact daily dates.
            "init": split.initializations.astype("datetime64[ns]"),
            "lead_week": np.arange(1, split.lead_count + 1, dtype=np.int16),
            "latitude": data.latitude,
            "longitude": data.longitude,
        },
        attrs={key: str(value) for key, value in attrs.items()},
    )
    for variable in variables:
        dataset[variable].attrs.update(
            {"units": "mm day-1", "temporal_statistic": "complete_7_day_mean"}
        )
    dataset.to_zarr(str(path), mode="w", consolidated=True)


def write_metric_tables(
    directory: Path,
    case_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    seasonal_summary: pd.DataFrame,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    case_metrics.to_csv(directory / "case_metrics.csv", index=False)
    summary.to_csv(directory / "summary_by_week_region.csv", index=False)
    seasonal_summary.to_csv(
        directory / "summary_by_week_region_season.csv", index=False
    )
