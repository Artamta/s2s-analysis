"""Small, non-interactive plotting helpers for adapter evaluation outputs."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import warnings

import matplotlib

# Evaluation normally runs on a compute node without a display.  Set the
# backend before importing pyplot so these functions are safe in batch jobs.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def _prepare_output_path(path: PathLike) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        "none of the required columns are present: " + ", ".join(candidates)
    )


def plot_metric_by_lead(
    summary: pd.DataFrame,
    metric: str,
    output_path: PathLike,
    split: str = "test",
    region: str = "india",
) -> Path:
    """Plot one verification metric against forecast lead for every model.

    Both common summary layouts are accepted:

    * wide: ``model, split, region, lead_week, acc, rmse, ...``
    * tidy: ``model, split, region, lead_week, metric, value``

    Parameters are deliberately simple so the same helper can be called from
    a notebook, CLI, or automated evaluation job.
    """

    if summary.empty:
        raise ValueError("summary is empty")

    data = summary.copy()
    if "split" in data.columns:
        data = data.loc[data["split"].astype(str).str.lower() == split.lower()]
    if "region" in data.columns:
        data = data.loc[data["region"].astype(str).str.lower() == region.lower()]
    if data.empty:
        raise ValueError(f"no rows found for split={split!r}, region={region!r}")

    lead_column = _first_present(data.columns, ("lead_week", "lead", "week"))
    model_column = _first_present(data.columns, ("model", "method", "predictor"))

    if {"metric", "value"}.issubset(data.columns):
        data = data.loc[
            data["metric"].astype(str).str.lower() == metric.lower()
        ].copy()
        value_column = "value"
    else:
        metric_lookup = {str(column).lower(): str(column) for column in data.columns}
        if metric.lower() not in metric_lookup:
            raise ValueError(f"metric {metric!r} is not present in summary")
        value_column = metric_lookup[metric.lower()]

    data[lead_column] = pd.to_numeric(data[lead_column], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna(subset=[lead_column, value_column])
    if data.empty:
        raise ValueError(f"metric {metric!r} has no finite values after filtering")

    # Averaging duplicate rows keeps plotting robust to summaries that retain
    # one row per seed.  Scientific uncertainty should still be reported from
    # the case-level paired bootstrap rather than inferred from these rows.
    data = (
        data.groupby([model_column, lead_column], as_index=False, sort=True)[
            value_column
        ]
        .mean()
        .sort_values([model_column, lead_column])
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    try:
        for model_name, model_rows in data.groupby(model_column, sort=False):
            axis.plot(
                model_rows[lead_column],
                model_rows[value_column],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=str(model_name).replace("_", " "),
            )

        metric_label = metric.upper().replace("_", " ")
        metric_units = (
            " (mm day$^{-1}$)"
            if metric.lower() in {"rmse", "mae", "bias"}
            else ""
        )
        axis.set_xlabel("Lead week")
        axis.set_ylabel(metric_label + metric_units)
        axis.set_title(f"{metric_label} by lead — {region.title()} ({split})")
        axis.grid(True, alpha=0.25, linewidth=0.7)
        axis.set_xticks(sorted(data[lead_column].unique()))
        if metric.lower() == "bias":
            axis.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
        axis.legend(frameon=False, fontsize=8)

        destination = _prepare_output_path(output_path)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    return destination


def plot_training_history(history: pd.DataFrame, path: PathLike) -> Path:
    """Plot training and validation losses from one training run."""

    if history.empty:
        raise ValueError("training history is empty")

    epoch_column = _first_present(history.columns, ("epoch", "step"))
    train_column = _first_present(
        history.columns, ("train_loss", "training_loss", "loss")
    )
    validation_column = _first_present(
        history.columns, ("validation_loss", "val_loss", "valid_loss")
    )

    epochs = pd.to_numeric(history[epoch_column], errors="coerce")
    train_loss = pd.to_numeric(history[train_column], errors="coerce")
    validation_loss = pd.to_numeric(history[validation_column], errors="coerce")
    finite = epochs.notna() & train_loss.notna() & validation_loss.notna()
    if not finite.any():
        raise ValueError("training history has no finite loss rows")

    epochs = epochs.loc[finite]
    train_loss = train_loss.loc[finite]
    validation_loss = validation_loss.loc[finite]

    figure, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    try:
        axis.plot(epochs, train_loss, label="Training", linewidth=1.8)
        axis.plot(epochs, validation_loss, label="Validation", linewidth=1.8)
        best_position = int(np.argmin(validation_loss.to_numpy()))
        axis.scatter(
            [epochs.iloc[best_position]],
            [validation_loss.iloc[best_position]],
            color="black",
            s=22,
            zorder=3,
            label="Best validation",
        )
        axis.set_xlabel("Epoch" if epoch_column == "epoch" else "Step")
        axis.set_ylabel("Loss")
        axis.set_title("Training history")
        axis.grid(True, alpha=0.25, linewidth=0.7)
        axis.legend(frameon=False)

        destination = _prepare_output_path(path)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    return destination


def _map_coordinates(
    latitude: np.ndarray,
    longitude: np.ndarray,
    spatial_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    height, width = spatial_shape

    if latitude.ndim == longitude.ndim == 1:
        if latitude.size != height or longitude.size != width:
            raise ValueError(
                "one-dimensional latitude/longitude lengths must match map shape"
            )
        return np.meshgrid(longitude, latitude)
    if latitude.shape == longitude.shape == spatial_shape:
        return longitude, latitude
    raise ValueError(
        "latitude and longitude must both be one-dimensional or both match map shape"
    )


def plot_mean_maps(
    predictions: Dict[str, np.ndarray],
    latitude: np.ndarray,
    longitude: np.ndarray,
    mask: np.ndarray,
    path: PathLike,
    lead_index: Optional[int] = None,
) -> Path:
    """Plot case-mean maps for several forecasts and/or observations.

    Each mapping value must have shape ``[case, lead, height, width]``.  When
    ``lead_index`` is omitted, the figure shows the mean over both cases and
    all lead weeks.  A shared colour scale makes comparisons visually honest.
    """

    if not predictions:
        raise ValueError("predictions mapping is empty")

    arrays = {name: np.asarray(values) for name, values in predictions.items()}
    first_shape = next(iter(arrays.values())).shape
    if len(first_shape) != 4:
        raise ValueError("prediction arrays must have shape [case, lead, height, width]")
    if any(values.shape != first_shape for values in arrays.values()):
        raise ValueError("all prediction arrays must have the same shape")

    _, leads, height, width = first_shape
    if lead_index is not None and not 0 <= lead_index < leads:
        raise ValueError(f"lead_index must be between 0 and {leads - 1}")

    support = np.asarray(mask, dtype=bool)
    if support.shape != (height, width):
        raise ValueError("mask shape must match prediction spatial shape")
    if not support.any():
        raise ValueError("mask contains no valid grid cells")
    map_longitude, map_latitude = _map_coordinates(
        latitude, longitude, (height, width)
    )

    mean_maps = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for name, values in arrays.items():
            if lead_index is None:
                field = np.nanmean(values, axis=(0, 1))
            else:
                field = np.nanmean(values[:, lead_index], axis=0)
            mean_maps[name] = np.where(support, field, np.nan)

    finite_values = np.concatenate(
        [field[np.isfinite(field)] for field in mean_maps.values()]
    )
    if finite_values.size == 0:
        raise ValueError("prediction maps contain no finite values inside the mask")

    low, high = np.nanpercentile(finite_values, [2.0, 98.0])
    if np.all(finite_values >= 0.0):
        colour_map = "Blues"
        value_min = 0.0
        value_max = max(float(high), np.finfo(float).eps)
    else:
        colour_map = "RdBu_r"
        extent = max(abs(float(low)), abs(float(high)), np.finfo(float).eps)
        value_min, value_max = -extent, extent

    panel_count = len(mean_maps)
    columns = 2 if panel_count == 4 else min(3, panel_count)
    rows = int(np.ceil(panel_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.1 * columns, 3.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    map_artist = None
    try:
        for axis, (name, field) in zip(flat_axes, mean_maps.items()):
            map_artist = axis.pcolormesh(
                map_longitude,
                map_latitude,
                field,
                shading="auto",
                cmap=colour_map,
                vmin=value_min,
                vmax=value_max,
                rasterized=True,
            )
            title = str(name).replace("_", " ").title()
            title = title.replace("Imerg", "IMERG").replace("Fuxi", "FuXi")
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            axis.set_aspect("auto")

        for unused_axis in flat_axes[panel_count:]:
            unused_axis.set_visible(False)

        period = "all leads" if lead_index is None else f"lead week {lead_index + 1}"
        figure.suptitle(f"Mean precipitation maps — {period}", fontsize=12)
        if map_artist is not None:
            colour_bar = figure.colorbar(
                map_artist,
                ax=list(flat_axes[:panel_count]),
                shrink=0.86,
                pad=0.02,
            )
            colour_bar.set_label("Precipitation (mm day$^{-1}$)")

        destination = _prepare_output_path(path)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    return destination
