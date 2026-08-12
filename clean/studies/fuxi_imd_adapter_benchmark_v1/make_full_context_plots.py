#!/usr/bin/env python3
"""Create publication-ready plots for the frozen full-context IMD adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = (
    HERE.parent.parent
    / "bias-correction/results/fuxi_imd_full_context_compact_allweeks/"
    "full_20260811T152024Z"
)
DEFAULT_AUDIT = HERE / "results/full_context_jjas_2022_2024_job91439"
METHODS = ("raw_fuxi", "log_bias", "selected_adapter")
LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Log-bias",
    "selected_adapter": "Full-context adapter",
}
COLORS = {
    "raw_fuxi": "#555555",
    "log_bias": "#CC79A7",
    "selected_adapter": "#0072B2",
}
MARKERS = {"raw_fuxi": "o", "log_bias": "s", "selected_adapter": "^"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(figure: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(figure)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "semibold",
            "legend.fontsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def setup_map(axis: plt.Axes) -> None:
    axis.set_extent((65.0, 100.0, 4.0, 39.5), crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F7F7F7", zorder=0)
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=0)
    axis.coastlines(resolution="50m", linewidth=0.65, color="0.20", zorder=4)
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.50,
        edgecolor="0.30",
        zorder=4,
    )
    axis.set_xticks((70, 80, 90, 100), crs=ccrs.PlateCarree())
    axis.set_yticks((10, 20, 30, 40), crs=ccrs.PlateCarree())
    axis.tick_params(labelsize=8, length=2.5)
    axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.35,
        color="0.75",
        alpha=0.45,
        linestyle=":",
    )


def masked(field: np.ndarray, support: np.ndarray) -> np.ndarray:
    return np.where(support, field, np.nan)


def positive_area_fraction(
    skill: np.ndarray, area_weight: np.ndarray, support: np.ndarray
) -> float:
    valid = support & np.isfinite(skill) & (area_weight > 0.0)
    weights = np.where(valid, area_weight, 0.0)
    return float(100.0 * weights[skill > 0.0].sum() / weights.sum())


def load_and_verify(
    run: Path, audit: Path
) -> tuple[
    xr.Dataset,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict,
    np.ndarray,
]:
    run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    audit_manifest = json.loads(
        (audit / "manifest.json").read_text(encoding="utf-8")
    )
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if (
        run_manifest.get("status") != "complete"
        or audit_manifest.get("status") != "complete"
        or selection.get("status") != "frozen"
    ):
        raise ValueError("training run, audit, or selection is not complete/frozen")
    if selection.get("selected_model") != "normal_climo_model":
        raise ValueError("plot contract expects the selected compact normal model")
    if audit_manifest.get("adapter_selection_sha256") != sha256_file(
        run / "selection.json"
    ):
        raise ValueError("audit does not reference the requested frozen selection")
    cases = pd.read_csv(audit / "case_metrics.csv")
    summary = pd.read_csv(audit / "summary_by_lead_region.csv")
    if len(cases) != 9_000 or cases.init.nunique() != 100:
        raise ValueError("audit case table is not the frozen 100-case contract")
    if set(pd.DatetimeIndex(cases.init).year) != {2022, 2023, 2024}:
        raise ValueError("audit years differ from 2022-2024")
    dataset = xr.open_zarr(audit / "predictions.zarr", consolidated=True).load()
    expected_sizes = {
        "method": 3,
        "init": 100,
        "lead_week": 6,
        "latitude": 27,
        "longitude": 27,
    }
    if dict(dataset.sizes) != expected_sizes:
        raise ValueError(f"audit prediction dimensions changed: {dict(dataset.sizes)}")
    if tuple(dataset.method.values.tolist()) != METHODS:
        raise ValueError("audit prediction methods changed")
    support = np.asarray(dataset.adapter_support.values, dtype=bool)
    if int(support.sum()) != 171:
        raise ValueError("IMD support is not 171 cells")
    with xr.open_zarr(
        run / "predictions.zarr", consolidated=True
    ) as training_predictions:
        area_weight = np.asarray(
            training_predictions.area_weight_km2.load().values, dtype=np.float64
        )
    if np.count_nonzero(area_weight) != 171:
        raise ValueError("training area weights differ from 171-cell support")
    if not np.array_equal(area_weight > 0.0, support):
        raise ValueError("positive training area weights differ from IMD support")

    rebuilt = (
        cases.groupby(
            ["region", "region_label", "method", "method_label", "lead_week"],
            as_index=False,
        )[["rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"]]
        .mean()
        .sort_values(["region", "method", "lead_week"])
        .reset_index(drop=True)
    )
    stored = summary.sort_values(["region", "method", "lead_week"]).reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        rebuilt[stored.columns],
        stored,
        check_dtype=False,
        check_exact=False,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    return dataset, cases, summary, run_manifest, audit_manifest, area_weight


def compute_spatial_diagnostics(
    dataset: xr.Dataset, support: np.ndarray
) -> xr.Dataset:
    prediction = np.asarray(dataset.prediction.values, dtype=np.float64)
    truth = np.asarray(dataset.truth_imd.values, dtype=np.float64)
    height, width = support.shape
    mean_prediction = np.full((len(METHODS), height, width), np.nan)
    mean_truth = np.full((height, width), np.nan)
    rmse = np.full((len(METHODS), height, width), np.nan)
    bias = np.full((len(METHODS), height, width), np.nan)
    lead_rmse = np.full((len(METHODS), 6, height, width), np.nan)
    supported_prediction = prediction[..., support]
    supported_truth = truth[..., support]
    truth_is_valid = np.isfinite(supported_truth)
    common_prediction = np.where(
        truth_is_valid[None], supported_prediction, np.nan
    )
    supported_error = common_prediction - supported_truth[None]
    mean_prediction[:, support] = np.nanmean(common_prediction, axis=(1, 2))
    mean_truth[support] = np.nanmean(supported_truth, axis=(0, 1))
    rmse[:, support] = np.sqrt(np.nanmean(supported_error**2, axis=(1, 2)))
    bias[:, support] = np.nanmean(supported_error, axis=(1, 2))
    lead_rmse[:, :, support] = np.sqrt(
        np.nanmean(supported_error**2, axis=1)
    )
    finite_fields = {
        "mean_prediction": mean_prediction[:, support],
        "mean_truth": mean_truth[support],
        "rmse": rmse[:, support],
        "bias": bias[:, support],
        "lead_rmse": lead_rmse[:, :, support],
    }
    for name, values in finite_fields.items():
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {name} remains on IMD-supported cells")
    fields = xr.Dataset(
        {
            "mean_prediction": (
                ("method", "latitude", "longitude"),
                np.where(support[None], mean_prediction, np.nan).astype(np.float32),
            ),
            "mean_truth": (
                ("latitude", "longitude"),
                masked(mean_truth, support).astype(np.float32),
            ),
            "rmse": (
                ("method", "latitude", "longitude"),
                np.where(support[None], rmse, np.nan).astype(np.float32),
            ),
            "bias": (
                ("method", "latitude", "longitude"),
                np.where(support[None], bias, np.nan).astype(np.float32),
            ),
            "lead_rmse": (
                ("method", "lead_week", "latitude", "longitude"),
                np.where(support[None, None], lead_rmse, np.nan).astype(np.float32),
            ),
            "rmse_reduction_vs_raw": (
                ("method", "latitude", "longitude"),
                np.where(support[None], rmse[0][None] - rmse, np.nan).astype(
                    np.float32
                ),
            ),
            "lead_rmse_reduction_vs_raw": (
                ("method", "lead_week", "latitude", "longitude"),
                np.where(
                    support[None, None], lead_rmse[0][None] - lead_rmse, np.nan
                ).astype(np.float32),
            ),
        },
        coords={
            "method": list(METHODS),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": dataset.latitude.values,
            "longitude": dataset.longitude.values,
        },
        attrs={
            "scope": "2022-2024 JJAS; 100 starts; W1-W6",
            "rmse_definition": "cellwise sqrt(mean squared error over cases and leads)",
            "lead_rmse_definition": "cellwise sqrt(mean squared error over cases)",
            "positive_reduction_is_better": "true",
            "units": "mm day-1",
        },
    )
    return fields


def plot_result_dashboard(summary: pd.DataFrame, output: Path) -> None:
    selected = summary.loc[summary.region.eq("all_india")]
    panels = (
        ("rmse_mm_day", "RMSE (mm day$^{-1}$)", "lower"),
        ("acc", "Spatial ACC", "higher"),
        ("mae_mm_day", "MAE (mm day$^{-1}$)", "lower"),
        ("bias_mm_day", "Bias (mm day$^{-1}$)", "zero"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), sharex=True)
    for panel_index, (axis, (metric, label, direction)) in enumerate(
        zip(axes.flat, panels)
    ):
        for method in METHODS:
            values = selected.loc[selected.method.eq(method)].sort_values(
                "lead_week"
            )
            axis.plot(
                values.lead_week,
                values[metric],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.4 if method == "selected_adapter" else 1.8,
                markersize=6,
                markerfacecolor=("white" if method == "raw_fuxi" else COLORS[method]),
                markeredgecolor=COLORS[method],
                linestyle="--" if method == "raw_fuxi" else "-",
                label=LABELS[method],
            )
        axis.set_title(f"({chr(97 + panel_index)}) {label} · {direction} is better")
        axis.set_xticks(range(1, 7), [f"W{week}" for week in range(1, 7)])
        axis.set_xlabel("Lead week")
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
        if metric == "bias_mm_day":
            axis.axhline(0.0, color="0.40", linewidth=0.9, linestyle=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.suptitle(
        "FuXi–IMD full-context adapter · frozen 2022–2024 JJAS audit\n"
        "100 operational starts · 171 India cells · no 2025 initialization",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.5,
        0.015,
        "Mean case-wise scores; ACC uses one common fixed 2002–2017 IMD climatology.",
        ha="center",
        color="0.35",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.90))
    save_figure(figure, output / "01_generalization_result_dashboard")


def plot_spatial_overview(
    diagnostics: xr.Dataset, support: np.ndarray, output: Path
) -> None:
    longitude = diagnostics.longitude.values
    latitude = diagnostics.latitude.values
    mean_fields = [
        diagnostics.mean_truth.values,
        diagnostics.mean_prediction.sel({"method": "raw_fuxi"}).values,
        diagnostics.mean_prediction.sel({"method": "log_bias"}).values,
        diagnostics.mean_prediction.sel({"method": "selected_adapter"}).values,
    ]
    rmse_fields = [
        diagnostics.rmse.sel({"method": "raw_fuxi"}).values,
        diagnostics.rmse.sel({"method": "log_bias"}).values,
        diagnostics.rmse.sel({"method": "selected_adapter"}).values,
    ]
    correction = mean_fields[-1] - mean_fields[1]
    skill = rmse_fields[0] - rmse_fields[-1]
    mean_limit = float(
        np.nanpercentile(np.stack(mean_fields[:3])[:, support], 99.0)
    )
    rmse_limit = float(np.nanpercentile(np.stack(rmse_fields)[:, support], 99.0))
    correction_limit = max(
        0.25, float(np.nanpercentile(np.abs(correction[support]), 98.0))
    )
    skill_limit = max(0.25, float(np.nanpercentile(np.abs(skill[support]), 98.0)))

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(15.2, 8.3),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    top_titles = ("IMD truth", "Raw FuXi", "Log-bias", "Full-context adapter")
    mean_images = []
    for axis, field, title in zip(axes[0], mean_fields, top_titles):
        setup_map(axis)
        image = axis.pcolormesh(
            longitude,
            latitude,
            masked(field, support),
            shading="nearest",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=mean_limit,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        mean_images.append(image)
        axis.set_title(title)
    bottom_fields = [*rmse_fields, skill]
    bottom_titles = (
        "Raw FuXi RMSE",
        "Log-bias RMSE",
        "Adapter RMSE",
        "RMSE reduction\n(raw − adapter)",
    )
    rmse_images = []
    for index, (axis, field, title) in enumerate(
        zip(axes[1], bottom_fields, bottom_titles)
    ):
        setup_map(axis)
        if index < 3:
            image = axis.pcolormesh(
                longitude,
                latitude,
                masked(field, support),
                shading="nearest",
                cmap="magma",
                vmin=0.0,
                vmax=rmse_limit,
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
        else:
            image = axis.pcolormesh(
                longitude,
                latitude,
                masked(field, support),
                shading="nearest",
                cmap="RdBu",
                norm=TwoSlopeNorm(vmin=-skill_limit, vcenter=0.0, vmax=skill_limit),
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
        rmse_images.append(image)
        axis.set_title(title)
    figure.colorbar(
        mean_images[0],
        ax=axes[0, :],
        orientation="horizontal",
        fraction=0.045,
        pad=0.055,
        extend="max",
        label="Mean rainfall (mm day$^{-1}$)",
    )
    figure.colorbar(
        rmse_images[0],
        ax=axes[1, :3],
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
        extend="max",
        label="Cellwise RMSE (mm day$^{-1}$)",
    )
    figure.colorbar(
        rmse_images[-1],
        ax=axes[1, 3],
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
        label="Positive = adapter improves (mm day$^{-1}$)",
        extend="both",
    )
    figure.suptitle(
        "Spatial comparison over India · frozen 2022–2024 JJAS audit\n"
        "Pooled over 100 starts and all six lead weeks",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output / "02_spatial_field_comparison")


def plot_spatial_skill_baselines(
    diagnostics: xr.Dataset,
    support: np.ndarray,
    area_weight: np.ndarray,
    output: Path,
) -> None:
    longitude = diagnostics.longitude.values
    latitude = diagnostics.latitude.values
    raw = diagnostics.rmse.sel({"method": "raw_fuxi"}).values
    log_bias = diagnostics.rmse.sel({"method": "log_bias"}).values
    adapter = diagnostics.rmse.sel({"method": "selected_adapter"}).values
    skills = (raw - adapter, log_bias - adapter)
    limit = max(
        0.20,
        float(
            np.nanpercentile(
                np.abs(np.stack(skills)[:, support]),
                98.0,
            )
        ),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    images = []
    for axis, field, baseline in zip(axes, skills, ("raw FuXi", "log-bias")):
        setup_map(axis)
        image = axis.pcolormesh(
            longitude,
            latitude,
            masked(field, support),
            shading="nearest",
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        images.append(image)
        fraction = positive_area_fraction(field, area_weight, support)
        axis.set_title(f"Adapter vs {baseline}\n{fraction:.1f}% of India area improved")
    figure.colorbar(
        images[0],
        ax=axes,
        orientation="horizontal",
        fraction=0.075,
        pad=0.08,
        extend="both",
        label="Local W1–W6 RMSE reduction (mm day$^{-1}$); positive is better",
    )
    figure.suptitle(
        "Where the full-context adapter improves rainfall error\n"
        "2022–2024 JJAS · 100 starts",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output / "03_spatial_rmse_skill_vs_baselines")


def plot_spatial_skill_by_lead(
    diagnostics: xr.Dataset,
    support: np.ndarray,
    area_weight: np.ndarray,
    output: Path,
    *,
    baseline: str,
    stem: str,
) -> None:
    longitude = diagnostics.longitude.values
    latitude = diagnostics.latitude.values
    baseline_rmse = diagnostics.lead_rmse.sel({"method": baseline}).values
    adapter_rmse = diagnostics.lead_rmse.sel(
        {"method": "selected_adapter"}
    ).values
    skill = baseline_rmse - adapter_rmse
    limit = max(
        0.25,
        float(np.nanpercentile(np.abs(skill[:, support]), 98.0)),
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.7, 8.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    images = []
    for lead_index, axis in enumerate(axes.flat):
        setup_map(axis)
        field = skill[lead_index]
        image = axis.pcolormesh(
            longitude,
            latitude,
            masked(field, support),
            shading="nearest",
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        images.append(image)
        fraction = positive_area_fraction(field, area_weight, support)
        axis.set_title(f"W{lead_index + 1} · {fraction:.1f}% area improved")
    figure.colorbar(
        images[0],
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.055,
        extend="both",
        label=(
            f"Local RMSE reduction vs {LABELS[baseline]} "
            "(mm day$^{-1}$); positive is better"
        ),
    )
    figure.suptitle(
        "Spatial RMSE improvement at every lead week\n"
        f"Full-context adapter versus {LABELS[baseline]} · frozen 2022–2024 JJAS audit",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output / stem)


def plot_spatial_bias(
    diagnostics: xr.Dataset,
    support: np.ndarray,
    area_weight: np.ndarray,
    output: Path,
) -> None:
    longitude = diagnostics.longitude.values
    latitude = diagnostics.latitude.values
    fields = [
        diagnostics.bias.sel({"method": method}).values for method in METHODS
    ]
    limit = max(
        0.5,
        float(np.nanpercentile(np.abs(np.stack(fields)[:, support]), 98.0)),
    )
    weights = np.where(support, area_weight, 0.0)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    images = []
    for axis, field, method in zip(axes, fields, METHODS):
        setup_map(axis)
        image = axis.pcolormesh(
            longitude,
            latitude,
            masked(field, support),
            shading="nearest",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        images.append(image)
        weighted_bias = float(np.nansum(field * weights) / weights.sum())
        axis.set_title(f"{LABELS[method]}\nIndia bias {weighted_bias:+.3f} mm day$^{{-1}}$")
    figure.colorbar(
        images[0],
        ax=axes,
        orientation="horizontal",
        fraction=0.07,
        pad=0.08,
        extend="both",
        label="Local mean bias (mm day$^{-1}$); blue = dry, red = wet",
    )
    figure.suptitle(
        "Spatial rainfall-bias comparison · pooled W1–W6\n"
        "Frozen 2022–2024 JJAS audit",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, output / "06_spatial_bias_comparison")


def plot_train_validation(run: Path, run_manifest: dict, output: Path) -> None:
    selected_name = str(run_manifest["selected_model"])
    metadata = run_manifest["training"][selected_name]
    records = metadata["runs"]
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), sharey=True)
    colors = ("#0072B2", "#E69F00", "#009E73")
    all_losses: list[np.ndarray] = []
    for axis, color, record in zip(axes, colors, records):
        history = pd.read_csv(run / record["history"])
        display_epoch = history.epoch.to_numpy() + 1
        all_losses.extend(
            [history.train_loss.to_numpy(), history.validation_loss.to_numpy()]
        )
        axis.plot(
            display_epoch,
            history.train_loss,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label="Train",
        )
        axis.plot(
            display_epoch,
            history.validation_loss,
            color=color,
            linewidth=2.2,
            label="Validation",
        )
        best_epoch = int(record["best_epoch"]) + 1
        best_loss = float(record["best_validation_loss"])
        axis.scatter(
            [best_epoch],
            [best_loss],
            marker="*",
            s=120,
            color="black",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
            label="Saved checkpoint",
        )
        axis.annotate(
            f"best epoch {best_epoch}\n{best_loss:.4f}",
            xy=(best_epoch, best_loss),
            xytext=(5, -32),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.7},
        )
        axis.set_title(f"Seed {int(record['seed'])}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.22)
    low = min(float(np.min(values)) for values in all_losses)
    high = max(float(np.max(values)) for values in all_losses)
    margin = 0.05 * (high - low)
    for axis in axes:
        axis.set_ylim(low - margin, high + margin)
    axes[0].set_ylabel("Composite objective (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    figure.suptitle(
        "Selected full-context model: training and blocked validation curves\n"
        "2002–2017 train · 2018–2019 validation · early-stopped checkpoints",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    figure.text(
        0.5,
        -0.01,
        "Objective = 0.75 Smooth-L1 residual + 0.20 spatial ACC loss + 0.05 mean-bias penalty.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    figure.tight_layout(rect=(0.01, 0.04, 0.99, 0.87))
    save_figure(figure, output / "07_train_validation_curves")


def output_files(output: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "plot_manifest.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    audit = args.audit.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_style()
    dataset, _, summary, run_manifest, audit_manifest, area_weight = load_and_verify(
        run, audit
    )
    support = np.asarray(dataset.adapter_support.values, dtype=bool)
    diagnostics = compute_spatial_diagnostics(dataset, support)
    diagnostics.to_netcdf(output / "spatial_diagnostics.nc")
    plot_result_dashboard(summary, output)
    plot_spatial_overview(diagnostics, support, output)
    plot_spatial_skill_baselines(diagnostics, support, area_weight, output)
    plot_spatial_skill_by_lead(
        diagnostics,
        support,
        area_weight,
        output,
        baseline="raw_fuxi",
        stem="04_spatial_rmse_skill_vs_raw_by_lead",
    )
    plot_spatial_skill_by_lead(
        diagnostics,
        support,
        area_weight,
        output,
        baseline="log_bias",
        stem="05_spatial_rmse_skill_vs_logbias_by_lead",
    )
    plot_spatial_bias(diagnostics, support, area_weight, output)
    plot_train_validation(run, run_manifest, output)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    readme = [
        "# Full-context FuXi–IMD figure set",
        "",
        "Frozen model and audit plots; generating these figures does not retrain or reselect the model.",
        "",
        "- `01_generalization_result_dashboard`: W1–W6 RMSE, ACC, MAE, and bias",
        "- `02_spatial_field_comparison`: mean rainfall and local RMSE maps",
        "- `03_spatial_rmse_skill_vs_baselines`: pooled spatial improvement versus raw FuXi and log-bias",
        "- `04_spatial_rmse_skill_vs_raw_by_lead`: spatial improvement versus raw FuXi separately for W1–W6",
        "- `05_spatial_rmse_skill_vs_logbias_by_lead`: spatial improvement versus log-bias separately for W1–W6",
        "- `06_spatial_bias_comparison`: pooled local rainfall bias for all three methods",
        "- `07_train_validation_curves`: selected-model curves and saved checkpoints for all three seeds",
        "- `spatial_diagnostics.nc`: numerical fields plotted in the spatial figures",
        "",
        f"Training run: `{run}`",
        f"Audit: `{audit}`",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_run": str(run),
        "training_manifest_sha256": sha256_file(run / "manifest.json"),
        "audit": str(audit),
        "audit_manifest_sha256": sha256_file(audit / "manifest.json"),
        "selected_model": run_manifest["selected_model"],
        "selected_alpha": run_manifest["selected_alpha"],
        "audit_case_count": audit_manifest["audit_case_count"],
        "spatial_support_cells": int(support.sum()),
        "outputs": {
            str(path.relative_to(output)): sha256_file(path)
            for path in output_files(output)
        },
    }
    (output / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: generated and numerically reconciled seven-figure set: {output}")


if __name__ == "__main__":
    main()
