#!/usr/bin/env python3
"""Render professional bar summaries and all 31 IMERG anomaly-map pages."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_SCRIPT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_SCRIPT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import PathPatch
from cartopy.mpl.path import shapely_to_path
import numpy as np
import pandas as pd
from PIL import Image
from shapely import contains_xy
import xarray as xr


HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
MAP_SUPPORT_SCRIPT = (
    WORKSPACE
    / "deliverables/fuxi_erpas_acc_multiseason_2023_2024/scripts/"
    "plot_jjas_composite_anomaly_maps.py"
)
FIELDS = HERE / "data/processed/review_fields_2023_2024.nc"
METRICS_FILE = HERE / "metrics/per_case_metrics_2023_2024.csv"
SUMMARY_FILE = HERE / "metrics/summary_metrics_2023_2024.csv"
SELECTED_FILE = HERE / "metrics/selected_spatial_initializations_2023_2024.csv"
FIGURES = HERE / "figures"
IC_FIGURES = FIGURES / "all_ic_maps"
PRESENTATION = FIGURES / "presentation_ic_maps"
LOGS = HERE / "logs"

FUXI = "#1B6CA8"
ERPAS = "#C65D32"
INK = "#172A36"
MUTED = "#536773"
GRID = "#D8E1E6"
MODELS = ("FuXi-S2S", "ERPAS")
REFERENCES = ("IMD", "IMERG Final V07B")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


maps = import_file("review_map_support", MAP_SUPPORT_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--skip-maps", action="store_true")
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grouped_bars(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    subtitle: str,
    stem: Path,
    higher_is_better: bool,
    dpi: int,
) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7), sharey=True, facecolor="white")
    weeks = np.arange(1, 5)
    width = 0.34
    offsets = {"FuXi-S2S": -width / 2, "ERPAS": width / 2}
    colors = {"FuXi-S2S": FUXI, "ERPAS": ERPAS}
    all_q25 = summary[f"{metric}_q25"].to_numpy(dtype=float)
    all_q75 = summary[f"{metric}_q75"].to_numpy(dtype=float)
    if metric == "acc":
        lower = min(-0.10, math.floor((float(np.min(all_q25)) - 0.05) * 10) / 10)
        upper = max(0.65, math.ceil((float(np.max(all_q75)) + 0.07) * 10) / 10)
    else:
        lower = 0.0
        upper = math.ceil((float(np.max(all_q75)) + 0.35) * 2) / 2

    for axis, reference in zip(axes, REFERENCES):
        panel = summary[summary.reference == reference]
        value_lookup: dict[str, np.ndarray] = {}
        for model in MODELS:
            rows = panel[panel.model == model].sort_values("week")
            if len(rows) != 4 or not (rows.n_cases == 31).all():
                raise ValueError(f"bar input failed for {reference}/{model}/{metric}")
            values = rows[f"{metric}_mean"].to_numpy(dtype=float)
            q25 = rows[f"{metric}_q25"].to_numpy(dtype=float)
            q75 = rows[f"{metric}_q75"].to_numpy(dtype=float)
            value_lookup[model] = values
            positions = weeks + offsets[model]
            bars = axis.bar(
                positions,
                values,
                width=width * 0.88,
                color=colors[model],
                edgecolor="white",
                linewidth=0.8,
                label=model,
                zorder=3,
            )
            whisker_color = colors[model]
            axis.vlines(
                positions,
                q25,
                q75,
                color=whisker_color,
                linewidth=1.35,
                alpha=0.78,
                zorder=5,
            )
            for x, low, high in zip(positions, q25, q75):
                axis.plot(
                    [x - 0.045, x + 0.045],
                    [low, low],
                    color=whisker_color,
                    lw=1.35,
                    alpha=0.78,
                    zorder=5,
                )
                axis.plot(
                    [x - 0.045, x + 0.045],
                    [high, high],
                    color=whisker_color,
                    lw=1.35,
                    alpha=0.78,
                    zorder=5,
                )
            for bar, value in zip(bars, values):
                vertical = 0.018 * (upper - lower)
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + vertical,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.0,
                    fontweight="bold",
                    color=colors[model],
                    bbox={
                        "boxstyle": "round,pad=0.10",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.94,
                    },
                )

        if metric == "acc":
            differences = value_lookup["FuXi-S2S"] - value_lookup["ERPAS"]
        else:
            differences = value_lookup["ERPAS"] - value_lookup["FuXi-S2S"]
        delta_y = lower + 0.055 * (upper - lower)
        for week, difference in zip(weeks, differences):
            axis.text(
                week,
                delta_y,
                f"{difference:+.2f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color=FUXI if difference >= 0 else ERPAS,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": GRID,
                    "linewidth": 0.7,
                    "alpha": 0.95,
                },
                zorder=7,
            )
        axis.set_title(
            "Verified against IMD" if reference == "IMD" else "Verified against IMERG Final V07B",
            fontsize=13,
            fontweight="bold",
            color=INK,
            pad=12,
        )
        axis.set_xticks(weeks, [f"Week {week}" for week in weeks])
        axis.set_xlim(0.55, 4.45)
        axis.set_ylim(lower, upper)
        axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8, zorder=0)
        axis.axhline(0, color="#75868F", linewidth=0.8, zorder=1)
        axis.tick_params(axis="x", labelsize=9.5)
    axes[0].set_ylabel(ylabel, fontsize=11.2, fontweight="semibold")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.965, 0.935), frameon=False, ncol=2)
    fig.suptitle(title, x=0.065, y=0.985, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.065, 0.925, subtitle, fontsize=10.0, color=MUTED)
    direction = "Higher is better" if higher_is_better else "Lower is better"
    delta_definition = (
        "Boxed Δ = FuXi−ERPAS ACC"
        if metric == "acc"
        else "Boxed reduction = ERPAS−FuXi MAE"
    )
    fig.text(
        0.065,
        0.035,
        f"{direction}. {delta_definition}; positive favors FuXi. Bars: arithmetic mean across 31 paired initializations. Whiskers: interquartile case range.",
        fontsize=8.8,
        color=MUTED,
    )
    fig.text(
        0.065,
        0.014,
        "Same 1.5° India grid and area weights. FuXi uses native 2002–2021 climatology; ERPAS uses provider climatology; IMD 1991–2020; IMERG 2001–2022.",
        fontsize=8.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.13, wspace=0.10)
    save_figure(fig, stem, dpi)


def imerg_acc_mae_rmse_headline(summary: pd.DataFrame, stem: Path, dpi: int) -> None:
    """Render the IMERG-only, meeting-ready ACC, MAE and RMSE headline."""
    set_style()
    panel = summary[summary.reference == "IMERG Final V07B"].copy()
    if len(panel) != 8 or set(panel.model) != set(MODELS):
        raise ValueError("IMERG combined figure requires two models x four weeks")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 7.6),
        facecolor="white",
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.20},
    )
    weeks = np.arange(1, 5)
    width = 0.34
    offsets = {"FuXi-S2S": -width / 2, "ERPAS": width / 2}
    colors = {"FuXi-S2S": FUXI, "ERPAS": ERPAS}
    specifications = (
        {
            "metric": "acc",
            "label": "Spatial anomaly correlation (ACC)",
            "title": "A   ACC: anomaly-pattern skill",
            "limits": (-0.17, 0.79),
            "delta_y": -0.115,
            "better": "Higher is better",
            "delta_definition": "ΔACC = FuXi − ERPAS",
            "value_format": ".2f",
        },
        {
            "metric": "mae_mm_day",
            "label": "Mean absolute error (mm/day)",
            "title": "B   MAE: rainfall-magnitude error",
            "limits": (0.0, 6.65),
            "delta_y": 0.42,
            "better": "Lower is better",
            "delta_definition": "MAE reduction = ERPAS − FuXi",
            "value_format": ".2f",
        },
        {
            "metric": "rmse_mm_day",
            "label": "Root-mean-square error (mm/day)",
            "title": "C   RMSE: large-error sensitivity",
            "limits": (0.0, 8.35),
            "delta_y": 0.52,
            "better": "Lower is better",
            "delta_definition": "RMSE reduction = ERPAS − FuXi",
            "value_format": ".2f",
        },
    )

    for axis, spec in zip(axes, specifications):
        metric = spec["metric"]
        model_values: dict[str, np.ndarray] = {}
        axis.set_facecolor("#FBFCFD")
        for model in MODELS:
            rows = panel[panel.model == model].sort_values("week")
            if len(rows) != 4 or not (rows.n_cases == 31).all():
                raise ValueError(f"invalid IMERG combined input for {model}/{metric}")
            values = rows[f"{metric}_mean"].to_numpy(dtype=float)
            q25 = rows[f"{metric}_q25"].to_numpy(dtype=float)
            q75 = rows[f"{metric}_q75"].to_numpy(dtype=float)
            model_values[model] = values
            positions = weeks + offsets[model]
            bars = axis.bar(
                positions,
                values,
                width=width * 0.88,
                color=colors[model],
                edgecolor="white",
                linewidth=0.9,
                label=model,
                zorder=3,
            )
            whisker_color = colors[model]
            axis.vlines(
                positions,
                q25,
                q75,
                color=whisker_color,
                linewidth=1.4,
                alpha=0.80,
                zorder=5,
            )
            for x, low, high in zip(positions, q25, q75):
                axis.plot(
                    [x - 0.045, x + 0.045],
                    [low, low],
                    color=whisker_color,
                    lw=1.4,
                    alpha=0.80,
                    zorder=5,
                )
                axis.plot(
                    [x - 0.045, x + 0.045],
                    [high, high],
                    color=whisker_color,
                    lw=1.4,
                    alpha=0.80,
                    zorder=5,
                )
            value_offset = 0.022 if metric == "acc" else 0.10
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + value_offset,
                    format(value, spec["value_format"]),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                    color=colors[model],
                    bbox={
                        "boxstyle": "round,pad=0.11",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.95,
                    },
                    zorder=7,
                )

        differences = (
            model_values["FuXi-S2S"] - model_values["ERPAS"]
            if metric == "acc"
            else model_values["ERPAS"] - model_values["FuXi-S2S"]
        )
        for week, difference in zip(weeks, differences):
            axis.text(
                week,
                spec["delta_y"],
                f"+{difference:.2f}",
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold",
                color=FUXI,
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": "white",
                    "edgecolor": "#A9BAC3",
                    "linewidth": 0.8,
                },
                zorder=8,
            )

        axis.set_ylim(*spec["limits"])
        axis.set_xlim(0.55, 4.45)
        axis.set_ylabel(spec["label"], fontsize=11.0, fontweight="semibold")
        axis.set_title(
            spec["title"], loc="left", fontsize=13.2, fontweight="bold", color=INK, pad=9
        )
        axis.text(
            0.99,
            0.975,
            f"{spec['better']}  •  {spec['delta_definition']}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            color=MUTED,
        )
        axis.axhline(0, color="#7D8E96", linewidth=0.85, zorder=1)
        axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9, zorder=0)
        axis.tick_params(axis="both", labelsize=9.5)

    for axis in axes:
        axis.set_xticks(weeks, [f"Week {week}" for week in weeks])
        axis.set_xlabel(
            "Forecast verification week", fontsize=10.6, fontweight="semibold"
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.895),
        frameon=False,
        ncol=2,
        fontsize=10.2,
    )
    fig.suptitle(
        "FuXi-S2S shows higher pattern skill and lower rainfall error",
        x=0.055,
        y=0.980,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.925,
        "IMERG Final V07B verification  •  31 paired JJAS starts, 2023–2024  •  identical valid dates and common 1.5° India grid",
        fontsize=9.4,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.030,
        "Bars: arithmetic case mean  •  Whiskers: interquartile range  •  Area-weighted over fixed India support  •  Positive boxed values favor FuXi.",
        fontsize=8.4,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.010,
        "ACC climatologies: FuXi native 2002–2021 lead/init; ERPAS provider reforecast; IMERG 2001–2022. MAE and RMSE use raw weekly rainfall; RMSE emphasizes large errors.",
        fontsize=8.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.15)
    save_figure(fig, stem, dpi)


def anomaly_scale(dataset: xr.Dataset) -> tuple[np.ndarray, ListedColormap, BoundaryNorm, float, float]:
    mask = dataset.spatial_weight.values > 0
    fields = [
        dataset.observed_weekly_anomaly.sel(reference="IMERG Final V07B").values,
        dataset.forecast_weekly_anomaly.values,
    ]
    absolute = np.concatenate([np.abs(value[..., mask]).ravel() for value in fields])
    q99 = float(np.percentile(absolute[np.isfinite(absolute)], 99))
    # Exact rainfall-anomaly scale used by the supplied FuXi and IMERG
    # reference figures in fuxi_s2s_Hindcast.
    levels = np.asarray([-20, -15, -10, -5, -2, 2, 5, 10, 15, 20], dtype=float)
    limit = 20.0
    colors = [
        "#ff5200",
        "#ff8e1d",
        "#ffca59",
        "#fff4a5",
        "#ffffff",
        "#c8c8e9",
        "#8c8cbf",
        "#6464a3",
        "#3c3c87",
    ]
    cmap = ListedColormap(colors, name="imd_rainfall_anomaly_exact")
    cmap.set_under("#d70e00")
    cmap.set_over("#00001e")
    return levels, cmap, BoundaryNorm(levels, cmap.N), limit, q99


def boundary_segments(geometries: list[object]) -> list[np.ndarray]:
    """Convert polygon boundaries to lightweight line segments for plotting."""
    segments: list[np.ndarray] = []

    def collect(geometry: object) -> None:
        if geometry.geom_type in ("LineString", "LinearRing"):
            segments.append(np.asarray(geometry.coords, dtype=float))
        elif hasattr(geometry, "geoms"):
            for child in geometry.geoms:
                collect(child)

    for geometry in geometries:
        collect(geometry.boundary)
    if not segments:
        raise ValueError("official state geometries produced no boundary segments")
    return segments


def plot_ic_maps(
    dataset: xr.Dataset,
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    dpi: int,
) -> tuple[pd.DataFrame, dict]:
    set_style()
    print("preparing fixed anomaly-map scale and official boundaries", flush=True)
    IC_FIGURES.mkdir(parents=True, exist_ok=True)
    PRESENTATION.mkdir(parents=True, exist_ok=True)
    for stale in list(PRESENTATION.glob("03_*.png")) + list(PRESENTATION.glob("03_*.pdf")):
        stale.unlink()
    _, states, outline = maps.presentation_boundaries()
    state_geometries = list(states.geometries())
    if len(state_geometries) != 40:
        raise ValueError(f"official boundary contains {len(state_geometries)} records, expected 40")
    display_tolerance_degrees = 0.02
    state_geometries = [
        geometry.simplify(display_tolerance_degrees, preserve_topology=True)
        for geometry in state_geometries
    ]
    outline_path = shapely_to_path(outline)
    state_segments = boundary_segments(state_geometries)
    extent = maps.full_india_extent(outline)
    levels, cmap, norm, limit, q99 = anomaly_scale(dataset)
    print(
        f"map scale ready: q99={q99:.3f} mm/day, fixed limit=±{limit:.1f} mm/day",
        flush=True,
    )
    metric_lookup = {
        (row.case_id, row.model, int(row.week)): float(row.acc)
        for row in metrics[metrics.reference == "IMERG Final V07B"].itertuples()
    }
    selected_lookup = {
        row.case_id: row.selection for row in selected.itertuples()
    }
    pdf_path = FIGURES / "04_all_31_imerg_spatial_anomaly_maps.pdf"
    index_rows: list[dict] = []
    with PdfPages(pdf_path) as pdf:
        for case_number, case_id in enumerate(dataset.case.values.astype(str), start=1):
            print(f"drawing anomaly map {case_number}/{dataset.sizes['case']}: {case_id}", flush=True)
            erpas_init = pd.Timestamp(dataset.erpas_initialization.sel(case=case_id).item())
            fuxi_init = pd.Timestamp(dataset.fuxi_initialization.sel(case=case_id).item())
            fuxi_mean = float(np.mean([metric_lookup[(case_id, "FuXi-S2S", week)] for week in range(1, 5)]))
            erpas_mean = float(np.mean([metric_lookup[(case_id, "ERPAS", week)] for week in range(1, 5)]))
            category = selected_lookup.get(case_id, "")
            png_path = IC_FIGURES / f"imerg_anomaly_{case_id}.png"
            index_rows.append(
                {
                    "case_id": case_id,
                    "erpas_initialization": erpas_init.strftime("%Y-%m-%d"),
                    "fuxi_initialization": fuxi_init.strftime("%Y-%m-%d"),
                    "fuxi_four_week_mean_acc": fuxi_mean,
                    "erpas_four_week_mean_acc": erpas_mean,
                    "fuxi_minus_erpas_four_week_mean_acc": fuxi_mean - erpas_mean,
                    "presentation_selection": category,
                    "png": str(png_path),
                }
            )

            fig, axes = plt.subplots(3, 4, figsize=(13.3, 9.8), facecolor="white")
            image = None
            for row_index, source_name in enumerate(("IMERG observed", "FuXi-S2S", "ERPAS")):
                for week_index in range(4):
                    axis = axes[row_index, week_index]
                    if row_index == 0:
                        field = dataset.observed_weekly_anomaly.sel(
                            reference="IMERG Final V07B", case=case_id, week=week_index + 1
                        ).values
                    else:
                        field = dataset.forecast_weekly_anomaly.sel(
                            model=source_name, case=case_id, week=week_index + 1
                        ).values
                    fine_lon, fine_lat, fine_field = maps.smooth_display_field(
                        dataset.longitude.values,
                        dataset.latitude.values,
                        field,
                        extent,
                    )
                    # A 0.30-degree display mesh is already five times finer
                    # than the verified 1.5-degree data and avoids creating
                    # unnecessarily large contour geometries.
                    fine_lon = fine_lon[::2]
                    fine_lat = fine_lat[::2]
                    fine_field = fine_field[::2, ::2]
                    fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)
                    fine_field = np.where(
                        contains_xy(outline, fine_lon_grid, fine_lat_grid),
                        fine_field,
                        np.nan,
                    )
                    image = axis.pcolormesh(
                        fine_lon,
                        fine_lat,
                        fine_field,
                        cmap=cmap,
                        norm=norm,
                        shading="auto",
                        rasterized=True,
                    )
                    axis.add_patch(
                        PathPatch(
                            outline_path,
                            transform=axis.transData,
                            facecolor="none",
                            edgecolor="#202A30",
                            linewidth=1.05,
                            zorder=5,
                        )
                    )
                    axis.add_collection(
                        LineCollection(
                            state_segments,
                            colors="#65777F",
                            linewidths=0.26,
                            zorder=5,
                        )
                    )
                    axis.set_xlim(extent[0], extent[1])
                    axis.set_ylim(extent[2], extent[3])
                    axis.set_aspect(1.08)
                    axis.grid(color=GRID, linewidth=0.3, alpha=0.5, linestyle=":")
                    axis.tick_params(labelsize=6.5, colors=MUTED)
                    if week_index != 0:
                        axis.set_yticklabels([])
                    if row_index != 2:
                        axis.set_xticklabels([])
                    if row_index == 0:
                        start = pd.Timestamp(
                            dataset.week_start.sel(case=case_id, week=week_index + 1).item()
                        )
                        end = pd.Timestamp(
                            dataset.week_end_exclusive.sel(case=case_id, week=week_index + 1).item()
                        ) - pd.Timedelta(days=1)
                        axis.set_title(
                            f"Week {week_index + 1}  |  {start:%d %b}–{end:%d %b}",
                            fontsize=9.2,
                            fontweight="bold",
                            color=INK,
                            pad=6,
                        )
                    if week_index == 0:
                        label = source_name if row_index == 0 else f"{source_name} forecast"
                        color = INK if row_index == 0 else (FUXI if source_name == "FuXi-S2S" else ERPAS)
                        axis.text(
                            -0.20,
                            0.5,
                            label,
                            rotation=90,
                            transform=axis.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9.4,
                            fontweight="bold",
                            color=color,
                        )
                    if row_index > 0:
                        score = metric_lookup[(case_id, source_name, week_index + 1)]
                        axis.text(
                            0.04,
                            0.04,
                            f"ACC {score:.2f}",
                            transform=axis.transAxes,
                            fontsize=7.4,
                            fontweight="bold",
                            color=FUXI if source_name == "FuXi-S2S" else ERPAS,
                            bbox={
                                "boxstyle": "round,pad=0.20",
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.90,
                            },
                            zorder=7,
                        )

            selection_note = f"  •  Representative: {category.replace('_', ' ')}" if category else ""
            fig.suptitle(
                f"IMERG-verified weekly rainfall anomalies — ERPAS IC {erpas_init:%d %B %Y}",
                x=0.075,
                y=0.993,
                ha="left",
                fontsize=16.5,
                fontweight="bold",
                color=INK,
            )
            fig.text(
                0.075,
                0.959,
                f"FuXi IC {fuxi_init:%d %b %Y}  •  identical valid dates  •  four-week mean ACC: FuXi {fuxi_mean:.2f}, ERPAS {erpas_mean:.2f}{selection_note}",
                fontsize=8.8,
                color=MUTED,
            )
            colorbar_axis = fig.add_axes([0.22, 0.059, 0.60, 0.020])
            colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal", extend="both")
            colorbar.set_ticks(levels)
            colorbar.set_label("Weekly rainfall anomaly (mm/day): drier  ←  0  →  wetter", fontsize=8.6)
            colorbar.ax.tick_params(labelsize=7.2)
            fig.text(
                0.075,
                0.017,
                "IMERG anomaly: Final V07B minus fixed 2001–2022 climatology. Model anomalies use system-specific climatologies. Native 1.5° values are used for ACC; interpolation is display-only.",
                fontsize=7.2,
                color=MUTED,
            )
            fig.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.115, hspace=0.10, wspace=0.06)
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            if category:
                selected_stem = PRESENTATION / f"03_{category}_{case_id}"
                fig.savefig(selected_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
                shutil.copy2(png_path, selected_stem.with_suffix(".png"))
            plt.close(fig)
            print(
                f"rendered anomaly map {case_number}/{dataset.sizes['case']}: {case_id}",
                flush=True,
            )

    index = pd.DataFrame(index_rows)
    index.to_csv(HERE / "metrics/spatial_map_index_2023_2024.csv", index=False)
    return index, {
        "global_scale_limit_mm_day": limit,
        "global_absolute_anomaly_q99_mm_day": q99,
        "levels_mm_day": levels.tolist(),
        "official_state_record_count": len(state_geometries),
        "display_boundary_simplification_degrees": display_tolerance_degrees,
        "india_masking": "official outline applied directly to the interpolated display mesh; no effect on native-grid verification",
        "all_maps_pdf": str(pdf_path),
    }


def main() -> int:
    args = parse_args()
    for directory in (FIGURES, IC_FIGURES, PRESENTATION, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(FIELDS) as source:
        dataset = source.load()
    metrics = pd.read_csv(METRICS_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    selected = pd.read_csv(SELECTED_FILE)
    if dataset.sizes.get("case") != 31 or len(metrics) != 496 or len(summary) != 16:
        raise ValueError("validated metric/field sample contract failed before plotting")

    grouped_bars(
        summary,
        metric="acc",
        ylabel="Spatial anomaly correlation (ACC)",
        title="Rainfall anomaly-pattern skill",
        subtitle="FuXi-S2S vs ERPAS  •  31 paired JJAS starts across 2023–2024  •  same valid dates and 1.5° India grid",
        stem=FIGURES / "01_acc_grouped_bars_imd_imerg_2023_2024",
        higher_is_better=True,
        dpi=args.dpi,
    )
    grouped_bars(
        summary,
        metric="mae_mm_day",
        ylabel="Mean absolute error (mm/day)",
        title="Weekly rainfall magnitude error",
        subtitle="FuXi-S2S vs ERPAS  •  raw weekly rainfall verified against IMD and IMERG  •  31 paired JJAS starts",
        stem=FIGURES / "02_mae_grouped_bars_imd_imerg_2023_2024",
        higher_is_better=False,
        dpi=args.dpi,
    )
    imerg_acc_mae_rmse_headline(
        summary,
        stem=FIGURES / "00_imerg_acc_mae_headline_2023_2024",
        dpi=args.dpi,
    )

    if args.skip_maps:
        map_index_path = HERE / "metrics/spatial_map_index_2023_2024.csv"
        map_index = pd.read_csv(map_index_path) if map_index_path.is_file() else pd.DataFrame()
        _, existing_states, _ = maps.presentation_boundaries()
        map_audit = {
            "skipped_rendering": True,
            "existing_outputs_revalidated": True,
            "official_state_record_count": len(list(existing_states.geometries())),
            "all_maps_pdf": str(FIGURES / "04_all_31_imerg_spatial_anomaly_maps.pdf"),
        }
    else:
        map_index, map_audit = plot_ic_maps(dataset, metrics, selected, args.dpi)

    expected_bar_files = [
        FIGURES / f"{stem}.{suffix}"
        for stem in (
            "00_imerg_acc_mae_headline_2023_2024",
            "01_acc_grouped_bars_imd_imerg_2023_2024",
            "02_mae_grouped_bars_imd_imerg_2023_2024",
        )
        for suffix in ("png", "pdf")
    ]
    checks = {
        "six_bar_outputs_exist": all(path.is_file() for path in expected_bar_files),
        "bar_summary_has_16_rows": len(summary) == 2 * 2 * 4,
        "bar_sample_is_31": bool((summary.n_cases == 31).all()),
        "maps_rendered_or_existing_index_has_31": len(map_index) == 31,
        "all_31_ic_pngs_exist": len(map_index) == 31
        and all(Path(path).is_file() for path in map_index.png),
        "three_selected_map_pairs_exist": len(list(PRESENTATION.glob("03_*.png"))) == 3
        and len(list(PRESENTATION.glob("03_*.pdf"))) == 3,
        "official_boundary_has_40_records": map_audit.get("official_state_record_count") == 40,
    }
    image_sizes = {}
    for path in expected_bar_files:
        if path.suffix == ".png":
            with Image.open(path) as image:
                image_sizes[path.name] = list(image.size)
    audit = {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Professor-requested IMERG ACC+MAE+RMSE headline, reference bars, and different-IC anomaly maps",
        "bar_method": "arithmetic mean; IQR whiskers; ACC higher is better; MAE and RMSE lower are better",
        "map_method": map_audit,
        "png_dimensions": image_sizes,
        "checks": checks,
    }
    (LOGS / "figure_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
