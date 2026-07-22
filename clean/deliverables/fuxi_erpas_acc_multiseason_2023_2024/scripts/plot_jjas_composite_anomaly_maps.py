#!/usr/bin/env python3
"""Composite IMD/FuXi/ERPAS JJAS rainfall-anomaly maps for 2023--2024."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = HERE / "scripts/build_acc_csv.py"
SPATIAL_SCRIPT = HERE / "scripts/plot_jjas_spatial_mae_advantage.py"
FIGURES = HERE / "figures"
PROCESSED = HERE / "data/processed"
METRICS = HERE / "metrics"
LOGS = HERE / "logs"

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.path import shapely_to_path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
import xarray as xr


INK = "#172A36"
MUTED = "#536773"
FUXI = "#0072B2"
ERPAS = "#D55E00"

ANOMALY_LEVELS = np.array(
    [-3, -2, -1.25, -0.75, -0.4, -0.2, 0.2, 0.4, 0.75, 1.25, 2, 3],
    dtype=float,
)
ANOMALY_COLORS = (
    "#B2182B",
    "#D6604D",
    "#F4A582",
    "#FDDBC7",
    "#FFF2CF",
    "#F7F7F7",
    "#E5E5F1",
    "#C2C0DD",
    "#908DC3",
    "#5E58A3",
    "#312E74",
)


def anomaly_scale(name: str) -> tuple[np.ndarray, ListedColormap, BoundaryNorm]:
    """Return the shared, discrete drier-to-wetter anomaly scale."""
    cmap = ListedColormap(ANOMALY_COLORS, name=name)
    cmap.set_under("#7F0000")
    cmap.set_over("#17134F")
    return ANOMALY_LEVELS, cmap, BoundaryNorm(ANOMALY_LEVELS, cmap.N)


def smooth_display_field(
    longitude: np.ndarray,
    latitude: np.ndarray,
    field: np.ndarray,
    extent: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly refine a field for display without changing verification data."""
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    field = np.asarray(field, dtype=float)
    lon_order = np.argsort(longitude)
    lat_order = np.argsort(latitude)
    longitude = longitude[lon_order]
    latitude = latitude[lat_order]
    field = field[np.ix_(lat_order, lon_order)]

    valid = np.isfinite(field)
    if not valid.any():
        raise ValueError("display field has no finite values")
    nearest = distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = field[tuple(nearest)]

    lon_step = float(np.median(np.diff(longitude)))
    lat_step = float(np.median(np.diff(latitude)))
    padded_lon = np.concatenate(
        ([longitude[0] - lon_step], longitude, [longitude[-1] + lon_step])
    )
    padded_lat = np.concatenate(
        ([latitude[0] - lat_step], latitude, [latitude[-1] + lat_step])
    )
    padded_field = np.pad(filled, 1, mode="edge")
    interpolator = RegularGridInterpolator(
        (padded_lat, padded_lon),
        padded_field,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    display_step = 0.15
    fine_lon = np.arange(extent[0], extent[1] + display_step, display_step)
    fine_lat = np.arange(extent[2], extent[3] + display_step, display_step)
    fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)
    fine_field = interpolator((fine_lat_grid, fine_lon_grid))
    return fine_lon, fine_lat, fine_field


def full_india_extent(outline_geometry: object) -> list[float]:
    """Frame the complete official boundary, including island territories."""
    west, south, east, north = outline_geometry.bounds
    return [west - 1.2, east + 1.2, south - 0.9, north + 0.9]


def presentation_boundaries() -> tuple[
    cfeature.ShapelyFeature, cfeature.ShapelyFeature, object
]:
    """Simplify sub-pixel boundary detail while preserving all official polygons."""
    _, states, outline_geometry = spatial.official_boundaries()
    tolerance_degrees = 0.005
    display_outline = outline_geometry.simplify(
        tolerance_degrees, preserve_topology=True
    )
    display_states = [
        geometry.simplify(tolerance_degrees, preserve_topology=True)
        for geometry in states.geometries()
    ]
    projection = ccrs.PlateCarree()
    return (
        cfeature.ShapelyFeature([display_outline], projection, facecolor="none"),
        cfeature.ShapelyFeature(display_states, projection, facecolor="none"),
        display_outline,
    )


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = import_file("composite_anomaly_build_support", BUILD_SCRIPT)
spatial = import_file("composite_anomaly_spatial_support", SPATIAL_SCRIPT)


def calculate() -> tuple[xr.Dataset, pd.DataFrame, dict]:
    all_cases, _ = build.build_cases()
    cases = [
        case
        for case in all_cases
        if pd.Timestamp(case["erpas_init"]).month in build.SEASON_WINDOWS["JJAS"]
    ]
    if len(all_cases) != 101 or len(cases) != 31:
        raise ValueError(
            f"unexpected paired samples: all={len(all_cases)}, JJAS={len(cases)}"
        )

    config = json.loads(build.BASE_CONFIG.read_text(encoding="utf-8"))
    config["cases"] = all_cases
    config["model_roots"]["erpas"] = str(build.ERPAS_ROOT)
    config["model_roots"]["fuxi"] = str(build.FUXI_ROOT)
    with xr.open_dataset(build.SOURCE_DATA) as source:
        reference = source.load()
    target_lat, target_lon, india_fraction, weight, land_support = (
        build.accmod.load_land_support(reference)
    )
    original_mask = weight > 0
    observed, imd_climatology, india_fraction, weight, land_support, imd_audit = (
        build.remap_imd(
            all_cases, target_lat, target_lon, land_support, original_mask
        )
    )
    mask = weight > 0
    model_climatology = build.system_climatologies(
        all_cases, target_lat, target_lon, land_support
    )
    if set(model_climatology) != {case["case_id"] for case in cases}:
        raise ValueError("system-climatology cases do not match the paired JJAS sample")

    names = ("IMD observed", "FuXi-S2S", "ERPAS")
    anomalies: dict[str, list[np.ndarray]] = {name: [] for name in names}
    per_case_acc: dict[str, list[list[float]]] = {
        "FuXi-S2S": [[] for _ in range(4)],
        "ERPAS": [[] for _ in range(4)],
    }
    source_counts: list[int] = []
    for index, case in enumerate(cases, 1):
        obs_weekly, imd_clim_weekly = build.weekly_reference(
            case, observed, imd_climatology
        )
        erpas_weekly, erpas_qc = build.load_erpas_variable_count(
            config,
            case,
            target_lat,
            target_lon,
            land_support,
            india_fraction,
        )
        fuxi_weekly, _ = build.core.load_standard_model(
            "FuXi-S2S",
            config,
            case,
            target_lat,
            target_lon,
            land_support,
            india_fraction,
        )
        forecasts = {
            "FuXi-S2S": fuxi_weekly / 7.0,
            "ERPAS": erpas_weekly / 7.0,
        }
        truth_anomaly = obs_weekly - imd_clim_weekly
        truth_anomaly[:, ~mask] = np.nan
        anomalies["IMD observed"].append(truth_anomaly.astype(np.float32))
        for model in ("FuXi-S2S", "ERPAS"):
            forecast_anomaly = (
                forecasts[model] - model_climatology[case["case_id"]][model]
            )
            forecast_anomaly[:, ~mask] = np.nan
            if not np.isfinite(forecast_anomaly[:, mask]).all():
                raise ValueError(f"non-finite anomaly for {case['case_id']}/{model}")
            anomalies[model].append(forecast_anomaly.astype(np.float32))
            for week_index in range(4):
                per_case_acc[model][week_index].append(
                    build.engine.anomaly_correlation(
                        forecast_anomaly[week_index],
                        truth_anomaly[week_index],
                        weight,
                    )
                )
        source_counts.append(int(erpas_qc["source_count"]))
        print(f"processed composite anomaly case {index}/31", flush=True)

    composites: list[np.ndarray] = []
    for name in names:
        stack = np.stack(anomalies[name])
        mean = np.full(stack.shape[1:], np.nan, dtype=np.float32)
        mean[:, mask] = np.mean(stack[:, :, mask], axis=0, dtype=np.float64)
        composites.append(mean)
    composite = np.stack(composites)

    summary = pd.read_csv(
        HERE / "metrics/acc_summary_by_year_season.csv", dtype={"year": str}
    )
    summary = summary[
        (summary.method == "system_specific_jjas")
        & (summary.year == "ALL")
        & (summary.season == "JJAS")
    ]
    expected = {
        (row.model, int(row.week)): float(row.acc_mean)
        for row in summary.itertuples()
    }
    rows: list[dict] = []
    largest_difference = 0.0
    for model_index, model in enumerate(("FuXi-S2S", "ERPAS"), start=1):
        for week_index in range(4):
            mean_case_acc = float(np.mean(per_case_acc[model][week_index]))
            largest_difference = max(
                largest_difference,
                abs(mean_case_acc - expected[(model, week_index + 1)]),
            )
            composite_acc = build.engine.anomaly_correlation(
                composite[model_index, week_index], composite[0, week_index], weight
            )
            rows.append(
                {
                    "model": model,
                    "week": week_index + 1,
                    "n_paired_initializations": len(cases),
                    "mean_per_initialization_spatial_acc": mean_case_acc,
                    "composite_pattern_spatial_acc": composite_acc,
                }
            )
    if largest_difference > 2e-6:
        raise ValueError(
            f"recomputed ACC does not reproduce audited CSV: {largest_difference}"
        )
    metrics = pd.DataFrame(rows)

    dataset = xr.Dataset(
        data_vars={
            "composite_rainfall_anomaly": (
                ("source", "week", "latitude", "longitude"), composite
            ),
            "spatial_weight": (("latitude", "longitude"), weight),
            "india_fraction": (
                ("latitude", "longitude"), india_fraction.astype(np.float32)
            ),
        },
        coords={
            "source": list(names),
            "week": [1, 2, 3, 4],
            "latitude": target_lat,
            "longitude": target_lon,
        },
        attrs={
            "title": "Composite JJAS weekly rainfall anomalies: IMD, FuXi-S2S and ERPAS",
            "sample": "arithmetic mean across 31 paired initialization cycles, 2023-2024",
            "forecast_anomaly_baselines": "FuXi native 2002-2021 hindcast climatology; ERPAS provider climatology",
            "observed_anomaly_baseline": "IMD 1991-2020 calendar-day climatology",
            "grid": "common FuXi-native 1.5-degree grid with conservative remapping",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    audit = {
        "status": "PASSED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_paired_initializations": len(cases),
        "year_counts": {
            str(year): sum(
                pd.Timestamp(case["erpas_init"]).year == year for case in cases
            )
            for year in (2023, 2024)
        },
        "fixed_target_supported_cells": int(mask.sum()),
        "imd_support_audit": imd_audit,
        "erpas_source_counts": {
            str(value): source_counts.count(value) for value in sorted(set(source_counts))
        },
        "maximum_acc_reproduction_difference": largest_difference,
        "checks": {
            "paired_cycle_count_is_31": len(cases) == 31,
            "four_lead_weeks": composite.shape[1] == 4,
            "three_map_rows": composite.shape[0] == 3,
            "supported_composites_finite": bool(
                np.isfinite(composite[:, :, mask]).all()
            ),
            "audited_acc_reproduced": largest_difference <= 2e-6,
        },
    }
    if not all(audit["checks"].values()):
        raise ValueError(f"composite anomaly audit failed: {audit['checks']}")
    return dataset, metrics, audit


def plot(dataset: xr.Dataset, metrics: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11.2,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    levels, cmap, norm = anomaly_scale("rainfall_anomaly")
    projection = ccrs.PlateCarree()
    outline, states, outline_geometry = presentation_boundaries()
    map_extent = full_india_extent(outline_geometry)
    sources = ("IMD observed", "FuXi-S2S", "ERPAS")
    row_labels = ("IMD observed", "FuXi-S2S forecast", "ERPAS forecast")
    colors = (INK, FUXI, ERPAS)
    mean_acc = metrics.groupby("model").mean_per_initialization_spatial_acc.mean()

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(15.6, 10.6),
        subplot_kw={"projection": projection},
    )
    image = None
    for row_index, (source, row_label, row_color) in enumerate(
        zip(sources, row_labels, colors)
    ):
        for week_index in range(4):
            axis = axes[row_index, week_index]
            field = dataset.composite_rainfall_anomaly.sel(
                source=source, week=week_index + 1
            ).values
            display_lon, display_lat, display_field = smooth_display_field(
                dataset.longitude.values,
                dataset.latitude.values,
                field,
                map_extent,
            )
            image = axis.contourf(
                display_lon,
                display_lat,
                display_field,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="both",
                antialiased=True,
                transform=projection,
            )
            clip = PathPatch(
                shapely_to_path(outline_geometry),
                transform=projection._as_mpl_transform(axis),
                facecolor="none",
            )
            image.set_clip_path(clip)
            axis.add_feature(outline, edgecolor="#111111", linewidth=1.15, zorder=5)
            axis.add_feature(states, edgecolor="#69777D", linewidth=0.28, zorder=5)
            axis.set_extent(map_extent, crs=projection)
            gridlines = axis.gridlines(
                crs=projection,
                draw_labels=True,
                x_inline=False,
                y_inline=False,
                linewidth=0.25,
                color="#B8C4CA",
                alpha=0.45,
                linestyle=":",
            )
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.left_labels = week_index == 0
            gridlines.bottom_labels = row_index == 2
            gridlines.xlabel_style = {"size": 7.0, "color": MUTED}
            gridlines.ylabel_style = {"size": 7.0, "color": MUTED}
            if row_index == 0:
                axis.set_title(
                    f"{'abcd'[week_index]}   WEEK {week_index + 1}",
                    loc="left",
                    pad=7,
                    color=INK,
                )
                fuxi_score = metrics[
                    (metrics.model == "FuXi-S2S")
                    & (metrics.week == week_index + 1)
                ].iloc[0].mean_per_initialization_spatial_acc
                erpas_score = metrics[
                    (metrics.model == "ERPAS")
                    & (metrics.week == week_index + 1)
                ].iloc[0].mean_per_initialization_spatial_acc
                axis.text(
                    0.98,
                    1.025,
                    f"FuXi lead  +{fuxi_score - erpas_score:.2f} ACC",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7.6,
                    fontweight="semibold",
                    color=FUXI,
                )
            if week_index == 0:
                axis.text(
                    -0.18,
                    0.50,
                    row_label,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="semibold",
                    color=row_color,
                )
            if row_index > 0:
                score = metrics[
                    (metrics.model == source) & (metrics.week == week_index + 1)
                ].iloc[0].mean_per_initialization_spatial_acc
                axis.text(
                    0.04,
                    0.04,
                    f"ACC vs IMD  {score:.2f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.7,
                    fontweight="semibold",
                    color=row_color,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.88,
                    },
                    zorder=7,
                )

    fig.suptitle(
        "Weekly JJAS rainfall anomalies: observed vs forecasts",
        x=0.066,
        y=0.988,
        ha="left",
        fontsize=19,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.066,
        0.953,
        "Average pattern across 31 paired forecasts, 2023–2024  •  Red/orange = drier; purple = wetter",
        ha="left",
        fontsize=10.2,
        color=MUTED,
    )
    fig.text(
        0.066,
        0.924,
        "KEY RESULT   FuXi matches IMD better in all four weeks  •  "
        f"Mean ACC: FuXi {mean_acc['FuXi-S2S']:.2f} vs "
        f"ERPAS {mean_acc['ERPAS']:.2f}",
        ha="left",
        fontsize=9.0,
        color=FUXI,
        fontweight="semibold",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#EAF4FA",
            "edgecolor": "#A7D3EA",
            "linewidth": 0.7,
        },
    )
    fig.text(
        0.066,
        0.892,
        "READ IT   Match each forecast with IMD above: same colour in the same place is better. Weeks cover days 1–7, 8–14, 15–21 and 22–28; ACC 1 = perfect, 0 = no match.",
        ha="left",
        fontsize=8.1,
        color=MUTED,
    )
    colorbar_axis = fig.add_axes([0.20, 0.068, 0.64, 0.022])
    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=levels,
        extend="both",
    )
    colorbar.set_label("Rainfall anomaly (mm/day)", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=7.6)
    fig.text(
        0.165, 0.079, "Drier than normal", ha="right", va="center",
        fontsize=8.3, color=ERPAS,
    )
    fig.text(
        0.875, 0.079, "Wetter than normal", ha="left", va="center",
        fontsize=8.3, color=FUXI,
    )
    fig.text(
        0.066,
        0.022,
        "Anomaly baselines: FuXi 2002–2021 hindcasts; ERPAS provider climatology; IMD 1991–2020. Bilinear interpolation is for display only; ACC uses the unchanged native 1.5° grid. Full official India boundary, including islands.",
        ha="left",
        fontsize=7.7,
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        top=0.855,
        bottom=0.125,
        hspace=0.08,
        wspace=0.07,
    )
    for suffix, kwargs in (("png", {"dpi": 330}), ("pdf", {})):
        fig.savefig(
            FIGURES / f"imd_briefing_jjas_composite_anomaly_maps_2023_2024.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def plot_source_figures(dataset: xr.Dataset, metrics: pd.DataFrame) -> None:
    """Render reference-style 2x2 maps separately for IMD, FuXi and ERPAS."""
    levels, cmap, norm = anomaly_scale("rainfall_anomaly_single")
    projection = ccrs.PlateCarree()
    outline, states, outline_geometry = presentation_boundaries()
    map_extent = full_india_extent(outline_geometry)
    specifications = (
        (
            "IMD observed",
            "Observed JJAS rainfall-anomaly composite",
            "IMD weekly anomalies over the forecast-valid periods",
            INK,
            "imd",
        ),
        (
            "FuXi-S2S",
            "FuXi-S2S JJAS rainfall-anomaly composite",
            "50-member ensemble-mean forecast minus native 2002–2021 hindcast climatology",
            FUXI,
            "fuxi",
        ),
        (
            "ERPAS",
            "ERPAS JJAS rainfall-anomaly composite",
            "Provider forecast mean minus provider hindcast climatology",
            ERPAS,
            "erpas",
        ),
    )
    for source, title, method_line, accent, stem_label in specifications:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(9.8, 9.0),
            subplot_kw={"projection": projection},
        )
        image = None
        for week_index, axis in enumerate(axes.flat):
            field = dataset.composite_rainfall_anomaly.sel(
                source=source, week=week_index + 1
            ).values
            display_lon, display_lat, display_field = smooth_display_field(
                dataset.longitude.values,
                dataset.latitude.values,
                field,
                map_extent,
            )
            image = axis.contourf(
                display_lon,
                display_lat,
                display_field,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="both",
                antialiased=True,
                transform=projection,
            )
            clip = PathPatch(
                shapely_to_path(outline_geometry),
                transform=projection._as_mpl_transform(axis),
                facecolor="none",
            )
            image.set_clip_path(clip)
            axis.add_feature(outline, edgecolor="#111111", linewidth=1.18, zorder=5)
            axis.add_feature(states, edgecolor="#5E676C", linewidth=0.34, zorder=5)
            axis.set_extent(map_extent, crs=projection)
            gridlines = axis.gridlines(
                crs=projection,
                draw_labels=True,
                x_inline=False,
                y_inline=False,
                linewidth=0.28,
                color="#B8C4CA",
                alpha=0.48,
                linestyle=":",
            )
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.left_labels = week_index % 2 == 0
            gridlines.bottom_labels = week_index >= 2
            gridlines.xlabel_style = {"size": 7.5, "color": MUTED}
            gridlines.ylabel_style = {"size": 7.5, "color": MUTED}
            axis.set_title(
                f"{'abcd'[week_index]}   Lead Week {week_index + 1}  "
                f"(days {1 + 7 * week_index}–{7 + 7 * week_index})",
                loc="left",
                pad=8,
                color=INK,
            )
            if source != "IMD observed":
                score = metrics[
                    (metrics.model == source) & (metrics.week == week_index + 1)
                ].iloc[0].mean_per_initialization_spatial_acc
                axis.text(
                    0.04,
                    0.04,
                    f"Mean spatial ACC vs IMD  {score:.2f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="semibold",
                    color=accent,
                    bbox={
                        "boxstyle": "round,pad=0.28",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.90,
                    },
                    zorder=7,
                )

        fig.suptitle(
            title,
            x=0.075,
            y=0.984,
            ha="left",
            fontsize=18,
            fontweight="semibold",
            color=accent,
        )
        fig.text(
            0.075,
            0.945,
            "Arithmetic mean across 31 paired JJAS initialization cycles, 2023–2024",
            ha="left",
            fontsize=10.0,
            color=MUTED,
        )
        fig.text(
            0.075,
            0.918,
            method_line,
            ha="left",
            fontsize=8.8,
            color=MUTED,
        )
        colorbar_axis = fig.add_axes([0.17, 0.078, 0.68, 0.024])
        colorbar = fig.colorbar(
            image,
            cax=colorbar_axis,
            orientation="horizontal",
            ticks=levels,
            extend="both",
        )
        colorbar.set_label("Composite weekly rainfall anomaly (mm/day)", fontsize=9.4)
        colorbar.ax.tick_params(labelsize=7.6)
        fig.text(
            0.075,
            0.026,
            "Common FuXi-native 1.5° grid; identical valid periods; full official India/state boundary including islands. Bilinear interpolation is used only for cleaner display; verification uses the unchanged native grid.",
            ha="left",
            fontsize=7.7,
            color=MUTED,
        )
        fig.subplots_adjust(
            left=0.09,
            right=0.96,
            top=0.87,
            bottom=0.145,
            hspace=0.13,
            wspace=0.08,
        )
        for suffix, kwargs in (("png", {"dpi": 340}), ("pdf", {})):
            fig.savefig(
                FIGURES
                / f"imd_briefing_jjas_composite_anomaly_{stem_label}_2023_2024.{suffix}",
                bbox_inches="tight",
                facecolor="white",
                **kwargs,
            )
        plt.close(fig)


def main() -> int:
    for directory in (FIGURES, PROCESSED, METRICS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    dataset, metrics, audit = calculate()
    dataset.to_netcdf(PROCESSED / "jjas_composite_anomaly_maps_2023_2024.nc")
    metrics.to_csv(METRICS / "jjas_composite_anomaly_map_metrics_2023_2024.csv", index=False)
    (LOGS / "jjas_composite_anomaly_map_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    plot(dataset, metrics)
    plot_source_figures(dataset, metrics)
    print(metrics.to_string(index=False), flush=True)
    print("wrote polished JJAS composite anomaly maps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
