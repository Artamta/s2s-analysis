#!/usr/bin/env python3
"""Map where FuXi-S2S reduces JJAS weekly-rainfall MAE versus ERPAS.

The calculation reuses the audited paired-cycle loaders and frozen All-India
support from ``build_acc_csv.py``.  Local MAE is averaged over the same 31
paired 2023--2024 JJAS initializations used by the system-climatology ACC
headline.  The mapped difference is ERPAS MAE minus FuXi-S2S MAE, so positive
(blue) values consistently mean lower FuXi error.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = HERE / "scripts/build_acc_csv.py"
FIGURES = HERE / "figures"
PROCESSED = HERE / "data/processed"
METRICS = HERE / "metrics"
LOGS = HERE / "logs"
OFFICIAL_STATE_SHAPEFILE = Path(
    "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"
)

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.path import shapely_to_path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from shapely.ops import transform as transform_geometry, unary_union
import xarray as xr


FUXI = "#0072B2"
ERPAS = "#D55E00"
INK = "#172A36"
MUTED = "#536773"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = import_file("spatial_mae_build_support", BUILD_SCRIPT)


def official_boundaries() -> tuple[
    cfeature.ShapelyFeature, cfeature.ShapelyFeature, object
]:
    """Load and transform the project official 40-record state boundary."""
    if not OFFICIAL_STATE_SHAPEFILE.is_file():
        raise FileNotFoundError(OFFICIAL_STATE_SHAPEFILE)
    source_geometries = list(
        shpreader.Reader(str(OFFICIAL_STATE_SHAPEFILE)).geometries()
    )
    if len(source_geometries) != 40:
        raise ValueError(
            f"official state shapefile has {len(source_geometries)} records; expected 40"
        )
    source_crs = ccrs.LambertConformal(
        central_longitude=80.0,
        central_latitude=24.0,
        standard_parallels=(12.472944, 35.172806),
        false_easting=4_000_000.0,
        false_northing=4_000_000.0,
    )
    target_crs = ccrs.PlateCarree()

    def to_wgs84(x, y, z=None):
        points = target_crs.transform_points(source_crs, np.asarray(x), np.asarray(y))
        return points[:, 0], points[:, 1]

    states = [transform_geometry(to_wgs84, geometry) for geometry in source_geometries]
    outline = unary_union(states)
    if not outline.is_valid:
        raise ValueError("official India boundary union invalid after WGS84 transform")
    return (
        cfeature.ShapelyFeature([outline], target_crs, facecolor="none"),
        cfeature.ShapelyFeature(states, target_crs, facecolor="none"),
        outline,
    )


def weighted_mean(field: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(field) & np.isfinite(weight) & (weight > 0)
    return float(np.sum(field[valid] * weight[valid]) / np.sum(weight[valid]))


def calculate() -> tuple[xr.Dataset, pd.DataFrame, dict]:
    all_cases, excluded = build.build_cases()
    if len(all_cases) != 101:
        raise ValueError(f"expected 101 paired cycles, got {len(all_cases)}")
    cases = [
        case
        for case in all_cases
        if pd.Timestamp(case["erpas_init"]).month in build.SEASON_WINDOWS["JJAS"]
    ]
    if len(cases) != 31:
        raise ValueError(f"expected 31 paired JJAS cycles, got {len(cases)}")

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
    if int(original_mask.sum()) != 171:
        raise ValueError("initial frozen India support is not 171 cells")

    # Use all 101 cycles here so the missing-data support is exactly the same
    # fixed all-season support used by the audited ACC/MAE CSV calculation.
    observed, imd_climatology, india_fraction, weight, land_support, imd_audit = build.remap_imd(
        all_cases, target_lat, target_lon, land_support, original_mask
    )
    mask = weight > 0
    if int(mask.sum()) != int(imd_audit["fixed_all_season_target_supported_cells"]):
        raise ValueError("fixed target support count mismatch")

    errors: dict[str, list[np.ndarray]] = {"FuXi-S2S": [], "ERPAS": []}
    source_counts: list[int] = []
    for index, case in enumerate(cases, 1):
        obs_weekly, _ = build.weekly_reference(case, observed, imd_climatology)
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
            "ERPAS": erpas_weekly / 7.0,
            "FuXi-S2S": fuxi_weekly / 7.0,
        }
        for model, forecast in forecasts.items():
            local_error = np.abs(forecast - obs_weekly)
            local_error[:, ~mask] = np.nan
            if not np.isfinite(local_error[:, mask]).all():
                raise ValueError(f"non-finite supported errors for {case['case_id']}/{model}")
            errors[model].append(local_error.astype(np.float32))
        source_counts.append(int(erpas_qc["source_count"]))
        print(f"processed JJAS spatial case {index}/31", flush=True)

    local_mae = np.stack(
        [
            np.mean(np.stack(errors[model]), axis=0, dtype=np.float64)
            for model in ("FuXi-S2S", "ERPAS")
        ]
    ).astype(np.float32)
    advantage = local_mae[1] - local_mae[0]
    advantage[:, ~mask] = np.nan

    rows: list[dict] = []
    for week_index in range(4):
        fuxi_mean = weighted_mean(local_mae[0, week_index], weight)
        erpas_mean = weighted_mean(local_mae[1, week_index], weight)
        delta_mean = weighted_mean(advantage[week_index], weight)
        fuxi_better_area = float(
            100.0
            * np.sum(weight[(advantage[week_index] > 0) & mask])
            / np.sum(weight[mask])
        )
        rows.append(
            {
                "week": week_index + 1,
                "n_paired_initializations": len(cases),
                "fuxi_mae_mm_day": fuxi_mean,
                "erpas_mae_mm_day": erpas_mean,
                "erpas_minus_fuxi_mae_mm_day": delta_mean,
                "india_area_percent_fuxi_lower_mae": fuxi_better_area,
            }
        )
    metrics = pd.DataFrame(rows)

    # Independent numerical agreement check against the already audited CSV.
    summary = pd.read_csv(HERE / "metrics/acc_summary_by_year_season.csv", dtype={"year": str})
    check = summary[
        (summary.method == "system_specific_jjas")
        & (summary.year == "ALL")
        & (summary.season == "JJAS")
    ]
    expected = {
        (row.model, int(row.week)): float(row.mae_mean_mm_day)
        for row in check.itertuples()
    }
    observed_values = {
        ("FuXi-S2S", int(row.week)): float(row.fuxi_mae_mm_day)
        for row in metrics.itertuples()
    } | {
        ("ERPAS", int(row.week)): float(row.erpas_mae_mm_day)
        for row in metrics.itertuples()
    }
    largest_difference = max(
        abs(observed_values[key] - expected[key]) for key in expected
    )
    if largest_difference > 2e-5:
        raise ValueError(
            f"spatial MAE does not reproduce audited CSV; max difference={largest_difference}"
        )

    dataset = xr.Dataset(
        data_vars={
            "local_mae": (
                ("model", "week", "latitude", "longitude"), local_mae
            ),
            "erpas_minus_fuxi_mae": (
                ("week", "latitude", "longitude"), advantage.astype(np.float32)
            ),
            "spatial_weight": (("latitude", "longitude"), weight),
            "india_fraction": (
                ("latitude", "longitude"), india_fraction.astype(np.float32)
            ),
        },
        coords={
            "model": ["FuXi-S2S", "ERPAS"],
            "week": [1, 2, 3, 4],
            "latitude": target_lat,
            "longitude": target_lon,
        },
        attrs={
            "title": "JJAS local weekly-rainfall MAE: FuXi-S2S versus ERPAS",
            "sample": "31 paired initialization cycles, 2023-2024",
            "difference_sign": "ERPAS MAE minus FuXi-S2S MAE; positive means FuXi-S2S has lower error",
            "verification": "IMD daily gridded rainfall conservatively remapped to the FuXi-native 1.5-degree grid",
            "valid_period_alignment": "identical four consecutive seven-day windows for each paired cycle",
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
        "available_fraction": "31 of 34 possible weekly JJAS cycles",
        "excluded_total_all_seasons": len(excluded),
        "fixed_target_supported_cells": int(mask.sum()),
        "official_boundary": str(OFFICIAL_STATE_SHAPEFILE),
        "erpas_source_counts": {
            str(value): int(source_counts.count(value)) for value in sorted(set(source_counts))
        },
        "maximum_mae_reproduction_difference_mm_day": largest_difference,
        "checks": {
            "paired_cycle_count_is_31": len(cases) == 31,
            "four_lead_weeks": local_mae.shape[1] == 4,
            "all_supported_errors_finite": bool(np.isfinite(local_mae[:, :, mask]).all()),
            "audited_csv_mae_reproduced": largest_difference <= 2e-5,
            "official_boundary_has_40_records": len(
                list(shpreader.Reader(str(OFFICIAL_STATE_SHAPEFILE)).geometries())
            ) == 40,
        },
    }
    if not all(audit["checks"].values()):
        raise ValueError(f"spatial audit failed: {audit['checks']}")
    return dataset, metrics, audit


def plot(dataset: xr.Dataset, metrics: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12.2,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    levels = np.array([-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3], dtype=float)
    cmap = ListedColormap(
        [
            "#7F0000", "#B30000", "#D7301F", "#EF6548", "#FC8D59", "#FEC49A",
            "#D7E9F3", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594",
        ]
    )
    norm = BoundaryNorm(levels, cmap.N, extend="both")
    projection = ccrs.PlateCarree()
    outline, states, outline_geometry = official_boundaries()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 9.7),
        subplot_kw={"projection": projection},
    )
    image = None
    for index, axis in enumerate(axes.flat):
        field = dataset.erpas_minus_fuxi_mae.isel(week=index).values
        image = axis.pcolormesh(
            dataset.longitude,
            dataset.latitude,
            field,
            cmap=cmap,
            norm=norm,
            shading="nearest",
            transform=projection,
            rasterized=True,
        )
        clip = PathPatch(
            shapely_to_path(outline_geometry),
            transform=projection._as_mpl_transform(axis),
            facecolor="none",
        )
        image.set_clip_path(clip)
        axis.add_feature(outline, edgecolor="#111111", linewidth=1.35, zorder=5)
        axis.add_feature(states, edgecolor="#5E676C", linewidth=0.38, zorder=5)
        axis.set_extent([67.0, 98.5, 6.0, 38.5], crs=projection)
        gridlines = axis.gridlines(
            crs=projection,
            draw_labels=True,
            x_inline=False,
            y_inline=False,
            linewidth=0.35,
            color="#B8C4CA",
            alpha=0.55,
            linestyle=":",
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.left_labels = index % 2 == 0
        gridlines.bottom_labels = index >= 2
        gridlines.xlabel_style = {"size": 8, "color": MUTED}
        gridlines.ylabel_style = {"size": 8, "color": MUTED}
        row = metrics.iloc[index]
        axis.set_title(
            f"{'abcd'[index]}   Week {index + 1}", loc="left", pad=8, color=INK
        )
        axis.text(
            0.03,
            0.04,
            f"India mean: +{row.erpas_minus_fuxi_mae_mm_day:.2f} mm/day\n"
            f"FuXi lower over {row.india_area_percent_fuxi_lower_mae:.0f}% of area",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.3,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            zorder=7,
        )

    fig.suptitle(
        "FuXi-S2S reduces JJAS rainfall error across most of India from Week 2",
        x=0.08,
        y=0.982,
        ha="left",
        fontsize=19,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.08,
        0.943,
        "Local MAE difference: ERPAS minus FuXi-S2S  |  blue = lower FuXi error  |  IMD verification",
        ha="left",
        fontsize=10.2,
        color=MUTED,
    )
    cbar_axis = fig.add_axes([0.17, 0.083, 0.66, 0.025])
    colorbar = fig.colorbar(
        image,
        cax=cbar_axis,
        orientation="horizontal",
        ticks=levels,
        extend="both",
    )
    colorbar.set_label(
        "ERPAS MAE − FuXi-S2S MAE (mm/day)", fontsize=9.5, color=INK, labelpad=5
    )
    colorbar.ax.tick_params(labelsize=8)
    fig.text(
        0.08,
        0.025,
        "Mean over 31 paired initialization cycles (2023–2024); identical valid weeks; common FuXi-native 1.5° grid; official India state boundary. Descriptive, not a significance map.",
        ha="left",
        fontsize=8.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.95, top=0.90, bottom=0.14, hspace=0.16, wspace=0.08)
    png_path = FIGURES / "imd_briefing_jjas_spatial_mae_advantage_2023_2024.png"
    pdf_path = FIGURES / "imd_briefing_jjas_spatial_mae_advantage_2023_2024.pdf"
    fig.savefig(png_path, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # A fully vector PDF repeats the detailed 40-part official boundary for
    # every clipped map and grows to tens of megabytes.  Embed the audited
    # 360-dpi render instead; this remains presentation-grade and Git-friendly.
    rendered = plt.imread(png_path)
    raster_fig = plt.figure(
        figsize=(rendered.shape[1] / 240, rendered.shape[0] / 240)
    )
    raster_axis = raster_fig.add_axes([0, 0, 1, 1])
    raster_axis.imshow(rendered)
    raster_axis.axis("off")
    raster_fig.savefig(pdf_path, dpi=240)
    plt.close(raster_fig)


def main() -> int:
    for directory in (FIGURES, PROCESSED, METRICS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    dataset, metrics, audit = calculate()
    dataset.to_netcdf(PROCESSED / "jjas_spatial_mae_advantage_2023_2024.nc")
    metrics.to_csv(METRICS / "jjas_spatial_mae_advantage_2023_2024.csv", index=False)
    (LOGS / "jjas_spatial_mae_advantage_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    plot(dataset, metrics)
    print(metrics.to_string(index=False), flush=True)
    print("wrote polished JJAS spatial MAE advantage figure", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
