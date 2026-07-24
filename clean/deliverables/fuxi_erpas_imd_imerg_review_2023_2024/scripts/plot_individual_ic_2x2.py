#!/usr/bin/env python3
"""Render separate 2x2 Week-1--4 anomaly figures for individual paired cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.path import shapely_to_path
import numpy as np
import pandas as pd
from shapely import contains_xy
import xarray as xr


SUPPORT_SCRIPT = HERE / "scripts/plot_review_figures.py"
NATIVE_SUPPORT_SCRIPT = HERE / "scripts/plot_native_imerg_erpas_maps.py"
FIELDS = HERE / "data/processed/review_fields_2023_2024.nc"
METRICS = HERE / "metrics/per_case_metrics_2023_2024.csv"
FUXI_FORECAST_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/forecasts"
)
FUXI_CLIMATOLOGY = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_jjas_2002_2021/"
    "fuxi_s2s_jjas_model_climatology_2002_2021_loyo.nc"
)
OUTPUT = HERE / "figures/individual_ic_2x2_unclipped_native"
AUDIT_PATH = HERE / "logs/individual_ic_2x2_unclipped_native_audit.json"

LEVELS = np.asarray([-20, -15, -10, -5, -2, 2, 5, 10, 15, 20], dtype=float)
COLORS = [
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
INK = "#172A36"
MUTED = "#536773"
GRID = "#D8E1E6"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


support = import_file("individual_ic_2x2_support", SUPPORT_SCRIPT)
native_support = import_file(
    "individual_ic_2x2_native_support", NATIVE_SUPPORT_SCRIPT
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help=(
            "Paired case to render, for example paired_20240724. Repeat for "
            "multiple cases; the default preview is paired_20240724."
        ),
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Render all cases in the audited 2023--2024 dataset.",
    )
    parser.add_argument(
        "--south-latitude",
        type=float,
        default=5.0,
        help="Southern map boundary in degrees north (default: 5.0).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Validate inputs but do not redraw complete PNG/PDF pairs.",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#63727A",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_date_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start.year == end.year and start.month == end.month:
        return f"{start:%d}–{end:%d %b %Y}"
    if start.year == end.year:
        return f"{start:%d %b}–{end:%d %b %Y}"
    return f"{start:%d %b %Y}–{end:%d %b %Y}"


def product_specifications(
    dataset: xr.Dataset,
    case_id: str,
    metrics: pd.DataFrame,
    imerg_native_anomaly: xr.DataArray,
    fuxi_native_anomaly: xr.DataArray,
    erpas_native_anomaly: xr.DataArray,
) -> list[dict]:
    erpas_init = pd.Timestamp(dataset.erpas_initialization.sel(case=case_id).item())
    fuxi_init = pd.Timestamp(dataset.fuxi_initialization.sel(case=case_id).item())
    case_metrics = metrics[
        (metrics.reference == "IMERG Final V07B")
        & (metrics.case_id == case_id)
    ]
    score_lookup = {
        (row.model, int(row.week)): float(row.acc)
        for row in case_metrics.itertuples()
    }
    expected_scores = {
        (model, week)
        for model in ("FuXi-S2S", "ERPAS")
        for week in range(1, 5)
    }
    if set(score_lookup) != expected_scores:
        raise ValueError(f"{case_id}: incomplete IMERG ACC records")

    return [
        {
            "key": "imerg",
            "label": "IMERG Final V07B",
            "title": "Observed weekly rainfall anomaly — IMERG",
            "subtitle": (
                f"IMERG Final V07B  •  paired verification case  "
                f"•  ERPAS IC {erpas_init:%d %b %Y}  "
                f"•  FuXi IC {fuxi_init:%d %b %Y}"
            ),
            "note": (
                "IMERG anomaly = observed weekly rainfall minus fixed "
                "2001–2022 IMERG Final V07B calendar-day climatology."
            ),
            "accent": "#D83F3F",
            "field": imerg_native_anomaly,
            "display_mode": "native",
            "render_mode": "native_cells",
            "grid_note": (
                "IMERG is shown on its native 0.1° grid over the full map "
                "domain; no display interpolation is applied."
            ),
            "scores": None,
            "filename": f"01_imerg_observed_anomaly_{case_id}",
        },
        {
            "key": "fuxi",
            "label": "FuXi-S2S",
            "title": "FuXi-S2S weekly rainfall anomaly forecast",
            "subtitle": (
                f"IC {fuxi_init:%d %b %Y}  •  50-member ensemble mean  "
                "•  identical Thursday–Wednesday valid weeks"
            ),
            "note": (
                "FuXi anomaly = forecast weekly mean minus native 2002–2021 "
                "lead/init-aware FuXi-S2S model climatology."
            ),
            "accent": "#1254C4",
            "field": fuxi_native_anomaly,
            "display_mode": "native",
            "render_mode": "native_contours",
            "grid_note": (
                "FuXi-S2S uses its native 1.5° nodes over the full map domain; "
                "filled contours do not resample or modify anomaly values."
            ),
            "scores": {
                week: score_lookup[("FuXi-S2S", week)]
                for week in range(1, 5)
            },
            "filename": f"02_fuxi_s2s_anomaly_{case_id}",
        },
        {
            "key": "erpas",
            "label": "ERPAS",
            "title": "ERPAS weekly rainfall anomaly forecast",
            "subtitle": (
                f"IC {erpas_init:%d %b %Y}  •  provider-precomputed ensemble mean  "
                "•  identical Thursday–Wednesday valid weeks"
            ),
            "note": (
                "ERPAS anomaly = provider forecast mean minus provider "
                "reforecast climatology."
            ),
            "accent": "#7A173A",
            "field": erpas_native_anomaly,
            "display_mode": "native",
            "render_mode": "native_contours",
            "grid_note": (
                "ERPAS uses its native 1.0° nodes over the full map domain; "
                "filled contours do not resample or modify anomaly values."
            ),
            "scores": {
                week: score_lookup[("ERPAS", week)]
                for week in range(1, 5)
            },
            "filename": f"03_erpas_anomaly_{case_id}",
        },
    ]


def load_fuxi_climatology() -> tuple[xr.DataArray, dict]:
    with xr.open_dataset(FUXI_CLIMATOLOGY) as source:
        if (
            source.attrs.get("loyo_definition") is None
            or source.tp_model_climatology_mean.attrs.get("units") != "mm day-1"
            or source.sizes.get("hindcast_year") != 20
        ):
            raise ValueError("FuXi native climatology contract failed")
        daily = source.tp_model_climatology_mean.sel(
            lead_day=slice(4, 31)
        ).load()
        attrs = dict(source.attrs)
    if daily.shape != (35, 28, 27, 27):
        raise ValueError(f"unexpected FuXi climatology shape {daily.shape}")
    if not np.isfinite(daily.values).all():
        raise ValueError("FuXi climatology contains non-finite values")
    return daily, attrs


def load_fuxi_native_anomaly(
    fuxi_init: pd.Timestamp,
    erpas_init: pd.Timestamp,
    fuxi_climatology: xr.DataArray,
    climatology_attrs: dict,
) -> tuple[xr.DataArray, dict]:
    forecast_path = (
        FUXI_FORECAST_ROOT
        / f"annual{fuxi_init.year}"
        / f"{fuxi_init:%Y%m%d}.nc"
    )
    with xr.open_dataset(forecast_path) as source:
        if (
            source.attrs.get("model") != "FuXi-S2S"
            or source.attrs.get("strict_operational") != "true"
            or source.sizes.get("member") != 50
            or source.sizes.get("lead_day") != 42
            or source.tp.attrs.get("units") != "mm h-1"
            or source.attrs.get("valid_time_role") != "period_end"
        ):
            raise ValueError(f"{forecast_path}: FuXi forecast contract failed")
        daily_forecast = (
            source.tp.sel(lead_day=slice(4, 31))
            .mean("member")
            .load()
            .astype(np.float64)
            * 24.0
        )
        forecast_starts = pd.DatetimeIndex(
            source.forecast_period_start.sel(lead_day=slice(4, 31)).values
        )
        forecast_ends = pd.DatetimeIndex(
            source.forecast_period_end.sel(lead_day=slice(4, 31)).values
        )
        forecast_attrs = dict(source.attrs)
    expected_starts = pd.date_range(
        erpas_init + pd.Timedelta(days=1),
        periods=28,
        freq="D",
    )
    expected_ends = expected_starts + pd.Timedelta(days=1)
    if not forecast_starts.equals(expected_starts) or not forecast_ends.equals(
        expected_ends
    ):
        raise ValueError("FuXi daily periods do not match the paired valid weeks")

    available_slots = list(fuxi_climatology.init_slot.values.astype(str))
    left, right, alpha = native_support.interpolation_bracket(
        fuxi_init.strftime("%m%d"),
        available_slots,
    )
    left_field = fuxi_climatology.sel(init_slot=left)
    right_field = (
        left_field
        if left == right
        else fuxi_climatology.sel(init_slot=right)
    )
    if not (
        np.array_equal(left_field.latitude.values, daily_forecast.latitude.values)
        and np.array_equal(
            left_field.longitude.values,
            daily_forecast.longitude.values,
        )
    ):
        raise ValueError("FuXi forecast and native climatology grids differ")
    daily_climatology = (1.0 - alpha) * left_field + alpha * right_field
    if not (
        np.isfinite(daily_forecast.values).all()
        and np.isfinite(daily_climatology.values).all()
    ):
        raise ValueError("FuXi forecast or climatology contains non-finite values")

    daily_anomaly = daily_forecast.values - daily_climatology.values
    weekly_anomaly = daily_anomaly.reshape(
        4,
        7,
        daily_forecast.sizes["latitude"],
        daily_forecast.sizes["longitude"],
    ).mean(axis=1)
    anomaly = xr.DataArray(
        weekly_anomaly.astype(np.float32),
        dims=("week", "latitude", "longitude"),
        coords={
            "week": np.arange(1, 5),
            "latitude": daily_forecast.latitude.values,
            "longitude": daily_forecast.longitude.values,
        },
        attrs={"units": "mm day-1"},
    )
    return anomaly, {
        "forecast_path": str(forecast_path),
        "forecast_ensemble_members": int(daily_forecast.sizes.get("member", 50)),
        "forecast_run_label": forecast_attrs.get("run_label"),
        "climatology_path": str(FUXI_CLIMATOLOGY),
        "climatology_slots": [left, right],
        "climatology_right_weight": alpha,
        "climatology_baseline_years": "2002-2021",
        "climatology_title": climatology_attrs.get("title"),
        "units": "mm day-1",
    }


def plot_product(
    dataset: xr.Dataset,
    case_id: str,
    specification: dict,
    outline: object,
    outline_path: object,
    state_segments: list[np.ndarray],
    extent: list[float],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    dpi: int,
    skip_existing: bool,
) -> list[Path]:
    field_set = specification["field"]
    if field_set.dims != ("week", "latitude", "longitude"):
        raise ValueError(
            f"{case_id}/{specification['key']}: unexpected dimensions "
            f"{field_set.dims}"
        )
    if field_set.sizes["week"] != 4:
        raise ValueError(f"{case_id}/{specification['key']}: expected four weeks")

    case_output = OUTPUT / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    stem = case_output / specification["filename"]
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    if skip_existing and png_path.is_file() and pdf_path.is_file():
        return [png_path, pdf_path]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.5), facecolor="white")
    image = None
    for week_index, axis in enumerate(axes.flat, start=1):
        field = field_set.sel(week=week_index)
        if specification["display_mode"] == "native":
            fine_lon = field.longitude.values
            fine_lat = field.latitude.values
            fine_field = field.values
        else:
            fine_lon, fine_lat, fine_field = support.maps.smooth_display_field(
                field.longitude.values,
                field.latitude.values,
                field.values,
                extent,
            )
        if specification["render_mode"] == "native_cells":
            image = axis.pcolormesh(
                fine_lon,
                fine_lat,
                fine_field,
                cmap=cmap,
                norm=norm,
                shading="nearest",
                antialiased=False,
                snap=True,
                rasterized=True,
            )
        else:
            image = axis.contourf(
                fine_lon,
                fine_lat,
                fine_field,
                levels=LEVELS,
                cmap=cmap,
                norm=norm,
                extend="both",
                antialiased=False,
            )
        axis.add_patch(
            PathPatch(
                outline_path,
                transform=axis.transData,
                facecolor="none",
                edgecolor="#172126",
                linewidth=1.30,
                zorder=5,
            )
        )
        axis.add_collection(
            LineCollection(
                state_segments,
                colors="#66777F",
                linewidths=0.34,
                zorder=5,
            )
        )
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect(1.08)
        axis.set_facecolor("#FAFBFC")
        axis.grid(color=GRID, linewidth=0.35, alpha=0.60, linestyle=":")
        axis.tick_params(labelsize=8.4, colors=MUTED, length=3.0)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}°E"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}°N"))

        start = pd.Timestamp(
            dataset.week_start.sel(case=case_id, week=week_index).item()
        )
        end = (
            pd.Timestamp(
                dataset.week_end_exclusive.sel(case=case_id, week=week_index).item()
            )
            - pd.Timedelta(days=1)
        )
        axis.set_title(
            f"Week {week_index}  |  {format_date_range(start, end)}",
            fontsize=12.0,
            fontweight="bold",
            color=specification["accent"],
            pad=8,
        )
        scores = specification["scores"]
        if scores is not None:
            axis.text(
                0.040,
                0.045,
                f"ACC vs IMERG  {scores[week_index]:.2f}",
                transform=axis.transAxes,
                fontsize=8.2,
                fontweight="bold",
                color=specification["accent"],
                bbox={
                    "boxstyle": "round,pad=0.26",
                    "facecolor": "white",
                    "edgecolor": specification["accent"],
                    "linewidth": 0.65,
                    "alpha": 0.94,
                },
                zorder=7,
            )

    fig.suptitle(
        specification["title"],
        x=0.072,
        y=0.985,
        ha="left",
        fontsize=20.0,
        fontweight="bold",
        color=specification["accent"],
    )
    fig.text(
        0.072,
        0.945,
        specification["subtitle"],
        fontsize=9.5,
        color=MUTED,
    )
    if image is None:
        raise RuntimeError("no map image was created")
    colorbar_axis = fig.add_axes([0.18, 0.092, 0.66, 0.027])
    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        extend="both",
    )
    colorbar.set_ticks(LEVELS)
    colorbar.set_label(
        "Weekly rainfall anomaly (mm/day): drier  ←  0  →  wetter",
        fontsize=9.2,
    )
    colorbar.ax.tick_params(labelsize=8.0)
    fig.text(
        0.072,
        0.045,
        specification["note"],
        fontsize=7.7,
        color=MUTED,
    )
    fig.text(
        0.072,
        0.023,
        (
            f"{specification['grid_note']} India boundaries are overlaid; "
            "ACC badges still use only the audited common 1.5° India support."
        ),
        fontsize=7.5,
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        top=0.885,
        bottom=0.165,
        hspace=0.18,
        wspace=0.08,
    )

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> int:
    args = parse_args()
    if args.all_cases and args.case_id:
        raise ValueError("use either --all-cases or --case-id, not both")
    set_style()

    with xr.open_dataset(FIELDS) as source:
        dataset = source.load()
    metrics = pd.read_csv(METRICS)
    available_cases = list(dataset.case.values.astype(str))
    requested_cases = (
        available_cases
        if args.all_cases
        else (args.case_id or ["paired_20240724"])
    )
    if not 0.0 <= args.south_latitude <= 10.0:
        raise ValueError("--south-latitude must be between 0 and 10 degrees north")
    missing = sorted(set(requested_cases) - set(available_cases))
    if missing:
        raise ValueError(f"unknown case IDs: {missing}")
    requested_cases = list(dict.fromkeys(requested_cases))

    _, states, outline = support.maps.presentation_boundaries()
    state_geometries = [
        geometry.simplify(0.02, preserve_topology=True)
        for geometry in states.geometries()
    ]
    if len(state_geometries) != 40:
        raise ValueError(
            f"official boundary contains {len(state_geometries)} records, expected 40"
        )
    state_segments = support.boundary_segments(state_geometries)
    outline_path = shapely_to_path(outline)
    extent = support.maps.full_india_extent(outline)
    extent[2] = float(args.south_latitude)

    cmap = ListedColormap(COLORS, name="imd_rainfall_anomaly_exact")
    cmap.set_under("#d70e00")
    cmap.set_over("#00001e")
    norm = BoundaryNorm(LEVELS, cmap.N)

    native_climatology_audit = native_support.build_native_imerg_climatology()
    imerg_observed, imerg_climatology, imerg_observed_attrs = (
        native_support.load_imerg_native()
    )
    available_erpas_slots = sorted(
        path.name
        for path in native_support.ERPAS_CLIMO_ROOT.iterdir()
        if path.is_dir() and (path / "APCP.grb").is_file()
    )
    erpas_climatology_cache: dict[str, xr.DataArray] = {}
    fuxi_climatology, fuxi_climatology_attrs = load_fuxi_climatology()

    audit_cases = []
    all_outputs: list[Path] = []
    erpas_source_counts = []
    for case_number, case_id in enumerate(requested_cases, start=1):
        print(
            f"rendering individual case {case_number}/{len(requested_cases)}: "
            f"{case_id}",
            flush=True,
        )
        erpas_init = pd.Timestamp(
            dataset.erpas_initialization.sel(case=case_id).item()
        )
        fuxi_init = pd.Timestamp(
            dataset.fuxi_initialization.sel(case=case_id).item()
        )
        imerg_native_anomaly = native_support.imerg_case_anomaly(
            imerg_observed,
            imerg_climatology,
            erpas_init,
        )
        erpas_native_anomaly, erpas_audit = (
            native_support.load_erpas_native_anomaly(
                erpas_init,
                available_erpas_slots,
                erpas_climatology_cache,
            )
        )
        erpas_source_counts.append(int(erpas_audit["source_count"]))
        fuxi_native_anomaly, fuxi_audit = load_fuxi_native_anomaly(
            fuxi_init,
            erpas_init,
            fuxi_climatology,
            fuxi_climatology_attrs,
        )
        audited_fuxi = dataset.forecast_weekly_anomaly.sel(
            model="FuXi-S2S",
            case=case_id,
        )
        aligned_fuxi = fuxi_native_anomaly.sel(
            latitude=audited_fuxi.latitude,
            longitude=audited_fuxi.longitude,
        )
        audited_mask = np.isfinite(audited_fuxi.values)
        fuxi_audit_delta = np.abs(
            aligned_fuxi.values[audited_mask]
            - audited_fuxi.values[audited_mask]
        )
        fuxi_audit["india_audited_cell_count"] = int(audited_mask.sum())
        fuxi_audit["india_max_abs_difference_mm_day"] = float(
            fuxi_audit_delta.max()
        )
        fuxi_audit["india_mean_abs_difference_mm_day"] = float(
            fuxi_audit_delta.mean()
        )
        specifications = product_specifications(
            dataset,
            case_id,
            metrics,
            imerg_native_anomaly,
            fuxi_native_anomaly,
            erpas_native_anomaly,
        )
        case_outputs = []
        for specification in specifications:
            outputs = plot_product(
                dataset,
                case_id,
                specification,
                outline,
                outline_path,
                state_segments,
                extent,
                cmap,
                norm,
                args.dpi,
                args.skip_existing,
            )
            case_outputs.extend(outputs)
            print(
                f"  rendered {specification['label']}: "
                f"{outputs[0].name}",
                flush=True,
            )
        all_outputs.extend(case_outputs)
        audit_cases.append(
            {
                "case_id": case_id,
                "fuxi_initialization": str(
                    pd.Timestamp(
                        dataset.fuxi_initialization.sel(case=case_id).item()
                    ).date()
                ),
                "erpas_initialization": str(
                    pd.Timestamp(
                        dataset.erpas_initialization.sel(case=case_id).item()
                    ).date()
                ),
                "imerg_native_shape": list(imerg_native_anomaly.shape),
                "fuxi_native_shape": list(fuxi_native_anomaly.shape),
                "erpas_native_shape": list(erpas_native_anomaly.shape),
                "fuxi": fuxi_audit,
                "erpas_climatology_slots": erpas_audit["climatology_slots"],
                "erpas_climatology_right_weight": erpas_audit[
                    "climatology_right_weight"
                ],
                "erpas_source_count": int(erpas_audit["source_count"]),
                "native_anomaly_min_max_mm_day": {
                    "IMERG": [
                        float(np.nanmin(imerg_native_anomaly.values)),
                        float(np.nanmax(imerg_native_anomaly.values)),
                    ],
                    "FuXi-S2S": [
                        float(np.nanmin(fuxi_native_anomaly.values)),
                        float(np.nanmax(fuxi_native_anomaly.values)),
                    ],
                    "ERPAS": [
                        float(np.nanmin(erpas_native_anomaly.values)),
                        float(np.nanmax(erpas_native_anomaly.values)),
                    ],
                },
                "outputs": [str(path) for path in case_outputs],
            }
        )

    checks = {
        "requested_cases_exist": not missing,
        "three_products_per_case": all(
            len(item["outputs"]) == 6 for item in audit_cases
        ),
        "all_cases_mode_has_31_cases": (
            not args.all_cases or len(audit_cases) == 31
        ),
        "all_png_and_pdf_outputs_exist": all(path.is_file() for path in all_outputs),
        "four_weeks_in_dataset": dataset.sizes["week"] == 4,
        "fuxi_grid_is_1p5_degrees": bool(
            all(
                item["fuxi_native_shape"] == [4, 27, 27]
                for item in audit_cases
            )
        ),
        "imerg_native_grid_is_0p1_degrees": bool(
            np.allclose(np.abs(np.diff(imerg_observed.latitude.values)), 0.1)
            and np.allclose(np.diff(imerg_observed.longitude.values), 0.1)
        ),
        "erpas_native_grid_is_1p0_degrees": all(
            item["erpas_native_shape"] == [4, 36, 35]
            for item in audit_cases
        ),
        "erpas_source_counts_positive": all(
            count > 0 for count in erpas_source_counts
        ),
        "all_native_fields_extend_outside_india": all(
            item["imerg_native_shape"] == [4, 348, 333]
            and item["fuxi_native_shape"] == [4, 27, 27]
            and item["erpas_native_shape"] == [4, 36, 35]
            for item in audit_cases
        ),
        "fuxi_unmasked_matches_audited_india_values": all(
            item["fuxi"]["india_max_abs_difference_mm_day"] < 1.0e-4
            for item in audit_cases
        ),
        "official_boundary_has_40_records": len(state_geometries) == 40,
        "exact_requested_color_levels": LEVELS.tolist()
        == [-20.0, -15.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 15.0, 20.0],
        "southern_boundary_matches_request": bool(
            np.isclose(
                extent[2],
                args.south_latitude,
            )
        ),
    }
    audit = {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "individual paired-case 2x2 Week-1--4 rainfall-anomaly figures "
            "for presentation"
        ),
        "data_source": str(FIELDS),
        "imerg_native_observation_source": str(native_support.IMERG_OBS),
        "imerg_native_observation_product": imerg_observed_attrs.get("product"),
        "imerg_native_climatology": native_climatology_audit,
        "verification_reference": "IMERG Final V07B",
        "map_extent_degrees": {
            "west": float(extent[0]),
            "east": float(extent[1]),
            "south": float(extent[2]),
            "north": float(extent[3]),
        },
        "shared_levels_mm_day": LEVELS.tolist(),
        "shared_colors": COLORS,
        "under_color": "#d70e00",
        "over_color": "#00001e",
        "scientific_grids": (
            "IMERG 0.1-degree native; FuXi-S2S 1.5-degree native; "
            "ERPAS 1.0-degree native"
        ),
        "display": (
            "IMERG is drawn as direct native cells; FuXi-S2S and ERPAS use "
            "filled contours evaluated on their native nodes. Fields are not "
            "clipped to India; no array resampling or amplitude scaling is applied"
        ),
        "color_scale_policy": (
            "input anomalies are not clipped or rescaled; the fixed shared "
            "colorbar saturates values beyond +/-20 mm/day using extend colors"
        ),
        "checks": checks,
        "cases": audit_cases,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
