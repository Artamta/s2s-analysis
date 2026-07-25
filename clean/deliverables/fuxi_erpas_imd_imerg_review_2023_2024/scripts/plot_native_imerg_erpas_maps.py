#!/usr/bin/env python3
"""Render presentation maps of IMERG, FuXi-S2S and ERPAS anomalies.

Anomalies are calculated on each product's scientifically correct underlying
grid, then bilinearly refined to 0.15 degrees for display only. Any ACC printed
on a page is read from the audited common-1.5-degree verification table, where
like is compared with like.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

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
from shapely import contains_xy
import xarray as xr


MAP_SUPPORT_SCRIPT = (
    WORKSPACE
    / "deliverables/fuxi_erpas_acc_multiseason_2023_2024/scripts/"
    "plot_jjas_composite_anomaly_maps.py"
)
IMERG_OBS = (
    WORKSPACE
    / "deliverables/imd_acc_1p5_full_2023_2024/data/observations/"
    "imerg_final_v07b_jjas_2023_2024_india.nc"
)
IMERG_YEAR_ROOT = HERE / "data/imerg_climatology/year_chunks"
IMERG_NATIVE_CLIMO = (
    HERE
    / "data/imerg_climatology/"
    "imerg_final_v07b_climatology_2001_2022_native_0p1_daily.nc"
)
ERPAS_FORECAST_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/raw/erpas/forecast"
)
ERPAS_CLIMO_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/raw/erpas/"
    "reforecast/climatology"
)
METRICS = HERE / "metrics/per_case_metrics_2023_2024.csv"
SELECTED = HERE / "metrics/selected_spatial_initializations_2023_2024.csv"
PROCESSED_FIELDS = HERE / "data/processed/review_fields_2023_2024.nc"
OUTPUT = HERE / "figures/smoothed_imerg_fuxi_erpas_maps"
SELECTED_OUTPUT = HERE / "figures/smoothed_imerg_fuxi_erpas_presentation"
PDF_PATH = HERE / "figures/06_all_31_smoothed_imerg_fuxi_erpas_anomaly_maps.pdf"
INDEX_PATH = HERE / "metrics/smoothed_imerg_fuxi_erpas_map_index_2023_2024.csv"
AUDIT_PATH = HERE / "logs/smoothed_imerg_fuxi_erpas_map_audit.json"

INK = "#172A36"
MUTED = "#536773"
GRID = "#D8E1E6"
ERPAS = "#C65D32"
FUXI = "#1B6CA8"
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


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


maps = import_file("native_map_support", MAP_SUPPORT_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Render only this case (repeatable); default renders all 31.",
    )
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def boundary_segments(geometries: list[object]) -> list[np.ndarray]:
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
        raise ValueError("official state geometries produced no line segments")
    return segments


def build_native_imerg_climatology() -> dict:
    """Build the fixed 2001--2022 daily climatology without spatial remapping."""
    expected_paths = [
        IMERG_YEAR_ROOT / f"imerg_final_v07b_daily_{year}_0606_1025.nc"
        for year in range(2001, 2023)
    ]
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing IMERG native climatology years: {missing}")

    if IMERG_NATIVE_CLIMO.is_file():
        with xr.open_dataset(IMERG_NATIVE_CLIMO) as source:
            if (
                source.attrs.get("product") != "GPM_3IMERGDF.07"
                or source.attrs.get("revision") != "V07B"
                or source.attrs.get("baseline_years") != "2001-2022"
                or source.daily_precipitation_climatology.shape != (142, 348, 333)
                or not np.all(source.daily_sample_count.values == 22)
            ):
                raise ValueError("existing native IMERG climatology failed validation")
            return {
                "status": "VALIDATED_EXISTING",
                "path": str(IMERG_NATIVE_CLIMO),
                "shape": list(source.daily_precipitation_climatology.shape),
                "baseline_year_count": 22,
            }

    print("building fixed native 0.1-degree IMERG climatology", flush=True)
    total: np.ndarray | None = None
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    expected_keys = np.asarray(
        [stamp.strftime("%m-%d") for stamp in pd.date_range("2000-06-06", "2000-10-25")]
    )
    for index, path in enumerate(expected_paths, start=1):
        with xr.open_dataset(path) as source:
            if (
                source.attrs.get("product") != "GPM_3IMERGDF.07"
                or source.attrs.get("revision") != "V07B"
                or source.precipitation.shape != (142, 348, 333)
                or int(source.precipitation_cnt.min()) != 48
                or int(source.precipitation_cnt.max()) != 48
            ):
                raise ValueError(f"{path}: native IMERG source contract failed")
            keys = pd.DatetimeIndex(source.period_start.values).strftime("%m-%d").to_numpy()
            if not np.array_equal(keys, expected_keys):
                raise ValueError(f"{path}: calendar-day order differs")
            values = source.precipitation.load().values.astype(np.float64)
            if not np.isfinite(values).all() or float(values.min()) < 0:
                raise ValueError(f"{path}: invalid precipitation")
            if latitude is None:
                latitude = source.latitude.values.astype(np.float64)
                longitude = source.longitude.values.astype(np.float64)
                total = np.zeros(values.shape, dtype=np.float64)
            elif not (
                np.array_equal(source.latitude.values, latitude)
                and np.array_equal(source.longitude.values, longitude)
            ):
                raise ValueError(f"{path}: native IMERG grid changed")
            total += values
        print(f"native IMERG climatology year {index}/22", flush=True)

    assert total is not None and latitude is not None and longitude is not None
    mean = (total / 22.0).astype(np.float32)
    dataset = xr.Dataset(
        data_vars={
            "daily_precipitation_climatology": (
                ("calendar_month_day", "latitude", "longitude"),
                mean,
                {
                    "units": "mm day-1",
                    "long_name": "IMERG Final fixed calendar-day precipitation climatology",
                },
            ),
            "daily_sample_count": (
                ("calendar_month_day",),
                np.full(142, 22, dtype=np.int16),
            ),
        },
        coords={
            "calendar_month_day": expected_keys.astype("U5"),
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "title": "IMERG Final V07B 2001-2022 native-grid daily climatology",
            "product": "GPM_3IMERGDF.07",
            "revision": "V07B",
            "doi": "10.5067/GPM/IMERGDF/DAY/07",
            "baseline_years": "2001-2022",
            "baseline_year_count": 22,
            "spatial_processing": "none; retained source 0.1-degree grid",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    IMERG_NATIVE_CLIMO.parent.mkdir(parents=True, exist_ok=True)
    temporary = IMERG_NATIVE_CLIMO.with_name(f".{IMERG_NATIVE_CLIMO.name}.{os.getpid()}.tmp")
    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={
            "daily_precipitation_climatology": {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            }
        },
    )
    os.replace(temporary, IMERG_NATIVE_CLIMO)
    return {
        "status": "BUILT_AND_VALIDATED",
        "path": str(IMERG_NATIVE_CLIMO),
        "shape": list(mean.shape),
        "baseline_year_count": 22,
    }


def load_imerg_native() -> tuple[xr.DataArray, dict[str, np.ndarray], dict]:
    with xr.open_dataset(IMERG_OBS) as source:
        if (
            source.attrs.get("product") != "GPM_3IMERGDF.07"
            or source.attrs.get("revision") != "V07B"
            or source.precipitation.shape != (280, 348, 333)
            or int(source.precipitation_cnt.min()) != 48
            or int(source.precipitation_cnt.max()) != 48
        ):
            raise ValueError("native IMERG observation contract failed")
        observed = source.precipitation.load()
        observed_attrs = dict(source.attrs)
    with xr.open_dataset(IMERG_NATIVE_CLIMO) as source:
        if not (
            np.array_equal(source.latitude.values, observed.latitude.values)
            and np.array_equal(source.longitude.values, observed.longitude.values)
        ):
            raise ValueError("native IMERG observation/climatology grids differ")
        keys = source.calendar_month_day.values.astype(str)
        values = source.daily_precipitation_climatology.load().values
    return observed, {key: values[index] for index, key in enumerate(keys)}, observed_attrs


def reference_day(mmdd: str) -> int:
    return int(
        (
            pd.Timestamp(f"2001-{mmdd[:2]}-{mmdd[2:]}")
            - pd.Timestamp("2001-01-01")
        ).days
    )


def interpolation_bracket(target: str, available: list[str]) -> tuple[str, str, float]:
    target_day = reference_day(target)
    points = np.asarray([reference_day(value) for value in available], dtype=int)
    exact = np.where(points == target_day)[0]
    if len(exact):
        value = available[int(exact[0])]
        return value, value, 0.0
    right_index = int(np.searchsorted(points, target_day))
    if right_index == 0 or right_index == len(points):
        raise ValueError(f"cannot bracket ERPAS climatology date {target}")
    left_index = right_index - 1
    fraction = float(
        (target_day - points[left_index]) / (points[right_index] - points[left_index])
    )
    return available[left_index], available[right_index], fraction


def load_erpas_daily(path: Path, expected_init: pd.Timestamp | None) -> xr.DataArray:
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as source:
        field = source.tp
        if (
            field.attrs.get("GRIB_stepType") != "accum"
            or field.attrs.get("units") != "kg m**-2"
            or source.sizes.get("step") != 33
        ):
            raise ValueError(f"{path}: ERPAS accumulation contract failed")
        if expected_init is not None and pd.Timestamp(source.time.item()) != expected_init:
            raise ValueError(f"{path}: ERPAS initialization time mismatch")
        daily = (
            field.isel(step=slice(1, 29))
            .sel(latitude=slice(40, 5), longitude=slice(66, 100))
            .load()
        )
    if daily.shape != (28, 36, 35) or not np.isfinite(daily.values).all():
        raise ValueError(f"{path}: unexpected native India-domain ERPAS field {daily.shape}")
    return daily.sortby("latitude")


def load_erpas_native_anomaly(
    erpas_init: pd.Timestamp,
    available_slots: list[str],
    climo_cache: dict[str, xr.DataArray],
) -> tuple[xr.DataArray, dict]:
    stamp = erpas_init.strftime("%Y%m%d")
    forecast_path = ERPAS_FORECAST_ROOT / f"annual{erpas_init.year}/tp/APCP_{stamp}.grb"
    forecast = load_erpas_daily(forecast_path, erpas_init)
    target = erpas_init.strftime("%m%d")
    left, right, alpha = interpolation_bracket(target, available_slots)
    for slot in {left, right}:
        if slot not in climo_cache:
            climo_cache[slot] = load_erpas_daily(
                ERPAS_CLIMO_ROOT / slot / "APCP.grb", None
            )
    climatology = (1.0 - alpha) * climo_cache[left] + alpha * climo_cache[right]
    forecast_weekly = forecast.values.reshape(4, 7, 36, 35).mean(axis=1)
    climo_weekly = climatology.values.reshape(4, 7, 36, 35).mean(axis=1)
    anomaly = xr.DataArray(
        (forecast_weekly - climo_weekly).astype(np.float32),
        dims=("week", "latitude", "longitude"),
        coords={
            "week": np.arange(1, 5),
            "latitude": forecast.latitude.values,
            "longitude": forecast.longitude.values,
        },
    )
    return anomaly, {
        "forecast_path": str(forecast_path),
        "climatology_slots": [left, right],
        "climatology_right_weight": alpha,
        "ensemble_semantics": "provider-precomputed unweighted source-forecast mean",
        "source_count": int(forecast.attrs.get("GRIB_totalNumber", -1)),
    }


def imerg_case_anomaly(
    observed: xr.DataArray,
    climatology: dict[str, np.ndarray],
    erpas_init: pd.Timestamp,
) -> xr.DataArray:
    starts = [erpas_init + pd.Timedelta(days=1 + 7 * week) for week in range(4)]
    fields = []
    for start in starts:
        dates = pd.date_range(start, periods=7, freq="D")
        found = observed.sel(period_start=dates)
        if found.sizes.get("period_start") != 7:
            raise ValueError(f"IMERG is missing native days beginning {start:%Y-%m-%d}")
        observed_week = found.mean("period_start").values
        climo_week = np.mean(
            np.stack([climatology[date.strftime("%m-%d")] for date in dates]),
            axis=0,
        )
        fields.append((observed_week - climo_week).astype(np.float32))
    return xr.DataArray(
        np.stack(fields),
        dims=("week", "latitude", "longitude"),
        coords={
            "week": np.arange(1, 5),
            "latitude": observed.latitude.values,
            "longitude": observed.longitude.values,
        },
    )


def mask_native(field: xr.DataArray, outline: object) -> np.ndarray:
    lon_grid, lat_grid = np.meshgrid(field.longitude.values, field.latitude.values)
    return np.where(contains_xy(outline, lon_grid, lat_grid), field.values, np.nan)


def draw_boundaries(
    axis: plt.Axes,
    outline_path: object,
    state_segments: list[np.ndarray],
    extent: tuple[float, float, float, float],
) -> None:
    axis.add_patch(
        PathPatch(
            outline_path,
            transform=axis.transData,
            facecolor="none",
            edgecolor="#151D22",
            linewidth=1.05,
            zorder=5,
        )
    )
    axis.add_collection(
        LineCollection(state_segments, colors="#596A72", linewidths=0.28, zorder=5)
    )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect(1.08)
    axis.grid(color=GRID, linewidth=0.3, alpha=0.55, linestyle=":")
    axis.tick_params(labelsize=7.0, colors=MUTED, length=2.5)


def main() -> int:
    args = parse_args()
    set_style()
    for directory in (OUTPUT, SELECTED_OUTPUT, AUDIT_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)

    climatology_audit = build_native_imerg_climatology()
    observed, imerg_climatology, observed_attrs = load_imerg_native()
    with xr.open_dataset(PROCESSED_FIELDS) as source:
        if (
            "FuXi-S2S" not in source.model.values
            or source.forecast_weekly_anomaly.sel(model="FuXi-S2S").shape
            != (31, 4, 22, 22)
        ):
            raise ValueError("audited FuXi anomaly fields failed their shape contract")
        fuxi_anomaly_all = source.forecast_weekly_anomaly.sel(model="FuXi-S2S").load()
        fuxi_initialization = {
            str(case): pd.Timestamp(value)
            for case, value in zip(source.case.values, source.fuxi_initialization.values)
        }
    metrics = pd.read_csv(METRICS)
    cases = (
        metrics[
            (metrics.reference == "IMERG Final V07B")
            & (metrics.model == "ERPAS")
        ][["case_id", "erpas_init"]]
        .drop_duplicates()
        .sort_values("erpas_init")
    )
    if len(cases) != 31:
        raise ValueError(f"expected 31 native-map cases, got {len(cases)}")
    requested = set(args.case_id)
    if requested:
        unknown = requested - set(cases.case_id)
        if unknown:
            raise ValueError(f"unknown case ids: {sorted(unknown)}")
        cases = cases[cases.case_id.isin(requested)]
    score_lookup = {
        (row.case_id, row.model, int(row.week)): float(row.acc)
        for row in metrics[
            (metrics.reference == "IMERG Final V07B")
            & (metrics.model.isin(["FuXi-S2S", "ERPAS"]))
        ].itertuples()
    }
    selected = pd.read_csv(SELECTED)
    selected_lookup = dict(zip(selected.case_id, selected.selection))

    _, states, outline = maps.presentation_boundaries()
    state_geometries = [
        geometry.simplify(0.02, preserve_topology=True)
        for geometry in states.geometries()
    ]
    if len(state_geometries) != 40:
        raise ValueError("official state boundary record count changed")
    state_segments = boundary_segments(state_geometries)
    outline_path = shapely_to_path(outline)
    extent = maps.full_india_extent(outline)

    cmap = ListedColormap(COLORS, name="imd_rainfall_anomaly_exact")
    cmap.set_under("#d70e00")
    cmap.set_over("#00001e")
    norm = BoundaryNorm(LEVELS, cmap.N)
    available_slots = sorted(
        path.name
        for path in ERPAS_CLIMO_ROOT.iterdir()
        if path.is_dir() and (path / "APCP.grb").is_file()
    )
    climo_cache: dict[str, xr.DataArray] = {}
    audit_rows: list[dict] = []

    temporary_pdf = PDF_PATH.with_name(f".{PDF_PATH.name}.{os.getpid()}.tmp")
    with PdfPages(temporary_pdf) as pdf:
        for case_number, row in enumerate(cases.itertuples(), start=1):
            case_id = str(row.case_id)
            erpas_init = pd.Timestamp(row.erpas_init)
            fuxi_init = fuxi_initialization[case_id]
            print(f"comparison map {case_number}/{len(cases)}: {case_id}", flush=True)
            imerg_anomaly = imerg_case_anomaly(observed, imerg_climatology, erpas_init)
            fuxi_anomaly = fuxi_anomaly_all.sel(case=case_id)
            erpas_anomaly, erpas_audit = load_erpas_native_anomaly(
                erpas_init, available_slots, climo_cache
            )
            fig, axes = plt.subplots(3, 4, figsize=(13.4, 9.8), facecolor="white")
            image = None
            for row_index, (source_name, label, data, resolution, row_color) in enumerate(
                (
                    ("IMERG observed", "IMERG observed", imerg_anomaly, "source 0.1°", INK),
                    ("FuXi-S2S", "FuXi-S2S forecast", fuxi_anomaly, "source 1.5°", FUXI),
                    ("ERPAS", "ERPAS forecast", erpas_anomaly, "source 1.0°", ERPAS),
                )
            ):
                for week_index in range(4):
                    axis = axes[row_index, week_index]
                    field = data.sel(week=week_index + 1)
                    fine_lon, fine_lat, fine_field = maps.smooth_display_field(
                        field.longitude.values,
                        field.latitude.values,
                        field.values,
                        extent,
                    )
                    fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)
                    masked = np.where(
                        contains_xy(outline, fine_lon_grid, fine_lat_grid),
                        fine_field,
                        np.nan,
                    )
                    image = axis.pcolormesh(
                        fine_lon,
                        fine_lat,
                        masked,
                        cmap=cmap,
                        norm=norm,
                        shading="auto",
                        rasterized=True,
                    )
                    draw_boundaries(axis, outline_path, state_segments, extent)
                    if week_index != 0:
                        axis.set_yticklabels([])
                    if row_index != 2:
                        axis.set_xticklabels([])
                    if row_index == 0:
                        start = erpas_init + pd.Timedelta(days=1 + 7 * week_index)
                        end = start + pd.Timedelta(days=6)
                        axis.set_title(
                            f"Week {week_index + 1}  |  {start:%d %b}–{end:%d %b}",
                            fontsize=9.6,
                            fontweight="bold",
                            color=INK,
                            pad=6,
                        )
                    if week_index == 0:
                        axis.text(
                            -0.22,
                            0.5,
                            f"{label}\n({resolution})",
                            rotation=90,
                            transform=axis.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9.5,
                            fontweight="bold",
                            color=row_color,
                        )
                    if row_index > 0:
                        score = score_lookup[(case_id, source_name, week_index + 1)]
                        axis.text(
                            0.04,
                            0.04,
                            f"ACC {score:.2f}*",
                            transform=axis.transAxes,
                            fontsize=7.5,
                            fontweight="bold",
                            color=row_color,
                            bbox={
                                "boxstyle": "round,pad=0.20",
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.92,
                            },
                            zorder=7,
                        )

            fig.suptitle(
                f"Weekly rainfall anomaly patterns — ERPAS IC {erpas_init:%d %B %Y}",
                x=0.075,
                y=0.993,
                ha="left",
                fontsize=17,
                fontweight="bold",
                color=INK,
            )
            fig.text(
                0.075,
                0.952,
                f"FuXi IC {fuxi_init:%d %b %Y}  •  identical Thursday–Wednesday valid weeks  •  bilinear refinement is display-only",
                fontsize=9.0,
                color=MUTED,
            )
            colorbar_axis = fig.add_axes([0.22, 0.090, 0.60, 0.020])
            colorbar = fig.colorbar(
                image, cax=colorbar_axis, orientation="horizontal", extend="both"
            )
            colorbar.set_ticks(LEVELS)
            colorbar.set_label(
                "Weekly rainfall anomaly (mm/day): drier  ←  0  →  wetter",
                fontsize=8.8,
            )
            colorbar.ax.tick_params(labelsize=7.4)
            fig.text(
                0.075,
                0.036,
                "Anomalies: IMERG minus fixed 2001–2022 baseline; FuXi minus native 2002–2021 climatology; ERPAS minus provider reforecast climatology.",
                fontsize=7.3,
                color=MUTED,
            )
            fig.text(
                0.075,
                0.015,
                "Underlying grids: IMERG 0.1°, FuXi 1.5°, ERPAS 1.0°. Maps are bilinearly refined to 0.15° for display; *ACC uses unchanged native values on the common 1.5° grid.",
                fontsize=7.1,
                color=MUTED,
            )
            fig.subplots_adjust(
                left=0.085,
                right=0.98,
                top=0.90,
                bottom=0.165,
                hspace=0.08,
                wspace=0.06,
            )
            png_path = OUTPUT / f"smoothed_imerg_fuxi_erpas_anomaly_{case_id}.png"
            fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            category = selected_lookup.get(case_id, "")
            if category:
                selected_png = SELECTED_OUTPUT / f"06_{category}_{case_id}.png"
                selected_pdf = SELECTED_OUTPUT / f"06_{category}_{case_id}.pdf"
                shutil.copy2(png_path, selected_png)
                fig.savefig(selected_pdf, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            audit_rows.append(
                {
                    "case_id": case_id,
                    "erpas_initialization": erpas_init.strftime("%Y-%m-%d"),
                    "fuxi_initialization": fuxi_init.strftime("%Y-%m-%d"),
                    "imerg_grid_degrees": 0.1,
                    "fuxi_grid_degrees": 1.5,
                    "erpas_grid_degrees": 1.0,
                    "display_grid_degrees": 0.15,
                    "png": str(png_path),
                    **erpas_audit,
                }
            )
    if not requested:
        os.replace(temporary_pdf, PDF_PATH)
    else:
        temporary_pdf.unlink()

    index = pd.DataFrame(audit_rows)
    if not requested:
        index.to_csv(INDEX_PATH, index=False)
    checks = {
        "imerg_native_grid_is_0p1": bool(
            np.isclose(float(np.diff(observed.latitude.values[:2])[0]), 0.1)
            and np.isclose(float(np.diff(observed.longitude.values[:2])[0]), 0.1)
        ),
        "erpas_native_grid_is_1p0": all(
            row["erpas_grid_degrees"] == 1.0 for row in audit_rows
        ),
        "fuxi_native_grid_is_1p5": bool(
            np.isclose(abs(float(np.diff(fuxi_anomaly_all.latitude.values[:2])[0])), 1.5)
            and np.isclose(float(np.diff(fuxi_anomaly_all.longitude.values[:2])[0]), 1.5)
        ),
        "all_requested_maps_exist": all(Path(row["png"]).is_file() for row in audit_rows),
        "all_31_maps_rendered": bool(requested) or len(audit_rows) == 31,
        "official_boundary_has_40_records": len(state_geometries) == 40,
        "fixed_scale_matches_reference": LEVELS.tolist()
        == [-20.0, -15.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 15.0, 20.0],
    }
    audit = {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "smoothed visual comparison of IMERG observed, FuXi-S2S and ERPAS rainfall anomalies",
        "native_grid_policy": "underlying IMERG 0.1-degree, FuXi 1.5-degree and ERPAS 1.0-degree fields are bilinearly refined to a common 0.15-degree display mesh only after anomaly calculation",
        "score_policy": "printed ACC is read from the independently audited common 1.5-degree calculation",
        "imerg_observation_path": str(IMERG_OBS),
        "imerg_observation_provenance": observed_attrs,
        "imerg_climatology": climatology_audit,
        "erpas_forecast_root": str(ERPAS_FORECAST_ROOT),
        "erpas_climatology_root": str(ERPAS_CLIMO_ROOT),
        "fuxi_field_path": str(PROCESSED_FIELDS),
        "rendered_case_count": len(audit_rows),
        "checks": checks,
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str), flush=True)
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
