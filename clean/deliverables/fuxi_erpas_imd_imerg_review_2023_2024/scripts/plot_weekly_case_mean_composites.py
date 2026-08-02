#!/usr/bin/env python3
"""Plot separate 2x2 Week-1--4 case-mean anomaly composites."""

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
from cartopy.mpl.path import shapely_to_path
import numpy as np
import pandas as pd
from shapely import contains_xy
import xarray as xr


SUPPORT_SCRIPT = HERE / "scripts/plot_native_imerg_erpas_maps.py"
OUTPUT = HERE / "figures/weekly_case_mean_composites_2x2"
AUDIT_PATH = HERE / "logs/weekly_case_mean_composites_audit.json"
COMPOSITE_LEVELS = np.asarray(
    [-5.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 5.0],
    dtype=float,
)


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


support = import_file("weekly_composite_support", SUPPORT_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=280)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": support.INK,
            "xtick.color": support.MUTED,
            "ytick.color": support.MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_composite(
    data: xr.DataArray,
    source_name: str,
    title: str,
    subtitle: str,
    method_note: str,
    stem: Path,
    outline: object,
    outline_path: object,
    state_segments: list[np.ndarray],
    extent: list[float],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    dpi: int,
) -> None:
    if data.dims != ("week", "latitude", "longitude") or data.sizes["week"] != 4:
        raise ValueError(f"{source_name}: unexpected composite dimensions {data.dims}")
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 9.1), facecolor="white")
    image = None
    for week_index, axis in enumerate(axes.flat):
        field = data.sel(week=week_index + 1)
        fine_lon, fine_lat, fine_field = support.maps.smooth_display_field(
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
        axis.add_patch(
            PathPatch(
                outline_path,
                transform=axis.transData,
                facecolor="none",
                edgecolor="#172126",
                linewidth=1.25,
                zorder=5,
            )
        )
        axis.add_collection(
            LineCollection(
                state_segments,
                colors="#5E7078",
                linewidths=0.32,
                zorder=5,
            )
        )
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect(1.08)
        axis.grid(color=support.GRID, linewidth=0.35, alpha=0.55, linestyle=":")
        axis.tick_params(labelsize=8.2, colors=support.MUTED, length=2.8)
        axis.set_title(
            f"Week {week_index + 1}",
            fontsize=13.2,
            fontweight="bold",
            color=support.INK,
            pad=8,
        )
        if week_index % 2:
            axis.set_yticklabels([])
        if week_index < 2:
            axis.set_xticklabels([])

    fig.suptitle(
        title,
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=support.INK,
    )
    fig.text(0.075, 0.945, subtitle, fontsize=9.2, color=support.MUTED)
    colorbar_axis = fig.add_axes([0.20, 0.100, 0.64, 0.025])
    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        extend="both",
    )
    colorbar.set_ticks(COMPOSITE_LEVELS)
    colorbar.set_label(
        "Case-mean weekly rainfall anomaly (mm/day): drier  ←  0  →  wetter",
        fontsize=9.0,
    )
    colorbar.ax.tick_params(labelsize=7.8)
    fig.text(0.075, 0.043, method_note, fontsize=7.6, color=support.MUTED)
    fig.text(
        0.075,
        0.020,
        "Arithmetic mean across 31 cases at each grid cell. Bilinear refinement to 0.15° is display-only; no smoothing changes the underlying anomaly values.",
        fontsize=7.3,
        color=support.MUTED,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.89,
        bottom=0.165,
        hspace=0.10,
        wspace=0.07,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    support.build_native_imerg_climatology()
    observed, imerg_climatology, _ = support.load_imerg_native()
    metrics = pd.read_csv(support.METRICS)
    cases = (
        metrics[
            (metrics.reference == "IMERG Final V07B")
            & (metrics.model == "ERPAS")
        ][["case_id", "erpas_init"]]
        .drop_duplicates()
        .sort_values("erpas_init")
    )
    if len(cases) != 31 or cases.case_id.nunique() != 31:
        raise ValueError("weekly composites require exactly 31 paired cases")

    print("building native IMERG case-mean anomaly", flush=True)
    imerg_fields = []
    for row in cases.itertuples():
        imerg_fields.append(
            support.imerg_case_anomaly(
                observed,
                imerg_climatology,
                pd.Timestamp(row.erpas_init),
            ).values
        )
    imerg_composite = xr.DataArray(
        np.mean(np.stack(imerg_fields), axis=0, dtype=np.float64).astype(np.float32),
        dims=("week", "latitude", "longitude"),
        coords={
            "week": np.arange(1, 5),
            "latitude": observed.latitude.values,
            "longitude": observed.longitude.values,
        },
    )

    print("building audited FuXi-S2S case-mean anomaly", flush=True)
    with xr.open_dataset(support.PROCESSED_FIELDS) as source:
        if list(source.case.values.astype(str)) != list(cases.case_id.astype(str)):
            raise ValueError("processed-field case order differs from composite sample")
        fuxi_composite = (
            source.forecast_weekly_anomaly.sel(model="FuXi-S2S")
            .mean("case", skipna=True)
            .load()
        )

    print("building native ERPAS case-mean anomaly", flush=True)
    available_slots = sorted(
        path.name
        for path in support.ERPAS_CLIMO_ROOT.iterdir()
        if path.is_dir() and (path / "APCP.grb").is_file()
    )
    climo_cache: dict[str, xr.DataArray] = {}
    erpas_fields = []
    source_counts = []
    for case_number, row in enumerate(cases.itertuples(), start=1):
        field, audit = support.load_erpas_native_anomaly(
            pd.Timestamp(row.erpas_init), available_slots, climo_cache
        )
        erpas_fields.append(field.values)
        source_counts.append(int(audit["source_count"]))
        if case_number == 1 or case_number % 10 == 0 or case_number == 31:
            print(f"ERPAS composite source {case_number}/31", flush=True)
    erpas_composite = xr.DataArray(
        np.mean(np.stack(erpas_fields), axis=0, dtype=np.float64).astype(np.float32),
        dims=("week", "latitude", "longitude"),
        coords={
            "week": np.arange(1, 5),
            "latitude": field.latitude.values,
            "longitude": field.longitude.values,
        },
    )

    _, states, outline = support.maps.presentation_boundaries()
    state_geometries = [
        geometry.simplify(0.02, preserve_topology=True)
        for geometry in states.geometries()
    ]
    if len(state_geometries) != 40:
        raise ValueError("official state boundary record count changed")
    state_segments = support.boundary_segments(state_geometries)
    outline_path = shapely_to_path(outline)
    extent = support.maps.full_india_extent(outline)
    cmap = ListedColormap(support.COLORS, name="imd_rainfall_anomaly_exact")
    cmap.set_under("#d70e00")
    cmap.set_over("#00001e")
    norm = BoundaryNorm(COMPOSITE_LEVELS, cmap.N)

    specifications = (
        (
            imerg_composite,
            "IMERG",
            "Observed rainfall anomaly composite",
            "IMERG Final V07B  •  31 paired JJAS starts, 2023–2024  •  observed Week 1–4 patterns",
            "IMERG anomaly = observed weekly rainfall minus fixed 2001–2022 IMERG Final V07B calendar-day climatology. Source grid: 0.1°.",
            OUTPUT / "01_imerg_observed_case_mean_anomaly_weeks1_4",
        ),
        (
            fuxi_composite,
            "FuXi-S2S",
            "FuXi-S2S rainfall anomaly composite",
            "50-member ensemble mean  •  31 paired JJAS starts, 2023–2024  •  forecast Week 1–4 patterns",
            "FuXi anomaly = forecast weekly mean minus native 2002–2021 lead/init-aware model climatology. Source grid: 1.5°.",
            OUTPUT / "02_fuxi_s2s_case_mean_anomaly_weeks1_4",
        ),
        (
            erpas_composite,
            "ERPAS",
            "ERPAS rainfall anomaly composite",
            "Provider-precomputed ensemble mean  •  31 paired JJAS starts, 2023–2024  •  forecast Week 1–4 patterns",
            "ERPAS anomaly = provider forecast mean minus provider reforecast climatology. Source grid: 1.0°.",
            OUTPUT / "03_erpas_case_mean_anomaly_weeks1_4",
        ),
    )
    for data, name, title, subtitle, note, stem in specifications:
        print(f"rendering {name} 2x2 composite", flush=True)
        plot_composite(
            data,
            name,
            title,
            subtitle,
            note,
            stem,
            outline,
            outline_path,
            state_segments,
            extent,
            cmap,
            norm,
            args.dpi,
        )

    output_files = [
        stem.with_suffix(suffix)
        for *_, stem in specifications
        for suffix in (".png", ".pdf")
    ]
    checks = {
        "paired_case_count_31": len(cases) == 31,
        "three_sources_rendered": len(specifications) == 3,
        "six_outputs_exist": all(path.is_file() for path in output_files),
        "four_weeks_each": all(data.sizes["week"] == 4 for data, *_ in specifications),
        "imerg_source_grid_0p1": bool(
            np.isclose(float(np.diff(imerg_composite.longitude.values[:2])[0]), 0.1)
        ),
        "fuxi_source_grid_1p5": bool(
            np.isclose(float(np.diff(fuxi_composite.longitude.values[:2])[0]), 1.5)
        ),
        "erpas_source_grid_1p0": bool(
            np.isclose(float(np.diff(erpas_composite.longitude.values[:2])[0]), 1.0)
        ),
        "official_boundary_has_40_records": len(state_geometries) == 40,
        "erpas_source_counts_positive": all(value > 0 for value in source_counts),
    }
    audit = {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "arithmetic mean of the case-wise weekly anomaly field at each source-grid cell",
        "sample": "31 paired JJAS initializations: 17 in 2023 and 14 in 2024",
        "display": "bilinear refinement to 0.15 degrees after composite calculation",
        "shared_composite_levels_mm_day": COMPOSITE_LEVELS.tolist(),
        "checks": checks,
        "outputs": [str(path) for path in output_files],
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
