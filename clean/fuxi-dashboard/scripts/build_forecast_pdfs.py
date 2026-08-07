#!/usr/bin/env python3
"""Build product-aware PDFs from compact India forecast JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atmos42_pdf_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/atmos42_pdf_xdg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT / "public"
DATA = PUBLIC / "data"
BRAND_ASSETS = ROOT / "scripts" / "assets"

from science.catalog import build_catalog  # noqa: E402
from science.validators import write_json  # noqa: E402

PRODUCTS = {
    "rainfall_total": {
        "title": "Weekly mean rainfall",
        "units": "mm day⁻¹",
        "boundaries": [0, 1, 2, 5, 10, 20, 40, 60],
        "colors": [
            "#ffffff", "#b7ffb8", "#71f27b", "#24d13b",
            "#009a18", "#006b12", "#003d0c",
        ],
        "under": "#ffffff",
        "over": "#002807",
        "operation": "divide_by_seven",
        "note": "Mean daily rainfall rate over each 7-day forecast week.",
    },
    "rainfall_anomaly": {
        "title": "Weekly mean rainfall anomaly",
        "units": "mm day⁻¹",
        "boundaries": [-20, -15, -10, -5, -2, 2, 5, 10, 15, 20],
        "colors": [
            "#ff5200", "#ff8e1d", "#ffca59", "#fff4a5", "#ffffff",
            "#c8c8e9", "#8c8cbf", "#6464a3", "#3c3c87",
        ],
        "under": "#d70e00",
        "over": "#00001e",
        "operation": "identity",
        "note": "Difference from the model's typical rainfall for the same season and forecast lead (2002–2021).",
    },
    "temperature_mean": {
        "title": "Weekly mean 2 m temperature",
        "units": "°C",
        "boundaries": [10, 14, 18, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42],
        "colors": [
            "#bebee2", "#fffacd", "#fff191", "#ffe271", "#ffca59",
            "#ffa635", "#ff8e1d", "#ff6a00", "#ff3a00", "#eb1800",
            "#c30400", "#9b0000", "#730000",
        ],
        "under": "#9696c6",
        "over": "#5f0000",
        "operation": "identity",
        "note": "Mean of seven daily-mean 2 m temperature fields.",
    },
    "temperature_anomaly": {
        "title": "Weekly mean temperature anomaly",
        "units": "°C",
        "boundaries": [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
        "colors": [
            "#383880", "#5a5a9c", "#8282b8", "#aaaad4", "#c8c8e9",
            "#dcdcf7", "#fffacd", "#ffa635", "#ff7605", "#ff5e00",
            "#ff2e00", "#c30400",
        ],
        "under": "#10103a",
        "over": "#730000",
        "operation": "identity",
        "note": "Difference from the model's typical temperature for the same season and forecast lead (2002–2021).",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=DATA / "index.json", help="Issue catalog"
    )
    parser.add_argument("--source", choices=("gfs", "era5"))
    parser.add_argument("--issue", help="Optional YYYYMMDD issue filter")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rings(geometry: dict[str, Any]) -> Iterable[list[list[float]]]:
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield from polygon


def geometry_segments(payload: dict[str, Any]) -> list[np.ndarray]:
    """Extract regional geometry once for efficient reuse across PDF panels."""

    features = payload.get("features")
    geometries = (
        (feature.get("geometry") for feature in features)
        if features is not None
        else (payload.get("geometry"),)
    )
    segments: list[np.ndarray] = []
    for geometry in geometries:
        if not geometry:
            continue
        for ring in rings(geometry):
            array = np.asarray(ring, dtype=np.float64)
            if array.size == 0:
                continue
            if (
                array[:, 0].max() < 59.0
                or array[:, 0].min() > 100.0
                or array[:, 1].max() < -1.0
                or array[:, 1].min() > 40.0
            ):
                continue
            segments.append(array[:, :2])
    return segments


def draw_geometry(
    axes: plt.Axes,
    segments: list[np.ndarray],
    color: str,
    width: float,
) -> None:
    axes.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=width,
            joinstyle="round",
            capstyle="round",
            zorder=4,
        )
    )


def bilinear_display(values: np.ndarray, size: int = 162) -> np.ndarray:
    """Match the website's clamped bilinear 1.5°→visual 0.25° rendering."""

    y = np.arange(size, dtype=np.float64)
    x = np.arange(size, dtype=np.float64)
    sampled_latitude = 39.75 - ((y + 0.5) / size) * 40.5
    sampled_longitude = 59.25 + ((x + 0.5) / size) * 40.5
    row = (39.0 - sampled_latitude) / 1.5
    column = (sampled_longitude - 60.0) / 1.5
    row0 = np.floor(row).astype(int)
    column0 = np.floor(column).astype(int)
    row_weight = row - row0
    column_weight = column - column0
    r0 = np.clip(row0, 0, values.shape[0] - 1)
    r1 = np.clip(row0 + 1, 0, values.shape[0] - 1)
    c0 = np.clip(column0, 0, values.shape[1] - 1)
    c1 = np.clip(column0 + 1, 0, values.shape[1] - 1)
    top = (
        values[r0[:, None], c0[None, :]] * (1.0 - column_weight[None, :])
        + values[r0[:, None], c1[None, :]] * column_weight[None, :]
    )
    bottom = (
        values[r1[:, None], c0[None, :]] * (1.0 - column_weight[None, :])
        + values[r1[:, None], c1[None, :]] * column_weight[None, :]
    )
    return top * (1.0 - row_weight[:, None]) + bottom * row_weight[:, None]


def friendly_day(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%d %b %Y")


def build_pdf(
    forecast_path: Path,
    source: dict[str, Any],
    issue: dict[str, Any],
    world: dict[str, Any],
    outline: dict[str, Any],
    admin: dict[str, Any],
) -> Path:
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    source_id = source["id"]
    issue_id = issue["id"]
    output_root = PUBLIC / "downloads" / source_id / issue_id
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / (
        f"S2S_Research_India_Weeks1-4_{source_id.upper()}_{issue_id}.pdf"
    )
    for pattern in (
        "Atmosphere42_India_*.pdf",
        "SCDLDS_India_Experimental_Forecast_*.pdf",
        "S2S_Research_India_Forecast_*.pdf",
        "S2S_Research_India_Weeks1-4_*.pdf",
    ):
        for stale_pdf in output_root.glob(pattern):
            if stale_pdf != output:
                stale_pdf.unlink()

    world_segments = geometry_segments(world)
    outline_segments = geometry_segments(outline)
    admin_segments = geometry_segments(admin)
    initialization = forecast["issue"]["initialization"][:10]
    source_label = forecast["issue"]["initial_condition_source"]["label"]
    members = forecast["issue"]["members"]
    warning = (
        "Rapid 5-member experimental prototype"
        if members < 100
        else "Experimental research guidance · not an operational warning"
    )
    ashoka_logo = plt.imread(BRAND_ASSETS / "ashoka-university.png")
    centre_logo = plt.imread(BRAND_ASSETS / "scdlds-centre.png")
    metadata = {
        "Title": f"S2S Research India experimental forecast · {source_id.upper()} · {issue_id}",
        "Author": "S2S Research",
        "Subject": "Experimental India subseasonal forecast guidance, Weeks 1–4",
        "Keywords": "India, S2S, rainfall, temperature, anomaly, experimental",
    }
    available_products = [
        key for key in PRODUCTS if key in forecast.get("products", {})
    ]
    if not available_products:
        raise ValueError(f"{forecast_path}: no supported products")
    with PdfPages(output, metadata=metadata) as pdf:
        for product_key in available_products:
            definition = PRODUCTS[product_key]
            cmap = ListedColormap(definition["colors"])
            cmap.set_under(definition["under"])
            cmap.set_over(definition["over"])
            norm = BoundaryNorm(definition["boundaries"], cmap.N)
            figure, axes_grid = plt.subplots(
                2,
                2,
                figsize=(11.7, 16.5),
                facecolor="white",
            )
            figure.subplots_adjust(
                left=0.075,
                right=0.94,
                top=0.76,
                bottom=0.12,
                wspace=0.10,
                hspace=0.15,
            )
            ashoka_axes = figure.add_axes((0.06, 0.925, 0.047, 0.035))
            ashoka_axes.imshow(ashoka_logo)
            ashoka_axes.axis("off")
            centre_axes = figure.add_axes((0.115, 0.925, 0.16, 0.035))
            centre_axes.imshow(centre_logo)
            centre_axes.axis("off")
            figure.text(
                0.29,
                0.943,
                "EXPERIMENTAL SUBSEASONAL FORECASTING",
                color="#a63f2d",
                fontsize=9.5,
                fontweight="bold",
            )
            figure.text(
                0.06,
                0.875,
                definition["title"],
                color="#172033",
                fontsize=25,
                fontweight="bold",
            )
            figure.text(
                0.06,
                0.828,
                (
                    f"Initialized {friendly_day(initialization)} · {source_label} · "
                    f"{members}-member ensemble mean · Weeks 1–4"
                ),
                color="#52627a",
                fontsize=10,
            )
            figure.text(
                0.965,
                0.917,
                "EXPERIMENTAL",
                ha="right",
                color="#8a431f",
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.35,rounding_size=0.08",
                    "facecolor": "#fff1df",
                    "edgecolor": "#e7b477",
                    "linewidth": 0.8,
                },
            )

            image = None
            for axes, week in zip(
                axes_grid.flat, forecast["weeks"][:4], strict=True
            ):
                values = np.asarray(week["fields"][product_key], dtype=np.float64).reshape(27, 27)
                if definition["operation"] == "divide_by_seven":
                    values = values / 7.0
                axes.set_facecolor("#edf5fb")
                visual_values = bilinear_display(values)
                image = axes.imshow(
                    visual_values,
                    extent=(59.25, 99.75, -0.75, 39.75),
                    origin="upper",
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                    aspect="equal",
                    rasterized=True,
                    zorder=1,
                )
                draw_geometry(axes, world_segments, "#777777", 0.38)
                draw_geometry(axes, outline_segments, "#161616", 1.0)
                draw_geometry(axes, admin_segments, "#333333", 0.38)
                axes.set_xlim(59.25, 99.75)
                axes.set_ylim(-0.75, 39.75)
                axes.set_aspect("equal", adjustable="box")
                axes.set_xticks([60, 70, 80, 90, 100])
                axes.set_yticks([0, 10, 20, 30, 40])
                axes.tick_params(labelsize=7, length=2, colors="#455064")
                axes.grid(color="#a7b7c8", linewidth=0.35, alpha=0.45, linestyle=":")
                axes.set_title(
                    f"WEEK {week['week']}  |  {friendly_day(week['valid_start'])} – {friendly_day(week['valid_end'])}",
                    loc="left",
                    color="#173f8a",
                    fontsize=9.5,
                    fontweight="bold",
                    pad=6,
                )
                for spine in axes.spines.values():
                    spine.set_color("#303947")
                    spine.set_linewidth(0.55)

            if image is None:
                raise ValueError(f"{forecast_path}: no forecast weeks")
            color_axes = figure.add_axes((0.20, 0.075, 0.60, 0.025))
            colorbar = figure.colorbar(
                image,
                cax=color_axes,
                orientation="horizontal",
                extend="both",
                ticks=definition["boundaries"],
                spacing="proportional",
            )
            colorbar.ax.tick_params(labelsize=7, length=2)
            colorbar.set_label(definition["units"], fontsize=9, fontweight="bold")
            figure.text(
                0.06,
                0.035,
                f"{definition['note']}  Visual-only bilinear interpolation 1.5°→0.25°; values remain native-grid.",
                color="#52627a",
                fontsize=7,
            )
            figure.text(
                0.94,
                0.017,
                warning,
                ha="right",
                color="#8a431f",
                fontsize=7,
                fontweight="bold",
            )
            pdf.savefig(figure, dpi=180, facecolor="white")
            plt.close(figure)

    relative_pdf = output.relative_to(PUBLIC).as_posix()
    compact_json = forecast["issue"].get("downloads", {}).get(
        "compact_json", forecast_path.relative_to(PUBLIC).as_posix()
    )
    forecast["issue"]["downloads"] = {
        "compact_json": compact_json,
        "india_pdf": relative_pdf,
        "india_pdf_sha256": sha256(output),
    }
    forecast_path.write_text(
        json.dumps(forecast, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    args = parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    world = json.loads((DATA / "world-countries.geojson").read_text(encoding="utf-8"))
    outline = json.loads((DATA / "india-outline.json").read_text(encoding="utf-8"))
    admin = json.loads((DATA / "india-admin.json").read_text(encoding="utf-8"))
    built: list[Path] = []
    for source in index["initial_condition_sources"]:
        if args.source and source["id"] != args.source:
            continue
        for issue in source["issues"]:
            if args.issue and issue["id"] != args.issue:
                continue
            forecast_path = DATA / issue["forecast"]
            built.append(build_pdf(forecast_path, source, issue, world, outline, admin))
    if not built:
        raise ValueError("no forecast issue matched the requested PDF filters")
    legacy = DATA / "forecasts/20260728.json"
    canonical = DATA / "forecasts/gfs/20260728.json"
    legacy.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    refreshed_index = build_catalog(
        json.loads(args.index.read_text(encoding="utf-8")),
        PUBLIC,
        datetime.now(timezone.utc),
    )
    write_json(args.index, refreshed_index)
    print(json.dumps({"pdfs": len(built), "paths": [str(path) for path in built]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
