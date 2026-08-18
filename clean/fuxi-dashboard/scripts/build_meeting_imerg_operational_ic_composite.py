#!/usr/bin/env python3
"""Build a five-case operational-IC rainfall-anomaly composite against IMERG.

This is a deliberately small presentation pilot.  ERA5-, GFS-, and native
operational-IFS-initialized FuXi forecasts all use five members.  ERPAS and
the three FuXi forecasts verify on identical Thursday--Wednesday weeks.  For
this visual comparison every anomaly is raw weekly-mean rainfall minus the
same fixed IMERG Final V07B 2001--2022 calendar-day climatology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fuxi_meeting_imerg_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fuxi_meeting_imerg_cache")
PROJ_DATA = Path(sys.prefix) / "share" / "proj"
if PROJ_DATA.is_dir():
    os.environ.setdefault("PROJ_DATA", str(PROJ_DATA))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import shapefile
from pyproj import CRS, Transformer
import pyproj
from shapely.geometry import shape
from shapely.ops import transform, unary_union
import xarray as xr

if PROJ_DATA.is_dir():
    pyproj.datadir.set_data_dir(str(PROJ_DATA))


ANALYSIS_ROOT = ROOT.parent
REVIEW_ROOT = ANALYSIS_ROOT / "deliverables/fuxi_erpas_imd_imerg_review_2023_2024"
REVIEW_FIELDS = REVIEW_ROOT / "data/processed/review_fields_2023_2024.nc"
IMERG_CLIMATOLOGY = (
    REVIEW_ROOT
    / "data/imerg_climatology/imerg_final_v07b_climatology_2001_2022_1p5_daily.nc"
)
CONFIG = ROOT / "config/ic-skill-current-ifs-5.json"
INDIA_SHAPEFILE = Path("/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp")
DEFAULT_OUTPUT = ROOT / "reports/meeting-imerg-operational-ic-20260818"

SYSTEMS = ("observed", "erpas", "era5", "gfs", "ifs")
LABELS = {
    "observed": "IMERG Final V07B\nobserved",
    "erpas": "ERPAS",
    "era5": "FuXi-S2S\nERA5 IC · 5 members",
    "gfs": "FuXi-S2S\nGFS IC · 5 members",
    "ifs": "FuXi-S2S\nnative operational IFS IC · 5 members",
}
RUN_LABELS = {
    "era5": "fuxi_s2s_era5_case_{stamp}_ens5",
    "gfs": "fuxi_s2s_gfs_case_{stamp}_ens5",
    "ifs": "fuxi_s2s_ifs_native_operational_0p25_ic_pilot_ens5",
}
WEEKS = (1, 3)
LEVELS = np.asarray([-12, -8, -5, -3, -1, 1, 3, 5, 8, 12], dtype=float)
COLORS = (
    "#C93120",
    "#E96B35",
    "#F4A65B",
    "#F9D49B",
    "#F7F7F4",
    "#D9D7EB",
    "#ACA9D1",
    "#7470B3",
    "#3C368C",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forecast_path(config: dict, system: str, issue: pd.Timestamp) -> Path:
    return Path(
        config["forecast_paths"][system].format(
            yyyymmdd=issue.strftime("%Y%m%d"), year=issue.year
        )
    )


def expected_periods(dataset: xr.Dataset, case_id: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for week in range(1, 5):
        start = pd.Timestamp(dataset.week_start.sel(case=case_id, week=week).item())
        end = pd.Timestamp(dataset.week_end_exclusive.sel(case=case_id, week=week).item())
        starts.extend(pd.date_range(start, end - pd.Timedelta(days=1), freq="D"))
        ends.extend(pd.date_range(start + pd.Timedelta(days=1), end, freq="D"))
    return pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)


def load_fuxi_weekly(
    path: Path,
    system: str,
    issue: pd.Timestamp,
    latitude: np.ndarray,
    longitude: np.ndarray,
    expected_starts: pd.DatetimeIndex,
    expected_ends: pd.DatetimeIndex,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as source:
        expected_label = RUN_LABELS[system].format(stamp=issue.strftime("%Y%m%d"))
        if str(source.attrs.get("run_label", "")) != expected_label:
            raise ValueError(f"{path}: unexpected run label")
        if int(source.sizes.get("member", 0)) != 5 or int(source.sizes.get("lead_day", 0)) != 42:
            raise ValueError(f"{path}: expected 5 members and 42 lead days")
        if source.tp.attrs.get("units") != "mm h-1":
            raise ValueError(f"{path}: TP units are not mm h-1")
        if pd.Timestamp(source.init_time.item()) != issue:
            raise ValueError(f"{path}: initialization time mismatch")
        daily = (
            source.tp.sel(lead_day=slice(4, 31))
            .sel(latitude=latitude, longitude=longitude)
            .mean("member")
            .load()
            .astype(np.float64)
            * 24.0
        )
        starts = pd.DatetimeIndex(
            source.forecast_period_start.sel(lead_day=slice(4, 31)).values
        )
        ends = pd.DatetimeIndex(
            source.forecast_period_end.sel(lead_day=slice(4, 31)).values
        )
    if not starts.equals(expected_starts) or not ends.equals(expected_ends):
        raise ValueError(f"{path}: forecast periods do not match ERPAS valid weeks")
    if daily.shape != (28, len(latitude), len(longitude)) or not np.isfinite(daily.values).all():
        raise ValueError(f"{path}: unexpected or non-finite rainfall array")
    return daily.values.reshape(4, 7, len(latitude), len(longitude)).mean(axis=1)


def weekly_imerg_climatology(
    climatology: xr.Dataset, review: xr.Dataset, case_id: str
) -> np.ndarray:
    weeks = []
    available = set(climatology.calendar_month_day.values.astype(str))
    for week in range(1, 5):
        start = pd.Timestamp(review.week_start.sel(case=case_id, week=week).item())
        end = pd.Timestamp(review.week_end_exclusive.sel(case=case_id, week=week).item())
        dates = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
        month_days = [date.strftime("%m-%d") for date in dates]
        missing = sorted(set(month_days) - available)
        if missing:
            raise ValueError(f"{case_id}/W{week}: climatology lacks {missing}")
        field = climatology.daily_precipitation_climatology.sel(
            calendar_day=climatology.calendar_day.where(
                climatology.calendar_month_day.isin(month_days), drop=True
            )
        )
        if field.sizes.get("calendar_day") != 7:
            raise ValueError(f"{case_id}/W{week}: expected seven climatology days")
        weeks.append(field.mean("calendar_day").values.astype(np.float64))
    return np.stack(weeks)


def weighted_acc(forecast: np.ndarray, observed: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(forecast) & np.isfinite(observed) & np.isfinite(weight) & (weight > 0)
    if int(valid.sum()) < 3:
        return float("nan")
    values_f = forecast[valid]
    values_o = observed[valid]
    values_w = weight[valid]
    values_w = values_w / values_w.sum()
    centered_f = values_f - np.sum(values_w * values_f)
    centered_o = values_o - np.sum(values_w * values_o)
    denominator = np.sqrt(
        np.sum(values_w * centered_f**2) * np.sum(values_w * centered_o**2)
    )
    return float(np.sum(values_w * centered_f * centered_o) / denominator)


def boundary_segments() -> tuple[list[np.ndarray], object]:
    components = (INDIA_SHAPEFILE, INDIA_SHAPEFILE.with_suffix(".shx"), INDIA_SHAPEFILE.with_suffix(".dbf"))
    if any(not path.is_file() for path in components):
        raise FileNotFoundError("India boundary shapefile is incomplete")
    geometries = [shape(item.__geo_interface__) for item in shapefile.Reader(str(INDIA_SHAPEFILE)).shapes()]
    source_crs = CRS.from_wkt(INDIA_SHAPEFILE.with_suffix(".prj").read_text(encoding="utf-8"))
    target_crs = CRS.from_epsg(4326)
    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geometries = [transform(transformer.transform, geometry) for geometry in geometries]
    outline = unary_union(geometries).simplify(0.03, preserve_topology=True)
    segments: list[np.ndarray] = []

    def collect(geometry: object) -> None:
        if geometry.geom_type in {"LineString", "LinearRing"}:
            segments.append(np.asarray(geometry.coords, dtype=float))
        elif hasattr(geometry, "geoms"):
            for child in geometry.geoms:
                collect(child)

    for geometry in geometries:
        collect(geometry.simplify(0.04, preserve_topology=True).boundary)
    if not segments:
        raise ValueError("India boundary yielded no line segments")
    return segments, outline


def plot_composite(
    composites: dict[str, np.ndarray],
    mean_acc: dict[tuple[str, int], float],
    latitude: np.ndarray,
    longitude: np.ndarray,
    mask: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    cmap = ListedColormap(COLORS, name="meeting_imerg_anomaly")
    cmap.set_under("#8F140C")
    cmap.set_over("#21185E")
    norm = BoundaryNorm(LEVELS, cmap.N)
    segments, outline = boundary_segments()
    west, south, east, north = outline.bounds
    figure, axes = plt.subplots(
        2, 5, figsize=(17.4, 7.65), sharex=True, sharey=True, facecolor="white"
    )
    image = None
    for row, week in enumerate(WEEKS):
        for column, system in enumerate(SYSTEMS):
            axis = axes[row, column]
            field = np.where(mask, composites[system][week - 1], np.nan)
            image = axis.contourf(
                longitude,
                latitude,
                field,
                levels=LEVELS,
                cmap=cmap,
                norm=norm,
                extend="both",
                antialiased=False,
            )
            axis.add_collection(
                LineCollection(segments, colors="#34454E", linewidths=0.35, zorder=5)
            )
            axis.set_xlim(max(67.0, west - 0.5), min(99.0, east + 0.5))
            axis.set_ylim(max(7.0, south - 0.3), min(38.5, north + 0.5))
            axis.set_aspect("equal")
            axis.set_facecolor("#FAFBFC")
            axis.grid(color="#D7E0E5", linewidth=0.38, linestyle=":", alpha=0.75)
            axis.tick_params(labelsize=8, colors="#536773", length=2.5)
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}°E"))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}°N"))
            if row == 0:
                axis.set_title(LABELS[system], fontsize=9.4, weight="bold", color="#172A36", pad=7)
            if column == 0:
                axis.set_ylabel(f"Week {week}\nLatitude", fontsize=10.2, weight="bold")
            if row == 1:
                axis.set_xlabel("Longitude", fontsize=8.6)
            if system != "observed":
                axis.text(
                    0.04,
                    0.045,
                    f"mean ACC {mean_acc[(system, week)]:.2f}",
                    transform=axis.transAxes,
                    fontsize=7.6,
                    weight="bold",
                    color="#172A36",
                    bbox={
                        "boxstyle": "round,pad=0.22",
                        "facecolor": "white",
                        "edgecolor": "#9AABB5",
                        "linewidth": 0.55,
                        "alpha": 0.93,
                    },
                    zorder=7,
                )
    if image is None:
        raise RuntimeError("no composite map was created")
    figure.suptitle(
        "Matched operational initialization comparison against IMERG",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=20,
        weight="bold",
        color="#172A36",
    )
    figure.text(
        0.055,
        0.936,
        "Five JJAS 2024 cases · identical Thursday–Wednesday valid weeks · all anomalies use IMERG Final V07B 2001–2022 climatology",
        fontsize=9.7,
        color="#536773",
    )
    figure.text(
        0.975,
        0.952,
        "5 MATCHED DATES · EXPLORATORY PILOT",
        ha="right",
        va="center",
        fontsize=8.5,
        weight="bold",
        color="#8C2D2D",
        bbox={
            "boxstyle": "round,pad=0.34",
            "facecolor": "#FFF3F1",
            "edgecolor": "#D9A29C",
        },
    )
    colorbar_axis = figure.add_axes([0.19, 0.075, 0.62, 0.029])
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal", extend="both")
    colorbar.set_ticks(LEVELS)
    colorbar.set_label("Five-case mean rainfall anomaly (mm/day): drier  ←  0  →  wetter", fontsize=9.2)
    colorbar.ax.tick_params(labelsize=8)
    figure.text(
        0.055,
        0.022,
        "FuXi IC: preceding Monday; ERPAS IC: Wednesday. Panel badges are mean case-wise spatial ACC against IMERG on the common 169-cell India support.",
        fontsize=7.8,
        color="#536773",
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.17, wspace=0.08, hspace=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("members_per_source") != 5 or config.get("sources") != ["era5", "gfs", "ifs"]:
        raise ValueError("operational-IC configuration contract changed")
    if len(config.get("cases", [])) != 5:
        raise ValueError("expected exactly five configured cases")

    with xr.open_dataset(REVIEW_FIELDS) as source:
        review = source.load()
    with xr.open_dataset(IMERG_CLIMATOLOGY) as source:
        climatology = source.load()
    if climatology.attrs.get("baseline_years") != "2001-2022":
        raise ValueError("IMERG climatology baseline is not 2001-2022")
    if climatology.attrs.get("verification_years_excluded") != "2023-2024":
        raise ValueError("IMERG climatology does not exclude verification years")
    if not (
        np.array_equal(review.latitude.values, climatology.latitude.values)
        and np.array_equal(review.longitude.values, climatology.longitude.values)
    ):
        raise ValueError("review fields and IMERG climatology grids differ")

    latitude = review.latitude.values.astype(float)
    longitude = review.longitude.values.astype(float)
    mask = review.india_fraction.values > 0
    weight = review.spatial_weight.values.astype(float)
    if int(mask.sum()) != 169 or not np.array_equal(mask, weight > 0):
        raise ValueError("expected fixed 169-cell India support")

    sums = {
        system: np.zeros((4, len(latitude), len(longitude)), dtype=np.float64)
        for system in SYSTEMS
    }
    rows: list[dict] = []
    input_files = {"review_fields": REVIEW_FIELDS, "imerg_climatology": IMERG_CLIMATOLOGY}
    case_ids: list[str] = []
    for case_number, case in enumerate(config["cases"], start=1):
        issue = pd.Timestamp(case["issue_date"])
        erpas_date = issue + pd.Timedelta(days=2)
        case_id = f"paired_{erpas_date:%Y%m%d}"
        if case_id not in set(review.case.values.astype(str)):
            raise ValueError(f"{case_id} is absent from audited review fields")
        case_ids.append(case_id)
        clim = weekly_imerg_climatology(climatology, review, case_id)
        observed_raw = review.observed_weekly_rainfall.sel(
            reference="IMERG Final V07B", case=case_id
        ).values.astype(np.float64)
        observed_anomaly = observed_raw - clim
        saved_observed_anomaly = review.observed_weekly_anomaly.sel(
            reference="IMERG Final V07B", case=case_id
        ).values.astype(np.float64)
        finite = np.isfinite(saved_observed_anomaly) & np.isfinite(observed_anomaly)
        if float(np.max(np.abs(saved_observed_anomaly[finite] - observed_anomaly[finite]))) > 2e-5:
            raise ValueError(f"{case_id}: failed IMERG anomaly reproduction")

        expected_starts, expected_ends = expected_periods(review, case_id)
        raw = {
            "erpas": review.forecast_weekly_rainfall.sel(model="ERPAS", case=case_id)
            .values.astype(np.float64)
        }
        for system in ("era5", "gfs", "ifs"):
            path = forecast_path(config, system, issue)
            input_files[f"{system}_{issue:%Y%m%d}"] = path
            raw[system] = load_fuxi_weekly(
                path, system, issue, latitude, longitude, expected_starts, expected_ends
            )
        sums["observed"] += np.where(np.isfinite(observed_anomaly), observed_anomaly, 0.0)
        for system in SYSTEMS[1:]:
            anomaly = raw[system] - clim
            sums[system] += np.where(np.isfinite(anomaly), anomaly, 0.0)
            for week in range(1, 5):
                rows.append(
                    {
                        "case_id": case_id,
                        "fuxi_initialization": str(issue.date()),
                        "erpas_initialization": str(erpas_date.date()),
                        "week": week,
                        "system": system,
                        "acc_vs_imerg": weighted_acc(
                            anomaly[week - 1], observed_anomaly[week - 1], weight
                        ),
                    }
                )
        print(f"processed {case_number}/5: {case_id}", flush=True)

    composites = {system: values / 5.0 for system, values in sums.items()}
    metrics = pd.DataFrame(rows).sort_values(["case_id", "system", "week"])
    mean_acc = {
        (system, int(week)): float(group.acc_vs_imerg.mean())
        for (system, week), group in metrics.groupby(["system", "week"])
    }
    checks = {
        "exactly_five_matched_cases": len(case_ids) == 5 and len(set(case_ids)) == 5,
        "five_members_for_each_fuxi_source": True,
        "identical_valid_periods_checked": True,
        "imerg_2001_2022_climatology": climatology.attrs.get("baseline_years") == "2001-2022",
        "verification_years_excluded": climatology.attrs.get("verification_years_excluded") == "2023-2024",
        "fixed_169_cell_support": int(mask.sum()) == 169,
        "all_acc_finite_and_bounded": bool(
            np.isfinite(metrics.acc_vs_imerg).all() and metrics.acc_vs_imerg.between(-1, 1).all()
        ),
        "all_composites_finite_on_support": all(
            np.isfinite(values[:, mask]).all() for values in composites.values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"output contract failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "01_imerg_operational_ic_composite_anomalies_w1_w3.png"
    metrics_path = args.output_dir / "per_case_acc_against_imerg.csv"
    metrics.to_csv(metrics_path, index=False)
    plot_composite(composites, mean_acc, latitude, longitude, mask, figure_path, args.dpi)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASSED",
        "scope": "Five matched ERPAS/FuXi ERA5/GFS/native-operational-IFS cases verified against IMERG Final V07B",
        "case_ids": case_ids,
        "members_per_fuxi_source": 5,
        "valid_week_alignment": "FuXi Monday leads 4-31 and ERPAS Wednesday Weeks 1-4 share Thursday-Wednesday valid periods",
        "anomaly_contract": "Every field is raw weekly-mean rainfall minus the same fixed IMERG Final V07B 2001-2022 calendar-day climatology",
        "displayed_weeks": list(WEEKS),
        "common_color_levels_mm_day": LEVELS.tolist(),
        "mean_case_acc_against_imerg": {
            system: {f"W{week}": mean_acc[(system, week)] for week in range(1, 5)}
            for system in SYSTEMS[1:]
        },
        "checks": checks,
        "inputs": [
            {"name": name, "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in input_files.items()
        ],
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in (figure_path, metrics_path)
        ],
        "interpretation": "Exploratory five-case pilot only; panel ACC badges are descriptive mean case-wise scores and do not establish a general IC ranking.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASSED", "figure": str(figure_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
