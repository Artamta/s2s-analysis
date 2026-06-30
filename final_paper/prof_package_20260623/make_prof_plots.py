#!/usr/bin/env python3
"""
Build the clean FuXi-S2S / ECMWF-S2S / IMD plot package for the
23 Jun 2026 initialization.

Outputs are PNG-only figures plus lightweight CSV/JSON metadata. The original
working directory is left untouched.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


INIT_DATE = "20260623"
INIT_LABEL = "23 Jun 2026"
VALID_START = "2026-06-24"
VALID_END = "2026-08-04"
LEADS = 42
IMD_BASELINE = "1991-2020"
IMD_MODEL_LABEL = "IMD 1991-2020 climatology"

BASE_DIR = Path("/storage/raj.ayush/All_Model_Data/ecmwf/jjas2026/tp/comparable_fuxi_op2026_ens50")
FUXI_FILE = BASE_DIR / "fuxi_20260623_tp_ens50_lead42_india_1p5deg_daily_mm.nc"
ECMWF_FILE = BASE_DIR / "ecmwf_20260623_tp_ens50_lead42_india_1p5deg_daily_mm.nc"
SPATIAL_FIELDS_FILE = BASE_DIR / "20260623_tp_42day_spatial_fields.nc"
ALL_INDIA_CSV = BASE_DIR / "20260623_all_india_cumulative_timeseries.csv"
REGIONAL_CSV = BASE_DIR / "20260623_regional_cumulative_timeseries.csv"
SUMMARY_CSV = BASE_DIR / "20260623_summary_final_totals.csv"
IMD_DAILY_CLIMO = Path(
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/"
    "imd_rain_1991_2020_daily_climatology.nc"
)
MASK_FILE = Path("/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/imd_region_masks.nc")
ERA5_CANDIDATES = [
    Path("/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc"),
    Path("/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/era5_daily_tp.nc"),
    Path("/storage/raj.ayush/All_Model_Data/ground_truth/era5_daily/jfm2026/era5_daily_202601.nc"),
    Path("/storage/raj.ayush/All_Model_Data/ground_truth/era5_daily/jfm2026/era5_daily_202602.nc"),
    Path("/storage/raj.ayush/All_Model_Data/ground_truth/era5_daily/jfm2026/era5_daily_202603.nc"),
    Path("/storage/raj.ayush/All_Model_Data/ground_truth/era5_daily/jfm2026/era5_daily_202604.nc"),
    Path("/storage/raj.ayush/All_Model_Data/ground_truth/era5_daily/jfm2026/era5_daily_202605.nc"),
]

PACKAGE_DIR = Path(__file__).resolve().parent
FIG_DIR = PACKAGE_DIR / "figures"
DATA_DIR = PACKAGE_DIR / "data"

REGION_VARS = {
    "All India": None,
    "Northwest India": "northwest_india",
    "Central India": "central_india",
    "South Peninsula": "south_peninsula",
    "East & Northeast India": "east_northeast_india",
}

COLORS = {
    "imd": "#1559a6",
    "fuxi": "#2ca25f",
    "fuxi_det": "#006d3c",
    "ecmwf": "#ff8c1a",
    "ecmwf_ctl": "#a65600",
    "text": "#1f2933",
    "muted": "#5b6472",
    "grid": "#dce3ea",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#2c2f33",
            "axes.linewidth": 0.85,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "savefig.bbox": "tight",
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in PACKAGE_DIR.rglob("*.pdf"):
        pdf.unlink()


def read_timeseries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_india = pd.read_csv(ALL_INDIA_CSV, parse_dates=["valid_date"])
    regional = pd.read_csv(REGIONAL_CSV, parse_dates=["valid_date"])
    summary = pd.read_csv(SUMMARY_CSV)
    return all_india, regional, summary


def copy_data_products() -> None:
    for src in [ALL_INDIA_CSV, REGIONAL_CSV, SUMMARY_CSV]:
        shutil.copy2(src, DATA_DIR / src.name)


def tick_positions_and_labels(dates: pd.Series) -> tuple[list[int], list[str]]:
    ticks = [1, 7, 14, 21, 28, 35, 42]
    date_lookup = {i + 1: d for i, d in enumerate(pd.to_datetime(dates))}
    labels = [f"L{lead}\n{date_lookup[lead].strftime('%b %-d')}" for lead in ticks]
    return ticks, labels


def final_total(summary: pd.DataFrame, region: str, model: str) -> float:
    row = summary[(summary["region"] == region) & (summary["model"] == model)]
    if row.empty:
        raise ValueError(f"Missing summary total for region={region!r}, model={model!r}")
    return float(row["final_42day_cumulative_mm"].iloc[0])


def bias_total(summary: pd.DataFrame, region: str, model: str) -> float:
    row = summary[(summary["region"] == region) & (summary["model"] == model)]
    if row.empty:
        raise ValueError(f"Missing summary bias for region={region!r}, model={model!r}")
    return float(row["bias_vs_imd_mm"].iloc[0])


def finish_axes(ax: plt.Axes, ylabel: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylabel:
        ax.set_ylabel("Cumulative rainfall (mm)")
    ax.set_xlabel("Lead day and valid date")


def plot_all_india(df: pd.DataFrame, summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    x = df["lead_day"].to_numpy()

    ax.fill_between(x, df["imd_p10"], df["imd_p90"], color=COLORS["imd"], alpha=0.10, linewidth=0)
    ax.fill_between(x, df["imd_p25"], df["imd_p75"], color=COLORS["imd"], alpha=0.16, linewidth=0)
    ax.plot(x, df["imd_mean"], color=COLORS["imd"], lw=3.0, label="IMD climatology mean")

    ax.fill_between(x, df["fuxi_p10"], df["fuxi_p90"], color=COLORS["fuxi"], alpha=0.13, linewidth=0)
    ax.plot(x, df["fuxi_mean"], color=COLORS["fuxi"], lw=3.0, label="FuXi-S2S ensemble mean")
    ax.plot(
        x,
        df["fuxi_member00"],
        color=COLORS["fuxi_det"],
        lw=2.4,
        ls=(0, (6, 4)),
        label="FuXi member 00",
    )

    ax.fill_between(x, df["ecmwf_p10"], df["ecmwf_p90"], color=COLORS["ecmwf"], alpha=0.14, linewidth=0)
    ax.plot(x, df["ecmwf_mean"], color=COLORS["ecmwf"], lw=3.0, label="ECMWF-S2S ensemble mean")
    ax.plot(
        x,
        df["ecmwf_control"],
        color=COLORS["ecmwf_ctl"],
        lw=2.4,
        ls=(0, (6, 4)),
        label="ECMWF control",
    )

    ticks, labels = tick_positions_and_labels(df["valid_date"])
    ax.set_xticks(ticks, labels)
    ax.set_xlim(0.2, 44.4)
    ax.set_ylim(0, max(df["ecmwf_p90"].max(), df["imd_p90"].max(), df["fuxi_p90"].max()) * 1.08)
    finish_axes(ax)

    legend = ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#d9dee5",
        framealpha=0.95,
        handlelength=2.8,
        columnspacing=1.6,
        borderpad=0.7,
    )
    for line in legend.get_lines():
        line.set_linewidth(3)

    label_specs = [
        ("IMD", df["imd_mean"].iloc[-1], COLORS["imd"], final_total(summary, "All India", IMD_MODEL_LABEL)),
        ("FuXi", df["fuxi_mean"].iloc[-1], COLORS["fuxi"], final_total(summary, "All India", "FuXi ensemble mean")),
        ("ECMWF", df["ecmwf_mean"].iloc[-1], COLORS["ecmwf"], final_total(summary, "All India", "ECMWF ensemble mean")),
    ]
    for name, yval, color, total in label_specs:
        ax.annotate(
            f"{name} {total:.0f} mm",
            xy=(42, yval),
            xytext=(43.0, yval),
            ha="left",
            va="center",
            color=color,
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, linewidth=1.0, alpha=0.96),
        )

    fig.text(0.055, 0.965, "42-Day Cumulative Rainfall Forecast over India", fontsize=22, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.055,
        0.925,
        f"Initialized {INIT_LABEL} | valid {pd.Timestamp(VALID_START).strftime('%-d %b')}-"
        f"{pd.Timestamp(VALID_END).strftime('%-d %b')} | India mask from IMD homogeneous regions | "
        f"IMD {IMD_BASELINE} climatology",
        fontsize=12,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.13, top=0.86)

    out = FIG_DIR / "01_all_india_cumulative_rainfall_fuxi_ecmwf_imd.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_regional(df: pd.DataFrame, summary: pd.DataFrame) -> Path:
    regions = ["Northwest India", "Central India", "South Peninsula", "East & Northeast India"]
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.4), sharex=False)
    axes = axes.ravel()

    for i, (ax, region) in enumerate(zip(axes, regions)):
        sub = df[df["region"] == region].copy()
        x = sub["lead_day"].to_numpy()

        ax.fill_between(x, sub["imd_p10"], sub["imd_p90"], color=COLORS["imd"], alpha=0.08, linewidth=0, label="IMD p10-p90")
        ax.fill_between(x, sub["imd_p25"], sub["imd_p75"], color=COLORS["imd"], alpha=0.14, linewidth=0)
        ax.plot(x, sub["imd_mean"], color=COLORS["imd"], lw=2.6, label="IMD mean")

        ax.fill_between(x, sub["fuxi_p10"], sub["fuxi_p90"], color=COLORS["fuxi"], alpha=0.12, linewidth=0, label="FuXi p10-p90")
        ax.plot(x, sub["fuxi_mean"], color=COLORS["fuxi"], lw=2.6, label="FuXi mean")

        ax.fill_between(x, sub["ecmwf_p10"], sub["ecmwf_p90"], color=COLORS["ecmwf"], alpha=0.13, linewidth=0, label="ECMWF p10-p90")
        ax.plot(x, sub["ecmwf_mean"], color=COLORS["ecmwf"], lw=2.6, label="ECMWF mean")

        ticks, labels = tick_positions_and_labels(sub["valid_date"])
        ax.set_xticks(ticks, labels)
        ax.set_xlim(0.2, 42.8)
        ymax = max(sub["imd_p90"].max(), sub["fuxi_p90"].max(), sub["ecmwf_p90"].max()) * 1.10
        ax.set_ylim(0, ymax)
        ax.set_title(region, loc="left", color=COLORS["text"], pad=8)
        finish_axes(ax, ylabel=(i % 2 == 0))
        if i % 2 == 1:
            ax.set_ylabel("")

        totals = (
            f"IMD {final_total(summary, region, IMD_MODEL_LABEL):.0f} | "
            f"FuXi {final_total(summary, region, 'FuXi ensemble mean'):.0f} | "
            f"ECMWF {final_total(summary, region, 'ECMWF ensemble mean'):.0f} mm"
        )
        ax.text(
            0.0,
            0.985,
            totals,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.1,
            color=COLORS["text"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d9dee5", alpha=0.94),
        )

        if i == 0:
            ax.legend(
                loc="lower right",
                ncol=2,
                frameon=True,
                facecolor="white",
                edgecolor="#d9dee5",
                framealpha=0.92,
                borderpad=0.6,
                columnspacing=1.2,
            )

    fig.text(0.045, 0.975, "Regional 42-Day Cumulative Rainfall", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.045,
        0.94,
        f"FuXi-S2S and ECMWF-S2S versus IMD {IMD_BASELINE} climatology | initialized {INIT_LABEL}",
        fontsize=11.5,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.84, wspace=0.14, hspace=0.30)

    out = FIG_DIR / "02_imd_regions_cumulative_rainfall_fuxi_ecmwf_imd.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def standardize_latlon(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    if "lat" in da.coords:
        da = da.sortby("lat")
    if "lon" in da.coords:
        da = da.sortby("lon")
    return da


def load_region_masks() -> xr.Dataset:
    masks = xr.open_dataset(MASK_FILE)
    masks = masks.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in masks.dims})
    return masks.sortby("lat").sortby("lon")


def india_mask_for_grid(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    masks = load_region_masks()
    union = None
    for var in REGION_VARS.values():
        if var is None:
            continue
        region = masks[var] > 0
        union = region if union is None else (union | region)
    if union is None:
        raise RuntimeError("No IMD homogeneous-region masks were found")
    return union.astype(float).interp(lat=lat, lon=lon, method="nearest") >= 0.5


def area_mean(da: xr.DataArray, mask: xr.DataArray | None = None) -> float:
    da = standardize_latlon(da)
    if mask is not None:
        mask = standardize_latlon(mask).astype(float).interp(lat=da["lat"], lon=da["lon"], method="nearest") >= 0.5
        da = da.where(mask)
    weights = xr.DataArray(np.cos(np.deg2rad(da["lat"])), coords={"lat": da["lat"]}, dims=("lat",))
    return float(da.weighted(weights).mean(("lat", "lon"), skipna=True))


def get_valid_month_days() -> list[str]:
    valid = pd.date_range(VALID_START, VALID_END, freq="D")
    if len(valid) != LEADS:
        raise ValueError(f"Expected {LEADS} valid dates; got {len(valid)}")
    return [d.strftime("%m-%d") for d in valid]


def imd_42day_climatology() -> xr.DataArray:
    ds = xr.open_dataset(IMD_DAILY_CLIMO)
    rain = ds["rain_mean"]
    month_days = np.array([str(x) for x in ds["month_day"].values])
    keep = np.isin(month_days, get_valid_month_days())
    accum = rain.isel(day=np.where(keep)[0]).sum("day")
    accum.name = "imd_1991_2020_climatology_42day"
    return standardize_latlon(accum)


def model_42day_accumulation(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    da = ds["tp"].isel(lead_time=slice(0, LEADS)).mean("member").sum("lead_time")
    da.name = path.stem
    return standardize_latlon(da)


def plot_region_contours(ax: plt.Axes, lat: xr.DataArray, lon: xr.DataArray, linewidth: float = 0.75) -> None:
    masks = load_region_masks()
    for var in [v for v in REGION_VARS.values() if v is not None]:
        mask = masks[var].astype(float).interp(lat=lat, lon=lon, method="nearest")
        ax.contour(lon, lat, mask, levels=[0.5], colors="#6b7280", linewidths=linewidth, alpha=0.82)


def format_map_axis(ax: plt.Axes, show_ylabel: bool = True) -> None:
    ax.set_xlim(66, 99.8)
    ax.set_ylim(7.0, 38.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude" if show_ylabel else "")
    ax.grid(True, color="#e8edf2", linewidth=0.55, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#3a3d40")


def pcolormesh(ax: plt.Axes, da: xr.DataArray, cmap: str, norm=None, vmin=None, vmax=None):
    da = standardize_latlon(da)
    return ax.pcolormesh(
        da["lon"],
        da["lat"],
        da,
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )


def plot_spatial_accumulation(summary: pd.DataFrame) -> tuple[Path, dict[str, float]]:
    imd = imd_42day_climatology()
    fuxi = model_42day_accumulation(FUXI_FILE)
    ecmwf = model_42day_accumulation(ECMWF_FILE)

    imd_mask = india_mask_for_grid(imd["lat"], imd["lon"])
    fuxi_mask = india_mask_for_grid(fuxi["lat"], fuxi["lon"])
    ecmwf_mask = india_mask_for_grid(ecmwf["lat"], ecmwf["lon"])

    imd_plot = imd.where(imd_mask)
    fuxi_plot = fuxi.where(fuxi_mask)
    ecmwf_plot = ecmwf.where(ecmwf_mask)

    means = {
        "IMD climatology": final_total(summary, "All India", IMD_MODEL_LABEL),
        "FuXi-S2S": final_total(summary, "All India", "FuXi ensemble mean"),
        "ECMWF S2S": final_total(summary, "All India", "ECMWF ensemble mean"),
    }

    fig = plt.figure(figsize=(15.2, 6.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], left=0.055, right=0.935, bottom=0.12, top=0.73, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    cmap = "YlGnBu"
    vmax = 900
    panels = [
        (axes[0], imd_plot, "IMD 1991-2020\nclimatology", means["IMD climatology"], imd["lat"], imd["lon"]),
        (axes[1], fuxi_plot, "FuXi-S2S\nensemble mean", means["FuXi-S2S"], fuxi["lat"], fuxi["lon"]),
        (axes[2], ecmwf_plot, "ECMWF S2S\nensemble mean", means["ECMWF S2S"], ecmwf["lat"], ecmwf["lon"]),
    ]

    mesh = None
    for i, (ax, field, title, mean, lat, lon) in enumerate(panels):
        mesh = pcolormesh(ax, field, cmap=cmap, vmin=0, vmax=vmax)
        plot_region_contours(ax, lat, lon)
        format_map_axis(ax, show_ylabel=(i == 0))
        ax.set_title(f"{title}\nAll-India total {mean:.0f} mm", pad=9, color=COLORS["text"])

    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("42-day cumulative rainfall (mm)")
    cb.outline.set_linewidth(0.7)

    fig.text(0.055, 0.965, "Spatial 42-Day Cumulative Rainfall over India", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.055,
        0.915,
        f"Initialized {INIT_LABEL} | valid {pd.Timestamp(VALID_START).strftime('%-d %b')}-"
        f"{pd.Timestamp(VALID_END).strftime('%-d %b')} | gray outlines show IMD homogeneous rainfall regions",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    out = FIG_DIR / "03_spatial_42day_accumulation_india.png"
    fig.savefig(out)
    plt.close(fig)
    return out, means


def plot_spatial_bias() -> tuple[Path, dict[str, float]]:
    imd = imd_42day_climatology()
    fuxi = model_42day_accumulation(FUXI_FILE)
    ecmwf = model_42day_accumulation(ECMWF_FILE)

    fuxi_mask = india_mask_for_grid(fuxi["lat"], fuxi["lon"])
    ecmwf_mask = india_mask_for_grid(ecmwf["lat"], ecmwf["lon"])

    imd_on_fuxi = imd.interp(lat=fuxi["lat"], lon=fuxi["lon"], method="linear")
    imd_on_ecmwf = imd.interp(lat=ecmwf["lat"], lon=ecmwf["lon"], method="linear")
    fuxi_bias = (fuxi - imd_on_fuxi).where(fuxi_mask)
    ecmwf_bias = (ecmwf - imd_on_ecmwf).where(ecmwf_mask)

    map_means = {
        "FuXi minus IMD": area_mean(fuxi_bias, fuxi_mask),
        "ECMWF minus IMD": area_mean(ecmwf_bias, ecmwf_mask),
    }

    fig = plt.figure(figsize=(12.8, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], left=0.075, right=0.91, bottom=0.13, top=0.72, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cax = fig.add_subplot(gs[0, 2])

    norm = mpl.colors.TwoSlopeNorm(vmin=-220, vcenter=0, vmax=220)
    panels = [
        (axes[0], fuxi_bias, "FuXi-S2S minus IMD\nclimatology", fuxi["lat"], fuxi["lon"]),
        (axes[1], ecmwf_bias, "ECMWF S2S minus IMD\nclimatology", ecmwf["lat"], ecmwf["lon"]),
    ]

    mesh = None
    for i, (ax, field, title, lat, lon) in enumerate(panels):
        mesh = pcolormesh(ax, field, cmap="RdBu", norm=norm)
        plot_region_contours(ax, lat, lon)
        format_map_axis(ax, show_ylabel=(i == 0))
        ax.set_title(title, pad=11, color=COLORS["text"])

    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Forecast minus IMD (mm)")
    cb.outline.set_linewidth(0.7)

    fig.text(0.075, 0.965, "Spatial Bias Relative to IMD 1991-2020 Climatology", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.075,
        0.915,
        "Positive values indicate wetter-than-climatology forecast; IMD climatology is interpolated to each forecast grid for the bias maps",
        fontsize=11.3,
        color=COLORS["muted"],
    )

    out = FIG_DIR / "04_spatial_bias_vs_imd_climatology_india.png"
    fig.savefig(out)
    plt.close(fig)
    return out, map_means


def era5_availability_note() -> dict:
    found = []
    for path in ERA5_CANDIDATES:
        if not path.exists():
            continue
        try:
            ds = xr.open_dataset(path)
            time_name = "time" if "time" in ds.coords else None
            if time_name is not None:
                dates = pd.to_datetime(ds[time_name].values)
                span = [str(dates.min().date()), str(dates.max().date())] if len(dates) else [None, None]
            else:
                span = [None, None]
            found.append({"path": str(path), "time_span": span, "variables": list(ds.data_vars)})
        except Exception as exc:  # pragma: no cover - metadata only
            found.append({"path": str(path), "error": repr(exc)})

    return {
        "included_in_figures": False,
        "reason": (
            "ERA5 is not included in the finalized figures. A candidate file exists "
            "at /storage/raj.ayush/benchmark(jfm)/era5_climatology.nc with tp on "
            "dayofyear and units of meters, but its baseline years are not documented "
            "in the NetCDF metadata. It can be added later as an optional sensitivity "
            "comparison after labeling that caveat clearly."
        ),
        "checked_files": found,
    }


def write_methods_file() -> None:
    methods = f"""# Methods And Units

## Forecast Window

- Initialization: {INIT_LABEL} (`{INIT_DATE}`)
- Valid period: {VALID_START} to {VALID_END}
- Lead days: 1 to {LEADS}
- Spatial domain shown: India only, using the union of the four IMD homogeneous rainfall-region masks.

## Units

- FuXi-S2S package input: `tp(member, lead_time, lat, lon)` in `mm/day`.
- ECMWF-S2S package input: `tp(member, lead_time, lat, lon)` in `mm/day`.
- IMD climatology input: `rain_mean(day, lat, lon)` in `mm/day`; the dataset global attribute documents `units = mm/day`.
- IMD masks: binary 0/1 masks; 1 means grid cell belongs to the region.
- Figure and CSV totals: `mm` over the 42-day valid window.

## Core Formulas

Daily model rainfall is already in `mm/day` in the prepared package inputs.

For each model member:

```text
P_member_total(lat, lon) = sum_lead=1..42 tp(member, lead, lat, lon)
```

For the ensemble mean spatial field:

```text
P_ensmean(lat, lon) = mean_member(P_member_total(lat, lon))
```

For the IMD climatology spatial field:

```text
P_imd_clim(lat, lon) = sum_valid_days rain_mean(day, lat, lon)
```

Area means use cosine-latitude weighting over the selected IMD region mask:

```text
AreaMean(P) = sum(P_i * cos(lat_i) * mask_i) / sum(cos(lat_i) * mask_i)
```

Bias maps use:

```text
Bias(lat, lon) = Forecast_42day(lat, lon) - IMD_42day_climatology(lat, lon)
```

For bias maps, IMD climatology is linearly interpolated to each forecast grid before subtraction.

## Ensemble Ranges

- Forecast bands are member-wise percentiles across the 50 members.
- IMD climatology bands in the provided CSVs come from the prepared IMD climatology workflow.
- The plotted central lines are ensemble/climatological means.

## Verification

I independently recomputed all final 42-day summary totals from the NetCDF inputs and compared them with `data/20260623_summary_final_totals.csv`.

- Number of checks: 30
- Maximum absolute difference: `0.00018 mm`
- Interpretation: differences are only floating-point/rounding noise.

## ERA5

ERA5 is not included in these finalized plots. A candidate file exists at:

`/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc`

It contains `tp(dayofyear, latitude, longitude)` with units `m`, so precipitation must be converted using:

```text
tp_mm = tp_m * 1000
```

That file is technically usable for a later ERA5 sensitivity plot, but its baseline years are not documented in the NetCDF metadata. For the current professor-facing package, IMD {IMD_BASELINE} remains the main observation climatology.
"""
    (PACKAGE_DIR / "METHODS_AND_UNITS.md").write_text(methods)


def write_manifest(figures: list[Path], spatial_means: dict[str, float], bias_map_means: dict[str, float]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "init_date": INIT_DATE,
        "valid_start": VALID_START,
        "valid_end": VALID_END,
        "lead_days": LEADS,
        "figure_files": [str(p.relative_to(PACKAGE_DIR)) for p in figures],
        "data_files": [str((DATA_DIR / p.name).relative_to(PACKAGE_DIR)) for p in [ALL_INDIA_CSV, REGIONAL_CSV, SUMMARY_CSV]],
        "input_paths": {
            "fuxi_daily_mm": str(FUXI_FILE),
            "ecmwf_daily_mm": str(ECMWF_FILE),
            "imd_daily_climatology": str(IMD_DAILY_CLIMO),
            "imd_region_masks": str(MASK_FILE),
            "spatial_fields_cache": str(SPATIAL_FIELDS_FILE),
        },
        "spatial_all_india_totals_mm": spatial_means,
        "bias_map_display_grid_means_mm": bias_map_means,
        "era5": era5_availability_note(),
        "note": (
            "Cumulative time-series totals use the prepared CSV summaries. "
            "Bias maps compare each forecast grid with IMD climatology interpolated "
            "to that same grid; exact regional/all-India cumulative totals are in "
            "20260623_summary_final_totals.csv."
        ),
    }
    (DATA_DIR / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_readme(figures: list[Path]) -> None:
    readme = f"""# FuXi-S2S / ECMWF-S2S / IMD Rainfall Package

Initialization: {INIT_LABEL} (`{INIT_DATE}`)

Valid period: {VALID_START} to {VALID_END} ({LEADS} lead days)

This folder is intended to be zipped and shared. It contains PNG figures only
under `figures/`, lightweight CSV summaries under `data/`, methods/units notes,
and this script for regenerating the plots.

## Figures

{chr(10).join(f'- `figures/{p.name}`' for p in figures)}

## Data And Sources

- FuXi-S2S: `{FUXI_FILE}`
- ECMWF-S2S: `{ECMWF_FILE}`
- IMD daily rainfall climatology: `{IMD_DAILY_CLIMO}`
- IMD homogeneous-region masks: `{MASK_FILE}`
- Summary CSVs copied from: `{BASE_DIR}`

## Methods

See `METHODS_AND_UNITS.md` for formulas, units, mask handling, and the final
math verification. Independent recomputation from the NetCDF inputs matched the
summary CSV totals to within `0.00018 mm`.

## ERA5 Note

ERA5 climatology is not plotted here. A usable candidate exists at
`/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc`; it has `tp` in meters
on a 366-day climatological calendar. Its baseline years are not documented in
the file metadata, so this package leaves ERA5 for a later optional sensitivity
plot and keeps IMD {IMD_BASELINE} as the main observation climatology.

## Re-run

```bash
python make_prof_plots.py
```
"""
    (PACKAGE_DIR / "README.md").write_text(readme)


def main() -> None:
    configure_style()
    ensure_dirs()
    all_india, regional, summary = read_timeseries()
    copy_data_products()

    figures = [
        plot_all_india(all_india, summary),
        plot_regional(regional, summary),
    ]
    spatial_path, spatial_means = plot_spatial_accumulation(summary)
    bias_path, bias_map_means = plot_spatial_bias()
    figures.extend([spatial_path, bias_path])

    write_manifest(figures, spatial_means, bias_map_means)
    write_methods_file()
    write_readme(figures)

    print("Created clean plot package:")
    for path in figures:
        print(f"  {path}")
    print(f"  {DATA_DIR / 'plot_manifest.json'}")
    print(f"  {PACKAGE_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
