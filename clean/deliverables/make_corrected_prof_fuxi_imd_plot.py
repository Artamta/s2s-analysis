#!/usr/bin/env python3
"""Create the corrected professor-facing 23 June FuXi/IMD rainfall figure.

This rebuild avoids the two incompatible products used in the earlier slide:
the FuXi forecast is the strict, training-consistent daily-mean-IC run, and the
IMD climatology is recomputed directly from the native 1991--2020 NetCDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/imd_story_plots"
FUXI_FILE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "fuxi_s2s_strict00z_case_20260623_ens50/forecasts/annual2026/20260623.nc"
)
IMD_CLIMO_FILE = Path(
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/"
    "imd_rain_1991_2020_daily_climatology.nc"
)
MASK_FILE = Path("/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/imd_region_masks.nc")
OBS_FILE = ROOT / "deliverables/final_story_plots/02_partial_verification_20260623_data.csv"
OLD_FILE = Path(
    "/home/raj.ayush/s2s/s2s_anlysis/final_paper/case-study/data/"
    "20260623_all_india_cumulative_timeseries.csv"
)
REGIONS = (
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)

COLORS = {
    "fuxi": "#138a45",
    "member": "#0f766e",
    "imd_climo": "#1559a6",
    "imd_obs": "#111827",
    "grid": "#dbe3ea",
    "muted": "#5f6b76",
}


def union_mask() -> xr.DataArray:
    with xr.open_dataset(MASK_FILE) as source:
        union = xr.zeros_like(source[REGIONS[0]], dtype=bool)
        for region in REGIONS:
            union = union | (source[region] > 0)
        return union.load().sortby("lat").sortby("lon")


def spatial_mean(field: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    lat_name = "latitude" if "latitude" in field.dims else "lat"
    lon_name = "longitude" if "longitude" in field.dims else "lon"
    local_mask = mask
    if lat_name != "lat" or lon_name != "lon":
        local_mask = local_mask.rename({"lat": lat_name, "lon": lon_name})
    local_mask = local_mask.astype(float).interp(
        {lat_name: field[lat_name], lon_name: field[lon_name]}, method="nearest"
    ) >= 0.5
    weights = xr.DataArray(
        np.cos(np.deg2rad(field[lat_name].values)),
        dims=lat_name,
        coords={lat_name: field[lat_name]},
    )
    return field.where(local_mask).weighted(weights).mean((lat_name, lon_name), skipna=True)


def load_fuxi(mask: xr.DataArray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    with xr.open_dataset(FUXI_FILE) as source:
        expected_attrs = {
            "input_daily_statistic": "daily_mean",
            "benchmark_mode": "strict_information_matched_00utc",
            "information_cutoff_matches_issue_time": "true",
        }
        for key, expected in expected_attrs.items():
            if str(source.attrs.get(key)) != expected:
                raise ValueError(f"FuXi {key}: expected {expected!r}; found {source.attrs.get(key)!r}")
        if source.tp.attrs.get("units") != "mm h-1":
            raise ValueError(f"unexpected FuXi TP units: {source.tp.attrs.get('units')!r}")
        daily_mm = source.tp * 24.0
        cumulative = spatial_mean(daily_mm, mask).cumsum("lead_day").load()
        dates = pd.DatetimeIndex(pd.to_datetime(source.valid_time.values))
    values = np.asarray(cumulative.values, dtype=np.float64)
    if values.shape != (50, 42) or not np.isfinite(values).all():
        raise ValueError(f"unexpected FuXi cumulative array {values.shape}")
    return dates, values


def load_imd_climatology(dates: pd.DatetimeIndex, mask: xr.DataArray) -> np.ndarray:
    month_days = [date.strftime("%m-%d") for date in dates]
    with xr.open_dataset(IMD_CLIMO_FILE) as source:
        if source.attrs.get("baseline") != "1991-2020":
            raise ValueError("unexpected IMD climatology baseline")
        lookup = {str(value): index for index, value in enumerate(source.month_day.values)}
        missing = sorted(set(month_days).difference(lookup))
        if missing:
            raise ValueError(f"IMD climatology is missing dates {missing}")
        daily = source.rain_mean.isel(day=[lookup[value] for value in month_days]).load()
    cumulative = spatial_mean(daily, mask).cumsum("day")
    values = np.asarray(cumulative.values, dtype=np.float64)
    if values.shape != (42,) or not np.isfinite(values).all():
        raise ValueError("invalid IMD climatology cumulative series")
    return values


def load_imd_observation(dates: pd.DatetimeIndex) -> np.ndarray:
    data = pd.read_csv(OBS_FILE, parse_dates=["valid_date"])
    data = data.set_index("valid_date").reindex(dates)
    values = data.imd_gauge_cumulative_mm.to_numpy(dtype=float)
    available = np.isfinite(values)
    if not available.any():
        raise ValueError("no IMD observation is available")
    last = int(np.flatnonzero(available)[-1])
    if not available[: last + 1].all() or available[last + 1 :].any():
        raise ValueError("IMD observation must be a continuous prefix")
    return values


def period_total(cumulative: np.ndarray, dates: pd.DatetimeIndex, start: str, end: str) -> float:
    selected = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("requested period is absent")
    first, last = int(indices[0]), int(indices[-1])
    previous = 0.0 if first == 0 else float(cumulative[first - 1])
    return float(cumulative[last] - previous)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mask = union_mask()
    dates, members = load_fuxi(mask)
    imd_climo = load_imd_climatology(dates, mask)
    imd_obs = load_imd_observation(dates)

    fuxi_mean = members.mean(axis=0)
    fuxi_p10 = np.quantile(members, 0.10, axis=0)
    fuxi_p90 = np.quantile(members, 0.90, axis=0)
    fuxi_member00 = members[0]

    week_start, week_end = "2026-07-01", "2026-07-07"
    week_climo = period_total(imd_climo, dates, week_start, week_end)
    week_fuxi = period_total(fuxi_mean, dates, week_start, week_end)
    week_member00 = period_total(fuxi_member00, dates, week_start, week_end)
    week_observed = period_total(imd_obs, dates, week_start, week_end)
    fuxi_anomaly = 100.0 * (week_fuxi / week_climo - 1.0)
    observed_anomaly = 100.0 * (week_observed / week_climo - 1.0)
    forecast_error = week_fuxi - week_observed
    forecast_error_percent = 100.0 * forecast_error / week_observed

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.edgecolor": "#263238",
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(15.5, 7.3), constrained_layout=True)
    layout = fig.add_gridspec(1, 2, width_ratios=[4.65, 1.45])
    ax = fig.add_subplot(layout[0, 0])
    note = fig.add_subplot(layout[0, 1])
    lead = np.arange(1, 43)

    # Lead 8--14 has period endpoints 1--7 July.
    ax.axvspan(7.5, 14.5, color="#fef3c7", alpha=0.56, zorder=0)
    ax.text(11, 12, "1–7 July", ha="center", color="#8a5a00", fontsize=9.5)
    last_observed = int(np.flatnonzero(np.isfinite(imd_obs))[-1]) + 1
    ax.axvspan(last_observed + 0.5, 42.5, color="#f1f5f9", alpha=0.72, zorder=0)

    ax.fill_between(lead, fuxi_p10, fuxi_p90, color=COLORS["fuxi"], alpha=0.15, linewidth=0)
    ax.plot(lead, imd_climo, color=COLORS["imd_climo"], lw=2.8, label="IMD 1991–2020 climatology")
    ax.plot(lead, fuxi_mean, color=COLORS["fuxi"], lw=3.2, label="FuXi-S2S ensemble mean")
    ax.plot(lead, fuxi_member00, color=COLORS["member"], lw=2.4, ls=(0, (6, 4)), label="FuXi member 00")
    ax.plot(lead, imd_obs, color=COLORS["imd_obs"], lw=3.1, label="IMD real-time gauge observation")
    ax.scatter([last_observed], [imd_obs[last_observed - 1]], color=COLORS["imd_obs"], s=40, zorder=5)
    ax.axvline(last_observed, color="#94a3b8", lw=1.1, ls=":")

    ticks = [1, 7, 14, 21, 28, 35, 42]
    labels = [f"L{value}\n{dates[value - 1]:%b %-d}" for value in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_xlim(0.3, 42.7)
    ax.set_ylim(0, max(fuxi_p90.max(), imd_climo.max()) * 1.10)
    ax.set_ylabel("All-India cumulative rainfall (mm)", fontsize=12)
    ax.set_xlabel("Lead day and 24-hour forecast-period endpoint", fontsize=11)
    ax.grid(True, color=COLORS["grid"], lw=0.8, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=9.7)
    ax.set_title("FuXi-S2S Captured the Wet First Week of July", loc="left", fontsize=17, weight="bold", pad=25)
    ax.text(
        0, 1.015,
        "Issued 23 June 2026 | valid 24 June–4 August | complete 21–22 June UTC daily-mean inputs",
        transform=ax.transAxes, color="#4b5563", fontsize=10.4,
    )

    note.axis("off")
    note.text(0, 0.96, "1–7 JULY: SAME IMD MASK", fontsize=11, weight="bold", color="#334155", va="top")
    note.text(
        0, 0.87,
        f"IMD climatology   {week_climo:.1f} mm\n"
        f"FuXi forecast       {week_fuxi:.1f} mm\n"
        f"IMD observed       {week_observed:.1f} mm",
        fontsize=11.4, linespacing=1.55, va="top",
    )
    note.text(0, 0.66, f"FuXi predicted: +{fuxi_anomaly:.0f}%", fontsize=13.2, weight="bold", color=COLORS["fuxi"], va="top")
    note.text(0, 0.59, f"IMD observed: +{observed_anomaly:.0f}%", fontsize=13.2, weight="bold", color=COLORS["imd_obs"], va="top")
    note.text(
        0, 0.49,
        f"Forecast − observation\n{forecast_error:+.1f} mm ({forecast_error_percent:+.0f}%)",
        fontsize=10.9, linespacing=1.4, va="top",
    )
    note.text(0, 0.36, "42-DAY FORECAST CONTEXT", fontsize=10.8, weight="bold", va="top")
    note.text(
        0, 0.30,
        f"FuXi mean            {fuxi_mean[-1]:.0f} mm\n"
        f"FuXi member 00  {fuxi_member00[-1]:.0f} mm\n"
        f"IMD climatology   {imd_climo[-1]:.0f} mm",
        fontsize=10.9, linespacing=1.45, va="top",
    )
    note.text(
        0, 0.14,
        "Message: FuXi predicted the correct\nwet-anomaly sign and close weekly magnitude.",
        fontsize=10.6, weight="bold", color="#334155", linespacing=1.35, va="top",
    )
    note.text(
        0, 0.015,
        "IMD climatology is context; the black line is verification.\n"
        "IMD gauge day is valid near 03 UTC; forecast endpoints are 00 UTC.\n"
        "One case does not establish general model skill.",
        fontsize=8.3, color=COLORS["muted"], va="bottom",
    )

    png = OUT / "06_corrected_prof_fuxi_imd_20260623.png"
    csv = OUT / "06_corrected_prof_fuxi_imd_20260623.csv"
    audit = OUT / "06_corrected_prof_fuxi_imd_20260623_audit.json"
    fig.savefig(png, dpi=240)
    plt.close(fig)

    pd.DataFrame(
        {
            "lead_day": lead,
            "valid_date": dates,
            "fuxi_mean": fuxi_mean,
            "fuxi_p10": fuxi_p10,
            "fuxi_p90": fuxi_p90,
            "fuxi_member00": fuxi_member00,
            "imd_1991_2020_climatology": imd_climo,
            "imd_gauge_observation": imd_obs,
        }
    ).to_csv(csv, index=False)

    old = pd.read_csv(OLD_FILE, parse_dates=["valid_date"])
    old_week_climo = period_total(old.imd_mean.to_numpy(float), pd.DatetimeIndex(old.valid_date), week_start, week_end)
    old_week_fuxi = period_total(old.fuxi_mean.to_numpy(float), pd.DatetimeIndex(old.valid_date), week_start, week_end)
    old_ratio = float(old.imd_mean.iloc[-1] / imd_climo[-1])
    report = {
        "corrected_plot": str(png),
        "forecast_contract": {
            "input_daily_statistic": "daily_mean",
            "input_days": ["2026-06-21", "2026-06-22"],
            "information_cutoff": "2026-06-23T00:00:00Z",
            "forecast_period_endpoints": ["2026-06-24", "2026-08-04"],
            "tp_conversion": "FuXi mm h-1 daily mean rate multiplied by 24, then cumulatively summed",
        },
        "imd_contract": {
            "source": str(IMD_CLIMO_FILE),
            "baseline": "1991-2020",
            "spatial_method": "cosine-latitude weighted union of four IMD homogeneous-region masks",
            "temporal_method": "daily mm/day summed over matching forecast-period endpoint dates",
        },
        "first_week_july_mm": {
            "imd_climatology": week_climo,
            "fuxi_ensemble_mean": week_fuxi,
            "fuxi_member00": week_member00,
            "imd_gauge_observation": week_observed,
            "fuxi_anomaly_percent": fuxi_anomaly,
            "imd_observed_anomaly_percent": observed_anomaly,
            "fuxi_minus_observed": forecast_error,
        },
        "full_42day_mm": {
            "imd_climatology": float(imd_climo[-1]),
            "fuxi_ensemble_mean": float(fuxi_mean[-1]),
            "fuxi_member00": float(fuxi_member00[-1]),
        },
        "old_slide_audit": {
            "fuxi_input": "00 UTC snapshot sensitivity experiment; not training-consistent daily means",
            "first_week_july_imd_climatology_mm": old_week_climo,
            "first_week_july_fuxi_mean_mm": old_week_fuxi,
            "claimed_fuxi_anomaly_percent": 100.0 * (old_week_fuxi / old_week_climo - 1.0),
            "full_42day_imd_climatology_mm": float(old.imd_mean.iloc[-1]),
            "old_to_direct_imd_climatology_ratio": old_ratio,
            "finding": "old IMD curve is the direct climatology multiplied by one fixed factor on all 42 days",
        },
        "news_comparison_warning": (
            "Do not numerically equate the article's 11% statistic with this plot unless its exact dates, "
            "domain, area weighting, and climatological normal are confirmed to match."
        ),
    }
    audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(png)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
