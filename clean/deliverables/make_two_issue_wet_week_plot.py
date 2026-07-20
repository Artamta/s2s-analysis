#!/usr/bin/env python3
"""Compare two forecasts of the observed 1--7 July 2026 wet spell.

The figure is intentionally event-specific.  It uses the same All-India IMD
mask, the native IMD 1991--2020 climatology, IMD gauge verification, and a
matched 50-member comparison for FuXi-S2S and ECMWF S2S.
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
MASK_FILE = Path("/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/imd_region_masks.nc")
IMD_CLIMO_FILE = Path(
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/"
    "imd_rain_1991_2020_daily_climatology.nc"
)
IMD_OBS_FILE = ROOT / "deliverables/final_story_plots/01_full_verification_20260601_data.csv"
FUXI_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi"
)
ECMWF_ROOT = Path("/storage/raj.ayush/All_Model_Data/ecmwf/jjas2026/tp")
REGIONS = (
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
EVENT_START = pd.Timestamp("2026-07-01")
EVENT_END = pd.Timestamp("2026-07-07")
N_MEMBERS = 50

COLORS = {
    "fuxi": "#138a45",
    "ecmwf": "#e67e00",
    "imd": "#111827",
    "climo": "#1559a6",
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


def fuxi_event_totals(init: str, mask: xr.DataArray) -> np.ndarray:
    path = (
        FUXI_ROOT
        / f"fuxi_s2s_strict00z_case_{init}_ens50"
        / "forecasts/annual2026"
        / f"{init}.nc"
    )
    with xr.open_dataset(path) as source:
        expected = {
            "input_daily_statistic": "daily_mean",
            "benchmark_mode": "strict_information_matched_00utc",
            "information_cutoff_matches_issue_time": "true",
        }
        for key, value in expected.items():
            if str(source.attrs.get(key)) != value:
                raise ValueError(f"FuXi {key}: expected {value!r}; found {source.attrs.get(key)!r}")
        if source.tp.attrs.get("units") != "mm h-1":
            raise ValueError(f"unexpected FuXi precipitation units: {source.tp.attrs.get('units')!r}")
        dates = pd.DatetimeIndex(pd.to_datetime(source.valid_time.values))
        selected = (dates >= EVENT_START) & (dates <= EVENT_END)
        daily_mm = spatial_mean(source.tp * 24.0, mask).load()
    totals = np.asarray(daily_mm.values[:, selected].sum(axis=1), dtype=float)
    if totals.shape != (N_MEMBERS,) or not np.isfinite(totals).all():
        raise ValueError(f"unexpected FuXi event-total array: {totals.shape}")
    return totals


def ecmwf_event_totals(init: str, mask: xr.DataArray) -> np.ndarray:
    path = ECMWF_ROOT / f"{init}_pf.nc"
    with xr.open_dataset(path) as source:
        # Match the FuXi 50-member ensemble and the prior case-study products.
        tp = source.tp.isel(number=slice(0, N_MEMBERS), step=slice(0, 42)).clip(min=0)
        dates = pd.DatetimeIndex(pd.to_datetime(source.valid_time.values[:42]))
        cumulative = spatial_mean(tp, mask).load()
    selected = np.flatnonzero((dates >= EVENT_START) & (dates <= EVENT_END))
    if len(selected) != 7:
        raise ValueError(f"expected seven ECMWF event days; found {len(selected)}")
    values = np.asarray(cumulative.values, dtype=float)
    previous = np.zeros(N_MEMBERS) if selected[0] == 0 else values[:, selected[0] - 1]
    totals = values[:, selected[-1]] - previous
    if totals.shape != (N_MEMBERS,) or not np.isfinite(totals).all():
        raise ValueError(f"unexpected ECMWF event-total array: {totals.shape}")
    return totals


def imd_references(mask: xr.DataArray) -> tuple[float, float]:
    event_dates = pd.date_range(EVENT_START, EVENT_END, freq="D")
    month_days = [value.strftime("%m-%d") for value in event_dates]
    with xr.open_dataset(IMD_CLIMO_FILE) as source:
        if source.attrs.get("baseline") != "1991-2020":
            raise ValueError("unexpected IMD climatology baseline")
        lookup = {str(value): index for index, value in enumerate(source.month_day.values)}
        daily = source.rain_mean.isel(day=[lookup[value] for value in month_days]).load()
    climatology = float(spatial_mean(daily, mask).sum("day").values)

    observed = pd.read_csv(IMD_OBS_FILE, parse_dates=["valid_date"])
    observed = observed.loc[
        observed.valid_date.between(EVENT_START, EVENT_END), "imd_gauge_mm"
    ]
    if len(observed) != 7 or not observed.notna().all():
        raise ValueError("IMD observation does not completely cover 1--7 July")
    return climatology, float(observed.sum())


def summarize(values: np.ndarray, climatology: float, observed: float) -> dict[str, float]:
    return {
        "mean_mm": float(values.mean()),
        "median_mm": float(np.median(values)),
        "p10_mm": float(np.quantile(values, 0.10)),
        "p25_mm": float(np.quantile(values, 0.25)),
        "p75_mm": float(np.quantile(values, 0.75)),
        "p90_mm": float(np.quantile(values, 0.90)),
        "probability_above_imd_climatology_percent": float(100.0 * np.mean(values > climatology)),
        "mean_minus_imd_observed_mm": float(values.mean() - observed),
        "absolute_mean_error_mm": float(abs(values.mean() - observed)),
    }


def draw_distribution(
    ax: plt.Axes,
    position: float,
    values: np.ndarray,
    color: str,
    rng: np.random.Generator,
) -> None:
    jitter = rng.uniform(-0.085, 0.085, size=len(values))
    ax.scatter(
        position + jitter,
        values,
        s=16,
        color=color,
        alpha=0.22,
        linewidth=0,
        zorder=2,
    )
    p10, p25, median, p75, p90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    ax.vlines(position, p10, p90, color=color, lw=2.2, zorder=3)
    ax.vlines(position, p25, p75, color=color, lw=9.0, alpha=0.82, zorder=3)
    ax.scatter(position, values.mean(), marker="D", s=68, color="white", edgecolor=color, lw=2.0, zorder=5)
    ax.hlines(median, position - 0.085, position + 0.085, color="white", lw=2.0, zorder=4)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mask = union_mask()
    climatology, observed = imd_references(mask)

    cases: dict[str, dict[str, np.ndarray]] = {}
    for init in ("20260601", "20260623"):
        cases[init] = {
            "FuXi-S2S": fuxi_event_totals(init, mask),
            "ECMWF S2S": ecmwf_event_totals(init, mask),
        }

    summaries = {
        init: {
            model: summarize(values, climatology, observed)
            for model, values in forecasts.items()
        }
        for init, forecasts in cases.items()
    }

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
    fig = plt.figure(figsize=(15.5, 7.5), constrained_layout=True)
    layout = fig.add_gridspec(1, 2, width_ratios=[4.4, 1.75])
    ax = fig.add_subplot(layout[0, 0])
    note = fig.add_subplot(layout[0, 1])

    ax.axhspan(climatology, 112, color="#ecfdf5", alpha=0.55, zorder=0)
    ax.axhline(observed, color=COLORS["imd"], lw=2.7, zorder=1)
    ax.axhline(climatology, color=COLORS["climo"], lw=2.2, ls=(0, (6, 4)), zorder=1)
    ax.text(2.50, observed + 1.1, f"IMD observed  {observed:.1f} mm", color=COLORS["imd"], ha="right", weight="bold")
    ax.text(2.50, climatology - 1.5, f"IMD 1991–2020 climatology  {climatology:.1f} mm", color=COLORS["climo"], ha="right", va="top")

    rng = np.random.default_rng(20260719)
    positions = {
        ("20260601", "FuXi-S2S"): 0.86,
        ("20260601", "ECMWF S2S"): 1.14,
        ("20260623", "FuXi-S2S"): 1.86,
        ("20260623", "ECMWF S2S"): 2.14,
    }
    model_colors = {"FuXi-S2S": COLORS["fuxi"], "ECMWF S2S": COLORS["ecmwf"]}
    for init, forecasts in cases.items():
        for model, values in forecasts.items():
            position = positions[(init, model)]
            color = model_colors[model]
            draw_distribution(ax, position, values, color, rng)
            ax.text(
                position,
                values.mean() + 3.0,
                f"{values.mean():.1f}",
                color=color,
                ha="center",
                weight="bold",
                fontsize=11.2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
            )

    ax.scatter([], [], marker="D", s=60, facecolor="white", edgecolor=COLORS["fuxi"], lw=2, label="FuXi-S2S mean")
    ax.scatter([], [], marker="D", s=60, facecolor="white", edgecolor=COLORS["ecmwf"], lw=2, label="ECMWF S2S mean")
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(25, 112)
    ax.set_xticks([1.0, 2.0], ["Issued 1 June\nlead days 30–36", "Issued 23 June\nlead days 8–14"])
    ax.set_ylabel("All-India rainfall during 1–7 July 2026 (mm)", fontsize=11.8)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=9.6)
    ax.set_title(
        "FuXi-S2S Predicted the Same Wet Week at Two Lead Times",
        loc="left",
        fontsize=17.0,
        weight="bold",
        pad=26,
    )
    ax.text(
        0,
        1.015,
        "1–7 July 2026 | daily-mean FuXi inputs | same All-India IMD mask | matched 50-member comparison",
        transform=ax.transAxes,
        color="#4b5563",
        fontsize=10.2,
    )
    ax.text(
        0.99,
        0.012,
        "Dots: members  •  thick bar: 25–75%  •  thin bar: 10–90%  •  diamond: ensemble mean",
        transform=ax.transAxes,
        ha="right",
        color=COLORS["muted"],
        fontsize=8.4,
    )

    note.axis("off")
    note.text(0, 0.96, "EVENT VERIFICATION", fontsize=11.2, weight="bold", color="#334155", va="top")
    note.text(
        0,
        0.885,
        f"IMD observed       {observed:.1f} mm\nIMD climatology    {climatology:.1f} mm\nObserved anomaly  +{100 * (observed / climatology - 1):.0f}%",
        fontsize=11.2,
        linespacing=1.48,
        va="top",
    )

    f1 = summaries["20260601"]["FuXi-S2S"]
    e1 = summaries["20260601"]["ECMWF S2S"]
    f23 = summaries["20260623"]["FuXi-S2S"]
    e23 = summaries["20260623"]["ECMWF S2S"]
    note.text(0, 0.69, "ENSEMBLE-MEAN ERROR", fontsize=10.8, weight="bold", va="top")
    note.text(0, 0.625, "1 June issue", fontsize=10.7, weight="bold", va="top")
    note.text(0.02, 0.575, f"FuXi       {f1['mean_minus_imd_observed_mm']:+.1f} mm", color=COLORS["fuxi"], fontsize=11.1, weight="bold", va="top")
    note.text(0.48, 0.575, f"ECMWF  {e1['mean_minus_imd_observed_mm']:+.1f} mm", color=COLORS["ecmwf"], fontsize=11.1, weight="bold", va="top")
    note.text(0, 0.515, "23 June issue", fontsize=10.7, weight="bold", va="top")
    note.text(0.02, 0.465, f"FuXi       {f23['mean_minus_imd_observed_mm']:+.1f} mm", color=COLORS["fuxi"], fontsize=11.1, weight="bold", va="top")
    note.text(0.48, 0.465, f"ECMWF  {e23['mean_minus_imd_observed_mm']:+.1f} mm", color=COLORS["ecmwf"], fontsize=11.1, weight="bold", va="top")

    note.text(0, 0.36, "PROBABILITY ABOVE IMD CLIMATOLOGY", fontsize=10.5, weight="bold", va="top")
    note.text(
        0,
        0.295,
        f"1 June       FuXi {f1['probability_above_imd_climatology_percent']:.0f}%   |   ECMWF {e1['probability_above_imd_climatology_percent']:.0f}%\n"
        f"23 June     FuXi {f23['probability_above_imd_climatology_percent']:.0f}%  |   ECMWF {e23['probability_above_imd_climatology_percent']:.0f}%",
        fontsize=10.7,
        linespacing=1.55,
        va="top",
    )
    note.text(
        0,
        0.17,
        "CLEAR MESSAGE",
        fontsize=10.8,
        weight="bold",
        color="#334155",
        va="top",
    )
    note.text(
        0,
        0.115,
        "FuXi consistently signaled an above-normal\nweek and was closer to IMD in both issues.",
        fontsize=10.8,
        weight="bold",
        color=COLORS["fuxi"],
        linespacing=1.35,
        va="top",
    )
    note.text(
        0,
        0.015,
        "Case-study evidence only: two forecasts verify the same event;\nthis is not an independent multi-case skill estimate.",
        fontsize=8.2,
        color=COLORS["muted"],
        va="bottom",
    )

    png = OUT / "08_two_issue_wet_week_ensemble_20260701_07.png"
    csv = OUT / "08_two_issue_wet_week_ensemble_20260701_07.csv"
    audit = OUT / "08_two_issue_wet_week_ensemble_20260701_07_audit.json"
    fig.savefig(png, dpi=240)
    plt.close(fig)

    rows = []
    for init, forecasts in cases.items():
        issue = pd.Timestamp(init).strftime("%Y-%m-%d")
        for model, values in forecasts.items():
            for member, value in enumerate(values, start=1):
                rows.append(
                    {
                        "issue_date": issue,
                        "model": model,
                        "member": member,
                        "event_start": EVENT_START.date(),
                        "event_end": EVENT_END.date(),
                        "event_total_mm": value,
                        "imd_climatology_mm": climatology,
                        "imd_observed_mm": observed,
                    }
                )
    pd.DataFrame(rows).to_csv(csv, index=False)
    report = {
        "plot": str(png),
        "event": "1-7 July 2026 All-India rainfall",
        "forecast_contract": {
            "fuxi_input_daily_statistic": "daily_mean",
            "fuxi_members": N_MEMBERS,
            "ecmwf_members": "perturbed members 1-50, matching prior case plots",
            "spatial_aggregation": "cosine-latitude weighted union of four IMD homogeneous-region masks",
        },
        "imd_climatology_mm": climatology,
        "imd_observed_mm": observed,
        "cases": summaries,
        "interpretation": "Case-specific repeated forecast of one event; not an independent skill sample.",
    }
    audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(png)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
