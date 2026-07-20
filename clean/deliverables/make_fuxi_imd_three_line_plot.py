#!/usr/bin/env python3
"""Plot 24 June FuXi/ECMWF forecasts with IMD climatology and available truth."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deliverables/final_story_plots/02_partial_verification_20260623_data.csv"
OUT = ROOT / "deliverables/imd_story_plots"

FUXI = "#138a45"
ECMWF = "#e67e00"
IMD = "#111827"
CLIMO = "#64748b"
GRID = "#dbe3ea"
MUTED = "#5f6b76"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE, parse_dates=["valid_date"])
    required = {
        "lead_day",
        "valid_date",
        "fuxi_mean",
        "fuxi_p10",
        "fuxi_p90",
        "ecmwf_mean",
        "ecmwf_p10",
        "ecmwf_p90",
        "imd_climatology_mm",
        "imd_gauge_cumulative_mm",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"source data are missing {sorted(missing)}")
    if len(data) != 42 or data.valid_date.iloc[0] != pd.Timestamp("2026-06-24"):
        raise ValueError("expected the 42-day forecast valid from 24 June 2026")

    observed = data.imd_gauge_cumulative_mm.notna()
    if not observed.any():
        raise ValueError("no IMD gauge verification is available")
    last_index = int(np.flatnonzero(observed.to_numpy())[-1])
    cutoff = data.valid_date.iloc[last_index]
    if not observed.iloc[: last_index + 1].all() or observed.iloc[last_index + 1 :].any():
        raise ValueError("IMD verification must form one continuous prefix")

    observed_total = float(data.imd_gauge_cumulative_mm.iloc[last_index])
    fuxi_to_cutoff = float(data.fuxi_mean.iloc[last_index])
    ecmwf_to_cutoff = float(data.ecmwf_mean.iloc[last_index])
    climo_to_cutoff = float(data.imd_climatology_mm.iloc[last_index])
    fuxi_bias = fuxi_to_cutoff - observed_total
    ecmwf_bias = ecmwf_to_cutoff - observed_total
    observed_departure = observed_total - climo_to_cutoff
    full_fuxi = float(data.fuxi_mean.iloc[-1])
    full_ecmwf = float(data.ecmwf_mean.iloc[-1])
    full_climo = float(data.imd_climatology_mm.iloc[-1])

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
    fig = plt.figure(figsize=(15.4, 7.2), constrained_layout=True)
    layout = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.35])
    ax = fig.add_subplot(layout[0, 0])
    note = fig.add_subplot(layout[0, 1])
    x = data.valid_date

    future_start = cutoff + pd.Timedelta(hours=12)
    ax.axvspan(future_start, x.iloc[-1] + pd.Timedelta(hours=12), color="#f1f5f9", alpha=0.8, zorder=0)
    ax.fill_between(x, data.ecmwf_p10, data.ecmwf_p90, color=ECMWF, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(x, data.ecmwf_mean, color=ECMWF, lw=2.8, label="ECMWF-S2S ensemble mean", zorder=2)
    ax.fill_between(x, data.fuxi_p10, data.fuxi_p90, color=FUXI, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(x, data.fuxi_mean, color=FUXI, lw=3.3, label="FuXi-S2S ensemble mean", zorder=3)
    ax.plot(x, data.imd_climatology_mm, color=CLIMO, lw=2.2, ls="--", label="IMD 1991–2020 climatology", zorder=2)
    ax.plot(x, data.imd_gauge_cumulative_mm, color=IMD, lw=3.4, label="IMD real-time gauge observation", zorder=4)
    ax.scatter([cutoff], [observed_total], color=IMD, s=45, zorder=5)
    ax.axvline(cutoff, color="#94a3b8", lw=1.3, ls=":", zorder=1)
    ax.text(
        cutoff + pd.Timedelta(days=0.4), 10,
        f"IMD observation available\nthrough {cutoff:%-d %B}",
        color=MUTED, fontsize=9.5, va="bottom",
    )
    ax.text(
        cutoff + pd.Timedelta(days=7.7), 37,
        "forecast not yet verified",
        color="#64748b", fontsize=10, ha="center",
    )

    ax.set_title(
        "24 June S2S Forecasts: IMD Climatology and Available IMD Observation",
        loc="left", fontsize=16.5, weight="bold", pad=25,
    )
    ax.text(
        0, 1.015,
        "Issued 23 June 2026  |  42-day period: 24 June–4 August 2026",
        transform=ax.transAxes, color="#4b5563", fontsize=10.8,
    )
    ax.set_ylabel("All-India cumulative rainfall (mm)", fontsize=12)
    ax.set_xlabel("24-hour forecast-period endpoint", fontsize=11)
    ax.set_ylim(0, max(data.fuxi_p90.max(), data.ecmwf_p90.max(), data.imd_climatology_mm.max()) * 1.06)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=10)

    note.axis("off")
    note.text(0, 0.96, "IMD-CENTERED READING", fontsize=11, weight="bold", color="#334155", va="top")
    note.text(0, 0.85, f"Verified through {cutoff:%-d %B}", fontsize=10.8, weight="bold", va="top")
    note.text(
        0, 0.78,
        f"IMD observed       {observed_total:.0f} mm\n"
        f"IMD climatology   {climo_to_cutoff:.0f} mm\n"
        f"FuXi forecast       {fuxi_to_cutoff:.0f} mm\n"
        f"ECMWF forecast  {ecmwf_to_cutoff:.0f} mm",
        fontsize=11.2, linespacing=1.5, va="top",
    )
    note.text(
        0, 0.52,
        f"FuXi wet bias: +{fuxi_bias:.0f} mm ({100 * fuxi_bias / observed_total:.0f}%)",
        fontsize=11.7, color=FUXI, weight="bold", va="top",
    )
    note.text(
        0, 0.45,
        f"ECMWF wet bias: +{ecmwf_bias:.0f} mm ({100 * ecmwf_bias / observed_total:.0f}%)",
        fontsize=11.7, color=ECMWF, weight="bold", va="top",
    )
    note.text(
        0, 0.35,
        f"Observed departure from\nIMD climatology: {observed_departure:+.0f} mm",
        fontsize=10.8, linespacing=1.35, va="top",
    )
    note.text(0, 0.23, "Full 42-day context", fontsize=10.8, weight="bold", va="top")
    note.text(
        0, 0.17,
        f"FuXi forecast       {full_fuxi:.0f} mm\n"
        f"ECMWF forecast  {full_ecmwf:.0f} mm\n"
        f"IMD climatology   {full_climo:.0f} mm",
        fontsize=11.0, linespacing=1.45, va="top",
    )
    note.text(
        0, 0.015,
        "Only the observed segment is verification.\n"
        "Climatology is context, not ground truth.\n"
        "Shading: ensemble 10–90% ranges.",
        fontsize=8.7, color=MUTED, va="bottom",
    )

    png = OUT / "05_fuxi_ecmwf_imd_climatology_observed_from_20260624.png"
    csv = OUT / "05_fuxi_ecmwf_imd_climatology_observed_from_20260624.csv"
    metrics = OUT / "05_fuxi_ecmwf_imd_climatology_observed_from_20260624.json"
    fig.savefig(png, dpi=240)
    plt.close(fig)
    data[
        [
            "lead_day", "valid_date", "fuxi_mean", "fuxi_p10", "fuxi_p90",
            "ecmwf_mean", "ecmwf_p10", "ecmwf_p90",
            "imd_climatology_mm", "imd_gauge_cumulative_mm",
        ]
    ].to_csv(csv, index=False)
    metrics.write_text(
        json.dumps(
            {
                "forecast_issue": "2026-06-23T00:00:00Z",
                "first_forecast_period_endpoint": "2026-06-24T00:00:00Z",
                "last_forecast_period_endpoint": "2026-08-04T00:00:00Z",
                "imd_observation_available_through": cutoff.strftime("%Y-%m-%d"),
                "observed_days": last_index + 1,
                "at_observation_cutoff_mm": {
                    "imd_observed": observed_total,
                    "imd_1991_2020_climatology": climo_to_cutoff,
                    "fuxi_ensemble_mean": fuxi_to_cutoff,
                    "fuxi_minus_imd_observed": fuxi_bias,
                    "ecmwf_ensemble_mean": ecmwf_to_cutoff,
                    "ecmwf_minus_imd_observed": ecmwf_bias,
                },
                "full_42day_context_mm": {
                    "fuxi_ensemble_mean": full_fuxi,
                    "ecmwf_ensemble_mean": full_ecmwf,
                    "imd_1991_2020_climatology": full_climo,
                },
                "era5_used": False,
                "imerg_used": False,
                "ecmwf_used": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(png)
    print(metrics.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
