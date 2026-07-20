#!/usr/bin/env python3
"""Create the corrected professor-facing FuXi/IMD plot for the 1 June issue."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_corrected_prof_fuxi_imd_plot as common  # noqa: E402


OUT = ROOT / "deliverables/imd_story_plots"
FUXI_FILE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "fuxi_s2s_strict00z_case_20260601_ens50/forecasts/annual2026/20260601.nc"
)
OBS_FILE = ROOT / "deliverables/final_story_plots/01_full_verification_20260601_data.csv"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    common.FUXI_FILE = FUXI_FILE
    mask = common.union_mask()
    dates, members = common.load_fuxi(mask)
    if dates[0] != pd.Timestamp("2026-06-02") or dates[-1] != pd.Timestamp("2026-07-13"):
        raise ValueError(f"unexpected forecast dates {dates[0]} through {dates[-1]}")
    imd_climo = common.load_imd_climatology(dates, mask)

    observations = pd.read_csv(OBS_FILE, parse_dates=["valid_date"]).set_index("valid_date").reindex(dates)
    imd_obs = observations.imd_gauge_cumulative_mm.to_numpy(dtype=float)
    if not np.isfinite(imd_obs).all():
        raise ValueError("1 June case must have complete 42-day IMD verification")

    fuxi_mean = members.mean(axis=0)
    fuxi_p10 = np.quantile(members, 0.10, axis=0)
    fuxi_p90 = np.quantile(members, 0.90, axis=0)
    fuxi_member00 = members[0]

    # Cross-check the earlier corrected CSV against the direct-source rebuild.
    if not np.allclose(observations.fuxi_mean, fuxi_mean, atol=5e-4, rtol=0):
        raise ValueError("FuXi series does not match the prior corrected product")
    if not np.allclose(observations.imd_climatology_mm, imd_climo, atol=5e-4, rtol=0):
        raise ValueError("IMD climatology does not match the direct native product")

    week_start, week_end = "2026-07-01", "2026-07-07"
    week_climo = common.period_total(imd_climo, dates, week_start, week_end)
    week_fuxi = common.period_total(fuxi_mean, dates, week_start, week_end)
    week_member00 = common.period_total(fuxi_member00, dates, week_start, week_end)
    week_observed = common.period_total(imd_obs, dates, week_start, week_end)
    fuxi_anomaly = 100.0 * (week_fuxi / week_climo - 1.0)
    observed_anomaly = 100.0 * (week_observed / week_climo - 1.0)
    forecast_error = week_fuxi - week_observed
    forecast_error_percent = 100.0 * forecast_error / week_observed
    full_bias = float(fuxi_mean[-1] - imd_obs[-1])

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

    # For the 1 June issue, period endpoints 1--7 July are leads 30--36.
    ax.axvspan(29.5, 36.5, color="#fef3c7", alpha=0.56, zorder=0)
    ax.text(33, 10, "1–7 July", ha="center", color="#8a5a00", fontsize=9.5)
    ax.fill_between(lead, fuxi_p10, fuxi_p90, color=common.COLORS["fuxi"], alpha=0.15, linewidth=0)
    ax.plot(lead, imd_climo, color=common.COLORS["imd_climo"], lw=2.8, label="IMD 1991–2020 climatology")
    ax.plot(lead, fuxi_mean, color=common.COLORS["fuxi"], lw=3.2, label="FuXi-S2S ensemble mean")
    ax.plot(lead, fuxi_member00, color=common.COLORS["member"], lw=2.4, ls=(0, (6, 4)), label="FuXi member 00")
    ax.plot(lead, imd_obs, color=common.COLORS["imd_obs"], lw=3.1, label="IMD real-time gauge observation")

    ticks = [1, 7, 14, 21, 28, 35, 42]
    labels = [f"L{value}\n{dates[value - 1]:%b %-d}" for value in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_xlim(0.3, 42.7)
    ax.set_ylim(0, max(fuxi_p90.max(), imd_climo.max(), imd_obs.max()) * 1.11)
    ax.set_ylabel("All-India cumulative rainfall (mm)", fontsize=12)
    ax.set_xlabel("Lead day and 24-hour forecast-period endpoint", fontsize=11)
    ax.grid(True, color=common.COLORS["grid"], lw=0.8, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=9.7)
    ax.set_title(
        "FuXi-S2S Predicted the Wet First Week of July from the 1 June Issue",
        loc="left", fontsize=16.4, weight="bold", pad=25,
    )
    ax.text(
        0, 1.015,
        "Issued 1 June 2026 | valid 2 June–13 July | complete 30–31 May UTC daily-mean inputs",
        transform=ax.transAxes, color="#4b5563", fontsize=10.4,
    )

    note.axis("off")
    note.text(0, 0.96, "1–7 JULY: LEADS 30–36", fontsize=11, weight="bold", color="#334155", va="top")
    note.text(
        0, 0.87,
        f"IMD climatology   {week_climo:.1f} mm\n"
        f"FuXi forecast       {week_fuxi:.1f} mm\n"
        f"IMD observed       {week_observed:.1f} mm",
        fontsize=11.4, linespacing=1.55, va="top",
    )
    note.text(0, 0.66, f"FuXi predicted: +{fuxi_anomaly:.0f}%", fontsize=13.2, weight="bold", color=common.COLORS["fuxi"], va="top")
    note.text(0, 0.59, f"IMD observed: +{observed_anomaly:.0f}%", fontsize=13.2, weight="bold", color=common.COLORS["imd_obs"], va="top")
    note.text(
        0, 0.49,
        f"Forecast − observation\n{forecast_error:+.1f} mm ({forecast_error_percent:+.0f}%)",
        fontsize=10.9, linespacing=1.4, va="top",
    )
    note.text(0, 0.36, "COMPLETE 42-DAY VERIFICATION", fontsize=10.8, weight="bold", va="top")
    note.text(
        0, 0.30,
        f"FuXi mean            {fuxi_mean[-1]:.0f} mm\n"
        f"FuXi member 00  {fuxi_member00[-1]:.0f} mm\n"
        f"IMD climatology   {imd_climo[-1]:.0f} mm\n"
        f"IMD observed       {imd_obs[-1]:.0f} mm",
        fontsize=10.9, linespacing=1.4, va="top",
    )
    note.text(
        0, 0.12,
        f"Event captured closely; full 42-day\naccumulation was +{full_bias:.0f} mm too wet.",
        fontsize=10.3, weight="bold", color="#334155", linespacing=1.28, va="top",
    )
    note.text(
        0, 0.008,
        "Same IMD mask/dates/baseline; IMD ≈03 UTC vs forecast 00 UTC.\n"
        "One case does not establish general model skill.",
        fontsize=7.5, color=common.COLORS["muted"], va="bottom",
    )

    png = OUT / "07_corrected_prof_fuxi_imd_20260601.png"
    csv = OUT / "07_corrected_prof_fuxi_imd_20260601.csv"
    audit = OUT / "07_corrected_prof_fuxi_imd_20260601_audit.json"
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
    report = {
        "forecast_contract": {
            "issue": "2026-06-01T00:00:00Z",
            "input_daily_statistic": "daily_mean",
            "input_days": ["2026-05-30", "2026-05-31"],
            "information_cutoff": "2026-06-01T00:00:00Z",
            "valid_period_endpoints": ["2026-06-02", "2026-07-13"],
            "tp_conversion": "FuXi mm h-1 daily mean rate multiplied by 24, then cumulatively summed",
        },
        "first_week_july_mm": {
            "lead_days": [30, 36],
            "imd_climatology": week_climo,
            "fuxi_ensemble_mean": week_fuxi,
            "fuxi_member00": week_member00,
            "imd_gauge_observation": week_observed,
            "fuxi_anomaly_percent": fuxi_anomaly,
            "imd_observed_anomaly_percent": observed_anomaly,
            "fuxi_minus_observed": forecast_error,
        },
        "full_42day_mm": {
            "fuxi_ensemble_mean": float(fuxi_mean[-1]),
            "fuxi_member00": float(fuxi_member00[-1]),
            "imd_climatology": float(imd_climo[-1]),
            "imd_gauge_observation": float(imd_obs[-1]),
            "fuxi_minus_observed": full_bias,
        },
    }
    audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(png)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
