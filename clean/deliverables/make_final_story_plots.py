#!/usr/bin/env python3
"""Create presentation-ready, period-aligned June 2026 rainfall story plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_verified_case_plots as source


OUT = source.ROOT / "deliverables/final_story_plots"
OLD_23 = Path(
    "/home/raj.ayush/s2s/s2s_anlysis/final_paper/case-study/data/"
    "20260623_all_india_cumulative_timeseries.csv"
)

COLORS = {
    "fuxi": "#138a45",
    "ecmwf": "#e67e00",
    "imd": "#111827",
    "imerg": "#2563eb",
    "imd_climo": "#64748b",
    "era5": "#7c3aed",
    "grid": "#dbe3ea",
    "muted": "#5f6b76",
}


def case_data(init: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(pd.Timestamp(init) + pd.Timedelta(days=1), periods=42)
    data = source.fuxi_series(init)
    data = data.merge(source.ecmwf_series(init, dates), on=["lead_day", "valid_date"], validate="one_to_one")
    data["imd_climatology_mm"] = source.imd_climatology(dates)
    data["era5_climatology_mm"] = source.era5_climatology(dates)
    data = source.add_observations(data)
    return data, source.verification_metrics(data)


def metric(metrics: pd.DataFrame, observation: str, forecast: str, column: str) -> float:
    row = metrics[(metrics.observation == observation) & (metrics.forecast == forecast)]
    return float(row.iloc[0][column])


def style_axis(ax: plt.Axes) -> None:
    ax.set_ylabel("Cumulative rainfall (mm)", fontsize=12)
    ax.set_xlabel("24-hour forecast-period endpoint", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, color=COLORS["grid"], lw=0.8, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def plot_lines(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = data.valid_date
    ax.fill_between(x, data.ecmwf_p10, data.ecmwf_p90, color="#f59e0b", alpha=0.12, linewidth=0)
    ax.plot(x, data.ecmwf_mean, color=COLORS["ecmwf"], lw=2.6, label="ECMWF S2S mean")
    ax.fill_between(x, data.fuxi_p10, data.fuxi_p90, color="#16a34a", alpha=0.15, linewidth=0)
    ax.plot(x, data.fuxi_mean, color=COLORS["fuxi"], lw=3.2, label="FuXi-S2S — daily-mean IC")
    ax.plot(x, data.imd_gauge_cumulative_mm, color=COLORS["imd"], lw=3.0, label="IMD gauge observation")
    ax.plot(x, data.imerg_late_cumulative_mm, color=COLORS["imerg"], lw=2.7, label="IMERG Late observation")
    ax.plot(x, data.imd_climatology_mm, color=COLORS["imd_climo"], lw=1.9, ls="-.", label="IMD 1991–2020 climatology")
    ax.plot(x, data.era5_climatology_mm, color=COLORS["era5"], lw=1.8, ls=":", label="ERA5 climatology")


def story_figure(init: str, output_name: str) -> Path:
    data, metrics = case_data(init)
    full = int(metrics.n_days.min()) == 42
    fig = plt.figure(figsize=(15.5, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.45])
    ax = fig.add_subplot(grid[0, 0])
    note = fig.add_subplot(grid[0, 1])
    plot_lines(ax, data)
    style_axis(ax)

    init_label = pd.Timestamp(init).strftime("%-d %B %Y")
    mode = "Complete 42-day verification" if full else "Partial verification through latest observations"
    ax.set_title(f"All-India 42-Day Cumulative Rainfall — {init_label} Forecast", loc="left", fontsize=17, weight="bold", pad=25)
    ax.text(
        0, 1.015,
        f"{mode}  |  FuXi IC: two complete prior UTC daily means",
        transform=ax.transAxes, color="#4b5563", fontsize=10.8,
    )
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=10)

    last_imd = data.loc[data.imd_gauge_cumulative_mm.notna()].iloc[-1]
    last_imerg = data.loc[data.imerg_late_cumulative_mm.notna()].iloc[-1]
    common_date = min(last_imd.valid_date, last_imerg.valid_date)
    if not full:
        ax.axvline(common_date, color="#94a3b8", lw=1.2, ls="--")
        ax.text(common_date, ax.get_ylim()[1] * 0.97, f" verified through {common_date:%d %b}", color=COLORS["muted"], va="top", fontsize=9.5)

    imd_f = metric(metrics, "IMD gauge", "FuXi-S2S", "cumulative_rmse_mm")
    imd_e = metric(metrics, "IMD gauge", "ECMWF S2S", "cumulative_rmse_mm")
    imerg_f = metric(metrics, "IMERG Late", "FuXi-S2S", "cumulative_rmse_mm")
    imerg_e = metric(metrics, "IMERG Late", "ECMWF S2S", "cumulative_rmse_mm")
    imd_gain = 100 * (1 - imd_f / imd_e)
    imerg_gain = 100 * (1 - imerg_f / imerg_e)

    note.axis("off")
    note.text(0, 0.96, "WHAT THIS CASE SHOWS", fontsize=11, weight="bold", color="#334155", va="top")
    note.text(0, 0.87, "Cumulative-trajectory RMSE", fontsize=10.5, weight="bold", color="#111827", va="top")
    note.text(0, 0.81, f"Against IMD\n  FuXi      {imd_f:5.1f} mm\n  ECMWF  {imd_e:5.1f} mm", fontsize=11, linespacing=1.45, va="top")
    note.text(0, 0.63, f"FuXi RMSE is {imd_gain:.0f}% lower", fontsize=13, weight="bold", color=COLORS["fuxi"], va="top")
    note.text(0, 0.54, f"Against IMERG\n  FuXi      {imerg_f:5.1f} mm\n  ECMWF  {imerg_e:5.1f} mm", fontsize=11, linespacing=1.45, va="top")
    note.text(0, 0.36, f"FuXi RMSE is {imerg_gain:.0f}% lower", fontsize=13, weight="bold", color=COLORS["fuxi"], va="top")
    if full:
        note.text(
            0, 0.28,
            f"42-day totals\nIMD       {last_imd.imd_gauge_cumulative_mm:.0f} mm\n"
            f"IMERG  {last_imerg.imerg_late_cumulative_mm:.0f} mm\n"
            f"FuXi       {data.fuxi_mean.iloc[-1]:.0f} mm\nECMWF  {data.ecmwf_mean.iloc[-1]:.0f} mm",
            fontsize=10.8, linespacing=1.35, va="top",
        )
    else:
        note.text(
            0, 0.28,
            f"Observed to {common_date:%d %b}\nIMD       {last_imd.imd_gauge_cumulative_mm:.0f} mm\n"
            f"IMERG  {last_imerg.imerg_late_cumulative_mm:.0f} mm\n\n"
            "The remaining forecast\nperiod is not yet verified.",
            fontsize=10.8, linespacing=1.35, va="top",
        )
    note.text(
        0, 0.015,
        "Climatologies provide context; observations determine accuracy.\n"
        "IMERG UTC day → following 00 UTC endpoint; IMD is valid near 03 UTC.",
        fontsize=8.7, color=COLORS["muted"], va="bottom",
    )
    output = OUT / output_name
    fig.savefig(output, dpi=240)
    plt.close(fig)
    data.to_csv(OUT / output_name.replace(".png", "_data.csv"), index=False)
    metrics.to_csv(OUT / output_name.replace(".png", "_metrics.csv"), index=False)
    print(output)
    print(metrics.to_string(index=False))
    return output


def sensitivity_figure() -> Path:
    data, _ = case_data("20260623")
    old = pd.read_csv(OLD_23, parse_dates=["valid_date"])[
        ["lead_day", "valid_date", "fuxi_mean", "fuxi_p10", "fuxi_p90"]
    ].rename(columns={"fuxi_mean": "snapshot_mean", "fuxi_p10": "snapshot_p10", "fuxi_p90": "snapshot_p90"})
    data = data.merge(old, on=["lead_day", "valid_date"], validate="one_to_one")

    fig = plt.figure(figsize=(15.5, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.45])
    ax = fig.add_subplot(grid[0, 0])
    note = fig.add_subplot(grid[0, 1])
    x = data.valid_date
    ax.fill_between(x, data.fuxi_p10, data.fuxi_p90, color="#16a34a", alpha=0.14, linewidth=0)
    ax.plot(x, data.fuxi_mean, color=COLORS["fuxi"], lw=3.2, label="FuXi — complete daily-mean IC")
    ax.plot(x, data.snapshot_mean, color="#0f766e", lw=2.7, ls="--", label="FuXi — 00 UTC snapshot IC")
    ax.plot(x, data.ecmwf_mean, color=COLORS["ecmwf"], lw=2.2, alpha=0.85, label="ECMWF S2S mean")
    ax.plot(x, data.imd_gauge_cumulative_mm, color=COLORS["imd"], lw=3.0, label="IMD gauge observation")
    ax.plot(x, data.imd_climatology_mm, color=COLORS["imd_climo"], lw=1.9, ls="-.", label="IMD 1991–2020 climatology")
    style_axis(ax)
    ax.set_title("Why FuXi Input Construction Matters — 23 June 2026", loc="left", fontsize=17, weight="bold", pad=25)
    ax.text(0, 1.015, "Same model and issue date; only the two ERA5 input statistics change", transform=ax.transAxes, color="#4b5563", fontsize=10.8)
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=10)

    daily_total = float(data.fuxi_mean.iloc[-1])
    snapshot_total = float(data.snapshot_mean.iloc[-1])
    difference = daily_total - snapshot_total
    percent = 100 * difference / snapshot_total
    valid = data.imd_gauge_cumulative_mm.notna()
    daily_error = data.loc[valid, "fuxi_mean"] - data.loc[valid, "imd_gauge_cumulative_mm"]
    snap_error = data.loc[valid, "snapshot_mean"] - data.loc[valid, "imd_gauge_cumulative_mm"]

    note.axis("off")
    note.text(0, 0.96, "INPUT SENSITIVITY", fontsize=11, weight="bold", color="#334155", va="top")
    note.text(0, 0.85, f"42-day FuXi total\nDaily mean   {daily_total:.0f} mm\n00 snapshot  {snapshot_total:.0f} mm", fontsize=11.2, linespacing=1.45)
    note.text(0, 0.67, f"Change: +{difference:.0f} mm ({percent:.1f}%)", fontsize=13, weight="bold", color=COLORS["fuxi"])
    note.text(0, 0.55, "Daily-mean IC", fontsize=10.8, weight="bold")
    note.text(0, 0.49, "Two complete preceding UTC days; matches FuXi-S2S training statistics.", fontsize=10, wrap=True)
    note.text(0, 0.36, "00 UTC snapshot IC", fontsize=10.8, weight="bold")
    note.text(0, 0.30, "Instantaneous state plus one-hour TP/TTR accumulations; non-native sensitivity experiment.", fontsize=10, wrap=True)
    note.text(
        0, 0.15,
        f"Partial IMD RMSE\nDaily mean   {np.sqrt(np.mean(daily_error ** 2)):.1f} mm\n"
        f"00 snapshot  {np.sqrt(np.mean(snap_error ** 2)):.1f} mm",
        fontsize=10.5, linespacing=1.35,
    )
    note.text(0, 0.02, "Do not choose the input method from\none favourable incomplete case.", fontsize=9.3, color=COLORS["muted"], va="bottom")

    output = OUT / "03_input_sensitivity_20260623.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    data.to_csv(OUT / "03_input_sensitivity_20260623_data.csv", index=False)
    print(output)
    return output


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#263238", "axes.linewidth": 0.9})
    story_figure("20260601", "01_full_verification_20260601.png")
    story_figure("20260623", "02_partial_verification_20260623.png")
    sensitivity_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
