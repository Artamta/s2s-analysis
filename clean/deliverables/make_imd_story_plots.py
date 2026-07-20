#!/usr/bin/env python3
"""Create three presentation-ready FuXi/ECMWF precipitation plots for IMD.

The figures deliberately use IMD gauge rainfall as the only verification
reference.  IMD 1991--2020 climatology is retained only as context in the
42-day trajectory figure; ERA5 and IMERG are not plotted or scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deliverables/final_story_plots"
OUT = ROOT / "deliverables/imd_story_plots"

COLORS = {
    "fuxi": "#11823b",
    "ecmwf": "#e97800",
    "imd": "#111827",
    "climo": "#64748b",
    "grid": "#dbe3ea",
    "muted": "#596574",
    "positive": "#138a45",
    "negative": "#c2413b",
}


def load_case(prefix: str) -> pd.DataFrame:
    path = SOURCE / f"{prefix}_data.csv"
    data = pd.read_csv(path, parse_dates=["valid_date"])
    required = {
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
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return data


def cumulative_metrics(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    valid = data["imd_gauge_cumulative_mm"].notna()
    if not valid.any():
        raise ValueError("case has no IMD verification")
    result: dict[str, dict[str, float]] = {}
    for column, label in (("fuxi_mean", "FuXi-S2S"), ("ecmwf_mean", "ECMWF S2S")):
        error = (
            data.loc[valid, column].to_numpy(dtype=float)
            - data.loc[valid, "imd_gauge_cumulative_mm"].to_numpy(dtype=float)
        )
        result[label] = {
            "n_days": int(valid.sum()),
            "rmse_mm": float(np.sqrt(np.mean(error**2))),
            "mae_mm": float(np.mean(np.abs(error))),
            "endpoint_bias_mm": float(error[-1]),
            "endpoint_abs_error_mm": float(abs(error[-1])),
            "endpoint_forecast_mm": float(data.loc[valid, column].iloc[-1]),
        }
    result["IMD gauge"] = {
        "n_days": int(valid.sum()),
        "endpoint_observed_mm": float(data.loc[valid, "imd_gauge_cumulative_mm"].iloc[-1]),
    }
    return result


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.edgecolor": "#263238",
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_cumulative(data: pd.DataFrame) -> Path:
    metrics = cumulative_metrics(data)
    fuxi = metrics["FuXi-S2S"]
    ecmwf = metrics["ECMWF S2S"]
    imd = metrics["IMD gauge"]
    rmse_gain = 100.0 * (1.0 - fuxi["rmse_mm"] / ecmwf["rmse_mm"])
    endpoint_gain = 100.0 * (
        1.0 - fuxi["endpoint_abs_error_mm"] / ecmwf["endpoint_abs_error_mm"]
    )

    fig = plt.figure(figsize=(15.5, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.75, 1.35])
    ax = fig.add_subplot(grid[0, 0])
    note = fig.add_subplot(grid[0, 1])
    x = data.valid_date

    ax.fill_between(
        x, data.ecmwf_p10, data.ecmwf_p90,
        color=COLORS["ecmwf"], alpha=0.11, linewidth=0,
    )
    ax.plot(x, data.ecmwf_mean, color=COLORS["ecmwf"], lw=2.6, label="ECMWF S2S ensemble mean")
    ax.fill_between(
        x, data.fuxi_p10, data.fuxi_p90,
        color=COLORS["fuxi"], alpha=0.15, linewidth=0,
    )
    ax.plot(x, data.fuxi_mean, color=COLORS["fuxi"], lw=3.2, label="FuXi-S2S ensemble mean")
    ax.plot(
        x, data.imd_gauge_cumulative_mm,
        color=COLORS["imd"], lw=3.2, label="IMD gauge observation",
    )
    ax.plot(
        x, data.imd_climatology_mm,
        color=COLORS["climo"], lw=2.0, ls="-.", label="IMD 1991–2020 climatology",
    )

    ax.set_title(
        "FuXi Tracks IMD More Closely Than ECMWF in This 42-Day Case",
        loc="left", fontsize=17, pad=25,
    )
    ax.text(
        0, 1.015,
        "Forecast issued 1 June 2026  |  Valid 2 June–13 July 2026  |  Complete verification",
        transform=ax.transAxes, color="#4b5563", fontsize=10.8,
    )
    ax.set_ylabel("All-India cumulative rainfall (mm)", fontsize=12)
    ax.set_xlabel("24-hour forecast-period endpoint", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, color=COLORS["grid"], lw=0.8, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=10)

    note.axis("off")
    note.text(0, 0.96, "CLEAR MESSAGE FOR IMD", fontsize=11, weight="bold", color="#334155", va="top")
    note.text(0, 0.86, "Cumulative-trajectory RMSE", fontsize=10.5, weight="bold", va="top")
    note.text(
        0, 0.80,
        f"FuXi       {fuxi['rmse_mm']:.1f} mm\nECMWF  {ecmwf['rmse_mm']:.1f} mm",
        fontsize=11.2, linespacing=1.5, va="top",
    )
    note.text(
        0, 0.65, f"{rmse_gain:.0f}% lower RMSE",
        fontsize=14, color=COLORS["positive"], weight="bold", va="top",
    )
    note.text(0, 0.54, "42-day totals", fontsize=10.5, weight="bold", va="top")
    note.text(
        0, 0.48,
        f"IMD        {imd['endpoint_observed_mm']:.0f} mm\n"
        f"FuXi       {fuxi['endpoint_forecast_mm']:.0f} mm\n"
        f"ECMWF  {ecmwf['endpoint_forecast_mm']:.0f} mm",
        fontsize=11.2, linespacing=1.45, va="top",
    )
    note.text(
        0, 0.28,
        f"Endpoint wet bias\nFuXi       +{fuxi['endpoint_bias_mm']:.0f} mm\n"
        f"ECMWF  +{ecmwf['endpoint_bias_mm']:.0f} mm",
        fontsize=10.8, linespacing=1.4, va="top",
    )
    note.text(
        0, 0.12, f"FuXi reduces endpoint error by {endpoint_gain:.0f}%",
        fontsize=11.2, color=COLORS["positive"], weight="bold", va="top",
    )
    note.text(
        0, 0.015,
        "Case-study result, not a general skill claim.\n"
        "Verification: IMD real-time 0.25° gauge analysis.\n"
        "Shading: ensemble 10–90% range.",
        fontsize=8.6, color=COLORS["muted"], va="bottom",
    )

    output = OUT / "01_imd_42day_cumulative_20260601.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output


def weekly_table(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) != 42 or data.imd_gauge_cumulative_mm.isna().any():
        raise ValueError("weekly diagnostic requires a complete 42-day IMD record")
    rows = []
    for week in range(6):
        start = week * 7
        stop = start + 7
        row: dict[str, object] = {
            "week": week + 1,
            "date_range": f"{data.valid_date.iloc[start]:%-d %b}–{data.valid_date.iloc[stop - 1]:%-d %b}",
        }
        for column, label in (
            ("imd_gauge_cumulative_mm", "IMD"),
            ("fuxi_mean", "FuXi"),
            ("ecmwf_mean", "ECMWF"),
        ):
            previous = 0.0 if start == 0 else float(data[column].iloc[start - 1])
            row[label] = float(data[column].iloc[stop - 1] - previous)
        row["FuXi bias"] = float(row["FuXi"] - row["IMD"])
        row["ECMWF bias"] = float(row["ECMWF"] - row["IMD"])
        row["FuXi absolute error"] = abs(float(row["FuXi bias"]))
        row["ECMWF absolute error"] = abs(float(row["ECMWF bias"]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_weekly(data: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    weekly = weekly_table(data)
    closer = int((weekly["FuXi absolute error"] < weekly["ECMWF absolute error"]).sum())

    fig, (ax, bias_ax) = plt.subplots(
        2, 1, figsize=(13.5, 8.2), constrained_layout=True,
        gridspec_kw={"height_ratios": [2.25, 1.0]}, sharex=True,
    )
    index = np.arange(6)
    width = 0.24
    ax.bar(index - width, weekly.IMD, width, color=COLORS["imd"], label="IMD gauge observation")
    ax.bar(index, weekly.FuXi, width, color=COLORS["fuxi"], label="FuXi-S2S mean")
    ax.bar(index + width, weekly.ECMWF, width, color=COLORS["ecmwf"], label="ECMWF S2S mean")
    for x_pos, value in zip(index, weekly.FuXi, strict=True):
        ax.text(x_pos, value + 1.5, f"{value:.0f}", ha="center", va="bottom", fontsize=8.8, color=COLORS["fuxi"])
    for x_pos, value in zip(index - width, weekly.IMD, strict=True):
        ax.text(x_pos, value + 1.5, f"{value:.0f}", ha="center", va="bottom", fontsize=8.8, color=COLORS["imd"])
    for x_pos, value in zip(index + width, weekly.ECMWF, strict=True):
        ax.text(x_pos, value + 1.5, f"{value:.0f}", ha="center", va="bottom", fontsize=8.8, color=COLORS["ecmwf"])
    ax.set_ylabel("7-day rainfall total (mm)")
    ax.set_ylim(0, max(weekly[["IMD", "FuXi", "ECMWF"]].max()) * 1.23)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.set_title(
        "The FuXi Advantage Is Not Just an Endpoint Effect",
        loc="left", fontsize=17, pad=25,
    )
    ax.text(
        0, 1.015,
        f"Weekly rainfall against IMD  |  FuXi is closer in {closer} of 6 weeks",
        transform=ax.transAxes, color="#4b5563", fontsize=10.8,
    )

    bias_ax.axhline(0, color="#334155", lw=1.0)
    bias_ax.bar(index - width / 1.7, weekly["FuXi bias"], width * 1.15, color=COLORS["fuxi"], label="FuXi − IMD")
    bias_ax.bar(index + width / 1.7, weekly["ECMWF bias"], width * 1.15, color=COLORS["ecmwf"], label="ECMWF − IMD")
    bias_ax.set_ylabel("Weekly bias (mm)")
    bias_ax.set_xlabel("Lead week and valid dates")
    bias_ax.grid(axis="y", color=COLORS["grid"], lw=0.8, alpha=0.85)
    bias_ax.spines[["top", "right"]].set_visible(False)
    bias_ax.legend(frameon=False, ncols=2, loc="upper left")
    labels = [f"Week {row.week}\n{row.date_range}" for row in weekly.itertuples(index=False)]
    bias_ax.set_xticks(index, labels)
    bias_ax.text(
        0.995, 0.04,
        "Positive bias = forecast is wetter than IMD",
        transform=bias_ax.transAxes, ha="right", color=COLORS["muted"], fontsize=9,
    )

    output = OUT / "02_imd_weekly_rainfall_and_bias_20260601.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output, weekly


def plot_scorecard(full: pd.DataFrame, partial: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    cases = [
        ("1 Jun issue\n42/42 days", "Complete", full),
        ("23 Jun issue\n25/42 days", "Partial to 18 Jul", partial),
    ]
    rows = []
    for case_label, status, data in cases:
        metrics = cumulative_metrics(data)
        for model in ("FuXi-S2S", "ECMWF S2S"):
            rows.append({"case": case_label, "status": status, "model": model, **metrics[model]})
    table = pd.DataFrame(rows)

    fig, (rmse_ax, endpoint_ax) = plt.subplots(1, 2, figsize=(13.7, 6.7))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.78, wspace=0.18)
    positions = np.arange(2)
    width = 0.32
    colors = {"FuXi-S2S": COLORS["fuxi"], "ECMWF S2S": COLORS["ecmwf"]}

    for model, shift in (("FuXi-S2S", -width / 2), ("ECMWF S2S", width / 2)):
        subset = table[table.model == model]
        bars = rmse_ax.bar(positions + shift, subset.rmse_mm, width, color=colors[model], label=model)
        rmse_ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
        bars = endpoint_ax.bar(positions + shift, subset.endpoint_abs_error_mm, width, color=colors[model], label=model)
        endpoint_ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)

    for idx, case_label in enumerate([case[0] for case in cases]):
        f = table[(table.case == case_label) & (table.model == "FuXi-S2S")].iloc[0]
        e = table[(table.case == case_label) & (table.model == "ECMWF S2S")].iloc[0]
        rmse_gain = 100 * (1 - f.rmse_mm / e.rmse_mm)
        endpoint_gain = 100 * (1 - f.endpoint_abs_error_mm / e.endpoint_abs_error_mm)
        rmse_ax.text(idx, max(f.rmse_mm, e.rmse_mm) + 4.3, f"FuXi {rmse_gain:.0f}% lower", ha="center", color=COLORS["positive"], weight="bold")
        endpoint_ax.text(idx, max(f.endpoint_abs_error_mm, e.endpoint_abs_error_mm) + 5.5, f"FuXi {endpoint_gain:.0f}% lower", ha="center", color=COLORS["positive"], weight="bold")

    rmse_ax.set_title("A. Cumulative-trajectory RMSE", loc="left", fontsize=14)
    endpoint_ax.set_title("B. Absolute error at last verified day", loc="left", fontsize=14)
    rmse_ax.set_ylabel("Error against IMD (mm)")
    endpoint_ax.set_ylabel("Error against IMD (mm)")
    for ax in (rmse_ax, endpoint_ax):
        ax.set_xticks(positions, [case[0] for case in cases])
        ax.grid(axis="y", color=COLORS["grid"], lw=0.8, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
    rmse_ax.set_ylim(0, 72)
    endpoint_ax.set_ylim(0, 116)
    handles, labels = rmse_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncols=2,
        loc="upper left", bbox_to_anchor=(0.068, 0.855), borderaxespad=0,
    )

    fig.suptitle(
        "FuXi Is Closer to IMD in Both Available June 2026 Cases",
        x=0.015, y=0.965, ha="left", fontsize=17, weight="bold",
    )
    fig.text(
        0.015, 0.905,
        "Two overlapping case studies; the 23 June case is only verified through 18 July",
        color="#4b5563", fontsize=10.8,
    )
    fig.text(
        0.985, 0.035,
        "Case evidence only—use the multi-year hindcast for a general model-skill claim.",
        ha="right", color=COLORS["muted"], fontsize=9,
    )

    output = OUT / "03_imd_two_case_scorecard.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output, table


def write_readme(paths: list[Path], weekly: pd.DataFrame, scorecard: pd.DataFrame) -> None:
    full = scorecard[scorecard["case"].str.startswith("1 Jun")]
    fuxi = full[full.model == "FuXi-S2S"].iloc[0]
    ecmwf = full[full.model == "ECMWF S2S"].iloc[0]
    message = f"""# IMD precipitation story plots

These plots intentionally use **IMD gauge rainfall as the only verification
reference**. ERA5 and IMERG are excluded. IMD 1991–2020 climatology appears in
the first figure only as context, never as an observation or skill target.

## Recommended order and talk track

1. **Full 42-day verification:** In the 1 June 2026 case, FuXi follows the IMD
   cumulative trajectory more closely than ECMWF. Its trajectory RMSE is
   {fuxi.rmse_mm:.1f} mm versus {ecmwf.rmse_mm:.1f} mm ({100 * (1 - fuxi.rmse_mm / ecmwf.rmse_mm):.0f}% lower).
2. **Where the difference comes from:** FuXi has the smaller weekly absolute
   error in {int((weekly['FuXi absolute error'] < weekly['ECMWF absolute error']).sum())} of 6 weeks. Both forecasts are too wet overall; this plot makes that limitation visible.
3. **Does the signal repeat?:** FuXi is closer to IMD in the second June issue
   as well, but that forecast is only partially verified and the two cases
   overlap. Present this as case evidence—not as a general skill claim.

## Files

""" + "\n".join(f"- `{path.name}`" for path in paths) + """

## Method note

All-India means use the cosine-latitude-weighted union of the four IMD
homogeneous-region masks. Forecast trajectories are ensemble means; shading in
Figure 1 is the 10–90% ensemble range. IMD real-time 0.25° gauge rainfall uses
the product's stated daily convention (valid near 03 UTC), while forecast
period endpoints are at 00 UTC, leaving an approximately three-hour timing
offset.
"""
    (OUT / "README.md").write_text(message, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    full = load_case("01_full_verification_20260601")
    partial = load_case("02_partial_verification_20260623")
    first = plot_cumulative(full)
    second, weekly = plot_weekly(full)
    third, scorecard = plot_scorecard(full, partial)
    weekly.to_csv(OUT / "02_imd_weekly_rainfall_and_bias_20260601.csv", index=False)
    scorecard.to_csv(OUT / "03_imd_two_case_scorecard.csv", index=False)
    report = {
        "verification_reference": "IMD real-time daily rainfall 0.25-degree gauge analysis",
        "era5_used": False,
        "imerg_used": False,
        "figures": [path.name for path in (first, second, third)],
        "full_case_metrics": cumulative_metrics(full),
        "partial_case_metrics": cumulative_metrics(partial),
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_readme([first, second, third], weekly, scorecard)
    for path in (first, second, third):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
