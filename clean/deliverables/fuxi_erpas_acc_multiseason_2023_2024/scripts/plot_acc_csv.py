#!/usr/bin/env python3
"""Render meeting figures from the audited multiseason ACC CSV only."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SUMMARY = HERE / "metrics/acc_summary_by_year_season.csv"
COMPARISON = HERE / "metrics/acc_fuxi_minus_erpas_by_year_season.csv"
REGIONAL_COMPARISON = HERE / "metrics/regional_acc_fuxi_minus_erpas.csv"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


COLORS = {"ERPAS": "#D45532", "FuXi-S2S": "#008F80"}
MARKERS = {"ERPAS": "o", "FuXi-S2S": "D"}
SEASONS = ("JF", "MAM", "JJAS", "OND")
REGIONS = (
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
REGION_LABELS = {
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D8E1E6",
            "grid.alpha": 0.82,
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    output = HERE / "figures" / stem
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=280, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_curves(summary: pd.DataFrame) -> None:
    frame = summary[
        (summary.method == "common_imd_1991_2020")
        & (summary.year.astype(str) == "ALL")
        & summary.season.isin(SEASONS)
    ].copy()
    style()
    fig = plt.figure(figsize=(16.4, 7.8), facecolor="white")
    grid = fig.add_gridspec(2, 4, height_ratios=(3.1, 1.2), hspace=0.14, wspace=0.20)
    weeks = np.arange(1, 5)
    for column, season in enumerate(SEASONS):
        top = fig.add_subplot(grid[0, column])
        bottom = fig.add_subplot(grid[1, column], sharex=top)
        season_frame = frame[frame.season == season]
        n_cases = int(season_frame.n_cases.max())
        values_by_model = {}
        for model in ("ERPAS", "FuXi-S2S"):
            model_frame = season_frame[season_frame.model == model].sort_values("week")
            values = model_frame.acc_mean.to_numpy()
            values_by_model[model] = values
            top.fill_between(
                weeks,
                model_frame.acc_q25,
                model_frame.acc_q75,
                color=COLORS[model],
                alpha=0.09,
                linewidth=0,
            )
            top.plot(
                weeks,
                values,
                color=COLORS[model],
                marker=MARKERS[model],
                linewidth=2.7,
                markersize=7,
                label=model,
            )
            for x, value in zip(weeks, values):
                top.annotate(
                    f"{value:.2f}",
                    (x, value),
                    xytext=(0, 10 if model == "FuXi-S2S" else -17),
                    textcoords="offset points",
                    ha="center",
                    color=COLORS[model],
                    fontsize=9,
                    fontweight="bold",
                )
        delta = values_by_model["FuXi-S2S"] - values_by_model["ERPAS"]
        winner_count = int(np.sum(delta > 0))
        if winner_count == 4:
            story = "FuXi higher at all four weeks"
        elif winner_count == 0:
            story = "ERPAS higher at all four weeks"
        else:
            story = f"Mixed ranking: FuXi leads {winner_count}/4 weeks"
        top.set_title(f"{season}  ·  n={n_cases}\n{story}", fontsize=12, pad=10)
        top.axhline(0, color="#71818B", linewidth=1.0)
        top.set_xlim(0.75, 4.25)
        top.set_ylim(-0.12, 0.82)
        top.set_xticks(weeks)
        top.set_xticklabels([])
        if column == 0:
            top.set_ylabel("Mean spatial ACC")
        if column == 3:
            top.legend(frameon=False, loc="upper right", fontsize=10)

        bar_colors = [COLORS["FuXi-S2S"] if value >= 0 else COLORS["ERPAS"] for value in delta]
        bottom.bar(weeks, delta, width=0.58, color=bar_colors, alpha=0.92)
        bottom.axhline(0, color="#27353D", linewidth=1.1)
        bottom.set_ylim(-0.22, 0.22)
        bottom.set_xticks(weeks, [f"W{week}" for week in weeks])
        if column == 0:
            bottom.set_ylabel("FuXi − ERPAS\nACC")
        for x, value, color in zip(weeks, delta, bar_colors):
            bottom.annotate(
                f"{value:+.2f}",
                (x, value),
                xytext=(0, 6 if value >= 0 else -14),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=8.8,
                fontweight="bold",
            )

    fig.suptitle(
        "Season changes the FuXi–ERPAS rainfall-skill ranking",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.945,
        "101 paired starts across 2023–2024  •  four disjoint issue-month seasons  •  common IMD 1991–2020 anomaly baseline  •  same 1.5° India support",
        fontsize=10.3,
        color="#43545E",
    )
    fig.text(
        0.065,
        0.028,
        "Primary statistic follows paper_v2: arithmetic mean of per-initialization area-weighted spatial ACC. Shading is the interquartile case range (descriptive).",
        fontsize=8.8,
        color="#485A64",
    )
    fig.text(
        0.065,
        0.010,
        "Seasons use ERPAS issue month: JF, MAM, JJAS and OND. The separate JFM paper_v2 window remains available in the CSV.",
        fontsize=8.5,
        color="#5B6971",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.865, bottom=0.105)
    save(fig, "seasonal_acc_fuxi_vs_erpas_2023_2024")


def plot_heatmap(comparison: pd.DataFrame) -> None:
    frame = comparison[
        (comparison.method == "common_imd_1991_2020")
        & comparison.season.isin(SEASONS)
    ].copy()
    rows = []
    labels = []
    counts = []
    for year in ("2023", "2024", "ALL"):
        for season in SEASONS:
            subset = frame[
                (frame.year.astype(str) == year) & (frame.season == season)
            ].sort_values("week")
            if len(subset) != 4:
                raise ValueError(f"missing heatmap scores for {year}/{season}")
            rows.append(subset.acc_mean_fuxi_minus_erpas.to_numpy())
            year_label = "Combined" if year == "ALL" else year
            labels.append(f"{year_label} · {season}")
            counts.append(int(subset.n_cases.iloc[0]))
    values = np.stack(rows)
    limit = max(0.15, np.ceil(np.max(np.abs(values)) * 20) / 20)
    cmap = LinearSegmentedColormap.from_list(
        "erpas_neutral_fuxi", [COLORS["ERPAS"], "#F7F8F7", COLORS["FuXi-S2S"]]
    )
    style()
    fig, ax = plt.subplots(figsize=(8.8, 7.3), facecolor="white")
    image = ax.imshow(
        values,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    ax.grid(False)
    ax.set_xticks(np.arange(4), ["Week 1", "Week 2", "Week 3", "Week 4"])
    ax.set_yticks(
        np.arange(len(labels)),
        [f"{label}   n={count}" for label, count in zip(labels, counts)],
    )
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            ax.text(
                x,
                y,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if abs(value) > 0.65 * limit else "#26343C",
            )
    for boundary in (3.5, 7.5):
        ax.axhline(boundary, color="white", linewidth=4)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.set_label("FuXi − ERPAS mean ACC", fontsize=10)
    ax.set_title("Where does each model lead?", loc="left", fontsize=18, pad=32)
    ax.text(
        0,
        1.025,
        "Teal = FuXi advantage   •   Orange = ERPAS advantage   •   Positive values favor FuXi",
        transform=ax.transAxes,
        fontsize=9.4,
        color="#45565F",
    )
    fig.text(
        0.13,
        0.022,
        "Paired 2023–2024 starts; common IMD 1991–2020 anomaly baseline; arithmetic mean of per-initialization spatial ACC.",
        fontsize=8.5,
        color="#53636C",
    )
    fig.subplots_adjust(left=0.24, right=0.91, top=0.88, bottom=0.09)
    save(fig, "year_season_acc_advantage_heatmap")


def plot_regional_heatmap(comparison: pd.DataFrame) -> None:
    frame = comparison[
        (comparison.method == "common_imd_1991_2020")
        & (comparison.year.astype(str) == "ALL")
        & comparison.season.isin(SEASONS)
        & comparison.region.isin(REGIONS)
    ].copy()
    expected = len(SEASONS) * len(REGIONS) * 4
    if len(frame) != expected:
        raise ValueError(f"expected {expected} combined regional scores, got {len(frame)}")

    values_by_season: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for season in SEASONS:
        subset = frame[frame.season == season]
        pivot = subset.pivot(index="region", columns="week", values="acc_mean_fuxi_minus_erpas")
        values_by_season[season] = pivot.reindex(index=REGIONS, columns=range(1, 5)).to_numpy()
        counts[season] = int(subset.n_cases.max())

    maximum = max(float(np.nanmax(np.abs(value))) for value in values_by_season.values())
    limit = max(0.20, np.ceil(maximum * 20) / 20)
    cmap = LinearSegmentedColormap.from_list(
        "erpas_neutral_fuxi", [COLORS["ERPAS"], "#F7F8F7", COLORS["FuXi-S2S"]]
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), facecolor="white")
    images = []
    for panel, (axis, season) in enumerate(zip(axes.flat, SEASONS)):
        values = values_by_season[season]
        image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        images.append(image)
        axis.grid(False)
        axis.set_title(f"{season}  ·  n={counts[season]}", fontsize=12.5, pad=9)
        axis.set_xticks(np.arange(4), [f"W{week}" for week in range(1, 5)])
        axis.set_yticks(np.arange(4), [REGION_LABELS[name] for name in REGIONS])
        axis.tick_params(axis="y", labelsize=9.2)
        for row in range(4):
            for column in range(4):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    fontweight="bold",
                    color="white" if abs(value) > 0.64 * limit else "#26343C",
                )
        if panel < 2:
            axis.set_xlabel("")
        else:
            axis.set_xlabel("Lead week")

    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("FuXi − ERPAS mean ACC", fontsize=10)
    fig.suptitle(
        "Regional rainfall skill depends on season and lead",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.942,
        "Official IMD homogeneous rainfall regions  •  teal favors FuXi-S2S  •  orange favors ERPAS",
        fontsize=10,
        color="#43545E",
    )
    fig.text(
        0.075,
        0.025,
        "Arithmetic mean of per-start area-weighted spatial ACC; common IMD 1991–2020 anomaly baseline; four disjoint issue-month seasons.",
        fontsize=8.7,
        color="#53636C",
    )
    fig.subplots_adjust(left=0.15, right=0.90, top=0.88, bottom=0.10, hspace=0.30, wspace=0.31)
    save(fig, "imd_homogeneous_region_acc_advantage_2023_2024")


def plot_jjas_system_climatologies(summary: pd.DataFrame) -> None:
    frame = summary[
        (summary.method == "system_specific_jjas")
        & (summary.year.astype(str) == "ALL")
        & (summary.season == "JJAS")
    ].copy()
    if len(frame) != 8:
        raise ValueError("expected two models x four weeks for system-specific JJAS")
    style()
    fig = plt.figure(figsize=(9.8, 7.2), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(3.2, 1.15), hspace=0.13)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    weeks = np.arange(1, 5)
    values_by_model = {}
    for model in ("ERPAS", "FuXi-S2S"):
        model_frame = frame[frame.model == model].sort_values("week")
        values = model_frame.acc_mean.to_numpy()
        values_by_model[model] = values
        top.fill_between(
            weeks,
            model_frame.acc_q25,
            model_frame.acc_q75,
            color=COLORS[model],
            alpha=0.10,
            linewidth=0,
        )
        top.plot(
            weeks,
            values,
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=2.9,
            markersize=8,
            label=model,
        )
        for x, value in zip(weeks, values):
            top.annotate(
                f"{value:.2f}",
                (x, value),
                xytext=(0, 12 if model == "FuXi-S2S" else -19),
                textcoords="offset points",
                ha="center",
                color=COLORS[model],
                fontweight="bold",
            )
    top.axhline(0, color="#71818B", linewidth=1.0)
    top.set_xlim(0.75, 4.25)
    top.set_ylim(-0.12, 0.72)
    top.set_xticks(weeks)
    top.set_xticklabels([])
    top.set_ylabel("Mean spatial ACC")
    top.legend(frameon=False, ncol=2, loc="upper right")
    top.text(
        0.0,
        1.02,
        "Higher ACC = better placement of the observed rainfall-anomaly pattern",
        transform=top.transAxes,
        fontsize=9.5,
        color="#52636D",
    )

    delta = values_by_model["FuXi-S2S"] - values_by_model["ERPAS"]
    bottom.bar(weeks, delta, width=0.58, color=COLORS["FuXi-S2S"], alpha=0.92)
    bottom.axhline(0, color="#27353D", linewidth=1.1)
    bottom.set_ylim(-0.08, 0.23)
    bottom.set_xticks(weeks, [f"Week {week}" for week in weeks])
    bottom.set_ylabel("FuXi − ERPAS\nACC")
    for x, value in zip(weeks, delta):
        bottom.annotate(
            f"FuXi {value:+.2f}",
            (x, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["FuXi-S2S"],
            fontweight="bold",
        )

    fig.suptitle(
        "FuXi retains the JJAS advantage with system-specific climatologies",
        x=0.09,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.925,
        "31 paired starts across 2023–2024  •  same 1.5° all-season India support  •  arithmetic mean of per-initialization ACC",
        fontsize=9.6,
        color="#43545E",
    )
    fig.text(
        0.09,
        0.037,
        "FuXi forecast anomaly: native 2002–2021 reforecast climatology. ERPAS: provider 20-source climatology. IMD truth: IMD 1991–2020 climatology.",
        fontsize=8.5,
        color="#485A64",
    )
    fig.text(
        0.09,
        0.017,
        "Shading is the interquartile range across starts (descriptive, not a confidence interval).",
        fontsize=8.4,
        color="#5B6971",
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.115)
    save(fig, "jjas_acc_system_specific_climatologies_2023_2024")


def main() -> int:
    summary = pd.read_csv(SUMMARY, dtype={"year": str})
    comparison = pd.read_csv(COMPARISON, dtype={"year": str})
    regional_comparison = pd.read_csv(REGIONAL_COMPARISON, dtype={"year": str})
    plot_curves(summary)
    plot_heatmap(comparison)
    plot_regional_heatmap(regional_comparison)
    plot_jjas_system_climatologies(summary)
    print("wrote four-window ACC, year/window advantage, regional, and JJAS system-climatology figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
