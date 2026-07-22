#!/usr/bin/env python3
"""Create restrained, presentation-grade IMD briefing figures.

The figures read only the audited CSV outputs.  They deliberately avoid
inferential ribbons: the displayed statistic is the paper_v2-aligned
arithmetic mean of per-initialization spatial ACC, and the paired sample size
is printed in every seasonal panel.
"""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SUMMARY = HERE / "metrics/acc_summary_by_year_season.csv"
REGIONAL = HERE / "metrics/regional_acc_fuxi_minus_erpas.csv"
REGIONAL_SUMMARY = HERE / "metrics/regional_acc_summary_by_year_season.csv"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FUXI = "#0072B2"
ERPAS = "#D55E00"
INK = "#172A36"
MUTED = "#536773"
GRID = "#DCE5E9"
SEASONS = (
    ("JF", "Winter (JF)"),
    ("MAM", "Pre-monsoon (MAM)"),
    ("JJAS", "Monsoon (JJAS)"),
    ("OND", "Post-monsoon (OND)"),
)
REGIONS = (
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
REGION_LABELS = (
    "Northwest India",
    "Central India",
    "South Peninsula",
    "East & Northeast India",
)
DOMAIN_LABELS = ("All India",) + REGION_LABELS


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#85959E",
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    output = HERE / "figures" / stem
    fig.savefig(
        output.with_suffix(".png"),
        dpi=360,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def all_india_acc(summary: pd.DataFrame) -> None:
    frame = summary[
        (summary.method == "common_imd_1991_2020")
        & (summary.year.astype(str) == "ALL")
        & summary.season.isin([key for key, _ in SEASONS])
    ].copy()
    if len(frame) != 32:
        raise ValueError(f"expected 32 all-India seasonal summary rows, got {len(frame)}")

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.333, 7.5), sharex=True, sharey=True)
    weeks = np.arange(1, 5)
    panel_letters = "abcd"
    for index, (axis, (season, season_label)) in enumerate(zip(axes.flat, SEASONS)):
        subset = frame[frame.season == season]
        values: dict[str, np.ndarray] = {}
        for model, color, marker in (
            ("ERPAS", ERPAS, "o"),
            ("FuXi-S2S", FUXI, "D"),
        ):
            model_frame = subset[subset.model == model].sort_values("week")
            values[model] = model_frame.acc_mean.to_numpy()
            axis.plot(
                weeks,
                values[model],
                color=color,
                linewidth=2.4,
                marker=marker,
                markersize=6.2,
                markerfacecolor="white",
                markeredgewidth=1.8,
                zorder=3,
            )

        delta = values["FuXi-S2S"] - values["ERPAS"]
        fuxi_wins = int(np.sum(delta > 0))
        if fuxi_wins == 4:
            result, result_color = "FuXi higher · 4/4 weeks", FUXI
        elif fuxi_wins == 0:
            result, result_color = "ERPAS higher · 4/4 weeks", ERPAS
        else:
            result, result_color = f"Mixed · FuXi higher {fuxi_wins}/4", MUTED

        n_cases = int(subset.n_cases.max())
        axis.set_title(
            f"{panel_letters[index]}   {season_label}",
            loc="left",
            pad=15,
            color=INK,
        )
        axis.text(
            1.0,
            1.055,
            f"n = {n_cases}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.4,
            color=MUTED,
        )
        axis.text(
            0.98,
            0.91,
            result,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9.3,
            fontweight="semibold",
            color=result_color,
        )
        axis.axhline(0, color="#7E909A", linewidth=0.9, zorder=1)
        axis.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.9)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0.78, 4.22)
        axis.set_ylim(-0.10, 0.72)
        axis.set_xticks(weeks, ["Week 1", "Week 2", "Week 3", "Week 4"])
        axis.set_yticks(np.arange(0.0, 0.8, 0.2))

    for axis in axes[:, 0]:
        axis.set_ylabel("Spatial ACC")
    for axis in axes[1, :]:
        axis.set_xlabel("Lead time")

    handles = [
        Line2D(
            [0], [0], color=FUXI, marker="D", markerfacecolor="white",
            markeredgewidth=1.6, linewidth=2.4, label="FuXi-S2S"
        ),
        Line2D(
            [0], [0], color=ERPAS, marker="o", markerfacecolor="white",
            markeredgewidth=1.6, linewidth=2.4, label="ERPAS"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.955, 0.925),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Seasonal rainfall anomaly correlation",
        x=0.072,
        y=0.982,
        ha="left",
        fontsize=20,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.072,
        0.932,
        "FuXi-S2S and ERPAS  |  IMD verification  |  101 paired weekly issues, 2023–2024",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.072,
        0.028,
        "Mean of per-initialization area-weighted spatial ACC · common IMD 1991–2020 daily climatology · common 1.5° grid (FuXi native) · higher is better",
        fontsize=8.7,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.12, hspace=0.38, wspace=0.16)
    save(fig, "imd_briefing_all_india_acc_2023_2024")


def regional_scorecard(regional: pd.DataFrame) -> None:
    frame = regional[
        (regional.method == "common_imd_1991_2020")
        & (regional.year.astype(str) == "ALL")
        & regional.season.isin([key for key, _ in SEASONS])
        & regional.region.isin(REGIONS)
    ].copy()
    if len(frame) != 64:
        raise ValueError(f"expected 64 regional comparison rows, got {len(frame)}")

    matrices: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for season, _ in SEASONS:
        subset = frame[frame.season == season]
        pivot = subset.pivot(
            index="region", columns="week", values="acc_mean_fuxi_minus_erpas"
        )
        matrices[season] = pivot.reindex(index=REGIONS, columns=range(1, 5)).to_numpy()
        counts[season] = int(subset.n_cases.max())

    set_style()
    cmap = LinearSegmentedColormap.from_list(
        "erpas_white_fuxi", [ERPAS, "#F8FAFA", FUXI]
    )
    norm = TwoSlopeNorm(vmin=-0.40, vcenter=0, vmax=0.40)
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 5.25), sharey=True)
    images = []
    for index, (axis, (season, season_label)) in enumerate(zip(axes, SEASONS)):
        values = matrices[season]
        image = axis.imshow(values, cmap=cmap, norm=norm, aspect="equal")
        images.append(image)
        axis.set_title(f"{season_label}\nn = {counts[season]}", pad=12, color=INK)
        axis.set_xticks(np.arange(4), ["W1", "W2", "W3", "W4"])
        axis.set_yticks(np.arange(4), REGION_LABELS)
        axis.tick_params(axis="y", labelleft=index == 0, length=0)
        axis.tick_params(axis="x", length=0, pad=7)
        axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.spines[:].set_visible(False)
        for row in range(4):
            for column in range(4):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    fontweight="semibold",
                    color="white" if abs(value) >= 0.25 else INK,
                )

    colorbar = fig.colorbar(
        images[0],
        ax=axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.22,
        aspect=55,
        shrink=0.72,
    )
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(labelsize=8.8, colors=INK)
    fig.suptitle(
        "Regional difference in rainfall anomaly correlation",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.055,
        0.900,
        "Official IMD homogeneous rainfall regions  |  ΔACC = FuXi-S2S − ERPAS  |  blue: positive, orange: negative",
        fontsize=10.3,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.025,
        "Paired 2023–2024 issues · area-weighted spatial ACC · common IMD 1991–2020 daily climatology · common 1.5° grid (FuXi native)",
        fontsize=8.7,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.70, bottom=0.23, wspace=0.055)
    save(fig, "imd_briefing_homogeneous_regions_acc_2023_2024")


def jjas_all_india_and_regions(
    summary: pd.DataFrame, regional_summary: pd.DataFrame
) -> None:
    """Actual ACC curves for All India plus four IMD regions in JJAS."""
    all_india = summary[
        (summary.method == "common_imd_1991_2020")
        & (summary.year.astype(str) == "ALL")
        & (summary.season == "JJAS")
    ].copy()
    regions = regional_summary[
        (regional_summary.method == "common_imd_1991_2020")
        & (regional_summary.year.astype(str) == "ALL")
        & (regional_summary.season == "JJAS")
        & regional_summary.region.isin(REGIONS)
    ].copy()
    if len(all_india) != 8 or len(regions) != 32:
        raise ValueError("incomplete JJAS All-India/regional summary")

    set_style()
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        left=0.075,
        right=0.975,
        top=0.80,
        bottom=0.12,
        hspace=0.58,
        wspace=0.55,
    )
    axes = [
        fig.add_subplot(grid[0, 1:3]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[2, 0:2]),
        fig.add_subplot(grid[2, 2:4]),
    ]
    domains = [("All India", all_india)] + [
        (label, regions[regions.region == key])
        for key, label in zip(REGIONS, REGION_LABELS)
    ]
    weeks = np.arange(1, 5)
    for index, (axis, (label, frame)) in enumerate(zip(axes, domains)):
        values: dict[str, np.ndarray] = {}
        for model, color, marker in (
            ("ERPAS", ERPAS, "o"),
            ("FuXi-S2S", FUXI, "D"),
        ):
            model_frame = frame[frame.model == model].sort_values("week")
            values[model] = model_frame.acc_mean.to_numpy()
            if len(values[model]) != 4:
                raise ValueError(f"incomplete JJAS curve for {label}/{model}")
            axis.plot(
                weeks,
                values[model],
                color=color,
                linewidth=2.15,
                marker=marker,
                markersize=5.6,
                markerfacecolor="white",
                markeredgewidth=1.6,
                zorder=3,
            )
        delta = values["FuXi-S2S"] - values["ERPAS"]
        fuxi_wins = int(np.sum(delta > 0))
        if fuxi_wins == 4:
            result, result_color = "FuXi higher · 4/4", FUXI
        elif fuxi_wins == 0:
            result, result_color = "ERPAS higher · 4/4", ERPAS
        else:
            result, result_color = f"Mixed · FuXi {fuxi_wins}/4", MUTED
        axis.set_title(f"{'abcde'[index]}   {label}", loc="left", pad=9, color=INK)
        axis.text(
            0.98,
            0.92,
            result,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.6,
            fontweight="semibold",
            color=result_color,
        )
        axis.axhline(0, color="#7E909A", linewidth=0.8, zorder=1)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0.78, 4.22)
        axis.set_ylim(-0.15, 0.76)
        axis.set_xticks(weeks, ["W1", "W2", "W3", "W4"])
        axis.set_yticks(np.arange(0.0, 0.8, 0.2))
    for axis in (axes[0], axes[1], axes[3]):
        axis.set_ylabel("Spatial ACC")
    for axis in axes[3:]:
        axis.set_xlabel("Lead week")

    handles = [
        Line2D(
            [0], [0], color=FUXI, marker="D", markerfacecolor="white",
            markeredgewidth=1.5, linewidth=2.2, label="FuXi-S2S"
        ),
        Line2D(
            [0], [0], color=ERPAS, marker="o", markerfacecolor="white",
            markeredgewidth=1.5, linewidth=2.2, label="ERPAS"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.925),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.4,
    )
    fig.suptitle(
        "Monsoon rainfall anomaly correlation by region",
        x=0.075,
        y=0.982,
        ha="left",
        fontsize=20,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.075,
        0.932,
        "All India and four official IMD homogeneous regions  |  JJAS  |  n = 31 paired issues, 2023–2024",
        fontsize=10.4,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.027,
        "Mean of per-initialization area-weighted spatial ACC · common IMD 1991–2020 daily climatology · common 1.5° grid (FuXi native) · higher is better",
        fontsize=8.6,
        color=MUTED,
    )
    save(fig, "imd_briefing_jjas_all_india_and_regions_acc_2023_2024")


def all_domains_scorecard(
    summary: pd.DataFrame, regional: pd.DataFrame
) -> None:
    """Four-season delta-ACC scorecard including All India as the first row."""
    all_india = summary[
        (summary.method == "common_imd_1991_2020")
        & (summary.year.astype(str) == "ALL")
        & summary.season.isin([key for key, _ in SEASONS])
    ].copy()
    region_frame = regional[
        (regional.method == "common_imd_1991_2020")
        & (regional.year.astype(str) == "ALL")
        & regional.season.isin([key for key, _ in SEASONS])
        & regional.region.isin(REGIONS)
    ].copy()
    if len(all_india) != 32 or len(region_frame) != 64:
        raise ValueError("incomplete all-domain scorecard inputs")

    matrices: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for season, _ in SEASONS:
        season_all = all_india[all_india.season == season]
        all_pivot = season_all.pivot(index="model", columns="week", values="acc_mean")
        all_delta = (
            all_pivot.loc["FuXi-S2S", range(1, 5)].to_numpy()
            - all_pivot.loc["ERPAS", range(1, 5)].to_numpy()
        )
        season_regions = region_frame[region_frame.season == season]
        region_pivot = season_regions.pivot(
            index="region", columns="week", values="acc_mean_fuxi_minus_erpas"
        )
        matrices[season] = np.vstack(
            [all_delta, region_pivot.reindex(index=REGIONS, columns=range(1, 5)).to_numpy()]
        )
        counts[season] = int(season_all.n_cases.max())

    set_style()
    cmap = LinearSegmentedColormap.from_list(
        "erpas_white_fuxi", [ERPAS, "#F8FAFA", FUXI]
    )
    norm = TwoSlopeNorm(vmin=-0.40, vcenter=0, vmax=0.40)
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 5.8), sharey=True)
    images = []
    for index, (axis, (season, season_label)) in enumerate(zip(axes, SEASONS)):
        values = matrices[season]
        image = axis.imshow(values, cmap=cmap, norm=norm, aspect="equal")
        images.append(image)
        axis.set_title(f"{season_label}\nn = {counts[season]}", pad=11, color=INK)
        axis.set_xticks(np.arange(4), ["W1", "W2", "W3", "W4"])
        axis.set_yticks(np.arange(5), DOMAIN_LABELS)
        axis.tick_params(axis="y", labelleft=index == 0, length=0)
        axis.tick_params(axis="x", length=0, pad=7)
        axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2.0)
        axis.axhline(0.5, color="white", linewidth=5.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.spines[:].set_visible(False)
        for row in range(5):
            for column in range(4):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="semibold",
                    color="white" if abs(value) >= 0.25 else INK,
                )

    colorbar = fig.colorbar(
        images[0],
        ax=axes,
        orientation="horizontal",
        fraction=0.048,
        pad=0.18,
        aspect=58,
        shrink=0.72,
    )
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(labelsize=8.7, colors=INK)
    fig.suptitle(
        "All-India and regional ACC advantage",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=19,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.055,
        0.917,
        "ΔACC = FuXi-S2S − ERPAS  |  blue: positive  |  orange: negative  |  official IMD homogeneous regions",
        fontsize=10.2,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.025,
        "Paired 2023–2024 issues · area-weighted spatial ACC · common IMD 1991–2020 daily climatology · common 1.5° grid (FuXi native)",
        fontsize=8.6,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.76, bottom=0.25, wspace=0.055)
    save(fig, "imd_briefing_all_india_and_regions_scorecard_2023_2024")


def all_season_domain_summary(
    summary: pd.DataFrame, regional: pd.DataFrame
) -> None:
    """Compact headline: lead-average delta ACC for five domains by season."""
    all_india = summary[
        (summary.method == "common_imd_1991_2020")
        & (summary.year.astype(str) == "ALL")
        & summary.season.isin([key for key, _ in SEASONS])
    ].copy()
    region_frame = regional[
        (regional.method == "common_imd_1991_2020")
        & (regional.year.astype(str) == "ALL")
        & regional.season.isin([key for key, _ in SEASONS])
        & regional.region.isin(REGIONS)
    ].copy()
    if len(all_india) != 32 or len(region_frame) != 64:
        raise ValueError("incomplete compact-summary inputs")

    records: list[dict] = []
    for season, _ in SEASONS:
        season_all = all_india[all_india.season == season]
        pivot = season_all.pivot(index="model", columns="week", values="acc_mean")
        all_delta = (
            pivot.loc["FuXi-S2S", range(1, 5)].to_numpy()
            - pivot.loc["ERPAS", range(1, 5)].to_numpy()
        )
        records.append(
            {
                "season": season,
                "domain": "All India",
                "delta": float(np.mean(all_delta)),
                "n_cases": int(season_all.n_cases.max()),
            }
        )
        season_regions = region_frame[region_frame.season == season]
        for key, label in zip(REGIONS, REGION_LABELS):
            subset = season_regions[season_regions.region == key].sort_values("week")
            records.append(
                {
                    "season": season,
                    "domain": label,
                    "delta": float(subset.acc_mean_fuxi_minus_erpas.mean()),
                    "n_cases": int(subset.n_cases.max()),
                }
            )
    frame = pd.DataFrame(records)
    if len(frame) != 20 or not np.isfinite(frame.delta).all():
        raise ValueError("compact-summary construction failed")

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), sharex=True, sharey=True)
    y = np.arange(len(DOMAIN_LABELS))
    panel_letters = "abcd"
    for index, (axis, (season, season_label)) in enumerate(zip(axes.flat, SEASONS)):
        subset = frame[frame.season == season].set_index("domain").reindex(DOMAIN_LABELS)
        values = subset.delta.to_numpy()
        colors = [FUXI if value >= 0 else ERPAS for value in values]
        axis.axvspan(-0.30, 0, color=ERPAS, alpha=0.035, zorder=0)
        axis.axvspan(0, 0.30, color=FUXI, alpha=0.035, zorder=0)
        axis.barh(y, values, height=0.56, color=colors, alpha=0.92, zorder=3)
        axis.axvline(0, color="#667A85", linewidth=1.0, zorder=4)
        axis.set_title(
            f"{panel_letters[index]}   {season_label}",
            loc="left",
            pad=13,
            color=INK,
        )
        axis.text(
            1.0,
            1.045,
            f"n = {int(subset.n_cases.max())}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            color=MUTED,
        )
        for row, value, color in zip(y, values, colors):
            axis.annotate(
                f"{value:+.2f}",
                xy=(value, row),
                xytext=(6 if value >= 0 else -6, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=9.0,
                fontweight="semibold",
                color=color,
            )
        axis.set_yticks(y, DOMAIN_LABELS)
        axis.set_xlim(-0.30, 0.30)
        axis.set_xticks(np.arange(-0.3, 0.31, 0.1))
        axis.grid(axis="x", color=GRID, linewidth=0.7, zorder=1)
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=8)
        if index % 2 == 1:
            axis.tick_params(axis="y", labelleft=False)
    axes[0, 0].invert_yaxis()
    for axis in axes[:, 0]:
        axis.get_yticklabels()[0].set_fontweight("semibold")
    for axis in axes[1, :]:
        axis.set_xlabel("Mean ΔACC across Weeks 1–4")

    fig.suptitle(
        "Seasonal ACC advantage across India",
        x=0.075,
        y=0.978,
        ha="left",
        fontsize=19,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.075,
        0.928,
        "All India and four official IMD homogeneous regions  |  blue: FuXi-S2S higher  |  orange: ERPAS higher",
        fontsize=10.2,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.028,
        "ΔACC = FuXi-S2S − ERPAS · equal-weight mean of the four lead-specific seasonal ACC values · paired 2023–2024 issues · common 1.5° grid",
        fontsize=8.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.19, right=0.97, top=0.83, bottom=0.12, hspace=0.35, wspace=0.16)
    save(fig, "imd_briefing_all_seasons_all_india_regions_summary_2023_2024")


def all_seasons_domain_line_plot(
    summary: pd.DataFrame, regional_summary: pd.DataFrame
) -> None:
    """Simple seasonal line profiles for All India and the four IMD regions."""
    season_keys = [key for key, _ in SEASONS]
    season_counts: dict[str, int] = {}
    all_india = summary[
        (summary.method == "common_imd_1991_2020")
        & (summary.year.astype(str) == "ALL")
        & summary.season.isin(season_keys)
    ].copy()
    regions = regional_summary[
        (regional_summary.method == "common_imd_1991_2020")
        & (regional_summary.year.astype(str) == "ALL")
        & regional_summary.season.isin(season_keys)
        & regional_summary.region.isin(REGIONS)
    ].copy()
    if len(all_india) != 32 or len(regions) != 128:
        raise ValueError("incomplete all-season line-plot inputs")
    for season in season_keys:
        season_counts[season] = int(
            all_india[all_india.season == season].n_cases.max()
        )

    domain_frames = [("All India", all_india)] + [
        (label, regions[regions.region == key])
        for key, label in zip(REGIONS, REGION_LABELS)
    ]
    x = np.arange(len(SEASONS))
    xlabels = [f"{key}\nn={season_counts[key]}" for key, _ in SEASONS]
    set_style()
    fig = plt.figure(figsize=(12.6, 7.7), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        left=0.075,
        right=0.975,
        top=0.80,
        bottom=0.125,
        hspace=0.57,
        wspace=0.52,
    )
    axes = [
        fig.add_subplot(grid[0, 1:3]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[2, 0:2]),
        fig.add_subplot(grid[2, 2:4]),
    ]
    for index, (axis, (domain, frame)) in enumerate(zip(axes, domain_frames)):
        for model, color, marker in (
            ("ERPAS", ERPAS, "o"),
            ("FuXi-S2S", FUXI, "D"),
        ):
            values = []
            for season in season_keys:
                subset = frame[
                    (frame.season == season) & (frame.model == model)
                ].sort_values("week")
                if len(subset) != 4:
                    raise ValueError(f"incomplete seasonal mean for {domain}/{model}/{season}")
                values.append(float(subset.acc_mean.mean()))
            axis.plot(
                x,
                values,
                color=color,
                linewidth=2.35,
                marker=marker,
                markersize=6.2,
                markerfacecolor="white",
                markeredgewidth=1.7,
                zorder=3,
            )
        axis.set_title(f"{'abcde'[index]}   {domain}", loc="left", pad=9, color=INK)
        axis.axhline(0, color="#7E909A", linewidth=0.8, zorder=1)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-0.18, 3.18)
        axis.set_ylim(-0.06, 0.70)
        axis.set_xticks(x, xlabels)
        axis.set_yticks(np.arange(0.0, 0.71, 0.1))
    for axis in (axes[0], axes[1], axes[3]):
        axis.set_ylabel("Mean spatial ACC")
    for axis in axes[3:]:
        axis.set_xlabel("Season (issue month)")

    handles = [
        Line2D(
            [0], [0], color=FUXI, marker="D", markerfacecolor="white",
            markeredgewidth=1.6, linewidth=2.35, label="FuXi-S2S"
        ),
        Line2D(
            [0], [0], color=ERPAS, marker="o", markerfacecolor="white",
            markeredgewidth=1.6, linewidth=2.35, label="ERPAS"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.925),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.4,
    )
    fig.suptitle(
        "Seasonal rainfall anomaly correlation across India",
        x=0.075,
        y=0.982,
        ha="left",
        fontsize=20,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.075,
        0.932,
        "All India and four official IMD homogeneous regions  |  each point is the mean across lead Weeks 1–4",
        fontsize=10.3,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.027,
        "Arithmetic mean of four lead-specific, per-initialization mean spatial ACC values · paired 2023–2024 issues · IMD 1991–2020 climatology · common 1.5° grid",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "imd_briefing_all_seasons_all_india_regions_line_2023_2024")


def fuxi_jjas_climatology_robustness(summary: pd.DataFrame) -> None:
    """Headline FuXi defense: JJAS advantage under two anomaly baselines."""
    methods = (
        (
            "common_imd_1991_2020",
            "Common IMD baseline",
            "Forecast and observed anomalies use IMD 1991–2020",
        ),
        (
            "system_specific_jjas",
            "System-specific baselines",
            "FuXi 2002–2021 · ERPAS provider · observed IMD 1991–2020",
        ),
    )
    frame = summary[
        (summary.year.astype(str) == "ALL")
        & (summary.season == "JJAS")
        & summary.method.isin([method for method, _, _ in methods])
    ].copy()
    if len(frame) != 16:
        raise ValueError(f"expected 16 JJAS robustness rows, got {len(frame)}")

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), sharex=True, sharey=True)
    weeks = np.arange(1, 5)
    for index, (axis, (method, title, baseline_note)) in enumerate(zip(axes, methods)):
        subset = frame[frame.method == method]
        values: dict[str, np.ndarray] = {}
        for model, color, marker in (
            ("ERPAS", ERPAS, "o"),
            ("FuXi-S2S", FUXI, "D"),
        ):
            model_frame = subset[subset.model == model].sort_values("week")
            values[model] = model_frame.acc_mean.to_numpy()
            if len(values[model]) != 4:
                raise ValueError(f"incomplete robustness curve for {method}/{model}")
            axis.plot(
                weeks,
                values[model],
                color=color,
                linewidth=2.7,
                marker=marker,
                markersize=7.2,
                markerfacecolor="white",
                markeredgewidth=1.9,
                zorder=4,
            )

        delta = values["FuXi-S2S"] - values["ERPAS"]
        for x_value, erpas_value, fuxi_value, difference in zip(
            weeks, values["ERPAS"], values["FuXi-S2S"], delta
        ):
            axis.annotate(
                f"+{difference:.2f}",
                (x_value, max(erpas_value, fuxi_value)),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="semibold",
                color=FUXI,
            )

        axis.set_title(f"{'ab'[index]}   {title}", loc="left", pad=29, color=INK)
        axis.text(
            0.0,
            1.035,
            baseline_note,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            color=MUTED,
        )
        axis.text(
            0.98,
            0.91,
            "FuXi higher at 4/4 weeks",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9.4,
            fontweight="semibold",
            color=FUXI,
        )
        axis.axhline(0, color="#7E909A", linewidth=0.8, zorder=1)
        axis.grid(axis="y", color=GRID, linewidth=0.75)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0.75, 4.25)
        axis.set_ylim(-0.02, 0.66)
        axis.set_xticks(weeks, ["Week 1", "Week 2", "Week 3", "Week 4"])
        axis.set_yticks(np.arange(0.0, 0.7, 0.1))
        axis.set_xlabel("Lead time")
    axes[0].set_ylabel("Mean spatial ACC")

    handles = [
        Line2D(
            [0], [0], color=FUXI, marker="D", markerfacecolor="white",
            markeredgewidth=1.7, linewidth=2.7, label="FuXi-S2S"
        ),
        Line2D(
            [0], [0], color=ERPAS, marker="o", markerfacecolor="white",
            markeredgewidth=1.7, linewidth=2.7, label="ERPAS"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.955, 0.905),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.4,
    )
    fig.suptitle(
        "FuXi-S2S shows higher JJAS rainfall ACC than ERPAS",
        x=0.072,
        y=0.982,
        ha="left",
        fontsize=17,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.072,
        0.915,
        "31 paired JJAS issues, 2023–2024  |  IMD verification  |  labels show FuXi-minus-ERPAS ACC",
        fontsize=10.3,
        color=MUTED,
    )
    fig.text(
        0.072,
        0.028,
        "Arithmetic mean of per-initialization area-weighted spatial ACC · identical valid periods · common 1.5° grid (FuXi native) · higher is better",
        fontsize=8.6,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.13, wspace=0.15)
    save(fig, "imd_briefing_fuxi_jjas_climatology_robustness_2023_2024")


def fuxi_jjas_system_specific_headline(summary: pd.DataFrame) -> None:
    """Standalone, meeting-ready version of the system-specific JJAS result."""
    frame = summary[
        (summary.method == "system_specific_jjas")
        & (summary.year.astype(str) == "ALL")
        & (summary.season == "JJAS")
    ].copy()
    if len(frame) != 8:
        raise ValueError(f"expected 8 system-specific JJAS rows, got {len(frame)}")
    if set(frame.n_cases.astype(int)) != {31}:
        raise ValueError(f"unexpected paired sample sizes: {sorted(frame.n_cases.unique())}")

    weeks = np.arange(1, 5)
    values: dict[str, np.ndarray] = {}
    for model in ("FuXi-S2S", "ERPAS"):
        model_frame = frame[frame.model == model].sort_values("week")
        values[model] = model_frame.acc_mean.to_numpy()
        if len(values[model]) != 4:
            raise ValueError(f"incomplete system-specific curve for {model}")

    set_style()
    fig, axis = plt.subplots(figsize=(13.333, 7.5))
    for model, color, marker in (
        ("FuXi-S2S", FUXI, "D"),
        ("ERPAS", ERPAS, "o"),
    ):
        axis.plot(
            weeks,
            values[model],
            color=color,
            linewidth=3.2,
            marker=marker,
            markersize=9,
            markerfacecolor="white",
            markeredgewidth=2.2,
            label=model,
            zorder=4,
        )

    for x_value, fuxi_value, erpas_value in zip(
        weeks, values["FuXi-S2S"], values["ERPAS"]
    ):
        axis.annotate(
            f"{fuxi_value:.2f}",
            (x_value, fuxi_value),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color=FUXI,
        )
        axis.annotate(
            f"{erpas_value:.2f}",
            (x_value, erpas_value),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="semibold",
            color=ERPAS,
        )

    deltas = values["FuXi-S2S"] - values["ERPAS"]
    delta_text = "   ".join(
        f"W{week}  +{delta:.2f}" for week, delta in zip(weeks, deltas)
    )
    axis.text(
        0.5,
        0.975,
        f"FuXi minus ERPAS:   {delta_text}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10.4,
        fontweight="semibold",
        color=FUXI,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E9F4FA", "edgecolor": "none"},
    )

    axis.axhline(0, color="#7E909A", linewidth=0.9, zorder=1)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlim(0.75, 4.25)
    axis.set_ylim(-0.02, 0.68)
    axis.set_xticks(weeks, ["Week 1", "Week 2", "Week 3", "Week 4"])
    axis.set_yticks(np.arange(0.0, 0.7, 0.1))
    axis.set_xlabel("Forecast lead")
    axis.set_ylabel("Mean spatial ACC")
    axis.legend(loc="upper right", frameon=False, ncol=2, columnspacing=2.0)
    axis.text(
        0.99,
        0.82,
        "FuXi higher at all 4 lead weeks",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color=FUXI,
    )

    fig.suptitle(
        "FuXi-S2S retains more JJAS rainfall-pattern skill through Week 4",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=20,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.075,
        0.918,
        "Paired weekly forecasts, 2023–2024  |  IMD rainfall verification  |  common 1.5° India grid",
        fontsize=10.8,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.058,
        "Model anomalies: FuXi 2002–2021 native hindcast climatology; ERPAS provider climatology. IMD anomalies: 1991–2020 climatology.",
        fontsize=8.9,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.029,
        "Mean of 31 paired initialization cycles per week (17 in 2023; 14 in 2024; 31 of 34 possible) = 124 forecast-week evaluations per system. Descriptive means; no significance claim.",
        fontsize=8.9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.975, top=0.80, bottom=0.16)
    save(fig, "imd_briefing_fuxi_jjas_system_specific_acc_2023_2024")


def main() -> int:
    summary = pd.read_csv(SUMMARY, dtype={"year": str})
    regional = pd.read_csv(REGIONAL, dtype={"year": str})
    regional_summary = pd.read_csv(REGIONAL_SUMMARY, dtype={"year": str})
    all_india_acc(summary)
    regional_scorecard(regional)
    jjas_all_india_and_regions(summary, regional_summary)
    all_domains_scorecard(summary, regional)
    all_season_domain_summary(summary, regional)
    all_seasons_domain_line_plot(summary, regional_summary)
    fuxi_jjas_climatology_robustness(summary)
    fuxi_jjas_system_specific_headline(summary)
    print("wrote eight IMD briefing figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
