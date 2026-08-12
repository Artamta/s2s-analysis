#!/usr/bin/env python3
"""Build the meeting-ready scientific summary from saved validation artifacts.

This script is deliberately plot-only: it reads completed CSV/PNG artifacts and
does not import the training package, rescore forecasts, or access prediction
stores.  If ``neural_adapter_academic_metrics.png`` exists, it is used as Page
4; otherwise a compact results/conclusions page is generated from the saved
headline CSV files.
"""

from __future__ import annotations

import argparse
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


PAGE_SIZE = (13.333, 7.5)  # 16:9 meeting format
NAVY = "#17324D"
BLUE = "#2864B7"
GREEN = "#16845B"
ORANGE = "#D97706"
PURPLE = "#8B4A9C"
RED = "#A63D40"
INK = "#202832"
MUTED = "#5C6875"
PALE_BLUE = "#EAF2FA"
PALE_GREEN = "#EAF6F0"
PALE_ORANGE = "#FFF3E3"
GRID = "#D8DEE6"


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": NAVY,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _new_page() -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    fig.add_artist(Rectangle((0, 0.985), 1, 0.015, transform=fig.transFigure,
                             facecolor=NAVY, edgecolor="none"))
    return fig


def _footer(fig: plt.Figure, page: int, label: str) -> None:
    fig.text(0.035, 0.024, label, color=MUTED, fontsize=7.5, va="bottom")
    fig.text(0.965, 0.024, f"{page} / 4", color=MUTED, fontsize=7.5,
             va="bottom", ha="right")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.048, 0.048],
                             transform=fig.transFigure, color=GRID, lw=0.7))


def _wrapped(fig: plt.Figure, x: float, y: float, text: str, width: int,
             **kwargs: object) -> None:
    fig.text(x, y, textwrap.fill(text, width=width), **kwargs)


def _box(fig: plt.Figure, xy: tuple[float, float], width: float, height: float,
         face: str, edge: str = GRID, radius: float = 0.012) -> None:
    fig.add_artist(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            transform=fig.transFigure,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
        )
    )


def _cover_page() -> plt.Figure:
    fig = _new_page()
    fig.text(0.045, 0.914, "FUXI–IMERG NEURAL ADAPTER", color=GREEN,
             fontsize=10, fontweight="bold", va="top")
    fig.text(0.045, 0.865,
             "Deterministic subseasonal precipitation correction over India",
             color=NAVY, fontsize=24, fontweight="bold", va="top")
    fig.text(0.045, 0.800,
             "A bias-anchored spatial–temporal adapter · 2023 development validation",
             color=MUTED, fontsize=12.5, va="top")

    # Executive conclusion.
    _box(fig, (0.045, 0.646), 0.91, 0.105, PALE_GREEN, "#B8DCC8")
    fig.text(0.065, 0.723, "EXECUTIVE RESULT", color=GREEN, fontsize=9,
             fontweight="bold", va="top")
    _wrapped(
        fig,
        0.065,
        0.690,
        "At Weeks 3–6, the temporal adapter raises spatial ACC from 0.179 "
        "(raw FuXi) to 0.319 and lowers MAE from 2.364 to 2.108 mm day⁻¹. "
        "It improves the earlier spatial U-Net, but added ACC beyond a strong "
        "training-only log-bias correction remains inconclusive (+0.003; 95% "
        "block interval −0.014 to +0.025).",
        width=155,
        fontsize=10.5,
        va="top",
        linespacing=1.35,
    )

    # Left: data contract.
    fig.text(0.045, 0.603, "DATA AND EVALUATION CONTRACT", color=NAVY,
             fontsize=12, fontweight="bold", va="top")
    ax_table = fig.add_axes([0.045, 0.304, 0.43, 0.272])
    ax_table.axis("off")
    rows = [
        ("Forecast", "FuXi-S2S TP/T2M; 50 members"),
        ("Target / verifier", "GPM IMERG Final V07B precipitation"),
        ("Reference climate", "Fixed IMERG 2001–2019 climatology"),
        ("Train", "2020–2022 · 302 initializations"),
        ("Development", "2023 · 93 initializations"),
        ("Domain", "27 × 27 at 1.5° · 174 supported India cells"),
        ("Output", "Deterministic weekly mean · mm day⁻¹"),
    ]
    table = ax_table.table(
        cellText=rows,
        colLabels=("Component", "Frozen definition"),
        colWidths=(0.34, 0.66),
        cellLoc="left",
        colLoc="left",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.PAD = 0.055
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "white")
            if col == 0:
                cell.get_text().set_fontweight("bold")

    # Right: time-scale contract.
    fig.text(0.515, 0.603, "INFERENCE TIME SCALE", color=NAVY, fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.515, 0.570,
             "One forecast per twice-weekly initialization; six non-overlapping 7-day means",
             fontsize=9.2, color=MUTED, va="top")
    ax_t = fig.add_axes([0.515, 0.420, 0.44, 0.118])
    ax_t.set_xlim(0, 42)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")
    week_colors = ["#DCEAF7", "#CEE2F3", "#BFE4D2", "#ACE0C3", "#98D9B3", "#84D1A2"]
    for i in range(6):
        ax_t.add_patch(Rectangle((7 * i, 0.25), 7, 0.55,
                                 facecolor=week_colors[i], edgecolor="white", lw=1.5))
        ax_t.text(7 * i + 3.5, 0.60, f"W{i + 1}", ha="center", va="center",
                  color=NAVY, fontsize=9, fontweight="bold")
        ax_t.text(7 * i + 3.5, 0.38, f"D{7*i}–{7*i+6}", ha="center", va="center",
                  color=INK, fontsize=7.6)
    ax_t.text(0, 0.08, "Initialization", ha="left", va="center", color=MUTED, fontsize=7.5)
    ax_t.text(42, 0.08, "42-day horizon", ha="right", va="center", color=MUTED, fontsize=7.5)

    _box(fig, (0.515, 0.292), 0.44, 0.099, PALE_ORANGE, "#F1C98E")
    fig.text(0.533, 0.374, "INTERPRETATION", fontsize=8.5, color=ORANGE,
             fontweight="bold", va="top")
    _wrapped(
        fig,
        0.533,
        0.345,
        "The network does not infer daily rainfall. It jointly post-processes "
        "six weekly-mean precipitation-rate fields; learned corrections are active "
        "only at W3–W6 (D14–41). W1–W2 exactly retain log-bias correction.",
        width=90,
        fontsize=8.5,
        va="top",
        linespacing=1.25,
    )

    # Bottom callouts.
    _box(fig, (0.045, 0.102), 0.282, 0.135, PALE_BLUE, "#BDD2E8")
    _box(fig, (0.359, 0.102), 0.282, 0.135, PALE_GREEN, "#B8DCC8")
    _box(fig, (0.673, 0.102), 0.282, 0.135, "#F7F0FA", "#D9C2E0")
    callouts = [
        (0.065, BLUE, "+0.140 ACC", "vs raw FuXi · W3–W6\n95% CI +0.068 to +0.220"),
        (0.379, GREEN, "+0.027 ACC", "vs v2 spatial U-Net · W3–W6\n95% CI +0.006 to +0.047"),
        (0.693, PURPLE, "+0.003 ACC", "vs log-bias · W3–W6\n95% CI −0.014 to +0.025"),
    ]
    for x, color, main, sub in callouts:
        fig.text(x, 0.211, main, color=color, fontsize=15, fontweight="bold", va="top")
        fig.text(x, 0.164, sub, color=INK, fontsize=8.6, va="top", linespacing=1.35)

    _footer(fig, 1, "Development evidence · not a confirmatory or operational evaluation")
    return fig


def _image_page(image_path: Path, page: int, heading: str, subtitle: str,
                footer: str) -> plt.Figure:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    fig = _new_page()
    fig.text(0.045, 0.938, heading, color=NAVY, fontsize=16,
             fontweight="bold", va="top")
    fig.text(0.045, 0.895, subtitle, color=MUTED, fontsize=9.2, va="top")
    ax = fig.add_axes([0.035, 0.078, 0.93, 0.785])
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.axis("off")
    _footer(fig, page, footer)
    return fig


def _fallback_results_page(root: Path) -> plt.Figure:
    late = pd.read_csv(root / "v3_headline_weeks_3_to_6_india.csv").set_index("predictor")
    intervals = pd.read_csv(root / "v3_headline_bootstrap_weeks_3_to_6_india.csv")
    fig = _new_page()
    fig.text(0.045, 0.938, "Results, uncertainty and scientific interpretation",
             color=NAVY, fontsize=16, fontweight="bold", va="top")
    fig.text(0.045, 0.895,
             "India-wide 2023 development validation · W3–W6 = D14–41 · 93 initializations",
             color=MUTED, fontsize=9.2, va="top")

    labels = ["Raw FuXi", "Log-bias", "v2 U-Net", "v3 temporal"]
    keys = ["raw_fuxi", "log_bias_correction", "v2_residual_unet", "late_lead_temporal_unet"]
    colors = ["#737B86", BLUE, ORANGE, GREEN]
    x = np.arange(4)
    ax1 = fig.add_axes([0.055, 0.49, 0.405, 0.32])
    vals = late.loc[keys, "acc_mean"].to_numpy()
    ax1.bar(x, vals, color=colors, width=0.68)
    ax1.set_ylim(0, 0.37)
    ax1.set_ylabel("Spatial ACC")
    ax1.set_title("Pattern skill", loc="left", fontweight="bold")
    ax1.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    for i, value in enumerate(vals):
        ax1.text(i, value + 0.009, f"{value:.3f}", ha="center", fontsize=8,
                 fontweight="bold")

    ax2 = fig.add_axes([0.54, 0.49, 0.405, 0.32])
    vals = late.loc[keys, "mae_mean"].to_numpy()
    ax2.bar(x, vals, color=colors, width=0.68)
    ax2.set_ylim(1.95, 2.45)
    ax2.set_ylabel("MAE (mm day⁻¹)")
    ax2.set_title("Absolute error (lower is better)", loc="left", fontweight="bold")
    ax2.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for i, value in enumerate(vals):
        ax2.text(i, value + 0.014, f"{value:.3f}", ha="center", fontsize=8,
                 fontweight="bold")

    # Saved paired intervals, if table uses metric/baseline rows.
    ax3 = fig.add_axes([0.055, 0.145, 0.455, 0.235])
    acc = intervals.loc[intervals["metric"].eq("acc")].copy()
    order = ["raw_fuxi", "v2_residual_unet", "log_bias_correction"]
    acc = acc.set_index("baseline").loc[order].reset_index()
    y = np.arange(3)[::-1]
    mean = acc["mean_difference"].to_numpy()
    lo = acc["ci_lower"].to_numpy()
    hi = acc["ci_upper"].to_numpy()
    ax3.axvline(0, color=MUTED, lw=0.9)
    ax3.errorbar(mean, y, xerr=np.vstack([mean - lo, hi - mean]), fmt="o",
                 color=GREEN, ecolor=GREEN, capsize=3, lw=1.4)
    ax3.set_yticks(y, ["vs raw FuXi", "vs v2 U-Net", "vs log-bias"])
    ax3.set_xlabel("Paired ΔACC (v3 − baseline)")
    ax3.set_title("13-initialization moving-block 95% intervals", loc="left",
                  fontsize=10, fontweight="bold")
    ax3.grid(axis="x", color=GRID, lw=0.6)
    ax3.set_axisbelow(True)

    _box(fig, (0.555, 0.125), 0.40, 0.27, "#F7F9FB", GRID)
    fig.text(0.575, 0.365, "SCIENTIFIC READING", color=NAVY, fontsize=10,
             fontweight="bold", va="top")
    bullets = [
        "Clear added value over raw FuXi and the earlier spatial U-Net.",
        "Neural ACC added value beyond log-bias is not statistically resolved.",
        "MAE improves modestly vs log-bias (−0.027 mm day⁻¹), but dry bias worsens.",
        "Next evidence needed: year-held-out hindcasts, IMD verification, MOS/XGBoost and probabilistic baselines.",
    ]
    y_text = 0.326
    for bullet in bullets:
        fig.text(0.578, y_text, "•", color=GREEN, fontsize=11, va="top")
        _wrapped(fig, 0.596, y_text, bullet, width=70, fontsize=8.5,
                 va="top", linespacing=1.25)
        y_text -= 0.060

    _footer(fig, 4, "Publication-safe conclusion: strong pilot; multi-year independent testing remains essential")
    return fig


def build(root: Path, output: Path, cover_png: Path) -> None:
    architecture = root / "neural_adapter_architecture.png"
    comparison = root / "neural_adapter_experiment_comparison.png"
    academic = root / "neural_adapter_academic_metrics.png"

    pages = [
        _cover_page(),
        _image_page(
            architecture,
            2,
            "Implemented architecture",
            "All six weekly fields are processed jointly; correction is anchored to training-only log-bias and active at W3–W6 (D14–41).",
            "Exact implemented model · deterministic nonnegative weekly precipitation output",
        ),
        _image_page(
            comparison,
            3,
            "Multi-experiment development comparison",
            "Inference target: six non-overlapping 7-day mean precipitation fields (W1 D0–6, W2 D7–13, W3 D14–20, W4 D21–27, W5 D28–34, W6 D35–41).",
            "2023 development validation · comparisons use saved, provenance-checked metrics",
        ),
    ]
    if academic.is_file():
        pages.append(
            _image_page(
                academic,
                4,
                "All-metric evaluation",
                "ACC, RMSE, MAE and bias by lead week · deterministic weekly mean precipitation · 2023 development validation.",
                "Results are development evidence, not a confirmatory test or operational scorecard",
            )
        )
    else:
        pages.append(_fallback_results_page(root))

    output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].savefig(cover_png, dpi=300, bbox_inches=None, facecolor="white")
    metadata = {
        "Title": "FuXi–IMERG deterministic precipitation neural adapter",
        "Author": "Neural Adapter Study",
        "Subject": "2023 development-validation meeting summary",
        "Keywords": "FuXi, IMERG, precipitation, subseasonal, India, neural adapter",
        "Creator": "make_meeting_summary.py",
        "Producer": "Matplotlib",
        "CreationDate": datetime(2026, 8, 7, 0, 0, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 8, 7, 0, 0, tzinfo=timezone.utc),
    }
    with PdfPages(output, metadata=metadata) as pdf:
        for page in pages:
            pdf.savefig(page, facecolor="white")
            plt.close(page)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path,
                        default=default_root / "neural_adapter_meeting_summary.pdf")
    parser.add_argument("--cover-png", type=Path,
                        default=default_root / "neural_adapter_meeting_summary_cover.png")
    return parser.parse_args()


def main() -> None:
    _configure_style()
    args = parse_args()
    build(args.root.resolve(), args.output.resolve(), args.cover_png.resolve())
    print(args.output.resolve())
    print(args.cover_png.resolve())


if __name__ == "__main__":
    main()
