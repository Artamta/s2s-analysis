#!/usr/bin/env python3
"""Render a one-page, presentation-ready comparison of the two correctors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]

NAVY = "#16324F"
BLUE = "#2563EB"
BLUE_LIGHT = "#EAF2FF"
GREEN = "#15803D"
GREEN_LIGHT = "#EAF7EF"
PURPLE = "#7C3AED"
PURPLE_LIGHT = "#F3EEFF"
ORANGE = "#C65D08"
ORANGE_LIGHT = "#FFF4E5"
SLATE = "#475569"
SLATE_LIGHT = "#F1F5F9"
RED = "#991B1B"


def card(axis, x, y, width, height, face, edge, title, body, *, title_color=NAVY):
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=axis.transAxes, facecolor=face, edgecolor=edge,
            linewidth=1.15,
        )
    )
    axis.text(
        x + 0.025, y + height - 0.035, title, transform=axis.transAxes,
        ha="left", va="top", fontsize=10.5, fontweight="bold", color=title_color,
    )
    axis.text(
        x + 0.025, y + height - 0.080, body, transform=axis.transAxes,
        ha="left", va="top", fontsize=8.7, color="#1F2937", linespacing=1.30,
    )


def output_card(axis, x, y, width, face, edge, title, body):
    """Compact horizontal output strip for short cards."""
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, 0.055,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=axis.transAxes, facecolor=face, edgecolor=edge,
            linewidth=1.15,
        )
    )
    axis.text(x + 0.020, y + 0.028, title, transform=axis.transAxes,
              ha="left", va="center", fontsize=9.8, fontweight="bold", color=GREEN)
    axis.text(x + 0.115, y + 0.028, body, transform=axis.transAxes,
              ha="left", va="center", fontsize=8.45, color="#1F2937")


def arrow(axis, x1, y1, x2, y2, color):
    axis.annotate(
        "", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8},
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(15.4, 9.0), facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_axis_off()

    figure.text(0.5, 0.965, "Two ways to correct FuXi weekly rainfall", ha="center",
                va="top", fontsize=24, fontweight="bold", color=NAVY)
    figure.text(
        0.5, 0.927,
        "Same FuXi forecast and same IMERG training target. The difference is fixed bias lookup versus event-dependent learned residual.",
        ha="center", va="top", fontsize=11.5, color=SLATE,
    )

    # Shared training rule.
    axis.add_patch(
        FancyBboxPatch(
            (0.055, 0.835), 0.89, 0.060,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=axis.transAxes, facecolor="#FFF7ED", edgecolor="#FDBA74",
            linewidth=1.1,
        )
    )
    axis.text(0.075, 0.865, "TRAINING ONLY (2020–2022)", transform=axis.transAxes,
              ha="left", va="center", fontsize=10.2, fontweight="bold", color=ORANGE)
    axis.text(
        0.30, 0.865,
        "FuXi forecasts + matching IMERG weekly rainfall teach both correctors.  2023 selects/checks; 2024 is reserved.",
        transform=axis.transAxes, ha="left", va="center", fontsize=10.2, color="#3F3F46",
    )

    # Column containers.
    for x, title, color, subtitle in (
        (0.055, "A   Standard log-bias correction", BLUE, "A fixed, transparent correction table"),
        (0.515, "B   Neural temporal adapter", GREEN, "A small event-dependent residual model"),
    ):
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.140), 0.43, 0.655,
                boxstyle="round,pad=0.014,rounding_size=0.020",
                transform=axis.transAxes, facecolor="white", edgecolor="#CBD5E1",
                linewidth=1.2,
            )
        )
        axis.text(x + 0.025, 0.765, title, transform=axis.transAxes, ha="left",
                  va="center", fontsize=15, fontweight="bold", color=color)
        axis.text(x + 0.025, 0.735, subtitle, transform=axis.transAxes, ha="left",
                  va="center", fontsize=9.7, color=SLATE)

    # Baseline: left.
    card(
        axis, 0.080, 0.570, 0.380, 0.140, BLUE_LIGHT, "#93C5FD",
        "Inference inputs", "• Raw FuXi ensemble-mean weekly rain\n• Forecast lead week, verification-month, grid-cell location", title_color=BLUE,
    )
    card(
        axis, 0.080, 0.390, 0.380, 0.145, ORANGE_LIGHT, "#FDBA74",
        "Fitted quantity (not a neural network)",
        "Fixed lead × month × grid-cell log-rain bias table.\n6 × 12 × 174 = 12,528 active coefficients;\nshrinkage = 10 toward each lead's India-wide mean bias.",
        title_color=ORANGE,
    )
    card(
        axis, 0.080, 0.250, 0.380, 0.105, SLATE_LIGHT, "#CBD5E1",
        "Operation", "b = exp[log(1 + FuXi) + fixed bias] − 1\nSame correction for every event in the same lead/month/cell bin.", title_color=NAVY,
    )
    output_card(axis, 0.080, 0.165, 0.380, GREEN_LIGHT, "#86EFAC", "Output", "b: deterministic, non-negative weekly rainfall map (mm day⁻¹)")
    arrow(axis, 0.270, 0.568, 0.270, 0.538, BLUE)
    arrow(axis, 0.270, 0.388, 0.270, 0.358, ORANGE)
    arrow(axis, 0.270, 0.248, 0.270, 0.222, NAVY)

    # Neural: right.
    card(
        axis, 0.540, 0.555, 0.380, 0.155, BLUE_LIGHT, "#93C5FD",
        "Inference inputs: 11 channels × 6 weeks × 27 × 27", 
        "Rain: log FuXi mean, log spread, fixed IMERG climatology, FuXi anomaly\nContext: latitude, longitude, season sin/cos, lead, India-support mask\nAtmosphere: FuXi weekly 2-m temperature", title_color=BLUE,
    )
    card(
        axis, 0.540, 0.390, 0.380, 0.135, PURPLE_LIGHT, "#C4B5FD",
        "Trainable model", 
        "Two-level U-Net (16 → 32 → 64 channels) + one 4-head Transformer layer across six lead weeks.\n144,689 weights per seed; three independently trained seeds are averaged.", title_color=PURPLE,
    )
    card(
        axis, 0.540, 0.250, 0.380, 0.100, SLATE_LIGHT, "#CBD5E1",
        "Learned output", 
        "z: standardized log-rain residual map.\nW1–2: z = 0. W3–6: z is learned from the current forecast event.", title_color=NAVY,
    )
    output_card(axis, 0.540, 0.165, 0.380, GREEN_LIGHT, "#86EFAC", "Final", "ŷ = exp[log(1 + b) + residual(z)] − 1: deterministic rain")
    arrow(axis, 0.730, 0.553, 0.730, 0.528, BLUE)
    arrow(axis, 0.730, 0.388, 0.730, 0.353, PURPLE)
    arrow(axis, 0.730, 0.248, 0.730, 0.222, NAVY)

    # Key distinction and GT note.
    axis.add_patch(
        FancyBboxPatch(
            (0.055, 0.062), 0.89, 0.052,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=axis.transAxes, facecolor="#FEF2F2", edgecolor="#FCA5A5",
            linewidth=1.1,
        )
    )
    axis.text(0.075, 0.088, "CRITICAL INFERENCE RULE", transform=axis.transAxes,
              ha="left", va="center", fontsize=8.9, fontweight="bold", color=RED)
    axis.text(
        0.275, 0.088,
        "Neither model sees current IMERG at forecast time. IMERG is ground truth for fit/evaluation; only fixed 2001–2019 IMERG climatology is an input.",
        transform=axis.transAxes, ha="left", va="center", fontsize=8.45, color="#3F3F46",
    )

    figure.text(
        0.5, 0.012,
        "Both methods predict six non-overlapping 7-day mean precipitation maps (W1 D0–6 … W6 D35–41). Green model shown: 11-channel hybrid v3, not the optional 14-channel distribution ablation.",
        ha="center", va="bottom", fontsize=8.6, color=SLATE,
    )

    stem = ROOT / "neural_adapter_baseline_vs_neural"
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(
        stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white",
        metadata={
            "Title": "FuXi baseline versus neural adapter",
            "Author": "FuXi-IMERG neural adapter project",
            "Subject": "Model inputs, fitted quantities, and outputs",
            "CreationDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
            "ModDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
        },
    )
    plt.close(figure)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
