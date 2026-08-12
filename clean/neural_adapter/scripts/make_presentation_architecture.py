#!/usr/bin/env python3
"""Create a simple presentation diagram for the implemented v3 hybrid adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]

INK = "#16324F"
MUTED = "#52677D"
BLUE = "#2563EB"
BLUE_LIGHT = "#EAF2FF"
CYAN = "#0E7490"
CYAN_LIGHT = "#E6F7FB"
PURPLE = "#7C3AED"
PURPLE_LIGHT = "#F3EEFF"
GREEN = "#15803D"
GREEN_LIGHT = "#EAF7EF"
AMBER = "#C65D08"
AMBER_LIGHT = "#FFF4E5"
GREY_LIGHT = "#F1F5F9"
RED = "#991B1B"


def card(ax, x, y, w, h, face, edge, title, body, *, color=INK, title_size=12, body_size=9.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        facecolor=face, edgecolor=edge, linewidth=1.35,
    ))
    ax.text(x + 0.02, y + h - 0.042, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=title_size, fontweight="bold", color=color)
    ax.text(x + 0.02, y + h - 0.102, body, transform=ax.transAxes,
            ha="left", va="top", fontsize=body_size, color="#1F2937", linespacing=1.30)


def arrow(ax, x1, y1, x2, y2, color=INK, *, lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw,
                            "shrinkA": 0, "shrinkB": 0})


def main() -> None:
    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    fig.text(0.5, 0.962, "FuXi–IMERG neural adapter: implemented architecture", ha="center",
             va="top", fontsize=24, fontweight="bold", color=INK)
    fig.text(0.5, 0.922, "A small deterministic residual corrector for six weekly precipitation forecasts over India",
             ha="center", va="top", fontsize=11.5, color=MUTED)

    # Data discipline is deliberately visible on a meeting slide.
    for x, w, face, edge, heading, detail, color in (
        (.045, .430, GREEN_LIGHT, "#86EFAC", "TRAIN · 2020–2022", "fit features, log-bias anchor, residual scale and network", GREEN),
        (.485, .235, AMBER_LIGHT, "#FDBA74", "DEVELOP · 2023", "choose checkpoint / model", AMBER),
        (.730, .225, GREY_LIGHT, "#CBD5E1", "LOCKED TEST · 2024", "not used for any choice", MUTED),
    ):
        ax.add_patch(FancyBboxPatch((x, .838), w, .052, transform=ax.transAxes,
                     boxstyle="round,pad=0.010,rounding_size=0.016", facecolor=face, edgecolor=edge, linewidth=1.1))
        ax.text(x+.016, .864, heading, transform=ax.transAxes, ha="left", va="center", fontsize=9.1, fontweight="bold", color=color)
        ax.text(x+w-.016, .864, detail, transform=ax.transAxes, ha="right", va="center", fontsize=8.1, color=color)

    ax.text(.045, .797, "Forecast-time flow", transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=INK)
    ax.text(.955, .797, "Each case outputs W1 D0–6 … W6 D35–41", transform=ax.transAxes,
            fontsize=9.4, ha="right", color=MUTED)

    # Principal forecast-time path.
    card(ax, .045, .470, .205, .275, BLUE_LIGHT, "#93C5FD", "1. Input tensor",
         "x ∈ R[B, 6, 11, 27, 27]\n\nRain: FuXi mean/spread, fixed IMERG climo, FuXi anomaly\nContext: lat/lon, season, lead, India mask\nAtmosphere: weekly FuXi T2M",
         color=BLUE, title_size=12.4, body_size=8.2)
    card(ax, .290, .470, .192, .275, CYAN_LIGHT, "#67C6D6", "2. Spatial U-Net",
         "Same 2-D U-Net for every lead week\n\n16 → 32 → 64 channels\n2 × downsample + skip connections\n64 → 32 → 16 decoder\n\nLearns spatial rainfall structure",
         color=CYAN, title_size=12.4, body_size=8.7)
    card(ax, .522, .470, .178, .275, PURPLE_LIGHT, "#C4B5FD", "3. Temporal mixing",
         "At the 6×6 bottleneck:\n\n6 lead tokens per grid cell\n1 Transformer layer\n4 attention heads · d = 64\nlearned lead position\n\nMixes information across leads",
         color=PURPLE, title_size=12.4, body_size=8.7)
    card(ax, .740, .470, .215, .275, GREEN_LIGHT, "#86EFAC", "4. Anchored output",
         "Network predicts z: standardized log residual\n\nW1–2: gate = 0 → exact log-bias baseline\nW3–6: gate = 1 → learned correction\n\nŷ = expm1[log(1+b) + s·g·z]",
         color=GREEN, title_size=12.4, body_size=8.6)
    arrow(ax, .250, .607, .290, .607, BLUE)
    arrow(ax, .482, .607, .522, .607, CYAN)
    arrow(ax, .700, .607, .740, .607, PURPLE)

    # Log-bias is a baseline stream, not an output leakage source.
    card(ax, .045, .265, .330, .130, AMBER_LIGHT, "#FDBA74", "Training-only log-bias anchor b",
         "Fixed lead × month × grid-cell log-bias table: 12,528 active coefficients.\nInference uses FuXi rain, lead, month and location—not current IMERG.",
         color=AMBER, title_size=11.2, body_size=7.7)
    arrow(ax, .148, .470, .148, .397, AMBER, lw=1.6)
    arrow(ax, .375, .333, .740, .520, AMBER, lw=1.7)
    # Model capacity and physical output are explicit.
    card(ax, .415, .265, .260, .130, GREY_LIGHT, "#CBD5E1", "Small model, averaged robustly",
         "144,689 weights per seed; three independently trained seeds are averaged.\n11-channel hybrid v3 (not the optional 14-channel distribution ablation).",
         color=INK, title_size=11.2, body_size=7.7)
    card(ax, .715, .265, .240, .130, GREEN_LIGHT, "#86EFAC", "Physical forecast",
         "Deterministic, non-negative precipitation in mm day⁻¹.\nIMERG is ground truth only—not an operational input.",
         color=GREEN, title_size=11.2, body_size=7.9)

    ax.text(.045, .207, "How it is trained", transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=INK)
    card(ax, .045, .040, .440, .135, "#F8FAFC", "#CBD5E1", "Target: residual relative to the strong baseline",
         "z* = [log(1 + IMERG) − log(1 + b)] / s, where s is a per-lead training-only residual scale.\nThe model learns what log-bias cannot: case-specific spatial and cross-lead adjustments.",
         color=INK, title_size=10.6, body_size=7.7)
    card(ax, .515, .040, .265, .135, PURPLE_LIGHT, "#C4B5FD", "Area-weighted objective",
         "0.75 Smooth-L1 + 0.20 physical-space spatial ACC + 0.05 bias².\nOnly W3–6 receive gradients: lead weights [0, 0, .25, .25, .25, .25].",
         color=PURPLE, title_size=10.6, body_size=7.45)
    card(ax, .810, .040, .145, .135, "#FEF2F2", "#FCA5A5", "Interpretation",
         "2023 is development validation—not final evidence.\nA locked 2024 test is required for the paper claim.",
         color=RED, title_size=10.6, body_size=7.35)

    out = ROOT / "neural_adapter_architecture_presentation"
    fig.savefig(out.with_suffix(".png"), dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", metadata={
        "Title": "FuXi-IMERG neural adapter architecture", "Author": "neural adapter project",
        "Subject": "Implemented hybrid v3 architecture", "CreationDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
    })
    print(out.with_suffix(".png"))
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
