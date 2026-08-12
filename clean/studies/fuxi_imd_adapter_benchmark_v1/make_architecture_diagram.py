#!/usr/bin/env python3
"""Create a slide-ready architecture diagram for the selected FuXi–IMD adapter."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "presentation"

NAVY = "#18324A"
BLUE = "#28658C"
TEAL = "#2B847C"
ORANGE = "#D8753D"
GREEN = "#3A7D66"
SLATE = "#5C6B78"
PALE_BLUE = "#EEF5F9"
PALE_TEAL = "#EDF7F5"
PALE_ORANGE = "#FCF2EA"
PALE_GREEN = "#EDF6F1"
PALE_GREY = "#F5F7F9"
LINE = "#CBD5DD"
WHITE = "#FFFFFF"


def box(
    ax,
    x,
    y,
    width,
    height,
    *,
    facecolor=WHITE,
    edgecolor=LINE,
    linewidth=1.2,
    radius=0.12,
    zorder=2,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=SLATE,
    linewidth=1.5,
    style="-|>",
    connectionstyle="arc3",
    linestyle="-",
    mutation_scale=11,
    zorder=4,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        mutation_scale=mutation_scale,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, color=SLATE):
    ax.text(
        x,
        y,
        text,
        fontsize=8.4,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        zorder=6,
    )


def card(ax, x, y, width, height, title, lines, *, color, facecolor):
    box(ax, x, y, width, height, facecolor=facecolor, edgecolor=color, linewidth=1.1)
    ax.text(
        x + 0.16,
        y + height - 0.20,
        title,
        fontsize=9.5,
        fontweight="bold",
        color=color,
        ha="left",
        va="top",
        zorder=6,
    )
    ax.text(
        x + 0.16,
        y + height - 0.50,
        "\n".join(lines),
        fontsize=8.15,
        color=NAVY,
        ha="left",
        va="top",
        linespacing=1.25,
        zorder=6,
    )


def stage(ax, x, number, title, color):
    ax.text(
        x,
        7.78,
        number,
        fontsize=8.5,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
    )
    ax.text(
        x + 0.34,
        7.78,
        title,
        fontsize=9.1,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )


def network_block(ax, x, y, width, height, title, subtitle, *, facecolor, edgecolor):
    box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.3,
        radius=0.08,
        zorder=5,
    )
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        fontsize=8.4,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + height * 0.29,
        subtitle,
        fontsize=7.1,
        color=SLATE,
        ha="center",
        va="center",
        zorder=7,
    )


def draw() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(16, 9), facecolor=WHITE)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        0.55,
        8.62,
        "FuXi–IMD spatiotemporal residual adapter",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )
    ax.text(
        0.57,
        8.25,
        "Selected regularized architecture  ·  six weekly leads  ·  2.54 million trainable parameters",
        fontsize=10.5,
        color=SLATE,
        ha="left",
        va="center",
    )
    box(ax, 13.40, 8.25, 2.02, 0.46, facecolor=PALE_GREEN, edgecolor=GREEN, radius=0.16)
    ax.text(
        14.41,
        8.48,
        "3-SEED ENSEMBLE",
        fontsize=8.7,
        fontweight="bold",
        color=GREEN,
        ha="center",
        va="center",
    )

    stage(ax, 0.58, "01", "INPUTS & PREPROCESSING", BLUE)
    stage(ax, 4.36, "02", "SPATIOTEMPORAL ADAPTER", TEAL)
    stage(ax, 11.78, "03", "PHYSICAL RECONSTRUCTION", ORANGE)

    # Input panel.
    box(ax, 0.50, 3.25, 3.45, 4.28, facecolor=PALE_GREY, edgecolor=LINE)
    card(
        ax,
        0.72,
        6.25,
        3.01,
        1.02,
        "FuXi-S2S predictors",
        ["TP ensemble mean + spread", "weekly 2-m temperature"],
        color=BLUE,
        facecolor=PALE_BLUE,
    )
    card(
        ax,
        0.72,
        5.08,
        3.01,
        1.02,
        "IMD reference context",
        ["calendar-day climatology", "FuXi − IMD climatology anomaly"],
        color=TEAL,
        facecolor=PALE_TEAL,
    )
    card(
        ax,
        0.72,
        3.90,
        3.01,
        1.02,
        "Space, season & lead",
        ["latitude, longitude, sin/cos season", "lead index + valid India mask"],
        color=SLATE,
        facecolor=WHITE,
    )
    box(ax, 0.72, 3.43, 3.01, 0.30, facecolor=NAVY, edgecolor=NAVY, radius=0.12)
    ax.text(
        2.225,
        3.58,
        "STANDARDIZE PER LEAD  •  LOG1P RAINFALL",
        fontsize=7.45,
        fontweight="bold",
        color=WHITE,
        ha="center",
        va="center",
    )
    ax.text(
        0.72,
        3.08,
        "Prepared: [B, 6, 29, 27, 27]  ·  selected backbone uses channels 1–11",
        fontsize=7.2,
        color=SLATE,
        ha="left",
        va="center",
    )

    # Training-only anchor.
    box(ax, 0.50, 2.13, 3.45, 0.72, facecolor=PALE_ORANGE, edgecolor=ORANGE)
    ax.text(
        0.72,
        2.59,
        "TRAINING-ONLY LOG-BIAS ANCHOR",
        fontsize=8.5,
        fontweight="bold",
        color=ORANGE,
        ha="left",
        va="center",
    )
    ax.text(
        0.72,
        2.31,
        r"lead × initialization month × grid  →  baseline  $P_{LB}$",
        fontsize=8.1,
        color=NAVY,
        ha="left",
        va="center",
    )

    arrow(ax, (3.95, 5.30), (4.38, 5.30), color=BLUE, linewidth=2.0)

    # Model panel.
    box(ax, 4.35, 3.25, 7.00, 4.28, facecolor=PALE_TEAL, edgecolor=TEAL, linewidth=1.4)
    ax.text(
        4.62,
        7.26,
        "Two-scale temporal U-Net",
        fontsize=12.0,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )
    ax.text(
        10.98,
        7.26,
        "shared spatial weights across W1–W6",
        fontsize=7.5,
        color=TEAL,
        ha="right",
        va="center",
    )

    network_block(ax, 4.65, 5.12, 1.00, 0.76, "ENC 1", "48 × 27²", facecolor=WHITE, edgecolor=BLUE)
    network_block(ax, 6.00, 4.69, 1.05, 0.76, "ENC 2", "96 × 13²", facecolor=WHITE, edgecolor=BLUE)
    network_block(ax, 7.52, 4.28, 1.18, 0.82, "BOTTLENECK", "192 × 6²", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    network_block(ax, 9.14, 4.69, 1.05, 0.76, "DEC 2", "96 × 13²", facecolor=WHITE, edgecolor=TEAL)
    network_block(ax, 10.35, 5.12, 0.78, 0.76, "DEC 1", "48 × 27² → 1", facecolor=PALE_GREEN, edgecolor=GREEN)

    arrow(ax, (5.65, 5.42), (6.00, 5.07), color=SLATE)
    arrow(ax, (7.05, 4.92), (7.52, 4.69), color=SLATE)
    arrow(ax, (8.70, 4.69), (9.14, 5.07), color=SLATE)
    arrow(ax, (10.19, 5.07), (10.35, 5.42), color=SLATE)

    # Skip connections.
    ax.plot(
        [5.15, 5.15, 10.74],
        [5.90, 6.92, 6.92],
        color=BLUE,
        linewidth=1.2,
        linestyle="--",
        zorder=4,
    )
    arrow(
        ax,
        (10.74, 6.92),
        (10.74, 5.90),
        color=BLUE,
        linewidth=1.2,
        linestyle="--",
        mutation_scale=9,
    )
    arrow(
        ax,
        (6.52, 5.47),
        (9.67, 5.47),
        color=BLUE,
        linewidth=1.2,
        connectionstyle="arc3,rad=0",
        linestyle="--",
        mutation_scale=9,
    )

    # Temporal attention at both scales.
    box(ax, 5.77, 6.15, 1.48, 0.66, facecolor=WHITE, edgecolor=ORANGE, radius=0.10)
    ax.text(6.51, 6.55, "LEAD MIXER", fontsize=7.8, fontweight="bold", color=ORANGE, ha="center")
    ax.text(6.51, 6.30, "2 layers · 8 heads", fontsize=7.0, color=SLATE, ha="center")
    arrow(ax, (6.51, 6.15), (6.51, 5.47), color=ORANGE, linewidth=1.2)
    box(ax, 7.36, 5.58, 1.50, 0.66, facecolor=WHITE, edgecolor=ORANGE, radius=0.10)
    ax.text(8.11, 5.98, "LEAD MIXER", fontsize=7.8, fontweight="bold", color=ORANGE, ha="center")
    ax.text(8.11, 5.73, "3 layers · 8 heads", fontsize=7.0, color=SLATE, ha="center")
    arrow(ax, (8.11, 5.58), (8.11, 5.10), color=ORANGE, linewidth=1.2)

    ax.text(
        4.66,
        3.77,
        "Each convolution block: 2 × (3×3 Conv → GroupNorm → SiLU) → spatial dropout",
        fontsize=7.75,
        color=NAVY,
        ha="left",
        va="center",
    )
    ax.text(
        4.66,
        3.48,
        "Regularization: spatial dropout 0.30  ·  temporal dropout 0.25  ·  weight decay 3×10⁻³",
        fontsize=7.55,
        color=SLATE,
        ha="left",
        va="center",
    )
    box(ax, 9.06, 3.55, 2.02, 0.38, facecolor=WHITE, edgecolor=GREEN, radius=0.12)
    ax.text(
        10.07,
        3.74,
        "zero-initialized 1×1 residual head",
        fontsize=7.15,
        color=GREEN,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Residual and reconstruction.
    arrow(ax, (11.35, 5.30), (11.76, 5.30), color=TEAL, linewidth=2.0)
    box(ax, 11.78, 5.82, 3.72, 1.44, facecolor=PALE_GREEN, edgecolor=GREEN, linewidth=1.3)
    ax.text(
        11.98,
        6.98,
        "STANDARDIZED LOG-RESIDUAL",
        fontsize=8.6,
        fontweight="bold",
        color=GREEN,
        ha="left",
        va="center",
    )
    ax.text(
        13.64,
        6.51,
        r"$\widehat{r}\;\in\;\mathbb{R}^{6\,\times\,27\,\times\,27}$",
        fontsize=14,
        color=NAVY,
        ha="center",
        va="center",
    )
    ax.text(
        13.64,
        6.10,
        "mean of seeds 42, 43 and 44",
        fontsize=7.6,
        color=SLATE,
        ha="center",
        va="center",
    )
    arrow(ax, (13.64, 5.82), (13.64, 5.48), color=ORANGE, linewidth=1.5)
    box(ax, 11.78, 4.14, 3.72, 1.34, facecolor=PALE_ORANGE, edgecolor=ORANGE, linewidth=1.3)
    ax.text(
        11.98,
        5.20,
        "ANCHOR + RESIDUAL",
        fontsize=8.6,
        fontweight="bold",
        color=ORANGE,
        ha="left",
        va="center",
    )
    ax.text(
        13.64,
        4.73,
        r"$\widehat{P}=\mathrm{expm1}\!\left[\log(1+P_{LB})+\alpha\,s_\ell\widehat{r}\right]$",
        fontsize=11.5,
        color=NAVY,
        ha="center",
        va="center",
    )
    ax.text(
        13.64,
        4.36,
        r"validation-selected residual gate  $\alpha=1.0$",
        fontsize=7.6,
        color=SLATE,
        ha="center",
        va="center",
    )
    arrow(ax, (13.64, 4.14), (13.64, 3.83), color=ORANGE, linewidth=1.5)
    box(ax, 11.78, 3.25, 3.72, 0.58, facecolor=GREEN, edgecolor=GREEN, linewidth=1.3)
    ax.text(
        13.64,
        3.54,
        "IMD-CALIBRATED PRECIPITATION  ·  W1–W6  ·  mm day⁻¹",
        fontsize=7.55,
        fontweight="bold",
        color=WHITE,
        ha="center",
        va="center",
    )

    # Bias-anchor connection to reconstruction.
    ax.plot(
        [3.95, 11.56, 11.56],
        [2.49, 2.49, 4.55],
        color=ORANGE,
        linewidth=1.25,
        linestyle="--",
        zorder=3,
    )
    arrow(
        ax,
        (11.56, 4.55),
        (11.78, 4.55),
        color=ORANGE,
        linewidth=1.25,
        linestyle="--",
        mutation_scale=9,
        zorder=3,
    )
    ax.text(
        7.67,
        2.74,
        r"$P_{LB}$",
        fontsize=8.2,
        fontweight="bold",
        color=ORANGE,
        ha="center",
        va="center",
    )

    # Objective panel.
    box(ax, 0.50, 0.48, 15.00, 1.26, facecolor=PALE_GREY, edgecolor=LINE, linewidth=1.2)
    label(ax, 0.72, 1.50, "TRAINING TARGET", color=BLUE)
    ax.text(
        0.72,
        1.13,
        r"$r^*=\dfrac{\log(1+P_{IMD})-\log(1+P_{LB})}{s_\ell}$",
        fontsize=12.0,
        color=NAVY,
        ha="left",
        va="center",
    )
    ax.text(
        0.72,
        0.74,
        "sℓ: training-only RMS residual scale for each lead",
        fontsize=7.4,
        color=SLATE,
        ha="left",
        va="center",
    )

    ax.plot([4.55, 4.55], [0.66, 1.56], color=LINE, linewidth=1.0)
    label(ax, 4.82, 1.50, "COMPOSITE AREA-WEIGHTED LOSS", color=TEAL)
    ax.text(
        4.82,
        1.10,
        r"$\mathcal{L}=0.75\,\mathcal{L}_{Smooth\,L1}+0.20\,(1-ACC_{spatial})+0.05\,\mathcal{L}_{bias^2}$",
        fontsize=11.5,
        color=NAVY,
        ha="left",
        va="center",
    )
    ax.text(
        4.82,
        0.72,
        "equal W1–W6 weights  ·  IMD valid support  ·  early stopping on validation loss",
        fontsize=7.5,
        color=SLATE,
        ha="left",
        va="center",
    )

    ax.plot([11.46, 11.46], [0.66, 1.56], color=LINE, linewidth=1.0)
    label(ax, 11.72, 1.50, "DATA SPLIT", color=ORANGE)
    ax.text(
        11.72,
        1.13,
        "TRAIN  2002–2017  ·  560 cases",
        fontsize=8.3,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )
    ax.text(
        11.72,
        0.86,
        "VALIDATE  2018–2019  ·  70 cases",
        fontsize=8.3,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )
    ax.text(
        11.72,
        0.61,
        "June–September initializations",
        fontsize=7.3,
        color=SLATE,
        ha="left",
        va="center",
    )

    # A precise note about the unused candidate-bank features.
    ax.text(
        15.46,
        0.18,
        "The prepared tensor also contains 18 shifted-climatology candidate channels; they belong to the evaluated attention variant and are not consumed by the selected fixed-climatology backbone.",
        fontsize=6.6,
        color=SLATE,
        ha="right",
        va="center",
    )

    return figure


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = draw()
    stem = OUTPUT / "fuxi_imd_adapter_architecture"
    figure.savefig(stem.with_suffix(".png"), dpi=300, facecolor=WHITE)
    figure.savefig(stem.with_suffix(".pdf"), facecolor=WHITE)
    figure.savefig(stem.with_suffix(".svg"), facecolor=WHITE)
    plt.close(figure)
    print(f"Wrote {stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
