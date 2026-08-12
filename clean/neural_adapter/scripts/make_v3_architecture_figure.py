#!/usr/bin/env python3
"""Render the implemented v3 FuXi--IMERG adapter as a publication-ready figure.

The diagram is deliberately code-generated and mirrors ``models.py``,
``v3_workflow.py``, ``anchored.py``, and the frozen v3 experiment configs.
It does not load predictions or alter any experiment artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#102A43"
MUTED = "#52677D"
FAINT = "#D8E2EC"
PAPER = "#F4F7FB"
WHITE = "#FFFFFF"
BLUE = "#2563EB"
BLUE_LIGHT = "#EAF2FF"
CYAN = "#0E7490"
CYAN_LIGHT = "#E6F7FB"
PURPLE = "#7C3AED"
PURPLE_LIGHT = "#F2ECFF"
GREEN = "#15803D"
GREEN_LIGHT = "#EAF8EF"
AMBER = "#B45309"
AMBER_LIGHT = "#FFF5DF"
RED = "#B42318"
RED_LIGHT = "#FFF0EE"
GREY_LIGHT = "#EEF2F6"


def _box(
    axis: Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = FAINT,
    radius: float = 0.14,
    linewidth: float = 1.2,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _arrow(
    axis: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.5,
    style: str = "-|>",
    connection: str = "arc3",
    linestyle: str = "-",
    mutation_scale: float = 11.0,
    zorder: int = 4,
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        connectionstyle=connection,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        mutation_scale=mutation_scale,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    axis.add_patch(arrow)
    return arrow


def _section_title(axis: Axes, x: float, y: float, number: str, title: str) -> None:
    axis.text(
        x,
        y,
        number,
        ha="left",
        va="center",
        color=WHITE,
        fontsize=8.8,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.30,rounding_size=0.7",
            "facecolor": INK,
            "edgecolor": "none",
        },
        zorder=8,
    )
    axis.text(
        x + 0.38,
        y,
        title,
        ha="left",
        va="center",
        color=INK,
        fontsize=10.6,
        fontweight="bold",
        zorder=8,
    )


def _pill(
    axis: Axes,
    x: float,
    y: float,
    text: str,
    *,
    face: str,
    color: str,
    width: float,
    height: float = 0.28,
    fontsize: float = 7.4,
) -> None:
    _box(axis, (x, y), width, height, face=face, edge=face, radius=0.12, linewidth=0.0)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=fontsize,
        fontweight="semibold",
        zorder=6,
    )


def _network_block(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    details: Iterable[str],
    *,
    face: str,
    edge: str,
) -> None:
    _box(axis, (x, y), width, height, face=face, edge=edge, radius=0.12, linewidth=1.35, zorder=3)
    axis.text(
        x + width / 2,
        y + height - 0.23,
        title,
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    axis.text(
        x + width / 2,
        y + height - 0.55,
        "\n".join(details),
        ha="center",
        va="top",
        fontsize=6.9,
        linespacing=1.28,
        color=MUTED,
        zorder=6,
    )


def _draw_split_strip(axis: Axes) -> None:
    x0, y0, total_width, height = 0.45, 7.74, 15.10, 0.48
    widths = (6.85, 4.05, 4.20)
    styles = (
        (GREEN_LIGHT, GREEN, "TRAIN · 2020–2022", "fit normalization, log-bias anchor, residual scales + weights"),
        (AMBER_LIGHT, AMBER, "DEVELOPMENT · 2023", "checkpoint selection + validation only"),
        (
            GREY_LIGHT,
            MUTED,
            "LOCKED · 2024",
            "not used for v3 fitting, selection or scoring",
        ),
    )
    cursor = x0
    for width, (face, color, heading, detail) in zip(widths, styles):
        _box(axis, (cursor, y0), width, height, face=face, edge=WHITE, radius=0.10, linewidth=1.5)
        axis.text(
            cursor + 0.18,
            y0 + height / 2,
            heading,
            ha="left",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color=color,
            zorder=6,
        )
        axis.text(
            cursor + width - 0.16,
            y0 + height / 2,
            detail,
            ha="right",
            va="center",
            fontsize=6.9,
            color=color,
            zorder=6,
        )
        cursor += width


def _draw_inputs(axis: Axes) -> None:
    x, y, width, height = 0.45, 2.22, 3.05, 5.22
    _box(axis, (x, y), width, height)
    _section_title(axis, x + 0.18, y + height - 0.32, "1", "INPUT TENSOR")

    _box(axis, (x + 0.18, y + height - 1.10), width - 0.36, 0.48, face=BLUE_LIGHT, edge="#B9D2FF", radius=0.09)
    axis.text(
        x + width / 2,
        y + height - 0.86,
        r"$\mathbf{x}\;\in\;\mathbb{R}^{B\,\times\,6\,\times\,C\,\times\,27\,\times\,27}$",
        ha="center",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color=BLUE,
        zorder=6,
    )
    axis.text(
        x + 0.20,
        y + height - 1.38,
        "Core C = 11 · one India map per lead week",
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="semibold",
        color=INK,
    )

    entries = (
        ("4", "precipitation", "log FuXi mean · log spread\nlog IMERG climo · FuXi log anomaly", BLUE_LIGHT, BLUE),
        ("1", "atmosphere", "weekly ensemble-mean T2M", CYAN_LIGHT, CYAN),
        ("6", "context", "latitude · longitude · sin/cos season\nlead week · India/IMERG support", GREY_LIGHT, MUTED),
    )
    top = y + height - 1.72
    entry_height = 0.62
    for index, (count, label, detail, face, color) in enumerate(entries):
        entry_y = top - index * 0.72 - entry_height
        _box(axis, (x + 0.18, entry_y), width - 0.36, entry_height, face=face, edge=face, radius=0.09, linewidth=0.0)
        axis.text(
            x + 0.37,
            entry_y + entry_height / 2,
            count,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
        axis.text(
            x + 0.63,
            entry_y + entry_height / 2 + 0.12,
            label.upper(),
            ha="left",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=color,
        )
        axis.text(
            x + 0.63,
            entry_y + entry_height / 2 - 0.10,
            detail,
            ha="left",
            va="center",
            fontsize=6.7,
            linespacing=1.30,
            color=INK,
        )

    optional_y = y + 0.48
    _box(
        axis,
        (x + 0.18, optional_y),
        width - 0.36,
        0.94,
        face=PURPLE_LIGHT,
        edge="#DCCBFF",
        radius=0.10,
        linewidth=1.0,
    )
    _pill(
        axis,
        x + 0.31,
        optional_y + 0.58,
        "OPTIONAL ABLATION  +3",
        face="#E3D6FF",
        color=PURPLE,
        width=1.57,
        height=0.25,
        fontsize=6.7,
    )
    axis.text(
        x + 0.31,
        optional_y + 0.35,
        "member log-median anomaly · member log-IQR",
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK,
    )
    axis.text(
        x + 0.31,
        optional_y + 0.15,
        "P(member TP > IMERG climatology)  →  C = 14",
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK,
    )
    axis.text(
        x + width / 2,
        y + 0.20,
        "All learned moments are fitted on training years only",
        ha="center",
        va="center",
        fontsize=6.5,
        color=MUTED,
        style="italic",
    )


def _draw_network(axis: Axes) -> None:
    x, y, width, height = 3.70, 2.22, 7.86, 5.22
    _box(axis, (x, y), width, height)
    _section_title(axis, x + 0.18, y + height - 0.32, "2", "SHARED SPATIAL U-NET + TEMPORAL MIXING")
    axis.text(
        x + width - 0.18,
        y + height - 0.32,
        "same convolution weights for all 6 leads",
        ha="right",
        va="center",
        fontsize=6.9,
        color=MUTED,
        style="italic",
    )

    block_width, block_height = 1.03, 1.36
    e1 = (x + 0.25, y + 2.50)
    e2 = (x + 1.60, y + 1.58)
    bn = (x + 2.95, y + 0.73)
    transformer = (x + 4.15, y + 0.64)
    d2 = (x + 5.52, y + 1.58)
    d1 = (x + 6.75, y + 2.50)

    _network_block(
        axis,
        *e1,
        block_width,
        block_height,
        "ENCODER 1",
        ("2 × 3×3 Conv", "GroupNorm · SiLU", "16 × 27 × 27"),
        face=BLUE_LIGHT,
        edge="#9FC0FA",
    )
    _network_block(
        axis,
        *e2,
        block_width,
        block_height,
        "ENCODER 2",
        ("2 × 3×3 Conv", "GroupNorm · SiLU", "32 × 13 × 13"),
        face=CYAN_LIGHT,
        edge="#9FDCE5",
    )
    _network_block(
        axis,
        *bn,
        block_width,
        block_height,
        "BOTTLENECK",
        ("2 × 3×3 Conv", "GroupNorm · SiLU", "64 × 6 × 6"),
        face=PURPLE_LIGHT,
        edge="#C8B1F8",
    )
    _network_block(
        axis,
        *transformer,
        1.18,
        1.54,
        "TEMPORAL",
        ("6 lead tokens / cell", "4-head Transformer", "1 layer · d=64", "learned lead position"),
        face="#F7EDFF",
        edge=PURPLE,
    )
    _network_block(
        axis,
        *d2,
        block_width,
        block_height,
        "DECODER 2",
        ("bilinear ↑ + skip", "1×1 + ConvBlock", "32 × 13 × 13"),
        face=CYAN_LIGHT,
        edge="#9FDCE5",
    )
    _network_block(
        axis,
        *d1,
        0.90,
        block_height,
        "DECODER 1",
        ("bilinear ↑ + skip", "1×1 + ConvBlock", "16 × 27 × 27"),
        face=BLUE_LIGHT,
        edge="#9FC0FA",
    )

    def right_mid(block: Tuple[float, float], width_: float = block_width) -> Tuple[float, float]:
        return block[0] + width_, block[1] + block_height / 2

    def left_mid(block: Tuple[float, float]) -> Tuple[float, float]:
        return block[0], block[1] + block_height / 2

    _arrow(axis, right_mid(e1), left_mid(e2), color=BLUE)
    _arrow(axis, right_mid(e2), left_mid(bn), color=CYAN)
    _arrow(
        axis,
        right_mid(bn),
        (transformer[0], transformer[1] + 1.54 / 2),
        color=PURPLE,
    )
    _arrow(
        axis,
        (transformer[0] + 1.18, transformer[1] + 1.54 / 2),
        left_mid(d2),
        color=PURPLE,
    )
    _arrow(axis, right_mid(d2), left_mid(d1), color=CYAN)

    axis.text(x + 1.48, y + 2.50, "2×2\nmax-pool", ha="center", va="center", fontsize=6.1, color=MUTED)
    axis.text(x + 2.83, y + 1.59, "2×2\nmax-pool", ha="center", va="center", fontsize=6.1, color=MUTED)
    axis.text(x + 5.40, y + 1.60, "bilinear\nresize", ha="center", va="center", fontsize=6.1, color=MUTED)
    axis.text(x + 6.65, y + 2.50, "bilinear\nresize", ha="center", va="center", fontsize=6.1, color=MUTED)

    # Skip paths are drawn above the blocks to make the U shape explicit.
    _arrow(
        axis,
        (e1[0] + block_width * 0.66, e1[1] + block_height),
        (d1[0] + 0.31, d1[1] + block_height),
        color="#7C93AB",
        linewidth=1.15,
        connection="arc3,rad=-0.24",
        linestyle="--",
        mutation_scale=9,
        zorder=2,
    )
    _arrow(
        axis,
        (e2[0] + block_width * 0.66, e2[1] + block_height),
        (d2[0] + 0.37, d2[1] + block_height),
        color="#7C93AB",
        linewidth=1.15,
        connection="arc3,rad=-0.22",
        linestyle="--",
        mutation_scale=9,
        zorder=2,
    )
    axis.text(x + 3.95, y + 4.36, "skip connections preserve local spatial detail", ha="center", va="center", fontsize=6.6, color=MUTED)

    # Residual head and the six lead outputs.
    head_x, head_y = x + 7.69, y + 3.00
    axis.add_patch(Rectangle((head_x, head_y), 0.10, 0.40, facecolor=GREEN, edgecolor=GREEN, zorder=5))
    _arrow(axis, (d1[0] + 0.90, d1[1] + block_height / 2), (head_x, head_y + 0.20), color=BLUE)
    axis.text(
        head_x + 0.05,
        head_y + 0.54,
        "1×1 zero-init head",
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=GREEN,
        rotation=90,
    )
    axis.text(
        x + width / 2,
        y + 0.25,
        "Lead weeks fold into the batch for 2-D convolutions; only bottleneck attention mixes time",
        ha="center",
        va="center",
        fontsize=6.7,
        color=MUTED,
        style="italic",
    )


def _draw_anchor(axis: Axes) -> None:
    x, y, width, height = 11.76, 2.22, 3.79, 5.22
    _box(axis, (x, y), width, height)
    _section_title(axis, x + 0.18, y + height - 0.32, "3", "ANCHORED CORRECTION")

    _box(axis, (x + 0.18, y + height - 1.22), width - 0.36, 0.63, face=GREEN_LIGHT, edge="#B4DFC2", radius=0.09)
    axis.text(
        x + width / 2,
        y + height - 0.91,
        "b = training-only log-bias-corrected FuXi",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=GREEN,
    )
    axis.text(
        x + width / 2,
        y + height - 1.45,
        r"target  $z_l^* = [\log(1+y_l)-\log(1+b_l)]\,/\,s_l$",
        ha="center",
        va="center",
        fontsize=7.9,
        color=INK,
    )
    axis.text(
        x + width / 2,
        y + height - 1.72,
        r"network predicts standardized log residual  $z_l$",
        ha="center",
        va="center",
        fontsize=7.1,
        color=MUTED,
    )

    axis.text(x + 0.18, y + height - 2.06, "EXACT LEAD GATE", ha="left", va="center", fontsize=7.0, fontweight="bold", color=INK)
    gate_x, gate_y = x + 0.18, y + height - 2.58
    gap = 0.05
    gate_width = (width - 0.36 - 5 * gap) / 6
    for lead in range(1, 7):
        active = lead >= 3
        face, color = (GREEN_LIGHT, GREEN) if active else (GREY_LIGHT, MUTED)
        _box(axis, (gate_x, gate_y), gate_width, 0.39, face=face, edge=face, radius=0.06, linewidth=0.0)
        axis.text(
            gate_x + gate_width / 2,
            gate_y + 0.195,
            f"W{lead}\n{'1' if active else '0'}",
            ha="center",
            va="center",
            fontsize=6.6,
            fontweight="bold",
            linespacing=0.95,
            color=color,
        )
        gate_x += gate_width + gap

    equation_y = y + 1.40
    _box(axis, (x + 0.18, equation_y), width - 0.36, 1.12, face="#F8FAFC", edge=FAINT, radius=0.09)
    axis.text(
        x + width / 2,
        equation_y + 0.79,
        r"$\widehat{y}_l = \operatorname{expm1}($",
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        x + width / 2,
        equation_y + 0.48,
        r"$\operatorname{clip}\{\log(1+b_l)+s_l\,g_l\,z_l,\ 0,\ 20\})$",
        ha="center",
        va="center",
        fontsize=8.8,
        color=INK,
    )
    axis.text(
        x + width / 2,
        equation_y + 0.18,
        r"$g=(0,0,1,1,1,1)$  ·  $s_l$ fitted on train only",
        ha="center",
        va="center",
        fontsize=7.0,
        color=MUTED,
    )

    _box(axis, (x + 0.18, y + 0.28), width - 0.36, 0.84, face=GREEN_LIGHT, edge="#B4DFC2", radius=0.10)
    axis.text(
        x + width / 2,
        y + 0.87,
        "W1–2: bit-exact baseline  ·  W3–6: active correction",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=GREEN,
    )
    axis.text(
        x + width / 2,
        y + 0.54,
        "deterministic weekly precipitation · mm day⁻¹ · finite · ≥ 0",
        ha="center",
        va="center",
        fontsize=6.8,
        color=INK,
    )

    # Connect the U-Net head into the residual/anchor panel.
    _arrow(axis, (11.49, 5.42), (x + 0.10, 5.42), color=GREEN, linewidth=2.0, mutation_scale=12)
    axis.text(11.64, 5.60, "z", ha="center", va="bottom", fontsize=7.3, color=GREEN, fontweight="bold")


def _draw_loss_footer(axis: Axes) -> None:
    x, y, width, height = 0.45, 0.35, 15.10, 1.55
    _box(axis, (x, y), width, height)
    _section_title(axis, x + 0.18, y + height - 0.30, "4", "AREA-WEIGHTED TRAINING OBJECTIVE")

    loss_x, loss_y, loss_w, loss_h = x + 0.18, y + 0.18, 8.05, 0.76
    _box(axis, (loss_x, loss_y), loss_w, loss_h, face="#F8FAFC", edge=FAINT, radius=0.09)
    axis.text(
        loss_x + 0.16,
        loss_y + 0.49,
        r"$\mathcal{L} =$",
        ha="left",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
    )
    chunks = (
        ("0.75  Smooth-L1(z, z*)", BLUE),
        (" + 0.20  [1 − spatial ACC]", PURPLE),
        (" + 0.05  normalized bias²", GREEN),
    )
    positions = (loss_x + 0.82, loss_x + 3.22, loss_x + 5.85)
    for (text, color), position in zip(chunks, positions):
        axis.text(position, loss_y + 0.50, text, ha="left", va="center", fontsize=8.2, fontweight="bold", color=color)
    axis.text(
        loss_x + 0.82,
        loss_y + 0.20,
        "ACC uses physical precipitation anomalies relative to fixed IMERG climatology",
        ha="left",
        va="center",
        fontsize=6.7,
        color=MUTED,
    )

    lead_x, lead_w = loss_x + loss_w + 0.20, 3.10
    _box(axis, (lead_x, loss_y), lead_w, loss_h, face=GREEN_LIGHT, edge="#B4DFC2", radius=0.09)
    axis.text(lead_x + 0.16, loss_y + 0.51, "lead weights", ha="left", va="center", fontsize=7.1, color=GREEN, fontweight="bold")
    axis.text(
        lead_x + lead_w - 0.16,
        loss_y + 0.51,
        "[0, 0, .25, .25, .25, .25]",
        ha="right",
        va="center",
        fontsize=7.6,
        color=INK,
        fontweight="bold",
    )
    axis.text(
        lead_x + 0.16,
        loss_y + 0.20,
        "only Weeks 3–6 contribute gradients",
        ha="left",
        va="center",
        fontsize=6.7,
        color=MUTED,
    )

    contract_x = lead_x + lead_w + 0.20
    contract_w = x + width - 0.18 - contract_x
    _box(axis, (contract_x, loss_y), contract_w, loss_h, face=RED_LIGHT, edge="#FFD0CB", radius=0.09)
    axis.text(
        contract_x + contract_w / 2,
        loss_y + 0.50,
        "2023 = development evidence",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=RED,
    )
    axis.text(
        contract_x + contract_w / 2,
        loss_y + 0.20,
        "not a confirmatory test",
        ha="center",
        va="center",
        fontsize=6.8,
        color=RED,
    )


def make_figure(output_directory: Path) -> Tuple[Path, Path]:
    """Create PNG and vector PDF copies and return their paths."""

    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / "neural_adapter_architecture.png"
    pdf_path = output_directory / "neural_adapter_architecture.pdf"

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(16, 9), facecolor=PAPER)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")

    axis.text(
        0.45,
        8.72,
        "FuXi–IMERG v3 late-lead neural adapter",
        ha="left",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.45,
        8.41,
        "A deterministic, climatology-aware residual correction for weekly precipitation over India",
        ha="left",
        va="center",
        fontsize=10.2,
        color=MUTED,
    )
    _pill(
        axis,
        13.24,
        8.44,
        "IMPLEMENTED v3 CONTRACT",
        face=GREEN_LIGHT,
        color=GREEN,
        width=2.31,
        height=0.32,
        fontsize=7.6,
    )

    _draw_split_strip(axis)
    _draw_inputs(axis)
    _draw_network(axis)
    _draw_anchor(axis)
    _draw_loss_footer(axis)

    metadata = {
        "Title": "FuXi-IMERG v3 late-lead neural adapter architecture",
        "Author": "neural_adapter/scripts/make_v3_architecture_figure.py",
        "Subject": "Implemented deterministic precipitation correction architecture",
        "Keywords": "FuXi, IMERG, subseasonal, precipitation, U-Net, Transformer",
        "CreationDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
    }
    figure.savefig(png_path, dpi=240, facecolor=PAPER, bbox_inches=None)
    figure.savefig(pdf_path, facecolor=PAPER, bbox_inches=None, metadata=metadata)
    plt.close(figure)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=default_output,
        help="Directory for neural_adapter_architecture.{png,pdf}",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    png_path, pdf_path = make_figure(arguments.output_directory.resolve())
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
