#!/usr/bin/env python3
"""Make a minimal teaching plot from the saved 2023 development metrics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "neural_adapter_simple_model_comparison_data.csv"
METHODS = ("raw_fuxi", "log_bias_correction", "hybrid_v3")
LABELS = {
    "raw_fuxi": "Raw FuXi forecast",
    "log_bias_correction": "Standard log-bias correction",
    "hybrid_v3": "Neural temporal adapter",
}
STYLES = {
    "raw_fuxi": ("#6B7280", "o", "--", "white"),
    "log_bias_correction": ("#2563EB", "D", "-", "white"),
    "hybrid_v3": ("#15803D", "o", "-", "#15803D"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    table = pd.read_csv(SOURCE)
    if set(METHODS) - set(table["method"]):
        raise ValueError("source table lacks a required teaching-plot method")
    for method in METHODS:
        rows = table.loc[table["method"].eq(method)].sort_values("lead")
        if rows["lead"].astype(int).tolist() != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"{method} lacks Weeks 1–6")
        if not rows["case_count"].eq(93).all():
            raise ValueError(f"{method} does not use the 93-case validation sample")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.3), facecolor="white")
    panels = (
        (axes[0], "acc_mean", "a   Does it get the rainfall pattern right?", "Spatial ACC"),
        (axes[1], "mae_mean", "b   How large is the rainfall error?", "MAE (mm day$^{-1}$)"),
    )
    for axis, metric, title, ylabel in panels:
        axis.axvspan(0.75, 2.5, color="#F1F5F9", zorder=0)
        axis.axvspan(2.5, 6.25, color="#EAF7EF", zorder=0)
        axis.axvline(2.5, color="#84A98C", linewidth=1.1, linestyle="--")
        for method in METHODS:
            rows = table.loc[table["method"].eq(method)].sort_values("lead")
            color, marker, linestyle, marker_face = STYLES[method]
            axis.plot(
                rows["lead"], rows[metric], color=color, marker=marker,
                linestyle=linestyle, linewidth=2.8 if method == "hybrid_v3" else 2.0,
                markersize=7, markerfacecolor=marker_face, markeredgecolor=color,
                markeredgewidth=1.4, label=LABELS[method],
                zorder=6 if method == "hybrid_v3" else 5,
            )
        axis.set_title(title, loc="left", fontsize=13.2, fontweight="bold", pad=11)
        axis.set_xlabel("Forecast lead week")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(1, 7))
        axis.set_xlim(0.8, 6.2)
        axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
        axis.text(
            1.62, 0.97, "Weeks 1–2\nstandard correction only",
            transform=axis.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.8, color="#64748B", linespacing=1.25,
        )
        axis.text(
            4.35, 0.97, "Weeks 3–6\nneural residual correction",
            transform=axis.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.8, color="#2F855A", linespacing=1.25,
        )

    axes[0].set_ylim(0.09, 0.71)
    axes[1].set_ylim(1.35, 2.45)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.80),
        ncol=3, frameon=False, columnspacing=1.7, handlelength=2.8,
    )
    figure.suptitle(
        "What the neural adapter changes", fontsize=19, fontweight="bold", y=0.975
    )
    figure.text(
        0.5, 0.915,
        "FuXi weekly rainfall over India → conventional bias correction → learned late-lead residual correction",
        ha="center", fontsize=10.6, color="#334155",
    )
    figure.text(
        0.5, 0.875,
        "2023 development validation: 93 twice-weekly initializations | IMERG verification | "
        "six non-overlapping 7-day means (W1 D0–6 … W6 D35–41)",
        ha="center", fontsize=9.4, color="#7F1D1D", fontweight="semibold",
    )
    figure.text(
        0.5, 0.04,
        "Higher ACC is better; lower MAE is better. The neural line exactly overlaps standard log-bias "
        "at Weeks 1–2 by design. This is validation, not the reserved 2024 test.",
        ha="center", fontsize=8.7, color="#475569",
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.15, top=0.66, wspace=0.20)

    stem = ROOT / "neural_adapter_explainer_plot"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    csv = Path(str(stem) + "_data.csv")
    manifest = Path(str(stem) + "_manifest.json")
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(
        pdf, bbox_inches="tight", facecolor="white",
        metadata={
            "Title": "What the neural adapter changes",
            "Author": "FuXi-IMERG neural adapter project",
            "Subject": "2023 development validation teaching figure",
            "CreationDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
            "ModDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
        },
    )
    plt.close(figure)
    table.loc[table["method"].isin(METHODS)].to_csv(csv, index=False)
    manifest.write_text(
        json.dumps(
            {
                "status": "derived_from_saved_2023_development_metrics",
                "confirmatory": False,
                "operations_excluded": ["training", "prediction", "rescoring"],
                "purpose": "minimal teaching figure; does not replace the full comparison",
                "split": "2023 development validation",
                "case_count": 93,
                "timescale": "six non-overlapping 7-day means; W1 D0-6 through W6 D35-41",
                "methods": [LABELS[value] for value in METHODS],
                "source": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
                "outputs": {
                    "png": {"path": str(png), "sha256": sha256(png)},
                    "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
                    "csv": {"path": str(csv), "sha256": sha256(csv)},
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
