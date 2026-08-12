#!/usr/bin/env python3
"""Make the formal four-metric lead-time figure from saved validation summaries.

This is a presentation-only renderer.  It verifies and reads the compact table
produced by the completed validation audit; it does not open predictions,
recompute scores, tune a model, or train anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLE = ROOT / "neural_adapter_experiment_comparison_by_week.csv"
SOURCE_MANIFEST = ROOT / "neural_adapter_experiment_comparison_manifest.json"
OUTPUT_STEM = ROOT / "neural_adapter_academic_metrics"

METHODS = (
    "raw_fuxi",
    "log_bias_correction",
    "v2_residual_unet",
    "hybrid_v3",
    "distribution_v3",
)
DISPLAY = {
    "raw_fuxi": "Raw FuXi",
    "log_bias_correction": "Log-bias correction",
    "v2_residual_unet": "v2 spatial U-Net",
    "hybrid_v3": "v3 temporal hybrid",
    "distribution_v3": "v3 + distribution features",
}
COLORS = {
    "raw_fuxi": "#68717D",
    "log_bias_correction": "#2764B7",
    "v2_residual_unet": "#D97800",
    "hybrid_v3": "#168354",
    "distribution_v3": "#9B4B96",
}
MARKERS = {
    "raw_fuxi": "o",
    "log_bias_correction": "D",
    "v2_residual_unet": "^",
    "hybrid_v3": "o",
    "distribution_v3": "s",
}
LINESTYLES: dict[str, Any] = {
    "raw_fuxi": (0, (4, 2)),
    "log_bias_correction": "-",
    "v2_residual_unet": (0, (1.5, 1.5)),
    "hybrid_v3": "-",
    "distribution_v3": (0, (5, 1.5)),
}
PANELS = (
    ("acc_mean", "Spatial anomaly correlation", "ACC", "a"),
    ("rmse_mean", "Root-mean-square error", "RMSE (mm day$^{-1}$)", "b"),
    ("mae_mean", "Mean absolute error", "MAE (mm day$^{-1}$)", "c"),
    ("bias_mean", "Mean bias", "Bias (mm day$^{-1}$)", "d"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not SOURCE_TABLE.is_file() or not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError("The verified plotting table or its manifest is absent")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    recorded = source_manifest["outputs"]["weekly_table"]["sha256"]
    observed = sha256(SOURCE_TABLE)
    if observed != recorded:
        raise ValueError("Saved weekly table fails its recorded SHA-256 check")
    if source_manifest.get("selection_split") != "2023_validation":
        raise ValueError("Source is not the frozen 2023 development-validation summary")
    if source_manifest.get("confirmatory") is not False:
        raise ValueError("Source is not marked non-confirmatory")

    table = pd.read_csv(SOURCE_TABLE)
    required = {
        "method", "lead", "case_count", "acc_mean", "rmse_mean",
        "mae_mean", "bias_mean",
    }
    if missing := sorted(required.difference(table.columns)):
        raise ValueError(f"Saved weekly table is missing columns: {missing}")
    if set(table["method"]) != set(METHODS):
        raise ValueError("Saved weekly table does not contain exactly the five methods")
    for method in METHODS:
        rows = table.loc[table["method"].eq(method)].sort_values("lead")
        if rows["lead"].astype(int).tolist() != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"{method} does not contain exactly Weeks 1–6")
        if not rows["case_count"].eq(93).all():
            raise ValueError(f"{method} does not contain 93 cases per lead")
        if not np.isfinite(rows[[p[0] for p in PANELS]].to_numpy(float)).all():
            raise ValueError(f"{method} contains a non-finite plotted score")
    return table, source_manifest


def plotting_table(table: pd.DataFrame) -> pd.DataFrame:
    compact = table.copy()
    compact["method_label"] = compact["method"].map(DISPLAY)
    compact["valid_day_start"] = (compact["lead"].astype(int) - 1) * 7
    compact["valid_day_end"] = compact["valid_day_start"] + 6
    compact["target_aggregation"] = "non-overlapping 7-day mean"
    compact["initialization_frequency"] = "twice weekly"
    compact["verification"] = "IMERG"
    compact["evaluation_split"] = "2023 development validation"
    compact["confirmatory"] = False
    return compact[
        [
            "method", "method_label", "lead", "valid_day_start",
            "valid_day_end", "target_aggregation", "initialization_frequency",
            "verification", "evaluation_split", "confirmatory", "case_count",
            "acc_mean", "rmse_mean", "mae_mean", "bias_mean",
        ]
    ].sort_values(["method", "lead"])


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_panel(
    axis: mpl.axes.Axes,
    table: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    panel: str,
) -> None:
    # Structural contract: v3 is fixed to log-bias for W1--2 and learned for W3--6.
    axis.axvspan(0.75, 2.5, color="#F0F2F4", alpha=0.90, zorder=0)
    axis.axvspan(2.5, 6.25, color="#EAF5EE", alpha=0.90, zorder=0)
    axis.axvline(2.5, color="#7F9B8A", linestyle="--", linewidth=0.8, zorder=1)
    if metric == "bias_mean":
        axis.axhline(0.0, color="#42474D", linewidth=0.85, zorder=1)

    for method in METHODS:
        rows = table.loc[table["method"].eq(method)].sort_values("lead")
        is_v3 = method in ("hybrid_v3", "distribution_v3")
        axis.plot(
            rows["lead"],
            rows[metric],
            label=DISPLAY[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.25 if is_v3 else 1.75,
            markersize=5.3,
            markerfacecolor=COLORS[method] if is_v3 else "white",
            markeredgecolor=COLORS[method],
            markeredgewidth=1.1,
            zorder=4 if is_v3 else 3,
        )

    axis.set_title(title, loc="left", pad=17)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Lead week")
    axis.set_xticks(range(1, 7))
    axis.set_xlim(0.78, 6.22)
    axis.grid(axis="y", color="#D5DAE0", alpha=0.85, linewidth=0.65)
    axis.tick_params(direction="out", length=3.5, width=0.8)

    transform = blended_transform_factory(axis.transData, axis.transAxes)
    axis.text(
        1.5, 1.015, "W1–2: anchored", transform=transform,
        ha="center", va="bottom", fontsize=8.0, color="#59616A",
    )
    axis.text(
        4.25, 1.015, "W3–6: learned correction", transform=transform,
        ha="center", va="bottom", fontsize=8.0, color="#2F7450",
    )
    axis.text(
        -0.105, 1.095, panel, transform=axis.transAxes,
        ha="left", va="top", fontsize=12.5, fontweight="bold", color="#202428",
    )


def main() -> None:
    table, source_manifest = load_verified_table()
    compact = plotting_table(table)
    csv_path = OUTPUT_STEM.with_suffix(".csv")
    png_path = OUTPUT_STEM.with_suffix(".png")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    manifest_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_manifest.json")
    compact.to_csv(csv_path, index=False, float_format="%.15g")

    set_style()
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 9.25))
    try:
        for axis, panel in zip(axes.flat, PANELS):
            draw_panel(axis, table, *panel)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.858),
            ncol=5, frameon=False, handlelength=2.6, columnspacing=1.35,
        )
        figure.suptitle(
            "Lead-dependent verification of FuXi–IMERG precipitation correction over India",
            fontsize=16.0, fontweight="semibold", y=0.986,
        )
        figure.text(
            0.5, 0.950,
            "2023 DEVELOPMENT VALIDATION (NON-CONFIRMATORY)  |  93 twice-weekly initializations  |  IMERG verification",
            ha="center", va="center", fontsize=10.3, fontweight="semibold",
            color="#8A2F2F",
        )
        figure.text(
            0.5, 0.921,
            "Inference target per initialization: six non-overlapping 7-day mean precipitation fields (mm day$^{-1}$)",
            ha="center", va="center", fontsize=10.2, color="#30363C",
        )
        figure.text(
            0.5, 0.895,
            "W1 d0–6  |  W2 d7–13  |  W3 d14–20  |  W4 d21–27  |  W5 d28–34  |  W6 d35–41",
            ha="center", va="center", fontsize=9.6, color="#30363C",
        )
        figure.text(
            0.5, 0.024,
            (
                "India area-weighted case means. v3 curves are equal-weight ensembles of seeds 42/43/44. "
                "At W1–2 the v3 residual is structurally inactive and predictions equal the training-only "
                "log-bias anchor exactly; learned correction is applied only at W3–6."
            ),
            ha="center", va="bottom", fontsize=8.45, color="#4A5158",
        )
        figure.subplots_adjust(
            left=0.075, right=0.985, top=0.775, bottom=0.105,
            hspace=0.40, wspace=0.24,
        )
        figure.savefig(
            png_path, dpi=320, facecolor="white", bbox_inches="tight",
            metadata={"Software": Path(__file__).name},
        )
        figure.savefig(
            pdf_path, facecolor="white", bbox_inches="tight",
            metadata={
                "Title": "Lead-dependent FuXi–IMERG precipitation verification over India",
                "Author": "FuXi–IMERG neural adapter project",
                "Subject": "2023 non-confirmatory development validation",
                "Creator": Path(__file__).name,
                "CreationDate": None,
                "ModDate": None,
            },
        )
    finally:
        plt.close(figure)

    script_path = Path(__file__).resolve()
    manifest = {
        "status": "presentation_render_from_existing_verified_summary_only",
        "operations_excluded": [
            "training", "prediction", "prediction_rescoring", "model_selection",
        ],
        "evaluation": {
            "split": "2023_development_validation",
            "confirmatory": False,
            "case_count_per_lead": 93,
            "initialization_frequency": "twice_weekly",
            "verification": "IMERG",
            "spatial_summary": "India_area_weighted_case_mean",
        },
        "temporal_inference_contract": {
            "quantity": "precipitation_rate",
            "units": "mm day-1",
            "aggregation": "six_non_overlapping_7_day_means_per_initialization",
            "lead_windows_inclusive_days": {
                "week_1": [0, 6], "week_2": [7, 13], "week_3": [14, 20],
                "week_4": [21, 27], "week_5": [28, 34], "week_6": [35, 41],
            },
            "v3_anchor_weeks": [1, 2],
            "v3_learned_correction_weeks": [3, 4, 5, 6],
        },
        "methods": [{"key": method, "label": DISPLAY[method]} for method in METHODS],
        "generator": {"path": str(script_path), "sha256": sha256(script_path)},
        "sources": {
            "verified_weekly_table": {
                "path": str(SOURCE_TABLE), "sha256": sha256(SOURCE_TABLE),
            },
            "source_provenance_manifest": {
                "path": str(SOURCE_MANIFEST), "sha256": sha256(SOURCE_MANIFEST),
                "selection_split": source_manifest["selection_split"],
                "confirmatory": source_manifest["confirmatory"],
            },
        },
        "outputs": {
            "png": {"path": str(png_path), "sha256": sha256(png_path)},
            "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "plotting_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
