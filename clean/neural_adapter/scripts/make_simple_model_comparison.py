#!/usr/bin/env python3
"""Make a simple ACC/MAE comparison from existing verified summaries only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
BASE_TABLE = ROOT / "neural_adapter_experiment_comparison_by_week.csv"
BASE_MANIFEST = ROOT / "neural_adapter_experiment_comparison_manifest.json"
HUBER_RESULT = Path(
    "/storage/raj.ayush/neural_adapter_data/validation_results/"
    "fuxi_imerg_late_acc_v3_huber__20260807T004923Z"
)
HYBRID_RESULT = Path(
    "/storage/raj.ayush/neural_adapter_data/validation_results/"
    "fuxi_imerg_late_acc_v3_hybrid__20260807T010828Z"
)
DISTRIBUTION_RESULT = Path(
    "/storage/raj.ayush/neural_adapter_data/validation_results/"
    "fuxi_imerg_late_acc_v3_distribution__20260807T052222Z"
)
METHODS = (
    "raw_fuxi",
    "log_bias_correction",
    "v2_residual_unet",
    "huber_v3",
    "hybrid_v3",
    "distribution_v3",
)
PRIMARY_METHODS = (
    "raw_fuxi",
    "log_bias_correction",
    "v2_residual_unet",
    "hybrid_v3",
)
ABLATION_RESULTS = {
    "huber_v3": HUBER_RESULT,
    "hybrid_v3": HYBRID_RESULT,
    "distribution_v3": DISTRIBUTION_RESULT,
}
LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias_correction": "Log-bias correction",
    "v2_residual_unet": "v2 spatial U-Net",
    "huber_v3": "v3 Huber",
    "hybrid_v3": "v3 hybrid",
    "distribution_v3": "v3 + distribution",
}
STYLES = {
    "raw_fuxi": ("#6B7280", "o", "--", "white"),
    "log_bias_correction": ("#2563EB", "D", "-", "white"),
    "v2_residual_unet": ("#D97706", "^", ":", "white"),
    "huber_v3": ("#7C6AB0", "X", "-.", "#7C6AB0"),
    "hybrid_v3": ("#15803D", "o", "-", "#15803D"),
    "distribution_v3": ("#A349A4", "s", "--", "#A349A4"),
}
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table() -> pd.DataFrame:
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("confirmatory") is not False:
        raise ValueError("source comparison is not explicitly non-confirmatory")
    if manifest.get("test_predictions_evaluated") is not False:
        raise ValueError("source comparison does not affirm zero test evaluation")
    table = pd.read_csv(BASE_TABLE)
    expected_base = {
        "raw_fuxi",
        "log_bias_correction",
        "v2_residual_unet",
        "hybrid_v3",
        "distribution_v3",
    }
    if set(table["method"]) != expected_base:
        raise ValueError("unexpected methods in verified base table")

    huber = pd.read_csv(HUBER_RESULT / "metrics" / "summary_by_week_region.csv")
    huber = huber.loc[
        huber["split"].eq("validation")
        & huber["region"].eq("india")
        & huber["predictor"].eq("late_lead_temporal_unet"),
        ["lead", "case_count", "acc_mean", "rmse_mean", "mae_mean", "bias_mean"],
    ].copy()
    huber.insert(0, "method", "huber_v3")
    combined = pd.concat([table, huber], ignore_index=True)
    for method in METHODS:
        rows = combined.loc[combined["method"].eq(method)].sort_values("lead")
        if rows["lead"].astype(int).tolist() != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"{method} does not contain exactly Weeks 1-6")
        if not rows["case_count"].eq(93).all():
            raise ValueError(f"{method} does not contain 93 cases at every lead")
    return combined.sort_values(["method", "lead"])


def load_ablation_stats() -> pd.DataFrame:
    """Read saved development-gate summaries; do not rescore predictions."""
    records: list[dict[str, object]] = []
    for method, result_directory in ABLATION_RESULTS.items():
        gate_path = result_directory / "development_gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("selection_split") != "2023_validation":
            raise ValueError(f"unexpected selection split in {gate_path}")
        if gate.get("confirmatory") is not False:
            raise ValueError(f"gate is not explicitly non-confirmatory: {gate_path}")
        difference = gate["candidate_minus_log_bias"]
        records.append(
            {
                "method": method,
                "delta_acc": float(difference["acc"]),
                "delta_mae": float(difference["mae"]),
                "gate_path": str(gate_path),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    table = load_table()
    ablations = load_ablation_stats()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(16.0, 6.35), facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.08, 1.08, 0.78),
        left=0.055,
        right=0.982,
        bottom=0.145,
        top=0.665,
        wspace=0.22,
    )
    axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    summary_axis = figure.add_subplot(grid[0, 2])
    for axis, metric, title, ylabel in (
        (axes[0], "acc_mean", "a   Spatial anomaly correlation", "ACC"),
        (axes[1], "mae_mean", "b   Mean absolute error", "MAE (mm day$^{-1}$)"),
    ):
        axis.axvspan(0.75, 2.5, color="#F1F5F9", zorder=0)
        axis.axvspan(2.5, 6.25, color="#EAF7EF", zorder=0)
        axis.axvline(2.5, color="#7CA890", linestyle="--", linewidth=1.0)
        for method in PRIMARY_METHODS:
            rows = table.loc[table["method"].eq(method)].sort_values("lead")
            color, marker, linestyle, marker_face = STYLES[method]
            axis.plot(
                rows["lead"],
                rows[metric],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.5 if method == "hybrid_v3" else 1.8,
                markersize=6.5,
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markeredgewidth=1.3,
                label=LABELS[method],
                zorder=7 if method == "hybrid_v3" else 5,
            )
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
        axis.set_xlabel("Lead week")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(1, 7))
        axis.set_xlim(0.8, 6.2)
        axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.65)
        axis.text(1.5, 0.98, "W1-2: anchored", transform=axis.get_xaxis_transform(),
                  ha="center", va="top", color="#64748B", fontsize=8.5)
        axis.text(4.3, 0.98, "W3-6: learned correction",
                  transform=axis.get_xaxis_transform(), ha="center", va="top",
                  color="#2F855A", fontsize=8.5)

    summary_axis.set_axis_off()
    summary_axis.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.8,
            edgecolor="#CBD5E1",
            facecolor="#F8FAFC",
            transform=summary_axis.transAxes,
            clip_on=False,
        )
    )
    summary_axis.text(
        0.055, 0.91, "c   V3 ablations: Weeks 3-6",
        transform=summary_axis.transAxes, fontsize=12.5, fontweight="bold",
        ha="left", va="center", color="#111827",
    )
    summary_axis.text(
        0.055, 0.825, "Change relative to log-bias baseline",
        transform=summary_axis.transAxes, fontsize=9.0,
        ha="left", va="center", color="#475569",
    )
    summary_axis.text(
        0.055, 0.72, "Experiment", transform=summary_axis.transAxes,
        fontsize=8.5, fontweight="bold", color="#475569", ha="left",
    )
    summary_axis.text(
        0.68, 0.72, "$\\Delta$ACC", transform=summary_axis.transAxes,
        fontsize=8.5, fontweight="bold", color="#475569", ha="right",
    )
    summary_axis.text(
        0.955, 0.72, "$\\Delta$MAE", transform=summary_axis.transAxes,
        fontsize=8.5, fontweight="bold", color="#475569", ha="right",
    )
    summary_axis.plot(
        [0.055, 0.955], [0.68, 0.68], transform=summary_axis.transAxes,
        color="#CBD5E1", linewidth=0.8,
    )
    row_positions = (0.57, 0.445, 0.32)
    short_labels = {
        "huber_v3": "Huber",
        "hybrid_v3": "Hybrid",
        "distribution_v3": "+ distribution",
    }
    for row_y, method in zip(row_positions, ABLATION_RESULTS, strict=True):
        record = ablations.loc[ablations["method"].eq(method)].iloc[0]
        color = STYLES[method][0]
        summary_axis.scatter(
            [0.075], [row_y], transform=summary_axis.transAxes,
            s=38, color=color, zorder=3, clip_on=False,
        )
        summary_axis.text(
            0.12, row_y, short_labels[method], transform=summary_axis.transAxes,
            fontsize=9.5, fontweight="bold" if method == "hybrid_v3" else "normal",
            ha="left", va="center", color="#1F2937",
        )
        summary_axis.text(
            0.68, row_y, f'{record["delta_acc"]:+.3f}',
            transform=summary_axis.transAxes, fontsize=9.5,
            ha="right", va="center", color="#1F2937",
        )
        summary_axis.text(
            0.955, row_y, f'{record["delta_mae"]:+.3f}',
            transform=summary_axis.transAxes, fontsize=9.5,
            ha="right", va="center", color="#1F2937",
        )
    summary_axis.text(
        0.055, 0.205,
        "Higher $\\Delta$ACC is better; lower $\\Delta$MAE is better.\n"
        "ACC intervals include zero for all three variants.",
        transform=summary_axis.transAxes, fontsize=8.3,
        ha="left", va="top", color="#475569", linespacing=1.5,
    )
    summary_axis.text(
        0.055, 0.065, "Development comparison only - not test evidence",
        transform=summary_axis.transAxes, fontsize=8.3,
        ha="left", va="bottom", color="#991B1B", fontweight="semibold",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.385, 0.805),
        ncol=4,
        frameon=False,
        columnspacing=1.45,
        handlelength=2.7,
        fontsize=9.7,
    )
    figure.suptitle(
        "FuXi–IMERG precipitation correction over India",
        fontsize=18, fontweight="bold", y=0.975,
    )
    figure.text(
        0.5,
        0.915,
        "PLOTTED: 2023 development validation | initializations 02 Jan–20 Nov 2023 "
        "(n=93, twice weekly) | verification reaches 01 Jan 2024",
        ha="center",
        fontsize=10.3,
        color="#7F1D1D",
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.865,
        "TRAINING: 02 Jan 2020–21 Nov 2022 (n=302; verification reaches 02 Jan 2023) | "
        "six non-overlapping 7-day means: W1 D0–6 … W6 D35–41",
        ha="center",
        fontsize=9.2,
        color="#475569",
    )
    figure.text(
        0.5,
        0.035,
        "India area-weighted case means · IMERG verification · 2001–2019 IMERG climatology for ACC · "
        "weekly mean precipitation rate (mm day$^{-1}$) · 2024 test is reserved and not shown",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )

    stem = ROOT / "neural_adapter_simple_model_comparison"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    csv = Path(str(stem) + "_data.csv")
    manifest = Path(str(stem) + "_manifest.json")
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": "Simple FuXi-IMERG model comparison over India",
            "Author": "FuXi-IMERG neural adapter project",
            "Subject": "2023 development validation",
            "CreationDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
            "ModDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
        },
    )
    plt.close(figure)
    table.to_csv(csv, index=False)
    payload = {
        "status": "derived_from_saved_verified_2023_development_metrics",
        "confirmatory": False,
        "operations_excluded": ["training", "prediction", "rescoring"],
        "methods": [LABELS[value] for value in METHODS],
        "main_panel_methods": [LABELS[value] for value in PRIMARY_METHODS],
        "ablation_table_methods": [LABELS[value] for value in ABLATION_RESULTS],
        "timescale": "six non-overlapping 7-day means; W1 D0-6 through W6 D35-41",
        "plotted_period": {
            "split": "2023 development validation",
            "initialization_start": "2023-01-02",
            "initialization_end": "2023-11-20",
            "initialization_count": 93,
            "last_verification_boundary": "2024-01-01",
        },
        "training_period": {
            "initialization_start": "2020-01-02",
            "initialization_end": "2022-11-21",
            "initialization_count": 302,
            "last_verification_boundary": "2023-01-02",
        },
        "test_period": {
            "initialization_start": "2024-01-01",
            "initialization_end": "2024-12-30",
            "initialization_count": 100,
            "last_verification_boundary": "2025-02-10",
            "shown": False,
        },
        "display_note": "four primary lead curves; Huber/hybrid/distribution shown separately as saved W3-6 deltas",
        "sources": {
            "base_table": {"path": str(BASE_TABLE), "sha256": sha256(BASE_TABLE)},
            "base_manifest": {"path": str(BASE_MANIFEST), "sha256": sha256(BASE_MANIFEST)},
            "huber_summary": {
                "path": str(HUBER_RESULT / "metrics" / "summary_by_week_region.csv"),
                "sha256": sha256(HUBER_RESULT / "metrics" / "summary_by_week_region.csv"),
            },
            "development_gates": {
                method: {
                    "path": str(result_directory / "development_gate.json"),
                    "sha256": sha256(result_directory / "development_gate.json"),
                }
                for method, result_directory in ABLATION_RESULTS.items()
            },
        },
        "outputs": {
            "png": {"path": str(png), "sha256": sha256(png)},
            "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
            "csv": {"path": str(csv), "sha256": sha256(csv)},
        },
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
