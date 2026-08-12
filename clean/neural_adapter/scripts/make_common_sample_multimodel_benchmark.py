#!/usr/bin/env python3
"""Plot a common-date multi-model S2S benchmark from saved case metrics.

This script deliberately keeps the IMD/common-1.5-degree benchmark separate
from the neural adapter's IMERG-grid scores.  It performs no forecast inference
and no spatial rescoring; it only selects the adapter's exact 2023 validation
initialization dates and averages already-saved case-wise metrics.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIGGYCAST_ROOT = Path(
    "/storage/raj.ayush/ashoka_storage/piggycast_s2s/runs/slurm_84636"
)
PIGGYCAST_CASE_METRICS = PIGGYCAST_ROOT / "metrics" / "case_metrics.csv"
PIGGYCAST_MANIFEST = PIGGYCAST_ROOT / "manifest.json"
ADAPTER_RESULT = Path(
    "/storage/raj.ayush/neural_adapter_data/validation_results/"
    "fuxi_imerg_late_acc_v3_distribution__20260807T052222Z"
)
ADAPTER_CASE_METRICS = ADAPTER_RESULT / "metrics" / "case_metrics.csv"

PHYSICS_MODELS = ("ecmwf", "ukmo", "ncep", "cma")
LEARNED_MODELS = ("fuxi_s2s", "neuralgcm", "dlesym_v0", "piggycast")
ALL_MODELS = PHYSICS_MODELS + LEARNED_MODELS

LABELS = {
    "ecmwf": "ECMWF",
    "ukmo": "UKMO",
    "ncep": "NCEP",
    "cma": "CMA",
    "fuxi_s2s": "FuXi-S2S",
    "neuralgcm": "NeuralGCM",
    "dlesym_v0": "DLESyM v0",
    "piggycast": "PiggyCast XGBoost†",
}
STYLES = {
    "ecmwf": ("#2563EB", "o", "-"),
    "ukmo": ("#7C3AED", "s", "--"),
    "ncep": ("#D97706", "^", "-."),
    "cma": ("#0F766E", "D", ":"),
    "fuxi_s2s": ("#2563EB", "D", "-"),
    "neuralgcm": ("#A349A4", "s", "--"),
    "dlesym_v0": ("#64748B", "^", "-."),
    "piggycast": ("#15803D", "o", "-"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_common_sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Return saved metrics restricted to the adapter's exact 93 dates."""
    adapter = pd.read_csv(ADAPTER_CASE_METRICS, parse_dates=["case_id"])
    adapter_dates = pd.DatetimeIndex(
        adapter.loc[
            adapter["split"].eq("validation") & adapter["region"].eq("india"),
            "case_id",
        ].drop_duplicates().sort_values()
    )
    if len(adapter_dates) != 93:
        raise ValueError(f"expected 93 adapter validation dates, found {len(adapter_dates)}")
    if adapter_dates.min() != pd.Timestamp("2023-01-02"):
        raise ValueError("unexpected first adapter validation initialization")
    if adapter_dates.max() != pd.Timestamp("2023-11-20"):
        raise ValueError("unexpected last adapter validation initialization")

    case_metrics = pd.read_csv(PIGGYCAST_CASE_METRICS, parse_dates=["initialization"])
    common = case_metrics.loc[
        case_metrics["split"].eq("validation")
        & case_metrics["region"].eq("india")
        & case_metrics["predictor"].isin(ALL_MODELS)
        & case_metrics["initialization"].isin(adapter_dates)
    ].copy()

    expected_dates = set(adapter_dates)
    for model in ALL_MODELS:
        for lead_week in range(1, 7):
            rows = common.loc[
                common["predictor"].eq(model)
                & common["lead_week"].eq(lead_week)
            ]
            actual_dates = set(rows["initialization"])
            if actual_dates != expected_dates:
                raise ValueError(
                    f"{model} Week {lead_week} does not contain the exact common dates"
                )
            if len(rows) != 93:
                raise ValueError(f"{model} Week {lead_week} has {len(rows)} rows")
            if rows[["acc", "rmse", "mae", "bias"]].isna().any().any():
                raise ValueError(f"{model} Week {lead_week} contains missing metrics")

    summary = (
        common.groupby(["predictor", "lead_week"], as_index=False)
        .agg(
            case_count=("initialization", "nunique"),
            acc_mean=("acc", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias", "mean"),
            valid_cells_min=("valid_cells", "min"),
            valid_cells_max=("valid_cells", "max"),
        )
        .sort_values(["predictor", "lead_week"])
    )
    if not summary["case_count"].eq(93).all():
        raise ValueError("common-sample aggregation changed the case count")
    return common, summary, adapter_dates


def plot_panel(
    axis: plt.Axes,
    summary: pd.DataFrame,
    models: tuple[str, ...],
    metric: str,
    title: str,
    ylabel: str,
    panel_letter: str,
    show_legend: bool,
) -> None:
    axis.axvspan(2.5, 6.2, color="#EFF8F2", zorder=0)
    axis.axvline(2.5, color="#94B8A3", linestyle="--", linewidth=0.9, zorder=1)
    for model in models:
        rows = summary.loc[summary["predictor"].eq(model)].sort_values("lead_week")
        color, marker, linestyle = STYLES[model]
        highlighted = model == "piggycast"
        axis.plot(
            rows["lead_week"],
            rows[metric],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.6 if highlighted else 1.9,
            markersize=6.4,
            markerfacecolor=color if highlighted else "white",
            markeredgecolor=color,
            markeredgewidth=1.25,
            label=LABELS[model],
            zorder=7 if highlighted else 5,
        )
    axis.set_title(
        f"{panel_letter}   {title}", loc="left", fontsize=12.5,
        fontweight="bold", pad=9,
    )
    axis.set_xlabel("Lead week")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0.8, 6.2)
    axis.set_xticks(range(1, 7))
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    axis.text(
        4.35, 0.975, "Weeks 3–6", transform=axis.get_xaxis_transform(),
        ha="center", va="top", fontsize=8.3, color="#2F855A",
    )
    if show_legend:
        axis.legend(
            loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1",
            framealpha=0.94, fontsize=8.7, handlelength=2.5,
        )


def main() -> None:
    _, summary, adapter_dates = load_common_sample()
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
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 8.6), facecolor="white")
    plot_panel(
        axes[0, 0], summary, PHYSICS_MODELS, "acc_mean",
        "Physics-based systems: spatial ACC", "ACC", "a", True,
    )
    plot_panel(
        axes[0, 1], summary, LEARNED_MODELS, "acc_mean",
        "AI / learned systems: spatial ACC", "ACC", "b", True,
    )
    plot_panel(
        axes[1, 0], summary, PHYSICS_MODELS, "mae_mean",
        "Physics-based systems: MAE", "MAE (mm day$^{-1}$)", "c", False,
    )
    plot_panel(
        axes[1, 1], summary, LEARNED_MODELS, "mae_mean",
        "AI / learned systems: MAE", "MAE (mm day$^{-1}$)", "d", False,
    )
    for axis in axes[0, :]:
        axis.set_ylim(-0.08, 0.78)
    for axis in axes[1, :]:
        axis.set_ylim(1.2, 3.5)

    figure.suptitle(
        "Common-sample S2S precipitation benchmark over India",
        fontsize=18, fontweight="bold", y=0.975,
    )
    figure.text(
        0.5, 0.932,
        "2023 development validation | exact matched initializations: 02 Jan–20 Nov 2023 "
        "(n=93) | IMD verification on the common 1.5° grid",
        ha="center", fontsize=10.4, color="#7F1D1D", fontweight="semibold",
    )
    figure.text(
        0.5, 0.895,
        "Saved ensemble-mean forecasts | six non-overlapping 7-day means: "
        "W1 D0–6 … W6 D35–41 | identical timestamps, leads, mask and verification",
        ha="center", fontsize=9.2, color="#475569",
    )
    figure.text(
        0.5, 0.045,
        "Case-wise India area-weighted means; descriptive comparison. † PiggyCast is a locally trained "
        "XGBoost development stacker; its split lacks a 42-day verification gap before 2023.",
        ha="center", fontsize=8.2, color="#475569",
    )
    figure.text(
        0.5, 0.018,
        "The IMERG-trained neural adapter is intentionally not overlaid: its saved scores use a different "
        "verification product and grid. No forecast inference or spatial rescoring was run here.",
        ha="center", fontsize=8.2, color="#7F1D1D",
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.84, hspace=0.33, wspace=0.18)

    stem = ROOT / "common_sample_multimodel_benchmark"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    csv_path = Path(str(stem) + "_metrics.csv")
    dates_path = Path(str(stem) + "_dates.csv")
    manifest_path = Path(str(stem) + "_manifest.json")

    figure.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": "Common-sample S2S precipitation benchmark over India",
            "Author": "FuXi-IMERG neural adapter project",
            "Subject": "2023 common-date IMD development validation",
            "CreationDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
            "ModDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
        },
    )
    plt.close(figure)
    summary.to_csv(csv_path, index=False)
    pd.DataFrame({"initialization": adapter_dates}).to_csv(dates_path, index=False)

    source_manifest = json.loads(PIGGYCAST_MANIFEST.read_text(encoding="utf-8"))
    payload = {
        "status": "derived_from_saved_case_metrics_on_exact_common_dates",
        "confirmatory": False,
        "operations_excluded": ["training", "forecast_inference", "spatial_rescoring"],
        "selection_split": "2023 development validation",
        "initializations": {
            "start": adapter_dates.min().date().isoformat(),
            "end": adapter_dates.max().date().isoformat(),
            "count": int(len(adapter_dates)),
            "matching": "exact set equality for every model and lead week",
        },
        "forecast_contract": {
            "variable": "total precipitation rate",
            "units": "mm day-1",
            "lead_definition": "six non-overlapping 7-day means; W1 D0-6 through W6 D35-41",
            "grid": "common 1.5 degree India grid",
            "verification": "IMD daily precipitation aggregated to the matching weekly intervals",
            "metric_aggregation": "mean of saved case-wise India area-weighted metrics",
        },
        "families": {
            "physics_based": [LABELS[value] for value in PHYSICS_MODELS],
            "ai_or_learned": [LABELS[value] for value in LEARNED_MODELS],
        },
        "adapter_metrics_included": False,
        "adapter_exclusion_reason": (
            "saved neural-adapter metrics use IMERG and a different grid; direct overlay would not be comparable"
        ),
        "caveats": [
            "same initialization timestamps do not imply identical analysis initial states or ensemble sizes",
            "PiggyCast is development-only and its train/validation split does not impose a 42-day verification gap",
            "no uncertainty intervals are shown",
        ],
        "source_archive_root": source_manifest["config"]["archive_root"],
        "sources": {
            "multimodel_case_metrics": {
                "path": str(PIGGYCAST_CASE_METRICS),
                "sha256": sha256(PIGGYCAST_CASE_METRICS),
            },
            "multimodel_manifest": {
                "path": str(PIGGYCAST_MANIFEST),
                "sha256": sha256(PIGGYCAST_MANIFEST),
            },
            "adapter_case_metrics_for_date_selection_only": {
                "path": str(ADAPTER_CASE_METRICS),
                "sha256": sha256(ADAPTER_CASE_METRICS),
            },
        },
        "outputs": {},
    }
    for key, path in {
        "png": png_path,
        "pdf": pdf_path,
        "metrics_csv": csv_path,
        "dates_csv": dates_path,
    }.items():
        payload["outputs"][key] = {"path": str(path), "sha256": sha256(path)}
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
