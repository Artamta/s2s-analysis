#!/usr/bin/env python3
"""Create compact tables and uncluttered figures from a locked result.

This script does not rescore or alter predictions.  It selects the three main
comparisons from the complete evaluation tables and plots the corresponding
fields already saved in ``predictions.zarr``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fuxi_adapter.plotting import plot_mean_maps, plot_metric_by_lead
import matplotlib.pyplot as plt


PRIMARY_PREDICTORS = ("raw_fuxi", "log_bias_correction", "residual_unet")
DISPLAY_NAMES = {
    "raw_fuxi": "Raw FuXi",
    "log_bias_correction": "Log-bias correction",
    "residual_unet": "Residual U-Net ensemble",
}
SCORE_COLUMNS = ("acc_mean", "rmse_mean", "mae_mean", "bias_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    result = parse_args().result_directory.resolve()
    metrics = result / "metrics"
    figures = result / "figures"
    summary = pd.read_csv(metrics / "summary_by_week_region.csv")
    case_metrics = pd.read_csv(metrics / "case_metrics.csv")

    by_week = summary.loc[
        summary["region"].eq("india")
        & summary["predictor"].isin(PRIMARY_PREDICTORS),
        ["split", "predictor", "lead", "region", "case_count", *SCORE_COLUMNS],
    ].sort_values(["lead", "predictor"])
    by_week.to_csv(metrics / "headline_by_week_india.csv", index=False)

    lead_mean = (
        by_week.groupby(["split", "predictor", "region"], as_index=False)[
            list(SCORE_COLUMNS)
        ]
        .mean()
        .sort_values("predictor")
    )
    lead_mean.to_csv(metrics / "headline_lead_mean_india.csv", index=False)

    seasonal = (
        case_metrics.loc[
            case_metrics["region"].eq("india")
            & case_metrics["predictor"].isin(PRIMARY_PREDICTORS)
        ]
        .groupby(["split", "predictor", "region", "season"], as_index=False)
        .agg(
            case_leads=("case_id", "size"),
            unique_initializations=("case_id", "nunique"),
            acc_mean=("acc", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias", "mean"),
        )
        .sort_values(["season", "predictor"])
    )
    seasonal.to_csv(metrics / "headline_by_season_india.csv", index=False)

    bootstrap = pd.read_csv(metrics / "paired_block_bootstrap_lead_mean.csv")
    bootstrap.loc[
        bootstrap["region"].eq("india")
        & bootstrap["predictor"].eq("residual_unet")
        & bootstrap["baseline"].isin(("raw_fuxi", "log_bias_correction"))
    ].sort_values(["baseline", "metric"]).to_csv(
        metrics / "headline_bootstrap_lead_mean_india.csv", index=False
    )

    plot_table = by_week.copy()
    plot_table["predictor"] = plot_table["predictor"].map(DISPLAY_NAMES)
    for metric in ("acc", "rmse", "mae", "bias"):
        plot_table[metric] = plot_table[f"{metric}_mean"]
        plot_metric_by_lead(
            plot_table,
            metric,
            figures / f"headline_{metric}_india.png",
            split="test",
            region="india",
        )

    colours = {
        "raw_fuxi": "#6b7280",
        "log_bias_correction": "#2563eb",
        "residual_unet": "#16a34a",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    try:
        for predictor in PRIMARY_PREDICTORS:
            rows = by_week.loc[by_week["predictor"].eq(predictor)].sort_values("lead")
            for axis, metric, label in (
                (axes[0], "acc_mean", "Spatial ACC"),
                (axes[1], "mae_mean", "MAE (mm day$^{-1}$)"),
            ):
                axis.plot(
                    rows["lead"],
                    rows[metric],
                    marker="o",
                    linewidth=2.0,
                    color=colours[predictor],
                    label=DISPLAY_NAMES[predictor],
                )
                axis.set_xlabel("Lead week")
                axis.set_ylabel(label)
                axis.set_xticks(range(1, 7))
                axis.grid(True, alpha=0.25, linewidth=0.7)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=3,
            frameon=False,
        )
        figure.suptitle("Aligned FuXi–IMERG deterministic correction over India", y=0.90)
        figure.subplots_adjust(top=0.80, bottom=0.15, wspace=0.17)
        figure.text(
            0.5,
            0.005,
            "Corrected 2024 re-analysis; 100 available initializations",
            ha="center",
            fontsize=9,
            color="0.35",
        )
        figure.savefig(
            figures / "headline_acc_mae_two_panel.png",
            dpi=200,
            bbox_inches="tight",
        )
    finally:
        plt.close(figure)

    dataset = xr.open_zarr(str(result / "predictions.zarr"), consolidated=True).load()
    interval_rows = []
    for initialization in dataset["init"].values.astype("datetime64[D]"):
        for lead_week in dataset["lead_week"].values.astype(int):
            period_start = initialization + np.timedelta64((lead_week - 1) * 7, "D")
            period_end = period_start + np.timedelta64(7, "D")
            interval_rows.append(
                {
                    "initialization": np.datetime_as_string(initialization, unit="D"),
                    "lead_week": lead_week,
                    "period_start_inclusive": np.datetime_as_string(period_start, unit="D"),
                    "period_end_exclusive": np.datetime_as_string(period_end, unit="D"),
                }
            )
    pd.DataFrame(interval_rows).to_csv(
        metrics / "verification_intervals.csv", index=False
    )
    maps = {
        "IMERG truth": dataset["truth_imerg"].values,
        "Raw FuXi": dataset["raw_fuxi"].values,
        "Log-bias correction": dataset["log_bias_correction"].values,
        "Residual U-Net ensemble": dataset["residual_unet"].values,
    }
    support = np.isfinite(dataset["truth_imerg"].values).all(axis=(0, 1))
    plot_mean_maps(
        maps,
        dataset["latitude"].values,
        dataset["longitude"].values,
        support,
        figures / "headline_mean_maps_all_leads.png",
    )
    for lead_index in range(dataset.sizes["lead_week"]):
        plot_mean_maps(
            maps,
            dataset["latitude"].values,
            dataset["longitude"].values,
            support,
            figures / f"headline_mean_maps_week_{lead_index + 1}.png",
            lead_index=lead_index,
        )

    derived_files = sorted(
        list(metrics.glob("headline_*.csv"))
        + [metrics / "verification_intervals.csv"]
        + list(figures.glob("headline_*.png"))
    )
    manifest = {
        "derived_from": str(result),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "source_files": {
            "predictions_zmetadata": file_sha256(result / "predictions.zarr" / ".zmetadata"),
            "case_metrics": file_sha256(metrics / "case_metrics.csv"),
            "summary_by_week_region": file_sha256(
                metrics / "summary_by_week_region.csv"
            ),
        },
        "derived_files": {
            str(path.relative_to(result)): file_sha256(path) for path in derived_files
        },
    }
    (result / "HEADLINE_OUTPUTS.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Headline outputs written below {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
