#!/usr/bin/env python3
"""Create compact, explicitly developmental v3 tables and a presentation PNG.

This script never rescores predictions.  It only derives presentation outputs
from a completed, independently verified v3 validation-result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from fuxi_adapter.metrics import paired_moving_block_bootstrap


PREDICTORS = (
    "raw_fuxi",
    "log_bias_correction",
    "v2_residual_unet",
    "late_lead_temporal_unet",
)
DISPLAY_NAMES = {
    "raw_fuxi": "Raw FuXi",
    "log_bias_correction": "Log-bias correction",
    "v2_residual_unet": "v2 residual U-Net",
    "late_lead_temporal_unet": "v3 temporal adapter",
}
COLOURS = {
    "raw_fuxi": "#6b7280",
    "log_bias_correction": "#2563eb",
    "v2_residual_unet": "#d97706",
    "late_lead_temporal_unet": "#15803d",
}
LINE_STYLES = {
    "raw_fuxi": "--",
    "log_bias_correction": "-",
    "v2_residual_unet": ":",
    "late_lead_temporal_unet": "-",
}
SCORE_COLUMNS = ("acc_mean", "rmse_mean", "mae_mean", "bias_mean")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_sources(result: Path) -> tuple[pd.DataFrame, Mapping[str, object]]:
    manifest = _read_json(result / "ensemble_manifest.json")
    gate = _read_json(result / "development_gate.json")
    if manifest.get("status") != "validation_only_development":
        raise ValueError("result is not marked validation-only development")
    if manifest.get("confirmatory") is not False:
        raise ValueError("validation result must be explicitly nonconfirmatory")
    if manifest.get("test_predictions_evaluated") is not False:
        raise ValueError("manifest does not affirm that test predictions were unused")
    if gate.get("confirmatory") is not False:
        raise ValueError("development gate must be explicitly nonconfirmatory")

    summary = pd.read_csv(result / "metrics" / "summary_by_week_region.csv")
    required_columns = {
        "split",
        "predictor",
        "lead",
        "region",
        "case_count",
        *SCORE_COLUMNS,
    }
    missing = required_columns.difference(summary.columns)
    if missing:
        raise ValueError(f"summary table is missing columns: {sorted(missing)}")
    selected = summary.loc[
        summary["region"].eq("india")
        & summary["predictor"].isin(PREDICTORS),
        [
            "split",
            "predictor",
            "lead",
            "region",
            "case_count",
            *SCORE_COLUMNS,
        ],
    ].copy()
    observed_predictors = set(selected["predictor"])
    if observed_predictors != set(PREDICTORS):
        raise ValueError(
            "result must contain raw, log-bias, v2, and v3 predictors; observed "
            + repr(sorted(observed_predictors))
        )
    for predictor in PREDICTORS:
        leads = sorted(
            selected.loc[selected["predictor"].eq(predictor), "lead"]
            .astype(int)
            .tolist()
        )
        if leads != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"{predictor} does not contain exactly lead Weeks 1-6")
    if not selected["split"].eq("validation").all():
        raise ValueError("headline table contains a non-validation row")
    return selected.sort_values(["lead", "predictor"]), gate


def make_outputs(result: Path, output: Path) -> Mapping[str, Path]:
    result = result.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    by_week, gate = _validate_sources(result)

    lead_mean = (
        by_week.groupby(["split", "predictor", "region"], as_index=False)[
            list(SCORE_COLUMNS)
        ]
        .mean()
        .sort_values("predictor")
    )
    late = (
        by_week.loc[by_week["lead"].isin([3, 4, 5, 6])]
        .groupby(["split", "predictor", "region"], as_index=False)[
            list(SCORE_COLUMNS)
        ]
        .mean()
        .sort_values("predictor")
    )
    case_metrics = pd.read_csv(result / "metrics" / "case_metrics.csv")
    late_cases = (
        case_metrics.loc[case_metrics["lead"].isin([3, 4, 5, 6])]
        .groupby(["predictor", "case_id", "region"], as_index=False)[
            ["acc", "rmse", "mae", "bias"]
        ]
        .mean()
    )
    bootstrap_contract = pd.read_csv(
        result / "metrics" / "paired_block_bootstrap_late_weeks_3_to_6.csv"
    )
    contract_values = {
        name: bootstrap_contract[name].drop_duplicates().tolist()
        for name in ("block_length", "n_resamples", "seed")
    }
    if any(len(values) != 1 for values in contract_values.values()):
        raise ValueError("saved late-lead bootstrap contract is not unique")
    bootstraps = pd.concat(
        [
            paired_moving_block_bootstrap(
                late_cases,
                "late_lead_temporal_unet",
                baseline,
                metric_columns=("acc", "rmse", "mae", "bias"),
                group_columns=("region",),
                block_length=int(contract_values["block_length"][0]),
                n_resamples=int(contract_values["n_resamples"][0]),
                seed=int(contract_values["seed"][0]),
            )
            for baseline in (
                "raw_fuxi",
                "v2_residual_unet",
                "log_bias_correction",
            )
        ],
        ignore_index=True,
    )
    bootstraps = bootstraps.loc[bootstraps["region"].eq("india")].sort_values(
        ["baseline", "metric"]
    )
    table_paths = {
        "by_week": output / "v3_headline_by_week_india.csv",
        "lead_mean": output / "v3_headline_lead_mean_india.csv",
        "late_mean": output / "v3_headline_weeks_3_to_6_india.csv",
        "late_bootstrap": output / "v3_headline_bootstrap_weeks_3_to_6_india.csv",
    }
    by_week.to_csv(table_paths["by_week"], index=False)
    lead_mean.to_csv(table_paths["lead_mean"], index=False)
    late.to_csv(table_paths["late_mean"], index=False)
    bootstraps.to_csv(table_paths["late_bootstrap"], index=False)

    interval = gate["late_acc_block_interval"]
    differences = gate["candidate_minus_log_bias"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    try:
        for predictor in PREDICTORS:
            rows = by_week.loc[by_week["predictor"].eq(predictor)].sort_values(
                "lead"
            )
            for axis, metric, label in (
                (axes[0], "acc_mean", "Spatial ACC"),
                (axes[1], "mae_mean", "MAE (mm day$^{-1}$)"),
            ):
                axis.plot(
                    rows["lead"],
                    rows[metric],
                    color=COLOURS[predictor],
                    linestyle=LINE_STYLES[predictor],
                    linewidth=2.4 if predictor == "late_lead_temporal_unet" else 1.9,
                    marker="o",
                    markersize=5.0,
                    label=DISPLAY_NAMES[predictor],
                    zorder=4 if predictor == "late_lead_temporal_unet" else 3,
                )
        for axis in axes:
            axis.axvspan(2.5, 6.2, color="#dcfce7", alpha=0.30, zorder=0)
            axis.text(
                4.25,
                0.97,
                "trained correction active",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.5,
                color="#166534",
            )
            axis.set_xlabel("Lead week")
            axis.set_xticks(range(1, 7))
            axis.set_xlim(0.8, 6.2)
            axis.grid(True, alpha=0.23, linewidth=0.7)
        axes[0].set_ylabel("Spatial ACC")
        axes[1].set_ylabel("MAE (mm day$^{-1}$)")

        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=4,
            frameon=False,
            fontsize=9,
        )
        figure.suptitle(
            "FuXi–IMERG deterministic precipitation correction over India",
            fontsize=14,
            fontweight="semibold",
            y=0.985,
        )
        figure.text(
            0.5,
            0.935,
            "2023 development validation (93 initializations) — not confirmatory",
            ha="center",
            fontsize=10,
            color="#991b1b",
        )
        acc_rows = bootstraps.loc[bootstraps["metric"].eq("acc")].set_index(
            "baseline"
        )
        caption = (
            "Weeks 3–6 ΔACC (moving-block 95% CI): "
            f"vs raw {acc_rows.loc['raw_fuxi', 'mean_difference']:+.3f} "
            f"[{acc_rows.loc['raw_fuxi', 'ci_lower']:+.3f}, "
            f"{acc_rows.loc['raw_fuxi', 'ci_upper']:+.3f}]; "
            f"vs v2 {acc_rows.loc['v2_residual_unet', 'mean_difference']:+.3f} "
            f"[{acc_rows.loc['v2_residual_unet', 'ci_lower']:+.3f}, "
            f"{acc_rows.loc['v2_residual_unet', 'ci_upper']:+.3f}]; "
            f"vs log-bias {float(differences['acc']):+.3f} "
            f"[{float(interval['lower']):+.3f}, {float(interval['upper']):+.3f}]."
        )
        figure.text(0.5, 0.012, caption, ha="center", fontsize=8.8, color="0.25")
        figure.subplots_adjust(top=0.79, bottom=0.17, left=0.075, right=0.985, wspace=0.18)
        figure_path = output / "neural_adapter_v3_validation.png"
        figure.savefig(figure_path, dpi=240, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)

    source_paths = {
        "ensemble_manifest": result / "ensemble_manifest.json",
        "development_gate": result / "development_gate.json",
        "summary_by_week_region": result / "metrics" / "summary_by_week_region.csv",
        "case_metrics": result / "metrics" / "case_metrics.csv",
        "saved_late_bootstrap": result
        / "metrics"
        / "paired_block_bootstrap_late_weeks_3_to_6.csv",
    }
    derived_paths = {**table_paths, "figure": figure_path}
    derivation = {
        "status": "derived_from_verified_2023_development_validation",
        "confirmatory": False,
        "source_result": str(result),
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "derived_sha256": {
            name: _sha256(path) for name, path in derived_paths.items()
        },
    }
    manifest_path = output / "v3_headline_outputs.json"
    manifest_path.write_text(
        json.dumps(derivation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**derived_paths, "manifest": manifest_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = make_outputs(args.result_directory, args.output_directory)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
