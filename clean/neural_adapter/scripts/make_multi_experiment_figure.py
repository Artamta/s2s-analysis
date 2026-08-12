#!/usr/bin/env python3
"""Create a provenance-checked, multi-experiment 2023 validation figure.

The figure uses only the saved metrics and paired moving-block intervals from
completed validation-result directories.  It does not rescore, retrain, or
touch the held-out 2024 predictions.  A JSON manifest and compact plotting
tables are emitted beside the PNG/PDF so every plotted number is traceable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/storage/raj.ayush/neural_adapter_data/validation_results")
DEFAULT_RESULTS = {
    "Huber": DEFAULT_ROOT / "fuxi_imerg_late_acc_v3_huber__20260807T004923Z",
    "Hybrid": DEFAULT_ROOT / "fuxi_imerg_late_acc_v3_hybrid__20260807T010828Z",
    "Distribution": (
        DEFAULT_ROOT / "fuxi_imerg_late_acc_v3_distribution__20260807T052222Z"
    ),
}

BASELINE_KEYS = ("raw_fuxi", "log_bias_correction", "v2_residual_unet")
PLOT_METHODS = (
    "raw_fuxi",
    "log_bias_correction",
    "v2_residual_unet",
    "hybrid_v3",
    "distribution_v3",
)
DISPLAY_NAMES = {
    "raw_fuxi": "Raw FuXi",
    "log_bias_correction": "Log-bias correction",
    "v2_residual_unet": "v2 spatial U-Net",
    "hybrid_v3": "v3 temporal (hybrid loss)",
    "distribution_v3": "v3 + ensemble distribution",
}
COLORS = {
    "raw_fuxi": "#737B86",
    "log_bias_correction": "#2864B7",
    "v2_residual_unet": "#E28A22",
    "hybrid_v3": "#16845B",
    "distribution_v3": "#A349A4",
    "Huber": "#7561A8",
    "Hybrid": "#16845B",
    "Distribution": "#A349A4",
}
MARKERS = {
    "raw_fuxi": "o",
    "log_bias_correction": "D",
    "v2_residual_unet": "^",
    "hybrid_v3": "o",
    "distribution_v3": "s",
}
LINESTYLES = {
    "raw_fuxi": (0, (4, 2)),
    "log_bias_correction": "-",
    "v2_residual_unet": (0, (2, 1.5)),
    "hybrid_v3": "-",
    "distribution_v3": "-",
}
SCORE_COLUMNS = ("acc_mean", "rmse_mean", "mae_mean", "bias_mean")


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(config: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in config.items() if key != "config_path"}
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_columns(table: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _validation_flag_is_false(value: Any) -> bool:
    return value is False or value == "False"


def _audit_result(
    label: str,
    result: Path,
    expected_experiment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any], dict[str, str]]:
    """Audit one completed result and return its India summary/bootstrap rows."""

    result = result.resolve()
    files = {
        "ensemble_manifest": result / "ensemble_manifest.json",
        "development_gate": result / "development_gate.json",
        "prediction_attrs": result / "predictions.zarr" / ".zattrs",
        "summary": result / "metrics" / "summary_by_week_region.csv",
        "case_metrics": result / "metrics" / "case_metrics.csv",
        "bootstrap": (
            result
            / "metrics"
            / "paired_block_bootstrap_late_weeks_3_to_6.csv"
        ),
    }
    absent = [str(path) for path in files.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("missing result artifacts: " + ", ".join(absent))

    manifest = _read_json(files["ensemble_manifest"])
    gate = _read_json(files["development_gate"])
    attrs = _read_json(files["prediction_attrs"])
    if manifest.get("status") != "validation_only_development":
        raise ValueError(f"{label}: result is not validation-only development")
    if manifest.get("confirmatory") is not False or gate.get("confirmatory") is not False:
        raise ValueError(f"{label}: result must be marked non-confirmatory")
    if manifest.get("test_predictions_evaluated") is not False:
        raise ValueError(f"{label}: manifest does not affirm unused test predictions")
    if gate.get("selection_split") != "2023_validation":
        raise ValueError(f"{label}: gate is not for 2023 validation")
    if attrs.get("split") != "2023_validation" or attrs.get("status") != "development_only":
        raise ValueError(f"{label}: prediction-store development split is inconsistent")
    if not _validation_flag_is_false(attrs.get("test_predictions_evaluated")):
        raise ValueError(f"{label}: prediction store does not affirm unused 2024 output")
    if manifest.get("equal_weight_ensemble") is not True:
        raise ValueError(f"{label}: ensemble is not marked equal-weight")
    if sorted(int(value) for value in manifest.get("seeds", [])) != [42, 43, 44]:
        raise ValueError(f"{label}: expected the predeclared seeds 42, 43, and 44")
    if manifest.get("model") != "late_lead_temporal_unet":
        raise ValueError(f"{label}: unexpected model in ensemble manifest")
    config_hash = str(manifest.get("config_sha256", ""))
    if len(config_hash) != 64 or attrs.get("config_sha256") != config_hash:
        raise ValueError(f"{label}: missing or inconsistent configuration hash")

    run_directories = [Path(value).resolve() for value in manifest.get("run_directories", [])]
    if len(run_directories) != 3:
        raise ValueError(f"{label}: expected exactly three ensemble run directories")
    observed_seeds: list[int] = []
    for run in run_directories:
        success_path = run / "SUCCESS.json"
        config_path = run / "resolved_config.json"
        if not success_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"{label}: incomplete source run {run}")
        success = _read_json(success_path)
        config = _read_json(config_path)
        if success.get("status") != "success" or success.get("smoke") is not False:
            raise ValueError(f"{label}: source run is not a completed scientific run")
        if success.get("validation_only") is not True:
            raise ValueError(f"{label}: source run is not validation-only")
        no_test = success.get("test_predictions_evaluated") is False or success.get(
            "test_data_accessed"
        ) is False
        if not no_test:
            raise ValueError(f"{label}: source run does not affirm unused test output")
        observed_seeds.append(int(success["seed"]))
        if config.get("experiment_name") != expected_experiment:
            raise ValueError(f"{label}: source run experiment name is inconsistent")
        if _config_sha256(config) != config_hash:
            raise ValueError(f"{label}: resolved configuration hash is inconsistent")
    if sorted(observed_seeds) != [42, 43, 44]:
        raise ValueError(f"{label}: source-run seeds are inconsistent")

    summary = pd.read_csv(files["summary"])
    _require_columns(
        summary,
        {"split", "predictor", "lead", "region", "case_count", *SCORE_COLUMNS},
        files["summary"],
    )
    india = summary.loc[
        summary["region"].eq("india")
        & summary["predictor"].isin((*BASELINE_KEYS, "late_lead_temporal_unet")),
        ["split", "predictor", "lead", "region", "case_count", *SCORE_COLUMNS],
    ].copy()
    if not india["split"].eq("validation").all():
        raise ValueError(f"{label}: summary includes a non-validation split")
    for predictor in (*BASELINE_KEYS, "late_lead_temporal_unet"):
        rows = india.loc[india["predictor"].eq(predictor)]
        if sorted(rows["lead"].astype(int).tolist()) != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"{label}: {predictor} does not have exactly Weeks 1-6")
        if not rows["case_count"].eq(93).all():
            raise ValueError(f"{label}: expected 93 cases at each lead")
        if not np.isfinite(rows[list(SCORE_COLUMNS)].to_numpy(dtype=float)).all():
            raise ValueError(f"{label}: non-finite saved summary score")

    # The v3 residual is structurally inactive at Weeks 1-2.
    early_v3 = india.loc[
        india["predictor"].eq("late_lead_temporal_unet") & india["lead"].isin([1, 2]),
        list(SCORE_COLUMNS),
    ].to_numpy(dtype=float)
    early_bias = india.loc[
        india["predictor"].eq("log_bias_correction") & india["lead"].isin([1, 2]),
        list(SCORE_COLUMNS),
    ].to_numpy(dtype=float)
    if not np.array_equal(early_v3, early_bias):
        raise ValueError(f"{label}: v3 Weeks 1-2 are not exact log-bias anchors")

    case_metrics = pd.read_csv(files["case_metrics"])
    _require_columns(
        case_metrics,
        {"split", "predictor", "case_id", "lead", "region", "acc", "rmse", "mae", "bias"},
        files["case_metrics"],
    )
    recomputed = (
        case_metrics.loc[
            case_metrics["region"].eq("india")
            & case_metrics["predictor"].isin((*BASELINE_KEYS, "late_lead_temporal_unet"))
        ]
        .groupby(["predictor", "lead"], as_index=False)
        .agg(
            acc_mean=("acc", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias", "mean"),
        )
        .sort_values(["predictor", "lead"])
    )
    saved = india.sort_values(["predictor", "lead"])
    if not np.allclose(
        recomputed[list(SCORE_COLUMNS)].to_numpy(dtype=float),
        saved[list(SCORE_COLUMNS)].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-13,
    ):
        raise ValueError(f"{label}: saved summaries do not reproduce case means")

    bootstrap = pd.read_csv(files["bootstrap"])
    _require_columns(
        bootstrap,
        {
            "region",
            "metric",
            "predictor",
            "baseline",
            "paired_case_count",
            "mean_difference",
            "ci_lower",
            "ci_upper",
            "block_length",
            "n_resamples",
            "seed",
        },
        files["bootstrap"],
    )
    intervals = bootstrap.loc[
        bootstrap["region"].eq("india")
        & bootstrap["metric"].isin(["acc", "mae"])
        & bootstrap["predictor"].eq("late_lead_temporal_unet")
        & bootstrap["baseline"].eq("log_bias_correction")
    ].copy()
    if sorted(intervals["metric"].tolist()) != ["acc", "mae"]:
        raise ValueError(f"{label}: expected one India ACC and MAE interval")
    if not (
        intervals["paired_case_count"].eq(93).all()
        and intervals["block_length"].eq(13).all()
        and intervals["n_resamples"].eq(2000).all()
        and intervals["seed"].eq(42).all()
    ):
        raise ValueError(f"{label}: bootstrap contract differs from 93/13/2000/42")
    if not (
        np.isfinite(
            intervals[["mean_difference", "ci_lower", "ci_upper"]].to_numpy(dtype=float)
        ).all()
        and (intervals["ci_lower"] <= intervals["mean_difference"]).all()
        and (intervals["mean_difference"] <= intervals["ci_upper"]).all()
    ):
        raise ValueError(f"{label}: malformed bootstrap interval")

    gate_delta = gate.get("candidate_minus_log_bias", {})
    for metric in ("acc", "mae"):
        row = intervals.loc[intervals["metric"].eq(metric)].iloc[0]
        if not np.isclose(
            float(row["mean_difference"]),
            float(gate_delta[metric]),
            rtol=0.0,
            atol=5e-13,
        ):
            raise ValueError(f"{label}: bootstrap and development-gate deltas differ")

    hashes = {name: _sha256(path) for name, path in files.items()}
    hashes["config_sha256"] = config_hash
    return india.sort_values(["lead", "predictor"]), intervals, manifest, hashes


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _plot_lead_panel(
    axis: mpl.axes.Axes,
    weekly: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    axis.axvspan(2.5, 6.25, color="#E8F5EE", alpha=0.95, zorder=0)
    axis.axvline(2.5, color="#80AE96", linewidth=0.8, linestyle="--", zorder=1)
    for method in PLOT_METHODS:
        rows = weekly.loc[weekly["method"].eq(method)].sort_values("lead")
        line_width = 2.35 if method in ("hybrid_v3", "distribution_v3") else 1.8
        marker_face = "white" if method in BASELINE_KEYS else COLORS[method]
        axis.plot(
            rows["lead"],
            rows[metric],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=line_width,
            marker=MARKERS[method],
            markersize=5.3,
            markerfacecolor=marker_face,
            markeredgecolor=COLORS[method],
            markeredgewidth=1.15,
            label=DISPLAY_NAMES[method],
            zorder=4 if method in ("hybrid_v3", "distribution_v3") else 3,
        )
    axis.set_title(title, loc="left", pad=9)
    axis.set_xlabel("Lead week")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(1, 7))
    axis.set_xlim(0.8, 6.2)
    axis.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    axis.text(
        4.25,
        0.035,
        "learned residual active",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#397456",
    )


def _plot_forest_panel(
    axis: mpl.axes.Axes,
    intervals: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
) -> None:
    order = ("Huber", "Hybrid", "Distribution")
    rows = intervals.loc[intervals["metric"].eq(metric)].set_index("experiment")
    axis.axvline(0.0, color="#32363B", linewidth=1.0, zorder=1)
    axis.axvspan(axis.get_xlim()[0], 0.0, color="#F5F6F7", alpha=0.35, zorder=0)
    for position, experiment in zip((2, 1, 0), order):
        row = rows.loc[experiment]
        value = float(row["mean_difference"])
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        axis.errorbar(
            value,
            position,
            xerr=np.array([[value - lower], [upper - value]]),
            fmt="o",
            color=COLORS[experiment],
            ecolor=COLORS[experiment],
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            elinewidth=2.0,
            capsize=4.0,
            capthick=1.5,
            zorder=3,
        )
        axis.annotate(
            f"{value:+.3f}  [{lower:+.3f}, {upper:+.3f}]",
            (value, position),
            xytext=(8, 9),
            textcoords="offset points",
            fontsize=8.4,
            color="#31363B",
            va="bottom",
        )
    axis.set_yticks((2, 1, 0), labels=order)
    axis.set_ylim(-0.65, 2.65)
    axis.set_title(title, loc="left", pad=9)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    axis.tick_params(axis="y", length=0)
    if metric == "acc":
        axis.text(
            0.985,
            0.04,
            "higher skill  →",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.6,
            color="#397456",
        )
    else:
        axis.text(
            0.015,
            0.04,
            "←  lower error",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.6,
            color="#397456",
        )


def make_figure(
    results: Mapping[str, Path],
    output_stem: Path,
) -> Mapping[str, Path]:
    expected = {
        "Huber": "fuxi_imerg_late_acc_v3_huber",
        "Hybrid": "fuxi_imerg_late_acc_v3_hybrid",
        "Distribution": "fuxi_imerg_late_acc_v3_distribution",
    }
    audited: dict[str, pd.DataFrame] = {}
    interval_tables: list[pd.DataFrame] = []
    manifests: dict[str, Mapping[str, Any]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for label in ("Huber", "Hybrid", "Distribution"):
        summary, intervals, manifest, hashes = _audit_result(
            label, results[label], expected[label]
        )
        audited[label] = summary
        interval_tables.append(intervals.assign(experiment=label))
        manifests[label] = manifest
        source_hashes[label] = hashes

    # These are common data/baseline/reference predictions and must be identical.
    canonical_baselines = audited["Distribution"].loc[
        audited["Distribution"]["predictor"].isin(BASELINE_KEYS)
    ].sort_values(["predictor", "lead"])
    for label in ("Huber", "Hybrid"):
        compared = audited[label].loc[
            audited[label]["predictor"].isin(BASELINE_KEYS)
        ].sort_values(["predictor", "lead"])
        if not np.array_equal(
            canonical_baselines[["lead", "case_count", *SCORE_COLUMNS]].to_numpy(),
            compared[["lead", "case_count", *SCORE_COLUMNS]].to_numpy(),
        ):
            raise ValueError(f"{label}: baseline/v2 scores differ across experiments")
        if manifests[label].get("reference_runs") != manifests["Distribution"].get(
            "reference_runs"
        ):
            raise ValueError(f"{label}: v2 reference-run provenance differs")

    weekly_parts = [canonical_baselines.assign(method=canonical_baselines["predictor"])]
    for label, method in (("Hybrid", "hybrid_v3"), ("Distribution", "distribution_v3")):
        candidate = audited[label].loc[
            audited[label]["predictor"].eq("late_lead_temporal_unet")
        ].copy()
        candidate["method"] = method
        weekly_parts.append(candidate)
    weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly = weekly[
        ["method", "lead", "case_count", *SCORE_COLUMNS]
    ].sort_values(["method", "lead"])
    intervals = pd.concat(interval_tables, ignore_index=True)[
        [
            "experiment",
            "metric",
            "paired_case_count",
            "mean_difference",
            "ci_lower",
            "ci_upper",
            "bootstrap_std",
            "block_length",
            "n_resamples",
            "seed",
        ]
    ].sort_values(["metric", "experiment"])

    output_stem = output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    weekly_path = output_stem.with_name(output_stem.name + "_by_week.csv")
    intervals_path = output_stem.with_name(output_stem.name + "_late_intervals.csv")
    manifest_path = output_stem.with_name(output_stem.name + "_manifest.json")

    weekly.to_csv(weekly_path, index=False, float_format="%.15g")
    intervals.to_csv(intervals_path, index=False, float_format="%.15g")

    _set_style()
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    try:
        _plot_lead_panel(
            axes[0, 0],
            weekly,
            "acc_mean",
            "Skill across lead weeks",
            "Spatial anomaly correlation (ACC)",
        )
        _plot_lead_panel(
            axes[0, 1],
            weekly,
            "mae_mean",
            "Error across lead weeks",
            "Mean absolute error (mm day$^{-1}$)",
        )
        _plot_forest_panel(
            axes[1, 0],
            intervals,
            "acc",
            "Weeks 3–6: ACC relative to log-bias",
            "Paired ΔACC (candidate − log-bias)",
        )
        _plot_forest_panel(
            axes[1, 1],
            intervals,
            "mae",
            "Weeks 3–6: MAE relative to log-bias",
            "Paired ΔMAE (mm day$^{-1}$; candidate − log-bias)",
        )
        for axis, label in zip(axes.flat, ("a", "b", "c", "d")):
            _panel_label(axis, label)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.885),
            ncol=5,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.5,
        )
        figure.suptitle(
            "Neural adaptation of FuXi precipitation forecasts over India",
            fontsize=16.0,
            fontweight="semibold",
            y=0.985,
        )
        figure.text(
            0.5,
            0.946,
            "2023 development validation • 93 twice-weekly forecast initializations • IMERG verification",
            ha="center",
            va="center",
            fontsize=10.8,
            color="#32383E",
        )
        figure.text(
            0.5,
            0.915,
            "MODEL-SELECTION RESULT — descriptive, not confirmatory; the held-out 2024 test was not evaluated",
            ha="center",
            va="center",
            fontsize=9.7,
            fontweight="semibold",
            color="#A23131",
        )
        figure.text(
            0.5,
            0.018,
            (
                "India area-weighted case means. v3 Weeks 1–2 are fixed to the training-only log-bias baseline; "
                "the learned residual acts at Weeks 3–6. Bars show paired 95% moving-block intervals "
                "(13 initializations/block, 2,000 resamples)."
            ),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#4B5158",
        )
        figure.subplots_adjust(
            left=0.08,
            right=0.985,
            top=0.825,
            bottom=0.105,
            hspace=0.47,
            wspace=0.25,
        )
        figure.savefig(
            png_path,
            dpi=320,
            facecolor="white",
            bbox_inches="tight",
            metadata={"Software": "make_multi_experiment_figure.py"},
        )
        figure.savefig(
            pdf_path,
            facecolor="white",
            bbox_inches="tight",
            metadata={
                "Title": "Neural adaptation of FuXi precipitation forecasts over India",
                "Author": "FuXi–IMERG neural adapter project",
                "Subject": "2023 non-confirmatory development validation",
                "Creator": "make_multi_experiment_figure.py",
                "CreationDate": None,
                "ModDate": None,
            },
        )
    finally:
        plt.close(figure)

    script_path = Path(__file__).resolve()
    payload = {
        "status": "derived_from_verified_2023_development_validation",
        "confirmatory": False,
        "test_predictions_evaluated": False,
        "selection_split": "2023_validation",
        "case_count": 93,
        "active_leads": [3, 4, 5, 6],
        "ensemble": {"seeds": [42, 43, 44], "weights": "equal"},
        "bootstrap": {
            "paired": True,
            "block_length_initializations": 13,
            "n_resamples": 2000,
            "seed": 42,
            "interval": "percentile_95_percent",
        },
        "generator": str(script_path),
        "generator_sha256": _sha256(script_path),
        "sources": {
            label: {
                "result_directory": str(results[label].resolve()),
                "hashes": source_hashes[label],
            }
            for label in ("Huber", "Hybrid", "Distribution")
        },
        "outputs": {
            "png": {"path": str(png_path), "sha256": _sha256(png_path)},
            "pdf": {"path": str(pdf_path), "sha256": _sha256(pdf_path)},
            "weekly_table": {
                "path": str(weekly_path),
                "sha256": _sha256(weekly_path),
            },
            "interval_table": {
                "path": str(intervals_path),
                "sha256": _sha256(intervals_path),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "png": png_path,
        "pdf": pdf_path,
        "weekly_table": weekly_path,
        "interval_table": intervals_path,
        "manifest": manifest_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--huber-result", type=Path, default=DEFAULT_RESULTS["Huber"])
    parser.add_argument("--hybrid-result", type=Path, default=DEFAULT_RESULTS["Hybrid"])
    parser.add_argument(
        "--distribution-result", type=Path, default=DEFAULT_RESULTS["Distribution"]
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "neural_adapter_experiment_comparison",
        help="Output path without an extension.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = make_figure(
        {
            "Huber": args.huber_result,
            "Hybrid": args.hybrid_result,
            "Distribution": args.distribution_result,
        },
        args.output_stem,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
