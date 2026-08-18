#!/usr/bin/env python3
"""Build the 31-case IMERG verification of FuXi ERA5-init timing.

This is a meeting-focused companion to ``build_erpas_fuxi_jjas_story.py``.
It retains exactly the 31 audited IMERG Final V07B verification cases and
their Thursday--Wednesday valid weeks, while comparing ERPAS with three
five-member FuXi-S2S ERA5-initialization timings: exact date, two days old,
and six days old.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meeting-imerg-delay-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/meeting-imerg-delay-cache")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


REPO = Path(__file__).resolve().parents[1]
DELAY_SCRIPT = REPO / "scripts/build_erpas_fuxi_jjas_story.py"
IMERG_REVIEW = Path(
    "/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/"
    "fuxi_erpas_imd_imerg_review_2023_2024"
)
REVIEW_FIELDS = IMERG_REVIEW / "data/processed/review_fields_2023_2024.nc"
REVIEW_AUDIT = IMERG_REVIEW / "logs/method_audit.json"
DEFAULT_OUTPUT = REPO / "reports/meeting-imerg-delay-20260818"

SYSTEMS = {
    "erpas": {
        "label": "ERPAS provider ensemble mean",
        "color": "#D45532",
        "marker": "o",
    },
    "fuxi_exact": {
        "label": "FuXi · exact-date ERA5 · 5 members",
        "color": "#008F80",
        "marker": "D",
    },
    "fuxi_2day": {
        "label": "FuXi · ERA5 2 days old · first 5 members",
        "color": "#2878B5",
        "marker": "s",
    },
    "fuxi_6day": {
        "label": "FuXi · ERA5 6 days old · first 5 members",
        "color": "#7A5195",
        "marker": "^",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--case-limit", type=int)
    return parser.parse_args()


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (indices.shape[1],) or not np.isfinite(values).all():
        raise ValueError("bootstrap values are incomplete or misaligned")
    resampled = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(resampled, 2.5)),
        float(np.percentile(resampled, 97.5)),
    )


def summarize(
    metrics: pd.DataFrame, case_table: pd.DataFrame, indices: np.ndarray
) -> pd.DataFrame:
    order = case_table.erpas_init.tolist()
    rows: list[dict] = []
    for system in SYSTEMS:
        for week in range(1, 5):
            subset = metrics[
                metrics.system.eq(system) & metrics.week.eq(week)
            ].set_index("erpas_init").loc[order]
            for metric in ("acc", "rmse_mm_day"):
                mean, low, high = interval(subset[metric].to_numpy(dtype=float), indices)
                rows.append(
                    {
                        "system": system,
                        "week": week,
                        "metric": metric,
                        "n_cases": len(order),
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def paired_differences(
    metrics: pd.DataFrame, case_table: pd.DataFrame, indices: np.ndarray
) -> pd.DataFrame:
    order = case_table.erpas_init.tolist()
    rows: list[dict] = []
    for week in range(1, 5):
        pivot = metrics[metrics.week.eq(week)].pivot(
            index="erpas_init", columns="system", values=["acc", "rmse_mm_day"]
        )
        for system in ("fuxi_exact", "fuxi_2day", "fuxi_6day"):
            for metric, definition in (
                ("acc", "fuxi_minus_erpas"),
                ("rmse_mm_day", "erpas_minus_fuxi"),
            ):
                fuxi = pivot[metric][system].loc[order].to_numpy(dtype=float)
                erpas = pivot[metric]["erpas"].loc[order].to_numpy(dtype=float)
                improvement = fuxi - erpas if metric == "acc" else erpas - fuxi
                mean, low, high = interval(improvement, indices)
                rows.append(
                    {
                        "system": system,
                        "week": week,
                        "metric": metric,
                        "positive_definition": definition,
                        "n_cases": len(order),
                        "mean_improvement": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "interval_excludes_zero": bool(low > 0 or high < 0),
                    }
                )
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#AAB5C0", alpha=0.28, linewidth=0.8)
    axis.tick_params(colors="#263442")
    axis.set_xticks(range(1, 5), [f"W{week}" for week in range(1, 5)])
    axis.set_xlabel("Lead week")


def plot_figure(summary: pd.DataFrame, output: Path, case_count: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15.8, 6.8))
    specifications = (
        ("acc", "Spatial anomaly correlation (ACC) ↑", (-0.05, 0.72)),
        ("rmse_mm_day", "RMSE (mm day⁻¹) ↓", None),
    )
    for panel, (axis, (metric, ylabel, ylim)) in enumerate(
        zip(axes, specifications, strict=True)
    ):
        for system, style in SYSTEMS.items():
            values = summary[
                summary.system.eq(system) & summary.metric.eq(metric)
            ].sort_values("week")
            axis.plot(
                values.week,
                values["mean"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.65,
                markersize=7,
                label=style["label"],
                zorder=3,
            )
            axis.fill_between(
                values.week,
                values.ci95_low,
                values.ci95_high,
                color=style["color"],
                alpha=0.105,
                linewidth=0,
                zorder=1,
            )
        axis.set_title(
            f"{'a' if panel == 0 else 'b'}  {'Pattern skill' if panel == 0 else 'Rainfall error'}",
            loc="left",
            fontsize=14,
            weight="bold",
        )
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        style_axis(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=2,
        frameon=False,
        fontsize=10.2,
        columnspacing=2.1,
        handlelength=2.8,
    )
    figure.suptitle(
        "FuXi-S2S rainfall skill is sensitive to ERA5 initialization timing",
        x=0.055,
        y=0.982,
        ha="left",
        fontsize=21,
        weight="bold",
        color="#17232E",
    )
    figure.text(
        0.055,
        0.932,
        f"India · JJAS 2023–2024 · {case_count} paired starts · IMERG Final V07B verification · identical valid weeks",
        ha="left",
        fontsize=11,
        color="#52616E",
    )
    figure.text(
        0.965,
        0.938,
        "RETROSPECTIVE DELAY SENSITIVITY",
        ha="right",
        fontsize=8.6,
        weight="bold",
        color="#8C2D2D",
        bbox={
            "boxstyle": "round,pad=0.34",
            "facecolor": "#FFF3F1",
            "edgecolor": "#D9A29C",
        },
    )
    figure.text(
        0.5,
        0.035,
        "Curves: arithmetic mean of case-wise scores. Shading: paired, year-stratified 4-start moving-block 95% percentile intervals.\n"
        "ACC anomalies use one common IMERG 2001–2022 calendar-day baseline; RMSE uses raw weekly rainfall. FuXi curves use five members.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#536271",
        linespacing=1.45,
    )
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.17, top=0.76, wspace=0.22)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    if args.block_length < 1:
        raise ValueError("block-length must be positive")
    if args.case_limit is not None and not 2 <= args.case_limit <= 31:
        raise ValueError("case-limit must be between 2 and 31")
    if not REVIEW_FIELDS.is_file() or not REVIEW_AUDIT.is_file():
        raise FileNotFoundError("the audited 31-case IMERG review inputs are missing")

    delay = import_file("meeting_imerg_delay_support", DELAY_SCRIPT)
    study = delay.import_source()
    all_cases, _ = study.build_cases()
    cases = [
        case
        for case in all_cases
        if pd.Timestamp(case["erpas_init"]).month in study.SEASON_WINDOWS["JJAS"]
    ]
    if len(cases) != 31:
        raise ValueError(f"expected 31 JJAS cases, found {len(cases)}")
    if args.case_limit is not None:
        cases = cases[: args.case_limit]

    exact_cases = [delay.delayed_case(case, 0) for case in cases]
    delayed_cases = [delay.delayed_case(case, 6) for case in cases]
    source_paths: dict[str, list[str]] = {
        "fuxi_exact": [],
        "fuxi_2day": [],
        "fuxi_6day": [],
    }
    for current, exact, old in zip(cases, exact_cases, delayed_cases, strict=True):
        paths = {
            "fuxi_exact": delay.exact_path(pd.Timestamp(exact["comparison_init"])),
            "fuxi_2day": delay.archive_path(pd.Timestamp(current["comparison_init"])),
            "fuxi_6day": delay.archive_path(pd.Timestamp(old["comparison_init"])),
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing forecast source(s): {missing}")
        for system, path in paths.items():
            source_paths[system].append(str(path))

    review_audit = json.loads(REVIEW_AUDIT.read_text(encoding="utf-8"))
    if review_audit.get("status") != "PASSED":
        raise ValueError("upstream IMERG review audit is not PASSED")

    with xr.open_dataset(REVIEW_FIELDS) as source:
        review = source.load()
    if "IMERG Final V07B" not in review.reference.values:
        raise ValueError("IMERG Final V07B is absent from the review dataset")
    review_cases = review.case.values.astype(str).tolist()
    expected_case_ids = [case["case_id"] for case in cases]
    case_positions = [review_cases.index(case_id) for case_id in expected_case_ids]
    subset = review.isel(case=case_positions)
    if subset.case.values.astype(str).tolist() != expected_case_ids:
        raise ValueError("review case order does not match the delay experiment")

    target_lat = subset.latitude.values.astype(float)
    target_lon = subset.longitude.values.astype(float)
    weight = subset.spatial_weight.values.astype(float)
    india_fraction = subset.india_fraction.values.astype(float)
    mask = weight > 0
    if int(mask.sum()) != 169:
        raise ValueError(f"expected the audited 169-cell support, found {int(mask.sum())}")

    with xr.open_dataset(study.SOURCE_DATA) as source:
        reference = source.load()
    source_lat, source_lon, _, original_weight, original_land_support = (
        study.accmod.load_land_support(reference)
    )
    if not (
        np.array_equal(source_lat, target_lat) and np.array_equal(source_lon, target_lon)
    ):
        raise ValueError("the source-study and IMERG-review target grids differ")
    _, _, remapped_fraction, remapped_weight, land_support, _ = study.remap_imd(
        all_cases,
        target_lat,
        target_lon,
        original_land_support,
        original_weight > 0,
    )
    if not (
        np.allclose(remapped_fraction, india_fraction, rtol=0, atol=2e-7)
        and np.allclose(remapped_weight, weight, rtol=0, atol=2e-12)
    ):
        raise ValueError("reconstructed fixed India support differs from IMERG review")

    truth_raw = subset.observed_weekly_rainfall.sel(
        reference="IMERG Final V07B"
    ).values.astype(float)
    truth_anomaly = subset.observed_weekly_anomaly.sel(
        reference="IMERG Final V07B"
    ).values.astype(float)
    imerg_baseline = truth_raw - truth_anomaly
    erpas_raw = subset.forecast_weekly_rainfall.sel(model="ERPAS").values.astype(float)
    expected_starts = subset.week_start.values.astype("datetime64[ns]")
    expected_ends = subset.week_end_exclusive.values.astype("datetime64[ns]")
    rows: list[dict] = []
    valid_windows_match = True

    for case_index, (current, exact, old) in enumerate(
        zip(cases, exact_cases, delayed_cases, strict=True)
    ):
        erpas_init = pd.Timestamp(current["erpas_init"]).tz_localize(None)
        calculated_starts = np.asarray(
            [
                (erpas_init + pd.Timedelta(days=1 + 7 * week)).to_datetime64()
                for week in range(4)
            ],
            dtype="datetime64[ns]",
        )
        calculated_ends = calculated_starts + np.timedelta64(7, "D")
        valid_windows_match &= bool(
            np.array_equal(calculated_starts, expected_starts[case_index])
            and np.array_equal(calculated_ends, expected_ends[case_index])
        )

        forecasts = {
            "erpas": erpas_raw[case_index],
            "fuxi_exact": delay.load_fuxi_window(
                study,
                exact,
                Path(source_paths["fuxi_exact"][case_index]),
                2,
                delay.EXACT_MEMBERS,
                target_lat,
                target_lon,
                land_support,
                india_fraction,
            ),
            "fuxi_2day": delay.load_fuxi_window(
                study,
                current,
                Path(source_paths["fuxi_2day"][case_index]),
                4,
                delay.ARCHIVE_MEMBERS_AVAILABLE,
                target_lat,
                target_lon,
                land_support,
                india_fraction,
                delay.ARCHIVE_MEMBERS_USED,
            ),
            "fuxi_6day": delay.load_fuxi_window(
                study,
                old,
                Path(source_paths["fuxi_6day"][case_index]),
                8,
                delay.ARCHIVE_MEMBERS_AVAILABLE,
                target_lat,
                target_lon,
                land_support,
                india_fraction,
                delay.ARCHIVE_MEMBERS_USED,
            ),
        }
        for system, forecast in forecasts.items():
            if forecast.shape != (4, len(target_lat), len(target_lon)):
                raise ValueError(f"unexpected {system} shape for {current['case_id']}")
            forecast_anomaly = forecast - imerg_baseline[case_index]
            for week_index in range(4):
                error = forecast[week_index] - truth_raw[case_index, week_index]
                error_scores = study.engine.error_metrics(error, weight)
                rows.append(
                    {
                        "case_id": current["case_id"],
                        "erpas_init": erpas_init.strftime("%Y-%m-%d"),
                        "year": erpas_init.year,
                        "week": week_index + 1,
                        "valid_period_start": str(
                            pd.Timestamp(calculated_starts[week_index]).date()
                        ),
                        "valid_period_end_exclusive": str(
                            pd.Timestamp(calculated_ends[week_index]).date()
                        ),
                        "reference": "IMERG Final V07B",
                        "system": system,
                        "fuxi_member_count": 0 if system == "erpas" else 5,
                        "acc": study.engine.anomaly_correlation(
                            forecast_anomaly[week_index],
                            truth_anomaly[case_index, week_index],
                            weight,
                        ),
                        "rmse_mm_day": error_scores["rmse"],
                    }
                )
        print(f"processed {case_index + 1}/{len(cases)} IMERG cases", flush=True)

    metrics = pd.DataFrame(rows).sort_values(["erpas_init", "system", "week"])
    case_table = (
        metrics[["erpas_init", "year"]]
        .drop_duplicates()
        .sort_values("erpas_init")
        .reset_index(drop=True)
    )
    indices = delay.moving_block_indices(
        case_table, args.bootstrap_samples, args.block_length
    )
    summary = summarize(metrics, case_table, indices)
    paired = paired_differences(metrics, case_table, indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "per_case_metrics.csv"
    summary_path = args.output_dir / "summary_by_system_week.csv"
    paired_path = args.output_dir / "paired_differences_vs_erpas.csv"
    figure_path = args.output_dir / "imerg_acc_rmse_era5_delay_31cases.png"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    plot_figure(summary, figure_path, len(case_table))

    coverage = metrics.groupby(["case_id", "week"]).system.nunique()
    year_counts = case_table.groupby("year").size().to_dict()
    checks = {
        "upstream_imerg_audit_passed": review_audit.get("status") == "PASSED",
        "case_count": len(case_table),
        "case_count_expected": args.case_limit or 31,
        "year_counts": {str(year): int(count) for year, count in year_counts.items()},
        "full_sample_year_counts_17_14": args.case_limit is not None
        or year_counts == {2023: 17, 2024: 14},
        "expected_metric_rows": len(metrics) == len(case_table) * 4 * len(SYSTEMS),
        "four_systems_every_case_week": bool((coverage == len(SYSTEMS)).all()),
        "identical_saved_valid_windows": bool(valid_windows_match),
        "fuxi_member_count_exactly_5": bool(
            (metrics.loc[metrics.system.ne("erpas"), "fuxi_member_count"] == 5).all()
        ),
        "fixed_india_support_169_cells": int(mask.sum()) == 169,
        "acc_finite_and_bounded": bool(
            np.isfinite(metrics.acc).all() and metrics.acc.between(-1, 1).all()
        ),
        "rmse_finite_and_nonnegative": bool(
            np.isfinite(metrics.rmse_mm_day).all() and (metrics.rmse_mm_day >= 0).all()
        ),
        "summary_complete": len(summary) == len(SYSTEMS) * 4 * 2,
        "paired_difference_table_complete": len(paired) == 3 * 4 * 2,
        "figure_nonempty": figure_path.is_file() and figure_path.stat().st_size > 100_000,
    }
    status = "PASSED" if all(
        value
        for key, value in checks.items()
        if key not in {"case_count", "case_count_expected", "year_counts"}
    ) and checks["case_count"] == checks["case_count_expected"] else "FAILED"
    outputs = [metrics_path, summary_path, paired_path, figure_path]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "31-case JJAS 2023-2024 ERPAS/FuXi ERA5-initialization timing verification against IMERG Final V07B",
        "case_contract": {
            "erpas_issue": "Wednesday 00 UTC",
            "verification": "ERPAS issue +1 through +28 days in four disjoint Thursday-Wednesday weeks; identical for all systems",
            "fuxi_exact": "same Wednesday initialization; leads 2-29; 5 members",
            "fuxi_2day": "preceding Monday initialization; leads 4-31; deterministic first 5 of 50 archived members",
            "fuxi_6day": "preceding Thursday initialization; leads 8-35; deterministic first 5 of 50 archived members",
        },
        "reference": {
            "product": "GPM IMERG Final V07B",
            "raw_and_anomaly_fields": str(REVIEW_FIELDS),
            "upstream_audit": str(REVIEW_AUDIT),
            "anomaly_baseline": "fixed IMERG Final V07B 2001-2022 calendar-day climatology, used as one common baseline for forecast and observed ACC anomalies",
            "rmse": "area-weighted raw weekly forecast minus IMERG rainfall, mm day-1",
        },
        "uncertainty": {
            "method": "paired year-stratified circular moving-block percentile bootstrap",
            "samples": args.bootstrap_samples,
            "block_length_initializations": args.block_length,
            "seed": 20260818,
            "interpretation": "descriptive retrospective uncertainty; not independent prospective confirmation",
        },
        "source_forecasts": source_paths,
        "checks": checks,
        "outputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "PASSED":
        raise ValueError(f"output checks failed: {checks}")
    print(
        json.dumps(
            {"status": status, "output_dir": str(args.output_dir), "checks": checks},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
