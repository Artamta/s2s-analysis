#!/usr/bin/env python3
"""Build a valid-time-matched JJAS FuXi/ERPAS presentation comparison.

The study has two distinct FuXi experiments:

* a retrospective exact-Wednesday ERA5 initialization, when those forecasts
  have been generated; and
* a six-day-old Thursday initialization representing a 5--6 day ERA5
  availability lag at a Wednesday ERPAS issue time.

The validated preceding-Monday FuXi archive is retained as a two-day-lag
bridge. Every score compares exactly the same seven-day valid periods.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


REPO = Path(__file__).resolve().parents[1]
SOURCE_STUDY = Path(
    "/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/"
    "fuxi_erpas_acc_multiseason_2023_2024"
)
SOURCE_SCRIPT = SOURCE_STUDY / "scripts/build_acc_csv.py"
SOURCE_METRICS = SOURCE_STUDY / "metrics/acc_per_case_2023_2024.csv"
ARCHIVE_FUXI = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/forecasts"
)
EXACT_FUXI = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "operational/era5"
)
EXACT_MEMBERS = 5
ARCHIVE_MEMBERS_AVAILABLE = 50
ARCHIVE_MEMBERS_USED = 5

SYSTEMS = {
    "erpas": {
        "label": "ERPAS",
        "color": "#D45532",
        "marker": "o",
    },
    "fuxi_exact": {
        "label": f"FuXi exact-date ERA5 ({EXACT_MEMBERS}-member pilot)",
        "spatial_label": f"FuXi exact-date ERA5\n({EXACT_MEMBERS}-member pilot)",
        "color": "#008F80",
        "marker": "D",
    },
    "fuxi_2day": {
        "label": f"FuXi nearest cycle (2 d old; {ARCHIVE_MEMBERS_USED} members)",
        "spatial_label": (
            f"FuXi nearest cycle\n(2 d old; {ARCHIVE_MEMBERS_USED} members)"
        ),
        "color": "#2878B5",
        "marker": "s",
    },
    "fuxi_6day": {
        "label": f"FuXi ERA5-delay sensitivity (6 d old; {ARCHIVE_MEMBERS_USED} members)",
        "spatial_label": (
            f"FuXi ERA5-delay sensitivity\n(6 d old; {ARCHIVE_MEMBERS_USED} members)"
        ),
        "color": "#7A5195",
        "marker": "^",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "reports/erpas-fuxi-jjas-2023-2024",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--require-exact", action="store_true")
    return parser.parse_args()


def import_source():
    spec = importlib.util.spec_from_file_location("erpas_fuxi_source_study", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delayed_case(case: dict, days: int) -> dict:
    output = dict(case)
    erpas_init = pd.Timestamp(case["erpas_init"]).tz_localize(None)
    output["comparison_init"] = (
        erpas_init - pd.Timedelta(days=days)
    ).strftime("%Y-%m-%dT00:00:00Z")
    return output


def exact_path(init: pd.Timestamp) -> Path:
    stamp = init.strftime("%Y%m%d")
    return (
        EXACT_FUXI
        / stamp
        / f"ens{EXACT_MEMBERS}"
        / "forecasts"
        / f"annual{init.year}"
        / f"{stamp}.nc"
    )


def archive_path(init: pd.Timestamp) -> Path:
    return ARCHIVE_FUXI / f"annual{init.year}" / f"{init:%Y%m%d}.nc"


def validate_remapped_support(
    remapped: np.ndarray, support_mask: np.ndarray, label: str
) -> None:
    """Require finite/nonnegative rainfall only on the verified India support."""
    supported = np.asarray(remapped)[:, np.asarray(support_mask, dtype=bool)]
    if supported.size == 0 or not np.isfinite(supported).all():
        raise ValueError(f"{label} is non-finite on verified India support")
    if float(np.min(supported)) < -1e-6:
        raise ValueError(f"{label} contains negative remapped precipitation")


def mean_selected_members(
    values: xr.DataArray, members_to_average: int
) -> xr.DataArray:
    """Average a deterministic leading subset of an archived ensemble."""
    available = int(values.sizes.get("member", 1))
    if members_to_average < 1 or members_to_average > available:
        raise ValueError(
            f"requested {members_to_average} members from an ensemble containing {available}"
        )
    return values.isel(member=slice(0, members_to_average)).mean("member")


def load_fuxi_window(
    study,
    case: dict,
    path: Path,
    first_lead: int,
    expected_members: int,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    land_support: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_fraction: np.ndarray,
    members_to_average: int | None = None,
) -> np.ndarray:
    init = pd.Timestamp(case["comparison_init"]).tz_localize(None)
    last_lead = first_lead + 27
    with xr.open_dataset(path) as source:
        label = str(source.attrs.get("run_label", ""))
        if expected_members == 50:
            expected_label = "fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
            if label != expected_label:
                raise ValueError(f"{path} has unexpected archive run label {label!r}")
        elif not (
            label.startswith("fuxi_s2s_era5_case_")
            and label.endswith(f"_ens{expected_members}")
        ):
            raise ValueError(f"{path} has unexpected exact-date run label {label!r}")
        if pd.Timestamp(source.init_time.item()) != init:
            raise ValueError(f"{path} initialization does not match {init}")
        if "information_cutoff_time" in source.coords and pd.Timestamp(
            source.information_cutoff_time.item()
        ) != init:
            raise ValueError(f"{path} information cutoff does not match initialization")
        if source.tp.attrs.get("units") != "mm h-1":
            raise ValueError(f"{path} TP units are not mm h-1")
        selected = source.tp.sel(lead_day=slice(first_lead, last_lead)).load()
        if selected.sizes.get("lead_day") != 28:
            raise ValueError(f"{path} does not contain leads {first_lead}-{last_lead}")
        members = int(selected.sizes.get("member", 1))
        if members != expected_members:
            raise ValueError(f"{path} contains {members} members, expected {expected_members}")
        if not np.isfinite(selected.values).all():
            raise ValueError(f"{path} contains missing precipitation")
        selected = mean_selected_members(
            selected, members_to_average or expected_members
        ) * 24.0
        ends = source.valid_time.sel(lead_day=slice(first_lead, last_lead)).values
        starts = source.forecast_period_start.sel(
            lead_day=slice(first_lead, last_lead)
        ).values
        period_ends = source.forecast_period_end.sel(
            lead_day=slice(first_lead, last_lead)
        ).values

    expected = study.core.expected_period_ends(case)
    study.core.assert_periods(ends, expected, f"FuXi {path.name}")
    study.core.assert_periods(period_ends, expected, f"FuXi period-end {path.name}")
    study.core.assert_periods(
        starts, expected - pd.Timedelta(days=1), f"FuXi period-start {path.name}"
    )
    source_support = study.core.remap_support_fraction(
        *land_support, selected.latitude.values, selected.longitude.values
    )
    remapped, _, _, checks = study.core.remap_conservative(
        selected.values,
        selected.latitude.values,
        selected.longitude.values,
        target_lat,
        target_lon,
        support=source_support,
    )
    if not checks["full_target_coverage"]:
        raise ValueError(f"{path} failed remapping coverage checks")
    validate_remapped_support(remapped, reference_fraction > 0, str(path))
    return remapped.reshape(4, 7, len(target_lat), len(target_lon)).mean(axis=1)


def moving_block_indices(
    cases: pd.DataFrame, samples: int, block_length: int, seed: int = 20260818
) -> np.ndarray:
    """Year-stratified circular moving-block resamples of case indices."""
    if samples < 100 or block_length < 1:
        raise ValueError("bootstrap requires >=100 samples and a positive block length")
    generator = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    for _, group in cases.reset_index(drop=True).groupby("year", sort=True):
        positions = group.index.to_numpy(dtype=int)
        count = len(positions)
        blocks = int(np.ceil(count / block_length))
        starts = generator.integers(0, count, size=(samples, blocks))
        offsets = np.arange(block_length, dtype=int)
        local = (starts[..., None] + offsets) % count
        local = local.reshape(samples, -1)[:, :count]
        pieces.append(positions[local])
    return np.concatenate(pieces, axis=1)


def interval(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) != indices.shape[1] or not np.isfinite(values).all():
        raise ValueError("bootstrap values are incomplete or misaligned")
    boot = values[indices].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize(frame: pd.DataFrame, case_table: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    case_order = case_table.erpas_init.tolist()
    for (system, week), group in frame.groupby(["system", "week"], sort=False):
        ordered = group.set_index("erpas_init").loc[case_order]
        for metric in ("acc", "rmse_mm_day", "bias_mm_day"):
            mean, low, high = interval(ordered[metric].to_numpy(), indices)
            rows.append(
                {
                    "system": system,
                    "week": int(week),
                    "metric": metric,
                    "n_cases": len(case_order),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def paired_differences(
    frame: pd.DataFrame, case_table: pd.DataFrame, indices: np.ndarray
) -> pd.DataFrame:
    rows: list[dict] = []
    case_order = case_table.erpas_init.tolist()
    for week in range(1, 5):
        subset = frame[frame.week.eq(week)]
        pivot = subset.pivot(index="erpas_init", columns="system")
        for system in [name for name in SYSTEMS if name != "erpas" and name in frame.system.unique()]:
            for metric, positive_definition in (
                ("acc", "system_minus_erpas"),
                ("rmse_mm_day", "erpas_minus_system"),
            ):
                left = pivot[metric][system].loc[case_order].to_numpy(dtype=float)
                right = pivot[metric]["erpas"].loc[case_order].to_numpy(dtype=float)
                values = left - right if metric == "acc" else right - left
                mean, low, high = interval(values, indices)
                rows.append(
                    {
                        "system": system,
                        "week": week,
                        "metric": metric,
                        "positive_definition": positive_definition,
                        "n_cases": len(case_order),
                        "mean_difference": mean,
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


def plot_acc_story(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    figure, (curve, gain) = plt.subplots(
        1, 2, figsize=(14.2, 6.8), gridspec_kw={"width_ratios": (1.55, 1.0)}
    )
    present = [name for name in SYSTEMS if name in summary.system.unique()]
    for system in present:
        values = summary[(summary.system == system) & (summary.metric == "acc")].sort_values("week")
        style = SYSTEMS[system]
        curve.plot(
            values.week,
            values["mean"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.5,
            markersize=6.5,
            label=style["label"],
        )
        curve.fill_between(
            values.week, values.ci95_low, values.ci95_high,
            color=style["color"], alpha=0.10,
        )
    curve.set_title("a  JJAS-average forecast skill", loc="left", weight="bold")
    curve.set_ylabel("Mean spatial ACC against IMD")
    curve.set_xlabel("Lead week")
    curve.set_xticks(range(1, 5), [f"W{week}" for week in range(1, 5)])
    curve.set_ylim(-0.05, 0.72)
    curve.legend(frameon=False, fontsize=9, loc="upper right")
    style_axis(curve)

    for system in [name for name in present if name != "erpas"]:
        values = paired[(paired.system == system) & (paired.metric == "acc")].sort_values("week")
        style = SYSTEMS[system]
        effect = values.mean_difference.to_numpy(dtype=float)
        low = values.ci95_low.to_numpy(dtype=float)
        high = values.ci95_high.to_numpy(dtype=float)
        gain.errorbar(
            values.week,
            effect,
            yerr=np.vstack((effect - low, high - effect)),
            color=style["color"],
            marker=style["marker"],
            linewidth=2.1,
            capsize=4,
            label=style["label"],
        )
    gain.axhline(0, color="#475569", linewidth=1.1, linestyle="--")
    gain.axhspan(0, 0.4, color="#2C9C69", alpha=0.045)
    gain.set_title("b  Paired ACC advantage over ERPAS", loc="left", weight="bold")
    gain.set_ylabel("FuXi − ERPAS ACC")
    gain.set_xlabel("Lead week")
    gain.set_xticks(range(1, 5), [f"W{week}" for week in range(1, 5)])
    gain.legend(frameon=False, fontsize=8.3, loc="best")
    style_axis(gain)

    figure.suptitle(
        "JJAS spatial ACC: FuXi-S2S versus ERPAS",
        x=0.07, y=0.98, ha="left", fontsize=20, weight="bold", color="#17232E",
    )
    figure.text(
        0.07, 0.925,
        "India · JJAS 2023–2024 · 31 paired initialization dates · identical valid weeks",
        ha="left", fontsize=10.5, color="#52616E",
    )
    figure.text(
        0.98, 0.925,
        "RETROSPECTIVE / DELAY SENSITIVITY · NOT LIVE OPERATIONAL VALIDATION",
        ha="right", fontsize=8.1, weight="bold", color="#8C2D2D",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#FFF3F1", "edgecolor": "#D9A29C"},
    )
    figure.text(
        0.5, 0.025,
        "Shading/error bars: year-stratified 4-start moving-block 95% percentile intervals. "
        "The 6-day curve aligns FuXi leads 8–35 to ERPAS Weeks 1–4.",
        ha="center", fontsize=8.8, color="#536271",
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.84, wspace=0.22)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_three_metrics(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.4, 5.8))
    specs = (
        ("rmse_mm_day", "RMSE (mm day⁻¹) ↓"),
        ("acc", "Spatial ACC ↑"),
        ("bias_mm_day", "Bias (mm day⁻¹; zero is best)"),
    )
    present = [name for name in SYSTEMS if name in summary.system.unique()]
    for axis, (metric, ylabel) in zip(axes, specs, strict=True):
        for system in present:
            values = summary[(summary.system == system) & (summary.metric == metric)].sort_values("week")
            style = SYSTEMS[system]
            axis.plot(
                values.week, values["mean"], color=style["color"],
                marker=style["marker"], linewidth=2.3, markersize=6,
                label=style["label"],
            )
            axis.fill_between(
                values.week, values.ci95_low, values.ci95_high,
                color=style["color"], alpha=0.09,
            )
        if metric == "bias_mm_day":
            axis.axhline(0, color="#46515C", linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Lead week")
        axis.set_xticks(range(1, 5))
        style_axis(axis)
    axes[1].legend(
        handles=axes[1].get_legend_handles_labels()[0],
        labels=axes[1].get_legend_handles_labels()[1],
        frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2,
    )
    figure.suptitle(
        "FuXi-S2S and ERPAS rainfall skill over India · JJAS 2023–2024",
        fontsize=17, weight="bold", y=1.02,
    )
    figure.text(
        0.5, 0.015,
        "31 paired cases; IMD 1991–2020 climatology; identical verification dates for every system.",
        ha="center", fontsize=9, color="#536271",
    )
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.13, top=0.79, wspace=0.25)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_composites(
    composites: dict[str, np.ndarray],
    latitude: np.ndarray,
    longitude: np.ndarray,
    mask: np.ndarray,
    output: Path,
) -> None:
    columns = ["observed"] + [name for name in SYSTEMS if name in composites]
    weeks = (1, 3)
    values = np.stack(
        [np.where(mask, composites[name][week - 1], np.nan) for week in weeks for name in columns]
    )
    limit = max(1.0, float(np.nanpercentile(np.abs(values), 98)))
    figure, axes = plt.subplots(
        len(weeks), len(columns), figsize=(3.05 * len(columns), 6.8), squeeze=False,
        sharex=True, sharey=True,
    )
    image = None
    for row, week in enumerate(weeks):
        for column, name in enumerate(columns):
            axis = axes[row, column]
            field = np.where(mask, composites[name][week - 1], np.nan)
            image = axis.pcolormesh(
                longitude, latitude, field, shading="auto", cmap="RdBu_r",
                vmin=-limit, vmax=limit,
            )
            if row == 0:
                title = (
                    "IMD observed"
                    if name == "observed"
                    else SYSTEMS[name].get("spatial_label", SYSTEMS[name]["label"])
                )
                axis.set_title(title, fontsize=8.8, weight="semibold")
            if column == 0:
                axis.set_ylabel(f"W{week}\nLatitude")
            if row == len(weeks) - 1:
                axis.set_xlabel("Longitude")
            axis.grid(color="white", alpha=0.18, linewidth=0.4)
    figure.suptitle(
        "JJAS composite rainfall anomalies against IMD climatology",
        fontsize=17, weight="bold", y=0.98,
    )
    figure.text(
        0.5, 0.91,
        "31 paired cases · common color scale · identical valid dates",
        ha="center", fontsize=9.5, color="#536271",
    )
    colorbar = figure.colorbar(image, ax=axes, orientation="horizontal", fraction=0.055, pad=0.11)
    colorbar.set_label("Rainfall anomaly (mm day⁻¹)")
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.17, top=0.85, wspace=0.08, hspace=0.15)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.case_limit is not None and args.case_limit < 2:
        raise ValueError("case-limit must be at least two")
    study = import_source()
    all_cases, _ = study.build_cases()
    current_cases = [
        case for case in all_cases
        if pd.Timestamp(case["erpas_init"]).month in study.SEASON_WINDOWS["JJAS"]
    ]
    if len(current_cases) != 31:
        raise ValueError(f"expected 31 JJAS pairs, found {len(current_cases)}")
    if args.case_limit is not None:
        current_cases = current_cases[: args.case_limit]

    delayed_cases = [delayed_case(case, 6) for case in current_cases]
    exact_cases = [delayed_case(case, 0) for case in current_cases]
    missing_delayed = [
        str(archive_path(pd.Timestamp(case["comparison_init"])))
        for case in delayed_cases
        if not archive_path(pd.Timestamp(case["comparison_init"])).is_file()
    ]
    if missing_delayed:
        raise FileNotFoundError(f"missing delayed FuXi forecasts: {missing_delayed}")
    exact_available = all(
        exact_path(pd.Timestamp(case["comparison_init"])).is_file() for case in exact_cases
    )
    if args.require_exact and not exact_available:
        missing = [
            str(exact_path(pd.Timestamp(case["comparison_init"])))
            for case in exact_cases
            if not exact_path(pd.Timestamp(case["comparison_init"])).is_file()
        ]
        raise FileNotFoundError(f"exact-date FuXi forecasts are incomplete: {missing}")

    config = json.loads(study.BASE_CONFIG.read_text(encoding="utf-8"))
    config["model_roots"]["erpas"] = str(study.ERPAS_ROOT)
    with xr.open_dataset(study.SOURCE_DATA) as source:
        reference = source.load()
    target_lat, target_lon, india_fraction, weight, land_support = study.accmod.load_land_support(reference)
    original_mask = weight > 0
    observed, imd_climatology, india_fraction, weight, land_support, imd_audit = study.remap_imd(
        all_cases, target_lat, target_lon, land_support, original_mask
    )
    mask = weight > 0

    rows: list[dict] = []
    composite_sums: dict[str, np.ndarray] = {
        "observed": np.zeros((4, len(target_lat), len(target_lon)), dtype=np.float64),
        "erpas": np.zeros((4, len(target_lat), len(target_lon)), dtype=np.float64),
        "fuxi_2day": np.zeros((4, len(target_lat), len(target_lon)), dtype=np.float64),
        "fuxi_6day": np.zeros((4, len(target_lat), len(target_lon)), dtype=np.float64),
    }
    if exact_available:
        composite_sums["fuxi_exact"] = np.zeros_like(composite_sums["observed"])

    for index, (current, delayed, exact) in enumerate(
        zip(current_cases, delayed_cases, exact_cases, strict=True), 1
    ):
        erpas_init = pd.Timestamp(current["erpas_init"]).tz_localize(None)
        obs_weekly, imd_clim_weekly = study.weekly_reference(
            current, observed, imd_climatology
        )
        erpas_weekly, _ = study.load_erpas_variable_count(
            config, current, target_lat, target_lon, land_support, india_fraction
        )
        forecasts = {
            "erpas": erpas_weekly / 7.0,
            "fuxi_2day": load_fuxi_window(
                study, current,
                archive_path(pd.Timestamp(current["comparison_init"])),
                4, ARCHIVE_MEMBERS_AVAILABLE, target_lat, target_lon,
                land_support, india_fraction, ARCHIVE_MEMBERS_USED,
            ),
            "fuxi_6day": load_fuxi_window(
                study, delayed,
                archive_path(pd.Timestamp(delayed["comparison_init"])),
                8, ARCHIVE_MEMBERS_AVAILABLE, target_lat, target_lon,
                land_support, india_fraction, ARCHIVE_MEMBERS_USED,
            ),
        }
        if exact_available:
            forecasts["fuxi_exact"] = load_fuxi_window(
                study, exact, exact_path(pd.Timestamp(exact["comparison_init"])),
                2, EXACT_MEMBERS, target_lat, target_lon, land_support, india_fraction,
            )
        observed_anomaly = obs_weekly - imd_clim_weekly
        composite_sums["observed"] += observed_anomaly
        for system, forecast_weekly in forecasts.items():
            forecast_anomaly = forecast_weekly - imd_clim_weekly
            composite_sums[system] += forecast_anomaly
            for week_index in range(4):
                error = forecast_weekly[week_index] - obs_weekly[week_index]
                score = study.engine.error_metrics(error, weight)
                rows.append(
                    {
                        "case_id": current["case_id"],
                        "erpas_init": erpas_init.strftime("%Y-%m-%d"),
                        "year": erpas_init.year,
                        "week": week_index + 1,
                        "valid_period_start": (
                            erpas_init + pd.Timedelta(days=1 + 7 * week_index)
                        ).strftime("%Y-%m-%d"),
                        "valid_period_end_exclusive": (
                            erpas_init + pd.Timedelta(days=8 + 7 * week_index)
                        ).strftime("%Y-%m-%d"),
                        "system": system,
                        "acc": study.engine.anomaly_correlation(
                            forecast_anomaly[week_index], observed_anomaly[week_index], weight
                        ),
                        "rmse_mm_day": score["rmse"],
                        "mae_mm_day": score["mae"],
                        "bias_mm_day": score["bias"],
                    }
                )
        print(f"processed {index}/{len(current_cases)} paired JJAS cases", flush=True)

    frame = pd.DataFrame(rows).sort_values(["erpas_init", "system", "week"])
    expected_systems = {"erpas", "fuxi_2day", "fuxi_6day"}
    if exact_available:
        expected_systems.add("fuxi_exact")
    if set(frame.system.unique()) != expected_systems:
        raise ValueError("system coverage is incomplete")
    coverage = frame.groupby(["erpas_init", "week"]).system.nunique()
    if not (coverage == len(expected_systems)).all():
        raise ValueError("a case/week does not contain every system")

    base = pd.read_csv(SOURCE_METRICS)
    base = base[
        base.method.eq("common_imd_1991_2020")
        & base.season_memberships.str.contains("JJAS")
    ].copy()
    base["system"] = base.model.map({"ERPAS": "erpas", "FuXi-S2S": "fuxi_2day"})
    base = base[base.erpas_init.isin(frame.erpas_init.unique())]
    comparison = frame[frame.system.eq("erpas")].merge(
        base[["erpas_init", "week", "system", "acc", "rmse_mm_day", "mae_mm_day", "bias_mm_day"]],
        on=["erpas_init", "week", "system"], suffixes=("_new", "_saved"), validate="one_to_one",
    )
    reproduction_max = {
        metric: float(np.max(np.abs(comparison[f"{metric}_new"] - comparison[f"{metric}_saved"])))
        for metric in ("acc", "rmse_mm_day", "mae_mm_day", "bias_mm_day")
    }
    if any(value > 2e-6 for value in reproduction_max.values()):
        raise ValueError(f"saved-study reproduction failed: {reproduction_max}")

    case_table = (
        frame[["erpas_init", "year"]].drop_duplicates().sort_values("erpas_init").reset_index(drop=True)
    )
    indices = moving_block_indices(
        case_table, args.bootstrap_samples, args.block_length
    )
    summary = summarize(frame, case_table, indices)
    paired = paired_differences(frame, case_table, indices)
    composites = {
        name: values / len(current_cases) for name, values in composite_sums.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_case_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_by_system_week.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    plot_acc_story(summary, paired, args.output_dir / "jjas_erpas_fuxi_acc_story.png")
    plot_three_metrics(summary, args.output_dir / "jjas_erpas_fuxi_acc_rmse_bias.png")
    plot_spatial_composites(
        composites, target_lat, target_lon, mask,
        args.output_dir / "jjas_erpas_fuxi_composite_anomalies.png",
    )
    outputs = sorted(args.output_dir.glob("*.csv")) + sorted(args.output_dir.glob("*.png"))
    checks = {
        "case_count": len(case_table),
        "case_count_expected": args.case_limit or 31,
        "delayed_cycles_available": len(missing_delayed) == 0,
        "exact_date_forecasts_included": exact_available,
        "exact_requirement_satisfied": exact_available or not args.require_exact,
        "every_case_week_has_every_system": bool((coverage == len(expected_systems)).all()),
        "saved_erpAS_metrics_reproduced_within_2e_6": all(
            value <= 2e-6 for value in reproduction_max.values()
        ),
        "acc_in_bounds": bool(frame.acc.between(-1, 1).all()),
        "errors_finite_nonnegative": bool(
            np.isfinite(frame[["rmse_mm_day", "mae_mm_day"]]).all().all()
            and (frame[["rmse_mm_day", "mae_mm_day"]] >= 0).all().all()
        ),
    }
    required_checks = (
        checks["case_count"] == checks["case_count_expected"]
        and checks["delayed_cycles_available"]
        and checks["exact_requirement_satisfied"]
        and checks["every_case_week_has_every_system"]
        and checks["saved_erpAS_metrics_reproduced_within_2e_6"]
        and checks["acc_in_bounds"]
        and checks["errors_finite_nonnegative"]
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASSED" if required_checks else "FAILED",
        "scope": "JJAS 2023-2024 paired ERPAS/FuXi rainfall verification against IMD",
        "case_contract": {
            "erpas_issue": "Wednesday 00 UTC",
            "exact_fuxi": (
                "same Wednesday 00 UTC; retrospective ERA5 initialization; "
                f"{EXACT_MEMBERS} members"
            ),
            "nearest_archive_fuxi": (
                "preceding Monday 00 UTC; two-day-old initialization; first 5 members "
                "selected deterministically from the 50-member archive"
            ),
            "era5_delay_fuxi": (
                "preceding Thursday 00 UTC; six-day-old initialization approximating "
                "a 5-6 day ERA5 availability delay; first 5 members selected "
                "deterministically from the 50-member archive"
            ),
            "verification": "all systems use ERPAS issue +1 through +28 days, grouped into four disjoint seven-day weeks",
        },
        "uncertainty": {
            "method": "year-stratified circular moving-block percentile bootstrap",
            "samples": args.bootstrap_samples,
            "block_length_initializations": args.block_length,
            "interpretation": "descriptive paired uncertainty, not independent prospective confirmation",
        },
        "climatology": "common IMD 1991-2020 calendar-day climatology for forecast and observed anomalies",
        "source_study": str(SOURCE_STUDY),
        "source_metrics_sha256": sha256_file(SOURCE_METRICS),
        "members_per_fuxi_source": EXACT_MEMBERS,
        "archive_members_available": ARCHIVE_MEMBERS_AVAILABLE,
        "archive_member_selection": "first 5 members, preserving stored member order",
        "imd_remap_audit": imd_audit,
        "reproduction_max_absolute_difference": reproduction_max,
        "checks": checks,
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if manifest["status"] != "PASSED":
        raise ValueError(f"output contract failed: {checks}")
    print(json.dumps({"status": "PASSED", "output_dir": str(args.output_dir), **checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
