#!/usr/bin/env python3
"""Build publication-ready result tables and comparison figures.

This script consumes the validated weekly metric CSVs and writes a clean
``publication_results`` directory. The lead axis is the standard S2S weekly windows
already used by the verification pipeline: Week 1 through Week 6, days 1-42.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "S2S_OUTPUT_ROOT",
        str(PROJECT_ROOT / "outputs" / "verification"),
    )
)


@dataclass(frozen=True)
class RunConfig:
    season: str
    run_label: str
    publication_case: str
    publication_title: str
    note: str


PUBLICATION_RUNS = (
    RunConfig(
        season="jjas2019",
        run_label="full_jjas2019_operational35_plus_fuxi_tp_imdtruth",
        publication_case="jjas2019_operational35_plus_fuxi_tp_imd",
        publication_title="JJAS2019 Common 35 TP, IMD Truth",
        note=(
            "Common 35 Monday/Thursday init dates for ECMWF/UKMO/NCEP/FuXi/MME; "
            "TP verified against IMD."
        ),
    ),
    RunConfig(
        season="jjas2019",
        run_label="full_jjas2019_operational35_plus_fuxi_tp_era5truth",
        publication_case="jjas2019_operational35_plus_fuxi_tp_era5",
        publication_title="JJAS2019 Common 35 TP, ERA5 Truth",
        note=(
            "Common 35 Monday/Thursday init dates for ECMWF/UKMO/NCEP/FuXi/MME; "
            "TP verified against ERA5 for truth-source sensitivity."
        ),
    ),
    RunConfig(
        season="jjas2019",
        run_label="full_jjas2019_operational35_plus_fuxi_z500",
        publication_case="jjas2019_operational35_plus_fuxi_z500",
        publication_title="JJAS2019 Common 35 Z500",
        note="Common 35 Monday/Thursday init dates for ECMWF/UKMO/NCEP/FuXi/MME; Z500 verified against ERA5.",
    ),
    RunConfig(
        season="jjas2019",
        run_label="full_jjas2019_common17_fuxi_imd",
        publication_case="jjas2019_common17_delysm_sensitivity",
        publication_title="JJAS2019 Common 17 DLESyM Sensitivity",
        note=(
            "Common 17 Thursday init dates used only for the DLESyM-inclusive sensitivity comparison; "
            "TP verified against IMD."
        ),
    ),
    RunConfig(
        season="jfm2026",
        run_label="full_jfm2026_daily_spire",
        publication_case="jfm2026_daily_spire",
        publication_title="JFM2026 Daily Init Dates With SPIRE",
        note="Daily JFM2026 comparison with SPIRE included; ERA5/available ground truth.",
    ),
)


TABLE_NAMES = (
    "deterministic_weekly",
    "probabilistic_weekly",
    "brier_weekly",
    "reliability_weekly",
    "scatter_area_weekly",
)

DETERMINISTIC_METRICS = ("acc", "rmse", "mae", "bias", "mse_skill_clim")
PROBABILISTIC_METRICS = ("crps", "crpss_clim", "spread", "rmse_ensmean", "spread_skill_ratio")
BRIER_METRICS = ("brier", "brier_skill_clim", "base_rate")

MODEL_ORDER = ("ecmwf", "ukmo", "ncep", "fuxi", "delysm", "spire", "mme")
MODEL_LABELS = {
    "ecmwf": "ECMWF",
    "ukmo": "UKMO",
    "ncep": "NCEP",
    "fuxi": "FuXi",
    "delysm": "DLESyM",
    "spire": "SPIRE",
    "mme": "MME",
}
MODEL_COLORS = {
    "ecmwf": "#0072B2",
    "ukmo": "#D55E00",
    "ncep": "#009E73",
    "fuxi": "#CC79A7",
    "delysm": "#E69F00",
    "spire": "#56B4E9",
    "mme": "#111111",
}

VARIABLE_LABELS = {
    "tp": "Precipitation",
    "z500": "500 hPa Geopotential Height",
    "t2m": "2 m Temperature",
}

REGION_LABELS = {
    "All India": "All India",
    "central_india": "Central India",
    "east_northeast_india": "East and Northeast India",
    "northwest_india": "Northwest India",
    "south_peninsula": "South Peninsula",
}

METRIC_LABELS = {
    "acc": "ACC",
    "rmse": "RMSE",
    "mae": "MAE",
    "bias": "Bias",
    "mse_skill_clim": "MSE skill vs climatology",
    "crps": "CRPS",
    "crpss_clim": "CRPSS vs climatology",
    "spread": "Ensemble spread",
    "rmse_ensmean": "Ensemble-mean RMSE",
    "spread_skill_ratio": "Spread-skill ratio",
    "brier": "Brier score",
    "brier_skill_clim": "Brier skill vs climatology",
    "base_rate": "Observed event rate",
}

REFERENCE_LINES = {
    "acc": 0.0,
    "bias": 0.0,
    "mse_skill_clim": 0.0,
    "crpss_clim": 0.0,
    "spread_skill_ratio": 1.0,
    "brier_skill_clim": 0.0,
}

LOWER_BOUNDED_ZERO = {
    "rmse",
    "mae",
    "crps",
    "spread",
    "rmse_ensmean",
    "brier",
    "base_rate",
}


def safe_name(text: object) -> str:
    out = str(text).strip().lower()
    for old, new in (
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        ("(", ""),
        (")", ""),
        (",", ""),
        ("=", "_"),
        (">", "gt"),
        ("<", "lt"),
    ):
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def model_sort_key(model: object) -> tuple[int, str]:
    name = str(model).lower()
    try:
        return (MODEL_ORDER.index(name), name)
    except ValueError:
        return (len(MODEL_ORDER), name)


def label_model(model: object) -> str:
    name = str(model).lower()
    return MODEL_LABELS.get(name, str(model))


def label_region(region: object) -> str:
    return REGION_LABELS.get(str(region), str(region).replace("_", " ").title())


def label_variable(variable: object) -> str:
    return VARIABLE_LABELS.get(str(variable).lower(), str(variable).upper())


def metric_axis_label(metric: str, unit: str | None = None) -> str:
    label = METRIC_LABELS.get(metric, metric)
    if unit and metric in {"rmse", "mae", "bias", "crps", "spread", "rmse_ensmean"}:
        return f"{label} ({unit})"
    return label


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.0,
        }
    )


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


def add_run_columns(df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "publication_case", config.publication_case)
    out.insert(1, "publication_title", config.publication_title)
    out.insert(2, "publication_note", config.note)
    return out


def load_combined_tables(output_root: Path) -> dict[str, pd.DataFrame]:
    combined: dict[str, pd.DataFrame] = {}
    for table_name in TABLE_NAMES:
        frames: list[pd.DataFrame] = []
        for config in PUBLICATION_RUNS:
            path = output_root / config.season / "03_metrics" / config.run_label / f"{table_name}.csv"
            table = read_table(path)
            if table.empty:
                continue
            frames.append(add_run_columns(table, config))
        combined[table_name] = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined


def existing_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> list[str]:
    return [col for col in columns if col in df.columns]


def summarize_long(df: pd.DataFrame, metrics: tuple[str, ...], extra_dims: tuple[str, ...] = ()) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    value_vars = existing_columns(df, metrics)
    if not value_vars:
        return pd.DataFrame()
    dims = existing_columns(
        df,
        (
            "publication_case",
            "publication_title",
            "season",
            "run_label",
            "set_name",
            "variable",
            "model",
            "region",
            "forecast_type",
            "week",
            "week_name",
            "lead_start_day",
            "lead_end_day",
            "truth_request",
            "tp_truth_source",
            "grid_dgrid",
            "unit",
            "truth_source",
            "climatology_source",
            *extra_dims,
        ),
    )
    data = df.replace([np.inf, -np.inf], np.nan)
    long = data.melt(id_vars=dims, value_vars=value_vars, var_name="metric", value_name="value")
    long = long.dropna(subset=["value"])
    if long.empty:
        return pd.DataFrame()
    grouped = (
        long.groupby(dims + ["metric"], dropna=False)["value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    grouped["lead_mid_day"] = (grouped["lead_start_day"].astype(float) + grouped["lead_end_day"].astype(float)) / 2.0
    return grouped.sort_values(dims + ["metric"]).reset_index(drop=True)


def summarize_wide(df: pd.DataFrame, metrics: tuple[str, ...], extra_dims: tuple[str, ...] = ()) -> pd.DataFrame:
    long = summarize_long(df, metrics, extra_dims)
    if long.empty:
        return pd.DataFrame()
    dims = [col for col in long.columns if col not in {"metric", "value", "count", "mean", "median", "std", "min", "max", "sem"}]
    wide = long.pivot_table(
        index=dims,
        columns="metric",
        values=["count", "mean", "median", "std", "min", "max", "sem"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    return wide.reset_index()


def write_data_products(combined: dict[str, pd.DataFrame], data_dir: Path) -> dict[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    combined_dir = data_dir / "combined_weekly"
    summary_dir = data_dir / "lead_model_summary"
    combined_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    for name, table in combined.items():
        if table.empty:
            continue
        path = combined_dir / f"{name}_all_publication_runs.csv"
        table.to_csv(path, index=False)
        outputs[f"combined_{name}"] = str(path)

    det = combined.get("deterministic_weekly", pd.DataFrame())
    prob = combined.get("probabilistic_weekly", pd.DataFrame())
    brier = combined.get("brier_weekly", pd.DataFrame())

    products = {
        "deterministic_lead_model_summary_long": summarize_long(det, DETERMINISTIC_METRICS),
        "deterministic_lead_model_summary_wide": summarize_wide(det, DETERMINISTIC_METRICS),
        "probabilistic_lead_model_summary_long": summarize_long(prob, PROBABILISTIC_METRICS),
        "probabilistic_lead_model_summary_wide": summarize_wide(prob, PROBABILISTIC_METRICS),
        "brier_lead_model_summary_long": summarize_long(brier, BRIER_METRICS, ("event", "threshold")),
        "brier_lead_model_summary_wide": summarize_wide(brier, BRIER_METRICS, ("event", "threshold")),
    }
    all_long = []
    for name, table in products.items():
        if table.empty:
            continue
        path = summary_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = str(path)
        if name.endswith("_long"):
            metric_family = name.replace("_lead_model_summary_long", "")
            tmp = table.copy()
            tmp.insert(0, "metric_family", metric_family)
            all_long.append(tmp)
    if all_long:
        path = summary_dir / "all_metrics_lead_model_summary_long.csv"
        pd.concat(all_long, ignore_index=True, sort=False).to_csv(path, index=False)
        outputs["all_metrics_lead_model_summary_long"] = str(path)

    inventory = []
    for config in PUBLICATION_RUNS:
        inventory.append(
            {
                "publication_case": config.publication_case,
                "publication_title": config.publication_title,
                "season": config.season,
                "run_label": config.run_label,
                "note": config.note,
            }
        )
    inventory_path = data_dir / "run_inventory.csv"
    pd.DataFrame(inventory).to_csv(inventory_path, index=False)
    outputs["run_inventory"] = str(inventory_path)
    return outputs


def write_grid_scatter_sample(output_root: Path, data_dir: Path, max_points_per_model: int) -> str | None:
    """Write a compact India-grid sample for publication scatter reproduction."""

    usecols = [
        "season",
        "run_label",
        "init_date",
        "valid_start",
        "valid_end",
        "week",
        "week_name",
        "lead_start_day",
        "lead_end_day",
        "variable",
        "model",
        "region",
        "lat",
        "lon",
        "unit",
        "forecast_value",
        "truth_value",
        "forecast_anomaly",
        "truth_anomaly",
        "error",
    ]
    samples: list[pd.DataFrame] = []
    for config in PUBLICATION_RUNS:
        path = output_root / config.season / "03_metrics" / config.run_label / "scatter_grid_weekly.csv"
        if not path.exists() or path.stat().st_size <= 1:
            continue
        table = pd.read_csv(path, usecols=lambda c: c in usecols)
        if table.empty:
            continue
        table = add_run_columns(table, config)
        for (_, _, variable, model), group in table.groupby(
            ["publication_case", "publication_title", "variable", "model"], sort=True
        ):
            if len(group) > max_points_per_model:
                group = group.sample(max_points_per_model, random_state=42)
            samples.append(group)
    if not samples:
        return None
    sample_dir = data_dir / "scatter_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / "scatter_grid_sample_all_publication_runs.csv"
    pd.concat(samples, ignore_index=True, sort=False).to_csv(out_path, index=False)
    return str(out_path)


def save_figure(fig: plt.Figure, base_path: Path, pdf: bool = True) -> list[str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    png_path = base_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    paths.append(str(png_path))
    if pdf:
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        paths.append(str(pdf_path))
    plt.close(fig)
    return paths


def lead_metric_plot(
    df: pd.DataFrame,
    metrics: tuple[str, ...],
    title: str,
    out_base: Path,
    unit: str | None = None,
    pdf: bool = True,
) -> list[str]:
    metrics = tuple(metric for metric in metrics if metric in df.columns)
    if df.empty or not metrics:
        return []
    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["week"])
    if data.empty:
        return []
    grouped = data.groupby(["model", "week"], dropna=False)[list(metrics)].mean(numeric_only=True).reset_index()
    if grouped.empty:
        return []

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 3.05 * nrows), squeeze=False)
    handles = []
    labels = []
    models = sorted(grouped["model"].astype(str).unique(), key=model_sort_key)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, metric in zip(axes.ravel(), metrics):
        ax.set_visible(True)
        for model in models:
            part = grouped[grouped["model"].astype(str).eq(model)].sort_values("week")
            if part[metric].dropna().empty:
                continue
            line = ax.plot(
                part["week"],
                part[metric],
                marker="o",
                color=MODEL_COLORS.get(model.lower(), "0.35"),
                label=label_model(model),
            )[0]
            if label_model(model) not in labels:
                handles.append(line)
                labels.append(label_model(model))
        if metric in REFERENCE_LINES:
            ax.axhline(REFERENCE_LINES[metric], color="0.25", linestyle="--", linewidth=1.0)
        if metric in LOWER_BOUNDED_ZERO:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=min(0.0, ymin), top=ymax)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Lead week")
        ax.set_ylabel(metric_axis_label(metric, unit))
        ax.set_xticks([1, 2, 3, 4, 5, 6])
    fig.suptitle(title, y=1.02, fontsize=13.0, fontweight="bold")
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(7, len(labels)), bbox_to_anchor=(0.5, -0.025))
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return save_figure(fig, out_base, pdf=pdf)


def plot_deterministic_figures(det: pd.DataFrame, fig_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if det.empty:
        return records
    for (publication_case, publication_title, variable, region), group in det.groupby(
        ["publication_case", "publication_title", "variable", "region"], sort=True
    ):
        unit = str(group["unit"].dropna().iloc[0]) if "unit" in group and group["unit"].notna().any() else ""
        title = f"{publication_title}: {label_variable(variable)} | {label_region(region)}"
        out_base = fig_dir / "lead_skill" / "deterministic" / (
            f"{safe_name(publication_case)}_{safe_name(region)}_{safe_name(variable)}_deterministic_lead_metrics"
        )
        paths = lead_metric_plot(group, DETERMINISTIC_METRICS, title, out_base, unit=unit, pdf=True)
        for path in paths:
            records.append(
                {
                    "figure_type": "deterministic_lead_skill",
                    "publication_case": str(publication_case),
                    "variable": str(variable),
                    "region": str(region),
                    "path": path,
                }
            )
    return records


def plot_probabilistic_figures(prob: pd.DataFrame, fig_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if prob.empty:
        return records
    for (publication_case, publication_title, variable, region), group in prob.groupby(
        ["publication_case", "publication_title", "variable", "region"], sort=True
    ):
        unit = str(group["unit"].dropna().iloc[0]) if "unit" in group and group["unit"].notna().any() else ""
        title = f"{publication_title}: {label_variable(variable)} | {label_region(region)}"
        out_base = fig_dir / "lead_skill" / "probabilistic" / (
            f"{safe_name(publication_case)}_{safe_name(region)}_{safe_name(variable)}_probabilistic_lead_metrics"
        )
        paths = lead_metric_plot(group, PROBABILISTIC_METRICS, title, out_base, unit=unit, pdf=True)
        for path in paths:
            records.append(
                {
                    "figure_type": "probabilistic_lead_skill",
                    "publication_case": str(publication_case),
                    "variable": str(variable),
                    "region": str(region),
                    "path": path,
                }
            )
    return records


def plot_brier_figures(brier: pd.DataFrame, fig_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if brier.empty:
        return records
    group_cols = ["publication_case", "publication_title", "event", "threshold", "region"]
    for keys, group in brier.groupby(group_cols, sort=True):
        publication_case, publication_title, event, threshold, region = keys
        title = f"{publication_title}: {event} | {label_region(region)}"
        out_base = fig_dir / "lead_skill" / "brier" / (
            f"{safe_name(publication_case)}_{safe_name(region)}_{safe_name(event)}_brier_lead_metrics"
        )
        paths = lead_metric_plot(group, BRIER_METRICS, title, out_base, unit=None, pdf=True)
        for path in paths:
            records.append(
                {
                    "figure_type": "brier_lead_skill",
                    "publication_case": str(publication_case),
                    "variable": "tp",
                    "region": str(region),
                    "event": str(event),
                    "threshold": str(threshold),
                    "path": path,
                }
            )
    return records


def scatter_limits(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    values = pd.concat([x, y], ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if values.empty:
        return 0.0, 1.0
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(lo, hi):
        pad = 1.0 if np.isclose(lo, 0.0) else abs(lo) * 0.1
    else:
        pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def scatter_stats(part: pd.DataFrame) -> str:
    x = part["truth_value"].astype(float)
    y = part["forecast_value"].astype(float)
    err = y - x
    if len(part) > 1 and x.std() > 0 and y.std() > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = np.nan
    rmse = float(np.sqrt(np.nanmean(err**2)))
    bias = float(np.nanmean(err))
    return f"r={corr:.2f}\nRMSE={rmse:.2g}\nBias={bias:.2g}\nn={len(part)}"


def plot_scatter_facets(
    table: pd.DataFrame,
    title: str,
    out_base: Path,
    max_points_per_model: int,
    pdf: bool = False,
) -> list[str]:
    required = {"model", "week", "forecast_value", "truth_value"}
    if table.empty or not required.issubset(table.columns):
        return []
    data = table.replace([np.inf, -np.inf], np.nan).dropna(subset=["forecast_value", "truth_value", "model", "week"])
    if data.empty:
        return []
    models = sorted(data["model"].astype(str).unique(), key=model_sort_key)
    ncols = min(3, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.85 * ncols, 4.45 * nrows), squeeze=False)
    lo, hi = scatter_limits(data["truth_value"], data["forecast_value"])
    scatter = None
    for ax in axes.ravel():
        ax.set_visible(False)
    for idx, (ax, model) in enumerate(zip(axes.ravel(), models)):
        row = idx // ncols
        col = idx % ncols
        part = data[data["model"].astype(str).eq(model)]
        if len(part) > max_points_per_model:
            part = part.sample(max_points_per_model, random_state=42)
        ax.set_visible(True)
        scatter = ax.scatter(
            part["truth_value"],
            part["forecast_value"],
            c=part["week"].astype(float),
            cmap="viridis",
            vmin=1,
            vmax=6,
            s=20 if max_points_per_model < 1200 else 8,
            alpha=0.72 if max_points_per_model < 1200 else 0.38,
            linewidths=0,
            rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], color="0.2", linestyle="--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label_model(model), pad=8)
        ax.text(
            0.04,
            0.96,
            scatter_stats(part),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.78, "pad": 2.5},
        )
        ax.set_xlabel("Truth" if row == nrows - 1 else "")
        ax.set_ylabel("Forecast" if col == 0 else "")
    if scatter is not None:
        fig.subplots_adjust(left=0.06, right=0.89, bottom=0.08, top=0.90, wspace=0.18, hspace=0.30)
        cax = fig.add_axes([0.915, 0.18, 0.018, 0.64])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Lead week")
        cbar.set_ticks([1, 2, 3, 4, 5, 6])
    else:
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.90, wspace=0.18, hspace=0.30)
    fig.suptitle(title, y=0.975, fontsize=13.0, fontweight="bold")
    return save_figure(fig, out_base, pdf=pdf)


def plot_area_scatter_figures(scatter_area: pd.DataFrame, fig_dir: Path, max_points_per_model: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if scatter_area.empty:
        return records
    for (publication_case, publication_title, variable, region), group in scatter_area.groupby(
        ["publication_case", "publication_title", "variable", "region"], sort=True
    ):
        unit = str(group["unit"].dropna().iloc[0]) if "unit" in group and group["unit"].notna().any() else ""
        title = f"{publication_title}: {label_variable(variable)} area-mean scatter | {label_region(region)}"
        if unit:
            title += f" ({unit})"
        out_base = fig_dir / "scatter" / "area_mean" / (
            f"{safe_name(publication_case)}_{safe_name(region)}_{safe_name(variable)}_area_scatter_by_model"
        )
        paths = plot_scatter_facets(group, title, out_base, max_points_per_model=max_points_per_model, pdf=False)
        for path in paths:
            records.append(
                {
                    "figure_type": "area_mean_scatter",
                    "publication_case": str(publication_case),
                    "variable": str(variable),
                    "region": str(region),
                    "path": path,
                }
            )
    return records


def plot_grid_scatter_figures(output_root: Path, fig_dir: Path, max_points_per_model: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    usecols = [
        "season",
        "run_label",
        "week",
        "variable",
        "model",
        "region",
        "unit",
        "forecast_value",
        "truth_value",
    ]
    for config in PUBLICATION_RUNS:
        path = output_root / config.season / "03_metrics" / config.run_label / "scatter_grid_weekly.csv"
        if not path.exists() or path.stat().st_size <= 1:
            continue
        table = pd.read_csv(path, usecols=lambda c: c in usecols)
        if table.empty:
            continue
        table = add_run_columns(table, config)
        for variable, group in table.groupby("variable", sort=True):
            unit = str(group["unit"].dropna().iloc[0]) if "unit" in group and group["unit"].notna().any() else ""
            title = f"{config.publication_title}: {label_variable(variable)} grid-point scatter | India grid"
            if unit:
                title += f" ({unit})"
            out_base = fig_dir / "scatter" / "grid_sample" / (
                f"{safe_name(config.publication_case)}_{safe_name(variable)}_grid_scatter_by_model"
            )
            paths = plot_scatter_facets(group, title, out_base, max_points_per_model=max_points_per_model, pdf=False)
            for fig_path in paths:
                records.append(
                    {
                        "figure_type": "grid_point_scatter_sample",
                        "publication_case": config.publication_case,
                        "variable": str(variable),
                        "region": "India grid",
                        "path": fig_path,
                    }
                )
    return records


def make_figures(combined: dict[str, pd.DataFrame], output_root: Path, fig_dir: Path, max_scatter_points: int) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    records.extend(plot_deterministic_figures(combined.get("deterministic_weekly", pd.DataFrame()), fig_dir))
    records.extend(plot_probabilistic_figures(combined.get("probabilistic_weekly", pd.DataFrame()), fig_dir))
    records.extend(plot_brier_figures(combined.get("brier_weekly", pd.DataFrame()), fig_dir))
    records.extend(plot_area_scatter_figures(combined.get("scatter_area_weekly", pd.DataFrame()), fig_dir, max_scatter_points))
    records.extend(plot_grid_scatter_figures(output_root, fig_dir, max_scatter_points))
    return pd.DataFrame(records)


def write_readme(publication_dir: Path, data_outputs: dict[str, str], figure_count: int) -> None:
    lines = [
        "# S2S Publication Results",
        "",
        "This directory contains publication-facing products generated from the validated verification metric runs.",
        "",
        "Lead time convention: the current verification products are weekly S2S lead windows, Week 1 through Week 6, covering lead days 1-42.",
        "",
        "Included metric families:",
        "",
        "- Deterministic: ACC, RMSE, MAE, Bias, MSE skill versus climatology.",
        "- Probabilistic: CRPS, CRPSS versus climatology, ensemble spread, ensemble-mean RMSE, spread-skill ratio.",
        "- Event/probability: Brier score, Brier skill versus climatology, observed base rate.",
        "- Scatter data/figures: forecast versus truth area-mean scatter for every region and sampled grid-point scatter across the India verification grid.",
        "",
        "Important notes:",
        "",
        "- JJAS2019 common-17 uses all usable models on the common 17 init dates.",
        "- JJAS2019 operational-35 TP has separate ERA5-truth and IMD-truth products.",
        "- JFM2026 daily includes SPIRE. TP and T2M cover 90 init dates. Z500 is limited for late March inits where April 2026 truth is unavailable locally.",
        "",
        f"Generated figure files: {figure_count}",
        "",
        "Key CSV products:",
        "",
    ]
    for name, path in sorted(data_outputs.items()):
        rel = Path(path).relative_to(publication_dir)
        lines.append(f"- `{name}`: `{rel}`")
    lines.append("")
    (publication_dir / "README.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--publication-dir-name", default="publication_results")
    parser.add_argument("--max-scatter-points-per-model", type=int, default=2500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    publication_dir = args.output_root / args.publication_dir_name
    data_dir = publication_dir / "data"
    fig_dir = publication_dir / "figures"
    publication_dir.mkdir(parents=True, exist_ok=True)

    combined = load_combined_tables(args.output_root)
    data_outputs = write_data_products(combined, data_dir)
    grid_sample = write_grid_scatter_sample(args.output_root, data_dir, args.max_scatter_points_per_model)
    if grid_sample:
        data_outputs["scatter_grid_sample_all_publication_runs"] = grid_sample
    figure_manifest = make_figures(combined, args.output_root, fig_dir, args.max_scatter_points_per_model)

    manifest_path = publication_dir / "figure_manifest.csv"
    figure_manifest.to_csv(manifest_path, index=False)
    data_outputs["figure_manifest"] = str(manifest_path)

    json_manifest = {
        "publication_dir": str(publication_dir),
        "data_outputs": data_outputs,
        "figure_count": int(len(figure_manifest)),
        "runs": [config.__dict__ for config in PUBLICATION_RUNS],
    }
    (publication_dir / "manifest.json").write_text(json.dumps(json_manifest, indent=2))
    write_readme(publication_dir, data_outputs, len(figure_manifest))

    print(f"Publication results: {publication_dir}")
    print(f"Data products: {len(data_outputs)}")
    print(f"Figure files: {len(figure_manifest)}")
    print(f"Figure manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
