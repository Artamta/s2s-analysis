#!/usr/bin/env python3
"""Make quick-look plots from a weekly metrics run.

The plots are intentionally simple: they are smoke-test diagnostics and plotting
fixtures, not final paper figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "s2s_paper_outputs"


def read_table(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / f"{name}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_name(text: object) -> str:
    return str(text).replace(" ", "_").replace("/", "_").replace(":", "_").lower()


def line_limits(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    values = pd.concat([x, y]).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0, 1.0
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(lo, hi):
        pad = 1.0 if np.isclose(lo, 0.0) else abs(lo) * 0.1
    else:
        pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def init_mean_for_bars(data: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    """Collapse repeated init/region rows before compact smoke/full-run bar plots."""

    cols = [col for col in metrics if col in data.columns]
    if not cols:
        return pd.DataFrame()
    out = data.groupby(group_cols, as_index=False)[cols].mean(numeric_only=True)
    return out.sort_values(group_cols).reset_index(drop=True)


def plot_deterministic(det: pd.DataFrame, out_dir: Path) -> list[str]:
    figures: list[str] = []
    if det.empty:
        return figures
    data = det[det["region"].eq("All India")].copy()
    if data.empty:
        return figures
    metrics = [m for m in ("acc", "rmse", "bias", "mse_skill_clim") if m in data]
    for variable, group in data.groupby("variable", sort=True):
        group = init_mean_for_bars(group, ["week", "model"], metrics)
        if group.empty:
            continue
        group = group.sort_values(["week", "model"])
        labels = [f"W{int(w)}\n{m}" for w, m in zip(group["week"], group["model"])]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(max(8, 0.45 * len(group)), 2.4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            ax.bar(np.arange(len(group)), group[metric].astype(float), color="#4C78A8")
            ax.axhline(0, color="0.25", linewidth=0.8)
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(np.arange(len(group)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.suptitle(f"All-India init-mean deterministic metrics: {variable.upper()}")
        path = out_dir / f"deterministic_all_india_{safe_name(variable)}.png"
        savefig(path)
        figures.append(str(path))
    return figures


def plot_probabilistic(prob: pd.DataFrame, out_dir: Path) -> list[str]:
    figures: list[str] = []
    if prob.empty:
        return figures
    data = prob[prob["region"].eq("All India")].copy()
    if data.empty:
        return figures
    for variable, group in data.groupby("variable", sort=True):
        metrics = [m for m in ("crps", "spread_skill_ratio") if m in group]
        group = init_mean_for_bars(group, ["week", "model"], metrics)
        if group.empty:
            continue
        group = group.sort_values(["week", "model"])
        labels = [f"W{int(w)}\n{m}" for w, m in zip(group["week"], group["model"])]
        fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.45 * len(group)), 5.6))
        for ax, metric, color in zip(axes, ("crps", "spread_skill_ratio"), ("#F58518", "#54A24B")):
            if metric not in group:
                continue
            ax.bar(np.arange(len(group)), group[metric].astype(float), color=color)
            ax.axhline(0 if metric == "crps" else 1, color="0.25", linewidth=0.8, linestyle="--")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(np.arange(len(group)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.suptitle(f"All-India init-mean probabilistic metrics: {variable.upper()}")
        path = out_dir / f"probabilistic_all_india_{safe_name(variable)}.png"
        savefig(path)
        figures.append(str(path))
    return figures


def plot_brier(brier: pd.DataFrame, out_dir: Path) -> list[str]:
    figures: list[str] = []
    if brier.empty:
        return figures
    data = brier[brier["region"].eq("All India")].copy()
    if data.empty:
        return figures
    for event, group in data.groupby("event", sort=True):
        group = init_mean_for_bars(group, ["week", "model"], ["brier_skill_clim"])
        if group.empty:
            continue
        group = group.sort_values(["week", "model"])
        labels = [f"W{int(w)}\n{m}" for w, m in zip(group["week"], group["model"])]
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(group)), 3.6))
        ax.bar(np.arange(len(group)), group["brier_skill_clim"].astype(float), color="#E45756")
        ax.axhline(0, color="0.25", linewidth=0.8)
        ax.set_ylabel("Brier skill vs climatology")
        ax.set_xticks(np.arange(len(group)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"All-India Brier skill: {event}")
        path = out_dir / f"brier_skill_all_india_{safe_name(event)}.png"
        savefig(path)
        figures.append(str(path))
    return figures


def plot_scatter(table: pd.DataFrame, out_dir: Path, prefix: str, max_points: int) -> list[str]:
    figures: list[str] = []
    if table.empty:
        return figures
    for variable, group in table.groupby("variable", sort=True):
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["forecast_value", "truth_value"])
        if group.empty:
            continue
        if len(group) > max_points:
            group = group.sample(max_points, random_state=42)
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        models = sorted(group["model"].astype(str).unique())
        cmap = plt.get_cmap("tab10")
        for i, model in enumerate(models):
            part = group[group["model"].astype(str).eq(model)]
            ax.scatter(
                part["truth_value"],
                part["forecast_value"],
                s=28 if prefix == "scatter_area" else 12,
                alpha=0.75 if prefix == "scatter_area" else 0.45,
                label=model,
                color=cmap(i % 10),
                edgecolors="none",
            )
        lo, hi = line_limits(group["truth_value"], group["forecast_value"])
        ax.plot([lo, hi], [lo, hi], color="0.2", linewidth=1.0, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        unit = group["unit"].dropna().iloc[0] if "unit" in group and group["unit"].notna().any() else ""
        ax.set_xlabel(f"Truth ({unit})")
        ax.set_ylabel(f"Forecast ({unit})")
        ax.set_title(f"{prefix.replace('_', ' ').title()}: {variable.upper()}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        path = out_dir / f"{prefix}_forecast_vs_truth_{safe_name(variable)}.png"
        savefig(path)
        figures.append(str(path))
    return figures


def model_sort_key(model: object) -> tuple[int, str]:
    name = str(model)
    return (1 if name == "mme" else 0, name)


def robust_limit(values: pd.Series, symmetric: bool) -> tuple[float, float]:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(vals), 98))
        if not np.isfinite(vmax) or np.isclose(vmax, 0.0):
            vmax = 1.0
        return -vmax, vmax
    lo = 0.0
    hi = float(np.nanpercentile(vals, 98))
    if not np.isfinite(hi) or np.isclose(hi, 0.0):
        hi = float(vals.max()) if not vals.empty else 1.0
    if not np.isfinite(hi) or np.isclose(hi, 0.0):
        hi = 1.0
    return lo, hi


def plot_spatial_maps(scatter_grid: pd.DataFrame, out_dir: Path) -> list[str]:
    """Plot India-grid mean-error and RMSE maps from scatter_grid_weekly.csv."""

    figures: list[str] = []
    required = {"lat", "lon", "model", "variable", "week", "error"}
    if scatter_grid.empty or not required.issubset(scatter_grid.columns):
        return figures
    data = scatter_grid.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    if data.empty:
        return figures
    data["error_sq"] = data["error"].astype(float) ** 2
    grouped = (
        data.groupby(["variable", "week", "model", "lat", "lon"], dropna=False)
        .agg(mean_error=("error", "mean"), rmse=("error_sq", lambda x: float(np.sqrt(np.nanmean(x)))))
        .reset_index()
    )
    if grouped.empty:
        return figures

    map_dir = out_dir / "spatial_maps"
    for (variable, week), group in grouped.groupby(["variable", "week"], sort=True):
        models = sorted(group["model"].astype(str).unique(), key=model_sort_key)
        ncols = min(3, len(models))
        nrows = int(np.ceil(len(models) / ncols))
        unit = ""
        source_units = data.loc[data["variable"].astype(str).eq(str(variable)), "unit"] if "unit" in data else pd.Series(dtype=object)
        if not source_units.empty and source_units.notna().any():
            unit = str(source_units.dropna().iloc[0])
        for metric, title, cmap, symmetric in (
            ("mean_error", "Mean Error", "RdBu_r", True),
            ("rmse", "RMSE", "viridis", False),
        ):
            vmin, vmax = robust_limit(group[metric], symmetric=symmetric)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(4.1 * ncols, 4.2 * nrows),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            scatter = None
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, model in zip(axes.ravel(), models):
                part = group[group["model"].astype(str).eq(model)]
                ax.set_visible(True)
                scatter = ax.scatter(
                    part["lon"],
                    part["lat"],
                    c=part[metric],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    marker="s",
                    s=54,
                    linewidths=0.0,
                )
                ax.set_title(model)
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")
                ax.set_xlim(64, 101)
                ax.set_ylim(4, 39)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=0.2, linewidth=0.5)
            if scatter is not None:
                label = f"{title} ({unit})" if unit else title
                fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.78, label=label)
            fig.suptitle(f"{title} map: {str(variable).upper()} Week {int(week)}")
            path = map_dir / f"spatial_{safe_name(metric)}_{safe_name(variable)}_week{int(week)}.png"
            savefig(path)
            figures.append(str(path))
    return figures


def write_summary_tables(run_dir: Path, table_dir: Path) -> list[str]:
    outputs: list[str] = []
    det = read_table(run_dir, "deterministic_weekly")
    prob = read_table(run_dir, "probabilistic_weekly")
    brier = read_table(run_dir, "brier_weekly")
    table_dir.mkdir(parents=True, exist_ok=True)
    if not det.empty:
        cols = [
            c
            for c in (
                "season",
                "run_label",
                "init_date",
                "week",
                "variable",
                "model",
                "region",
                "acc",
                "rmse",
                "bias",
                "mae",
                "mse_skill_clim",
            )
            if c in det.columns
        ]
        path = table_dir / "deterministic_summary.csv"
        det[cols].to_csv(path, index=False)
        outputs.append(str(path))
    if not prob.empty:
        cols = [
            c
            for c in (
                "season",
                "run_label",
                "init_date",
                "week",
                "variable",
                "model",
                "region",
                "crps",
                "crpss_clim",
                "spread_skill_ratio",
            )
            if c in prob.columns
        ]
        path = table_dir / "probabilistic_summary.csv"
        prob[cols].to_csv(path, index=False)
        outputs.append(str(path))
    if not brier.empty:
        cols = [
            c
            for c in (
                "season",
                "run_label",
                "init_date",
                "week",
                "model",
                "region",
                "event",
                "brier",
                "brier_skill_clim",
            )
            if c in brier.columns
        ]
        path = table_dir / "brier_summary.csv"
        brier[cols].to_csv(path, index=False)
        outputs.append(str(path))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", choices=["jjas2019", "jfm2026"], required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-grid-points", type=int, default=5000)
    parser.add_argument(
        "--no-spatial-maps",
        action="store_true",
        help="Skip map diagnostics from scatter_grid_weekly.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / args.season / "03_metrics" / args.run_label
    fig_dir = args.output_root / args.season / "04_figures" / args.run_label / "test_plots"
    table_dir = args.output_root / args.season / "05_tables" / args.run_label
    if not run_dir.exists():
        raise FileNotFoundError(f"metrics run directory does not exist: {run_dir}")

    det = read_table(run_dir, "deterministic_weekly")
    prob = read_table(run_dir, "probabilistic_weekly")
    brier = read_table(run_dir, "brier_weekly")
    scatter_area = read_table(run_dir, "scatter_area_weekly")
    scatter_grid = read_table(run_dir, "scatter_grid_weekly")

    figures: list[str] = []
    figures.extend(plot_deterministic(det, fig_dir))
    figures.extend(plot_probabilistic(prob, fig_dir))
    figures.extend(plot_brier(brier, fig_dir))
    figures.extend(plot_scatter(scatter_area, fig_dir, "scatter_area", args.max_grid_points))
    figures.extend(plot_scatter(scatter_grid, fig_dir, "scatter_grid", args.max_grid_points))
    if not args.no_spatial_maps:
        figures.extend(plot_spatial_maps(scatter_grid, fig_dir))
    tables = write_summary_tables(run_dir, table_dir)

    manifest = {
        "season": args.season,
        "run_label": args.run_label,
        "run_dir": str(run_dir),
        "figure_dir": str(fig_dir),
        "table_dir": str(table_dir),
        "figures": figures,
        "tables": tables,
    }
    manifest_path = fig_dir / "plot_manifest.json"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(figures)} figures")
    for path in figures:
        print(f"  {path}")
    print(f"Wrote {len(tables)} summary tables")
    for path in tables:
        print(f"  {path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
