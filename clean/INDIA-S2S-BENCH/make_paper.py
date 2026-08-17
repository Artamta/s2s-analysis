#!/usr/bin/env python3
"""Generate manuscript numbers, tables, and figures from audited artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts/confirmatory_2025"
PAPER = ROOT / "paper"

DISPLAY_NAMES = {
    "cma": "CMA",
    "dlesym_v0": "DLESyM v0",
    "ecmwf": "ECMWF",
    "equal_weight": "Equal-weight mean",
    "fuxi_s2s": "FuXi-S2S",
    "location_calendar_only": "Location/calendar only",
    "ncep": "NCEP",
    "neuralgcm": "NeuralGCM",
    "piggycast_forecast_only": "PiggyCast, forecast only",
    "piggycast_full": "PiggyCast, full",
    "ukmo": "UKMO",
    "validation_selected_model": "Per-lead 2024-RMSE-selected",
    "validation_weighted": "Validation-weighted mean",
}

BASE_MODELS = ["cma", "dlesym_v0", "ecmwf", "fuxi_s2s", "ncep", "neuralgcm", "ukmo"]
HEADLINE_METHODS = [
    "piggycast_full",
    "piggycast_forecast_only",
    "validation_weighted",
    "equal_weight",
    "validation_selected_model",
    "ecmwf",
]
ALL_METHODS = [
    "piggycast_full",
    "piggycast_forecast_only",
    "validation_weighted",
    "equal_weight",
    "ecmwf",
    "validation_selected_model",
    "fuxi_s2s",
    "neuralgcm",
    "ukmo",
    "cma",
    "ncep",
    "dlesym_v0",
    "location_calendar_only",
]


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _write_score_table(
    path: Path,
    overall: pd.DataFrame,
    methods: list[str],
    *,
    include_wet_error: bool = False,
) -> None:
    if include_wet_error:
        lines = [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Method & ACC $\uparrow$ & RMSE $\downarrow$ & MAE $\downarrow$ & Bias & Wet-area error & Negative support (\%) \\",
            r"\midrule",
        ]
    else:
        lines = [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Method & ACC $\uparrow$ & RMSE $\downarrow$ & MAE $\downarrow$ & Bias \\",
            r"\midrule",
        ]
    for method in methods:
        row = overall.loc[method]
        values = (
            f"{DISPLAY_NAMES[method]} & {row.acc:.3f} & {row.rmse:.3f} & "
            f"{row.mae:.3f} & {row.bias:.3f}"
        )
        if include_wet_error:
            values += (
                f" & {row.wet_fraction_error:.3f} & "
                f"{100.0 * row.negative_fraction:.2f}"
            )
        lines.append(values + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")


def _make_figures(
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    regional: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    figure_dir = PAPER / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Main protocol schematic. It makes only the auditable workflow claim:
    # no 2025 fitting or method selection within this released evaluation.
    fig, ax = plt.subplots(figsize=(9.2, 2.0), constrained_layout=True)
    stages = [
        (0.07, "7 systems\n42 daily leads"),
        (0.29, "Common dates/grid\none IMD normal"),
        (0.50, "2020--23\ninitial fit"),
        (0.73, "2024 select trees\nrefit through 2024"),
        (0.94, "2025\nevaluate"),
    ]
    for index, (x, label) in enumerate(stages):
        ax.text(
            x,
            0.58,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=.42",
                "facecolor": "#E8F2F8",
                "edgecolor": "#3B6175",
            },
        )
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(stages[index + 1][0] - 0.085, 0.58),
                xytext=(x + 0.085, 0.58),
                arrowprops={"arrowstyle": "->", "lw": 1.25, "color": "#3B6175"},
            )
    ax.text(
        0.5,
        0.12,
        "JJAS initializations; Weeks 1--6 use +1...+42 period endpoints; "
        "35 common 2025 cases; no 2025 fitting or selection",
        ha="center",
        fontsize=8.5,
    )
    ax.set_axis_off()
    _save_figure(fig, figure_dir / "headline_pipeline")
    plt.close(fig)

    # Only claim-relevant curves appear in the main lead figure. The complete
    # seven-model inventory is retained in the appendix heatmap.
    india = aggregate[
        (aggregate.aggregation == "by_lead") & (aggregate.region == "india")
    ]
    styles = {
        "piggycast_full": ("#0072B2", "-", 2.7, "o"),
        "piggycast_forecast_only": ("#56B4E9", "--", 1.9, "s"),
        "validation_weighted": ("#009E73", "-.", 1.8, "v"),
        "equal_weight": ("#D55E00", "-", 2.2, "^"),
        "validation_selected_model": ("#333333", ":", 2.0, "D"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.25), constrained_layout=True)
    for method, (color, linestyle, linewidth, marker) in styles.items():
        subset = india[india.method == method].sort_values("lead_week")
        label = DISPLAY_NAMES[method]
        axes[0].plot(
            subset.lead_week,
            subset.acc,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=4.2,
            label=label,
        )
        axes[1].plot(
            subset.lead_week,
            subset.rmse,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=4.2,
            label=label,
        )
    axes[0].axhline(0, color="0.4", linewidth=0.7)
    axes[0].set(
        xlabel="Lead week",
        ylabel="Spatial ACC",
        title="(a) Anomaly-pattern skill",
    )
    axes[1].set(
        xlabel="Lead week",
        ylabel=r"RMSE (mm day$^{-1}$)",
        title="(b) Rainfall amount error",
    )
    for axis in axes:
        axis.set_xticks(range(1, 7))
        axis.grid(axis="y", color="0.9", linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        fontsize=7.5,
        handlelength=2.4,
        columnspacing=1.0,
        frameon=False,
    )
    _save_figure(fig, figure_dir / "headline_skill_by_lead")
    plt.close(fig)

    # One forest plot keeps the failed selected-model gate and the
    # east/northeast loss visible beside the positive national average.
    pair = paired.set_index(["method_a", "method_b", "metric"])
    rows = [
        (
            "India: full $-$ equal",
            pair.loc[("piggycast_full", "equal_weight", "acc")],
        ),
        (
            "India: full $-$ selected",
            pair.loc[("piggycast_full", "validation_selected_model", "acc")],
        ),
        (
            "India: forecast-only $-$ location/calendar",
            pair.loc[
                (
                    "piggycast_forecast_only",
                    "location_calendar_only",
                    "acc",
                )
            ],
        ),
    ]
    for region_name, label in [
        ("northwest_india", "Northwest: full $-$ equal"),
        ("central_india", "Central: full $-$ equal"),
        ("south_peninsula", "South peninsula: full $-$ equal"),
        ("east_northeast_india", "East/northeast: full $-$ equal"),
    ]:
        rows.append((label, regional[regional.region == region_name].iloc[0]))
    labels = [label for label, _ in rows]
    effects = np.asarray([row.effect for _, row in rows])
    lows = np.asarray([row.ci_low for _, row in rows])
    highs = np.asarray([row.ci_high for _, row in rows])
    colors = [
        "#0072B2" if low > 0 else "#D55E00" if high < 0 else "#777777"
        for low, high in zip(lows, highs)
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.75), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for index in range(len(rows)):
        ax.errorbar(
            effects[index],
            y[index],
            xerr=[
                [effects[index] - lows[index]],
                [highs[index] - effects[index]],
            ],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            markersize=5,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(y[2] - 0.5, color="0.82", linewidth=0.8)
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Paired mean spatial-ACC difference",
    )
    ax.grid(axis="x", color="0.92", linewidth=0.7)
    _save_figure(fig, figure_dir / "headline_effects")
    plt.close(fig)

    # Complete IMD-referenced all-model/week comparison for the appendix.
    # IMERG requires a separate, explicitly retrospective sensitivity run.
    heatmap_methods = [*BASE_MODELS, "equal_weight", "piggycast_full"]
    heatmap_labels = [DISPLAY_NAMES[item] for item in heatmap_methods]
    acc = (
        india[india.method.isin(heatmap_methods)]
        .pivot(index="method", columns="lead_week", values="acc")
        .loc[heatmap_methods]
    )
    rmse = (
        india[india.method.isin(heatmap_methods)]
        .pivot(index="method", columns="lead_week", values="rmse")
        .loc[heatmap_methods]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
    panels = [
        (
            axes[0],
            acc.to_numpy(),
            "Spatial ACC (higher is better)",
            "YlGnBu",
            -0.1,
            0.75,
            ".2f",
        ),
        (
            axes[1],
            rmse.to_numpy(),
            r"RMSE, mm day$^{-1}$ (lower is better)",
            "YlOrRd",
            3.8,
            7.4,
            ".1f",
        ),
    ]
    for axis, values, title, cmap, vmin, vmax, number_format in panels:
        image = axis.imshow(
            values,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set(
            xticks=np.arange(6),
            xticklabels=[f"W{week}" for week in range(1, 7)],
            yticks=np.arange(len(heatmap_labels)),
            yticklabels=heatmap_labels,
            title=title,
        )
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    format(values[row, column], number_format),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        fig.colorbar(image, ax=axis, shrink=0.78, pad=0.02)
    _save_figure(fig, figure_dir / "appendix_all_models_imd")
    plt.close(fig)

    # Region-by-lead diagnostic: the national mean does not hide local losses.
    regions = [
        "northwest_india",
        "central_india",
        "south_peninsula",
        "east_northeast_india",
    ]
    region_labels = [
        "Northwest",
        "Central",
        "South peninsula",
        "East/northeast",
    ]
    by_lead = aggregate[
        (aggregate.aggregation == "by_lead")
        & aggregate.region.isin(regions)
    ]
    full = (
        by_lead[by_lead.method == "piggycast_full"]
        .pivot(index="region", columns="lead_week", values="acc")
        .loc[regions]
    )
    equal = (
        by_lead[by_lead.method == "equal_weight"]
        .pivot(index="region", columns="lead_week", values="acc")
        .loc[regions]
    )
    delta = (full - equal).to_numpy()
    bound = max(0.10, float(np.nanmax(np.abs(delta))))
    fig, ax = plt.subplots(figsize=(7.1, 2.7), constrained_layout=True)
    image = ax.imshow(
        delta,
        aspect="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    )
    ax.set(
        xticks=np.arange(6),
        xticklabels=[f"W{week}" for week in range(1, 7)],
        yticks=np.arange(4),
        yticklabels=region_labels,
        title="Full PiggyCast minus equal-weight spatial ACC",
    )
    for row in range(delta.shape[0]):
        for column in range(delta.shape[1]):
            ax.text(
                column,
                row,
                f"{delta[row, column]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02, label=r"$\Delta$ACC")
    _save_figure(fig, figure_dir / "appendix_region_lead_delta")
    plt.close(fig)


def main() -> None:
    audit = json.loads((ARTIFACT / "audit_report.json").read_text())
    if audit["status"] != "passed":
        raise RuntimeError("paper generation requires a passing artifact audit")
    aggregate = pd.read_csv(ARTIFACT / "aggregate_metrics.csv")
    paired = pd.read_csv(ARTIFACT / "paired_intervals.csv")
    regional = pd.read_csv(ARTIFACT / "regional_acc_intervals.csv")
    sensitivity = pd.read_csv(ARTIFACT / "bootstrap_sensitivity.csv")
    overall = aggregate[
        (aggregate.aggregation == "all") & (aggregate.region == "india")
    ].set_index("method")
    pair = paired.set_index(["method_a", "method_b", "metric"])
    full_equal = pair.loc[("piggycast_full", "equal_weight", "acc")]
    full_selected = pair.loc[
        ("piggycast_full", "validation_selected_model", "acc")
    ]
    forecast_location = pair.loc[
        ("piggycast_forecast_only", "location_calendar_only", "acc")
    ]
    full_equal_rmse = pair.loc[("piggycast_full", "equal_weight", "rmse")]
    macros = {
        "TestCases": "35",
        "FullACC": fmt(overall.loc["piggycast_full", "acc"]),
        "EqualACC": fmt(overall.loc["equal_weight", "acc"]),
        "SelectedACC": fmt(
            overall.loc["validation_selected_model", "acc"]
        ),
        "ForecastOnlyACC": fmt(
            overall.loc["piggycast_forecast_only", "acc"]
        ),
        "WeightedACC": fmt(overall.loc["validation_weighted", "acc"]),
        "FullRMSE": fmt(overall.loc["piggycast_full", "rmse"]),
        "EqualRMSE": fmt(overall.loc["equal_weight", "rmse"]),
        "ForecastOnlyRMSE": fmt(
            overall.loc["piggycast_forecast_only", "rmse"]
        ),
        "FullMAE": fmt(overall.loc["piggycast_full", "mae"]),
        "EqualMAE": fmt(overall.loc["equal_weight", "mae"]),
        "FullBias": fmt(overall.loc["piggycast_full", "bias"]),
        "EqualBias": fmt(overall.loc["equal_weight", "bias"]),
        "FullNegativePercent": fmt(
            100.0 * overall.loc["piggycast_full", "negative_fraction"],
            1,
        ),
        "FullEqualEffect": fmt(full_equal.effect),
        "FullEqualLow": fmt(full_equal.ci_low),
        "FullEqualHigh": fmt(full_equal.ci_high),
        "FullSelectedEffect": fmt(full_selected.effect),
        "FullSelectedLow": fmt(full_selected.ci_low),
        "FullSelectedHigh": fmt(full_selected.ci_high),
        "ForecastLocationEffect": fmt(forecast_location.effect),
        "ForecastLocationLow": fmt(forecast_location.ci_low),
        "ForecastLocationHigh": fmt(forecast_location.ci_high),
        "FullEqualRMSEEffect": fmt(full_equal_rmse.effect),
        "FullEqualRMSELow": fmt(full_equal_rmse.ci_low),
        "FullEqualRMSEHigh": fmt(full_equal_rmse.ci_high),
    }

    lead_names = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
    }
    lead_rows = aggregate[
        (aggregate.aggregation == "by_lead")
        & (aggregate.region == "india")
    ]
    for lead, word in lead_names.items():
        full_lead = lead_rows[
            (lead_rows.method == "piggycast_full")
            & (lead_rows.lead_week == lead)
        ].iloc[0]
        equal_lead = lead_rows[
            (lead_rows.method == "equal_weight")
            & (lead_rows.lead_week == lead)
        ].iloc[0]
        macros[f"Lead{word}AccDelta"] = fmt(
            full_lead.acc - equal_lead.acc
        )
        macros[f"Lead{word}RmseDelta"] = fmt(
            full_lead.rmse - equal_lead.rmse
        )

    region_macro_names = {
        "northwest_india": "Northwest",
        "central_india": "Central",
        "south_peninsula": "South",
        "east_northeast_india": "EastNortheast",
    }
    for region_name, macro_name in region_macro_names.items():
        row = regional[regional.region == region_name].iloc[0]
        macros[f"{macro_name}Effect"] = fmt(row.effect)
        macros[f"{macro_name}Low"] = fmt(row.ci_low)
        macros[f"{macro_name}High"] = fmt(row.ci_high)

    for block, word in {2: "Two", 4: "Four", 8: "Eight"}.items():
        row = sensitivity[
            (sensitivity.method_a == "piggycast_full")
            & (
                sensitivity.method_b
                == "validation_selected_model"
            )
            & (sensitivity.metric == "acc")
            & (sensitivity.block_length == block)
        ].iloc[0]
        macros[f"SelectedBlock{word}Low"] = fmt(row.ci_low)

    PAPER.mkdir(exist_ok=True)
    (PAPER / "generated_numbers.tex").write_text(
        "\n".join(
            f"\\newcommand{{\\{key}}}{{{value}}}"
            for key, value in macros.items()
        )
        + "\n",
        encoding="utf-8",
    )

    tables = PAPER / "tables"
    tables.mkdir(exist_ok=True)
    _write_score_table(
        tables / "headline.tex",
        overall,
        HEADLINE_METHODS,
    )
    _write_score_table(
        tables / "all_methods.tex",
        overall,
        ALL_METHODS,
        include_wet_error=True,
    )

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Paired ACC comparison & Mean & 2.5th pct. & 97.5th pct. \\",
        r"\midrule",
    ]
    for label, row in [
        ("Full $-$ equal weight", full_equal),
        ("Full $-$ per-lead selected", full_selected),
        ("Forecast-only $-$ location/calendar", forecast_location),
    ]:
        lines.append(
            f"{label} & {row.effect:.3f} & {row.ci_low:.3f} & "
            f"{row.ci_high:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "ablations.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Domain: full $-$ equal weight & Mean & 2.5th pct. & 97.5th pct. \\",
        r"\midrule",
    ]
    for row in regional[regional.region != "india"].itertuples():
        label = row.region.replace("_", " ")
        lines.append(
            f"{label} & {row.effect:.3f} & {row.ci_low:.3f} & "
            f"{row.ci_high:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "regions.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    lines = [
        r"\begin{tabular}{rrr}",
        r"\toprule",
        r"Block length & Full $-$ equal & Full $-$ per-lead selected \\",
        r"\midrule",
    ]
    for block in (2, 4, 8):
        equal_row = sensitivity[
            (sensitivity.method_a == "piggycast_full")
            & (sensitivity.method_b == "equal_weight")
            & (sensitivity.metric == "acc")
            & (sensitivity.block_length == block)
        ].iloc[0]
        selected_row = sensitivity[
            (sensitivity.method_a == "piggycast_full")
            & (
                sensitivity.method_b
                == "validation_selected_model"
            )
            & (sensitivity.metric == "acc")
            & (sensitivity.block_length == block)
        ].iloc[0]
        lines.append(
            f"{block} & [{equal_row.ci_low:.3f}, "
            f"{equal_row.ci_high:.3f}] & "
            f"[{selected_row.ci_low:.3f}, "
            f"{selected_row.ci_high:.3f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "bootstrap.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    _make_figures(aggregate, paired, regional)


if __name__ == "__main__":
    main()
