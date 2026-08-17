#!/usr/bin/env python3
"""Generate manuscript macros and tables only from audited result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts/confirmatory_2025"
PAPER = ROOT / "paper"


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    audit = json.loads((ARTIFACT / "audit_report.json").read_text())
    if audit["status"] != "passed":
        raise RuntimeError("paper generation requires a passing artifact audit")
    aggregate = pd.read_csv(ARTIFACT / "aggregate_metrics.csv")
    paired = pd.read_csv(ARTIFACT / "paired_intervals.csv")
    regional = pd.read_csv(ARTIFACT / "regional_acc_intervals.csv")
    overall = aggregate[(aggregate.aggregation == "all") & (aggregate.region == "india")].set_index("method")
    pair = paired.set_index(["method_a", "method_b", "metric"])
    full_equal = pair.loc[("piggycast_full", "equal_weight", "acc")]
    full_selected = pair.loc[("piggycast_full", "validation_selected_model", "acc")]
    forecast_location = pair.loc[("piggycast_forecast_only", "location_calendar_only", "acc")]
    macros = {
        "TestCases": "35",
        "FullACC": fmt(overall.loc["piggycast_full", "acc"]),
        "EqualACC": fmt(overall.loc["equal_weight", "acc"]),
        "SelectedACC": fmt(overall.loc["validation_selected_model", "acc"]),
        "FullRMSE": fmt(overall.loc["piggycast_full", "rmse"]),
        "EqualRMSE": fmt(overall.loc["equal_weight", "rmse"]),
        "FullBias": fmt(overall.loc["piggycast_full", "bias"]),
        "EqualBias": fmt(overall.loc["equal_weight", "bias"]),
        "FullEqualEffect": fmt(full_equal.effect),
        "FullEqualLow": fmt(full_equal.ci_low),
        "FullEqualHigh": fmt(full_equal.ci_high),
        "FullSelectedEffect": fmt(full_selected.effect),
        "FullSelectedLow": fmt(full_selected.ci_low),
        "FullSelectedHigh": fmt(full_selected.ci_high),
        "ForecastLocationEffect": fmt(forecast_location.effect),
        "ForecastLocationLow": fmt(forecast_location.ci_low),
        "ForecastLocationHigh": fmt(forecast_location.ci_high),
    }
    PAPER.mkdir(exist_ok=True)
    (PAPER / "generated_numbers.tex").write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n", encoding="utf-8")

    tables = PAPER / "tables"
    tables.mkdir(exist_ok=True)
    methods = ["piggycast_full", "piggycast_forecast_only", "validation_weighted", "equal_weight", "ecmwf", "validation_selected_model", "fuxi_s2s", "neuralgcm", "ukmo", "cma", "ncep", "dlesym_v0", "location_calendar_only"]
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Method & ACC $\\uparrow$ & RMSE $\\downarrow$ & MAE $\\downarrow$ & Bias & Wet-area error \\\\", "\\midrule"]
    for method in methods:
        row = overall.loc[method]
        label = method.replace("_", "\\_")
        lines.append(f"{label} & {row.acc:.3f} & {row.rmse:.3f} & {row.mae:.3f} & {row.bias:.3f} & {row.wet_fraction_error:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables / "overall.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Paired ACC comparison & Mean & 2.5th pct. & 97.5th pct. \\\\", "\\midrule"]
    for label, row in [("Full $-$ equal weight", full_equal), ("Full $-$ validation-selected", full_selected), ("Forecast-only $-$ location/calendar", forecast_location)]:
        lines.append(f"{label} & {row.effect:.3f} & {row.ci_low:.3f} & {row.ci_high:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables / "ablations.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Domain: full $-$ equal weight & Mean & 2.5th pct. & 97.5th pct. \\\\", "\\midrule"]
    for row in regional.itertuples():
        lines.append(f"{row.region.replace('_', ' ')} & {row.effect:.3f} & {row.ci_low:.3f} & {row.ci_high:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables / "regions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
