#!/usr/bin/env python3
"""Validate generated full-run CSVs for publication-facing checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "s2s_paper_outputs"
TABLES = (
    "deterministic_weekly",
    "probabilistic_weekly",
    "brier_weekly",
    "reliability_weekly",
    "scatter_area_weekly",
    "scatter_grid_weekly",
    "model_status",
)
EXPECTED_UNITS = {"tp": "mm day-1", "z500": "gpm", "t2m": "K"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


def validate_run(output_root: Path, season: str, run_label: str) -> tuple[list[dict], list[dict]]:
    run_dir = output_root / season / "03_metrics" / run_label
    fig_dir = output_root / season / "04_figures" / run_label / "test_plots"
    table_dir = output_root / season / "05_tables" / run_label
    rows: list[dict] = []
    issues: list[dict] = []

    metadata_path = run_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        issues.append({"season": season, "run_label": run_label, "severity": "error", "issue": "missing run_metadata.json"})
    expected_init_count = int(metadata.get("init_count", 0) or 0)
    expected_weeks = set(metadata.get("weeks", []))

    for table in TABLES:
        path = run_dir / f"{table}.csv"
        df = read_csv(path)
        rows.append(
            {
                "season": season,
                "run_label": run_label,
                "artifact": table,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "rows": len(df),
                "columns": len(df.columns),
            }
        )
        if not path.exists() or path.stat().st_size <= 1:
            issues.append({"season": season, "run_label": run_label, "severity": "error", "issue": f"{table}.csv missing or empty"})

    status = read_csv(run_dir / "model_status.csv")
    if not status.empty:
        failed = status[status["status"].astype(str).str.lower().eq("failed")]
        if not failed.empty:
            issues.append(
                {
                    "season": season,
                    "run_label": run_label,
                    "severity": "error",
                    "issue": f"model_status has {len(failed)} failed rows",
                }
            )
        worker_failed = status[
            status["stage"].astype(str).str.contains("worker", na=False)
            & status["status"].astype(str).str.lower().eq("failed")
        ]
        if not worker_failed.empty:
            issues.append(
                {
                    "season": season,
                    "run_label": run_label,
                    "severity": "error",
                    "issue": f"worker failures remain: {len(worker_failed)} rows",
                }
            )

    det = read_csv(run_dir / "deterministic_weekly.csv")
    if not det.empty:
        if {"variable", "unit"}.issubset(det.columns):
            for variable, expected in EXPECTED_UNITS.items():
                actual = sorted(set(det.loc[det["variable"].eq(variable), "unit"].dropna().astype(str)))
                if actual and actual != [expected]:
                    issues.append(
                        {
                            "season": season,
                            "run_label": run_label,
                            "severity": "error",
                            "issue": f"{variable} unit mismatch: expected {expected}, got {actual}",
                        }
                    )
        if expected_init_count and "init_date" in det:
            observed = int(det["init_date"].nunique())
            if observed > expected_init_count:
                issues.append(
                    {
                        "season": season,
                        "run_label": run_label,
                        "severity": "error",
                        "issue": f"deterministic init count {observed} exceeds metadata init_count {expected_init_count}",
                    }
                )
        if expected_weeks and "week" in det:
            observed_weeks = set(int(w) for w in det["week"].dropna().unique())
            if not observed_weeks.issubset(expected_weeks):
                issues.append(
                    {
                        "season": season,
                        "run_label": run_label,
                        "severity": "error",
                        "issue": f"unexpected weeks in deterministic table: {sorted(observed_weeks - expected_weeks)}",
                    }
                )

    figures = list(fig_dir.rglob("*.png")) if fig_dir.exists() else []
    summary_tables = list(table_dir.glob("*.csv")) if table_dir.exists() else []
    rows.append(
        {
            "season": season,
            "run_label": run_label,
            "artifact": "figures_png",
            "exists": bool(figures),
            "bytes": sum(path.stat().st_size for path in figures),
            "rows": len(figures),
            "columns": "",
        }
    )
    rows.append(
        {
            "season": season,
            "run_label": run_label,
            "artifact": "summary_tables",
            "exists": bool(summary_tables),
            "bytes": sum(path.stat().st_size for path in summary_tables),
            "rows": len(summary_tables),
            "columns": "",
        }
    )
    if not figures:
        issues.append({"season": season, "run_label": run_label, "severity": "warning", "issue": "no PNG figures found"})
    if not summary_tables:
        issues.append({"season": season, "run_label": run_label, "severity": "warning", "issue": "no summary table CSVs found"})
    return rows, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("SEASON", "RUN_LABEL"),
        required=True,
        help="Run to validate, e.g. --run jjas2019 full_jjas2019_common17_fuxi_imd",
    )
    parser.add_argument("--out-label", default="publication_validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    issues: list[dict] = []
    for season, run_label in args.run:
        run_rows, run_issues = validate_run(args.output_root, season, run_label)
        rows.extend(run_rows)
        issues.extend(run_issues)

    out_dir = args.output_root / "common" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.out_label}_artifacts.csv"
    issues_path = out_dir / f"{args.out_label}_issues.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(issues, columns=["season", "run_label", "severity", "issue"]).to_csv(issues_path, index=False)
    print(f"Summary: {summary_path}")
    print(f"Issues: {issues_path}")
    if issues:
        print(pd.DataFrame(issues).to_string(index=False))
        return 1 if any(issue["severity"] == "error" for issue in issues) else 0
    print("No validation issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
