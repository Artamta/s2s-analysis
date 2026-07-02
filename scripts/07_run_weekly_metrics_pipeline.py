#!/usr/bin/env python3
"""Run the benchmark weekly S2S metrics pipeline.

Examples
--------
Focused smoke tests:
  python scripts/08_run_smoke_metric_cases.py --case jjas_tp
  python scripts/08_run_smoke_metric_cases.py --case jfm_tp_spire
  python scripts/07_run_weekly_metrics_pipeline.py --season jjas2019 --set-name operational_models --variables tp --models ecmwf ukmo ncep --smoke --run-label test_tp

Full runs:
  python scripts/07_run_weekly_metrics_pipeline.py --season jjas2019 --run-label full
  python scripts/07_run_weekly_metrics_pipeline.py --season jfm2026 --include-spire --run-label full_daily_spire --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from s2s_benchmark.analysis_pipeline import DEFAULT_OUTPUT_ROOT, run_weekly_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", choices=["jjas2019", "jfm2026"], required=True)
    parser.add_argument(
        "--set-name",
        default=None,
        help="Matched-init set from scripts/05. Defaults to delysm_operational for JJAS and all_usable_models for JFM.",
    )
    parser.add_argument("--variables", nargs="+", default=None, help="Default: tp z500 t2m")
    parser.add_argument("--models", nargs="+", default=None, help="Override models from matched-init set.")
    parser.add_argument(
        "--truth-source",
        choices=["auto", "era5", "imd", "both"],
        default="auto",
        help="TP verification truth. auto=JJAS IMD and JFM ERA5. both runs ERA5 and IMD into separate folders.",
    )
    parser.add_argument("--include-spire", action="store_true", help="Add daily SPIRE s2s-research.zarr inits for JFM2026.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=None, help="Output folder label under 03_metrics/.")
    parser.add_argument("--smoke", action="store_true", help="Use 1 init and Week 1 unless --max-inits/--weeks override.")
    parser.add_argument("--max-inits", type=int, default=None, help="Limit number of init dates. Use 0 for all.")
    parser.add_argument("--weeks", nargs="+", type=int, default=None, help="Week numbers to run, e.g. --weeks 1 2 3.")
    parser.add_argument(
        "--min-leads-for-mean",
        type=int,
        default=2,
        help="Minimum non-cumulative lead samples needed for a weekly mean.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel init-date workers. Use 1 for serial.")
    parser.add_argument(
        "--no-grid-scatter",
        action="store_true",
        help="Write only region-mean scatter pairs, not grid-cell scatter pairs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_name = args.set_name
    if set_name is None:
        set_name = "delysm_operational" if args.season == "jjas2019" else "all_usable_models"

    if args.max_inits == 0:
        max_inits = None
    elif args.max_inits is not None:
        max_inits = args.max_inits
    elif args.smoke:
        max_inits = 1
    else:
        max_inits = None

    weeks = args.weeks
    if weeks is None and args.smoke:
        weeks = [1]

    base_run_label = args.run_label or ("test" if args.smoke else "full")
    truth_sources = ["era5", "imd"] if args.truth_source == "both" else [args.truth_source]
    for truth_source in truth_sources:
        run_label = base_run_label
        if args.truth_source == "both" or (args.run_label is None and truth_source != "auto"):
            run_label = f"{base_run_label}_{truth_source}truth"

        result = run_weekly_pipeline(
            season=args.season,
            set_name=set_name,
            truth_source=truth_source,
            variables=args.variables,
            models=args.models,
            include_spire=args.include_spire,
            output_root=args.output_root,
            run_label=run_label,
            max_inits=max_inits,
            weeks=weeks,
            min_leads_for_mean=args.min_leads_for_mean,
            workers=args.workers,
            include_grid_scatter=not args.no_grid_scatter,
        )

        print(f"\ntruth_source={truth_source} run_label={run_label}")
        print(json.dumps(result["row_counts"], indent=2))
        for name, path in result["outputs"].items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
