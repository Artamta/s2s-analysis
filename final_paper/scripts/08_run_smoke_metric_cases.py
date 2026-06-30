#!/usr/bin/env python3
"""Run focused smoke cases for the final-paper weekly metrics pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from s2s_paper.analysis_pipeline import run_weekly_pipeline  # noqa: E402


CASES = {
    "jjas_tp": dict(
        season="jjas2019",
        set_name="operational_models",
        variables=["tp"],
        models=["ecmwf", "ukmo", "ncep"],
        run_label="test_tp",
    ),
    "jjas_z500_delysm": dict(
        season="jjas2019",
        set_name="delysm_operational",
        variables=["z500"],
        models=["delysm", "ukmo"],
        run_label="test_z500_delysm_ukmo",
    ),
    "jjas_tp_common17_fuxi": dict(
        season="jjas2019",
        set_name="all_usable_models",
        variables=["tp"],
        models=["ecmwf", "ukmo", "ncep", "fuxi"],
        run_label="test_tp_common17_fuxi",
    ),
    "jjas_z500_common17_fuxi": dict(
        season="jjas2019",
        set_name="all_usable_models",
        variables=["z500"],
        models=["delysm", "ecmwf", "ukmo", "ncep", "fuxi"],
        run_label="test_z500_common17_fuxi",
    ),
    "jfm_tp_spire": dict(
        season="jfm2026",
        set_name="all_usable_models",
        variables=["tp"],
        models=["spire", "ecmwf", "ukmo", "ncep", "fuxi"],
        include_spire=True,
        run_label="test_tp_spire",
    ),
    "jfm_z500_spire_delysm": dict(
        season="jfm2026",
        set_name="all_usable_models",
        variables=["z500"],
        models=["spire", "delysm", "ukmo"],
        include_spire=True,
        run_label="test_z500_spire_delysm_ukmo",
    ),
    "jfm_t2m_spire_delysm_fuxi": dict(
        season="jfm2026",
        set_name="all_usable_models",
        variables=["t2m"],
        models=["spire", "delysm", "fuxi"],
        include_spire=True,
        run_label="test_t2m_spire_delysm_fuxi",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument(
        "--truth-source",
        choices=["auto", "era5", "imd", "both"],
        default="auto",
        help="TP verification truth. both writes separate smoke folders for ERA5 and IMD.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = list(CASES) if args.case == "all" else [args.case]
    truth_sources = ["era5", "imd"] if args.truth_source == "both" else [args.truth_source]
    for name in names:
        for truth_source in truth_sources:
            kwargs = dict(CASES[name])
            kwargs.setdefault("include_spire", False)
            if args.truth_source == "both" or truth_source != "auto":
                kwargs["run_label"] = f"{kwargs['run_label']}_{truth_source}truth"
            kwargs.update(truth_source=truth_source, max_inits=1, weeks=[1])
            print(f"\n=== {name} truth_source={truth_source} ===", flush=True)
            result = run_weekly_pipeline(**kwargs)
            print(json.dumps(result["row_counts"], indent=2))
            for output_name, path in result["outputs"].items():
                print(f"{output_name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
