#!/usr/bin/env python3
"""Run ordered integrity, structure, and physical gates on prototype sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from science.validators import (  # noqa: E402
    ValidationCheck,
    combine_status,
    utc_now,
    validate_forecast,
    validate_fuxi_climatology,
    validate_imd_climatology,
    validate_imerg_climatology,
    write_json,
)

DEFAULT_FORECAST = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "fuxi_s2s_gfs_proxy_case_20260728_ens100/forecasts/annual2026/20260728.nc"
)
DEFAULT_FUXI_CLIMO = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_jjas_2002_2021/"
    "fuxi_s2s_jjas_model_climatology_2002_2021_loyo.nc"
)
DEFAULT_IMD_CLIMO = Path(
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/"
    "climatology/imd_rain_1991_2020_daily_climatology.nc"
)
DEFAULT_IMERG_CLIMO = (
    ROOT.parent
    / "deliverables/fuxi_erpas_imd_imerg_review_2023_2024/data/"
    "imerg_climatology/imerg_final_v07b_climatology_2001_2022_1p5_daily.nc"
)
SOURCE_DIGESTS = {
    "forecast": "59bf922f03159666603c38158b14f015fdd0d9c566249853e2cf90c96ffa3085",
    "fuxi_climo": "e0bc3c4faf44a8f9d47b8dfc22cf3640cf7018dbacf9b9c876c1dd8cb8b93f2d",
    "imd_climo": "a7b7780ad46dc9bbd3127f74714e76c33559b98ce377c8d92a9002c1fa73d62d",
    "imerg_climo": "38e066a3d0efccb38977e9d2a6f243d94e548a6a7d5dd33f67592efc2d7247b8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST)
    parser.add_argument("--fuxi-climatology", type=Path, default=DEFAULT_FUXI_CLIMO)
    parser.add_argument("--imd-climatology", type=Path, default=DEFAULT_IMD_CLIMO)
    parser.add_argument("--imerg-climatology", type=Path, default=DEFAULT_IMERG_CLIMO)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "public/data/validation.json"
    )
    return parser.parse_args()


def failed_check(identifier: str, label: str, error: Exception) -> ValidationCheck:
    """Convert a validation exception into a public-safe failure."""

    return ValidationCheck(
        id=identifier,
        label=label,
        group="source",
        status="failure",
        summary=f"Validation failed: {type(error).__name__}: {error}",
    )


def main() -> int:
    args = parse_args()
    jobs = (
        (
            "fuxi_forecast",
            "FuXi forecast",
            validate_forecast,
            args.forecast,
            SOURCE_DIGESTS["forecast"],
        ),
        (
            "fuxi_climatology",
            "FuXi climatology",
            validate_fuxi_climatology,
            args.fuxi_climatology,
            SOURCE_DIGESTS["fuxi_climo"],
        ),
        (
            "imd_climatology",
            "IMD climatology",
            validate_imd_climatology,
            args.imd_climatology,
            SOURCE_DIGESTS["imd_climo"],
        ),
        (
            "imerg_climatology",
            "IMERG climatology",
            validate_imerg_climatology,
            args.imerg_climatology,
            SOURCE_DIGESTS["imerg_climo"],
        ),
    )
    checks: list[ValidationCheck] = []
    for identifier, label, validator, path, digest in jobs:
        try:
            check = validator(path, digest)
        except Exception as error:  # validation failures belong in the report
            check = failed_check(identifier, label, error)
        checks.append(check)
        print(f"{check.status:7s} {check.label}: {check.summary}")
    overall_status = combine_status(checks)
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "overall_status": overall_status,
        "presentation_allowed": overall_status != "failure",
        "checks": [check.as_json() for check in checks],
        "warnings": [
            check.summary for check in checks if check.status == "warning"
        ],
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    return 1 if overall_status == "failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
