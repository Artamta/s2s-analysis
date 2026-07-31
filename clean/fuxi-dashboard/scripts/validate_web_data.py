#!/usr/bin/env python3
"""Run formula, independent-builder, schema, and publication validation gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from science.formulas import (  # noqa: E402
    anomaly,
    area_weights,
    calendar_interpolation,
    kelvin_to_celsius,
    tp_mm_hour_to_mm_day,
    verification_metrics,
    weekly_mean_rainfall,
    weekly_total,
)
from science.validators import (  # noqa: E402
    ValidationCheck,
    combine_status,
    contains_nonfinite_json,
    utc_now,
    write_json,
)
from scripts.stamp_deploy import build_manifest  # noqa: E402
from scripts.validate_sources import DEFAULT_FORECAST, DEFAULT_FUXI_CLIMO  # noqa: E402

LEGACY_BUILDER = (
    REPOSITORY_ROOT / "forecast/ic_20260728_gfs_proxy/build_imd_package.py"
)
FORBIDDEN_PUBLIC_PATTERNS = (
    re.compile(r"/storage/", re.IGNORECASE),
    re.compile(r"/home/", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"\b(password|secret|access[_-]?token)\b\s*[:=]", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST)
    parser.add_argument("--fuxi-climatology", type=Path, default=DEFAULT_FUXI_CLIMO)
    parser.add_argument(
        "--public-data", type=Path, default=ROOT / "public/data"
    )
    parser.add_argument(
        "--validation", type=Path, default=ROOT / "public/data/validation.json"
    )
    return parser.parse_args()


def run_check(
    identifier: str,
    label: str,
    group: str,
    function: Callable[[], tuple[str, dict[str, Any]]],
) -> ValidationCheck:
    """Run a validation function and convert exceptions to safe failures."""

    try:
        summary, details = function()
        return ValidationCheck(
            id=identifier,
            label=label,
            group=group,
            status="green",
            summary=summary,
            details=details,
        )
    except Exception as error:
        print(f"FAIL {identifier}: {type(error).__name__}: {error}")
        return ValidationCheck(
            id=identifier,
            label=label,
            group=group,
            status="failure",
            summary=f"{type(error).__name__} during validation; see local build log.",
        )


def formula_gate() -> tuple[str, dict[str, Any]]:
    """Independently exercise locked formulas with hand-calculated values."""

    converted = tp_mm_hour_to_mm_day([0.0, 0.5, 1.0])
    np.testing.assert_allclose(converted, [0.0, 12.0, 24.0], rtol=0.0, atol=0.0)
    daily = np.arange(1.0, 15.0)
    np.testing.assert_allclose(
        weekly_total(daily), [28.0, 77.0], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        weekly_mean_rainfall(daily), [4.0, 11.0], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        kelvin_to_celsius([273.15, 300.0]), [0.0, 26.85], atol=1e-12
    )
    np.testing.assert_allclose(anomaly([5.0, 2.0], [3.0, 4.0]), [2.0, -2.0])
    np.testing.assert_allclose(
        calendar_interpolation([1.0], [4.0], 2.0 / 3.0), [3.0]
    )
    weights = area_weights([0.0, 60.0], [[1.0], [1.0]])
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-15)
    metrics = verification_metrics(
        [2.0, 4.0, 6.0],
        [1.0, 5.0, 5.0],
        [1.0, 1.0, 1.0],
        forecast_anomaly=[-1.0, 0.0, 1.0],
        observation_anomaly=[-2.0, 0.0, 2.0],
    )
    np.testing.assert_allclose(metrics["bias"], 1.0 / 3.0)
    np.testing.assert_allclose(metrics["mae"], 1.0)
    np.testing.assert_allclose(metrics["rmse"], 1.0)
    np.testing.assert_allclose(metrics["acc"], 1.0)
    return (
        "Conversions, weekly aggregation, anomalies, interpolation, weights, and metrics passed.",
        {"formula_version": "1.0.0", "independent_examples": 9},
    )


def load_json(path: Path) -> Any:
    """Load strict JSON and reject non-finite values."""

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if contains_nonfinite_json(payload):
        raise ValueError(f"non-finite value in {path.name}")
    return payload


def schema_gate(public_data: Path) -> tuple[str, dict[str, Any]]:
    """Validate the forecast schema and exact six-by-four field topology."""

    schema = load_json(ROOT / "science/web-data.schema.json")
    forecast = load_json(public_data / "forecasts/20260728.json")
    jsonschema.Draft202012Validator(schema).validate(forecast)
    expected_products = {
        "rainfall_total",
        "rainfall_anomaly",
        "temperature_mean",
        "temperature_anomaly",
    }
    if set(forecast["products"]) != expected_products:
        raise ValueError("forecast product identities do not match the contract")
    for expected_week, week in enumerate(forecast["weeks"], start=1):
        if week["week"] != expected_week:
            raise ValueError("forecast week order is incorrect")
        if set(week["fields"]) != expected_products:
            raise ValueError("one or more weeks lacks the four required products")
        if set(week["summary"]) != expected_products:
            raise ValueError("one or more weeks lacks product summaries")
    return (
        "JSON Schema and all six weeks × four products passed.",
        {"weeks": 6, "products": 4, "finite_values": 6 * 4 * 729},
    )


def independent_builder_gate(
    public_data: Path, forecast_path: Path, climatology_path: Path
) -> tuple[str, dict[str, Any]]:
    """Compare every exported cell against the established 28 July builder."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fuxi_dashboard_mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fuxi_dashboard_xdg")
    spec = importlib.util.spec_from_file_location(
        "validated_20260728_builder", LEGACY_BUILDER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import the existing product builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analysis, alignment, _ = module.build_analysis(forecast_path, climatology_path)
    forecast = load_json(public_data / "forecasts/20260728.json")
    variable_map = {
        "rainfall_total": "rainfall_weekly_total",
        "rainfall_anomaly": "rainfall_anomaly",
        "temperature_mean": "temperature_weekly_mean",
        "temperature_anomaly": "temperature_anomaly",
    }
    maximum_difference = 0.0
    try:
        for week_index, week in enumerate(forecast["weeks"]):
            for product, variable in variable_map.items():
                actual = np.asarray(week["fields"][product], dtype=np.float64).reshape(
                    27, 27
                )
                expected = analysis[variable].isel(week=week_index).values.astype(
                    np.float64
                )
                difference = float(np.max(np.abs(actual - expected)))
                maximum_difference = max(maximum_difference, difference)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.5e-4)
    finally:
        analysis.close()
    exported_alignment = forecast["issue"]["climatology_alignment"]
    for key in ("left_slot", "right_slot", "right_weight"):
        if exported_alignment[key] != alignment[key]:
            raise ValueError(f"climatology alignment differs for {key}")
    return (
        "All exported fields match the established validated 28 July builder.",
        {
            "compared_values": 6 * 4 * 729,
            "absolute_tolerance": 0.00015,
            "maximum_absolute_difference": maximum_difference,
            "left_slot": alignment["left_slot"],
            "right_slot": alignment["right_slot"],
            "right_weight": alignment["right_weight"],
        },
    )


def publication_gate(public_data: Path) -> tuple[str, dict[str, Any]]:
    """Reject internal paths, credentials, raw assets, and malformed public JSON."""

    files = sorted(public_data.rglob("*.json"))
    required = {
        "index.json",
        "sources.json",
        "formulas.json",
        "validation.json",
        "india-outline.json",
        "india-admin.json",
        "20260728.json",
    }
    present = {path.name for path in files}
    missing = required - present
    if missing:
        raise ValueError(f"missing public files: {sorted(missing)}")
    scanned_bytes = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        scanned_bytes += len(text.encode("utf-8"))
        load_json(path)
        for pattern in FORBIDDEN_PUBLIC_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"forbidden content in public file {path.name}")
        if path.suffix.lower() in {".nc", ".grib", ".grb", ".onnx"}:
            raise ValueError(f"restricted raw asset in public data: {path.name}")
    return (
        "Public JSON is finite and free of internal paths, credentials, and raw assets.",
        {"json_files_scanned": len(files), "bytes_scanned": scanned_bytes},
    )


def main() -> int:
    args = parse_args()
    if not args.validation.is_file():
        raise FileNotFoundError("source validation report is absent")
    validation = load_json(args.validation)
    source_checks = [
        ValidationCheck(**check)
        for check in validation["checks"]
        if check["group"] == "source"
    ]
    new_checks = [
        run_check("formula_tests", "Formula tests", "formula", formula_gate),
        run_check(
            "legacy_builder_crosscheck",
            "Independent 28 July cross-check",
            "formula",
            lambda: independent_builder_gate(
                args.public_data, args.forecast, args.fuxi_climatology
            ),
        ),
        run_check(
            "web_schema",
            "Web export",
            "publication",
            lambda: schema_gate(args.public_data),
        ),
        run_check(
            "publication_safety",
            "Publication safety",
            "publication",
            lambda: publication_gate(args.public_data),
        ),
    ]
    checks = [*source_checks, *new_checks]
    overall = combine_status(checks)
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "overall_status": overall,
        "presentation_allowed": overall != "failure",
        "checks": [check.as_json() for check in checks],
        "warnings": [
            check.summary for check in checks if check.status == "warning"
        ],
        "verification_metrics": {
            "status": "withheld",
            "reason": (
                "Every required observation day is not yet available for the "
                "six complete valid weeks. No partial-period score is published."
            ),
        },
    }
    write_json(args.validation, payload)
    build_manifest()
    for check in new_checks:
        print(f"{check.status:7s} {check.label}: {check.summary}")
    return 1 if overall == "failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
