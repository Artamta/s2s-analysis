"""Contract tests for compact public website data."""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "public/data"


def load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON: {value}")
        ),
    )


def test_forecast_export_has_six_weeks_and_four_products() -> None:
    forecast = load(PUBLIC_DATA / "forecasts/20260728.json")
    schema = load(ROOT / "science/web-data.schema.json")
    jsonschema.Draft202012Validator(schema).validate(forecast)
    expected = {
        "rainfall_total",
        "rainfall_anomaly",
        "temperature_mean",
        "temperature_anomaly",
    }
    assert set(forecast["products"]) == expected
    assert len(forecast["weeks"]) == 6
    for week_number, week in enumerate(forecast["weeks"], start=1):
        assert week["week"] == week_number
        assert set(week["fields"]) == expected
        assert set(week["summary"]) == expected
        for values in week["fields"].values():
            array = np.asarray(values)
            assert array.shape == (729,)
            assert np.isfinite(array).all()


def test_export_records_exact_climatology_interpolation() -> None:
    forecast = load(PUBLIC_DATA / "forecasts/20260728.json")
    alignment = forecast["issue"]["climatology_alignment"]
    assert alignment["left_slot"] == "0725"
    assert alignment["right_slot"] == "0728"
    assert alignment["right_weight"] == 2.0 / 3.0
    assert forecast["issue"]["hindcast_years"] == list(range(2002, 2022))


def test_public_baselines_are_never_conflated() -> None:
    formulas = load(PUBLIC_DATA / "formulas.json")
    assert formulas["baseline_separation"] == {
        "fuxi": "2002–2021 native reforecast model climatology",
        "imd": "1991–2020 native daily gauge climatology",
        "imerg": "Final V07B 2001–2022 fixed daily climatology",
    }
    forecast = load(PUBLIC_DATA / "forecasts/20260728.json")
    for product in ("rainfall_anomaly", "temperature_anomaly"):
        assert forecast["products"][product]["baseline"].startswith("FuXi-S2S")


def test_failed_validation_would_block_presentation() -> None:
    validation = load(PUBLIC_DATA / "validation.json")
    assert validation["presentation_allowed"] is (
        validation["overall_status"] != "failure"
    )


def test_public_json_contains_no_internal_paths_or_nonfinite_constants() -> None:
    forbidden = [
        re.compile(r"/storage/", re.IGNORECASE),
        re.compile(r"/home/", re.IGNORECASE),
        re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    ]
    for path in PUBLIC_DATA.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        load(path)
        assert all(pattern.search(text) is None for pattern in forbidden), path
        assert "NaN" not in text
        assert "Infinity" not in text


def test_only_compact_json_is_present_in_public_data() -> None:
    forbidden_suffixes = {".nc", ".grib", ".grb", ".onnx", ".7z"}
    assert not [
        path
        for path in PUBLIC_DATA.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
