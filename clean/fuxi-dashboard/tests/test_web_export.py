"""Contract tests for compact public website data."""

from __future__ import annotations

import hashlib
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


def test_initial_condition_sources_are_never_conflated() -> None:
    index = load(PUBLIC_DATA / "index.json")
    assert index["default_view"] == "india"
    assert index["default_source"] == "gfs"
    sources = index["initial_condition_sources"]
    assert [source["id"] for source in sources] == ["gfs", "era5"]
    assert {issue["id"] for source in sources for issue in source["issues"]} >= {
        "20260715",
        "20260722",
        "20260728",
        "20260730",
        "20260731",
    }
    for source in sources:
        for issue in source["issues"]:
            forecast = load(PUBLIC_DATA / issue["forecast"])
            assert forecast["issue"]["initial_condition_source"]["id"] == source["id"]
            assert forecast["issue"]["members"] == issue["members"]


def test_regional_outlooks_use_complete_ensembles_and_locked_terciles() -> None:
    index = load(PUBLIC_DATA / "index.json")
    expected_regions = [
        "all_india",
        "northwest",
        "central",
        "south_peninsula",
        "east_northeast",
    ]
    regional_count = 0
    for source in index["initial_condition_sources"]:
        for issue in source["issues"]:
            regional_path = issue.get("regional_outlook")
            capabilities = issue.get("capabilities", {})
            regional_advertised = capabilities.get(
                "regional_probabilities", bool(regional_path)
            )
            if not regional_advertised:
                assert regional_path is None
                continue
            assert regional_path is not None
            regional_count += 1
            outlook = load(PUBLIC_DATA / regional_path)
            assert outlook["issue"]["source_id"] == source["id"]
            assert outlook["issue"]["initialization"] == issue["initialization"]
            assert outlook["issue"]["members"] == 100
            assert outlook["issue"]["hindcast_years"] == list(range(2002, 2022))
            assert outlook["region_definition"]["name"] == (
                "IMD four broad homogeneous rainfall regions"
            )
            assert len(outlook["weeks"]) == 6
            for week_number, week in enumerate(outlook["weeks"], start=1):
                assert week["week"] == week_number
                assert [region["id"] for region in week["regions"]] == expected_regions
                for variable in ("rainfall", "temperature"):
                    probability = week["probability_fields"][variable]
                    arrays = [
                        np.asarray(probability[category])
                        for category in (
                            "below_normal",
                            "near_normal",
                            "above_normal",
                        )
                    ]
                    assert all(array.shape == (729,) for array in arrays)
                    assert all(np.issubdtype(array.dtype, np.integer) for array in arrays)
                    assert all(np.all((array >= 0) & (array <= 100)) for array in arrays)
                    np.testing.assert_array_equal(arrays[0] + arrays[1] + arrays[2], 100)
                for region in week["regions"]:
                    for variable in ("rainfall", "temperature"):
                        probability = region[variable]["tercile_probability_percent"]
                        assert sum(
                            probability[category]
                            for category in (
                                "below_normal",
                                "near_normal",
                                "above_normal",
                            )
                        ) == 100
    expected_count = sum(
        issue.get("capabilities", {}).get(
            "regional_probabilities", bool(issue.get("regional_outlook"))
        )
        for source in index["initial_condition_sources"]
        for issue in source["issues"]
    )
    assert regional_count == expected_count


def test_each_issue_has_one_verified_pdf_download() -> None:
    index = load(PUBLIC_DATA / "index.json")
    public_root = PUBLIC_DATA.parent
    for source in index["initial_condition_sources"]:
        for issue in source["issues"]:
            forecast = load(PUBLIC_DATA / issue["forecast"])
            downloads = forecast["issue"]["downloads"]
            assert set(downloads) == {
                "compact_json",
                "india_pdf",
                "india_pdf_sha256",
            }
            pdf_path = public_root / downloads["india_pdf"]
            pdf_bytes = pdf_path.read_bytes()
            assert pdf_bytes.startswith(b"%PDF-")
            assert len(re.findall(rb"/Type /Page\b", pdf_bytes)) == len(
                forecast["products"]
            )
            assert pdf_path.stat().st_size < 10_000_000
            assert hashlib.sha256(pdf_bytes).hexdigest() == downloads["india_pdf_sha256"]


def test_india_map_geography_is_compact_and_source_locked() -> None:
    geography_path = PUBLIC_DATA / "india-map-geography.json"
    geography = load(geography_path)
    assert geography_path.stat().st_size < 200_000
    assert geography["view_box"] == [0, 0, 620, 620]
    for key in ("world_path", "india_outline_path", "india_admin_path"):
        assert geography[key].startswith("M")
        assert len(geography[key]) > 1_000
    sources = {
        "world_countries_sha256": PUBLIC_DATA / "world-countries.geojson",
        "india_outline_sha256": PUBLIC_DATA / "india-outline.json",
        "india_admin_sha256": PUBLIC_DATA / "india-admin.json",
    }
    for key, source_path in sources.items():
        assert geography["sources"][key] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
