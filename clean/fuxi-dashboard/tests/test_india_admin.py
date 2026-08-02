"""Checks for the compact Survey of India ABDB display derivative."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDIA_ADMIN = ROOT / "public/data/india-admin.json"


def test_india_admin_has_complete_state_layer_and_provenance() -> None:
    document = json.loads(INDIA_ADMIN.read_text(encoding="utf-8"))
    assert document["type"] == "FeatureCollection"
    assert len(document["features"]) == 40
    assert sum(feature["properties"]["label"] for feature in document["features"]) == 36
    names = {feature["properties"]["name"] for feature in document["features"]}
    assert {
        "JAMMU AND KASHMIR",
        "LADAKH",
        "ARUNACHAL PRADESH",
        "ANDAMAN & NICOBAR",
    } <= names
    source = document["source"]
    assert source["name"] == "Survey of India Administrative Boundary Database (ABDB)"
    assert len(source["source_sha256"]) == 64
    assert "legal or cadastral" in source["display_note"]


def test_india_admin_is_compact_and_contains_no_source_path() -> None:
    text = INDIA_ADMIN.read_text(encoding="utf-8")
    assert INDIA_ADMIN.stat().st_size < 500_000
    assert "/storage/" not in text
    assert "/home/" not in text
    for feature in json.loads(text)["features"]:
        assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}
        longitude = feature["properties"]["label_longitude"]
        latitude = feature["properties"]["label_latitude"]
        assert 60 <= longitude <= 100
        assert 0 <= latitude <= 40
