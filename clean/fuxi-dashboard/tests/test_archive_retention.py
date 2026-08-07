"""Retention tests for the bounded public forecast archive."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from scripts.apply_archive_retention import apply_retention
from scripts.publish_operational_site import allowed_retention_change

UTC = timezone.utc


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pdf_then_interactive_assets_expire_but_metadata_remains(
    tmp_path: Path,
) -> None:
    public = tmp_path / "public"
    forecast = public / "data/forecasts/gfs/20240101.json"
    pdf = public / "downloads/gfs/20240101/briefing.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-retention-test")
    write(
        forecast,
        {
            "issue": {
                "downloads": {
                    "compact_json": "data/forecasts/gfs/20240101.json",
                    "india_pdf": "downloads/gfs/20240101/briefing.pdf",
                    "india_pdf_sha256": "0" * 64,
                }
            }
        },
    )
    index_path = public / "data/index.json"
    write(
        index_path,
        {
            "current": {"source_id": "gfs", "issue_id": "20260805"},
            "latest_reference": {"source_id": "era5", "issue_id": "20260722"},
            "initial_condition_sources": [
                {
                    "id": "gfs",
                    "issues": [
                        {
                            "id": "20240101",
                            "initialization": "2024-01-01T00:00:00Z",
                            "forecast": "forecasts/gfs/20240101.json",
                            "pdf": "downloads/gfs/20240101/briefing.pdf",
                            "capabilities": {"pdf": True},
                            "checksums": {"pdf_sha256": "0" * 64},
                            "archive": {
                                "interactive_available": True,
                                "pdf_available": True,
                                "metadata": "indefinite",
                            },
                        }
                    ],
                }
            ],
        },
    )

    apply_retention(
        public,
        index_path,
        datetime(2024, 3, 5, tzinfo=UTC),
    )
    after_pdf = json.loads(index_path.read_text(encoding="utf-8"))
    record = after_pdf["initial_condition_sources"][0]["issues"][0]
    assert forecast.is_file()
    assert not pdf.exists()
    assert record["archive"]["pdf_available"] is False
    forecast_payload = json.loads(forecast.read_text(encoding="utf-8"))
    assert "india_pdf" not in forecast_payload["issue"]["downloads"]
    assert record["checksums"]["forecast_sha256"] == hashlib.sha256(
        forecast.read_bytes()
    ).hexdigest()

    apply_retention(
        public,
        index_path,
        datetime(2025, 1, 2, tzinfo=UTC),
    )
    after_maps = json.loads(index_path.read_text(encoding="utf-8"))
    record = after_maps["initial_condition_sources"][0]["issues"][0]
    assert not forecast.exists()
    assert record["archive"]["interactive_available"] is False
    assert record["archive"]["metadata"] == "indefinite"


def test_publisher_allows_only_expired_retention_paths() -> None:
    now = datetime(2026, 8, 7, tzinfo=UTC)
    assert allowed_retention_change(
        "clean/fuxi-dashboard/public/downloads/gfs/20260601/briefing.pdf",
        "clean/fuxi-dashboard",
        now,
    )
    assert not allowed_retention_change(
        "clean/fuxi-dashboard/public/downloads/gfs/20260805/briefing.pdf",
        "clean/fuxi-dashboard",
        now,
    )
    assert not allowed_retention_change(
        "clean/fuxi-dashboard/src/main.ts",
        "clean/fuxi-dashboard",
        now,
    )
