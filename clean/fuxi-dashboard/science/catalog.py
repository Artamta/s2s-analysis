"""Build the public, source-aware issue catalog from validated web assets.

The catalog is deliberately derived from files that are already safe for the
website.  It never scans private forecast storage and it advances the Current
pointer only when a complete 100-member GFS package is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .validators import write_json

UTC = timezone.utc
PRODUCTION_MEMBERS = 100
RAW_PRODUCTS = {"rainfall_total", "temperature_mean"}
ANOMALY_PRODUCTS = {"rainfall_anomaly", "temperature_anomaly"}
CADENCE_WEEKDAYS = (2, 5)  # Wednesday and Saturday, datetime.weekday().
TARGET_HOUR_UTC = 12


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_datetime(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_z(value: datetime) -> str:
    """Serialize a UTC timestamp using a stable ``Z`` suffix."""

    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def next_expected_update(now: datetime) -> datetime:
    """Return the next Wednesday/Saturday publication target at 12:00 UTC."""

    normalized = now.astimezone(UTC)
    for offset in range(8):
        candidate_day = (normalized + timedelta(days=offset)).date()
        candidate = datetime.combine(
            candidate_day,
            datetime.min.time(),
            tzinfo=UTC,
        ).replace(hour=TARGET_HOUR_UTC)
        if candidate.weekday() in CADENCE_WEEKDAYS and candidate > normalized:
            return candidate
    raise AssertionError("a twice-weekly update must exist within eight days")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _public_asset(public_root: Path, relative: str) -> Path:
    candidate = (public_root / relative).resolve()
    root = public_root.resolve()
    if root not in candidate.parents:
        raise ValueError(f"public path escapes the public root: {relative}")
    return candidate


def _forecast_asset(public_root: Path, relative: str) -> Path:
    return _public_asset(public_root, f"data/{relative}")


def enrich_issue(
    issue: dict[str, Any],
    source_id: str,
    public_root: Path,
    now: datetime,
) -> tuple[dict[str, Any], bool]:
    """Add capabilities, checksums, retention, and presentation metadata."""

    result = deepcopy(issue)
    forecast_path = _forecast_asset(public_root, str(result["forecast"]))
    existing_archive = result.get("archive", {})
    if (
        not forecast_path.is_file()
        and isinstance(existing_archive, dict)
        and existing_archive.get("interactive_available") is False
    ):
        # Metadata is retained indefinitely after the interactive asset expires.
        # The retention worker has already validated and removed the old file.
        return result, False
    forecast = _load_json(forecast_path)
    forecast_issue = forecast.get("issue", {})
    if forecast_issue.get("initial_condition_source", {}).get("id") != source_id:
        raise ValueError(f"source mismatch for {source_id}/{result['id']}")
    if int(forecast_issue.get("members", -1)) != int(result["members"]):
        raise ValueError(f"member mismatch for {source_id}/{result['id']}")

    products = list(forecast.get("products", {}))
    product_set = set(products)
    if not RAW_PRODUCTS.issubset(product_set):
        raise ValueError(f"raw rainfall/temperature fields are missing: {forecast_path}")
    anomalies = ANOMALY_PRODUCTS.issubset(product_set)

    regional_relative = result.get("regional_outlook")
    regional_path = (
        _forecast_asset(public_root, str(regional_relative))
        if regional_relative
        else None
    )
    regional_available = bool(regional_path and regional_path.is_file())
    if regional_relative and not regional_available:
        raise FileNotFoundError(regional_path)

    downloads = forecast_issue.get("downloads", {})
    pdf_relative = downloads.get("india_pdf")
    pdf_path = (
        _public_asset(public_root, str(pdf_relative)) if pdf_relative else None
    )
    pdf_available = bool(pdf_path and pdf_path.is_file())
    pdf_checksum = sha256(pdf_path) if pdf_available and pdf_path else None
    declared_pdf_checksum = downloads.get("india_pdf_sha256")
    if declared_pdf_checksum and pdf_checksum != declared_pdf_checksum:
        raise ValueError(f"PDF checksum mismatch for {source_id}/{result['id']}")

    initialization = parse_datetime(str(result["initialization"]))
    published_at = str(forecast.get("generated_at") or result["initialization"])
    valid_through = str(forecast.get("weeks", [{}])[-1].get("valid_end", ""))
    production = int(result["members"]) == PRODUCTION_MEMBERS
    member_class = "production" if production else "limited"
    if not production:
        label = "Limited experiment"
    elif source_id == "era5":
        label = "Delayed ERA5 reference"
    else:
        label = "Archived guidance"

    interactive_until = initialization + timedelta(days=365)
    pdf_until = initialization + timedelta(days=56)
    retained_interactive = forecast_path.is_file() and now <= interactive_until
    retained_pdf = pdf_available and now <= pdf_until
    result.update(
        {
            "available_products": products,
            "capabilities": {
                "raw_fields": True,
                "anomalies": anomalies,
                "regional_probabilities": regional_available,
                "pdf": pdf_available,
            },
            "presentation": {
                "section": "archive",
                "label": label,
                "member_class": member_class,
            },
            "checksums": {
                "forecast_sha256": sha256(forecast_path),
                **(
                    {"regional_sha256": sha256(regional_path)}
                    if regional_available and regional_path
                    else {}
                ),
                **({"pdf_sha256": pdf_checksum} if pdf_checksum else {}),
            },
            "valid_through": valid_through,
            "published_at": published_at,
            **({"pdf": str(pdf_relative)} if retained_pdf and pdf_relative else {}),
            "archive": {
                "interactive_available": retained_interactive,
                "pdf_available": retained_pdf,
                "interactive_until": iso_z(interactive_until),
                "pdf_until": iso_z(pdf_until),
                "metadata": "indefinite",
            },
        }
    )
    complete = bool(
        production
        and forecast_path.is_file()
        and retained_pdf
        and (not regional_relative or regional_available)
    )
    return result, complete


def _pointer(issue: dict[str, Any], source_id: str) -> dict[str, Any]:
    pointer = {
        "source_id": source_id,
        "issue_id": issue["id"],
        "initialization": issue["initialization"],
        "members": issue["members"],
        "forecast": issue["forecast"],
        "forecast_sha256": issue["checksums"]["forecast_sha256"],
        "valid_through": issue["valid_through"],
        "published_at": issue["published_at"],
        "available_products": issue["available_products"],
        "capabilities": issue["capabilities"],
    }
    if issue.get("regional_outlook"):
        pointer["regional_outlook"] = issue["regional_outlook"]
    if issue.get("pdf"):
        pointer["pdf"] = issue["pdf"]
    return pointer


def build_catalog(
    existing: dict[str, Any],
    public_root: Path,
    now: datetime,
) -> dict[str, Any]:
    """Return schema-v3 metadata while preserving unrelated viewer settings."""

    payload = deepcopy(existing)
    sources = payload.get("initial_condition_sources", [])
    if [source.get("id") for source in sources] != ["gfs", "era5"]:
        raise ValueError("catalog sources must be ordered GFS then ERA5")

    complete_by_source: dict[str, list[dict[str, Any]]] = {"gfs": [], "era5": []}
    for source in sources:
        source_id = str(source["id"])
        enriched: list[dict[str, Any]] = []
        for issue in source.get("issues", []):
            item, complete = enrich_issue(issue, source_id, public_root, now)
            enriched.append(item)
            if complete:
                complete_by_source[source_id].append(item)
        enriched.sort(key=lambda item: item["id"], reverse=True)
        source["issues"] = enriched
        if source_id == "era5":
            source["availability"] = {
                "mode": "delayed_reference",
                "typical_lag_days": "5–7",
                "notice": "ERA5 is a delayed research reference, never live guidance.",
            }

    if not complete_by_source["gfs"]:
        raise ValueError("no complete 100-member GFS issue; Current is unchanged")
    current_issue = max(complete_by_source["gfs"], key=lambda item: item["id"])
    current_issue["presentation"] = {
        "section": "current",
        "label": "Current guidance",
        "member_class": "production",
    }
    reference_issue = (
        max(complete_by_source["era5"], key=lambda item: item["id"])
        if complete_by_source["era5"]
        else None
    )

    current = _pointer(current_issue, "gfs")
    reference = _pointer(reference_issue, "era5") if reference_issue else None
    current_schedule_target = parse_datetime(
        str(current_issue["initialization"])
    ).replace(hour=TARGET_HOUR_UTC)
    next_expected = next_expected_update(current_schedule_target)
    stale_after = next_expected + timedelta(hours=12)
    last_successful_at = str(current_issue["published_at"])

    payload.update(
        {
            "schema_version": 3,
            "generated_at": iso_z(now),
            "latest_successful_issue": current_issue["id"],
            "current": current,
            "latest_reference": reference,
            "publication": {
                "cadence_days_utc": ["Wednesday", "Saturday"],
                "target_time_utc": "12:00",
                "last_successful_at": last_successful_at,
                "next_expected_at": iso_z(next_expected),
            },
            "retention": {
                "interactive_days": 365,
                "pdf_days": 56,
                "metadata": "indefinite",
            },
            "operations": {
                "status": "on_schedule" if now <= stale_after else "stale",
                "last_successful_at": last_successful_at,
                "next_expected_at": iso_z(next_expected),
                "stale_after": iso_z(stale_after),
            },
            "cache": {
                "catalog": "no-store",
                "assets": "checksum-versioned",
            },
        }
    )
    if payload.get("global_viewer"):
        payload["global_viewer"].update(
            {
                "status": "dated_demo",
                "label": "Dated global demo — 28 July 2026",
                "automated": False,
            }
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public/data/index.json",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public",
    )
    parser.add_argument(
        "--now",
        help="UTC ISO timestamp used for deterministic rehearsals and tests.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate derivation without modifying the catalog.",
    )
    args = parser.parse_args()
    now = parse_datetime(args.now) if args.now else datetime.now(UTC)
    existing = _load_json(args.index)
    payload = build_catalog(existing, args.public_root, now)
    if not args.check:
        write_json(args.index, payload)
        print(f"wrote schema-v3 catalog: {args.index}")
    else:
        print(
            "catalog valid: "
            f"current={payload['current']['issue_id']} "
            f"reference={payload.get('latest_reference', {}).get('issue_id', 'none')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
