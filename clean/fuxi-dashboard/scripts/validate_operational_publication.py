#!/usr/bin/env python3
"""Validate an operational web publication without reading private model data.

The validator deliberately starts at the public catalog.  It checks catalog
topology, the current 100-member GFS issue, the delayed ERA5 reference, dynamic
product/capability declarations, every referenced archive asset, and the
public SHA-256 inventory.  A single issue can be selected for a fast staging
gate; ``--scope all`` is intended for CI immediately before deployment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ISSUE_ID = re.compile(r"^\d{8}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RAW_PRODUCTS = {"rainfall_total", "temperature_mean"}
ANOMALY_PRODUCTS = {"rainfall_anomaly", "temperature_anomaly"}
PRIVATE_TOKENS = ("/home/", "/storage/", "private key", "password", "secret")


@dataclass(frozen=True)
class IssueRef:
    source_id: str
    record: dict[str, Any]

    @property
    def issue_id(self) -> str:
        return str(self.record["id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-root", type=Path, default=ROOT / "public", help="Static public root."
    )
    parser.add_argument(
        "--scope",
        choices=("current", "all", "issue"),
        default="current",
        help="Validate current pointers, the complete archive, or one issue.",
    )
    parser.add_argument("--source", choices=("gfs", "era5"))
    parser.add_argument("--issue", help="YYYYMMDD issue id for --scope issue.")
    parser.add_argument("--expected-gfs-issue")
    parser.add_argument("--expected-era5-issue")
    parser.add_argument(
        "--expected-forecast-sha256",
        help="Expected selected forecast checksum (only valid for --scope issue).",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON receipt.")
    parser.add_argument(
        "--live-inventory",
        action="store_true",
        help="Hash the public tree in memory instead of requiring a pre-stamped manifest.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_path(public_root: Path, relative: str) -> Path:
    """Resolve a catalog path and prevent absolute or parent traversal."""

    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe public path: {relative!r}")
    root = public_root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"public path escapes root: {relative!r}")
    return resolved


def manifest_inventory(public_root: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(public_root / "data/manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported public manifest schema")
    inventory: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("files", []):
        relative = str(entry.get("path", ""))
        if relative in inventory:
            raise ValueError(f"duplicate manifest path: {relative}")
        if not SHA256.fullmatch(str(entry.get("sha256", ""))):
            raise ValueError(f"invalid manifest checksum: {relative}")
        inventory[relative] = entry
    return inventory


def live_inventory(public_root: Path) -> dict[str, dict[str, Any]]:
    """Build a non-mutating checksum view for pre-publication rehearsals."""

    inventory: dict[str, dict[str, Any]] = {}
    manifest_path = (public_root / "data/manifest.json").resolve()
    for path in sorted(public_root.rglob("*")):
        if not path.is_file() or path.resolve() == manifest_path:
            continue
        relative = path.relative_to(public_root).as_posix()
        inventory[relative] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return inventory


def check_inventory_file(
    public_root: Path,
    inventory: dict[str, dict[str, Any]],
    relative: str,
    checked: dict[str, str],
) -> Path:
    path = public_path(public_root, relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    entry = inventory.get(relative)
    if not entry:
        raise ValueError(f"asset is absent from data/manifest.json: {relative}")
    actual_size = path.stat().st_size
    if entry.get("size_bytes") != actual_size:
        raise ValueError(f"manifest size mismatch: {relative}")
    actual_sha = checked.setdefault(relative, sha256(path))
    if entry["sha256"] != actual_sha:
        raise ValueError(f"manifest checksum mismatch: {relative}")
    return path


def source_map(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = index.get("initial_condition_sources")
    if not isinstance(sources, list):
        raise ValueError("catalog has no initial_condition_sources list")
    result: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_id = source.get("id")
        if source_id in result:
            raise ValueError(f"duplicate source id: {source_id}")
        result[str(source_id)] = source
    if set(result) != {"gfs", "era5"}:
        raise ValueError("catalog must expose exactly GFS and ERA5 sources")
    return result


def issue_records(sources: dict[str, dict[str, Any]]) -> dict[tuple[str, str], IssueRef]:
    records: dict[tuple[str, str], IssueRef] = {}
    for source_id, source in sources.items():
        issues = source.get("issues")
        if not isinstance(issues, list) or not issues:
            raise ValueError(f"{source_id} has no issue catalog")
        previous = "99999999"
        for record in issues:
            issue_id = str(record.get("id", ""))
            if not ISSUE_ID.fullmatch(issue_id):
                raise ValueError(f"invalid {source_id} issue id: {issue_id!r}")
            if issue_id > previous:
                raise ValueError(f"{source_id} issues are not newest-first")
            previous = issue_id
            key = (source_id, issue_id)
            if key in records:
                raise ValueError(f"duplicate issue: {source_id}/{issue_id}")
            initialization = str(record.get("initialization", ""))
            if initialization[:10].replace("-", "") != issue_id:
                raise ValueError(f"initialization/id mismatch: {source_id}/{issue_id}")
            members = record.get("members")
            if not isinstance(members, int) or members <= 0:
                raise ValueError(f"invalid member count: {source_id}/{issue_id}")
            forecast = str(record.get("forecast", ""))
            expected = f"forecasts/{source_id}/{issue_id}.json"
            if forecast != expected:
                raise ValueError(
                    f"forecast path for {source_id}/{issue_id} must be {expected}"
                )
            records[key] = IssueRef(source_id, record)
    return records


def pointer_id(index: dict[str, Any], source: dict[str, Any], source_id: str) -> str:
    """Read schema-v2 defaults or schema-v3 explicit current/reference pointers."""

    keys = (
        ("current", "current_gfs", "current_issue")
        if source_id == "gfs"
        else ("latest_reference", "latest_era5", "reference_issue")
    )
    for key in keys:
        pointer = index.get(key)
        if isinstance(pointer, str):
            return pointer
        if isinstance(pointer, dict):
            if pointer.get("source_id") not in (None, source_id):
                continue
            for id_key in ("id", "issue", "issue_id"):
                if pointer.get(id_key):
                    return str(pointer[id_key])
    return str(source.get("default_issue", ""))


def advertised_products(record: dict[str, Any], forecast: dict[str, Any]) -> set[str]:
    advertised = record.get("available_products")
    if advertised is None:
        presentation = record.get("presentation")
        if isinstance(presentation, dict):
            advertised = presentation.get("available_products")
    if advertised is None:
        advertised = list(forecast.get("products", {}))
    if not isinstance(advertised, list) or not all(
        isinstance(item, str) for item in advertised
    ):
        raise ValueError("available_products must be a string list")
    return set(advertised)


def capability(record: dict[str, Any], name: str, default: bool) -> bool:
    capabilities = record.get("capabilities")
    if isinstance(capabilities, list):
        return name in capabilities
    if isinstance(capabilities, dict):
        value = capabilities.get(name)
        if isinstance(value, dict):
            return bool(value.get("available", value.get("enabled", default)))
        if value is not None:
            return bool(value)
    return default


def issue_asset_paths(ref: IssueRef, forecast: dict[str, Any]) -> list[str]:
    record = ref.record
    assets = [f"data/{record['forecast']}"]
    regional = record.get("regional_outlook")
    if regional:
        assets.append(f"data/{regional}")
    issue = forecast.get("issue", {})
    downloads = issue.get("downloads", {})
    if isinstance(downloads, dict):
        pdf = downloads.get("india_pdf")
        if pdf:
            assets.append(str(pdf))
    comparison = issue.get("initialization_comparison")
    if isinstance(comparison, dict) and comparison.get("comparison"):
        assets.append(f"data/{comparison['comparison']}")
    archive = record.get("archive")
    if isinstance(archive, dict):
        for key in ("assets", "download_paths"):
            values = archive.get(key)
            if isinstance(values, list):
                assets.extend(str(value) for value in values)
    return list(dict.fromkeys(assets))


def declared_checksum(record: dict[str, Any], relative: str) -> str | None:
    checksums = record.get("checksums")
    if isinstance(checksums, dict):
        value = checksums.get(relative)
        if isinstance(value, str):
            return value
        basename = Path(relative).name
        value = checksums.get(basename)
        if isinstance(value, str):
            return value
        aliases = {
            "forecast": relative.startswith("data/forecasts/"),
            "regional_outlook": relative.startswith("data/regional/"),
            "pdf": relative.endswith(".pdf"),
        }
        for alias, matches in aliases.items():
            if matches:
                keys = [alias, f"{alias}_sha256"]
                if alias == "regional_outlook":
                    keys.append("regional_sha256")
                for key in keys:
                    value = checksums.get(key)
                    if isinstance(value, str):
                        return value
    return None


def archive_available(record: dict[str, Any], kind: str) -> bool:
    archive = record.get("archive")
    if not isinstance(archive, dict):
        return True
    return bool(archive.get(f"{kind}_available", True))


def grid_size(payload: dict[str, Any], label: str) -> int:
    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError(f"{label} has no grid definition")
    latitude = grid.get("latitude")
    longitude = grid.get("longitude")
    if not isinstance(latitude, list) or not isinstance(longitude, list):
        raise ValueError(f"{label} has invalid grid coordinates")
    expected_shape = [len(latitude), len(longitude)]
    if min(expected_shape) < 1 or grid.get("shape") != expected_shape:
        raise ValueError(f"{label} grid shape does not match its coordinates")
    return expected_shape[0] * expected_shape[1]


def validate_regional_payload(path: Path, ref: IssueRef) -> None:
    regional = load_json(path)
    issue = regional.get("issue", {})
    if (
        issue.get("source_id") != ref.source_id
        or str(issue.get("initialization", ""))[:10].replace("-", "")
        != ref.issue_id
        or issue.get("members") != 100
    ):
        raise ValueError(f"regional identity mismatch: {ref.source_id}/{ref.issue_id}")
    expected_values = grid_size(regional, "regional outlook")
    weeks = regional.get("weeks")
    if not isinstance(weeks, list) or len(weeks) != 6:
        raise ValueError(f"regional outlook must contain six weeks: {ref.issue_id}")
    for expected_week, week in enumerate(weeks, start=1):
        if week.get("week") != expected_week:
            raise ValueError(f"regional week order mismatch: {ref.issue_id}")
        fields = week.get("probability_fields", {})
        for variable in ("rainfall", "temperature"):
            categories = fields.get(variable, {})
            arrays = []
            for category in ("below_normal", "near_normal", "above_normal"):
                values = categories.get(category)
                if not isinstance(values, list) or len(values) != expected_values:
                    raise ValueError(
                        f"regional probability field has wrong size: "
                        f"{ref.source_id}/{ref.issue_id}/{variable}/{category}"
                    )
                if any(
                    not isinstance(value, (int, float)) or value < 0 or value > 100
                    for value in values
                ):
                    raise ValueError(
                        f"regional probability outside 0..100: "
                        f"{ref.source_id}/{ref.issue_id}/{variable}/{category}"
                    )
                arrays.append(values)
            if any(
                abs(sum(values) - 100.0) > 0.2 for values in zip(*arrays, strict=True)
            ):
                raise ValueError(
                    f"regional probabilities do not sum to 100: "
                    f"{ref.source_id}/{ref.issue_id}/{variable}"
                )


def validate_forecast(
    public_root: Path,
    inventory: dict[str, dict[str, Any]],
    checked: dict[str, str],
    ref: IssueRef,
) -> dict[str, Any]:
    relative = f"data/{ref.record['forecast']}"
    forecast_path = check_inventory_file(public_root, inventory, relative, checked)
    forecast = load_json(forecast_path)
    issue = forecast.get("issue", {})
    source = issue.get("initial_condition_source", {})
    if source.get("id") != ref.source_id:
        raise ValueError(f"forecast source mismatch: {ref.source_id}/{ref.issue_id}")
    if issue.get("members") != ref.record.get("members"):
        raise ValueError(f"forecast member mismatch: {ref.source_id}/{ref.issue_id}")
    if str(issue.get("initialization", ""))[:10].replace("-", "") != ref.issue_id:
        raise ValueError(f"forecast initialization mismatch: {ref.source_id}/{ref.issue_id}")

    products = forecast.get("products")
    if not isinstance(products, dict) or not RAW_PRODUCTS.issubset(products):
        raise ValueError(f"raw rainfall/temperature missing: {ref.source_id}/{ref.issue_id}")
    available = advertised_products(ref.record, forecast)
    if available != set(products):
        raise ValueError(
            f"available_products differs from forecast products: "
            f"{ref.source_id}/{ref.issue_id}"
        )
    anomaly_available = capability(
        ref.record, "anomalies", bool(ANOMALY_PRODUCTS & set(products))
    )
    if anomaly_available != ANOMALY_PRODUCTS.issubset(products):
        raise ValueError(f"anomaly capability/product mismatch: {ref.source_id}/{ref.issue_id}")

    expected_values = grid_size(forecast, "forecast")
    mask = forecast["grid"].get("india_mask")
    if not isinstance(mask, list) or len(mask) != expected_values or any(
        value not in (0, 1, False, True) for value in mask
    ):
        raise ValueError(f"forecast support mask is invalid: {ref.source_id}/{ref.issue_id}")

    weeks = forecast.get("weeks")
    if not isinstance(weeks, list) or len(weeks) != 6:
        raise ValueError(f"forecast must contain six weeks: {ref.source_id}/{ref.issue_id}")
    for expected_week, week in enumerate(weeks, start=1):
        if week.get("week") != expected_week:
            raise ValueError(f"forecast week order mismatch: {ref.source_id}/{ref.issue_id}")
        if set(week.get("fields", {})) != set(products):
            raise ValueError(f"week fields differ from products: {ref.source_id}/{ref.issue_id}")
        if set(week.get("summary", {})) != set(products):
            raise ValueError(f"week summaries differ from products: {ref.source_id}/{ref.issue_id}")
        for product, values in week["fields"].items():
            if not isinstance(values, list) or len(values) != expected_values:
                raise ValueError(
                    f"forecast field has wrong size: "
                    f"{ref.source_id}/{ref.issue_id}/{expected_week}/{product}"
                )
            if any(not isinstance(value, (int, float)) for value in values):
                raise ValueError(
                    f"forecast field is not numeric: "
                    f"{ref.source_id}/{ref.issue_id}/{expected_week}/{product}"
                )

    regional = ref.record.get("regional_outlook")
    regional_capability = capability(ref.record, "regional_probabilities", bool(regional))
    if regional_capability != bool(regional):
        raise ValueError(f"regional capability/path mismatch: {ref.source_id}/{ref.issue_id}")

    assets = issue_asset_paths(ref, forecast)
    if not archive_available(ref.record, "pdf"):
        assets = [asset for asset in assets if not asset.endswith(".pdf")]
    for asset in assets:
        asset_path = check_inventory_file(public_root, inventory, asset, checked)
        declared = declared_checksum(ref.record, asset)
        if declared is not None:
            if not SHA256.fullmatch(declared) or declared != checked[asset]:
                raise ValueError(f"declared issue checksum mismatch: {asset}")
        if regional and asset == f"data/{regional}":
            validate_regional_payload(asset_path, ref)
        if asset.startswith("data/comparisons/"):
            comparison = load_json(asset_path)
            if (
                str(comparison.get("issue_date", ""))[:10].replace("-", "")
                != ref.issue_id
                or set(comparison.get("sources", {})) != {"gfs", "era5"}
            ):
                raise ValueError(f"comparison identity mismatch: {asset}")

    downloads = issue.get("downloads", {})
    pdf_available = bool(
        archive_available(ref.record, "pdf")
        and isinstance(downloads, dict)
        and downloads.get("india_pdf")
    )
    if capability(ref.record, "pdf", pdf_available) != pdf_available:
        raise ValueError(f"PDF capability/asset mismatch: {ref.source_id}/{ref.issue_id}")
    if pdf_available:
        pdf = str(downloads["india_pdf"])
        pdf_sha = downloads.get("india_pdf_sha256")
        if pdf_sha != checked.get(pdf):
            raise ValueError(f"forecast PDF checksum mismatch: {ref.source_id}/{ref.issue_id}")

    serialized = json.dumps(forecast).lower()
    if any(token in serialized for token in PRIVATE_TOKENS):
        raise ValueError(f"private path or secret leaked: {ref.source_id}/{ref.issue_id}")
    return forecast


def choose_scope(
    args: argparse.Namespace,
    index: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    records: dict[tuple[str, str], IssueRef],
) -> tuple[list[IssueRef], str, str]:
    gfs_id = pointer_id(index, sources["gfs"], "gfs")
    era5_id = pointer_id(index, sources["era5"], "era5")
    if ("gfs", gfs_id) not in records:
        raise ValueError(f"current GFS pointer is not cataloged: {gfs_id}")
    if ("era5", era5_id) not in records:
        raise ValueError(f"latest ERA5 pointer is not cataloged: {era5_id}")
    if records[("gfs", gfs_id)].record.get("members") != 100:
        raise ValueError("current GFS issue must be a 100-member publication")
    if records[("era5", era5_id)].record.get("members") != 100:
        raise ValueError("latest ERA5 reference must be a 100-member publication")
    for label, issue_id in (("current GFS", gfs_id), ("latest ERA5", era5_id)):
        if dt.datetime.strptime(issue_id, "%Y%m%d").weekday() not in {2, 5}:
            raise ValueError(f"{label} pointer is not a Wednesday/Saturday issue")
    if args.expected_gfs_issue and args.expected_gfs_issue != gfs_id:
        raise ValueError(
            f"expected GFS {args.expected_gfs_issue}, catalog points to {gfs_id}"
        )
    if args.expected_era5_issue and args.expected_era5_issue != era5_id:
        raise ValueError(
            f"expected ERA5 {args.expected_era5_issue}, catalog points to {era5_id}"
        )
    if args.scope == "all":
        selected = [
            ref for ref in records.values() if archive_available(ref.record, "interactive")
        ]
    elif args.scope == "current":
        selected = [records[("gfs", gfs_id)], records[("era5", era5_id)]]
    else:
        if not args.source or not args.issue:
            raise ValueError("--scope issue requires --source and --issue")
        key = (args.source, args.issue)
        if key not in records:
            raise ValueError(f"issue is not cataloged: {args.source}/{args.issue}")
        selected = [records[key]]
    return selected, gfs_id, era5_id


def validate_v3_contract(
    index: dict[str, Any],
    records: dict[tuple[str, str], IssueRef],
    gfs_id: str,
    era5_id: str,
) -> None:
    if index.get("schema_version", 0) < 3:
        return
    retention = index.get("retention", {})
    if retention != {
        "interactive_days": 365,
        "pdf_days": 56,
        "metadata": "indefinite",
    }:
        raise ValueError("schema-v3 retention policy must be 365/56 days and indefinite metadata")
    cache = index.get("cache", {})
    if cache.get("catalog") != "no-store" or cache.get("assets") != "checksum-versioned":
        raise ValueError("schema-v3 cache policy is incomplete")
    publication = index.get("publication")
    operations = index.get("operations")
    if not isinstance(publication, dict) or not isinstance(operations, dict):
        raise ValueError("schema-v3 publication and operations metadata are required")
    cadence = " ".join(str(value).lower() for value in publication.get("cadence_days_utc", []))
    if "wed" not in cadence or "sat" not in cadence:
        raise ValueError("publication cadence must name Wednesday and Saturday UTC")
    if not publication.get("target_time_utc"):
        raise ValueError("schema-v3 publication target_time_utc is required")
    for field in ("last_successful_at", "next_expected_at"):
        if not publication.get(field) or not operations.get(field):
            raise ValueError(f"schema-v3 operational timestamp missing: {field}")
    for field in ("status", "stale_after"):
        if not operations.get(field):
            raise ValueError(f"schema-v3 operations field missing: {field}")

    for key, source_id, issue_id in (
        ("current", "gfs", gfs_id),
        ("latest_reference", "era5", era5_id),
    ):
        pointer = index.get(key)
        if not isinstance(pointer, dict):
            raise ValueError(f"schema-v3 pointer must be an object: {key}")
        ref = records[(source_id, issue_id)]
        required = {
            "source_id",
            "issue_id",
            "initialization",
            "members",
            "forecast",
            "forecast_sha256",
            "valid_through",
            "published_at",
        }
        if not required.issubset(pointer):
            raise ValueError(f"schema-v3 pointer is incomplete: {key}")
        if pointer["source_id"] != source_id or pointer["issue_id"] != issue_id:
            raise ValueError(f"schema-v3 pointer identity mismatch: {key}")
        for field in ("initialization", "members", "forecast", "valid_through", "published_at"):
            if pointer[field] != ref.record.get(field):
                raise ValueError(f"schema-v3 pointer/record mismatch: {key}.{field}")
        checksum = ref.record.get("checksums", {}).get("forecast_sha256")
        if pointer["forecast_sha256"] != checksum:
            raise ValueError(f"schema-v3 pointer checksum mismatch: {key}")

    for ref in records.values():
        record = ref.record
        for field in (
            "available_products",
            "capabilities",
            "presentation",
            "checksums",
            "valid_through",
            "published_at",
            "archive",
        ):
            if field not in record:
                raise ValueError(f"schema-v3 issue field missing: {ref.source_id}/{ref.issue_id}.{field}")
        capabilities = record["capabilities"]
        if not isinstance(capabilities, dict) or not capabilities.get("raw_fields"):
            raise ValueError(f"raw-field capability missing: {ref.source_id}/{ref.issue_id}")
        presentation = record["presentation"]
        if not isinstance(presentation, dict) or not {
            "section",
            "label",
            "member_class",
        }.issubset(presentation):
            raise ValueError(f"presentation metadata incomplete: {ref.source_id}/{ref.issue_id}")
        checksums = record["checksums"]
        if not isinstance(checksums, dict) or not SHA256.fullmatch(
            str(checksums.get("forecast_sha256", ""))
        ):
            raise ValueError(f"forecast checksum missing: {ref.source_id}/{ref.issue_id}")
        archive = record["archive"]
        if not isinstance(archive, dict) or archive.get("metadata") != "indefinite":
            raise ValueError(f"archive metadata policy missing: {ref.source_id}/{ref.issue_id}")
        for field in ("interactive_available", "pdf_available"):
            if not isinstance(archive.get(field), bool):
                raise ValueError(f"archive availability missing: {ref.source_id}/{ref.issue_id}.{field}")
        if archive["interactive_available"] and not archive.get("interactive_until"):
            raise ValueError(f"interactive retention date missing: {ref.source_id}/{ref.issue_id}")
        if archive["pdf_available"] and not archive.get("pdf_until"):
            raise ValueError(f"PDF retention date missing: {ref.source_id}/{ref.issue_id}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    public_root = args.public_root.resolve()
    index_path = public_root / "data/index.json"
    index = load_json(index_path)
    if index.get("default_source") != "gfs":
        raise ValueError("default source must remain GFS")
    if not isinstance(index.get("schema_version"), int):
        raise ValueError("catalog has no integer schema_version")
    serialized_index = json.dumps(index).lower()
    if any(token in serialized_index for token in PRIVATE_TOKENS):
        raise ValueError("catalog contains a private path or secret")

    inventory = (
        live_inventory(public_root) if args.live_inventory else manifest_inventory(public_root)
    )
    checked: dict[str, str] = {}
    check_inventory_file(public_root, inventory, "data/index.json", checked)
    sources = source_map(index)
    records = issue_records(sources)
    selected, gfs_id, era5_id = choose_scope(args, index, sources, records)
    validate_v3_contract(index, records, gfs_id, era5_id)

    selected_forecasts: dict[str, dict[str, Any]] = {}
    for ref in selected:
        key = f"{ref.source_id}/{ref.issue_id}"
        selected_forecasts[key] = validate_forecast(
            public_root, inventory, checked, ref
        )

    # Even in the fast current/issue scopes, no catalog record may point at a
    # missing archive forecast or regional package.  Full checksums run in CI.
    for ref in records.values():
        if not archive_available(ref.record, "interactive"):
            continue
        forecast_path = public_path(public_root, f"data/{ref.record['forecast']}")
        if not forecast_path.is_file():
            raise FileNotFoundError(forecast_path)
        regional = ref.record.get("regional_outlook")
        if regional and not public_path(public_root, f"data/{regional}").is_file():
            raise FileNotFoundError(public_path(public_root, f"data/{regional}"))

    if args.expected_forecast_sha256:
        if args.scope != "issue" or len(selected) != 1:
            raise ValueError("--expected-forecast-sha256 requires --scope issue")
        expected = args.expected_forecast_sha256.lower()
        if not SHA256.fullmatch(expected):
            raise ValueError("expected forecast checksum is not a SHA-256 digest")
        relative = f"data/{selected[0].record['forecast']}"
        if checked[relative] != expected:
            raise ValueError(
                f"expected forecast checksum {expected}, found {checked[relative]}"
            )

    return {
        "status": "validated",
        "catalog_schema": index["schema_version"],
        "current_gfs_issue": gfs_id,
        "latest_era5_issue": era5_id,
        "scope": args.scope,
        "issues_validated": sorted(selected_forecasts),
        "assets_hashed": len(checked),
        "catalog_issues": len(records),
    }


def main() -> int:
    args = parse_args()
    try:
        receipt = validate(args)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"operational publication validation failed: {error}") from error
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    else:
        print(
            "operational publication: validated "
            f"{len(receipt['issues_validated'])} issue(s), "
            f"GFS {receipt['current_gfs_issue']}, ERA5 {receipt['latest_era5_issue']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
