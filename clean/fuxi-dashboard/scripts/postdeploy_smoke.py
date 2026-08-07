#!/usr/bin/env python3
"""Smoke-test a deployed static dashboard against expected issue metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument(
        "--expected-index",
        type=Path,
        help="Local catalog whose current/reference pointers must match deployment.",
    )
    parser.add_argument("--expected-gfs-issue")
    parser.add_argument("--expected-era5-issue")
    parser.add_argument("--expected-source", choices=("gfs", "era5"))
    parser.add_argument("--expected-issue")
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def decode_json(payload: bytes, label: str) -> Any:
    return json.loads(
        payload.decode("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value!r} in {label}")
        ),
    )


def pointer_id(index: dict[str, Any], source_id: str) -> str:
    pointer_keys = (
        ("current", "current_gfs", "current_issue")
        if source_id == "gfs"
        else ("latest_reference", "latest_era5", "reference_issue")
    )
    for key in pointer_keys:
        pointer = index.get(key)
        if isinstance(pointer, str):
            return pointer
        if isinstance(pointer, dict):
            if pointer.get("source_id") not in (None, source_id):
                continue
            for id_key in ("issue_id", "id", "issue"):
                if pointer.get(id_key):
                    return str(pointer[id_key])
    for source in index.get("initial_condition_sources", []):
        if source.get("id") == source_id:
            return str(source.get("default_issue", ""))
    return ""


def fetch(base_url: str, relative: str, attempts: int, timeout: float) -> bytes:
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    base = base_url.rstrip("/") + "/"
    base_parts = urllib.parse.urlsplit(base)
    if base_parts.scheme not in {"http", "https"} or not base_parts.netloc:
        raise ValueError("deployment base URL must be HTTP(S)")
    url = urllib.parse.urljoin(base, relative)
    if urllib.parse.urlsplit(url).netloc != base_parts.netloc:
        raise ValueError(f"smoke-test path escaped deployment origin: {relative}")
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json,text/html;q=0.8,*/*;q=0.2",
                "Cache-Control": "no-cache",
                "User-Agent": "s2s-postdeploy-smoke/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise ValueError(f"HTTP {status} for {relative}")
                return response.read()
        except (OSError, urllib.error.URLError, ValueError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    raise ValueError(f"unable to fetch {relative} after {attempts} attempts: {last_error}")


def expected_issues(args: argparse.Namespace) -> tuple[str | None, str | None]:
    gfs = args.expected_gfs_issue
    era5 = args.expected_era5_issue
    if args.expected_index:
        local = decode_json(args.expected_index.read_bytes(), str(args.expected_index))
        gfs = gfs or pointer_id(local, "gfs")
        era5 = era5 or pointer_id(local, "era5")
    return gfs, era5


def smoke_once(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch one internally consistent deployment snapshot."""

    html = fetch(args.base_url, "index.html", 1, args.timeout)
    if b'id="app"' not in html and b"id='app'" not in html:
        raise ValueError("deployed index.html has no application root")
    index_bytes = fetch(args.base_url, "data/index.json", 1, args.timeout)
    index = decode_json(index_bytes, "deployed data/index.json")
    manifest = decode_json(
        fetch(args.base_url, "data/manifest.json", 1, args.timeout),
        "deployed data/manifest.json",
    )
    if args.expected_commit and manifest.get("deployed_commit") != args.expected_commit:
        raise ValueError(
            f"deployed commit is {manifest.get('deployed_commit')}, "
            f"expected {args.expected_commit}"
        )
    inventory = {entry["path"]: entry for entry in manifest.get("files", [])}
    index_entry = inventory.get("data/index.json")
    if not index_entry or index_entry.get("sha256") != sha256_bytes(index_bytes):
        raise ValueError("deployed catalog does not match its manifest checksum")

    actual_gfs = pointer_id(index, "gfs")
    actual_era5 = pointer_id(index, "era5")
    expected_gfs, expected_era5 = expected_issues(args)
    if expected_gfs and actual_gfs != expected_gfs:
        raise ValueError(f"deployed GFS issue is {actual_gfs}, expected {expected_gfs}")
    if expected_era5 and actual_era5 != expected_era5:
        raise ValueError(
            f"deployed ERA5 issue is {actual_era5}, expected {expected_era5}"
        )
    expected_source = getattr(args, "expected_source", None)
    expected_issue = getattr(args, "expected_issue", None)
    if bool(expected_source) != bool(expected_issue):
        raise ValueError("--expected-source and --expected-issue must be used together")
    if expected_source:
        actual_selected = actual_gfs if expected_source == "gfs" else actual_era5
        if actual_selected != expected_issue:
            raise ValueError(
                f"deployed {expected_source.upper()} issue is {actual_selected}, "
                f"expected {expected_issue}"
            )

    sources = {
        source["id"]: source for source in index.get("initial_condition_sources", [])
    }
    verified_assets: list[str] = []
    for source_id, issue_id in (("gfs", actual_gfs), ("era5", actual_era5)):
        source = sources.get(source_id, {})
        record = next(
            (item for item in source.get("issues", []) if item.get("id") == issue_id),
            None,
        )
        if not record:
            raise ValueError(f"deployed pointer is not cataloged: {source_id}/{issue_id}")
        if record.get("members") != 100:
            raise ValueError(f"deployed pointer is not 100-member: {source_id}/{issue_id}")
        relative = f"data/{record['forecast']}"
        forecast_bytes = fetch(args.base_url, relative, 1, args.timeout)
        entry = inventory.get(relative)
        if not entry or entry.get("sha256") != sha256_bytes(forecast_bytes):
            raise ValueError(f"deployed forecast checksum mismatch: {relative}")
        forecast = decode_json(forecast_bytes, relative)
        issue = forecast.get("issue", {})
        if (
            issue.get("initial_condition_source", {}).get("id") != source_id
            or str(issue.get("initialization", ""))[:10].replace("-", "")
            != issue_id
            or issue.get("members") != 100
        ):
            raise ValueError(f"deployed forecast metadata mismatch: {relative}")
        verified_assets.append(relative)

    return {
        "status": "healthy",
        "deployed_commit": manifest.get("deployed_commit"),
        "current_gfs_issue": actual_gfs,
        "latest_era5_issue": actual_era5,
        "verified_assets": verified_assets,
    }


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Retry the whole consistency check while a Pages deployment converges."""

    if args.attempts < 1:
        raise ValueError("attempts must be at least one")
    last_error: Exception | None = None
    for attempt in range(args.attempts):
        try:
            return smoke_once(args)
        except (KeyError, OSError, TypeError, ValueError) as error:
            last_error = error
            if attempt + 1 < args.attempts:
                time.sleep(min(2**attempt, 8))
    raise ValueError(
        f"deployment did not become internally consistent after "
        f"{args.attempts} attempts: {last_error}"
    )


def main() -> int:
    args = parse_args()
    try:
        receipt = smoke(args)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"post-deploy smoke test failed: {error}") from error
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    else:
        print(
            "post-deploy smoke: healthy; "
            f"GFS {receipt['current_gfs_issue']}, "
            f"ERA5 {receipt['latest_era5_issue']}, "
            f"commit {receipt['deployed_commit']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
