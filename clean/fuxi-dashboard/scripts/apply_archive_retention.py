#!/usr/bin/env python3
"""Apply the public 12-month map / 8-week PDF retention policy safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
UTC = timezone.utc


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)


def safe_asset(public_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe public retention path: {relative}")
    resolved = (public_root / candidate).resolve()
    root = public_root.resolve()
    if root not in resolved.parents:
        raise ValueError(f"retention path escapes public root: {relative}")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".retention-part")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remove_file(path: Path, removed: list[str], public_root: Path) -> None:
    if not path.is_file():
        return
    path.unlink()
    removed.append(path.relative_to(public_root).as_posix())
    parent = path.parent
    while parent != public_root and parent != public_root / "data":
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def apply_retention(
    public_root: Path,
    index_path: Path,
    now: datetime,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    protected = {
        (pointer.get("source_id"), pointer.get("issue_id"))
        for pointer in (index.get("current"), index.get("latest_reference"))
        if isinstance(pointer, dict)
    }
    removed: list[str] = []
    updated: list[str] = []
    for source in index.get("initial_condition_sources", []):
        source_id = str(source["id"])
        for record in source.get("issues", []):
            issue_id = str(record["id"])
            initialization = parse_datetime(str(record["initialization"]))
            pdf_expired = now > initialization + timedelta(days=56)
            interactive_expired = now > initialization + timedelta(days=365)
            if (source_id, issue_id) in protected:
                continue
            forecast_path = safe_asset(
                public_root, f"data/{record['forecast']}"
            )
            forecast = (
                json.loads(forecast_path.read_text(encoding="utf-8"))
                if forecast_path.is_file()
                else None
            )
            changed = False
            if pdf_expired:
                pdf_relative = record.get("pdf")
                if not pdf_relative and forecast:
                    pdf_relative = forecast.get("issue", {}).get("downloads", {}).get(
                        "india_pdf"
                    )
                if pdf_relative and not dry_run:
                    remove_file(
                        safe_asset(public_root, str(pdf_relative)),
                        removed,
                        public_root,
                    )
                if forecast:
                    downloads = forecast.get("issue", {}).get("downloads", {})
                    for key in ("india_pdf", "india_pdf_sha256"):
                        downloads.pop(key, None)
                    if not dry_run:
                        write_json(forecast_path, forecast)
                        record.setdefault("checksums", {})[
                            "forecast_sha256"
                        ] = sha256(forecast_path)
                record.pop("pdf", None)
                record.get("checksums", {}).pop("pdf_sha256", None)
                record.setdefault("capabilities", {})["pdf"] = False
                record.setdefault("archive", {})["pdf_available"] = False
                changed = True
            if interactive_expired:
                if not dry_run:
                    remove_file(forecast_path, removed, public_root)
                    regional = record.get("regional_outlook")
                    if regional:
                        remove_file(
                            safe_asset(public_root, f"data/{regional}"),
                            removed,
                            public_root,
                        )
                record.setdefault("archive", {})["interactive_available"] = False
                record.setdefault("capabilities", {})[
                    "regional_probabilities"
                ] = False
                changed = True
            if changed:
                updated.append(f"{source_id}/{issue_id}")
    if updated and not dry_run:
        write_json(index_path, index)
    return {
        "status": "dry-run" if dry_run else "applied",
        "updated_issues": updated,
        "removed_files": removed,
        "interactive_days": 365,
        "pdf_days": 56,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=PUBLIC)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--now", help="ISO UTC clock for rehearsals/tests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    index = args.index or args.public_root / "data/index.json"
    now = parse_datetime(args.now) if args.now else datetime.now(UTC)
    result = apply_retention(
        args.public_root.resolve(), index.resolve(), now, dry_run=args.dry_run
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
