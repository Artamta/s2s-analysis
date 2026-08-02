#!/usr/bin/env python3
"""Stamp the deployed commit and refresh the public-data checksum inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "public"
PUBLIC_DATA = PUBLIC_ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(commit: str = "local-uncommitted") -> Path:
    """Write the complete, non-self-referential public inventory."""

    files = []
    for path in sorted(PUBLIC_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path == PUBLIC_DATA / "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(PUBLIC_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deployed_commit": commit,
        "inventory_scope": (
            "Every deployed public file except data/manifest.json itself, "
            "whose self-checksum would be recursive."
        ),
        "files": files,
    }
    output = PUBLIC_DATA / "manifest.json"
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output} with {len(files)} checksums")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="local-uncommitted")
    args = parser.parse_args()
    build_manifest(args.commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
