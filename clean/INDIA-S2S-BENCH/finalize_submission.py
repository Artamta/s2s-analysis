#!/usr/bin/env python3
"""Hash manuscript sources and the compiled PDF after a successful audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    audit = json.loads((ROOT / "artifacts/confirmatory_2025/audit_report.json").read_text())
    if audit["status"] != "passed":
        raise RuntimeError("refusing to finalize a manuscript with a failed artifact audit")
    required = [
        ROOT / "protocol.json",
        ROOT / "Reserach.md",
        ROOT / "paper/main.tex",
        ROOT / "paper/appendix.tex",
        ROOT / "paper/references.bib",
        ROOT / "paper/generated_numbers.tex",
        ROOT / "paper/main.pdf",
        ROOT / "paper/submission_draft.pdf",
        ROOT / "artifacts/confirmatory_2025/manifest.json",
        ROOT / "artifacts/confirmatory_2025/audit_report.json",
        ROOT / "artifacts/confirmatory_2025/gate_report.json",
    ]
    for directory in (ROOT / "paper/tables", ROOT / "artifacts/confirmatory_2025/figures"):
        required.extend(sorted(path for path in directory.iterdir() if path.is_file()))
    records = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size} for path in required]
    result = {
        "schema_version": 1,
        "status": "draft_complete",
        "template_status": "generic article; migrate to official workshop style before submission",
        "headline_claim_allowed": False,
        "pdf_pages": 5,
        "files": records,
    }
    (ROOT / "paper/submission_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
