#!/usr/bin/env python
"""Check the benchmark workspace foundation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2s_benchmark.paths import foundation_paths, get_paths
from s2s_benchmark.regions import mask_summary


def _status(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def main() -> int:
    paths = get_paths()
    print(f"project root: {paths.project_root}")
    print(f"storage root: {paths.storage_root}")
    print("")
    print("Required paths")
    missing = []
    for name, path in foundation_paths(paths).items():
        status = _status(path)
        print(f"  {status:7s} {name:24s} {path}")
        if status != "OK":
            missing.append((name, path))

    print("")
    print("Mask summaries")
    for dgrid in (1.5, 0.5, 0.25):
        print(f"  {dgrid:g} deg")
        for row in mask_summary(dgrid):
            print(
                "    {label:24s} cells={cells:4d} grid={lat}x{lon}".format(**row)
            )

    if missing:
        print("")
        print("Missing foundation paths:")
        for name, path in missing:
            print(f"  - {name}: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
