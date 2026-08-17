#!/usr/bin/env python3
"""CLI for the frozen India S2S benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from india_s2s_bench.workflow import audit, preflight, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "audit"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == "preflight":
        output = args.output or root / "artifacts/preflight"
        result = preflight(root, output)
    elif args.command == "run":
        output = args.output or root / "artifacts/confirmatory_2025"
        result = run_experiment(root, output)
    else:
        output = args.output or root / "artifacts/confirmatory_2025"
        result = audit(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
