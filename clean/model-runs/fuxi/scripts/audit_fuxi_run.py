#!/usr/bin/env python3
"""Summarize FuXi-S2S inputs, forecasts, manifests, and unfinished work."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import run_fuxi_forecast as pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=pipeline.DEFAULT_CONFIG)
    parser.add_argument("--verify-outputs", action="store_true")
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    config = pipeline.load_config(args.config)
    dates = pipeline.load_dates(config)
    counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    missing_examples: list[str] = []
    invalid_examples: list[dict[str, str]] = []

    for date in dates:
        paths = pipeline.paths_for(config, date)
        if paths["input"].is_file():
            counts["inputs"] += 1
        if paths["output"].is_file():
            counts["outputs"] += 1
            if args.verify_outputs:
                try:
                    pipeline.validate_output(paths["output"], date, config)
                    counts["verified_outputs"] += 1
                except Exception as exc:  # noqa: BLE001
                    invalid_examples.append(
                        {"date": date.strftime("%Y%m%d"), "error": f"{type(exc).__name__}: {exc}"}
                    )
        elif len(missing_examples) < args.examples:
            missing_examples.append(date.strftime("%Y%m%d"))
        if paths["manifest"].is_file():
            counts["manifests"] += 1
            try:
                record = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                statuses[str(record.get("status", "missing_status"))] += 1
            except (OSError, json.JSONDecodeError):
                statuses["invalid_manifest"] += 1
        if paths["raw"].is_dir():
            counts["raw_work_directories"] += 1

    report = {
        "run_label": config["run_label"],
        "storage_root": config["storage_root"],
        "planned_dates": len(dates),
        "counts": dict(sorted(counts.items())),
        "manifest_statuses": dict(sorted(statuses.items())),
        "missing_output_examples": missing_examples,
        "invalid_output_examples": invalid_examples[: args.examples],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if invalid_examples else 0


if __name__ == "__main__":
    raise SystemExit(main())
