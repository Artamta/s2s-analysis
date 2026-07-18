#!/usr/bin/env python3
"""Materialize one dated NeuralGCM config from a frozen production template."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys


COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))

import neuralgcm_run_contract as run_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def member_seed(run_label: str, init_date: str, member: int) -> int:
    payload = f"{run_label}/{init_date}/{member}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def calendar_dates(config: dict) -> list[str]:
    path = run_contract.repo_path(config["calendar"]["path"])
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["init_date"] for row in csv.DictReader(handle)]


def materialize(template: Path, index: int) -> dict:
    config = run_contract.load_config(template)
    if config.get("run_mode") != "production":
        raise ValueError("template must have run_mode=production")

    dates = calendar_dates(config)
    if index < 0 or index >= len(dates):
        raise IndexError(f"calendar index {index} is outside 0..{len(dates) - 1}")
    init_date = dates[index]
    init = datetime.strptime(init_date, "%Y-%m-%d")
    init_time = f"{init_date}T00:00:00"
    forcing_time = f"{(init - timedelta(days=1)):%Y-%m-%d}T00:00:00"
    stamp = init.strftime("%Y%m%d")
    year = init.strftime("%Y")

    config["case"] = {
        "init_time": init_time,
        "forecast_reference_time": init_time,
        "information_cutoff_time": init_time,
    }
    config["initial_conditions"]["atmosphere_source_time"] = init_time
    config["initial_conditions"]["forcing_source_time"] = forcing_time
    count = int(config["forecast"]["member_count"])
    config["forecast"]["member_seeds"] = [
        member_seed(config["run_label"], init_date, member)
        for member in range(count)
    ]
    config["storage"].update(
        {
            "input": f"inputs/{year}/{stamp}.nc",
            "input_manifest": f"inputs/{year}/{stamp}.json",
            "output": f"forecasts/{year}/{stamp}.nc",
            "output_manifest": f"manifests/{year}/{stamp}.json",
        }
    )
    run_contract.validate_config(config)
    return config


def write_atomic(config: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    config = materialize(args.template, args.index)
    write_atomic(config, args.output)
    print(
        json.dumps(
            {
                "index": args.index,
                "init_time": config["case"]["init_time"],
                "member_seeds": config["forecast"]["member_seeds"],
                "output": config["storage"]["output"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
