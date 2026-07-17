#!/usr/bin/env python3
"""Validate the frozen one-date model pilot before data or GPU work starts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ARCO_URL = (
    "gs://gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
EXPECTED_STORAGE_PREFIX = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs"
)


class ContractError(ValueError):
    """Raised when a frozen pilot contract is internally inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def storage_path(config: dict[str, Any], key: str) -> Path:
    root = Path(config["storage"]["root"])
    relative = Path(config["storage"][key])
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"storage.{key} must be a safe relative path")
    return root / relative


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "calendar",
        "case",
        "initial_conditions",
        "model",
        "purpose",
        "run_label",
        "runtime",
        "schema_version",
        "smoke",
        "source",
        "storage",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ContractError(f"config missing top-level keys: {missing}")
    if config["schema_version"] != 1:
        raise ContractError("unsupported pilot schema_version")
    if "not for skill scores" not in config["purpose"]:
        raise ContractError("smoke purpose must explicitly forbid skill scoring")

    calendar_cfg = config["calendar"]
    calendar_path = repo_path(calendar_cfg["path"])
    if not calendar_path.is_file():
        raise ContractError(f"calendar does not exist: {calendar_path}")
    actual_calendar_hash = sha256_file(calendar_path)
    if actual_calendar_hash != calendar_cfg["sha256"]:
        raise ContractError(
            "calendar SHA256 mismatch: "
            f"expected {calendar_cfg['sha256']}, found {actual_calendar_hash}"
        )

    dates: list[datetime] = []
    year_counts: Counter[int] = Counter()
    with calendar_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            date = datetime.strptime(row["init_date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            dates.append(date)
            year_counts[date.year] += 1
            if int(row["lead_days"]) != 42:
                raise ContractError(f"calendar row {row['init_date']} is not 42 days")
    if len(dates) != calendar_cfg["date_count"]:
        raise ContractError(
            f"expected {calendar_cfg['date_count']} dates, found {len(dates)}"
        )
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ContractError("calendar must be sorted and unique")
    expected_year_counts = {
        int(year): int(count) for year, count in calendar_cfg["year_counts"].items()
    }
    if dict(sorted(year_counts.items())) != dict(sorted(expected_year_counts.items())):
        raise ContractError(
            f"calendar year counts differ: found {dict(sorted(year_counts.items()))}"
        )

    case = config["case"]
    init_time = parse_utc(case["init_time"])
    if init_time.hour != 0 or init_time.minute != 0 or init_time.second != 0:
        raise ContractError("pilot initialization must be exactly 00 UTC")
    if init_time not in dates:
        raise ContractError("pilot initialization is not in the frozen calendar")
    for key in ("forecast_reference_time", "information_cutoff_time"):
        if parse_utc(case[key]) != init_time:
            raise ContractError(f"case.{key} must equal initialization time")

    if config["source"]["url"] != EXPECTED_ARCO_URL:
        raise ContractError("pilot source must be the frozen hourly ARCO-ERA5 store")
    ic = config["initial_conditions"]
    if parse_utc(ic["atmosphere_source_time"]) != init_time:
        raise ContractError("atmospheric IC must be initialized at D 00 UTC")
    if parse_utc(ic["forcing_source_time"]) != init_time - timedelta(days=1):
        raise ContractError("NeuralGCM forcing IC must be exactly D-1 00 UTC")
    if "persistence" not in ic["forcing_policy"].lower():
        raise ContractError("primary NeuralGCM pilot must declare persisted forcing")

    checkpoint_hash = config["model"]["checkpoint_sha256"]
    if len(checkpoint_hash) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_hash):
        raise ContractError("checkpoint SHA256 must be lowercase hexadecimal")
    if int(config["model"]["checkpoint_size_bytes"]) <= 0:
        raise ContractError("checkpoint size must be positive")
    if config["model"]["available_requested_fields"] != ["tp"]:
        raise ContractError("this checkpoint may advertise only requested field tp")
    if "t2m" not in config["model"]["unavailable_requested_fields"]:
        raise ContractError("the missing NeuralGCM T2M decoder must be documented")

    smoke = config["smoke"]
    expected_frames = list(range(0, smoke["unroll_steps"] * smoke["output_interval_hours"], smoke["output_interval_hours"]))
    if smoke["expected_frame_hours"] != expected_frames:
        raise ContractError(
            f"smoke frame hours must be {expected_frames}, found {smoke['expected_frame_hours']}"
        )
    if smoke["start_with_input"] is not True:
        raise ContractError("NeuralGCM smoke must retain the +0 frame")
    if smoke["member_count"] != 1:
        raise ContractError("the first smoke test must use exactly one member")
    if smoke["output_fields"] != ["precipitation_cumulative_mean"]:
        raise ContractError("smoke output must contain only the native cumulative TP field")

    storage_root = Path(config["storage"]["root"])
    if not storage_root.is_absolute():
        raise ContractError("storage root must be absolute")
    try:
        storage_root.relative_to(EXPECTED_STORAGE_PREFIX)
    except ValueError as exc:
        raise ContractError(
            f"storage root must be under {EXPECTED_STORAGE_PREFIX}"
        ) from exc
    for key in (
        "checkpoint",
        "input",
        "input_manifest",
        "output",
        "output_manifest",
        "pilot_report",
    ):
        storage_path(config, key)

    return {
        "calendar_path": str(calendar_path),
        "calendar_sha256": actual_calendar_hash,
        "date_count": len(dates),
        "init_time": init_time.isoformat(),
        "forcing_source_time": parse_utc(ic["forcing_source_time"]).isoformat(),
        "frame_hours": expected_frames,
        "run_label": config["run_label"],
        "storage_root": str(storage_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(validate_config(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
