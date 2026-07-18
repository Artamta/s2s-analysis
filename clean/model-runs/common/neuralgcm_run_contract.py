#!/usr/bin/env python3
"""Validate frozen 42-day NeuralGCM run configurations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pilot_contract


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_STORAGE_PREFIX = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/neural-gcm"
)
PRODUCT_FIELDS = {
    "daily_tp": ["tp"],
    "daily_pressure_temperature": ["t850", "t1000"],
}


class RunContractError(ValueError):
    """Raised when a NeuralGCM run configuration is inconsistent."""


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def storage_path(config: dict[str, Any], key: str) -> Path:
    root = Path(config["storage"]["root"])
    relative = Path(config["storage"][key])
    if relative.is_absolute() or ".." in relative.parts:
        raise RunContractError(f"storage.{key} must be a safe relative path")
    return root / relative


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "calendar", "case", "domain", "forecast", "initial_conditions",
        "model", "purpose", "run_label", "runtime", "schema_version",
        "source", "storage",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise RunContractError(f"config missing top-level keys: {missing}")
    if config["schema_version"] != 1:
        raise RunContractError("unsupported schema_version")
    run_mode = config.get("run_mode", "pilot")
    if run_mode not in {"pilot", "production"}:
        raise RunContractError("run_mode must be pilot or production")
    if run_mode == "pilot" and "not for skill scores" not in config["purpose"]:
        raise RunContractError("pilot purpose must forbid skill scoring")
    if run_mode == "production" and "production" not in config["purpose"].lower():
        raise RunContractError("production purpose must identify production output")

    calendar_cfg = config["calendar"]
    calendar_path = repo_path(calendar_cfg["path"])
    actual_hash = pilot_contract.sha256_file(calendar_path)
    if actual_hash != calendar_cfg["sha256"]:
        raise RunContractError("calendar SHA256 mismatch")
    dates: list[datetime] = []
    counts: Counter[int] = Counter()
    with calendar_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            date = datetime.strptime(row["init_date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            dates.append(date)
            counts[date.year] += 1
            if int(row["lead_days"]) != 42:
                raise RunContractError(f"calendar lead mismatch at {row['init_date']}")
    expected_counts = {
        int(year): int(count) for year, count in calendar_cfg["year_counts"].items()
    }
    if len(dates) != int(calendar_cfg["date_count"]):
        raise RunContractError("calendar date_count mismatch")
    if dates != sorted(set(dates)):
        raise RunContractError("calendar must be sorted and unique")
    if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
        raise RunContractError("calendar year_counts mismatch")

    case = config["case"]
    init_time = parse_utc(case["init_time"])
    if init_time.hour != 0 or init_time not in dates:
        raise RunContractError("case must be an exact 00 UTC calendar initialization")
    for key in ("forecast_reference_time", "information_cutoff_time"):
        if parse_utc(case[key]) != init_time:
            raise RunContractError(f"case.{key} must equal initialization")
    ic = config["initial_conditions"]
    if parse_utc(ic["atmosphere_source_time"]) != init_time:
        raise RunContractError("atmospheric IC time mismatch")
    if parse_utc(ic["forcing_source_time"]) != init_time - timedelta(days=1):
        raise RunContractError("forcing must use D-1 00 UTC")
    if "persistence" not in ic["forcing_policy"].lower():
        raise RunContractError("forecast forcing policy must declare persistence")

    model = config["model"]
    product = model["product"]
    if product not in PRODUCT_FIELDS:
        raise RunContractError(f"unsupported product: {product}")
    if model["retained_fields"] != PRODUCT_FIELDS[product]:
        raise RunContractError("retained fields do not match product contract")
    if product == "daily_tp" and model["decoded_field"] != "precipitation_cumulative_mean":
        raise RunContractError("TP product must use cumulative precipitation")
    if product == "daily_pressure_temperature":
        if model.get("pressure_levels_hpa") != [850, 1000]:
            raise RunContractError("temperature product must retain 850 and 1000 hPa")
        if model.get("t2m_available") is not False:
            raise RunContractError("1.4 degree checkpoint must explicitly forbid T2M")
    if len(model["checkpoint_sha256"]) != 64:
        raise RunContractError("invalid checkpoint SHA256")
    if int(model["checkpoint_size_bytes"]) <= 0:
        raise RunContractError("invalid checkpoint size")
    if min(int(model["native_longitude_count"]), int(model["native_latitude_count"])) <= 0:
        raise RunContractError("invalid native grid")

    forecast = config["forecast"]
    interval = int(forecast["output_interval_hours"])
    lead_days = int(forecast["lead_days"])
    expected_steps = lead_days * 24 // interval + 1
    if lead_days != 42 or interval != 6 or int(forecast["unroll_steps"]) != expected_steps:
        raise RunContractError("42-day endpoint arithmetic must be 169 six-hour frames")
    if forecast["start_with_input"] is not True:
        raise RunContractError("forecast must retain the +0 boundary frame")
    seeds = [int(seed) for seed in forecast["member_seeds"]]
    if len(seeds) != int(forecast["member_count"]) or len(seeds) != len(set(seeds)):
        raise RunContractError("member seeds must be unique and match member_count")
    if any(seed < 0 or seed > 0xFFFFFFFF for seed in seeds):
        raise RunContractError("member seeds must be uint32")

    domain = config["domain"]
    if domain["latitude"] != [39.0, 0.0, -1.5] or domain["longitude"] != [60.0, 99.0, 1.5]:
        raise RunContractError("common domain must be the exact 27x27 physics grid")
    if int(domain["latitude_count"]) != 27 or int(domain["longitude_count"]) != 27:
        raise RunContractError("common domain counts must be 27x27")

    storage_root = Path(config["storage"]["root"])
    try:
        storage_root.relative_to(EXPECTED_STORAGE_PREFIX)
    except ValueError as exc:
        raise RunContractError("storage root is outside NeuralGCM storage") from exc
    storage_keys = ["checkpoint", "input", "input_manifest", "output", "output_manifest"]
    if run_mode == "pilot":
        storage_keys.append("pilot_report")
    for key in storage_keys:
        storage_path(config, key)

    return {
        "calendar_sha256": actual_hash,
        "date_count": len(dates),
        "init_time": init_time.isoformat(),
        "lead_days": lead_days,
        "member_count": len(seeds),
        "product": product,
        "run_mode": run_mode,
        "retained_fields": model["retained_fields"],
        "run_label": config["run_label"],
        "storage_root": str(storage_root),
        "unroll_steps": expected_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(validate_config(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
