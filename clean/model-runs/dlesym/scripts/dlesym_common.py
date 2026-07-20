#!/usr/bin/env python3
"""Shared contracts for the reproducible DLESyM benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    calendar = REPO_ROOT / config["calendar"]["path"]
    if sha256_file(calendar) != config["calendar"]["sha256"]:
        raise ValueError("calendar SHA256 does not match the frozen contract")
    dates = read_dates(config)
    if len(dates) != config["calendar"]["date_count"]:
        raise ValueError("calendar date count does not match the frozen contract")
    years, counts = np.unique([date[:4] for date in dates], return_counts=True)
    if [int(year) for year in years] != config["calendar"]["years"]:
        raise ValueError("calendar years do not match the frozen contract")
    if counts.tolist() != config["calendar"]["year_counts"]:
        raise ValueError("calendar per-year counts do not match the frozen contract")
    return config


def read_dates(config: dict[str, Any]) -> list[str]:
    calendar = REPO_ROOT / config["calendar"]["path"]
    with calendar.open(newline="", encoding="utf-8") as handle:
        dates = [row["init_date"] for row in csv.DictReader(handle)]
    if dates != sorted(set(dates)):
        raise ValueError("calendar dates must be unique and sorted")
    return dates


def select_date(
    config: dict[str, Any], index: int | None, init_date: str | None
) -> tuple[int, str]:
    dates = read_dates(config)
    if (index is None) == (init_date is None):
        raise ValueError("provide exactly one of --index or --init-date")
    if index is not None:
        if not 0 <= index < len(dates):
            raise IndexError(f"calendar index {index} is outside 0..{len(dates) - 1}")
        return index, dates[index]
    assert init_date is not None
    if init_date not in dates:
        raise ValueError(f"{init_date} is not in the frozen calendar")
    return dates.index(init_date), init_date


def product_paths(
    config: dict[str, Any], product: str, init_date: str
) -> dict[str, Path]:
    run_label = config["products"][product]["run_label"]
    root = Path(config["storage"]["root"]) / "dlesym" / run_label
    year = init_date[:4]
    return {
        "root": root,
        "stage": root / "initial-conditions" / year / f"{init_date.replace('-', '')}.nc",
        "stage_manifest": root
        / "initial-conditions"
        / year
        / f"{init_date.replace('-', '')}.json",
        "output": root / "forecasts" / year / f"{init_date.replace('-', '')}.nc",
        "manifest": root / "manifests" / year / f"{init_date.replace('-', '')}.json",
        "failed": root / "failures" / year / f"{init_date.replace('-', '')}.json",
        "inventory": root / "provenance" / "checkpoint_inventory.json",
    }


def existing_pair_is_valid(data_path: Path, manifest_path: Path, key: str) -> bool:
    if not data_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("status") == "passed" and manifest.get(key) == sha256_file(
            data_path
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def package_for(product: str):
    if product == "v1_t2m":
        from earth2studio.models.px import DLESyM

        return DLESyM.load_default_package()
    if product == "v0_tp_t2m":
        from earth2studio.models.px import DLESyMv0_ISCCP_ERA5

        return DLESyMv0_ISCCP_ERA5.load_default_package()
    raise ValueError(f"unknown product {product}")


def repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def coords_from_dataset(dataset: Any) -> OrderedDict[str, np.ndarray]:
    return OrderedDict(
        (dimension, np.asarray(dataset.coords[dimension].values))
        for dimension in dataset["state"].dims
    )


def target_coordinates(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lat_start, lat_stop, lat_step = config["domain"]["latitude"]
    lon_start, lon_stop, lon_step = config["domain"]["longitude"]
    latitude = np.arange(lat_start, lat_stop + lat_step / 2, lat_step, dtype=np.float64)
    longitude = np.arange(lon_start, lon_stop + lon_step / 2, lon_step, dtype=np.float64)
    if (len(latitude), len(longitude)) != tuple(config["domain"]["shape"]):
        raise ValueError("target coordinate shape is inconsistent with config")
    return latitude, longitude


def conservative_weights(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_spacing: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return separable spherical-cell overlap weights for area means."""
    source_dlat = abs(float(source_lat[1] - source_lat[0]))
    source_dlon = abs(float(source_lon[1] - source_lon[0]))
    src_lat_lo = np.maximum(source_lat - source_dlat / 2, -90.0)
    src_lat_hi = np.minimum(source_lat + source_dlat / 2, 90.0)
    lat_weights = np.zeros((len(target_lat), len(source_lat)), dtype=np.float64)
    for row, center in enumerate(target_lat):
        lo = center - target_spacing / 2
        hi = center + target_spacing / 2
        overlap_lo = np.maximum(src_lat_lo, lo)
        overlap_hi = np.minimum(src_lat_hi, hi)
        valid = overlap_hi > overlap_lo
        lat_weights[row, valid] = np.sin(np.deg2rad(overlap_hi[valid])) - np.sin(
            np.deg2rad(overlap_lo[valid])
        )
        lat_weights[row] /= lat_weights[row].sum()

    src_lon_lo = source_lon - source_dlon / 2
    src_lon_hi = source_lon + source_dlon / 2
    lon_weights = np.zeros((len(target_lon), len(source_lon)), dtype=np.float64)
    for row, center in enumerate(target_lon):
        lo = center - target_spacing / 2
        hi = center + target_spacing / 2
        overlap = np.maximum(
            0.0, np.minimum(src_lon_hi, hi) - np.maximum(src_lon_lo, lo)
        )
        lon_weights[row] = overlap / overlap.sum()

    payload = lat_weights.tobytes() + lon_weights.tobytes()
    return lat_weights, lon_weights, sha256_bytes(payload)


def period_coordinates(init_date: str, lead_days: int) -> dict[str, Any]:
    init = np.datetime64(init_date, "ns")
    lead = np.arange(1, lead_days + 1, dtype=np.int16)
    start = init + (lead - 1).astype("timedelta64[D]")
    end = init + lead.astype("timedelta64[D]")
    return {
        "lead_day": lead,
        "valid_time": ("lead_day", end),
        "forecast_period_start": ("lead_day", start),
        "forecast_period_end": ("lead_day", end),
        "forecast_period_bounds": (
            ("lead_day", "bounds"),
            np.stack([start, end], axis=1),
        ),
        "bounds": np.array([0, 1], dtype=np.int8),
        "init_time": init,
        "forecast_reference_time": init,
        "information_cutoff_time": init,
    }
