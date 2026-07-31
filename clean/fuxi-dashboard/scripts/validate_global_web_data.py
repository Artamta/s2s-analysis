#!/usr/bin/env python3
"""Validate the compact global browser package independently of the exporter."""

from __future__ import annotations

import argparse
import array
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys


EXPECTED_VARIABLES = {
    "precipitation",
    "temperature",
    "z500",
    "wind850",
    "mslp",
    "sst",
    "olr",
}
PHYSICAL_RANGES = {
    "precipitation": (0.0, 500.0),
    "temperature": (-150.0, 70.0),
    "z500": (250.0, 700.0),
    "wind850": (0.0, 100.0),
    "mslp": (800.0, 1150.0),
    "sst": (-100.0, 70.0),
    "olr": (0.0, 500.0),
}
SPREAD_RANGES = {
    "precipitation": (0.0, 500.0),
    "temperature": (0.0, 100.0),
    "z500": (0.0, 100.0),
    "wind850": (0.0, 100.0),
    "mslp": (0.0, 100.0),
    "sst": (0.0, 100.0),
    "olr": (0.0, 300.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public/data/global",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    metadata_path = args.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["schema_version"] != 2:
        raise ValueError("unsupported global schema version")
    if metadata["validation"]["status"] != "green":
        raise ValueError("global publication gate is not green")
    if metadata["issue"]["lead_days"] != 42:
        raise ValueError("global package must contain 42 daily leads")
    if metadata["grid"]["shape"] != [121, 240]:
        raise ValueError("global package must use the native 121 x 240 grid")
    if set(metadata["variables"]) != EXPECTED_VARIABLES:
        raise ValueError("global package must contain TP, T2M, and Z500 products")

    dates = [dt.date.fromisoformat(value) for value in metadata["valid_period_starts"]]
    if len(dates) != 42:
        raise ValueError("global valid-period list must contain 42 dates")
    if any(right - left != dt.timedelta(days=1) for left, right in zip(dates, dates[1:])):
        raise ValueError("global valid-period dates must be consecutive")

    expected_values = 42 * 121 * 240
    expected_bytes = expected_values * 2
    mask_definition = metadata["grid"].get("ocean_mask")
    if not isinstance(mask_definition, dict):
        raise ValueError("global grid must declare the ocean display mask")
    mask_path = args.data_dir / mask_definition["path"]
    if not mask_path.is_file() or mask_path.stat().st_size != 121 * 240:
        raise ValueError("ocean display mask has an invalid byte length")
    if mask_definition["size_bytes"] != 121 * 240:
        raise ValueError("ocean display mask declares an invalid byte length")
    if sha256(mask_path) != mask_definition["sha256"]:
        raise ValueError("ocean display mask failed SHA-256 validation")
    mask = mask_path.read_bytes()
    if set(mask) - {0, 1}:
        raise ValueError("ocean display mask must contain only 0 and 1")
    ocean_fraction = sum(mask) / len(mask)
    if ocean_fraction < 0.6 or ocean_fraction > 0.8:
        raise ValueError("ocean display support has an implausible fraction")

    for key, definition in metadata["variables"].items():
        if definition.get("family") not in {
            "surface",
            "circulation",
            "ocean-convection",
        }:
            raise ValueError(f"{key} has no valid interpretation family")
        if not definition.get("interpretation"):
            raise ValueError(f"{key} has no interpretation guidance")
        validate_binary(
            args.data_dir,
            key,
            definition,
            expected_values,
            expected_bytes,
            PHYSICAL_RANGES[key],
        )
        spread = definition.get("spread")
        if not isinstance(spread, dict):
            raise ValueError(f"{key} must declare ensemble spread")
        if spread.get("offset") != 0.0:
            raise ValueError(f"{key} spread must have zero offset")
        validate_binary(
            args.data_dir,
            f"{key} spread",
            spread,
            expected_values,
            expected_bytes,
            SPREAD_RANGES[key],
        )
        if len(spread.get("frame_area_means", [])) != 42:
            raise ValueError(f"{key} spread must declare 42 area means")
        if any(
            not math.isfinite(value) or value < 0
            for value in spread["frame_area_means"]
        ):
            raise ValueError(f"{key} spread contains invalid area means")
        statistic = spread.get("statistic", "").lower()
        if (
            "population standard deviation" not in statistic
            or "not calibrated confidence" not in statistic
        ):
            raise ValueError(f"{key} spread statistic is incompletely documented")
        vector = definition.get("vector")
        if key == "wind850":
            if not isinstance(vector, dict):
                raise ValueError("850 hPa wind must declare U/V vector components")
            for component in ("u", "v"):
                validate_binary(
                    args.data_dir,
                    f"wind850 {component}",
                    vector[component],
                    expected_values,
                    expected_bytes,
                    (-100.0, 100.0),
                )
        elif vector is not None:
            raise ValueError(f"{key} must not declare vector components")

    if metadata["variables"]["sst"].get("domain") != "ocean":
        raise ValueError("SST must be restricted to ocean display support")
    olr_interpretation = metadata["variables"]["olr"]["interpretation"].lower()
    if "not an mjo index" not in olr_interpretation:
        raise ValueError("OLR interpretation must distinguish it from an MJO index")
    sst_definition = metadata["variables"]["sst"]
    sst_encoded = array.array("H")
    sst_encoded.frombytes((args.data_dir / sst_definition["path"]).read_bytes())
    if sys.byteorder != "little":
        sst_encoded.byteswap()
    supported_indices = [index for index, supported in enumerate(mask) if supported]
    for day in range(42):
        start = day * 121 * 240
        supported_values = [
            sst_encoded[start + index] * sst_definition["scale"]
            + sst_definition["offset"]
            for index in supported_indices
        ]
        if min(supported_values) < -3.01 or max(supported_values) > 45.01:
            raise ValueError(
                f"SST day {day + 1} is outside physical open-ocean support"
            )

    serialized = json.dumps(metadata)
    forbidden = ("/storage/", "/home/", "password", "secret", "credential")
    if any(token in serialized.lower() for token in forbidden):
        raise ValueError("global metadata contains private or sensitive information")
    print("global web data: validated")


def validate_binary(
    data_dir: Path,
    key: str,
    definition: dict[str, object],
    expected_values: int,
    expected_bytes: int,
    physical_range: tuple[float, float],
) -> None:
    path = data_dir / str(definition["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path} has an invalid byte length")
    if definition["size_bytes"] != expected_bytes:
        raise ValueError(f"{key} declares an invalid byte length")
    if sha256(path) != definition["sha256"]:
        raise ValueError(f"{key} failed SHA-256 validation")
    encoded = array.array("H")
    encoded.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        encoded.byteswap()
    if len(encoded) != expected_values:
        raise ValueError(f"{key} has an invalid element count")
    scale = float(definition["scale"])
    offset = float(definition["offset"])
    value_minimum = min(encoded) * scale + offset
    value_maximum = max(encoded) * scale + offset
    minimum, maximum = physical_range
    if not math.isfinite(value_minimum) or not math.isfinite(value_maximum):
        raise ValueError(f"{key} contains non-finite decoded values")
    if value_minimum < minimum or value_maximum > maximum:
        raise ValueError(
            f"{key} range {value_minimum:.2f} to {value_maximum:.2f} "
            f"falls outside {minimum} to {maximum}"
        )
    if len(definition["frame_ranges"]) != 42:
        raise ValueError(f"{key} must declare 42 per-frame ranges")


if __name__ == "__main__":
    main()
