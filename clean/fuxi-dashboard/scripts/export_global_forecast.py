#!/usr/bin/env python3
"""Export compact global ensemble-mean fields for the static forecast viewer."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


GRAVITY_M_S2 = 9.80665
CHANNELS = ("tp", "t2m", "z500")
EXPECTED_LATITUDE = np.linspace(90.0, -90.0, 121, dtype=np.float32)
EXPECTED_LONGITUDE = np.linspace(0.0, 358.5, 240, dtype=np.float32)


@dataclass(frozen=True)
class ExportVariable:
    key: str
    source_channel: str
    label: str
    short_label: str
    units: str
    description: str
    offset: float
    scale: float
    legend_boundaries: tuple[float, ...]
    legend_colors: tuple[str, ...]
    under: str
    over: str


VARIABLES = (
    ExportVariable(
        key="precipitation",
        source_channel="tp",
        label="Daily precipitation",
        short_label="Precipitation",
        units="mm/day",
        description="Ensemble-mean precipitation rate converted to a 24-hour total.",
        offset=0.0,
        scale=0.01,
        legend_boundaries=(0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
        legend_colors=(
            "#081d2a",
            "#123d50",
            "#17687a",
            "#1da297",
            "#74c98e",
            "#e3d66d",
            "#e98e45",
        ),
        under="#07151e",
        over="#e8543f",
    ),
    ExportVariable(
        key="temperature",
        source_channel="t2m",
        label="2 m temperature",
        short_label="Temperature",
        units="°C",
        description="Daily ensemble-mean temperature two metres above the surface.",
        offset=-150.0,
        scale=0.01,
        legend_boundaries=(-60.0, -40.0, -20.0, 0.0, 15.0, 25.0, 35.0, 45.0),
        legend_colors=(
            "#5a4fa3",
            "#3c78bd",
            "#4db7d0",
            "#c8d8b5",
            "#f3c35b",
            "#ef8548",
            "#c8473c",
        ),
        under="#352b75",
        over="#7e1f35",
    ),
    ExportVariable(
        key="z500",
        source_channel="z500",
        label="500 hPa geopotential height",
        short_label="Z500",
        units="dam",
        description="Ensemble-mean 500 hPa geopotential converted to decametres.",
        offset=0.0,
        scale=0.01,
        legend_boundaries=(460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 580.0, 600.0),
        legend_colors=(
            "#49396d",
            "#345f83",
            "#288a91",
            "#62aa7f",
            "#b4bd72",
            "#d5a15f",
            "#bf6654",
        ),
        under="#30244c",
        over="#873b4a",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=100)
    parser.add_argument("--lead-days", type=int, default=42)
    parser.add_argument("--initialization", default="2026-07-28")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_path(raw_dir: Path, member: int, lead_day: int) -> Path:
    return raw_dir / "member" / f"{member:02d}" / f"{lead_day:02d}.nc"


def validate_coordinates(source: xr.DataArray, path: Path) -> None:
    if not np.allclose(source.lat.values, EXPECTED_LATITUDE):
        raise ValueError(f"{path} has an unexpected latitude coordinate")
    if not np.allclose(source.lon.values, EXPECTED_LONGITUDE):
        raise ValueError(f"{path} has an unexpected longitude coordinate")
    channels = {str(value) for value in source.channel.values.tolist()}
    missing = set(CHANNELS) - channels
    if missing:
        raise ValueError(f"{path} is missing channels {sorted(missing)}")


def read_selected(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    source = xr.open_dataarray(path, decode_times=False)
    try:
        validate_coordinates(source, path)
        selected = (
            source.sel(channel=list(CHANNELS))
            .squeeze(drop=True)
            .transpose("channel", "lat", "lon")
        )
        values = np.asarray(selected.values, dtype=np.float64)
        if values.shape != (3, 121, 240):
            raise ValueError(f"{path} produced shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains non-finite selected fields")
        if values[0].min() < -1e-5:
            raise ValueError(f"{path} contains negative precipitation")
        return values
    finally:
        source.close()


def convert_fields(mean: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "precipitation": np.maximum(mean[0], 0.0) * 24.0,
        "temperature": mean[1] - 273.15,
        "z500": mean[2] / GRAVITY_M_S2 / 10.0,
    }


def convert_spread(spread: np.ndarray) -> dict[str, np.ndarray]:
    """Convert population spread without applying absolute-field offsets."""

    return {
        "precipitation": spread[0] * 24.0,
        "temperature": spread[1],
        "z500": spread[2] / GRAVITY_M_S2 / 10.0,
    }


def quantize_with(
    values: np.ndarray, *, offset: float, scale: float, label: str
) -> np.ndarray:
    encoded = np.rint((values - offset) / scale)
    if encoded.min() < 0 or encoded.max() > np.iinfo(np.uint16).max:
        raise ValueError(
            f"{label} cannot be represented by its locked quantization: "
            f"{float(values.min()):.3f} to {float(values.max()):.3f}"
        )
    return encoded.astype("<u2")


def quantize(values: np.ndarray, variable: ExportVariable) -> np.ndarray:
    return quantize_with(
        values,
        offset=variable.offset,
        scale=variable.scale,
        label=variable.key,
    )


def global_area_mean(values: np.ndarray) -> float:
    """Return a cosine-latitude weighted global mean."""

    latitude_weights = np.cos(np.deg2rad(EXPECTED_LATITUDE))[:, None]
    weights = np.broadcast_to(latitude_weights, values.shape)
    return float(np.average(values, weights=weights))


def iso_day(initialization: dt.date, offset: int) -> str:
    return (initialization + dt.timedelta(days=offset)).isoformat()


def main() -> None:
    args = parse_args()
    if args.members < 1 or args.lead_days != 42:
        raise ValueError("the public global contract requires 42 leads and >=1 member")
    initialization = dt.date.fromisoformat(args.initialization)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variable_by_key = {variable.key: variable for variable in VARIABLES}
    frames: dict[str, list[np.ndarray]] = {key: [] for key in variable_by_key}
    spread_frames: dict[str, list[np.ndarray]] = {
        key: [] for key in variable_by_key
    }
    frame_ranges: dict[str, list[dict[str, float]]] = {
        key: [] for key in variable_by_key
    }
    spread_frame_ranges: dict[str, list[dict[str, float]]] = {
        key: [] for key in variable_by_key
    }
    spread_frame_means: dict[str, list[float]] = {
        key: [] for key in variable_by_key
    }

    for lead_day in range(1, args.lead_days + 1):
        mean = np.zeros((3, 121, 240), dtype=np.float64)
        second_moment = np.zeros_like(mean)
        for member in range(args.members):
            values = read_selected(raw_path(args.raw_dir, member, lead_day))
            delta = values - mean
            mean += delta / (member + 1)
            second_moment += delta * (values - mean)
        population_spread = np.sqrt(second_moment / args.members)
        converted = convert_fields(mean)
        converted_spread = convert_spread(population_spread)
        for key, values in converted.items():
            frames[key].append(quantize(values, variable_by_key[key]))
            frame_ranges[key].append(
                {"minimum": float(values.min()), "maximum": float(values.max())}
            )
            spread_values = converted_spread[key]
            spread_frames[key].append(
                quantize_with(
                    spread_values,
                    offset=0.0,
                    scale=variable_by_key[key].scale,
                    label=f"{key} spread",
                )
            )
            spread_frame_ranges[key].append(
                {
                    "minimum": float(spread_values.min()),
                    "maximum": float(spread_values.max()),
                }
            )
            spread_frame_means[key].append(global_area_mean(spread_values))
        print(f"aggregated global lead {lead_day:02d}/{args.lead_days}", flush=True)

    file_records: dict[str, dict[str, object]] = {}
    for variable in VARIABLES:
        binary_path = args.output_dir / f"{variable.key}.bin"
        stacked = np.stack(frames[variable.key], axis=0)
        binary_path.write_bytes(stacked.tobytes(order="C"))
        decoded = stacked.astype(np.float64) * variable.scale + variable.offset
        expected_size = args.lead_days * 121 * 240 * 2
        if binary_path.stat().st_size != expected_size:
            raise ValueError(
                f"{binary_path} has {binary_path.stat().st_size} bytes, "
                f"expected {expected_size}"
            )
        file_records[variable.key] = {
            "path": binary_path.name,
            "sha256": sha256(binary_path),
            "size_bytes": binary_path.stat().st_size,
            "dtype": "uint16-little-endian",
            "offset": variable.offset,
            "scale": variable.scale,
            "minimum": float(decoded.min()),
            "maximum": float(decoded.max()),
            "frame_ranges": frame_ranges[variable.key],
        }
        spread_path = args.output_dir / f"{variable.key}-spread.bin"
        stacked_spread = np.stack(spread_frames[variable.key], axis=0)
        spread_path.write_bytes(stacked_spread.tobytes(order="C"))
        if spread_path.stat().st_size != expected_size:
            raise ValueError(
                f"{spread_path} has {spread_path.stat().st_size} bytes, "
                f"expected {expected_size}"
            )
        decoded_spread = stacked_spread.astype(np.float64) * variable.scale
        file_records[variable.key]["spread"] = {
            "path": spread_path.name,
            "sha256": sha256(spread_path),
            "size_bytes": spread_path.stat().st_size,
            "dtype": "uint16-little-endian",
            "offset": 0.0,
            "scale": variable.scale,
            "minimum": float(decoded_spread.min()),
            "maximum": float(decoded_spread.max()),
            "frame_ranges": spread_frame_ranges[variable.key],
            "frame_area_means": spread_frame_means[variable.key],
            "statistic": (
                "Population standard deviation across 100 members (ddof=0); "
                "spread measures ensemble disagreement, not calibrated confidence."
            ),
        }

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata = {
        "schema_version": 1,
        "generated_at": generated_at,
        "issue": {
            "initialization": f"{initialization.isoformat()}T00:00:00Z",
            "members": args.members,
            "lead_days": args.lead_days,
            "status": "experimental",
            "public_label": "Experimental global ensemble guidance",
            "input_description": "Operational global analysis proxy",
            "ensemble_relation": (
                "Independent 100-member companion ensemble generated from the "
                "same frozen initialization input as the India validation case."
            ),
            "display_interpolation": (
                "Cross-fades are a visual transition between daily fields; "
                "they are not additional forecast times."
            ),
        },
        "grid": {
            "shape": [121, 240],
            "spacing_degrees": 1.5,
            "latitude_first": 90.0,
            "latitude_last": -90.0,
            "longitude_first": 0.0,
            "longitude_last": 358.5,
            "value_order": "lead_day, latitude north-to-south, longitude eastward",
        },
        "valid_period_starts": [
            iso_day(initialization, offset) for offset in range(args.lead_days)
        ],
        "variables": {
            variable.key: {
                "label": variable.label,
                "short_label": variable.short_label,
                "units": variable.units,
                "description": variable.description,
                "legend": {
                    "boundaries": variable.legend_boundaries,
                    "colors": variable.legend_colors,
                    "under": variable.under,
                    "over": variable.over,
                },
                **file_records[variable.key],
            }
            for variable in VARIABLES
        },
        "validation": {
            "status": "green",
            "checks": [
                "exact 121 x 240 global grid",
                f"all {args.members} members present for every lead",
                "companion-ensemble relationship declared",
                "all 42 daily leads present",
                "finite TP, T2M, and Z500 fields",
                "nonnegative precipitation",
                "locked unit conversions and quantization",
                "population ensemble spread with ddof=0",
                "cosine-latitude global spread summaries",
                "binary size and SHA-256 checks",
            ],
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
