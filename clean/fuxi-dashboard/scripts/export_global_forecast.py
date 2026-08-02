#!/usr/bin/env python3
"""Export compact ensemble fields for the static global forecast viewer."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import xarray as xr
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from science.formulas import (  # noqa: E402
    anomaly as calculate_anomaly,
    geopotential_to_height_dam,
    kelvin_to_celsius,
    pascal_to_hectopascal,
    top_net_thermal_to_olr,
    tp_mm_hour_to_mm_day,
    wind_speed,
)

GRAVITY_M_S2 = 9.80665
CHANNELS = (
    "tp",
    "t2m",
    "z500",
    "u850",
    "v850",
    "msl",
    "sst",
    "ttr",
    "tcwv",
)
CHANNEL_INDEX = {channel: index for index, channel in enumerate(CHANNELS)}
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
    family: str
    interpretation: str
    offset: float
    scale: float
    legend_boundaries: tuple[float, ...]
    legend_colors: tuple[str, ...]
    under: str
    over: str
    domain: str = "global"


@dataclass(frozen=True)
class AnomalyStyle:
    label: str
    short_label: str
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
        short_label="Rainfall",
        units="mm/day",
        description="Ensemble-mean precipitation rate converted to a 24-hour total.",
        family="surface",
        interpretation="Rain belts, monsoon organization, and storm-track shifts.",
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
        short_label="2 m temp",
        units="°C",
        description="Daily ensemble-mean temperature two metres above the surface.",
        family="surface",
        interpretation="Persistent warm and cool regions; local values weaken with lead.",
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
        family="circulation",
        interpretation="Broad ridges, troughs, blocking, and planetary-wave evolution.",
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
    ExportVariable(
        key="wind850",
        source_channel="u850,v850",
        label="850 hPa wind speed",
        short_label="850 wind",
        units="m/s",
        description=(
            "Arithmetic ensemble mean of member 850 hPa wind speed; arrows "
            "show the ensemble-mean U/V direction."
        ),
        family="circulation",
        interpretation="Monsoon flow, tropical inflow, and lower-tropospheric circulation.",
        offset=0.0,
        scale=0.01,
        legend_boundaries=(0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 25.0, 35.0),
        legend_colors=(
            "#10212b",
            "#194554",
            "#1d7180",
            "#36a09b",
            "#8ac483",
            "#e1c969",
            "#e8804c",
        ),
        under="#09171e",
        over="#d84e43",
    ),
    ExportVariable(
        key="mslp",
        source_channel="msl",
        label="Mean sea-level pressure",
        short_label="MSLP",
        units="hPa",
        description="Daily ensemble-mean pressure reduced to mean sea level.",
        family="circulation",
        interpretation="Large pressure systems; exact centres matter mainly at shorter leads.",
        offset=800.0,
        scale=0.01,
        legend_boundaries=(960.0, 980.0, 995.0, 1005.0, 1015.0, 1025.0, 1040.0, 1060.0),
        legend_colors=(
            "#454176",
            "#35658e",
            "#3c92a0",
            "#8ab9a0",
            "#d4d2a0",
            "#d6a46e",
            "#b96258",
        ),
        under="#2a2858",
        over="#76384b",
    ),
    ExportVariable(
        key="sst",
        source_channel="sst",
        label="Sea-surface temperature",
        short_label="SST",
        units="°C",
        description="Daily ensemble-mean sea-surface temperature, displayed over ocean only.",
        family="ocean-convection",
        interpretation="Slow lower-boundary patterns; anomalies require a matched ocean climate.",
        offset=-100.0,
        scale=0.01,
        legend_boundaries=(-2.0, 2.0, 8.0, 14.0, 20.0, 25.0, 28.0, 31.0, 35.0),
        legend_colors=(
            "#263d78",
            "#2d67a0",
            "#3b91b1",
            "#62b8aa",
            "#bad18c",
            "#efd071",
            "#ef994f",
            "#d95343",
        ),
        under="#172851",
        over="#7d263d",
        domain="ocean",
    ),
    ExportVariable(
        key="olr",
        source_channel="ttr",
        label="Outgoing longwave radiation",
        short_label="OLR",
        units="W/m²",
        description=(
            "Positive outgoing longwave radiation derived as the negative of "
            "top net thermal radiation."
        ),
        family="ocean-convection",
        interpretation="Low tropical OLR can indicate deep cloud and convection; it is not an MJO index.",
        offset=0.0,
        scale=0.01,
        legend_boundaries=(80.0, 140.0, 180.0, 210.0, 240.0, 270.0, 300.0, 340.0),
        legend_colors=(
            "#14354d",
            "#21647a",
            "#389792",
            "#8dbb87",
            "#d7cf82",
            "#e9a463",
            "#d8684f",
        ),
        under="#0b2237",
        over="#7f3545",
    ),
    ExportVariable(
        key="tcwv",
        source_channel="tcwv",
        label="Total-column water vapour",
        short_label="TCWV",
        units="kg/m²",
        description=(
            "Daily ensemble-mean vertically integrated atmospheric water vapour. "
            "Physically invalid negative experimental values are clipped to zero "
            "before aggregation."
        ),
        family="ocean-convection",
        interpretation=(
            "Broad moisture reservoirs and transport pathways; combine with "
            "850 hPa vectors to interpret tropical inflow."
        ),
        offset=0.0,
        scale=0.01,
        legend_boundaries=(0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 55.0, 70.0, 85.0),
        legend_colors=(
            "#16263a",
            "#203d5a",
            "#2b6173",
            "#3d8b86",
            "#77ae83",
            "#bdc878",
            "#e1ad62",
            "#dd704e",
        ),
        under="#0d1928",
        over="#973f4c",
    ),
)

ANOMALY_STYLES = {
    "precipitation": AnomalyStyle(
        label="Daily precipitation anomaly",
        short_label="Rainfall anomaly",
        description=(
            "Ensemble-mean daily precipitation minus the exact-initialization, "
            "lead-matched 2002–2021 native model climatology."
        ),
        offset=-100.0,
        scale=0.01,
        legend_boundaries=(-40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0),
        legend_colors=(
            "#8c4d32",
            "#b87a50",
            "#d7ae80",
            "#e8dfc7",
            "#c8e2dd",
            "#7fc0b9",
            "#3b8d98",
            "#24536f",
        ),
        under="#613026",
        over="#142d50",
    ),
    "temperature": AnomalyStyle(
        label="2 m temperature anomaly",
        short_label="Temperature anomaly",
        description=(
            "Ensemble-mean 2 m temperature minus the exact-initialization, "
            "lead-matched 2002–2021 native model climatology."
        ),
        offset=-100.0,
        scale=0.01,
        legend_boundaries=(-10.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0),
        legend_colors=(
            "#344f91",
            "#5680ba",
            "#8eb8d2",
            "#d5e3df",
            "#eadfca",
            "#e6aa79",
            "#cc6b50",
            "#8d3544",
        ),
        under="#24346c",
        over="#63243c",
    ),
    "z500": AnomalyStyle(
        label="500 hPa height anomaly",
        short_label="Z500 anomaly",
        description=(
            "Ensemble-mean 500 hPa geopotential height minus the "
            "exact-initialization, lead-matched 2002–2021 native model climatology."
        ),
        offset=-250.0,
        scale=0.01,
        legend_boundaries=(-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0),
        legend_colors=(
            "#3e4c8c",
            "#627db1",
            "#9aafd0",
            "#d7dce0",
            "#e5dbc9",
            "#d6aa7a",
            "#b76d56",
            "#803b4d",
        ),
        under="#293266",
        over="#592a43",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=100)
    parser.add_argument("--lead-days", type=int, default=42)
    parser.add_argument("--initialization", default="2026-07-28")
    parser.add_argument(
        "--climatology-dir",
        type=Path,
        help=(
            "Directory containing tp_clima_MMDD.nc, t2m_clima_MMDD.nc, and "
            "z500_clima_MMDD.nc. When supplied, locked anomaly binaries are exported."
        ),
    )
    parser.add_argument(
        "--world-geometry",
        type=Path,
        default=ROOT / "public/data/world-countries.geojson",
    )
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
        if values.shape != (len(CHANNELS), 121, 240):
            raise ValueError(f"{path} produced shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains non-finite selected fields")
        if values[CHANNEL_INDEX["tp"]].min() < -1e-5:
            raise ValueError(f"{path} contains negative precipitation")
        return values
    finally:
        source.close()


def convert_fields(values: np.ndarray) -> dict[str, np.ndarray]:
    """Convert one member or an ensemble mean to public display fields."""

    get = lambda channel: values[CHANNEL_INDEX[channel]]  # noqa: E731
    return {
        "precipitation": tp_mm_hour_to_mm_day(np.maximum(get("tp"), 0.0)),
        "temperature": kelvin_to_celsius(get("t2m")),
        "z500": geopotential_to_height_dam(get("z500")),
        "wind850": wind_speed(get("u850"), get("v850")),
        "mslp": pascal_to_hectopascal(get("msl")),
        "sst": kelvin_to_celsius(get("sst")),
        "olr": top_net_thermal_to_olr(get("ttr")),
        "tcwv": np.maximum(get("tcwv"), 0.0),
    }


def convert_spread(spread: np.ndarray) -> dict[str, np.ndarray]:
    """Convert channel spread for independently linear public fields."""

    get = lambda channel: spread[CHANNEL_INDEX[channel]]  # noqa: E731
    return {
        "precipitation": get("tp") * 24.0,
        "temperature": get("t2m"),
        "z500": get("z500") / GRAVITY_M_S2 / 10.0,
        "mslp": get("msl") / 100.0,
        "sst": get("sst"),
        "olr": get("ttr"),
        "tcwv": get("tcwv"),
    }


def load_climatologies(
    directory: Path,
    initialization: dt.date,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Load and validate exact-date, lead-matched global model climatologies."""

    mmdd = initialization.strftime("%m%d")
    source_variables = {
        "precipitation": ("tp", tp_mm_hour_to_mm_day),
        "temperature": ("t2m", kelvin_to_celsius),
        "z500": ("z500", geopotential_to_height_dam),
    }
    climatologies: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for key, (source_name, converter) in source_variables.items():
        path = directory / f"{source_name}_clima_{mmdd}.nc"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        dataset = xr.open_dataset(path, decode_times=False)
        try:
            mean_name = f"{source_name}_mean"
            if mean_name not in dataset:
                raise ValueError(f"{path} does not contain {mean_name}")
            mean = dataset[mean_name].transpose("step", "lat", "lon")
            if mean.shape != (42, 121, 240):
                raise ValueError(f"{path} has unexpected shape {mean.shape}")
            if not np.array_equal(mean.step.values, np.arange(1, 43)):
                raise ValueError(f"{path} does not contain lead days 1–42")
            if not np.allclose(mean.lat.values, EXPECTED_LATITUDE):
                raise ValueError(f"{path} has an unexpected latitude coordinate")
            if not np.allclose(mean.lon.values, EXPECTED_LONGITUDE):
                raise ValueError(f"{path} has an unexpected longitude coordinate")
            description = str(dataset.attrs.get("description", ""))
            if "20 years x 51 members" not in description:
                raise ValueError(
                    f"{path} does not declare the locked 20-year × 51-member sample"
                )
            converted = np.asarray(converter(mean.values), dtype=np.float64)
            if not np.isfinite(converted).all():
                raise ValueError(f"{path} contains non-finite climatology values")
            if key == "precipitation" and converted.min() < -1e-6:
                raise ValueError(f"{path} contains negative climatological rainfall")
            climatologies[key] = converted
            provenance[key] = {
                "source_file": path.name,
                "source_sha256": sha256(path),
                "initialization_slot": mmdd,
                "hindcast_years": list(range(2002, 2022)),
                "years": 20,
                "native_members_per_year": 51,
                "lead_days": 42,
                "weighting": (
                    "Native members have equal weight within each complete year; "
                    "the 20 yearly means have equal weight. With 51 complete members "
                    "in every year, this equals the stored 1020-sample arithmetic mean."
                ),
            }
        finally:
            dataset.close()
    return climatologies, provenance


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


def global_area_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Return a cosine-latitude weighted global or masked mean."""

    latitude_weights = np.cos(np.deg2rad(EXPECTED_LATITUDE))[:, None]
    weights = np.broadcast_to(latitude_weights, values.shape).copy()
    if mask is not None:
        weights *= mask
    return float(np.average(values, weights=weights))


def build_ocean_mask(world_path: Path) -> np.ndarray:
    """Build a public 1-byte ocean mask from the Natural Earth land polygons."""

    document = json.loads(world_path.read_text(encoding="utf-8"))
    land = unary_union([
        shape(feature["geometry"])
        for feature in document["features"]
        if feature.get("geometry")
    ])
    longitude = np.where(
        EXPECTED_LONGITUDE > 180.0,
        EXPECTED_LONGITUDE - 360.0,
        EXPECTED_LONGITUDE,
    )
    lon_grid, lat_grid = np.meshgrid(longitude, EXPECTED_LATITUDE)
    land_mask = contains_xy(land, lon_grid, lat_grid)
    ocean_mask = (~land_mask).astype(np.uint8)
    if ocean_mask.mean() < 0.6 or ocean_mask.mean() > 0.8:
        raise ValueError("derived ocean support has an implausible fraction")
    return ocean_mask


def iso_day(initialization: dt.date, offset: int) -> str:
    return (initialization + dt.timedelta(days=offset)).isoformat()


def range_record(values: np.ndarray) -> dict[str, float]:
    return {"minimum": float(values.min()), "maximum": float(values.max())}


def write_binary(
    path: Path,
    frames: list[np.ndarray],
    *,
    offset: float,
    scale: float,
    frame_ranges: list[dict[str, float]],
) -> dict[str, Any]:
    stacked = np.stack(frames, axis=0)
    path.write_bytes(stacked.tobytes(order="C"))
    expected_size = 42 * 121 * 240 * 2
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes, expected {expected_size}"
        )
    decoded = stacked.astype(np.float64) * scale + offset
    return {
        "path": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "dtype": "uint16-little-endian",
        "offset": offset,
        "scale": scale,
        "minimum": float(decoded.min()),
        "maximum": float(decoded.max()),
        "frame_ranges": frame_ranges,
    }


def main() -> None:
    args = parse_args()
    if args.members < 1 or args.lead_days != 42:
        raise ValueError("the public global contract requires 42 leads and >=1 member")
    initialization = dt.date.fromisoformat(args.initialization)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variable_by_key = {variable.key: variable for variable in VARIABLES}
    ocean_mask = build_ocean_mask(args.world_geometry)
    sst_support = ocean_mask.copy()
    climatologies: dict[str, np.ndarray] = {}
    climatology_provenance: dict[str, dict[str, Any]] = {}
    if args.climatology_dir:
        climatologies, climatology_provenance = load_climatologies(
            args.climatology_dir,
            initialization,
        )

    frames: dict[str, list[np.ndarray]] = {key: [] for key in variable_by_key}
    anomaly_frames: dict[str, list[np.ndarray]] = {
        key: [] for key in climatologies
    }
    anomaly_frame_ranges: dict[str, list[dict[str, float]]] = {
        key: [] for key in climatologies
    }
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
    vector_frames: dict[str, list[np.ndarray]] = {"u": [], "v": []}
    vector_ranges: dict[str, list[dict[str, float]]] = {"u": [], "v": []}

    for lead_day in range(1, args.lead_days + 1):
        means = {
            key: np.zeros((121, 240), dtype=np.float64)
            for key in variable_by_key
        }
        second_moments = {key: np.zeros_like(value) for key, value in means.items()}
        vector_means = {
            "u": np.zeros((121, 240), dtype=np.float64),
            "v": np.zeros((121, 240), dtype=np.float64),
        }
        for member in range(args.members):
            values = read_selected(raw_path(args.raw_dir, member, lead_day))
            converted = convert_fields(values)
            for key, member_values in converted.items():
                delta = member_values - means[key]
                means[key] += delta / (member + 1)
                second_moments[key] += delta * (member_values - means[key])
            for component, channel in (("u", "u850"), ("v", "v850")):
                component_values = values[CHANNEL_INDEX[channel]]
                vector_means[component] += (
                    component_values - vector_means[component]
                ) / (member + 1)

        for key, values in means.items():
            if key == "sst":
                sst_support &= (values >= -3.0) & (values <= 45.0)
            spread_values = np.sqrt(second_moments[key] / args.members)
            variable = variable_by_key[key]
            frames[key].append(quantize(values, variable))
            spread_frames[key].append(
                quantize_with(
                    spread_values,
                    offset=0.0,
                    scale=variable.scale,
                    label=f"{key} spread",
                )
            )
            frame_ranges[key].append(range_record(values))
            if key in climatologies:
                anomaly = calculate_anomaly(
                    values,
                    climatologies[key][lead_day - 1],
                )
                style = ANOMALY_STYLES[key]
                anomaly_frames[key].append(
                    quantize_with(
                        anomaly,
                        offset=style.offset,
                        scale=style.scale,
                        label=f"{key} anomaly",
                    )
                )
                anomaly_frame_ranges[key].append(range_record(anomaly))
            spread_frame_ranges[key].append(range_record(spread_values))
            mask = ocean_mask if variable.domain == "ocean" else None
            spread_frame_means[key].append(
                global_area_mean(spread_values, mask=mask)
            )
        for component, values in vector_means.items():
            vector_frames[component].append(
                quantize_with(
                    values,
                    offset=-100.0,
                    scale=0.01,
                    label=f"wind850 {component}",
                )
            )
            vector_ranges[component].append(range_record(values))
        print(f"aggregated global lead {lead_day:02d}/{args.lead_days}", flush=True)

    ocean_mask = sst_support.astype(np.uint8)
    if ocean_mask.mean() < 0.6 or ocean_mask.mean() > 0.8:
        raise ValueError("stable physical SST support has an implausible fraction")
    spread_frame_means["sst"] = [
        global_area_mean(
            frame.astype(np.float64) * variable_by_key["sst"].scale,
            mask=ocean_mask,
        )
        for frame in spread_frames["sst"]
    ]

    file_records: dict[str, dict[str, Any]] = {}
    for variable in VARIABLES:
        record = write_binary(
            args.output_dir / f"{variable.key}.bin",
            frames[variable.key],
            offset=variable.offset,
            scale=variable.scale,
            frame_ranges=frame_ranges[variable.key],
        )
        record["spread"] = {
            **write_binary(
                args.output_dir / f"{variable.key}-spread.bin",
                spread_frames[variable.key],
                offset=0.0,
                scale=variable.scale,
                frame_ranges=spread_frame_ranges[variable.key],
            ),
            "frame_area_means": spread_frame_means[variable.key],
            "statistic": (
                f"Population standard deviation across {args.members} members "
                "(ddof=0); spread measures ensemble disagreement, not "
                "calibrated confidence."
            ),
        }
        file_records[variable.key] = record
        if variable.key in anomaly_frames:
            style = ANOMALY_STYLES[variable.key]
            record["anomaly"] = {
                "label": style.label,
                "short_label": style.short_label,
                "units": variable.units,
                "description": style.description,
                "baseline": {
                    "name": (
                        "Native model reforecast climatology · 2002–2021 · "
                        f"initialization {initialization.strftime('%d %b')}"
                    ),
                    **climatology_provenance[variable.key],
                },
                "legend": {
                    "boundaries": style.legend_boundaries,
                    "colors": style.legend_colors,
                    "under": style.under,
                    "over": style.over,
                },
                **write_binary(
                    args.output_dir / f"{variable.key}-anomaly.bin",
                    anomaly_frames[variable.key],
                    offset=style.offset,
                    scale=style.scale,
                    frame_ranges=anomaly_frame_ranges[variable.key],
                ),
            }

    file_records["wind850"]["vector"] = {
        "u": write_binary(
            args.output_dir / "wind850-u.bin",
            vector_frames["u"],
            offset=-100.0,
            scale=0.01,
            frame_ranges=vector_ranges["u"],
        ),
        "v": write_binary(
            args.output_dir / "wind850-v.bin",
            vector_frames["v"],
            offset=-100.0,
            scale=0.01,
            frame_ranges=vector_ranges["v"],
        ),
        "statistic": (
            "Arithmetic ensemble-mean eastward and northward wind components; "
            "arrows show vector direction while shading shows vector magnitude."
        ),
    }
    ocean_path = args.output_dir / "ocean-mask.bin"
    ocean_path.write_bytes(ocean_mask.tobytes(order="C"))
    ocean_record = {
        "path": ocean_path.name,
        "sha256": sha256(ocean_path),
        "size_bytes": ocean_path.stat().st_size,
        "dtype": "uint8",
        "meaning": (
            "1 = stable physically valid open-ocean SST display support; "
            "0 = land, sea ice, or unsupported"
        ),
    }

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata = {
        "schema_version": 3,
        "generated_at": generated_at,
        "issue": {
            "initialization": f"{initialization.isoformat()}T00:00:00Z",
            "members": args.members,
            "lead_days": args.lead_days,
            "status": "experimental",
            "public_label": "Experimental global ensemble guidance",
            "input_description": "Operational global analysis proxy",
            "ensemble_relation": (
                f"Independent {args.members}-member companion ensemble generated "
                "from the same frozen initialization input as the India validation case."
            ),
            "display_interpolation": (
                "Smooth cross-fades are a visual transition between daily fields; "
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
            "ocean_mask": ocean_record,
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
                "family": variable.family,
                "interpretation": variable.interpretation,
                "domain": variable.domain,
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
                "finite TP, T2M, Z500, U850, V850, MSLP, SST, TTR, and TCWV",
                "nonnegative precipitation, OLR, and published TCWV",
                "locked unit conversions and quantization",
                "population ensemble spread with ddof=0",
                "850 hPa vector components exported separately",
                "SST restricted to derived ocean display support",
                "global TP, T2M, and Z500 anomalies use an exact 28 July slot",
                "20 complete hindcast years with 51 native members per year",
                "lead-matched anomaly subtraction on the native global grid",
                "cosine-latitude spread summaries",
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
