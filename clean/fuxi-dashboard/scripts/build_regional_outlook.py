#!/usr/bin/env python3
"""Build compact IMD-region summaries and raw ensemble tercile probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from shapely import contains_xy, make_valid
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from science.formulas import (  # noqa: E402
    area_weights,
    calendar_interpolation,
    climatology_terciles,
    kelvin_to_celsius,
    probability_above_normal,
    probability_below_normal,
    probability_near_normal,
    tp_mm_hour_to_mm_day,
)
from scripts.validate_sources import DEFAULT_FUXI_CLIMO  # noqa: E402


REGION_STATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "northwest": (
        "Northwest India",
        (
            "JAMMU AND KASHMIR",
            "LADAKH",
            "HIMACHAL PRADESH",
            "CHANDIGARH",
            "DELHI",
            "HARYANA",
            "PUNJAB",
            "UTTAR PRADESH",
            "RAJASTHAN",
            "UTTARAKHAND",
        ),
    ),
    "central": (
        "Central India",
        (
            "CHHATTISGARH",
            "ODISHA",
            "GUJARAT",
            "MADHYA PRADESH",
            "MAHARASHTRA",
            "GOA",
            "DADRA & NAGAR HAVELI & DAMAN & DIU",
        ),
    ),
    "south_peninsula": (
        "South Peninsula",
        (
            "KARNATAKA",
            "ANDHRA PRADESH",
            "KERALA",
            "PUDUCHERRY",
            "TAMIL NADU",
            "TELANGANA",
            "LAKSHADWEEP",
        ),
    ),
    "east_northeast": (
        "East and Northeast India",
        (
            "ARUNACHAL PRADESH",
            "ASSAM",
            "MANIPUR",
            "MEGHALAYA",
            "MIZORAM",
            "NAGALAND",
            "SIKKIM",
            "WEST BENGAL",
            "BIHAR",
            "JHARKHAND",
            "TRIPURA",
            "ANDAMAN & NICOBAR",
        ),
    ),
}

REGION_ORDER = ("all_india", *REGION_STATES)
REGION_SHORT_LABELS = {
    "all_india": "All India",
    "northwest": "Northwest",
    "central": "Central",
    "south_peninsula": "South Peninsula",
    "east_northeast": "East & Northeast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--public-forecast", type=Path, required=True)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_FUXI_CLIMO)
    parser.add_argument(
        "--india-admin",
        type=Path,
        default=ROOT / "public/data/india-admin.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def weekly_mean(values: np.ndarray) -> np.ndarray:
    if values.shape[1] != 42:
        raise ValueError("weekly aggregation requires exactly 42 lead days")
    return values.reshape(values.shape[0], 6, 7, *values.shape[2:]).mean(axis=2)


def probability_fields(
    forecast_members: np.ndarray,
    yearly_climatology: np.ndarray,
) -> dict[str, np.ndarray]:
    lower, upper = climatology_terciles(yearly_climatology, year_axis=0)
    fields = {
        "below_normal": probability_below_normal(
            forecast_members, lower, member_axis=0
        ),
        "near_normal": probability_near_normal(
            forecast_members, lower, upper, member_axis=0
        ),
        "above_normal": probability_above_normal(
            forecast_members, upper, member_axis=0
        ),
    }
    np.testing.assert_allclose(
        fields["below_normal"] + fields["near_normal"] + fields["above_normal"],
        100.0,
        rtol=0.0,
        atol=1e-12,
    )
    return fields


def region_geometries(admin: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    features = {
        feature["properties"]["name"]: make_valid(shape(feature["geometry"]))
        for feature in admin["features"]
    }
    declared_states = {
        state_name
        for _, states in REGION_STATES.values()
        for state_name in states
    }
    missing = declared_states - features.keys()
    if missing:
        raise ValueError(f"India state geometry is missing {sorted(missing)}")
    regions = {
        region_id: make_valid(
            unary_union([features[state_name] for state_name in states])
        )
        for region_id, (_, states) in REGION_STATES.items()
    }
    regions["all_india"] = make_valid(unary_union(list(regions.values())))
    excluded = sorted(features.keys() - declared_states)
    if not excluded or not all(name.startswith("DISPUTED") for name in excluded):
        raise ValueError(f"unexpected unassigned India features: {excluded}")
    return regions, excluded


def regional_land_fractions(
    geometry: Any,
    latitude: np.ndarray,
    longitude: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Estimate polygon coverage with deterministic 20-by-20 subcell sampling."""

    samples_per_axis = 20
    fractions = np.zeros((latitude.size, longitude.size), dtype=np.float64)
    half = spacing / 2.0
    offset = spacing / samples_per_axis
    for row, lat in enumerate(latitude):
        for column, lon in enumerate(longitude):
            sample_longitude = np.linspace(
                lon - half + offset / 2.0,
                lon + half - offset / 2.0,
                samples_per_axis,
            )
            sample_latitude = np.linspace(
                lat - half + offset / 2.0,
                lat + half - offset / 2.0,
                samples_per_axis,
            )
            sample_lon_grid, sample_lat_grid = np.meshgrid(
                sample_longitude, sample_latitude
            )
            fractions[row, column] = float(
                contains_xy(geometry, sample_lon_grid, sample_lat_grid).mean()
            )
    return fractions


def spatial_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(values, weights, axes=((-2, -1), (0, 1)))


def rounded_probability(value: float) -> int:
    rounded = int(round(value))
    if not 0 <= rounded <= 100:
        raise ValueError("probability lies outside 0–100 percent")
    return rounded


def probability_record(probabilities: dict[str, np.ndarray]) -> dict[str, Any]:
    values = {
        category: rounded_probability(float(value))
        for category, value in probabilities.items()
    }
    maximum = max(values.values())
    winners = [category for category, value in values.items() if value == maximum]
    return {
        **values,
        "dominant_category": winners[0] if len(winners) == 1 else "mixed",
        "dominant_probability": maximum,
    }


def variable_region_record(
    forecast_members: np.ndarray,
    yearly_climatology: np.ndarray,
    *,
    mean_key: str,
    anomaly_key: str,
    spread_key: str,
) -> dict[str, Any]:
    probabilities = probability_fields(forecast_members, yearly_climatology)
    record = {
        mean_key: round(float(forecast_members.mean()), 2),
        anomaly_key: round(
            float(forecast_members.mean() - yearly_climatology.mean()), 2
        ),
        spread_key: round(float(forecast_members.std(ddof=0)), 2),
        "tercile_probability_percent": probability_record(probabilities),
    }
    return record


def load_fields(
    forecast_path: Path,
    climatology_path: Path,
    public_forecast: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    with xr.open_dataset(forecast_path) as forecast:
        required = {"member": 100, "lead_day": 42, "latitude": 27, "longitude": 27}
        for dimension, expected in required.items():
            if int(forecast.sizes.get(dimension, -1)) != expected:
                raise ValueError(
                    f"forecast {dimension} expected {expected}, found "
                    f"{forecast.sizes.get(dimension)}"
                )
        initialization = public_forecast["issue"]["initialization"][:10]
        if str(forecast.attrs.get("init_date")) != initialization:
            raise ValueError("private forecast and public issue dates differ")
        latitude = forecast.latitude.values.astype(np.float64)
        longitude = forecast.longitude.values.astype(np.float64)
        rain_members = weekly_mean(
            tp_mm_hour_to_mm_day(
                forecast.tp.transpose(
                    "member", "lead_day", "latitude", "longitude"
                ).values
            )
        )
        temperature_members = weekly_mean(
            kelvin_to_celsius(
                forecast.t2m.transpose(
                    "member", "lead_day", "latitude", "longitude"
                ).values
            )
        )

    alignment = public_forecast["issue"]["climatology_alignment"]
    left = alignment["left_slot"]
    right = alignment["right_slot"]
    right_weight = float(alignment["right_weight"])
    with xr.open_dataset(climatology_path) as climatology:
        available_slots = set(climatology.init_slot.values.astype(str))
        if left not in available_slots or right not in available_slots:
            raise ValueError("declared climatology slots are unavailable")
        if climatology.sizes.get("hindcast_year") != 20:
            raise ValueError("probabilities require exactly 20 hindcast years")
        rain_left = climatology.tp_ensemble_mean.sel(init_slot=left).transpose(
            "hindcast_year", "lead_day", "latitude", "longitude"
        ).values
        rain_right = climatology.tp_ensemble_mean.sel(init_slot=right).transpose(
            "hindcast_year", "lead_day", "latitude", "longitude"
        ).values
        temperature_left = climatology.t2m_ensemble_mean.sel(init_slot=left).transpose(
            "hindcast_year", "lead_day", "latitude", "longitude"
        ).values
        temperature_right = climatology.t2m_ensemble_mean.sel(init_slot=right).transpose(
            "hindcast_year", "lead_day", "latitude", "longitude"
        ).values
        hindcast_years = climatology.hindcast_year.values.astype(int)

    rain_history = weekly_mean(
        calendar_interpolation(rain_left, rain_right, right_weight)
    )
    temperature_history = weekly_mean(
        kelvin_to_celsius(
            calendar_interpolation(temperature_left, temperature_right, right_weight)
        )
    )
    return (
        latitude,
        longitude,
        rain_members,
        temperature_members,
        rain_history,
        temperature_history,
        hindcast_years,
    )


def build(args: argparse.Namespace) -> Path:
    public_forecast = load_json(args.public_forecast)
    issue = public_forecast["issue"]
    if int(issue["members"]) != 100:
        raise ValueError("regional probabilities are published only for 100-member issues")
    source_id = issue["initial_condition_source"]["id"]
    issue_id = issue["initialization"][:10].replace("-", "")
    output = args.output or (
        ROOT / "public/data/regional" / source_id / f"{issue_id}.json"
    )
    (
        latitude,
        longitude,
        rain_members,
        temperature_members,
        rain_history,
        temperature_history,
        hindcast_years,
    ) = load_fields(args.forecast, args.climatology, public_forecast)

    admin = load_json(args.india_admin)
    geometries, excluded_features = region_geometries(admin)
    spacing = float(public_forecast["grid"]["spacing_degrees"])
    fractions = {
        region_id: regional_land_fractions(
            geometries[region_id], latitude, longitude, spacing
        )
        for region_id in REGION_ORDER
    }
    weights = {
        region_id: area_weights(latitude, region_fractions)
        for region_id, region_fractions in fractions.items()
    }

    rain_grid_probabilities = probability_fields(rain_members, rain_history)
    temperature_grid_probabilities = probability_fields(
        temperature_members, temperature_history
    )
    weeks: list[dict[str, Any]] = []
    for week_index, source_week in enumerate(public_forecast["weeks"]):
        region_records: list[dict[str, Any]] = []
        for region_id in REGION_ORDER:
            region_weights = weights[region_id]
            rain_forecast_region = spatial_mean(
                rain_members[:, week_index], region_weights
            )
            rain_history_region = spatial_mean(
                rain_history[:, week_index], region_weights
            )
            temperature_forecast_region = spatial_mean(
                temperature_members[:, week_index], region_weights
            )
            temperature_history_region = spatial_mean(
                temperature_history[:, week_index], region_weights
            )
            region_records.append(
                {
                    "id": region_id,
                    "label": (
                        "All India"
                        if region_id == "all_india"
                        else REGION_STATES[region_id][0]
                    ),
                    "short_label": REGION_SHORT_LABELS[region_id],
                    "rainfall": variable_region_record(
                        rain_forecast_region,
                        rain_history_region,
                        mean_key="weekly_mean_mm_day",
                        anomaly_key="anomaly_mm_day",
                        spread_key="ensemble_spread_mm_day",
                    ),
                    "temperature": variable_region_record(
                        temperature_forecast_region,
                        temperature_history_region,
                        mean_key="weekly_mean_deg_c",
                        anomaly_key="anomaly_deg_c",
                        spread_key="ensemble_spread_deg_c",
                    ),
                }
            )
        weeks.append(
            {
                "week": source_week["week"],
                "valid_start": source_week["valid_start"],
                "valid_end": source_week["valid_end"],
                "probability_fields": {
                    "rainfall": {
                        category: np.rint(values[week_index])
                        .astype(np.uint8)
                        .ravel()
                        .tolist()
                        for category, values in rain_grid_probabilities.items()
                    },
                    "temperature": {
                        category: np.rint(values[week_index])
                        .astype(np.uint8)
                        .ravel()
                        .tolist()
                        for category, values in temperature_grid_probabilities.items()
                    },
                },
                "regions": region_records,
            }
        )

    region_definitions = []
    for region_id in REGION_ORDER:
        region_definitions.append(
            {
                "id": region_id,
                "label": (
                    "All India"
                    if region_id == "all_india"
                    else REGION_STATES[region_id][0]
                ),
                "short_label": REGION_SHORT_LABELS[region_id],
                "states_and_union_territories": (
                    [] if region_id == "all_india" else list(REGION_STATES[region_id][1])
                ),
                "equivalent_native_grid_cells": round(float(fractions[region_id].sum()), 2),
            }
        )

    calibration = (
        "uncalibrated_gfs_proxy"
        if source_id == "gfs"
        else "raw_ensemble_reanalysis_reference"
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "issue": {
            "initialization": issue["initialization"],
            "source_id": source_id,
            "source_label": issue["initial_condition_source"]["label"],
            "members": 100,
            "lead_days": 42,
            "status": "experimental",
            "probability_type": "raw_ensemble_tercile_probability",
            "calibration": calibration,
            "forecast_sha256": sha256(args.forecast),
            "climatology_sha256": sha256(args.climatology),
            "climatology_alignment": issue["climatology_alignment"],
            "hindcast_years": hindcast_years.tolist(),
        },
        "grid": {
            "shape": [27, 27],
            "spacing_degrees": spacing,
            "latitude": latitude.tolist(),
            "longitude": longitude.tolist(),
            "value_order": "latitude-major row order; latitude north-to-south",
        },
        "region_definition": {
            "name": "IMD four broad homogeneous rainfall regions",
            "geometry_source": admin["source"],
            "geometry_sha256": sha256(args.india_admin),
            "aggregation": (
                "cos(latitude) multiplied by state/UT polygon coverage estimated "
                "from a deterministic 20-by-20 subcell mask within each native grid cell"
            ),
            "reference": "https://mausam.imd.gov.in/imd_latest/contents/rainfall_over_homogeneous.php",
            "interpretation": (
                "The four-region grouping follows IMD reporting. Display masks are "
                "derived from Survey of India state/UT geometry, not an IMD subdivision shapefile."
            ),
            "excluded_geometry_features": excluded_features,
            "regions": region_definitions,
        },
        "probability_definition": {
            "below_normal": (
                "percentage of 100 forecast members strictly below the lower tercile"
            ),
            "near_normal": (
                "percentage of members at or between the two terciles"
            ),
            "above_normal": (
                "percentage of members strictly above the upper tercile"
            ),
            "terciles": (
                "1/3 and 2/3 quantiles across 20 equally weighted yearly native-ensemble means"
            ),
            "warning": (
                "Raw ensemble guidance, not a calibrated operational probability or warning."
            ),
        },
        "weeks": weeks,
    }
    write_json(output, payload)
    print(f"wrote {output} ({len(weeks)} weeks, {len(REGION_ORDER)} regions)")
    return output


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
