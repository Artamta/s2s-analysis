#!/usr/bin/env python3
"""Build compact public fields from the validated 28-Jul-2026 FuXi sources."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import shapefile
import xarray as xr
from pyproj import CRS, Transformer
from shapely import contains_xy
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from science.contracts import FUXI_FORECAST  # noqa: E402
from science.formulas import (  # noqa: E402
    FORMULA_DEFINITIONS,
    anomaly,
    area_weights,
    calendar_interpolation,
    ensemble_mean,
    forecast_spread,
    kelvin_to_celsius,
    tp_mm_hour_to_mm_day,
    weekly_mean,
    weekly_mean_rainfall,
    weekly_total,
    weighted_mean,
)
from science.validators import utc_now, write_json  # noqa: E402
from scripts.validate_sources import (  # noqa: E402
    DEFAULT_FORECAST,
    DEFAULT_FUXI_CLIMO,
)

DEFAULT_SHAPEFILE = Path(
    "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"
)

PRODUCTS: dict[str, dict[str, Any]] = {
    "rainfall_total": {
        "label": "Rainfall weekly total",
        "short_label": "Rainfall total",
        "description": "100-member ensemble-mean seven-day accumulation",
        "units": "mm / 7 days",
        "baseline": None,
        "legend": {
            "boundaries": [0, 10, 25, 50, 75, 100, 150, 200, 300, 450],
            "colors": [
                "#f7fbff",
                "#deebf7",
                "#c6dbef",
                "#9ecae1",
                "#6baed6",
                "#4292c6",
                "#2171b5",
                "#08519c",
                "#08306b",
            ],
            "under": "#ffffff",
            "over": "#041f4a",
        },
    },
    "rainfall_anomaly": {
        "label": "Rainfall anomaly",
        "short_label": "Rain anomaly",
        "description": "Weekly-mean forecast minus lead-matched FuXi model climatology",
        "units": "mm / day",
        "baseline": "FuXi-S2S native reforecasts, 2002–2021",
        "legend": {
            "boundaries": [-20, -15, -10, -5, -2, 2, 5, 10, 15, 20],
            "colors": [
                "#b33b13",
                "#e56b2f",
                "#f3a45b",
                "#fbd7a5",
                "#f6f3e8",
                "#c9ddec",
                "#8fb8d5",
                "#477fae",
                "#164a79",
            ],
            "under": "#7d2106",
            "over": "#072f55",
        },
    },
    "temperature_mean": {
        "label": "Weekly mean temperature",
        "short_label": "Temperature",
        "description": "100-member ensemble-mean 2-metre temperature",
        "units": "°C",
        "baseline": None,
        "legend": {
            "boundaries": [10, 14, 18, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42],
            "colors": [
                "#4d63a8",
                "#7186bf",
                "#9eacd2",
                "#d6d9df",
                "#f4edcf",
                "#f8dda0",
                "#f7c66c",
                "#f3a548",
                "#e87836",
                "#d24c2d",
                "#b52d2c",
                "#8f1d27",
                "#641522",
            ],
            "under": "#32447d",
            "over": "#430b18",
        },
    },
    "temperature_anomaly": {
        "label": "Temperature anomaly",
        "short_label": "Temp anomaly",
        "description": "Weekly-mean forecast minus lead-matched FuXi model climatology",
        "units": "°C",
        "baseline": "FuXi-S2S native reforecasts, 2002–2021",
        "legend": {
            "boundaries": [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
            "colors": [
                "#2d3e8f",
                "#4965ae",
                "#6f8bc5",
                "#9db1d7",
                "#c9d4e6",
                "#edf0ec",
                "#f7e8c3",
                "#f5c887",
                "#ed9d58",
                "#df6d3e",
                "#c84331",
                "#952526",
            ],
            "under": "#19265f",
            "over": "#651217",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST)
    parser.add_argument("--fuxi-climatology", type=Path, default=DEFAULT_FUXI_CLIMO)
    parser.add_argument("--india-shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "public/data")
    parser.add_argument(
        "--validation", type=Path, default=ROOT / "public/data/validation.json"
    )
    return parser.parse_args()


def climatology_bracket(
    slots: np.ndarray, target_month_day: str
) -> tuple[str, str, float]:
    """Return surrounding MMDD slots and the right-slot linear weight."""

    target_month = int(target_month_day[:2])
    target_day = int(target_month_day[2:])
    target = np.datetime64(f"2000-{target_month:02d}-{target_day:02d}")
    labels = sorted(str(value) for value in slots)
    dates = np.asarray(
        [np.datetime64(f"2000-{value[:2]}-{value[2:]}") for value in labels]
    )
    matches = np.flatnonzero(dates == target)
    if matches.size:
        label = labels[int(matches[0])]
        return label, label, 0.0
    insertion = int(np.searchsorted(dates, target))
    if insertion == 0 or insertion == len(dates):
        raise ValueError(f"{target_month_day} is outside available climatology slots")
    left_date, right_date = dates[insertion - 1], dates[insertion]
    right_weight = float((target - left_date) / (right_date - left_date))
    return labels[insertion - 1], labels[insertion], right_weight


def read_india_geometry(path: Path):
    """Read, reproject, dissolve, and lightly simplify India state boundaries."""

    required = [path, path.with_suffix(".shx"), path.with_suffix(".dbf"), path.with_suffix(".prj")]
    missing = [item for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete India boundary: {missing}")
    reader = shapefile.Reader(str(path))
    geometries = [shape(item.__geo_interface__) for item in reader.shapes()]
    source_crs = CRS.from_wkt(path.with_suffix(".prj").read_text(encoding="utf-8"))
    target_crs = CRS.from_epsg(4326)
    if source_crs != target_crs:
        transformer = Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
        geometries = [
            transform(transformer.transform, geometry) for geometry in geometries
        ]
    dissolved = unary_union(geometries)
    if not dissolved.is_valid:
        dissolved = dissolved.buffer(0)
    return dissolved.simplify(0.04, preserve_topology=True)


def rounded_geometry(value: Any) -> Any:
    """Round nested GeoJSON coordinates without altering object keys."""

    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, tuple):
        return [rounded_geometry(item) for item in value]
    if isinstance(value, list):
        return [rounded_geometry(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded_geometry(item) for key, item in value.items()}
    return value


def flatten_rounded(values: np.ndarray) -> list[float]:
    """Flatten a finite field to stable four-decimal public values."""

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("derived field contains non-finite values")
    return np.round(array, 4).ravel().tolist()


def field_summary(
    values: np.ndarray, india_mask: np.ndarray, latitude: np.ndarray
) -> dict[str, float]:
    """Return fixed-mask, area-weighted public summary statistics."""

    weights = area_weights(latitude, india_mask.astype(np.float64))
    supported = values[india_mask]
    return {
        "india_weighted_mean": round(weighted_mean(values, weights), 3),
        "india_minimum": round(float(supported.min()), 3),
        "india_maximum": round(float(supported.max()), 3),
    }


def build_fields(
    forecast_path: Path, climatology_path: Path
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.datetime64],
    list[np.datetime64],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Calculate six-week fields solely through the locked formulas."""

    with xr.open_dataset(forecast_path) as forecast:
        latitude = forecast.latitude.values.astype(np.float64)
        longitude = forecast.longitude.values.astype(np.float64)
        rain_daily_members = tp_mm_hour_to_mm_day(forecast.tp.values)
        temperature_daily_members = kelvin_to_celsius(forecast.t2m.values)
        rain_total_members = weekly_total(rain_daily_members, day_axis=1)
        rain_mean_members = weekly_mean_rainfall(
            rain_daily_members, day_axis=1
        )
        temperature_mean_members = weekly_mean(
            temperature_daily_members, day_axis=1
        )
        week_start = list(
            forecast.forecast_period_start.values[::7].astype("datetime64[D]")
        )
        week_end_exclusive = list(
            forecast.forecast_period_end.values[6::7].astype("datetime64[D]")
        )

    with xr.open_dataset(climatology_path) as climatology:
        left, right, right_weight = climatology_bracket(
            climatology.init_slot.values, "0727"
        )
        rain_left = climatology.tp_ensemble_mean.sel(init_slot=left).values
        rain_right = climatology.tp_ensemble_mean.sel(init_slot=right).values
        temperature_left = climatology.t2m_ensemble_mean.sel(init_slot=left).values
        temperature_right = climatology.t2m_ensemble_mean.sel(init_slot=right).values
        years = climatology.hindcast_year.values.astype(int)

    rain_history = calendar_interpolation(rain_left, rain_right, right_weight)
    temperature_history = calendar_interpolation(
        temperature_left, temperature_right, right_weight
    )
    rain_climatology = ensemble_mean(
        weekly_mean_rainfall(rain_history, day_axis=1), member_axis=0
    )
    temperature_climatology = ensemble_mean(
        weekly_mean(kelvin_to_celsius(temperature_history), day_axis=1),
        member_axis=0,
    )
    fields = {
        "rainfall_total": ensemble_mean(rain_total_members, member_axis=0),
        "rainfall_anomaly": anomaly(
            ensemble_mean(rain_mean_members, member_axis=0), rain_climatology
        ),
        "temperature_mean": ensemble_mean(
            temperature_mean_members, member_axis=0
        ),
        "temperature_anomaly": anomaly(
            ensemble_mean(temperature_mean_members, member_axis=0),
            temperature_climatology,
        ),
    }
    diagnostics = {
        "alignment": {
            "target_model_state_calendar_day": "0727",
            "left_slot": left,
            "right_slot": right,
            "right_weight": right_weight,
            "operation_order": (
                "interpolate each yearly native-ensemble mean, aggregate seven "
                "lead days, then equally average 20 years"
            ),
        },
        "hindcast_years": years.tolist(),
        "forecast_population_spread": {
            "rainfall_weekly_mean_max_mm_day": float(
                forecast_spread(rain_mean_members, member_axis=0).max()
            ),
            "temperature_weekly_mean_max_deg_c": float(
                forecast_spread(temperature_mean_members, member_axis=0).max()
            ),
        },
    }
    return (
        latitude,
        longitude,
        week_start,
        week_end_exclusive,
        fields,
        diagnostics,
    )


def date_text(value: np.datetime64) -> str:
    """Serialize a day-resolution NumPy datetime."""

    return np.datetime_as_string(value.astype("datetime64[D]"), unit="D")


def main() -> int:
    args = parse_args()
    if not args.validation.is_file():
        raise FileNotFoundError(
            f"{args.validation} is absent; run validate_sources.py first"
        )
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if not validation.get("presentation_allowed", False):
        raise RuntimeError("source validation failed; refusing to build public data")
    (
        latitude,
        longitude,
        week_start,
        week_end_exclusive,
        fields,
        diagnostics,
    ) = build_fields(args.forecast, args.fuxi_climatology)
    india = read_india_geometry(args.india_shapefile)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    india_mask = contains_xy(india, longitude_grid, latitude_grid)
    if int(india_mask.sum()) < 100:
        raise ValueError("derived India support unexpectedly contains too few cells")

    weeks = []
    for index in range(6):
        week_fields = {
            product: flatten_rounded(values[index])
            for product, values in fields.items()
        }
        summaries = {
            product: field_summary(values[index], india_mask, latitude)
            for product, values in fields.items()
        }
        inclusive_end = week_end_exclusive[index] - np.timedelta64(1, "D")
        weeks.append(
            {
                "week": index + 1,
                "valid_start": date_text(week_start[index]),
                "valid_end": date_text(inclusive_end),
                "fields": week_fields,
                "summary": summaries,
            }
        )

    status = (
        "warning"
        if validation["overall_status"] == "warning"
        else "green"
    )
    forecast_payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "issue": {
            "initialization": FUXI_FORECAST.initialization + "Z",
            "information_cutoff": FUXI_FORECAST.information_cutoff + "Z",
            "model_state_time": FUXI_FORECAST.model_state_time + "Z",
            "input_days": ["2026-07-26", "2026-07-27"],
            "input_source": (
                "NCEP operational GFS 1.0° analyses and 0–6 h forecasts, "
                "aggregated to two UTC daily proxy inputs"
            ),
            "members": FUXI_FORECAST.members,
            "lead_days": FUXI_FORECAST.lead_days,
            "status": status,
            "scientific_status": (
                "Experimental GFS-proxy initialization; no matched "
                "GFS-initialized FuXi hindcast calibration"
            ),
            "climatology_alignment": diagnostics["alignment"],
            "hindcast_years": diagnostics["hindcast_years"],
            "observation_verification": {
                "status": "not_available",
                "message": (
                    "Verification is withheld until every required observation "
                    "day in a valid week is available."
                ),
            },
        },
        "grid": {
            "shape": [27, 27],
            "spacing_degrees": 1.5,
            "latitude": latitude.tolist(),
            "longitude": longitude.tolist(),
            "india_mask": india_mask.ravel().tolist(),
            "supported_cell_count": int(india_mask.sum()),
            "value_order": "latitude-major row order; latitude is north-to-south",
        },
        "products": PRODUCTS,
        "diagnostics": diagnostics["forecast_population_spread"],
        "weeks": weeks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_dir = args.output_dir / "forecasts"
    write_json(forecast_dir / "20260728.json", forecast_payload)
    write_json(
        args.output_dir / "india-outline.json",
        {
            "schema_version": 1,
            "description": "Simplified display-only India outline derived from the validated state boundary.",
            "geometry": rounded_geometry(mapping(india)),
        },
    )
    source_catalog = json.loads(
        (ROOT / "science/sources.json").read_text(encoding="utf-8")
    )
    source_status = {
        item["id"]: item["status"]
        for item in validation["checks"]
        if item["group"] == "source"
    }
    source_id_for_check = {
        "fuxi_forecast_20260728": "fuxi_forecast",
        "fuxi_climatology_2002_2021": "fuxi_climatology",
        "imd_climatology_1991_2020": "imd_climatology",
        "imerg_final_climatology_2001_2022": "imerg_climatology",
    }
    for source in source_catalog["sources"]:
        source["validation_status"] = source_status[
            source_id_for_check[source["id"]]
        ]
    source_catalog["generated_at"] = utc_now()
    write_json(args.output_dir / "sources.json", source_catalog)
    write_json(
        args.output_dir / "formulas.json",
        {
            "schema_version": 1,
            "formula_version": "1.1.0",
            "generated_at": utc_now(),
            "definitions": FORMULA_DEFINITIONS,
            "statistics": {
                "ensemble_mean": "arithmetic mean across 100 members",
                "forecast_spread": "population standard deviation across members (ddof=0)",
                "climatology_spread": "sample standard deviation across 20 yearly means (ddof=1)",
                "terciles": "1/3 and 2/3 quantiles across 20 yearly ensemble means",
                "probabilities": "100 × qualifying members ÷ 100",
                "verification": (
                    "Bias, MAE, RMSE, and weighted spatial Pearson ACC use "
                    "identical valid periods, target grid, India support, and weights."
                ),
            },
            "baseline_separation": {
                "fuxi": "2002–2021 native reforecast model climatology",
                "imd": "1991–2020 native daily gauge climatology",
                "imerg": "Final V07B 2001–2022 fixed daily climatology",
            },
        },
    )
    write_json(
        args.output_dir / "index.json",
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "latest_successful_issue": "20260728",
            "available_issues": [
                {
                    "id": "20260728",
                    "initialization": "2026-07-28T00:00:00Z",
                    "status": status,
                    "forecast": "forecasts/20260728.json",
                }
            ],
        },
    )
    print(
        "wrote six weeks × four products; "
        f"India support contains {int(india_mask.sum())} native-grid cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
