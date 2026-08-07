#!/usr/bin/env python3
"""Build compact public fields from the validated 28-Jul-2026 FuXi sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import shapefile
import xarray as xr
PROJ_DATA = Path(sys.prefix) / "share" / "proj"
if PROJ_DATA.is_dir():
    os.environ.setdefault("PROJ_DATA", str(PROJ_DATA))
import pyproj
if PROJ_DATA.is_dir():
    pyproj.datadir.set_data_dir(str(PROJ_DATA))
from pyproj import CRS, Transformer
from shapely import contains_xy
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

INITIAL_CONDITION_SOURCES: dict[str, dict[str, str]] = {
    "gfs": {
        "label": "GFS operational proxy",
        "short_label": "GFS",
        "category": "operational_proxy",
        "availability": "near_real_time",
        "description": (
            "Near-real-time operational analyses and short-range forecasts "
            "aggregated into the model's two daily input states. Experimental."
        ),
        "scientific_status": (
            "Experimental GFS-proxy initialization; no matched GFS-initialized "
            "hindcast calibration is available."
        ),
    },
    "era5": {
        "label": "ERA5 reanalysis reference",
        "short_label": "ERA5",
        "category": "reanalysis_reference",
        "availability": "delayed_reference",
        "description": (
            "Delayed ERA5 reanalysis daily means used as a scientifically "
            "matched reference initialization; not a real-time forecast feed."
        ),
        "scientific_status": (
            "ERA5 reference initialization matched to the native model "
            "reforecast system; research reference, not near-real-time guidance."
        ),
    },
}

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
        "--source-id",
        choices=sorted(INITIAL_CONDITION_SOURCES),
        help="Initialization source; inferred from input metadata when omitted.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Validated run manifest used to verify status and forecast checksum.",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Write only the source-aware issue and download package.",
    )
    parser.add_argument(
        "--validation", type=Path, default=ROOT / "public/data/validation.json"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_id_for(input_source: str, requested: str | None) -> str:
    """Resolve the declared initialization source without silent guessing."""

    inferred = "gfs" if "gfs" in input_source.lower() else "era5"
    if requested is not None and requested != inferred:
        raise ValueError(
            f"declared source {requested!r} conflicts with forecast metadata {input_source!r}"
        )
    return requested or inferred


def validate_run_manifest(path: Path | None, forecast: Path, members: int, leads: int) -> None:
    """Verify a forecast against its private validated run manifest."""

    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in {"generated_valid", "existing_valid"}:
        raise ValueError(f"run manifest is not publishable: {path}")
    if int(payload.get("members", -1)) != members:
        raise ValueError("run manifest member count does not match forecast")
    if int(payload.get("lead_days", -1)) != leads:
        raise ValueError("run manifest lead count does not match forecast")
    if payload.get("output_sha256") != sha256(forecast):
        raise ValueError("run manifest output checksum does not match forecast")


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
    """Calculate six-week fields solely through the locked formulas.

    The native 2002–2021 climatology currently covers JJAS initialization
    slots only.  Forecasts outside that window remain publishable, but only as
    raw rainfall totals and mean temperature; anomaly fields are withheld.
    """

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
        model_state_day = forecast.model_state_time.values.astype("datetime64[D]")
        target_month_day = np.datetime_as_string(model_state_day, unit="D")[5:].replace("-", "")

    fields = {
        "rainfall_total": ensemble_mean(rain_total_members, member_axis=0),
        "temperature_mean": ensemble_mean(
            temperature_mean_members, member_axis=0
        ),
    }
    alignment: dict[str, Any]
    years: np.ndarray
    with xr.open_dataset(climatology_path) as climatology:
        try:
            left, right, right_weight = climatology_bracket(
                climatology.init_slot.values, target_month_day
            )
        except ValueError:
            alignment = {
                "status": "unavailable_outside_jjas",
                "target_model_state_calendar_day": target_month_day,
                "available_slot_start": str(
                    min(str(value) for value in climatology.init_slot.values)
                ),
                "available_slot_end": str(
                    max(str(value) for value in climatology.init_slot.values)
                ),
                "message": (
                    "Anomalies are withheld because the validated native "
                    "2002–2021 climatology does not cover this calendar day."
                ),
            }
            years = np.asarray([], dtype=int)
        else:
            rain_left = climatology.tp_ensemble_mean.sel(init_slot=left).values
            rain_right = climatology.tp_ensemble_mean.sel(init_slot=right).values
            temperature_left = climatology.t2m_ensemble_mean.sel(init_slot=left).values
            temperature_right = climatology.t2m_ensemble_mean.sel(init_slot=right).values
            years = climatology.hindcast_year.values.astype(int)
            rain_history = calendar_interpolation(
                rain_left, rain_right, right_weight
            )
            temperature_history = calendar_interpolation(
                temperature_left, temperature_right, right_weight
            )
            rain_climatology = ensemble_mean(
                weekly_mean_rainfall(rain_history, day_axis=1), member_axis=0
            )
            temperature_climatology = ensemble_mean(
                weekly_mean(
                    kelvin_to_celsius(temperature_history), day_axis=1
                ),
                member_axis=0,
            )
            fields["rainfall_anomaly"] = anomaly(
                ensemble_mean(rain_mean_members, member_axis=0),
                rain_climatology,
            )
            fields["temperature_anomaly"] = anomaly(
                ensemble_mean(temperature_mean_members, member_axis=0),
                temperature_climatology,
            )
            alignment = {
                "status": "available",
                "target_model_state_calendar_day": target_month_day,
                "left_slot": left,
                "right_slot": right,
                "right_weight": right_weight,
                "operation_order": (
                    "interpolate each yearly native-ensemble mean, aggregate "
                    "seven lead days, then equally average 20 years"
                ),
            }
    diagnostics = {
        "alignment": alignment,
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
    with xr.open_dataset(args.forecast) as source:
        initialization_day = source.forecast_reference_time.values.astype("datetime64[D]")
        model_state_day = source.model_state_time.values.astype("datetime64[D]")
        issue_id = np.datetime_as_string(initialization_day, unit="D").replace("-", "")
        initialization = np.datetime_as_string(initialization_day, unit="D") + "T00:00:00Z"
        model_state_time = np.datetime_as_string(model_state_day, unit="D") + "T00:00:00Z"
        input_days = [
            np.datetime_as_string(model_state_day - np.timedelta64(1, "D"), unit="D"),
            np.datetime_as_string(model_state_day, unit="D"),
        ]
        members = int(source.sizes["member"])
        lead_days = int(source.sizes["lead_day"])
        input_source = str(source.attrs.get("input_source", "Experimental operational proxy inputs"))
    source_id = source_id_for(input_source, args.source_id)
    source_definition = INITIAL_CONDITION_SOURCES[source_id]
    validate_run_manifest(args.run_manifest, args.forecast, members, lead_days)
    relative_forecast_path = Path("data/forecasts") / source_id / f"{issue_id}.json"
    products = {
        key: json.loads(json.dumps(PRODUCTS[key]))
        for key in fields
    }
    for product in products.values():
        product["description"] = product["description"].replace(
            "100-member", f"{members}-member"
        )
    forecast_payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "issue": {
            "initialization": initialization,
            "information_cutoff": initialization,
            "model_state_time": model_state_time,
            "input_days": input_days,
            "input_source": input_source,
            "members": members,
            "lead_days": lead_days,
            "status": status,
            "scientific_status": source_definition["scientific_status"],
            "initial_condition_source": {
                key: source_definition[key]
                for key in (
                    "label",
                    "short_label",
                    "category",
                    "availability",
                    "description",
                )
            }
            | {"id": source_id},
            "downloads": {
                "compact_json": relative_forecast_path.as_posix(),
            },
            "available_products": list(products),
            "capabilities": {
                "raw_fields": True,
                "anomalies": {
                    "rainfall_anomaly",
                    "temperature_anomaly",
                }.issubset(products),
                "regional_probabilities_eligible": (
                    members == 100
                    and {
                        "rainfall_anomaly",
                        "temperature_anomaly",
                    }.issubset(products)
                ),
            },
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
        "products": products,
        "diagnostics": diagnostics["forecast_population_spread"],
        "weeks": weeks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = args.output_dir / "forecasts" / source_id / f"{issue_id}.json"
    write_json(forecast_path, forecast_payload)
    if source_id == "gfs" and issue_id == "20260728":
        write_json(args.output_dir / "forecasts" / "20260728.json", forecast_payload)
    if args.forecast_only:
        print(
            f"wrote {forecast_path} with {len(products)} products; "
            f"India support contains {int(india_mask.sum())} native-grid cells"
        )
        return 0
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
    print(
        f"wrote six weeks × {len(products)} products; "
        f"India support contains {int(india_mask.sum())} native-grid cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
