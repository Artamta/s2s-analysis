#!/usr/bin/env python3
"""Build a matched GFS-versus-ERA5 initialization-sensitivity comparison.

This is not a forecast-skill score. It quantifies how two 100-member FuXi
ensembles differ when the issue date and model are fixed but the initial-state
source changes. Skill can only be scored after observations are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
DATA = PUBLIC / "data"


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
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def private_forecast_path(public_forecast: dict[str, Any]) -> Path:
    source = public_forecast["issue"]["initial_condition_source"]["id"]
    issue = public_forecast["issue"]["initialization"][:10].replace("-", "")
    members = int(public_forecast["issue"]["members"])
    return Path(
        "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
        f"operational/{source}/{issue}/ens{members}/forecasts/annual{issue[:4]}/{issue}.nc"
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    selected = np.isfinite(values) & (weights > 0)
    if not selected.any():
        raise ValueError("comparison has no supported finite values")
    return float(np.sum(values[selected] * weights[selected]) / np.sum(weights[selected]))


def weighted_pattern_correlation(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> float:
    selected = np.isfinite(left) & np.isfinite(right) & (weights > 0)
    x = left[selected]
    y = right[selected]
    w = weights[selected]
    w = w / w.sum()
    x = x - np.sum(w * x)
    y = y - np.sum(w * y)
    denominator = np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))
    return float(np.sum(w * x * y) / denominator) if denominator > 0 else 0.0


def weekly_fields(path: Path) -> dict[str, Any]:
    with xr.open_dataset(path) as dataset:
        members = int(dataset.sizes.get("member", 0))
        if members not in {5, 100}:
            raise ValueError(f"{path}: expected 5 or 100 members, found {members}")
        required_sizes = {"lead_day": 42, "latitude": 27, "longitude": 27}
        for dimension, expected in required_sizes.items():
            if dataset.sizes.get(dimension) != expected:
                raise ValueError(
                    f"{path}: {dimension} expected {expected}, found "
                    f"{dataset.sizes.get(dimension)}"
                )
        rain = dataset.tp.values.astype(np.float64) * 24.0
        temperature = dataset.t2m.values.astype(np.float64) - 273.15
        rain = rain.reshape(members, 6, 7, 27, 27).mean(axis=2)
        temperature = temperature.reshape(members, 6, 7, 27, 27).mean(axis=2)
        return {
            "rain_members": rain,
            "temperature_members": temperature,
            "rain_mean": rain.mean(axis=0),
            "temperature_mean": temperature.mean(axis=0),
            "rain_spread": rain.std(axis=0, ddof=0),
            "temperature_spread": temperature.std(axis=0, ddof=0),
            "latitude": dataset.latitude.values.astype(np.float64),
            "longitude": dataset.longitude.values.astype(np.float64),
            "valid_start": dataset.forecast_period_start.values[::7].astype("datetime64[D]"),
            "valid_end": dataset.forecast_period_end.values[6::7].astype("datetime64[D]")
            - np.timedelta64(1, "D"),
            "forecast_sha256": sha256(path),
            "members": members,
        }


def build(
    date: str,
    gfs_forecast: Path | None = None,
    era5_forecast: Path | None = None,
) -> Path:
    public_paths = {
        source: DATA / "forecasts" / source / f"{date}.json"
        for source in ("gfs", "era5")
    }
    public_forecasts = {
        source: json.loads(path.read_text(encoding="utf-8"))
        for source, path in public_paths.items()
    }
    private_paths = {
        "gfs": gfs_forecast or private_forecast_path(public_forecasts["gfs"]),
        "era5": era5_forecast or private_forecast_path(public_forecasts["era5"]),
    }
    fields = {source: weekly_fields(path) for source, path in private_paths.items()}
    if not np.array_equal(fields["gfs"]["latitude"], fields["era5"]["latitude"]):
        raise ValueError("GFS and ERA5 latitude grids differ")
    if not np.array_equal(fields["gfs"]["longitude"], fields["era5"]["longitude"]):
        raise ValueError("GFS and ERA5 longitude grids differ")
    if not np.array_equal(fields["gfs"]["valid_start"], fields["era5"]["valid_start"]):
        raise ValueError("GFS and ERA5 forecast periods differ")

    mask = np.asarray(public_forecasts["gfs"]["grid"]["india_mask"], dtype=bool).reshape(27, 27)
    latitude = fields["gfs"]["latitude"]
    weights = np.cos(np.deg2rad(latitude))[:, None] * mask
    weeks = []
    for week in range(6):
        rain_difference = fields["gfs"]["rain_mean"][week] - fields["era5"]["rain_mean"][week]
        temperature_difference = (
            fields["gfs"]["temperature_mean"][week]
            - fields["era5"]["temperature_mean"][week]
        )
        weeks.append(
            {
                "week": week + 1,
                "valid_start": str(fields["gfs"]["valid_start"][week]),
                "valid_end": str(fields["gfs"]["valid_end"][week]),
                "rainfall": {
                    "gfs_minus_era5_india_mean_mm_day": round(
                        weighted_mean(rain_difference, weights), 4
                    ),
                    "mean_absolute_difference_mm_day": round(
                        weighted_mean(np.abs(rain_difference), weights), 4
                    ),
                    "ensemble_mean_pattern_correlation": round(
                        weighted_pattern_correlation(
                            fields["gfs"]["rain_mean"][week],
                            fields["era5"]["rain_mean"][week],
                            weights,
                        ),
                        4,
                    ),
                    "gfs_mean_spread_mm_day": round(
                        weighted_mean(fields["gfs"]["rain_spread"][week], weights), 4
                    ),
                    "era5_mean_spread_mm_day": round(
                        weighted_mean(fields["era5"]["rain_spread"][week], weights), 4
                    ),
                },
                "temperature": {
                    "gfs_minus_era5_india_mean_deg_c": round(
                        weighted_mean(temperature_difference, weights), 4
                    ),
                    "mean_absolute_difference_deg_c": round(
                        weighted_mean(np.abs(temperature_difference), weights), 4
                    ),
                    "ensemble_mean_pattern_correlation": round(
                        weighted_pattern_correlation(
                            fields["gfs"]["temperature_mean"][week],
                            fields["era5"]["temperature_mean"][week],
                            weights,
                        ),
                        4,
                    ),
                    "gfs_mean_spread_deg_c": round(
                        weighted_mean(
                            fields["gfs"]["temperature_spread"][week], weights
                        ),
                        4,
                    ),
                    "era5_mean_spread_deg_c": round(
                        weighted_mean(
                            fields["era5"]["temperature_spread"][week], weights
                        ),
                        4,
                    ),
                },
            }
        )

    output = DATA / "comparisons" / f"{date}.json"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "issue_date": f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z",
        "comparison_type": "matched_initialization_source_sensitivity",
        "skill_status": "not_yet_verifiable",
        "interpretation": (
            "GFS-minus-ERA5 differences measure sensitivity to the initial-state "
            "source. They do not show which forecast is better; skill requires "
            "observations for the complete valid period."
        ),
        "sources": {
            source: {
                "members": fields[source]["members"],
                "forecast_sha256": fields[source]["forecast_sha256"],
            }
            for source in ("gfs", "era5")
        },
        "weeks": weeks,
    }
    write_json(output, payload)

    week_one = weeks[0]
    for source, forecast_path in public_paths.items():
        forecast = public_forecasts[source]
        forecast["issue"]["initialization_comparison"] = {
            "status": "initialization_sensitivity_only",
            "counterpart_source_id": "era5" if source == "gfs" else "gfs",
            "comparison": output.relative_to(DATA).as_posix(),
            "week1_rainfall_gfs_minus_era5_mm_day": week_one["rainfall"][
                "gfs_minus_era5_india_mean_mm_day"
            ],
            "week1_temperature_gfs_minus_era5_deg_c": week_one["temperature"][
                "gfs_minus_era5_india_mean_deg_c"
            ],
            "message": payload["interpretation"],
        }
        write_json(forecast_path, forecast)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Issue date YYYYMMDD")
    parser.add_argument("--gfs-forecast", type=Path)
    parser.add_argument("--era5-forecast", type=Path)
    args = parser.parse_args()
    if len(args.date) != 8 or not args.date.isdigit():
        raise SystemExit("date must use YYYYMMDD")
    print(build(args.date, args.gfs_forecast, args.era5_forecast))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
