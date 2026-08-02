"""Validation gates for private scientific inputs and public web artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import xarray as xr

from .contracts import (
    BASELINES,
    FUXI_CLIMATOLOGY,
    FUXI_FORECAST,
    IMD_CLIMATOLOGY,
    IMERG_CLIMATOLOGY,
)
from .formulas import calendar_interpolation, climatology_spread

Status = Literal["green", "warning", "failure"]


@dataclass(frozen=True)
class ValidationCheck:
    """One public-safe validation result."""

    id: str
    label: str
    group: str
    status: Status
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def utc_now() -> str:
    """Return a timezone-aware ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combine_status(checks: Iterable[ValidationCheck]) -> Status:
    """Return failure over warning over green."""

    statuses = {check.status for check in checks}
    if "failure" in statuses:
        return "failure"
    if "warning" in statuses:
        return "warning"
    return "green"


def require_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate file existence, nonzero size, and SHA-256."""

    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("source file is empty")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {"size_bytes": size, "sha256": actual_sha256}


def _expected_coordinate(first: float, last: float, count: int) -> np.ndarray:
    return np.linspace(first, last, count, dtype=np.float64)


def validate_forecast(path: Path, expected_sha256: str) -> ValidationCheck:
    """Validate the complete FuXi forecast contract and physical ranges."""

    integrity = require_file(path, expected_sha256)
    contract = FUXI_FORECAST
    with xr.open_dataset(path) as dataset:
        expected_sizes = {
            "member": contract.members,
            "lead_day": contract.lead_days,
            "latitude": contract.grid.latitude_count,
            "longitude": contract.grid.longitude_count,
        }
        for name, expected in expected_sizes.items():
            if dataset.sizes.get(name) != expected:
                raise ValueError(
                    f"{name}={dataset.sizes.get(name)}, expected {expected}"
                )
        if not {"tp", "t2m"}.issubset(dataset.data_vars):
            raise ValueError("forecast lacks tp or t2m")
        if dataset.tp.attrs.get("units") != contract.tp_units:
            raise ValueError("TP unit contract failed")
        if dataset.t2m.attrs.get("units") != contract.t2m_units:
            raise ValueError("T2M unit contract failed")
        expected_latitude = _expected_coordinate(
            contract.grid.latitude_first,
            contract.grid.latitude_last,
            contract.grid.latitude_count,
        )
        expected_longitude = _expected_coordinate(
            contract.grid.longitude_first,
            contract.grid.longitude_last,
            contract.grid.longitude_count,
        )
        if not np.array_equal(dataset.latitude.values, expected_latitude):
            raise ValueError("latitude coordinate or orientation is incorrect")
        if not np.array_equal(dataset.longitude.values, expected_longitude):
            raise ValueError("longitude coordinate or orientation is incorrect")
        temporal_checks = {
            "forecast_reference_time": contract.initialization,
            "information_cutoff_time": contract.information_cutoff,
            "model_state_time": contract.model_state_time,
        }
        for coordinate, expected in temporal_checks.items():
            actual = np.datetime_as_string(
                dataset[coordinate].values.astype("datetime64[s]"), unit="s"
            )
            if actual != expected:
                raise ValueError(f"{coordinate}={actual}, expected {expected}")
        first_start = np.datetime_as_string(
            dataset.forecast_period_start.values[0].astype("datetime64[s]"), unit="s"
        )
        last_end = np.datetime_as_string(
            dataset.forecast_period_end.values[-1].astype("datetime64[s]"), unit="s"
        )
        if first_start != contract.period_start or last_end != contract.period_end_exclusive:
            raise ValueError("forecast-period bounds do not match the contract")
        tp = dataset.tp.values.astype(np.float64)
        t2m = dataset.t2m.values.astype(np.float64)
        if not np.isfinite(tp).all() or not np.isfinite(t2m).all():
            raise ValueError("forecast contains non-finite values")
        if tp.min() < 0.0 or tp.max() > 15.0:
            raise ValueError("forecast rainfall is outside the physical gate")
        if t2m.min() < 180.0 or t2m.max() > 350.0:
            raise ValueError("forecast temperature is outside the physical gate")
        tp_spread = tp.std(axis=0, ddof=0)
        t2m_spread = t2m.std(axis=0, ddof=0)
        if not np.any(tp_spread > 0.0) or not np.any(t2m_spread > 0.0):
            raise ValueError("forecast ensemble has no spread")
        details = {
            **integrity,
            "members": contract.members,
            "lead_days": contract.lead_days,
            "grid": "27 × 27 at 1.5°",
            "tp_range_mm_h-1": [float(tp.min()), float(tp.max())],
            "t2m_range_K": [float(t2m.min()), float(t2m.max())],
            "maximum_tp_population_spread_mm_h-1": float(tp_spread.max()),
            "maximum_t2m_population_spread_K": float(t2m_spread.max()),
            "initialization": contract.initialization + "Z",
            "information_cutoff": contract.information_cutoff + "Z",
        }
    return ValidationCheck(
        id="fuxi_forecast",
        label="FuXi forecast",
        group="source",
        status="green",
        summary="100 members, 42 lead days, timing, units, grid, physics, and spread validated.",
        details=details,
    )


def validate_fuxi_climatology(
    path: Path, expected_sha256: str
) -> ValidationCheck:
    """Validate FuXi sampling, grid, units, completeness, and annual weighting."""

    integrity = require_file(path, expected_sha256)
    with xr.open_dataset(path) as dataset:
        if dataset.sizes.get("hindcast_year") != FUXI_CLIMATOLOGY.sample_count:
            raise ValueError("FuXi climatology does not contain exactly 20 years")
        expected_years = np.arange(2002, 2022)
        if not np.array_equal(dataset.hindcast_year.values, expected_years):
            raise ValueError("FuXi climatology years are not exactly 2002-2021")
        if dataset.sizes.get("lead_day") != FUXI_FORECAST.lead_days:
            raise ValueError("FuXi climatology lead-day count is not 42")
        if (dataset.sizes.get("latitude"), dataset.sizes.get("longitude")) != (27, 27):
            raise ValueError("FuXi climatology grid shape does not match forecast")
        if not np.array_equal(
            dataset.latitude.values,
            _expected_coordinate(39.0, 0.0, 27),
        ) or not np.array_equal(
            dataset.longitude.values,
            _expected_coordinate(60.0, 99.0, 27),
        ):
            raise ValueError("FuXi climatology grid coordinates do not match forecast")
        required = {
            "tp_ensemble_mean": "mm day-1",
            "t2m_ensemble_mean": "K",
            "tp_model_climatology_mean": "mm day-1",
            "t2m_model_climatology_mean": "K",
        }
        for variable, units in required.items():
            if variable not in dataset:
                raise ValueError(f"missing {variable}")
            if dataset[variable].attrs.get("units") != units:
                raise ValueError(f"{variable} unit contract failed")
        if not np.all(dataset.hindcast_sample_count.values == 20):
            raise ValueError("one or more climatology slots is incomplete")
        slots = dataset.init_slot.values.astype(str)
        if not {"0725", "0728"}.issubset(slots):
            raise ValueError("required calendar interpolation slots are absent")
        tp_left = dataset.tp_ensemble_mean.sel(init_slot="0725").values.astype(
            np.float64
        )
        tp_right = dataset.tp_ensemble_mean.sel(init_slot="0728").values.astype(
            np.float64
        )
        t2m_left = dataset.t2m_ensemble_mean.sel(init_slot="0725").values.astype(
            np.float64
        )
        t2m_right = dataset.t2m_ensemble_mean.sel(init_slot="0728").values.astype(
            np.float64
        )
        for values, label in (
            (tp_left, "TP left slot"),
            (tp_right, "TP right slot"),
            (t2m_left, "T2M left slot"),
            (t2m_right, "T2M right slot"),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{label} contains incomplete fields")
        if min(tp_left.min(), tp_right.min()) < 0.0:
            raise ValueError("FuXi climatology contains negative rainfall")
        if min(t2m_left.min(), t2m_right.min()) < 180.0 or max(
            t2m_left.max(), t2m_right.max()
        ) > 350.0:
            raise ValueError("FuXi climatology temperature is implausible")
        interpolated = calendar_interpolation(tp_left, tp_right, 2.0 / 3.0)
        annual_equal_weight_mean = interpolated.mean(axis=0)
        if not np.isfinite(annual_equal_weight_mean).all():
            raise ValueError("equal-weight annual climatology is incomplete")
        selected_spread = climatology_spread(
            interpolated[:, 0, 13, 13], year_axis=0
        )
        if float(selected_spread) <= 0.0:
            raise ValueError("20-year climatology sample has no spread")
        details = {
            **integrity,
            "hindcast_years": expected_years.tolist(),
            "native_members_per_year": 51,
            "annual_weighting": "one native-member mean per year; 20 years equally weighted",
            "lead_days": 42,
            "available_init_slots": int(dataset.sizes["init_slot"]),
            "prototype_alignment": {
                "left_slot": "0725",
                "right_slot": "0728",
                "right_weight": 2.0 / 3.0,
            },
            "complete_sample_count_slots": int(
                np.count_nonzero(dataset.hindcast_sample_count.values == 20)
            ),
            "calendar_scope": "JJAS only",
        }
    return ValidationCheck(
        id="fuxi_climatology",
        label="FuXi climatology",
        group="source",
        status="warning",
        summary="Validated for this JJAS issue; full-year FuXi climatology is not yet available.",
        details=details,
    )


def validate_imd_climatology(path: Path, expected_sha256: str) -> ValidationCheck:
    """Validate the native IMD 1991-2020 daily rainfall climatology."""

    integrity = require_file(path, expected_sha256)
    with xr.open_dataset(path) as dataset:
        if dataset.attrs.get("baseline") != IMD_CLIMATOLOGY.baseline:
            raise ValueError("IMD baseline is not exactly 1991-2020")
        if dataset.attrs.get("units") != IMD_CLIMATOLOGY.units:
            raise ValueError("IMD unit contract failed")
        if dataset.sizes.get("day") != 365:
            raise ValueError("IMD climatology does not contain 365 calendar keys")
        month_day = dataset.month_day.values.astype(str)
        if len(set(month_day)) != 365 or "02-29" in set(month_day):
            raise ValueError("IMD calendar keys are incomplete or ambiguous")
        if not {"rain_mean", "rain_count"}.issubset(dataset.data_vars):
            raise ValueError("IMD climatology lacks rainfall or sample-count fields")
        count = dataset.rain_count.values
        support = count > 0
        if not np.all(support == support[0]):
            raise ValueError("IMD spatial support changes by calendar day")
        if not np.all(count[support] == 30):
            raise ValueError("IMD supported cells do not have exactly 30 samples")
        rainfall = dataset.rain_mean.values.astype(np.float64)
        if not np.isfinite(rainfall[support]).all() or np.any(rainfall[support] < 0.0):
            raise ValueError("IMD rainfall is non-finite or negative on support")
        details = {
            **integrity,
            "baseline": dataset.attrs["baseline"],
            "calendar_days": 365,
            "feb_29_rule": "removed from the 365-day climatology",
            "supported_native_cells": int(support[0].sum()),
            "samples_per_supported_cell": 30,
            "units": dataset.attrs["units"],
        }
    return ValidationCheck(
        id="imd_climatology",
        label="IMD climatology",
        group="source",
        status="green",
        summary="1991–2020 baseline, calendar, units, finite rainfall, and stable support validated.",
        details=details,
    )


def validate_imerg_climatology(
    path: Path, expected_sha256: str
) -> ValidationCheck:
    """Validate the fixed audited IMERG Final V07B climatology."""

    integrity = require_file(path, expected_sha256)
    with xr.open_dataset(path) as dataset:
        if dataset.attrs.get("baseline_years") != IMERG_CLIMATOLOGY.baseline:
            raise ValueError("IMERG baseline is not exactly 2001-2022")
        if int(dataset.attrs.get("baseline_year_count", -1)) != 22:
            raise ValueError("IMERG baseline does not contain exactly 22 years")
        if dataset.attrs.get("product") != "GPM_3IMERGDF.07":
            raise ValueError("IMERG product identity is not Final V07B")
        if dataset.sizes.get("calendar_day") != 142:
            raise ValueError("audited IMERG seasonal calendar must contain 142 keys")
        sample_count = dataset.daily_sample_count.values
        if not np.all(sample_count == 22):
            raise ValueError("IMERG does not have exactly 22 samples per calendar key")
        if (dataset.sizes.get("latitude"), dataset.sizes.get("longitude")) != (22, 22):
            raise ValueError("IMERG verification grid shape is incorrect")
        rainfall = dataset.daily_precipitation_climatology
        if rainfall.attrs.get("units") != IMERG_CLIMATOLOGY.units:
            raise ValueError("IMERG unit contract failed")
        values = rainfall.values.astype(np.float64)
        support = dataset.india_fraction.values.astype(np.float64) > 0.0
        supported_values = values[:, support]
        if not np.isfinite(supported_values).all() or np.any(supported_values < 0.0):
            raise ValueError("IMERG rainfall is non-finite or negative on support")
        if int(support.sum()) != 169:
            raise ValueError("IMERG frozen support is not 169 cells")
        if float(dataset.attrs.get("support_difference_fraction", 1.0)) >= 0.03:
            raise ValueError("IMERG conservative-remap support audit failed")
        month_days = set(dataset.calendar_month_day.values.astype(str))
        required = {
            (np.datetime64("2026-07-28") + np.timedelta64(day, "D")).astype(
                "datetime64[D]"
            )
            for day in range(42)
        }
        required_month_days = {
            str(day)[5:] for day in required
        }
        if not required_month_days.issubset(month_days):
            raise ValueError("IMERG climatology does not cover all prototype valid days")
        details = {
            **integrity,
            "baseline": dataset.attrs["baseline_years"],
            "samples_per_calendar_day": 22,
            "calendar_days": 142,
            "calendar_window": dataset.attrs["calendar_window"],
            "grid": dataset.attrs["grid"],
            "supported_cells": int(support.sum()),
            "remapping": dataset.attrs["remapping"],
            "observation_stream": "IMERG Late (separate from this Final climatology)",
        }
    return ValidationCheck(
        id="imerg_climatology",
        label="IMERG climatology",
        group="source",
        status="warning",
        summary="Final V07B 2001–2022 baseline is validated for all 42 valid days; the audited file is seasonal.",
        details=details,
    )


def validate_baseline_identity(source: str, baseline: str) -> None:
    """Prevent accidental exchange of FuXi, IMD, and IMERG baselines."""

    if source not in BASELINES:
        raise ValueError(f"unknown climatology source: {source}")
    if baseline != BASELINES[source]:
        raise ValueError(
            f"{source} requires baseline {BASELINES[source]}, received {baseline}"
        )


def contains_nonfinite_json(value: Any) -> bool:
    """Return whether a nested JSON-like value contains NaN or infinity."""

    if isinstance(value, float):
        return not np.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite_json(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_nonfinite_json(item) for item in value)
    return False


def write_json(path: Path, payload: Any) -> None:
    """Write strict, stable, human-readable JSON."""

    if contains_nonfinite_json(payload):
        raise ValueError(f"refusing to serialize non-finite JSON to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
