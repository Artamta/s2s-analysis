from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


COMMON_LAT = np.arange(39.0, -0.01, -1.5, dtype=np.float64)
COMMON_LON = np.arange(60.0, 99.01, 1.5, dtype=np.float64)


@dataclass(frozen=True)
class StandardField:
    model: str
    experiment_id: str
    variable: str
    initialization: str
    values: np.ndarray
    member: np.ndarray
    lead_day: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    units: str
    temporal_statistic: str
    distribution_representation: str
    source_paths: tuple[str, ...]
    source_ensemble_size: int | None = None
    pressure_hpa: np.ndarray | None = None
    attrs: dict[str, Any] | None = None

    def validate(self) -> None:
        expected = 5 if self.pressure_hpa is not None else 4
        if self.values.ndim != expected:
            raise ValueError(f"{self.model}/{self.variable}: expected {expected} dimensions")
        if self.values.shape[0] != len(self.member):
            raise ValueError("member coordinate does not match values")
        if self.values.shape[1] != len(self.lead_day):
            raise ValueError("lead coordinate does not match values")
        if self.values.shape[-2:] != (len(self.latitude), len(self.longitude)):
            raise ValueError("spatial coordinates do not match values")
        if not np.all(np.diff(self.lead_day) == 1):
            raise ValueError("lead days must be consecutive")
        if len(np.unique(self.member)) != len(self.member):
            raise ValueError("member IDs must be unique")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def daily_to_weekly(values: np.ndarray, variable: str) -> dict[str, np.ndarray]:
    """Aggregate complete non-overlapping seven-day blocks only."""
    nweek = values.shape[1] // 7
    if nweek == 0:
        shape = (values.shape[0], 0) + values.shape[2:]
        return {"weekly_mean": np.empty(shape, dtype=np.float32)}
    complete = values[:, : nweek * 7]
    reshaped = complete.reshape((values.shape[0], nweek, 7) + values.shape[2:])
    result = {"weekly_mean": np.mean(reshaped, axis=2, dtype=np.float64).astype(np.float32)}
    if variable == "tp":
        result["weekly_total"] = np.sum(reshaped, axis=2, dtype=np.float64).astype(np.float32)
    return result


def ensemble_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    count = finite.sum(axis=0).astype(np.int16)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(values, axis=0, dtype=np.float64).astype(np.float32)
        std = np.nanstd(values, axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    return mean, std, count


def field_to_dataset(field: StandardField, grid_id: str) -> xr.Dataset:
    field.validate()
    dims = ["member", "lead_day"]
    coords: dict[str, Any] = {
        "member": field.member.astype(np.int16),
        "lead_day": field.lead_day.astype(np.int16),
        "latitude": field.latitude.astype(np.float64),
        "longitude": field.longitude.astype(np.float64),
        "init": np.array([np.datetime64(field.initialization, "ns")]),
    }
    if field.pressure_hpa is not None:
        dims.append("pressure_hpa")
        coords["pressure_hpa"] = field.pressure_hpa.astype(np.float32)
    dims.extend(["latitude", "longitude"])
    data = field.values[np.newaxis, ...].astype(np.float32)
    daily_dims = ["init", *dims]
    mean, std, count = ensemble_statistics(field.values)
    stat_dims = ["init", *dims[1:]]
    ds = xr.Dataset(
        {
            "forecast": (daily_dims, data),
            "ensemble_mean": (stat_dims, mean[np.newaxis, ...]),
            "ensemble_std": (stat_dims, std[np.newaxis, ...]),
            "ensemble_member_count": (stat_dims, count[np.newaxis, ...]),
            "member_available": (
                ("init", "member"),
                np.any(np.isfinite(field.values), axis=tuple(range(1, field.values.ndim)))[np.newaxis, :],
            ),
        },
        coords=coords,
    )
    if field.source_ensemble_size is not None:
        if field.source_ensemble_size < 1:
            raise ValueError("source ensemble size must be positive")
        ds["source_ensemble_size"] = (
            ("init",), np.array([field.source_ensemble_size], dtype=np.int16)
        )
        ds["source_ensemble_size"].attrs.update(
            units="1",
            long_name="number of source forecasts represented by a provider-precomputed mean",
        )
    weekly = daily_to_weekly(field.values, field.variable)
    if weekly["weekly_mean"].shape[1]:
        ds = ds.assign_coords(lead_week=np.arange(1, weekly["weekly_mean"].shape[1] + 1, dtype=np.int16))
        weekly_dims = ["init", "member", "lead_week", *dims[2:]]
        ds["forecast_weekly_mean"] = (weekly_dims, weekly["weekly_mean"][np.newaxis, ...])
        wmean, wstd, wcount = ensemble_statistics(weekly["weekly_mean"])
        weekly_stat_dims = ["init", "lead_week", *dims[2:]]
        ds["ensemble_mean_weekly"] = (weekly_stat_dims, wmean[np.newaxis, ...])
        ds["ensemble_std_weekly"] = (weekly_stat_dims, wstd[np.newaxis, ...])
        ds["ensemble_member_count_weekly"] = (weekly_stat_dims, wcount[np.newaxis, ...])
        if "weekly_total" in weekly:
            ds["forecast_weekly_total"] = (weekly_dims, weekly["weekly_total"][np.newaxis, ...])
            ds["ensemble_mean_weekly_total"] = (
                weekly_stat_dims,
                ensemble_statistics(weekly["weekly_total"])[0][np.newaxis, ...],
            )
    valid_end = np.datetime64(field.initialization, "D") + field.lead_day.astype("timedelta64[D]")
    valid_start = valid_end - np.timedelta64(1, "D")
    ds = ds.assign_coords(
        valid_time=("lead_day", valid_end.astype("datetime64[ns]")),
        forecast_period_start=("lead_day", valid_start.astype("datetime64[ns]")),
        forecast_period_end=("lead_day", valid_end.astype("datetime64[ns]")),
    )
    daily_attrs = {"units": field.units, "temporal_statistic": field.temporal_statistic}
    for name in ("forecast", "ensemble_mean", "ensemble_std"):
        ds[name].attrs.update(daily_attrs)
    ds["ensemble_member_count"].attrs.update(
        units="1", long_name="number of finite ensemble members contributing at each point"
    )
    ds["member_available"].attrs.update(
        units="1", long_name="member has at least one finite value for this initialization"
    )
    if "forecast_weekly_mean" in ds:
        weekly_mean_attrs = {
            "units": field.units,
            "temporal_statistic": "mean_of_complete_7_day_block",
        }
        for name in ("forecast_weekly_mean", "ensemble_mean_weekly", "ensemble_std_weekly"):
            ds[name].attrs.update(weekly_mean_attrs)
        ds["ensemble_member_count_weekly"].attrs.update(
            units="1", long_name="number of finite ensemble members contributing at each point"
        )
    if "forecast_weekly_total" in ds:
        total_attrs = {
            "units": "mm",
            "temporal_statistic": "sum_of_complete_7_day_block",
        }
        ds["forecast_weekly_total"].attrs.update(total_attrs)
        ds["ensemble_mean_weekly_total"].attrs.update(total_attrs)
    ds.attrs.update(
        schema_version=1,
        archive_id="india_s2s_benchmark_v1",
        model=field.model,
        experiment_id=field.experiment_id,
        variable=field.variable,
        grid_id=grid_id,
        distribution_representation=field.distribution_representation,
        ensemble_std_ddof=0,
        source_paths=json.dumps(list(field.source_paths)),
        preprocessing_policy="preserve suspicious finite values and record QC; never clip",
    )
    if field.attrs:
        ds.attrs.update(
            {k: v for k, v in field.attrs.items() if v is not None and k != "remap_audit"}
        )
    return ds


def qc_summary(field: StandardField) -> dict[str, Any]:
    values = field.values
    finite = np.isfinite(values)
    return {
        "model": field.model,
        "experiment_id": field.experiment_id,
        "variable": field.variable,
        "initialization": field.initialization,
        "shape": list(values.shape),
        "members": int(values.shape[0]),
        "lead_days": int(values.shape[1]),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "minimum": float(np.nanmin(values)),
        "maximum": float(np.nanmax(values)),
        "negative_count": int(np.sum(values < 0)) if field.variable == "tp" else 0,
        "source_ensemble_size": field.source_ensemble_size,
        "preprocessing_attrs": field.attrs or {},
        "status": "passed" if finite.any() else "failed",
    }
