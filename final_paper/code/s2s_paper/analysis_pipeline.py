"""Weekly S2S verification pipeline for the final-paper branch.

The public entry point is :func:`run_weekly_pipeline`. It keeps season-specific
I/O here, while metric formulas remain in :mod:`s2s_paper.metrics`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import dask.array as dask_array
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from .constants import VARIABLE_UNITS, WEEKLY_WINDOWS, LeadWindow
from .grid import GridSpec, cosine_weights, crop_india_box, to_grid
from .metrics import (
    bias,
    brier_score,
    brier_skill_score,
    crps_gaussian,
    crps_skill_score,
    gaussian_probability_exceedance,
    mae,
    probability_exceedance,
    reliability_bins,
    rmse,
    score_deterministic,
    spread_skill_ratio,
    weighted_mean,
)
from .paths import get_paths
from .regions import REGION_KEYS, open_region_masks


G = 9.80665
REGIONS = ("All India",) + tuple(REGION_KEYS)
TP_THRESHOLDS_MM_DAY = (1.0, 10.0)
SIGMA_FLOOR = {"tp": 0.05, "z500": 1.0, "t2m": 0.1}
CUMULATIVE_TP_MODELS = {"ecmwf", "ukmo", "ncep"}
SUPPORTED_MODELS = ("spire", "delysm", "ecmwf", "ukmo", "ncep", "fuxi")
DEFAULT_OUTPUT_ROOT = get_paths().outputs_dir / "s2s_paper_outputs"
TRUTH_SOURCE_CHOICES = ("auto", "era5", "imd")


def realize(field: xr.DataArray) -> xr.DataArray:
    """Materialize a small common-grid field once so metrics do not recompute I/O."""

    return field.compute() if hasattr(field.data, "compute") else field.load()


def valid_time_count(da: xr.DataArray) -> int:
    """Count time steps with at least one finite grid cell."""

    if "time" not in da.dims:
        return int(np.isfinite(da).any())
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    if not spatial_dims:
        count = np.isfinite(da).sum()
    else:
        count = da.notnull().any(dim=spatial_dims).sum()
    if hasattr(count.data, "compute"):
        count = count.compute()
    return int(count.item())


@dataclass(frozen=True)
class FieldWithMeta:
    field: xr.DataArray | None
    source: str
    count: int = 0
    notes: str = ""
    spread: xr.DataArray | None = None


@dataclass(frozen=True)
class ForecastSource:
    model: str
    variable: str
    kind: str
    source_paths: tuple[str, ...]
    notes: str = ""
    ensemble: xr.DataArray | None = None
    mean: xr.DataArray | None = None
    spread: xr.DataArray | None = None


@dataclass(frozen=True)
class AggregatedForecast:
    model: str
    variable: str
    kind: str
    mean: xr.DataArray
    spread: xr.DataArray | None
    ensemble: xr.DataArray | None
    member_count: int
    lead_count: int
    source_paths: tuple[str, ...]
    notes: str


class RegionContext:
    """Loaded region masks and area weights for one verification grid."""

    def __init__(self, dgrid: float):
        self.dgrid = dgrid
        self.masks = open_region_masks(dgrid).load()
        self.weights = cosine_weights(self.masks["lat"])

    def mask(self, field: xr.DataArray, region: str) -> xr.DataArray:
        return field.where(self.masks[region].astype(bool))


def canonical_variable(variable: str) -> str:
    key = variable.strip().lower()
    aliases = {
        "precip": "tp",
        "precipitation": "tp",
        "rain": "tp",
        "z": "z500",
        "gh": "z500",
        "gh500": "z500",
        "2t": "t2m",
        "tas": "t2m",
    }
    key = aliases.get(key, key)
    if key not in {"tp", "z500", "t2m"}:
        raise ValueError(f"unsupported variable: {variable!r}")
    return key


def canonical_truth_source(truth_source: str) -> str:
    key = truth_source.strip().lower()
    aliases = {
        "obs": "auto",
        "default": "auto",
        "reanalysis": "era5",
        "cds": "era5",
        "wb2": "era5",
        "weatherbench2": "era5",
        "rainfall": "imd",
    }
    key = aliases.get(key, key)
    if key not in TRUTH_SOURCE_CHOICES:
        raise ValueError(f"unsupported truth source {truth_source!r}; use one of {TRUTH_SOURCE_CHOICES}")
    return key


def variable_label(variable: str) -> str:
    return canonical_variable(variable).upper()


def valid_dates_for_window(init_date: str, window: LeadWindow) -> list[str]:
    init = pd.Timestamp(init_date)
    return [
        (init + pd.Timedelta(days=lead)).strftime("%Y-%m-%d")
        for lead in range(window.start, window.end + 1)
    ]


@lru_cache(maxsize=1)
def _open_forecast_module():
    """Load scripts/06_open_comparable_forecasts.py as an importable module."""

    path = get_paths().project_root / "scripts" / "06_open_comparable_forecasts.py"
    spec = importlib.util.spec_from_file_location("open_comparable_forecasts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import forecast opener from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_matched_init_dates(
    season: str,
    set_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[str]:
    path = output_root / season / "02_processed" / "matched_init" / f"{set_name}_init_dates.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"matched-init file not found: {path}. Run scripts/05_scan_forecast_inits.py first."
        )
    frame = pd.read_csv(path, dtype={"init_date": str})
    return frame["init_date"].dropna().astype(str).tolist()


def read_models_for_set(
    season: str,
    set_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[str]:
    path = output_root / "common" / "inventory" / "forecast_comparable_init_sets.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = frame[
        (frame["season"].astype(str) == season)
        & (frame["set_name"].astype(str) == set_name)
    ]
    if rows.empty:
        return []
    models = str(rows.iloc[0]["models"])
    return [m for m in models.split(";") if m and m != "nan"]


def spire_init_dates() -> list[str]:
    paths = get_paths()
    group = zarr.open_group(str(paths.jfm_spire_zarr), mode="r")["mean_stddev"]
    ref = np.asarray(group["reference_time"][:])
    if np.issubdtype(ref.dtype, np.datetime64):
        return [pd.Timestamp(value).strftime("%Y%m%d") for value in ref]
    units = dict(group["reference_time"].attrs).get("units", "days since 2026-01-01 00:00:00")
    base_text = units.split(" since ", 1)[1].split()[0]
    base = pd.Timestamp(base_text)
    return [(base + pd.Timedelta(days=int(day))).strftime("%Y%m%d") for day in ref]


def prepare_run_selection(
    season: str,
    set_name: str,
    models: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
    *,
    include_spire: bool = False,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_inits: int | None = None,
) -> tuple[list[str], list[str], list[str], str]:
    season = season.lower()
    if season not in {"jfm2026", "jjas2019"}:
        raise ValueError(f"unsupported season: {season}")

    init_dates = read_matched_init_dates(season, set_name, output_root)
    selected_models = [m.lower() for m in models] if models else read_models_for_set(season, set_name, output_root)
    selected_models = [m for m in selected_models if m in SUPPORTED_MODELS]
    if include_spire and season == "jfm2026" and "spire" not in selected_models:
        selected_models = ["spire"] + selected_models

    notes = []
    if "spire" in selected_models:
        if season != "jfm2026":
            selected_models = [m for m in selected_models if m != "spire"]
            notes.append("SPIRE removed because it is only available for JFM2026")
        else:
            spire_dates = set(spire_init_dates())
            before = len(init_dates)
            init_dates = [d for d in init_dates if d in spire_dates]
            notes.append(f"SPIRE selected: intersected init dates with daily SPIRE archive from {before} to {len(init_dates)}")

    if max_inits is not None:
        init_dates = init_dates[:max_inits]

    if variables is None:
        selected_variables = ["tp", "z500", "t2m"]
    else:
        selected_variables = [canonical_variable(v) for v in variables]

    return init_dates, selected_models, selected_variables, "; ".join(notes)


class VerificationData:
    """Truth and climatology access for JFM2026 and JJAS2019."""

    def __init__(self, grid: GridSpec, truth_source: str = "auto"):
        self.paths = get_paths()
        self.grid = grid
        self.truth_source = canonical_truth_source(truth_source)
        self._imd_2019: xr.Dataset | None = None
        self._imd_years: dict[int, xr.Dataset] = {}
        self._imd_clim: xr.Dataset | None = None
        self._era5_clim: xr.Dataset | None = None
        self._era5_prev_tp: xr.Dataset | None = None
        self._era5_prev_t2m: xr.Dataset | None = None
        self._wb2: xr.Dataset | None = None
        self._jfm_truth_cache: dict[tuple[str, str], xr.DataArray] = {}
        self._jfm_era5_daily_cache: dict[str, xr.Dataset] = {}

    @property
    def imd_2019(self) -> xr.Dataset:
        if self._imd_2019 is None:
            self._imd_2019 = xr.open_dataset(self.paths.jjas2019_imd_rainfall_nc)
        return self._imd_2019

    def imd_year(self, year: int) -> xr.Dataset:
        if year not in self._imd_years:
            path = self.paths.imd_rainfall_root / "netcdf" / f"imd_rain_{year}.nc"
            if not path.exists():
                raise FileNotFoundError(path)
            self._imd_years[year] = xr.open_dataset(path)
        return self._imd_years[year]

    @property
    def imd_clim(self) -> xr.Dataset:
        if self._imd_clim is None:
            self._imd_clim = xr.open_dataset(self.paths.imd_daily_climatology_nc)
        return self._imd_clim

    @property
    def era5_clim(self) -> xr.Dataset:
        if self._era5_clim is None:
            self._era5_clim = xr.open_dataset(self.paths.era5_climatology, chunks={"dayofyear": 1})
        return self._era5_clim

    @property
    def era5_prev_tp(self) -> xr.Dataset:
        if self._era5_prev_tp is None:
            self._era5_prev_tp = xr.open_dataset(self.paths.era5_prev_daily_tp_nc)
        return self._era5_prev_tp

    @property
    def era5_prev_t2m(self) -> xr.Dataset:
        if self._era5_prev_t2m is None:
            self._era5_prev_t2m = xr.open_dataset(self.paths.era5_prev_daily_t2m_nc)
        return self._era5_prev_t2m

    @property
    def wb2(self) -> xr.Dataset:
        if self._wb2 is None:
            self._wb2 = xr.open_zarr(self.paths.weatherbench2_era5_zarr, consolidated=False)
        return self._wb2

    def truth_period(self, season: str, variable: str, dates: Sequence[str]) -> FieldWithMeta:
        variable = canonical_variable(variable)
        if variable == "tp":
            if season == "jfm2026":
                monthly = self._truth_jfm_era5_daily("tp", dates)
                compact = self._truth_era5_prev_daily("tp", dates)
                return self._best_truth(monthly, compact, expected_count=len(dates))
            source = self.tp_truth_source(season)
            if source == "imd":
                return self._truth_imd_tp(dates)
            return self._truth_era5_tp(season, dates)
        if season == "jfm2026" and variable == "t2m":
            monthly = self._truth_jfm_era5_daily("t2m", dates)
            compact = self._truth_era5_prev_daily("t2m", dates)
            daily = self._truth_jfm("t2m", dates)
            return self._best_truth(monthly, compact, daily, expected_count=len(dates))
        if season == "jfm2026" and variable == "z500":
            monthly = self._truth_jfm_era5_daily("z500", dates)
            daily = self._truth_jfm(variable, dates)
            return self._best_truth(monthly, daily, expected_count=len(dates))
        if season == "jjas2019" and variable in {"z500", "t2m"}:
            return self._truth_wb2(variable, dates)
        raise ValueError(f"no truth reader for {season} {variable}")

    def _best_truth(self, *candidates: FieldWithMeta, expected_count: int) -> FieldWithMeta:
        usable = [candidate for candidate in candidates if candidate.field is not None and candidate.count > 0]
        if not usable:
            return candidates[0]
        for candidate in usable:
            if candidate.count >= expected_count:
                return candidate
        return max(usable, key=lambda candidate: candidate.count)

    def climatology_period(self, season: str, variable: str, dates: Sequence[str]) -> FieldWithMeta:
        variable = canonical_variable(variable)
        if variable == "tp":
            if self.tp_truth_source(season) == "era5":
                return self._clim_era5(variable, dates)
            return self._clim_imd_tp(dates)
        return self._clim_era5(variable, dates)

    def tp_truth_source(self, season: str) -> str:
        if self.truth_source != "auto":
            return self.truth_source
        return "imd" if season == "jjas2019" else "era5"

    def _truth_jfm_daily(self, variable: str, date: str) -> xr.DataArray | None:
        key = (variable, date)
        if key in self._jfm_truth_cache:
            return self._jfm_truth_cache[key]
        path = self.paths.jfm_ground_truth_root / f"{pd.Timestamp(date).strftime('%Y%m%d')}.nc"
        if not path.exists():
            return None
        ds = xr.open_dataset(path)
        if variable not in ds:
            ds.close()
            return None
        da = ds[variable].load()
        ds.close()
        da = crop_india_box(da)
        self._jfm_truth_cache[key] = da
        return da

    def _truth_jfm(self, variable: str, dates: Sequence[str]) -> FieldWithMeta:
        pieces = []
        missing = []
        for date in dates:
            da = self._truth_jfm_daily(variable, date)
            if da is None:
                missing.append(date)
            else:
                pieces.append(da)
        if not pieces:
            return FieldWithMeta(None, "ERA5_CDS_daily_jfm2026", 0, f"missing all dates: {missing}")
        field = xr.concat(pieces, dim="time").mean("time", skipna=True)
        return FieldWithMeta(
            realize(to_grid(field, self.grid)),
            "ERA5_CDS_daily_jfm2026",
            len(pieces),
            f"missing_dates={','.join(missing)}" if missing else "",
        )

    def _jfm_era5_daily_month(self, month: str) -> xr.Dataset | None:
        if month in self._jfm_era5_daily_cache:
            return self._jfm_era5_daily_cache[month]
        path = self.paths.era5_jfm2026_daily_dir / f"era5_daily_{month}.nc"
        if not path.exists():
            return None
        ds = xr.open_dataset(path)
        self._jfm_era5_daily_cache[month] = ds
        return ds

    def _truth_jfm_era5_daily(self, variable: str, dates: Sequence[str]) -> FieldWithMeta:
        wanted = pd.to_datetime(list(dates)).normalize()
        pieces = []
        missing_files = []
        for month, group in pd.Series(wanted).groupby(wanted.strftime("%Y%m")):
            ds = self._jfm_era5_daily_month(str(month))
            if ds is None or variable not in ds:
                missing_files.extend([pd.Timestamp(date).strftime("%Y-%m-%d") for date in group])
                continue
            pieces.append(ds[variable].reindex(time=list(group)))
        source = f"ERA5_ARCO_daily_jfm2026_{variable}"
        if not pieces:
            return FieldWithMeta(None, source, 0, f"missing all dates: {','.join(missing_files)}")
        da = xr.concat(pieces, dim="time").sortby("time")
        count = valid_time_count(da)
        valid_by_time = da.notnull().any(dim=[dim for dim in da.dims if dim != "time"])
        present = {
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date, ok in zip(da["time"].values, valid_by_time.values)
            if bool(ok)
        }
        missing = missing_files + [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in wanted
            if pd.Timestamp(date).strftime("%Y-%m-%d") not in present
        ]
        if count == 0:
            return FieldWithMeta(None, source, 0, f"missing all dates: {','.join(missing)}")
        field = crop_india_box(da.mean("time", skipna=True))
        return FieldWithMeta(
            realize(to_grid(field, self.grid)),
            source,
            count,
            f"missing_dates={','.join(missing)}" if missing else "",
        )

    def _truth_jfm_t2m(self, dates: Sequence[str]) -> FieldWithMeta:
        compact = self._truth_era5_prev_daily("t2m", dates)
        if compact.field is not None:
            return compact
        return self._truth_jfm("t2m", dates)

    def _truth_imd_2019(self, dates: Sequence[str]) -> FieldWithMeta:
        wanted = pd.to_datetime(list(dates))
        try:
            da = self.imd_2019["rain"].sel(time=wanted)
        except Exception as exc:
            return FieldWithMeta(None, "IMD_daily_rainfall_2019", 0, f"{type(exc).__name__}: {exc}")
        count = int(da.sizes.get("time", 0))
        if count == 0:
            return FieldWithMeta(None, "IMD_daily_rainfall_2019", 0, "no requested dates found")
        field = crop_india_box(da.mean("time", skipna=True))
        return FieldWithMeta(realize(to_grid(field, self.grid)), "IMD_daily_rainfall_2019", count)

    def _truth_imd_tp(self, dates: Sequence[str]) -> FieldWithMeta:
        wanted = pd.to_datetime(list(dates))
        pieces = []
        missing_files = []
        for year, group in pd.Series(wanted).groupby(wanted.year):
            try:
                ds = self.imd_year(int(year))
                selected = ds["rain"].reindex(time=list(group))
                if selected.sizes.get("time", 0):
                    pieces.append(selected)
            except Exception as exc:
                missing_files.extend([pd.Timestamp(date).strftime("%Y-%m-%d") for date in group])
                if len(wanted) == len(missing_files):
                    return FieldWithMeta(None, f"IMD_daily_rainfall_{int(year)}", 0, f"{type(exc).__name__}: {exc}")
        if not pieces:
            return FieldWithMeta(None, "IMD_daily_rainfall", 0, f"missing all dates: {','.join(missing_files)}")
        da = xr.concat(pieces, dim="time").sortby("time")
        count = valid_time_count(da)
        valid_by_time = da.notnull().any(dim=[dim for dim in da.dims if dim != "time"])
        present = {
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date, ok in zip(da["time"].values, valid_by_time.values)
            if bool(ok)
        }
        missing = missing_files + [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in wanted
            if pd.Timestamp(date).strftime("%Y-%m-%d") not in present
        ]
        if count == 0:
            return FieldWithMeta(None, "IMD_daily_rainfall", 0, f"missing all dates: {','.join(missing)}")
        field = crop_india_box(da.mean("time", skipna=True))
        years = sorted({pd.Timestamp(date).year for date in da["time"].values})
        source = "IMD_daily_rainfall_" + "_".join(str(year) for year in years)
        return FieldWithMeta(
            realize(to_grid(field, self.grid)),
            source,
            count,
            f"missing_dates={','.join(missing)}" if missing else "",
        )

    def _truth_era5_tp(self, season: str, dates: Sequence[str]) -> FieldWithMeta:
        if season == "jfm2026":
            return self._truth_era5_prev_daily("tp", dates)
        if season == "jjas2019":
            return self._truth_wb2_tp(dates)
        raise ValueError(f"no ERA5 TP truth reader for {season}")

    def _truth_era5_prev_daily(self, variable: str, dates: Sequence[str]) -> FieldWithMeta:
        ds = self.era5_prev_tp if variable == "tp" else self.era5_prev_t2m
        var_name = {"tp": "tp", "t2m": "t2m"}[variable]
        wanted = pd.to_datetime(list(dates))
        try:
            da = ds[var_name].reindex(time=wanted)
        except Exception as exc:
            return FieldWithMeta(None, f"ERA5_ARCO_daily_{variable}", 0, f"{type(exc).__name__}: {exc}")
        count = valid_time_count(da)
        if count == 0:
            return FieldWithMeta(None, f"ERA5_ARCO_daily_{variable}", 0, "no requested dates found")
        valid_by_time = da.notnull().any(dim=[dim for dim in da.dims if dim != "time"])
        missing = [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date, ok in zip(da["time"].values, valid_by_time.values)
            if not bool(ok)
        ]
        field = crop_india_box(da.mean("time", skipna=True))
        return FieldWithMeta(
            realize(to_grid(field, self.grid)),
            f"ERA5_ARCO_daily_{variable}",
            count,
            f"missing_dates={','.join(missing)}" if missing else "",
        )

    def _truth_wb2_tp(self, dates: Sequence[str]) -> FieldWithMeta:
        wanted = pd.to_datetime(list(dates)).normalize()
        start = wanted.min()
        end = wanted.max() + pd.Timedelta(hours=18)
        source = "WeatherBench2_ERA5_total_precipitation_6hr_daily_sum"
        try:
            da6 = self.wb2["total_precipitation_6hr"].sel(time=slice(start, end))
            daily = (da6.resample(time="1D").sum() * 1000.0).rename("tp")
            daily = daily.reindex(time=wanted)
            count = valid_time_count(daily)
            if count == 0:
                return FieldWithMeta(None, source, 0, "no requested dates found")
            valid_by_time = daily.notnull().any(dim=[dim for dim in daily.dims if dim != "time"])
            missing = [
                pd.Timestamp(date).strftime("%Y-%m-%d")
                for date, ok in zip(daily["time"].values, valid_by_time.values)
                if not bool(ok)
            ]
            field = crop_india_box(daily.mean("time", skipna=True))
            return FieldWithMeta(
                realize(to_grid(field, self.grid)),
                source,
                count,
                f"missing_dates={','.join(missing)}" if missing else "",
            )
        except Exception as exc:
            return FieldWithMeta(None, source, 0, f"{type(exc).__name__}: {exc}")

    def _truth_wb2(self, variable: str, dates: Sequence[str]) -> FieldWithMeta:
        start = pd.Timestamp(dates[0])
        end = pd.Timestamp(dates[-1]) + pd.Timedelta(hours=23)
        if variable == "z500":
            src = self.wb2["geopotential"].sel(level=500) / G
            source = "WeatherBench2_ERA5_geopotential_500hPa"
        else:
            src = self.wb2["2m_temperature"]
            source = "WeatherBench2_ERA5_2m_temperature"
        try:
            da = src.sel(time=slice(start, end))
            count = int(da.sizes.get("time", 0))
            if count == 0:
                return FieldWithMeta(None, source, 0, "no requested times found")
            field = crop_india_box(da.mean("time", skipna=True))
            return FieldWithMeta(realize(to_grid(field, self.grid)), source, count)
        except Exception as exc:
            return FieldWithMeta(None, source, 0, f"{type(exc).__name__}: {exc}")

    def _imd_day_numbers(self, dates: Sequence[str]) -> list[int]:
        ds = self.imd_clim
        out = []
        for date in pd.to_datetime(list(dates)):
            match = ds["day"].where(
                (ds["month"] == date.month) & (ds["day_of_month"] == date.day),
                drop=True,
            )
            if match.size == 0:
                raise KeyError(f"IMD climatology has no month/day for {date:%m-%d}")
            out.append(int(match.values[0]))
        return out

    def _clim_imd_tp(self, dates: Sequence[str]) -> FieldWithMeta:
        try:
            day_numbers = self._imd_day_numbers(dates)
            mean = self.imd_clim["rain_mean"].sel(day=day_numbers).mean("day", skipna=True)
            spread = self.imd_clim["rain_std"].sel(day=day_numbers).mean("day", skipna=True)
            mean = realize(to_grid(crop_india_box(mean), self.grid))
            spread = realize(to_grid(crop_india_box(spread), self.grid))
            return FieldWithMeta(mean, "IMD_1991_2020_daily_climatology", len(day_numbers), spread=spread)
        except Exception as exc:
            return FieldWithMeta(None, "IMD_1991_2020_daily_climatology", 0, f"{type(exc).__name__}: {exc}")

    def _clim_era5(self, variable: str, dates: Sequence[str]) -> FieldWithMeta:
        var_name = {"z500": "z500", "t2m": "t2m", "tp": "tp"}[variable]
        scale = {"z500": 1.0 / G, "tp": 1000.0, "t2m": 1.0}[variable]
        doys = [pd.Timestamp(date).dayofyear for date in dates]
        try:
            da = self.era5_clim[var_name].sel(dayofyear=doys).mean("dayofyear", skipna=True) * scale
            field = realize(to_grid(crop_india_box(da), self.grid))
            return FieldWithMeta(field, "ERA5_1990_2019_dayofyear_climatology", len(doys))
        except Exception as exc:
            return FieldWithMeta(None, "ERA5_1990_2019_dayofyear_climatology", 0, f"{type(exc).__name__}: {exc}")


def _lead_indices(da: xr.DataArray, start_day: int, end_day: int) -> np.ndarray:
    lead = np.asarray(da["lead"].values, dtype=float)
    lo = (start_day - 1) * 24.0
    hi = end_day * 24.0
    return np.where((lead > lo + 1e-6) & (lead <= hi + 1e-6))[0]


def _exact_lead(da: xr.DataArray, hour: float) -> xr.DataArray:
    lead = np.asarray(da["lead"].values, dtype=float)
    idx = np.where(np.isclose(lead, hour, atol=1e-3))[0]
    if idx.size == 0:
        raise KeyError(f"lead hour {hour:g} not found; available {lead[:5]} ... {lead[-5:]}")
    return da.isel(lead=int(idx[0]))


def _convert_nonaccumulated_units(model: str, variable: str, field: xr.DataArray) -> xr.DataArray:
    model = model.lower()
    variable = canonical_variable(variable)
    if variable == "z500" and model in {"delysm", "fuxi"}:
        return field / G
    if variable == "tp" and model == "fuxi":
        return field * 24.0
    return field


def open_model_source(model: str, season: str, init_date: str, variable: str) -> ForecastSource:
    model = model.lower()
    variable = canonical_variable(variable)
    if model == "spire":
        return open_spire_source(init_date, variable)

    opener = _open_forecast_module()
    da, paths = opener.open_forecast(
        model=model,
        season=season,
        init_date=init_date,
        variable=variable,
        data_root=get_paths().all_model_data,
    )
    notes = f"native_units={da.attrs.get('units', '')}"
    if model == "fuxi" and variable == "tp":
        notes = "native_units=mm h-1 24-hour average; multiplied by 24 to mm/day during aggregation"
    return ForecastSource(
        model=model,
        variable=variable,
        kind="ensemble",
        ensemble=da,
        source_paths=tuple(str(p) for p in paths),
        notes=notes,
    )


def open_spire_source(init_date: str, variable: str) -> ForecastSource:
    variable = canonical_variable(variable)
    paths = get_paths()
    group = zarr.open_group(str(paths.jfm_spire_zarr), mode="r")["mean_stddev"]
    refs = spire_init_dates()
    init_compact = pd.Timestamp(init_date).strftime("%Y%m%d")
    if init_compact not in refs:
        raise FileNotFoundError(f"SPIRE has no init {init_date}; available {refs[0]} to {refs[-1]}")
    ref_idx = refs.index(init_compact)
    step_days = np.asarray(group["step"][:], dtype=float)
    lat = np.asarray(group["latitude"][:], dtype=float)
    lon = np.asarray(group["longitude"][:], dtype=float)

    if variable == "tp":
        mean_array = dask_array.from_zarr(group["precipitation_amount"])[ref_idx, :, :, :]
        spread_array = dask_array.from_zarr(group["precipitation_amount_stddev"])[ref_idx, :, :, :]
        notes = "SPIRE precipitation_amount kg m-2 per 24h treated as mm/day"
    elif variable == "z500":
        isobar = np.asarray(group["isobar"][:], dtype=float)
        lev_idx = int(np.where(np.isclose(isobar, 50000.0))[0][0])
        mean_array = dask_array.from_zarr(group["geopotential_height_at_isobaric_levels"])[
            ref_idx, :, lev_idx, :, :
        ]
        spread_array = dask_array.from_zarr(group["geopotential_height_at_isobaric_levels_stddev"])[
            ref_idx, :, lev_idx, :, :
        ]
        notes = "SPIRE z500 is geopotential height in m"
    else:
        mean_array = dask_array.from_zarr(group["air_temperature"])[ref_idx, :, :, :]
        spread_array = dask_array.from_zarr(group["air_temperature_stddev"])[ref_idx, :, :, :]
        notes = "SPIRE t2m is 24-hour ensemble mean air temperature"

    coords = {"lead": step_days * 24.0, "lat": lat, "lon": lon}
    mean = xr.DataArray(mean_array, dims=("lead", "lat", "lon"), coords=coords, name=variable)
    spread = xr.DataArray(spread_array, dims=("lead", "lat", "lon"), coords=coords, name=f"{variable}_spread")
    return ForecastSource(
        model="spire",
        variable=variable,
        kind="gaussian",
        mean=mean,
        spread=spread,
        source_paths=(str(paths.jfm_spire_zarr),),
        notes=notes,
    )


def aggregate_source(
    source: ForecastSource,
    window: LeadWindow,
    grid: GridSpec,
    *,
    min_leads_for_mean: int = 2,
) -> AggregatedForecast:
    variable = source.variable
    model = source.model

    if source.kind == "ensemble":
        if source.ensemble is None:
            raise ValueError(f"{model} {variable}: missing ensemble data")
        ens = source.ensemble
        if variable == "tp" and model in CUMULATIVE_TP_MODELS:
            end = _exact_lead(ens, window.end * 24.0)
            if window.start == 1:
                field = end / window.n_days
            else:
                start = _exact_lead(ens, (window.start - 1) * 24.0)
                field = (end - start) / window.n_days
            field = field.clip(min=0.0)
            lead_count = window.n_days
        else:
            idx = _lead_indices(ens, window.start, window.end)
            lead_count = int(idx.size)
            if lead_count < min_leads_for_mean:
                raise ValueError(
                    f"insufficient leads for weekly mean: {lead_count} < {min_leads_for_mean}"
                )
            field = ens.isel(lead=idx).mean("lead", skipna=True)
            field = _convert_nonaccumulated_units(model, variable, field)

        field = crop_india_box(field)
        ensemble = realize(to_grid(field, grid))
        mean = ensemble.mean("member", skipna=True)
        spread = ensemble.std("member", ddof=1, skipna=True)
        return AggregatedForecast(
            model=model,
            variable=variable,
            kind="ensemble",
            mean=mean,
            spread=spread,
            ensemble=ensemble,
            member_count=int(ensemble.sizes.get("member", 0)),
            lead_count=lead_count,
            source_paths=source.source_paths,
            notes=source.notes,
        )

    if source.mean is None:
        raise ValueError(f"{model} {variable}: missing mean data")
    idx = _lead_indices(source.mean, window.start, window.end)
    lead_count = int(idx.size)
    if lead_count < min_leads_for_mean:
        raise ValueError(f"insufficient leads for weekly mean: {lead_count} < {min_leads_for_mean}")
    mean = source.mean.isel(lead=idx).mean("lead", skipna=True)
    spread = None
    if source.spread is not None:
        spread = source.spread.isel(lead=idx).mean("lead", skipna=True)
    mean = realize(to_grid(crop_india_box(mean), grid))
    spread = realize(to_grid(crop_india_box(spread), grid)) if spread is not None else None
    return AggregatedForecast(
        model=model,
        variable=variable,
        kind="gaussian",
        mean=mean,
        spread=spread,
        ensemble=None,
        member_count=0,
        lead_count=lead_count,
        source_paths=source.source_paths,
        notes=source.notes,
    )


def mse_skill_score(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    reference: xr.DataArray,
    weights: xr.DataArray,
) -> float:
    mse_model = weighted_mean((forecast - truth) ** 2, weights)
    mse_ref = weighted_mean((reference - truth) ** 2, weights)
    return 1.0 - mse_model / mse_ref if mse_ref > 0 else float("nan")


def finite_cell_count(*fields: xr.DataArray) -> int:
    aligned = xr.align(*fields, join="inner")
    ok = None
    for field in aligned:
        this = np.isfinite(field)
        ok = this if ok is None else (ok & this)
    if ok is None:
        return 0
    value = ok.sum()
    if hasattr(value.data, "compute"):
        value = value.compute()
    return int(value.item())


def crps_ensemble_sorted(
    ensemble: xr.DataArray,
    truth: xr.DataArray,
    weights: xr.DataArray,
    member_dim: str = "member",
) -> float:
    """Finite-member CRPS using the sorted-member identity on the common grid."""

    ens, obs = xr.align(ensemble, truth, join="inner")
    ens = ens.transpose(member_dim, "lat", "lon")
    arr = np.asarray(ens.values, dtype=float)
    y = np.asarray(obs.values, dtype=float)
    out = np.full(y.shape, np.nan, dtype=float)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            yy = y[i, j]
            if not np.isfinite(yy):
                continue
            x = arr[:, i, j]
            x = np.sort(x[np.isfinite(x)])
            m = x.size
            if m == 0:
                continue
            term1 = np.mean(np.abs(x - yy))
            coeff = 2.0 * np.arange(1, m + 1) - m - 1.0
            mean_pair_abs = 2.0 * np.sum(coeff * x) / (m * m)
            out[i, j] = term1 - 0.5 * mean_pair_abs
    field = xr.DataArray(out, dims=("lat", "lon"), coords={"lat": obs["lat"], "lon": obs["lon"]})
    return weighted_mean(field, weights)


def deterministic_rows(
    forecast: AggregatedForecast,
    truth: FieldWithMeta,
    clim: FieldWithMeta,
    region_context: RegionContext,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    assert truth.field is not None and clim.field is not None
    for region in REGIONS:
        f = region_context.mask(forecast.mean, region)
        o = region_context.mask(truth.field, region)
        c = region_context.mask(clim.field, region)
        scores = score_deterministic(f, o, climatology=c, weights=region_context.weights)
        rows.append(
            {
                **metadata,
                "model": forecast.model,
                "region": region,
                "forecast_type": forecast.kind,
                "member_count": forecast.member_count,
                "lead_count": forecast.lead_count,
                "truth_source": truth.source,
                "climatology_source": clim.source,
                "unit": VARIABLE_UNITS[variable_label(forecast.variable)],
                "n_grid_cells": finite_cell_count(f, o, c),
                "acc": scores["acc"],
                "rmse": scores["rmse"],
                "bias": scores["bias"],
                "mae": scores["mae"],
                "mse_skill_clim": mse_skill_score(f, o, c, region_context.weights),
                "clim_rmse": rmse(c, o, region_context.weights),
                "forecast_mean": weighted_mean(f, region_context.weights),
                "truth_mean": weighted_mean(o, region_context.weights),
                "clim_mean": weighted_mean(c, region_context.weights),
                "notes": forecast.notes,
            }
        )
    return rows


def area_scatter_rows(
    forecast: AggregatedForecast,
    truth: FieldWithMeta,
    clim: FieldWithMeta,
    region_context: RegionContext,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    """Region-mean forecast/truth pairs for simple scatter plots."""

    rows = []
    assert truth.field is not None and clim.field is not None
    for region in REGIONS:
        f = region_context.mask(forecast.mean, region)
        o = region_context.mask(truth.field, region)
        c = region_context.mask(clim.field, region)
        fv = weighted_mean(f, region_context.weights)
        ov = weighted_mean(o, region_context.weights)
        cv = weighted_mean(c, region_context.weights)
        rows.append(
            {
                **metadata,
                "model": forecast.model,
                "region": region,
                "forecast_type": forecast.kind,
                "member_count": forecast.member_count,
                "lead_count": forecast.lead_count,
                "unit": VARIABLE_UNITS[variable_label(forecast.variable)],
                "truth_source": truth.source,
                "climatology_source": clim.source,
                "forecast_value": fv,
                "truth_value": ov,
                "climatology_value": cv,
                "forecast_anomaly": fv - cv if np.isfinite(fv) and np.isfinite(cv) else float("nan"),
                "truth_anomaly": ov - cv if np.isfinite(ov) and np.isfinite(cv) else float("nan"),
                "error": fv - ov if np.isfinite(fv) and np.isfinite(ov) else float("nan"),
                "notes": forecast.notes,
            }
        )
    return rows


def grid_scatter_rows(
    forecast: AggregatedForecast,
    truth: FieldWithMeta,
    clim: FieldWithMeta,
    region_context: RegionContext,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    """Grid-cell forecast/truth pairs inside the All-India mask."""

    assert truth.field is not None and clim.field is not None
    f, o, c = xr.align(forecast.mean, truth.field, clim.field, join="inner")
    all_india = region_context.masks["All India"].astype(bool).sel(lat=f["lat"], lon=f["lon"])
    region_values = np.full(all_india.shape, "", dtype=object)
    for region in REGION_KEYS:
        mask = region_context.masks[region].astype(bool).sel(lat=f["lat"], lon=f["lon"]).values
        region_values[mask] = region

    fv = np.asarray(f.values, dtype=float)
    ov = np.asarray(o.values, dtype=float)
    cv = np.asarray(c.values, dtype=float)
    mask = np.asarray(all_india.values, dtype=bool) & np.isfinite(fv) & np.isfinite(ov) & np.isfinite(cv)
    lat = np.asarray(f["lat"].values, dtype=float)
    lon = np.asarray(f["lon"].values, dtype=float)
    rows = []
    for i, lat_value in enumerate(lat):
        for j, lon_value in enumerate(lon):
            if not mask[i, j]:
                continue
            rows.append(
                {
                    **metadata,
                    "model": forecast.model,
                    "region": region_values[i, j],
                    "lat": lat_value,
                    "lon": lon_value,
                    "forecast_type": forecast.kind,
                    "member_count": forecast.member_count,
                    "lead_count": forecast.lead_count,
                    "unit": VARIABLE_UNITS[variable_label(forecast.variable)],
                    "truth_source": truth.source,
                    "climatology_source": clim.source,
                    "forecast_value": fv[i, j],
                    "truth_value": ov[i, j],
                    "climatology_value": cv[i, j],
                    "forecast_anomaly": fv[i, j] - cv[i, j],
                    "truth_anomaly": ov[i, j] - cv[i, j],
                    "error": fv[i, j] - ov[i, j],
                    "notes": forecast.notes,
                }
            )
    return rows


def probabilistic_rows(
    forecast: AggregatedForecast,
    truth: FieldWithMeta,
    clim: FieldWithMeta,
    region_context: RegionContext,
    metadata: dict[str, object],
    *,
    reliability_nbins: int = 10,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prob_rows: list[dict[str, object]] = []
    brier_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    if forecast.spread is None:
        return prob_rows, brier_rows, reliability_rows
    assert truth.field is not None and clim.field is not None

    for region in REGIONS:
        mu = region_context.mask(forecast.mean, region)
        sig = region_context.mask(forecast.spread, region).clip(min=SIGMA_FLOOR[forecast.variable])
        obs = region_context.mask(truth.field, region)
        cm = region_context.mask(clim.field, region)
        if forecast.kind == "ensemble" and forecast.ensemble is not None:
            ens = region_context.mask(forecast.ensemble, region)
            crps = crps_ensemble_sorted(ens, obs, region_context.weights)
        else:
            ens = None
            crps = crps_gaussian(mu, sig, obs, region_context.weights, SIGMA_FLOOR[forecast.variable])
        crps_ref = weighted_mean(abs(cm - obs), region_context.weights)
        rmse_mean = rmse(mu, obs, region_context.weights)
        spread_mean = weighted_mean(sig, region_context.weights)
        prob_rows.append(
            {
                **metadata,
                "model": forecast.model,
                "region": region,
                "forecast_type": forecast.kind,
                "member_count": forecast.member_count,
                "lead_count": forecast.lead_count,
                "truth_source": truth.source,
                "climatology_source": clim.source,
                "unit": VARIABLE_UNITS[variable_label(forecast.variable)],
                "crps": crps,
                "crps_clim": crps_ref,
                "crpss_clim": crps_skill_score(crps, crps_ref),
                "spread": spread_mean,
                "rmse_ensmean": rmse_mean,
                "spread_skill_ratio": spread_skill_ratio(sig, mu, obs, region_context.weights),
                "notes": forecast.notes,
            }
        )

        if forecast.variable != "tp":
            continue
        for threshold in TP_THRESHOLDS_MM_DAY:
            if ens is not None:
                probability = probability_exceedance(ens, threshold)
            else:
                probability = gaussian_probability_exceedance(
                    mu, sig, threshold, sigma_floor=SIGMA_FLOOR["tp"]
                )
            outcome = (obs > threshold).astype(float)
            if clim.spread is not None:
                clim_spread = region_context.mask(clim.spread, region).clip(min=SIGMA_FLOOR["tp"])
                ref_probability = gaussian_probability_exceedance(
                    cm, clim_spread, threshold, sigma_floor=SIGMA_FLOOR["tp"]
                )
            else:
                ref_probability = xr.full_like(cm, float(outcome.mean(skipna=True)))
            bs = brier_score(probability, outcome, region_context.weights)
            bs_ref = brier_score(ref_probability, outcome, region_context.weights)
            event_name = f"tp_gt_{threshold:g}_mm_day"
            brier_rows.append(
                {
                    **metadata,
                    "model": forecast.model,
                    "region": region,
                    "forecast_type": forecast.kind,
                    "event": event_name,
                    "threshold": threshold,
                    "brier": bs,
                    "brier_clim": bs_ref,
                    "brier_skill_clim": brier_skill_score(bs, bs_ref),
                    "base_rate": weighted_mean(outcome, region_context.weights),
                    "notes": forecast.notes,
                }
            )
            if region == "All India":
                for row in reliability_bins(probability, outcome, nbins=reliability_nbins):
                    reliability_rows.append(
                        {
                            **metadata,
                            "model": forecast.model,
                            "forecast_type": forecast.kind,
                            "event": event_name,
                            "threshold": threshold,
                            **row,
                        }
                    )
    return prob_rows, brier_rows, reliability_rows


def write_outputs(
    out_dir: Path,
    *,
    deterministic: list[dict[str, object]],
    probabilistic: list[dict[str, object]],
    brier: list[dict[str, object]],
    reliability: list[dict[str, object]],
    scatter_area: list[dict[str, object]],
    scatter_grid: list[dict[str, object]],
    status: list[dict[str, object]],
    metadata: dict[str, object],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    tables = {
        "deterministic_weekly": deterministic,
        "probabilistic_weekly": probabilistic,
        "brier_weekly": brier,
        "reliability_weekly": reliability,
        "scatter_area_weekly": scatter_area,
        "scatter_grid_weekly": scatter_grid,
        "model_status": status,
    }
    for name, rows in tables.items():
        path = out_dir / f"{name}.csv"
        frame = pd.DataFrame(rows)
        sort_cols = [
            col
            for col in (
                "season",
                "run_label",
                "set_name",
                "init_date",
                "variable",
                "week",
                "model",
                "region",
                "lat",
                "lon",
                "event",
                "bin",
            )
            if col in frame.columns
        ]
        if sort_cols:
            frame = frame.sort_values(sort_cols).reset_index(drop=True)
        frame.to_csv(path, index=False)
        outputs[name] = str(path)
    meta_path = out_dir / "run_metadata.json"
    meta_path.write_text(json.dumps({**metadata, "outputs": outputs}, indent=2, default=str))
    outputs["metadata"] = str(meta_path)
    return outputs


def process_init_date(
    *,
    season: str,
    run_label: str,
    set_name: str,
    truth_source: str,
    init_date: str,
    selected_models: Sequence[str],
    selected_variables: Sequence[str],
    selected_weeks: Sequence[LeadWindow],
    grid: GridSpec,
    min_leads_for_mean: int,
    include_grid_scatter: bool,
) -> dict[str, list[dict[str, object]]]:
    """Process all requested variables/weeks/models for one init date."""

    data = VerificationData(grid, truth_source=truth_source)
    regions = RegionContext(grid.dgrid)
    deterministic: list[dict[str, object]] = []
    probabilistic: list[dict[str, object]] = []
    brier: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    scatter_area: list[dict[str, object]] = []
    scatter_grid: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    source_cache: dict[tuple[str, str, str], ForecastSource | Exception] = {}

    print(f"[{season} {run_label}] init {init_date}", flush=True)
    for variable in selected_variables:
        for model in selected_models:
            key = (init_date, model, variable)
            if key not in source_cache:
                try:
                    source_cache[key] = open_model_source(model, season, init_date, variable)
                    status.append(
                        {
                            "season": season,
                            "run_label": run_label,
                            "init_date": init_date,
                            "model": model,
                            "variable": variable,
                            "stage": "open",
                            "status": "opened",
                            "notes": getattr(source_cache[key], "notes", ""),
                        }
                    )
                except Exception as exc:
                    source_cache[key] = exc
                    status.append(
                        {
                            "season": season,
                            "run_label": run_label,
                            "init_date": init_date,
                            "model": model,
                            "variable": variable,
                            "stage": "open",
                            "status": "skipped",
                            "notes": f"{type(exc).__name__}: {exc}",
                        }
                    )

        for window in selected_weeks:
            dates = valid_dates_for_window(init_date, window)
            truth = data.truth_period(season, variable, dates)
            clim = data.climatology_period(season, variable, dates)
            common_meta = {
                "season": season,
                "run_label": run_label,
                "set_name": set_name,
                "init_date": init_date,
                "valid_start": dates[0],
                "valid_end": dates[-1],
                "week": window.week,
                "week_name": window.name,
                "lead_start_day": window.start,
                "lead_end_day": window.end,
                "variable": variable,
                "truth_request": data.truth_source,
                "tp_truth_source": data.tp_truth_source(season) if variable == "tp" else "era5",
                "grid_dgrid": grid.dgrid,
                "truth_day_count": truth.count,
                "climatology_day_count": clim.count,
            }
            if truth.field is None or clim.field is None or truth.count < window.n_days:
                status.append(
                    {
                        **common_meta,
                        "model": "ALL",
                        "stage": "truth_climatology",
                        "status": "skipped",
                        "notes": f"truth={truth.notes}; climatology={clim.notes}",
                    }
                )
                continue

            aggregates: list[AggregatedForecast] = []
            for model in selected_models:
                source = source_cache[(init_date, model, variable)]
                if isinstance(source, Exception):
                    continue
                try:
                    forecast = aggregate_source(
                        source,
                        window,
                        grid,
                        min_leads_for_mean=min_leads_for_mean,
                    )
                    aggregates.append(forecast)
                    status.append(
                        {
                            **common_meta,
                            "model": model,
                            "stage": "aggregate_score",
                            "status": "scored",
                            "member_count": forecast.member_count,
                            "lead_count": forecast.lead_count,
                            "notes": forecast.notes,
                        }
                    )
                    deterministic.extend(deterministic_rows(forecast, truth, clim, regions, common_meta))
                    scatter_area.extend(area_scatter_rows(forecast, truth, clim, regions, common_meta))
                    if include_grid_scatter:
                        scatter_grid.extend(grid_scatter_rows(forecast, truth, clim, regions, common_meta))
                    p_rows, b_rows, r_rows = probabilistic_rows(forecast, truth, clim, regions, common_meta)
                    probabilistic.extend(p_rows)
                    brier.extend(b_rows)
                    reliability.extend(r_rows)
                except Exception as exc:
                    status.append(
                        {
                            **common_meta,
                            "model": model,
                            "stage": "aggregate_score",
                            "status": "skipped",
                            "notes": f"{type(exc).__name__}: {exc}",
                        }
                    )

            if len(aggregates) >= 2:
                mme_inputs = [fc.mean.reset_coords(drop=True) for fc in aggregates]
                mme_field = xr.concat(
                    mme_inputs,
                    dim=pd.Index([fc.model for fc in aggregates], name="model_member"),
                    coords="minimal",
                    compat="override",
                ).mean("model_member", skipna=True)
                mme = AggregatedForecast(
                    model="mme",
                    variable=variable,
                    kind="deterministic_mme",
                    mean=mme_field,
                    spread=None,
                    ensemble=None,
                    member_count=len(aggregates),
                    lead_count=min(fc.lead_count for fc in aggregates),
                    source_paths=(),
                    notes="mean of available model ensemble means on common grid",
                )
                deterministic.extend(deterministic_rows(mme, truth, clim, regions, common_meta))
                scatter_area.extend(area_scatter_rows(mme, truth, clim, regions, common_meta))
                if include_grid_scatter:
                    scatter_grid.extend(grid_scatter_rows(mme, truth, clim, regions, common_meta))

    return {
        "deterministic": deterministic,
        "probabilistic": probabilistic,
        "brier": brier,
        "reliability": reliability,
        "scatter_area": scatter_area,
        "scatter_grid": scatter_grid,
        "status": status,
    }


def _process_init_payload(payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    return process_init_date(**payload)


def run_weekly_pipeline(
    *,
    season: str,
    set_name: str,
    truth_source: str = "auto",
    variables: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    include_spire: bool = False,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_label: str = "test",
    max_inits: int | None = None,
    weeks: Sequence[int] | None = None,
    grid: GridSpec = GridSpec(),
    min_leads_for_mean: int = 2,
    workers: int = 1,
    include_grid_scatter: bool = True,
    write: bool = True,
) -> dict[str, object]:
    """Run weekly verification and optionally write tidy CSV outputs."""

    season = season.lower()
    truth_source = canonical_truth_source(truth_source)
    selected_weeks = [w for w in WEEKLY_WINDOWS if weeks is None or w.week in set(weeks)]
    init_dates, selected_models, selected_variables, selection_notes = prepare_run_selection(
        season,
        set_name,
        models=models,
        variables=variables,
        include_spire=include_spire,
        output_root=output_root,
        max_inits=max_inits,
    )

    deterministic: list[dict[str, object]] = []
    probabilistic: list[dict[str, object]] = []
    brier: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    scatter_area: list[dict[str, object]] = []
    scatter_grid: list[dict[str, object]] = []
    status: list[dict[str, object]] = []

    payloads = [
        {
            "season": season,
            "run_label": run_label,
            "set_name": set_name,
            "truth_source": truth_source,
            "init_date": init_date,
            "selected_models": selected_models,
            "selected_variables": selected_variables,
            "selected_weeks": selected_weeks,
            "grid": grid,
            "min_leads_for_mean": min_leads_for_mean,
            "include_grid_scatter": include_grid_scatter,
        }
        for init_date in init_dates
    ]

    def merge(result: dict[str, list[dict[str, object]]]) -> None:
        deterministic.extend(result["deterministic"])
        probabilistic.extend(result["probabilistic"])
        brier.extend(result["brier"])
        reliability.extend(result["reliability"])
        scatter_area.extend(result["scatter_area"])
        scatter_grid.extend(result["scatter_grid"])
        status.extend(result["status"])

    workers = max(1, int(workers))
    if workers == 1 or len(payloads) <= 1:
        for payload in payloads:
            merge(process_init_date(**payload))
    else:
        failed_payloads: list[tuple[str, dict[str, object], Exception]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_init = {
                executor.submit(_process_init_payload, payload): (payload["init_date"], payload)
                for payload in payloads
            }
            for future in as_completed(future_to_init):
                init_date, payload = future_to_init[future]
                try:
                    merge(future.result())
                except Exception as exc:
                    failed_payloads.append((str(init_date), payload, exc))
        for init_date, payload, parallel_exc in failed_payloads:
            try:
                merge(process_init_date(**payload))
                status.append(
                    {
                        "season": season,
                        "run_label": run_label,
                        "init_date": init_date,
                        "model": "ALL",
                        "variable": "ALL",
                        "stage": "worker_retry",
                        "status": "recovered",
                        "notes": (
                            f"parallel worker failed with {type(parallel_exc).__name__}: {parallel_exc}; "
                            "reran this init serially"
                        ),
                    }
                )
            except Exception as retry_exc:
                    status.append(
                        {
                            "season": season,
                            "run_label": run_label,
                            "init_date": init_date,
                            "model": "ALL",
                            "variable": "ALL",
                            "stage": "worker",
                            "status": "failed",
                            "notes": (
                                f"parallel worker failed with {type(parallel_exc).__name__}: {parallel_exc}; "
                                f"serial retry failed with {type(retry_exc).__name__}: {retry_exc}"
                            ),
                        }
                    )

    hard_failures = [row for row in status if row.get("stage") == "worker" and row.get("status") == "failed"]
    if hard_failures:
        examples = "; ".join(
            f"{row.get('init_date')}: {row.get('notes')}" for row in hard_failures[:5]
        )
        raise RuntimeError(
            f"{len(hard_failures)} init-date worker failures remain after serial retry; "
            f"not writing partial outputs. Examples: {examples}"
        )

    run_metadata = {
        "season": season,
        "set_name": set_name,
        "run_label": run_label,
        "truth_source": truth_source,
        "tp_truth_source": "imd" if truth_source == "auto" and season == "jjas2019" else "era5" if truth_source == "auto" else truth_source,
        "init_count": len(init_dates),
        "init_dates": init_dates,
        "models": selected_models,
        "variables": selected_variables,
        "weeks": [w.week for w in selected_weeks],
        "workers": workers,
        "include_grid_scatter": include_grid_scatter,
        "grid": {"lat0": grid.lat0, "lat1": grid.lat1, "lon0": grid.lon0, "lon1": grid.lon1, "dgrid": grid.dgrid},
        "selection_notes": selection_notes,
        "method_notes": [
            "weekly windows use lead day init+start through init+end inclusive",
            "TP truth can be selected with truth_source=auto|era5|imd; auto uses IMD for JJAS2019 TP and ERA5 for JFM2026 TP",
            "TP climatology follows the selected TP truth source: IMD 1991-2020 for IMD truth, ERA5 1990-2019 converted to mm/day for ERA5 truth",
            "Z500/T2M climatology uses ERA5 day-of-year climatology",
            "FuXi TP is a 24-hour hourly average and is multiplied by 24 to mm/day",
            "ECMWF/UKMO/NCEP TP are cumulative and are differenced to mm/day",
            "DLESyM/FuXi Z500 are converted from geopotential to gpm with /9.80665",
            "SPIRE daily JFM2026 mean/std are read from archive s2s-research.zarr/mean_stddev and scored as Gaussian forecasts",
            "SPIRE weekly stddev is averaged over lead days, matching the older analysis code",
            "scatter_area_weekly.csv contains region-mean forecast/truth pairs",
            "scatter_grid_weekly.csv contains grid-cell forecast/truth pairs inside the India mask",
        ],
    }
    out_dir = output_root / season / "03_metrics" / run_label
    outputs = {}
    if write:
        outputs = write_outputs(
            out_dir,
            deterministic=deterministic,
            probabilistic=probabilistic,
            brier=brier,
            reliability=reliability,
            scatter_area=scatter_area,
            scatter_grid=scatter_grid,
            status=status,
            metadata=run_metadata,
        )

    return {
        "metadata": run_metadata,
        "outputs": outputs,
        "row_counts": {
            "deterministic": len(deterministic),
            "probabilistic": len(probabilistic),
            "brier": len(brier),
            "reliability": len(reliability),
            "scatter_area": len(scatter_area),
            "scatter_grid": len(scatter_grid),
            "status": len(status),
        },
    }
