#!/usr/bin/env python3
"""Build paired FuXi-S2S/ERPAS rainfall ACC CSVs for 2023--2024.

The all-season branch follows the paper_v2 comparison convention: every
forecast and IMD truth field is expressed relative to the same IMD 1991--2020
calendar-day climatology.  A second JJAS-only branch uses each forecast
system's ready native/provider model climatology while retaining the IMD
climatology for the observed anomaly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
SOURCE_ROOT = WORKSPACE / "deliverables/imd_acc_1p5_full_2023_2024"
SOURCE_DATA = SOURCE_ROOT / "data/processed/available_cycles_1p5_weekly.nc"
ACC_SCRIPT = WORKSPACE / "deliverables/imd_acc_model_climo_fuxi_erpas_2023_2024/scripts/build_acc_figure.py"
BASE_CONFIG = SOURCE_ROOT / "config/full_study.json"
IMD_ROOT = Path("/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/netcdf")
IMD_CLIMO = Path(
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/"
    "imd_rain_1991_2020_daily_climatology.nc"
)
IMD_REGION_MASKS = Path(
    "/home/raj.ayush/s2s/s2s_anlysis/masks/imd_region_masks_0.25deg.nc"
)
ERPAS_ROOT = Path("/storage/raj.ayush/s2s_final_data/final_iteration/raw/erpas/forecast")
FUXI_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/forecasts"
)

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import numpy as np
import pandas as pd
import xarray as xr


MODELS = ("ERPAS", "FuXi-S2S")
REGIONS = (
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
SEASON_WINDOWS = {
    "ALL": tuple(range(1, 13)),
    "JF": (1, 2),
    "JFM": (1, 2, 3),
    "MAM": (3, 4, 5),
    "JJAS": (6, 7, 8, 9),
    "OND": (10, 11, 12),
}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


accmod = import_file("multiseason_acc_support", ACC_SCRIPT)
engine = accmod.engine
core = accmod.core


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cases() -> tuple[list[dict], list[dict]]:
    cases: list[dict] = []
    excluded: list[dict] = []
    for year in (2023, 2024):
        erpas_dir = ERPAS_ROOT / f"annual{year}/tp"
        fuxi_dir = FUXI_ROOT / f"annual{year}"
        for path in sorted(erpas_dir.glob("APCP_????????.grb")):
            erpas_init = pd.Timestamp(path.stem.removeprefix("APCP_"))
            fuxi_init = erpas_init - pd.Timedelta(days=2)
            fuxi_path = fuxi_dir / f"{fuxi_init:%Y%m%d}.nc"
            if not fuxi_path.is_file():
                excluded.append(
                    {
                        "erpas_init": f"{erpas_init:%Y-%m-%d}",
                        "reason": "matched preceding-Monday strict-00Z FuXi file missing",
                        "expected_fuxi_path": str(fuxi_path),
                    }
                )
                continue
            cases.append(
                {
                    "case_id": f"paired_{erpas_init:%Y%m%d}",
                    "erpas_init": f"{erpas_init:%Y-%m-%d}T00:00:00Z",
                    "comparison_init": f"{fuxi_init:%Y-%m-%d}T00:00:00Z",
                    "valid_start": f"{(erpas_init + pd.Timedelta(days=1)):%Y-%m-%d}T00:00:00Z",
                    "valid_end": f"{(erpas_init + pd.Timedelta(days=29)):%Y-%m-%d}T00:00:00Z",
                }
            )
    return cases, excluded


def remap_imd(
    cases: list[dict],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    land_support: tuple[np.ndarray, np.ndarray, np.ndarray],
    original_mask: np.ndarray,
) -> tuple[
    dict[pd.Timestamp, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict,
]:
    requested = pd.DatetimeIndex(
        sorted({stamp for case in cases for stamp in core.expected_period_ends(case)})
    )
    observed: dict[pd.Timestamp, np.ndarray] = {}
    observation_checks: dict[str, dict] = {}
    raw_by_year: dict[int, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    source_lat = source_lon = None
    for year in sorted(set(requested.year)):
        dates = requested[requested.year == year]
        path = IMD_ROOT / f"imd_rain_{year}.nc"
        with xr.open_dataset(path) as source:
            field = source.rain.sel(time=dates).load()
            if field.attrs.get("units") != "mm/day":
                raise ValueError(f"IMD {year} units are not mm/day")
            year_lat = source.lat.values.astype(float)
            year_lon = source.lon.values.astype(float)
        if source_lat is None:
            source_lat, source_lon = year_lat, year_lon
        elif not (np.array_equal(source_lat, year_lat) and np.array_equal(source_lon, year_lon)):
            raise ValueError("IMD source grid changed between years")
        raw_by_year[year] = (dates, field.values.astype(np.float32))

    original_support = land_support[0].astype(bool)
    finite_every_day = np.ones_like(original_support, dtype=bool)
    for _, values in raw_by_year.values():
        finite_every_day &= np.isfinite(values).all(axis=0)
    fixed_support = original_support & finite_every_day
    if int(fixed_support.sum()) < 4000:
        raise ValueError("all-season fixed IMD source support is unexpectedly small")
    fixed_land_support = (fixed_support.astype(np.float64), source_lat, source_lon)
    new_india_fraction = new_weight = None
    for year, (dates, values) in raw_by_year.items():
        remapped_values, denominators, target_area, checks = core.remap_conservative(
            values,
            source_lat,
            source_lon,
            target_lat,
            target_lon,
            support=fixed_support,
        )
        if new_india_fraction is None:
            new_india_fraction = denominators[0] / target_area
            new_weight = np.where(
                new_india_fraction > 0, target_area * new_india_fraction, 0.0
            )
        if not np.isfinite(remapped_values[:, new_weight > 0]).all() or float(np.nanmin(remapped_values[:, new_weight > 0])) < -1e-6:
            raise ValueError(f"IMD {year} truth failed finite/nonnegative checks")
        if not np.allclose(denominators, denominators[0], rtol=0, atol=1e-12):
            raise ValueError(f"IMD {year} remap support changes with date")
        observed.update({stamp: remapped_values[index] for index, stamp in enumerate(dates)})
        observation_checks[str(year)] = checks

    with xr.open_dataset(IMD_CLIMO) as source:
        if source.attrs.get("baseline") != "1991-2020":
            raise ValueError("unexpected IMD climatology baseline")
        month_days = source.month_day.values.astype(str)
        daily = source.rain_mean.load()
        climo_lat = source.lat.values.astype(float)
        climo_lon = source.lon.values.astype(float)
    remapped, denominators, _, climo_checks = core.remap_conservative(
        daily.values,
        climo_lat,
        climo_lon,
        target_lat,
        target_lon,
        support=fixed_support,
    )
    if not np.isfinite(remapped[:, new_weight > 0]).all():
        raise ValueError("IMD climatology failed finite check")
    climatology = {key: remapped[index] for index, key in enumerate(month_days)}
    climatology["02-29"] = 0.5 * (climatology["02-28"] + climatology["03-01"])
    return observed, climatology, new_india_fraction.astype(np.float32), new_weight, fixed_land_support, {
        "requested_period_end_count": len(requested),
        "first_period_end": requested.min().isoformat(),
        "last_period_end": requested.max().isoformat(),
        "truth_remap_checks": observation_checks,
        "climatology_remap_checks": climo_checks,
        "feb29_rule": "mean of Feb-28 and Mar-01 climatology fields",
        "original_source_support_cells": int(original_support.sum()),
        "fixed_all_season_source_support_cells": int(fixed_support.sum()),
        "source_cells_removed_for_any_date_missingness": int(np.sum(original_support & ~finite_every_day)),
        "original_target_supported_cells": int(original_mask.sum()),
        "fixed_all_season_target_supported_cells": int(np.sum(new_weight > 0)),
    }


def weekly_reference(
    case: dict,
    observed: dict[pd.Timestamp, np.ndarray],
    climatology: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    ends = core.expected_period_ends(case)
    obs = np.stack([observed[stamp] for stamp in ends])
    clim = np.stack([climatology[stamp.strftime("%m-%d")] for stamp in ends])
    shape = (4, 7) + obs.shape[1:]
    return obs.reshape(shape).mean(axis=1), clim.reshape(shape).mean(axis=1)


def load_region_weights(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    india_fraction: np.ndarray,
    all_india_weight: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Transfer official IMD homogeneous-region masks to the analysis grid.

    The four fine-grid masks are conservatively represented as target-cell
    fractions.  Inside each target cell, the frozen All-India weight is split
    among regions in proportion to those fractions.  This preserves the exact
    All-India support used by the headline scores while allowing boundary
    cells to contribute fractionally to adjacent IMD regions.
    """
    with xr.open_dataset(IMD_REGION_MASKS) as source:
        missing = [name for name in REGIONS if name not in source]
        if missing:
            raise ValueError(f"IMD region mask file is missing {missing}")
        source_lat = source.lat.values.astype(float)
        source_lon = source.lon.values.astype(float)
        masks = {
            name: source[name].values.astype(np.float64) for name in REGIONS
        }
        citation = source.attrs.get("citation", "")
        description = source.attrs.get("description", "")

    fractions = {
        name: core.remap_support_fraction(
            mask, source_lat, source_lon, target_lat, target_lon
        )
        for name, mask in masks.items()
    }
    union_fraction = np.sum(np.stack(list(fractions.values())), axis=0)
    weights: dict[str, np.ndarray] = {}
    for name, fraction in fractions.items():
        share = np.divide(
            fraction,
            union_fraction,
            out=np.zeros_like(fraction),
            where=union_fraction > 0,
        )
        region_weight = all_india_weight * share
        region_weight[(india_fraction <= 0) | (share <= 0)] = 0.0
        if np.nanmin(region_weight) < 0 or int(np.sum(region_weight > 0)) < 8:
            raise ValueError(f"invalid/too-small target support for {name}")
        weights[name] = region_weight

    allocated = np.sum(np.stack(list(weights.values())), axis=0)
    represented = union_fraction > 0
    if not np.allclose(
        allocated[represented], all_india_weight[represented], rtol=0, atol=1e-12
    ):
        raise ValueError("regional weights do not partition represented All-India support")
    audit = {
        "mask_path": str(IMD_REGION_MASKS),
        "mask_sha256": sha256_file(IMD_REGION_MASKS),
        "description": description,
        "citation": citation,
        "remapping": "conservative spherical overlap from 0.25-degree boolean masks to 1.5-degree fractional support",
        "weighting": "frozen All-India spatial weight partitioned by within-cell IMD-region fractional area",
        "target_cells_with_positive_weight": {
            name: int(np.sum(value > 0)) for name, value in weights.items()
        },
        "fractional_weight_sum_matches_all_india_where_region_union_is_represented": True,
    }
    return weights, audit


def fisher_mean(values: np.ndarray) -> float:
    clipped = np.clip(np.asarray(values, dtype=float), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def load_erpas_variable_count(
    config: dict,
    case: dict,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    land_support: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_fraction: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Load the provider ERPAS mean while auditing its date-varying source count."""
    init = pd.Timestamp(case["erpas_init"]).tz_localize(None)
    stamp = init.strftime("%Y%m%d")
    path = Path(config["model_roots"]["erpas"]) / f"annual{init.year}/tp/APCP_{stamp}.grb"
    contract = core.grib_accumulation_contract(path)
    counts = np.unique(contract["total"])
    if not (
        np.array_equal(contract["start"], np.arange(33) * 24)
        and np.array_equal(contract["end"], (np.arange(33) + 1) * 24)
        and np.all(contract["step_type"] == "accum")
        and len(counts) == 1
        and int(counts[0]) > 0
    ):
        raise ValueError(f"ERPAS {stamp} daily accumulation/source-count contract failed")
    source_count = int(counts[0])
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as source:
        field = source.tp
        if (
            field.attrs.get("GRIB_stepType") != "accum"
            or field.attrs.get("units") != "kg m**-2"
            or int(field.attrs.get("GRIB_totalNumber", -1)) != source_count
        ):
            raise ValueError(f"ERPAS {stamp} field metadata contract failed")
        if pd.Timestamp(source.time.item()) != init:
            raise ValueError(f"ERPAS {stamp} initialization time mismatch")
        selected = core.subset_box(field, "latitude", "longitude").isel(step=slice(1, 29)).load()
        ends = source.valid_time.isel(step=slice(1, 29)).values
    core.assert_periods(ends, core.expected_period_ends(case), f"ERPAS {stamp}")
    if not np.isfinite(selected.values).all():
        raise ValueError(f"ERPAS {stamp} contains missing precipitation")
    source_support = core.remap_support_fraction(
        *land_support, selected.latitude.values, selected.longitude.values
    )
    remapped, denominator, target_area, checks = core.remap_conservative(
        selected.values,
        selected.latitude.values,
        selected.longitude.values,
        target_lat,
        target_lon,
        support=source_support,
    )
    if not checks["full_target_coverage"] or float(np.nanmin(remapped)) < -1e-6:
        raise ValueError(f"ERPAS {stamp} remap/nonnegative check failed")
    weekly = remapped.reshape(4, 7, len(target_lat), len(target_lon)).sum(axis=1)
    return weekly, {
        "path": str(path),
        "source_count": source_count,
        "member_semantics": "provider precomputed unweighted source-forecast mean",
        "unit_conversion": "kg m-2 equals mm; disjoint 24-hour accumulations",
        **core.support_representation_audit(denominator, target_area, reference_fraction),
    }


def system_climatologies(
    cases: list[dict],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    land_support: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    jjases = [case for case in cases if pd.Timestamp(case["erpas_init"]).month in SEASON_WINDOWS["JJAS"]]
    fuxi_slots, fuxi_values, full_loyo_difference = accmod.load_fuxi_weekly(
        target_lat, target_lon
    )
    if full_loyo_difference >= 1e-4:
        raise ValueError("FuXi full/mean-LOYO climatology consistency failed")
    fuxi_lookup = {slot: index for index, slot in enumerate(fuxi_slots)}
    erpas_slots = sorted(
        path.name
        for path in accmod.ERPAS_CLIMO.iterdir()
        if path.is_dir() and (path / "APCP.grb").is_file()
    )
    support, support_lat, support_lon = land_support
    cache_args = (
        tuple(target_lat),
        tuple(target_lon),
        np.ascontiguousarray(support, dtype=np.float64).tobytes(),
        tuple(support_lat),
        tuple(support_lon),
        support.shape,
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    for case in jjases:
        erpas_init = pd.Timestamp(case["erpas_init"]).tz_localize(None)
        fuxi_init = pd.Timestamp(case["comparison_init"]).tz_localize(None)
        ftarget = fuxi_init.strftime("%m%d")
        fl, fr, fa = accmod.interpolation_bracket(ftarget, fuxi_slots)
        fclim = (1 - fa) * fuxi_values[fuxi_lookup[fl]] + fa * fuxi_values[fuxi_lookup[fr]]
        etarget = erpas_init.strftime("%m%d")
        el, er, ea = accmod.interpolation_bracket(etarget, erpas_slots)
        eleft = accmod.load_erpas_weekly(el, *cache_args)
        eright = eleft if el == er else accmod.load_erpas_weekly(er, *cache_args)
        output[case["case_id"]] = {
            "ERPAS": ((1 - ea) * eleft + ea * eright).astype(np.float32),
            "FuXi-S2S": fclim.astype(np.float32),
        }
    return output


def summarize(per_case: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for method in per_case.method.unique():
        method_frame = per_case[per_case.method == method]
        for season, months in SEASON_WINDOWS.items():
            subset = method_frame[method_frame.init_month.isin(months)]
            if subset.empty:
                continue
            year_groups = [("ALL", subset)] + [
                (str(year), subset[subset.year == year]) for year in (2023, 2024)
            ]
            for year_label, year_frame in year_groups:
                if year_frame.empty:
                    continue
                for (model, week), group in year_frame.groupby(["model", "week"], sort=True):
                    acc = group.acc.to_numpy(dtype=float)
                    rows.append(
                        {
                            "method": method,
                            "season": season,
                            "year": year_label,
                            "model": model,
                            "week": int(week),
                            "n_cases": int(group.case_id.nunique()),
                            "acc_mean": float(np.mean(acc)),
                            "acc_fisher_z_mean": fisher_mean(acc),
                            "acc_median": float(np.median(acc)),
                            "acc_q25": float(np.percentile(acc, 25)),
                            "acc_q75": float(np.percentile(acc, 75)),
                            "rmse_mean_mm_day": float(group.rmse_mm_day.mean()),
                            "mae_mean_mm_day": float(group.mae_mm_day.mean()),
                            "bias_mean_mm_day": float(group.bias_mm_day.mean()),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_regional(per_case: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for method in per_case.method.unique():
        method_frame = per_case[per_case.method == method]
        for season, months in SEASON_WINDOWS.items():
            subset = method_frame[method_frame.init_month.isin(months)]
            if subset.empty:
                continue
            year_groups = [("ALL", subset)] + [
                (str(year), subset[subset.year == year]) for year in (2023, 2024)
            ]
            for year_label, year_frame in year_groups:
                if year_frame.empty:
                    continue
                for (region, model, week), group in year_frame.groupby(
                    ["region", "model", "week"], sort=True
                ):
                    acc = group.acc.to_numpy(dtype=float)
                    rows.append(
                        {
                            "method": method,
                            "season": season,
                            "year": year_label,
                            "region": region,
                            "model": model,
                            "week": int(week),
                            "n_cases": int(group.case_id.nunique()),
                            "region_target_cells": int(group.region_target_cells.iloc[0]),
                            "acc_mean": float(np.mean(acc)),
                            "acc_fisher_z_mean": fisher_mean(acc),
                            "acc_median": float(np.median(acc)),
                            "acc_q25": float(np.percentile(acc, 25)),
                            "acc_q75": float(np.percentile(acc, 75)),
                            "rmse_mean_mm_day": float(group.rmse_mm_day.mean()),
                            "mae_mean_mm_day": float(group.mae_mm_day.mean()),
                            "bias_mean_mm_day": float(group.bias_mm_day.mean()),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> int:
    for directory in (HERE / "metrics", HERE / "logs", HERE / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    cases, excluded = build_cases()
    if len(cases) != 101:
        raise ValueError(f"expected 101 paired cycles, got {len(cases)}")
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["cases"] = cases
    config["model_roots"]["erpas"] = str(ERPAS_ROOT)
    config["model_roots"]["fuxi"] = str(FUXI_ROOT)

    with xr.open_dataset(SOURCE_DATA) as source:
        reference = source.load()
    target_lat, target_lon, india_fraction, weight, land_support = accmod.load_land_support(reference)
    mask = weight > 0
    if int(mask.sum()) != 171:
        raise ValueError("frozen India support is not 171 cells")
    observed, imd_climatology, india_fraction, weight, land_support, imd_audit = remap_imd(
        cases, target_lat, target_lon, land_support, mask
    )
    mask = weight > 0
    region_weights, region_audit = load_region_weights(
        target_lat, target_lon, india_fraction, weight
    )
    model_climatology = system_climatologies(
        cases, target_lat, target_lon, land_support
    )

    rows: list[dict] = []
    regional_rows: list[dict] = []
    source_qc: dict[str, dict] = {}
    for case_index, case in enumerate(cases, 1):
        erpas_init = pd.Timestamp(case["erpas_init"]).tz_localize(None)
        fuxi_init = pd.Timestamp(case["comparison_init"]).tz_localize(None)
        obs_weekly, imd_clim_weekly = weekly_reference(case, observed, imd_climatology)
        forecasts = {}
        erpas_weekly, erpas_qc = load_erpas_variable_count(
            config, case, target_lat, target_lon, land_support, india_fraction
        )
        forecasts["ERPAS"] = erpas_weekly / 7.0
        fuxi_weekly, fuxi_qc = core.load_standard_model(
            "FuXi-S2S", config, case, target_lat, target_lon, land_support, india_fraction
        )
        forecasts["FuXi-S2S"] = fuxi_weekly / 7.0
        source_qc[case["case_id"]] = {
            "ERPAS": erpas_qc,
            "FuXi-S2S": fuxi_qc,
        }
        methods = {"common_imd_1991_2020": {model: imd_clim_weekly for model in MODELS}}
        if case["case_id"] in model_climatology:
            methods["system_specific_jjas"] = model_climatology[case["case_id"]]
        for method, baselines in methods.items():
            for model in MODELS:
                for week_index in range(4):
                    forecast = forecasts[model][week_index]
                    truth = obs_weekly[week_index]
                    common_truth_anomaly = truth - imd_clim_weekly[week_index]
                    forecast_anomaly = forecast - baselines[model][week_index]
                    score = engine.error_metrics(forecast - truth, weight)
                    rows.append(
                        {
                            "method": method,
                            "case_id": case["case_id"],
                            "erpas_init": erpas_init.strftime("%Y-%m-%d"),
                            "fuxi_init": fuxi_init.strftime("%Y-%m-%d"),
                            "year": erpas_init.year,
                            "init_month": erpas_init.month,
                            "season_memberships": ";".join(
                                name for name, months in SEASON_WINDOWS.items() if name != "ALL" and erpas_init.month in months
                            ),
                            "week": week_index + 1,
                            "valid_period_start": (
                                erpas_init + pd.Timedelta(days=1 + 7 * week_index)
                            ).strftime("%Y-%m-%d"),
                            "valid_period_end_exclusive": (
                                erpas_init + pd.Timedelta(days=8 + 7 * week_index)
                            ).strftime("%Y-%m-%d"),
                            "model": model,
                            "erpas_source_count": int(erpas_qc["source_count"]),
                            "acc": engine.anomaly_correlation(
                                forecast_anomaly, common_truth_anomaly, weight
                            ),
                            "rmse_mm_day": score["rmse"],
                            "mae_mm_day": score["mae"],
                            "bias_mm_day": score["bias"],
                        }
                    )
                    for region, region_weight in region_weights.items():
                        region_score = engine.error_metrics(
                            forecast - truth, region_weight
                        )
                        regional_rows.append(
                            {
                                "method": method,
                                "case_id": case["case_id"],
                                "erpas_init": erpas_init.strftime("%Y-%m-%d"),
                                "fuxi_init": fuxi_init.strftime("%Y-%m-%d"),
                                "year": erpas_init.year,
                                "init_month": erpas_init.month,
                                "season_memberships": ";".join(
                                    name
                                    for name, months in SEASON_WINDOWS.items()
                                    if name != "ALL" and erpas_init.month in months
                                ),
                                "region": region,
                                "region_target_cells": int(np.sum(region_weight > 0)),
                                "week": week_index + 1,
                                "valid_period_start": (
                                    erpas_init
                                    + pd.Timedelta(days=1 + 7 * week_index)
                                ).strftime("%Y-%m-%d"),
                                "valid_period_end_exclusive": (
                                    erpas_init
                                    + pd.Timedelta(days=8 + 7 * week_index)
                                ).strftime("%Y-%m-%d"),
                                "model": model,
                                "erpas_source_count": int(erpas_qc["source_count"]),
                                "acc": engine.anomaly_correlation(
                                    forecast_anomaly,
                                    common_truth_anomaly,
                                    region_weight,
                                ),
                                "rmse_mm_day": region_score["rmse"],
                                "mae_mm_day": region_score["mae"],
                                "bias_mm_day": region_score["bias"],
                            }
                        )
        if case_index == 1 or case_index % 10 == 0 or case_index == len(cases):
            print(f"processed {case_index}/{len(cases)} paired cycles", flush=True)

    per_case = pd.DataFrame(rows).sort_values(
        ["method", "erpas_init", "model", "week"]
    )
    regional_per_case = pd.DataFrame(regional_rows).sort_values(
        ["method", "region", "erpas_init", "model", "week"]
    )
    summary = summarize(per_case).sort_values(
        ["method", "season", "year", "model", "week"]
    )
    regional_summary = summarize_regional(regional_per_case).sort_values(
        ["method", "season", "year", "region", "model", "week"]
    )
    comparison = summary.pivot_table(
        index=["method", "season", "year", "week", "n_cases"],
        columns="model",
        values=["acc_mean", "acc_fisher_z_mean"],
    )
    comparison.columns = [f"{metric}_{model.lower().replace('-', '_')}" for metric, model in comparison.columns]
    comparison = comparison.reset_index()
    comparison["acc_mean_fuxi_minus_erpas"] = (
        comparison["acc_mean_fuxi_s2s"] - comparison["acc_mean_erpas"]
    )
    comparison["acc_fisher_fuxi_minus_erpas"] = (
        comparison["acc_fisher_z_mean_fuxi_s2s"]
        - comparison["acc_fisher_z_mean_erpas"]
    )
    regional_comparison = regional_summary.pivot_table(
        index=[
            "method",
            "season",
            "year",
            "region",
            "week",
            "n_cases",
            "region_target_cells",
        ],
        columns="model",
        values=["acc_mean", "acc_fisher_z_mean"],
    )
    regional_comparison.columns = [
        f"{metric}_{model.lower().replace('-', '_')}"
        for metric, model in regional_comparison.columns
    ]
    regional_comparison = regional_comparison.reset_index()
    regional_comparison["acc_mean_fuxi_minus_erpas"] = (
        regional_comparison["acc_mean_fuxi_s2s"]
        - regional_comparison["acc_mean_erpas"]
    )
    regional_comparison["acc_fisher_fuxi_minus_erpas"] = (
        regional_comparison["acc_fisher_z_mean_fuxi_s2s"]
        - regional_comparison["acc_fisher_z_mean_erpas"]
    )

    per_case_path = HERE / "metrics/acc_per_case_2023_2024.csv"
    summary_path = HERE / "metrics/acc_summary_by_year_season.csv"
    comparison_path = HERE / "metrics/acc_fuxi_minus_erpas_by_year_season.csv"
    regional_per_case_path = HERE / "metrics/regional_acc_per_case_2023_2024.csv"
    regional_summary_path = HERE / "metrics/regional_acc_summary_by_year_season.csv"
    regional_comparison_path = HERE / "metrics/regional_acc_fuxi_minus_erpas.csv"
    per_case.to_csv(per_case_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    regional_per_case.to_csv(regional_per_case_path, index=False)
    regional_summary.to_csv(regional_summary_path, index=False)
    regional_comparison.to_csv(regional_comparison_path, index=False)

    checks = {
        "paired_case_count_101": per_case[per_case.method == "common_imd_1991_2020"].case_id.nunique() == 101,
        "year_counts_52_49": per_case[per_case.method == "common_imd_1991_2020"].drop_duplicates("case_id").groupby("year").size().to_dict() == {2023: 52, 2024: 49},
        "fixed_support_has_at_least_150_cells": int(mask.sum()) >= 150,
        "all_acc_in_bounds": bool(per_case.acc.between(-1, 1).all()),
        "all_errors_finite_nonnegative": bool(np.isfinite(per_case[["rmse_mm_day", "mae_mm_day"]]).all().all() and (per_case[["rmse_mm_day", "mae_mm_day"]] >= 0).all().all()),
        "system_specific_only_jjas": bool((per_case.loc[per_case.method == "system_specific_jjas", "init_month"].isin(SEASON_WINDOWS["JJAS"])).all()),
        "both_models_every_case_method_week": bool((per_case.groupby(["method", "case_id", "week"]).model.nunique() == 2).all()),
        "four_imd_regions_present": set(regional_per_case.region.unique()) == set(REGIONS),
        "regional_acc_in_bounds": bool(regional_per_case.acc.between(-1, 1).all()),
        "regional_errors_finite_nonnegative": bool(np.isfinite(regional_per_case[["rmse_mm_day", "mae_mm_day"]]).all().all() and (regional_per_case[["rmse_mm_day", "mae_mm_day"]] >= 0).all().all()),
        "both_models_every_region_case_method_week": bool((regional_per_case.groupby(["method", "region", "case_id", "week"]).model.nunique() == 2).all()),
    }
    audit = {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "paired Wednesday ERPAS and preceding-Monday strict-00Z FuXi-S2S cycles in 2023-2024",
        "paper_v2_alignment": "primary acc_mean is arithmetic mean of per-initialization spatial ACC; Fisher-z mean retained as sensitivity",
        "methods": {
            "common_imd_1991_2020": "forecast and IMD truth anomalies both subtract the same IMD 1991-2020 calendar-day climatology; available all year",
            "system_specific_jjas": "FuXi forecast subtracts native 2002-2021 model climatology; ERPAS forecast subtracts provider 20-source climatology; IMD truth subtracts IMD 1991-2020; ready only for JJAS because the standardized FuXi model-climatology NetCDF is JJAS-only",
        },
        "season_windows": {key: list(value) for key, value in SEASON_WINDOWS.items()},
        "season_window_note": "The headline four-season partition is JF, MAM, JJAS and OND. JFM is retained only as a separate paper_v2 comparison window and overlaps MAM in March; it is not part of the disjoint four-season partition.",
        "forecast_roots": {"ERPAS": str(ERPAS_ROOT), "FuXi-S2S": str(FUXI_ROOT)},
        "erpas_source_count_distribution": {
            str(int(key)): int(value)
            for key, value in per_case[
                per_case.method == "common_imd_1991_2020"
            ].drop_duplicates("case_id").erpas_source_count.value_counts().sort_index().items()
        },
        "erpas_source_count_note": "ERPAS is a provider-precomputed deterministic mean; GRIB numberOfForecastsInEnsemble varies by initialization and is retained explicitly in the per-case CSV.",
        "forecast_climatologies": {
            "FuXi-S2S": str(accmod.FUXI_CLIMO),
            "ERPAS": str(accmod.ERPAS_CLIMO),
        },
        "truth": str(IMD_ROOT),
        "truth_climatology": str(IMD_CLIMO),
        "truth_climatology_sha256": sha256_file(IMD_CLIMO),
        "grid": f"22x22 at 1.5 degrees; frozen all-season India support with {int(mask.sum())} cells",
        "spatial_weighting": "spherical target-cell area multiplied by frozen IMD India fraction",
        "case_aggregation": "paper_v2 arithmetic mean primary; Fisher-z mean also provided",
        "excluded_missing_pairs": excluded,
        "imd": imd_audit,
        "imd_homogeneous_regions": region_audit,
        "checks": checks,
        "outputs": [str(per_case_path), str(summary_path), str(comparison_path), str(regional_per_case_path), str(regional_summary_path), str(regional_comparison_path)],
    }
    (HERE / "logs/method_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": audit["status"], "checks": checks}, indent=2))
    print(f"wrote {per_case_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {comparison_path}")
    print(f"wrote {regional_per_case_path}")
    print(f"wrote {regional_summary_path}")
    print(f"wrote {regional_comparison_path}")
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
