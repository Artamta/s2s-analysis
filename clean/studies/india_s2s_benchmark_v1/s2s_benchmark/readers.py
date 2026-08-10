from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .core import COMMON_LAT, COMMON_LON, StandardField
from .remap import conservative_remap, intensive_remap


EXPERIMENT_IDS = {
    "ecmwf": "physics/ecmwf_operational_2020_2025",
    "ukmo": "physics/ukmo_operational_2020_2025",
    "cma": "physics/cma_operational_2020_2025",
    "ncep": "physics/ncep_operational_2020_2025",
    "cnrm": "physics/cnrm_operational_2020_2025",
    "fuxi_s2s": "model-run/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50",
    "dlesym_v0": "model-run/dlesym/dlesym_v0_isccp_era5_tpdiag_t2m_00z_2020_2024_ens1",
    "dlesym_v1": "model-run/dlesym/dlesym_v1_era5_t2m_00z_2020_2024_ens4",
    "neuralgcm": "model-run/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2020_2024_ens10",
    "fcn3": "model-run/fcn3/fcn3_v1_t2m_00z_2020_2024_ens3",
    "erpas": "provider/erpas_forecast_2023_2025",
}

MODEL_VARIABLES = {
    "ecmwf": ("tp", "t2m"),
    "ukmo": ("tp", "t2m"),
    "cma": ("tp", "t2m"),
    "ncep": ("tp", "t2m_proxy"),
    "cnrm": ("tp", "t2m"),
    "fuxi_s2s": ("tp", "t2m"),
    "dlesym_v0": ("tp", "t2m"),
    "dlesym_v1": ("t2m",),
    "neuralgcm": ("tp",),
    "fcn3": ("t2m",),
    "erpas": ("tp", "tp_india_0p5", "tsfc", "gh"),
}


def _open(path: str) -> xr.Dataset:
    if Path(path).suffix.lower() in {".grib", ".grb"}:
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    return xr.open_dataset(path)


def _files_for(case: dict[str, Any], variable: str) -> list[dict[str, Any]]:
    if variable == "t2m_proxy":
        return [item for item in case["files"] if item["variable"] == "surface"]
    source_name = {"tsfc": "surface_temperature", "gh": "geopotential_height"}.get(variable, variable)
    matches = []
    for item in case["files"]:
        declared = item["variable"].split(",")
        manifest_names = item.get("manifest_variables") or list((item.get("manifest_fields") or {}).keys())
        if (
            item["variable"] == source_name
            or source_name in declared
            or source_name in manifest_names
            or (source_name == "tp" and item["variable"] == "surface")
        ):
            matches.append(item)
    return matches


def _member_array(da: xr.DataArray, fallback: int) -> tuple[np.ndarray, np.ndarray]:
    if "number" in da.dims:
        return da.transpose("number", "step", "latitude", "longitude").values, da.number.values
    if "member" in da.dims:
        trailing = [dim for dim in da.dims if dim not in {"member", "lead_day"}]
        return da.transpose("member", "lead_day", *trailing).values, da.member.values
    step = "step" if "step" in da.dims else "lead_day"
    trailing = [dim for dim in da.dims if dim != step]
    return da.transpose(step, *trailing).values[np.newaxis, ...], np.array([fallback])


def _physics_field(
    model: str, variable: str, initialization: str, files: list[dict[str, Any]], lead_limit: int
) -> StandardField:
    arrays: list[np.ndarray] = []
    members: list[np.ndarray] = []
    lat = lon = None
    for file_index, item in enumerate(sorted(files, key=lambda x: x["forecast_type"])):
        with _open(item["path"]) as ds:
            if variable == "t2m_proxy":
                raw = ((ds["mx2t6"] + ds["mn2t6"]) / 2).values
                if "number" in ds["mx2t6"].dims:
                    raw = ((ds["mx2t6"] + ds["mn2t6"]) / 2).transpose(
                        "number", "step", "latitude", "longitude"
                    ).values
                    ids = ds.number.values
                else:
                    raw = raw[np.newaxis, ...]
                    ids = np.array([0])
                raw = raw[:, : lead_limit * 4].reshape(
                    raw.shape[0], lead_limit, 4, raw.shape[-2], raw.shape[-1]
                ).mean(axis=2) - 273.15
            else:
                raw, ids = _member_array(ds[variable], 0)
                if variable == "tp":
                    if model == "ncep":
                        raw = raw[:, 3 : lead_limit * 4 : 4]
                    else:
                        raw = raw[:, :lead_limit]
                    raw = np.diff(
                        np.concatenate([np.zeros_like(raw[:, :1]), raw], axis=1), axis=1
                    )
                else:
                    raw = raw[:, :lead_limit] - 273.15
            arrays.append(raw.astype(np.float32))
            if item["forecast_type"] == "cf":
                members.append(np.array([0], dtype=np.int16))
            else:
                members.append(np.asarray(ids, dtype=np.int16))
            lat = ds.latitude.values
            lon = ds.longitude.values
    values = np.concatenate(arrays, axis=0)
    member = np.concatenate(members)
    order = np.argsort(member)
    return StandardField(
        model=model,
        experiment_id=EXPERIMENT_IDS[model],
        variable=variable,
        initialization=initialization,
        values=values[order],
        member=member[order],
        lead_day=np.arange(1, lead_limit + 1),
        latitude=np.asarray(lat),
        longitude=np.asarray(lon),
        units="mm day-1" if variable == "tp" else "degC",
        temporal_statistic="daily_mean_rate" if variable == "tp" else (
            "daily_proxy_mean" if variable == "t2m_proxy" else "daily_mean"
        ),
        distribution_representation="members",
        source_paths=tuple(item["path"] for item in files),
        attrs={"source_grid_equals_common": True},
    )


def _native_ai_field(
    model: str, variable: str, initialization: str, item: dict[str, Any], lead_limit: int
) -> StandardField:
    with _open(item["path"]) as ds:
        raw_name = variable
        da = ds[raw_name].isel(lead_day=slice(0, lead_limit))
        values, members = _member_array(da, 0)
        units = da.attrs.get("units", "")
        if model == "fuxi_s2s" and variable == "tp":
            if units != "mm h-1":
                raise ValueError(f"unexpected FuXi TP units: {units}")
            values = values * 24.0
            units = "mm day-1"
        elif model == "fuxi_s2s" and variable == "t2m":
            if units != "K":
                raise ValueError(f"unexpected FuXi T2M units: {units}")
            values = values - 273.15
            units = "degC"
        expected_units = "mm day-1" if variable == "tp" else "degC"
        if units != expected_units:
            raise ValueError(f"{model}/{variable}: expected {expected_units}, got {units}")
        lat = ds.latitude.values
        lon = ds.longitude.values
        source_attrs = dict(ds.attrs)
    representation = "deterministic" if len(members) == 1 else "members"
    return StandardField(
        model=model,
        experiment_id=EXPERIMENT_IDS[model],
        variable=variable,
        initialization=initialization,
        values=values.astype(np.float32),
        member=np.asarray(members),
        lead_day=np.arange(1, lead_limit + 1),
        latitude=np.asarray(lat),
        longitude=np.asarray(lon),
        units=expected_units,
        temporal_statistic="daily_mean_rate" if variable == "tp" else "daily_mean",
        distribution_representation=representation,
        source_paths=(item["path"],),
        attrs={
            "source_grid_equals_common": True,
            "source_model_run_label": source_attrs.get("run_label"),
            "source_native_grid": source_attrs.get("native_grid"),
        },
    )


def _erpas_field(
    variable: str,
    initialization: str,
    item: dict[str, Any],
    lead_limit: int,
    grid: str,
    native_box: dict[str, float],
) -> StandardField:
    raw_name = {"tsfc": "t", "gh": "gh"}.get(variable, "tp")
    with _open(item["path"]) as ds:
        da = ds[raw_name].isel(step=slice(0, lead_limit))
        pressure = ds.isobaricInhPa.values if "isobaricInhPa" in da.dims else None
        if pressure is not None:
            values = da.transpose("step", "isobaricInhPa", "latitude", "longitude").values[np.newaxis]
        else:
            values = da.transpose("step", "latitude", "longitude").values[np.newaxis]
        lat = ds.latitude.values
        lon = ds.longitude.values
        raw_units = da.attrs.get("units")
        step_type = da.attrs.get("GRIB_stepType")
        provider_members = da.attrs.get("GRIB_totalNumber")
    if provider_members is None or int(provider_members) < 1:
        raise ValueError("ERPAS must declare a positive GRIB_totalNumber")
    if variable == "tsfc":
        if raw_units != "K" or step_type != "instant":
            raise ValueError("ERPAS tsfc must be instantaneous kelvin")
        values = values - 273.15
        units = "degC"
        statistic = "instantaneous_daily_sample"
    elif variable.startswith("tp"):
        if raw_units != "kg m**-2":
            raise ValueError(f"unexpected ERPAS TP units: {raw_units}")
        units = "mm day-1"
        statistic = "daily_accumulation_as_mean_rate"
    else:
        if raw_units != "gpm" or step_type != "instant":
            raise ValueError("ERPAS gh must be instantaneous gpm")
        units = "gpm"
        statistic = "instantaneous_daily_sample"
    lat_keep = (lat >= native_box["latitude_min"]) & (lat <= native_box["latitude_max"])
    lon_keep = (lon >= native_box["longitude_min"]) & (lon <= native_box["longitude_max"])
    native_values = values[..., lat_keep, :][..., :, lon_keep]
    native_lat = lat[lat_keep]
    native_lon = lon[lon_keep]
    attrs: dict[str, Any] = {
        "source_grid_equals_common": False,
        "source_grib_step_type": step_type,
        "source_member_note": "provider-precomputed unweighted ensemble mean; raw members unavailable",
    }
    if grid == "source_native_india":
        final_values, final_lat, final_lon = native_values, native_lat, native_lon
    elif grid == "common_1p5":
        remapper = conservative_remap if variable.startswith("tp") else intensive_remap
        final_values, audit = remapper(values, lat, lon, COMMON_LAT, COMMON_LON)
        final_lat, final_lon = COMMON_LAT, COMMON_LON
        attrs["remap_audit"] = str(audit)
        attrs["horizontal_remapping"] = (
            "spherical_conservative_cell_overlap" if variable.startswith("tp") else "bilinear"
        )
    else:
        raise ValueError(f"unsupported ERPAS grid: {grid}")
    return StandardField(
        model="erpas",
        experiment_id=EXPERIMENT_IDS["erpas"] + ("/india_0p5_sensitivity" if variable == "tp_india_0p5" else ""),
        variable="tp" if variable == "tp_india_0p5" else variable,
        initialization=initialization,
        values=final_values.astype(np.float32),
        member=np.array([0], dtype=np.int16),
        lead_day=np.arange(1, lead_limit + 1),
        latitude=np.asarray(final_lat),
        longitude=np.asarray(final_lon),
        units=units,
        temporal_statistic=statistic,
        distribution_representation="mean_only",
        source_paths=(item["path"],),
        source_ensemble_size=int(provider_members),
        pressure_hpa=pressure,
        attrs=attrs,
    )


def load_field(
    inventory: dict[str, Any],
    model: str,
    variable: str,
    initialization: str,
    lead_limit: int,
    grid: str,
    native_box: dict[str, float],
) -> StandardField:
    if variable not in MODEL_VARIABLES[model]:
        raise ValueError(f"{variable} is not registered for {model}")
    experiment = inventory["experiments"][EXPERIMENT_IDS[model]]
    case = experiment["cases"].get(initialization)
    if case is None:
        raise FileNotFoundError(f"{model} has no {initialization} case")
    files = _files_for(case, variable)
    if not files:
        raise FileNotFoundError(f"{model}/{variable}/{initialization}: no source file")
    if model in {"ecmwf", "ukmo", "cma", "ncep", "cnrm"}:
        if grid != "common_1p5":
            raise ValueError(f"{model} source archive already equals common_1p5")
        return _physics_field(model, variable, initialization, files, lead_limit)
    if model == "erpas":
        return _erpas_field(variable, initialization, files[0], lead_limit, grid, native_box)
    if grid != "common_1p5":
        raise ValueError(f"{model} retained archive already equals common_1p5")
    return _native_ai_field(model, variable, initialization, files[0], lead_limit)
