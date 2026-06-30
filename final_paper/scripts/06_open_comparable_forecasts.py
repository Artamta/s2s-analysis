#!/usr/bin/env python3
"""
Open comparable forecast initializations in a metric-ready shape.

The loader normalizes supported forecast products to:

    member, lead, lat, lon

It opens all available ensemble members lazily where possible. The command-line
mode is a smoke test and metadata writer; the functions are intended to be
imported by the later ACC/RMSE computation script.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cfgrib
import dask.array as dask_array
import numpy as np
import pandas as pd
import xarray as xr
import zarr


DEFAULT_DATA_ROOT = Path("/storage/raj.ayush/All_Model_Data")
DEFAULT_OUTPUT_ROOT = Path("/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs")
SUPPORTED_MODELS = ("delysm", "ecmwf", "ukmo", "ncep", "fuxi")


VARIABLE_ALIASES = {
    "z": "z500",
    "gh": "z500",
    "gh500": "z500",
    "geopotential_height_500": "z500",
    "precip": "tp",
    "rain": "tp",
    "precipitation": "tp",
    "2t": "t2m",
}


@dataclass
class ForecastOpenResult:
    season: str
    set_name: str
    init_date: str
    model: str
    variable: str
    status: str
    data: xr.DataArray | None
    paths: list[Path]
    notes: str = ""

    def summary_row(self, load_sample: bool = False) -> dict:
        data = self.data
        sample_value = ""
        sample_status = ""
        if data is not None and load_sample:
            try:
                sample = data.isel(member=0, lead=0, lat=0, lon=0).compute()
                sample_value = float(sample.values)
                sample_status = "sample_loaded"
            except Exception as exc:
                sample_status = f"sample_load_failed={type(exc).__name__}: {exc}"
        return {
            "season": self.season,
            "set_name": self.set_name,
            "init_date": self.init_date,
            "model": self.model,
            "variable": self.variable,
            "status": self.status,
            "paths": ";".join(str(p) for p in self.paths),
            "dims": ",".join(data.dims) if data is not None else "",
            "shape": "x".join(map(str, data.shape)) if data is not None else "",
            "member_count": int(data.sizes.get("member", 0)) if data is not None else 0,
            "lead_count": int(data.sizes.get("lead", 0)) if data is not None else 0,
            "lead_min": float(np.nanmin(data["lead"].values)) if data is not None and "lead" in data.coords and data.sizes.get("lead", 0) else "",
            "lead_max": float(np.nanmax(data["lead"].values)) if data is not None and "lead" in data.coords and data.sizes.get("lead", 0) else "",
            "lat_count": int(data.sizes.get("lat", 0)) if data is not None else 0,
            "lon_count": int(data.sizes.get("lon", 0)) if data is not None else 0,
            "units": data.attrs.get("units", "") if data is not None else "",
            "standard_dims": "member,lead,lat,lon" if data is not None else "",
            "sample_status": sample_status,
            "sample_value": sample_value,
            "notes": self.notes,
        }


def canonical_variable(variable: str) -> str:
    key = variable.strip()
    return VARIABLE_ALIASES.get(key.lower(), key)


def get_matched_init_dates(output_root: Path, season: str, set_name: str) -> list[str]:
    path = output_root / season / "02_processed" / "matched_init" / f"{set_name}_init_dates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Matched-init file not found: {path}")
    df = pd.read_csv(path, dtype={"init_date": str})
    return df["init_date"].dropna().astype(str).tolist()


def get_models_for_set(output_root: Path, season: str, set_name: str) -> list[str]:
    path = output_root / "common" / "inventory" / "forecast_comparable_init_sets.csv"
    if not path.exists():
        return list(SUPPORTED_MODELS)
    df = pd.read_csv(path)
    match = df[(df["season"].astype(str) == season) & (df["set_name"].astype(str) == set_name)]
    if match.empty:
        return list(SUPPORTED_MODELS)
    models = str(match.iloc[0]["models"])
    if models == "nan":
        return []
    return [m for m in models.split(";") if m in SUPPORTED_MODELS]


def hours_from_coord(values: np.ndarray) -> np.ndarray:
    vals = np.array(values).reshape(-1)
    if vals.size == 0:
        return vals.astype(float)
    if np.issubdtype(vals.dtype, np.timedelta64):
        return (vals / np.timedelta64(1, "h")).astype(float)
    return vals.astype(float)


def normalize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    return da


def normalize_lead(da: xr.DataArray, ds: xr.Dataset | None = None) -> xr.DataArray:
    if "step" in da.dims:
        lead_hours = hours_from_coord(da["step"].values)
        da = da.rename({"step": "lead"})
        da = da.assign_coords(lead=lead_hours)
    elif "lead_time" in da.dims:
        lead_vals = np.array(da["lead_time"].values).reshape(-1)
        if np.issubdtype(lead_vals.dtype, np.timedelta64):
            lead = hours_from_coord(lead_vals)
            units = "hours"
        else:
            lead = lead_vals.astype(float) * 24.0
            units = "hours_from_days"
        da = da.rename({"lead_time": "lead"})
        da = da.assign_coords(lead=lead)
        da.attrs["lead_units_note"] = units
    else:
        step_source = None
        if "step" in da.coords:
            step_source = da.coords["step"].values
        elif ds is not None and "step" in ds.variables:
            step_source = ds.variables["step"][:]
        if step_source is not None:
            lead = hours_from_coord(step_source)
            if lead.size == 0:
                lead = np.array([0.0])
        else:
            lead = np.array([0.0])
        da = da.expand_dims(lead=lead)
    return da


def ensure_member_dim(da: xr.DataArray, control: bool = False) -> xr.DataArray:
    if "number" in da.dims:
        da = da.rename({"number": "member"})
    elif "ensemble" in da.dims:
        da = da.rename({"ensemble": "member"})
    elif "member" not in da.dims:
        da = da.expand_dims(member=[0 if control else 0])
    if control:
        da = da.assign_coords(member=[0])
    return da


def finalize_standard_da(da: xr.DataArray, standard_name: str, units: str | None = None) -> xr.DataArray:
    da = normalize_lat_lon(da)
    missing = [dim for dim in ("member", "lead", "lat", "lon") if dim not in da.dims]
    if missing:
        raise ValueError(f"Cannot standardize {standard_name}; missing dims {missing}; got dims {da.dims}")
    da = da.transpose("member", "lead", "lat", "lon")
    da.name = standard_name
    da.attrs["standard_variable"] = standard_name
    if units is not None:
        da.attrs["units"] = units
    return da


def variable_for_operational(variable: str) -> tuple[str, str, str]:
    var = canonical_variable(variable)
    if var == "tp":
        return "tp", "tp", "tp"
    if var == "msl":
        return "msl", "msl", "msl"
    if var == "t2m":
        return "2t", "t2m", "t2m"
    if var == "z500":
        return "z/500", "gh", "z500"
    if var in {"mx2t6", "mn2t6"}:
        return "surface", var, var
    raise KeyError(f"Unsupported operational variable: {variable}")


def variable_for_ncep(variable: str) -> tuple[str, str, str]:
    var = canonical_variable(variable)
    if var == "tp":
        return "surface", "tp", "tp"
    if var == "z500":
        return "z/500", "gh", "z500"
    if var in {"mx2t6", "mn2t6"}:
        return "surface", var, var
    raise KeyError(f"NCEP does not provide requested variable in this tree: {variable}")


def open_delysm(data_root: Path, season: str, init_date: str, variable: str, nonnegative_leads: bool) -> tuple[xr.DataArray, list[Path]]:
    var = canonical_variable(variable)
    store = data_root / "delysm" / season / init_date / "india" / "forecast.zarr"
    if not store.exists():
        raise FileNotFoundError(store)
    group = zarr.open_group(str(store), mode="r")
    if var not in group.array_keys():
        raise KeyError(f"DLESyM variable not found: {var}")
    raw = dask_array.from_zarr(group[var])
    raw = raw[:, 0, :, :, :]
    lead = np.array(group["lead_time"][:])
    lead_hours = hours_from_coord(lead)
    if nonnegative_leads:
        keep = np.where(lead_hours >= 0)[0]
        lead_hours = lead_hours[keep]
        raw = raw[:, keep, :, :]
    member = np.array(group["ensemble"][:])
    lat = np.array(group["lat"][:])
    lon = np.array(group["lon"][:])
    da = xr.DataArray(
        raw,
        dims=("member", "lead", "lat", "lon"),
        coords={"member": member, "lead": lead_hours, "lat": lat, "lon": lon},
        name=var,
        attrs={"source_model": "delysm"},
    )
    # Guard: incomplete DLESyM stores (missing chunks) read back as fill-value
    # zeros instead of erroring (QC flagged e.g. jjas2019/20190603). Refuse the
    # store rather than silently score an all-zero/all-NaN field.
    sample = np.asarray(da.isel(member=0, lead=0).values, dtype=float)
    if not np.isfinite(sample).any() or float(np.nanmax(np.abs(sample))) == 0.0:
        raise ValueError(f"DLESyM store reads as all-zero/empty (likely missing chunks): {store}")
    return finalize_standard_da(da, var), [store]


def open_netcdf_control_perturbed(
    data_root: Path,
    model: str,
    season: str,
    init_date: str,
    variable: str,
) -> tuple[xr.DataArray, list[Path]]:
    product, source_var, standard_var = variable_for_operational(variable)
    root = data_root / model / season / product
    cf = root / f"{init_date}_cf.nc"
    pf = root / f"{init_date}_pf.nc"
    paths = [p for p in (cf, pf) if p.exists()]
    if not paths:
        raise FileNotFoundError(f"No {model} files found for {season} {init_date} {variable}: {root}")

    arrays = []
    for path, control in ((cf, True), (pf, False)):
        if not path.exists():
            continue
        ds = xr.open_dataset(path, chunks={})
        if source_var not in ds:
            raise KeyError(f"{source_var} not found in {path}")
        da = ds[source_var]
        da = ensure_member_dim(da, control=control)
        da = normalize_lead(da, ds)
        da = normalize_lat_lon(da)
        da.attrs.update(ds[source_var].attrs)
        arrays.append(da)
    out = xr.concat(arrays, dim="member").sortby("member")
    return finalize_standard_da(out, standard_var, arrays[0].attrs.get("units")), paths


def pick_grib_dataset(path: Path, source_var: str) -> xr.Dataset:
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    for ds in datasets:
        if source_var in ds.data_vars:
            return ds
    available = [list(ds.data_vars) for ds in datasets]
    raise KeyError(f"{source_var} not found in {path}; available={available}")


def open_ncep(data_root: Path, season: str, init_date: str, variable: str) -> tuple[xr.DataArray, list[Path]]:
    product, source_var, standard_var = variable_for_ncep(variable)
    root = data_root / "ncep" / season / product
    cf = root / "cf" / f"{init_date}.grib"
    pf = root / "pf" / f"{init_date}.grib"
    paths = [p for p in (cf, pf) if p.exists()]
    if not paths:
        raise FileNotFoundError(f"No NCEP GRIB files found for {season} {init_date} {variable}: {root}")
    arrays = []
    for path, control in ((cf, True), (pf, False)):
        if not path.exists():
            continue
        ds = pick_grib_dataset(path, source_var)
        da = ds[source_var]
        da = ensure_member_dim(da, control=control)
        da = normalize_lead(da, ds)
        da = normalize_lat_lon(da)
        arrays.append(da)
    out = xr.concat(arrays, dim="member").sortby("member")
    return finalize_standard_da(out, standard_var, arrays[0].attrs.get("units")), paths


def open_fuxi(data_root: Path, season: str, init_date: str, variable: str) -> tuple[xr.DataArray, list[Path]]:
    var = canonical_variable(variable)
    path = data_root / "fuxi" / season / "combined" / f"{init_date}.nc"
    if not path.exists() and season == "jjas2019":
        path = Path("/storage/raj.ayush/s2s_final_data/jjas/fuxi_combined") / f"{init_date}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path, chunks={"member": 1, "lead_time": 1})
    if "forecast" not in ds:
        raise KeyError(f"forecast variable missing in {path}")
    channels = [str(x) for x in ds["channel"].values]
    if var not in channels:
        raise KeyError(f"FuXi channel not found: {var}; available first channels={channels[:10]}")
    da = ds["forecast"].sel(channel=var).drop_vars("channel")
    da = normalize_lead(da, ds)
    da = normalize_lat_lon(da)
    return finalize_standard_da(da, var), [path]


def open_forecast(
    model: str,
    season: str,
    init_date: str,
    variable: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nonnegative_leads: bool = True,
) -> tuple[xr.DataArray, list[Path]]:
    model = model.lower()
    if model == "delysm":
        return open_delysm(data_root, season, init_date, variable, nonnegative_leads)
    if model in {"ecmwf", "ukmo"}:
        return open_netcdf_control_perturbed(data_root, model, season, init_date, variable)
    if model == "ncep":
        return open_ncep(data_root, season, init_date, variable)
    if model == "fuxi":
        return open_fuxi(data_root, season, init_date, variable)
    raise KeyError(f"Unsupported model for opening forecasts: {model}")


def open_comparable_forecasts(
    season: str,
    set_name: str,
    variable: str,
    models: Iterable[str] | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_inits: int | None = None,
    nonnegative_leads: bool = True,
) -> Iterator[ForecastOpenResult]:
    init_dates = get_matched_init_dates(output_root, season, set_name)
    if max_inits is not None:
        init_dates = init_dates[:max_inits]
    selected_models = list(models) if models is not None else get_models_for_set(output_root, season, set_name)
    for init_date in init_dates:
        for model in selected_models:
            try:
                data, paths = open_forecast(
                    model=model,
                    season=season,
                    init_date=init_date,
                    variable=variable,
                    data_root=data_root,
                    nonnegative_leads=nonnegative_leads,
                )
                yield ForecastOpenResult(season, set_name, init_date, model, canonical_variable(variable), "opened", data, paths)
            except Exception as exc:
                yield ForecastOpenResult(
                    season=season,
                    set_name=set_name,
                    init_date=init_date,
                    model=model,
                    variable=canonical_variable(variable),
                    status="failed",
                    data=None,
                    paths=[],
                    notes=f"{type(exc).__name__}: {exc}",
                )


def write_smoke_summary(rows: list[dict], output_root: Path, season: str, variable: str, set_name: str) -> Path:
    out_common = output_root / "common" / "inventory" / f"forecast_open_smoke_{season}_{set_name}_{variable}.csv"
    out_latest = output_root / "common" / "inventory" / "forecast_open_smoke_latest.csv"
    out_common.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_common, index=False)
    frame.to_csv(out_latest, index=False)

    out_season = output_root / season / "01_qc" / "metadata" / f"open_smoke_{set_name}_{variable}.csv"
    out_season.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_season, index=False)

    manifest = {
        "common_summary": str(out_common),
        "latest_summary": str(out_latest),
        "season_summary": str(out_season),
        "season": season,
        "set_name": set_name,
        "variable": variable,
    }
    (out_season.parent / f"open_smoke_{set_name}_{variable}.json").write_text(json.dumps(manifest, indent=2))
    return out_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", choices=["jfm2026", "jjas2019"], default="jjas2019")
    parser.add_argument("--set-name", default="delysm_operational")
    parser.add_argument("--variable", default="z500")
    parser.add_argument("--models", nargs="*", default=None, help="Models to open. Default comes from comparable set.")
    parser.add_argument("--max-inits", type=int, default=1, help="Limit init dates for smoke testing. Use 0 for all.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-negative-delysm-leads", action="store_true")
    parser.add_argument("--load-sample", action="store_true", help="Actually load first member/lead/grid value for each opened forecast.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_inits = None if args.max_inits == 0 else args.max_inits
    rows = []
    print(
        f"Opening forecasts: season={args.season} set={args.set_name} "
        f"variable={args.variable} max_inits={max_inits if max_inits is not None else 'all'}"
    )
    for result in open_comparable_forecasts(
        season=args.season,
        set_name=args.set_name,
        variable=args.variable,
        models=args.models,
        data_root=args.data_root,
        output_root=args.output_root,
        max_inits=max_inits,
        nonnegative_leads=not args.include_negative_delysm_leads,
    ):
        row = result.summary_row(load_sample=args.load_sample)
        rows.append(row)
        shape = row["shape"] or "-"
        print(f"  {result.init_date} {result.model:12s} {result.status:7s} shape={shape} {row['notes']}")
    out = write_smoke_summary(rows, args.output_root, args.season, canonical_variable(args.variable), args.set_name)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
