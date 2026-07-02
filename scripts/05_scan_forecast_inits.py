#!/usr/bin/env python3
"""
Scan forecast availability and comparable initialization dates.

This is the first lightweight pre-compute script for the benchmark workflow. It
checks forecast stores/files without loading full forecast arrays, records
sample metadata, and writes matched/common initialization-date tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import netCDF4 as nc
import numpy as np
import pandas as pd
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("S2S_DATA_ROOT", "/storage/raj.ayush/All_Model_Data"))
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "S2S_OUTPUT_ROOT",
        str(PROJECT_ROOT / "outputs" / "verification"),
    )
)
SEASONS = ("jfm2026", "jjas2019")
MODELS = ("delysm", "ecmwf", "ukmo", "ncep", "fuxi")
OPERATIONAL_MODELS = ("ecmwf", "ukmo", "ncep")
AI_MODELS = ("delysm", "fuxi")


@dataclass(frozen=True)
class Product:
    label: str
    path: Path
    format: str


def ymd_from_text(text: str) -> str | None:
    match = re.search(r"(\d{8})", text)
    return match.group(1) if match else None


def date_entries(root: Path) -> list[str]:
    if not root.exists():
        return []
    with os.scandir(root) as entries:
        return sorted(entry.name for entry in entries if entry.name.isdigit() and len(entry.name) == 8)


def file_size_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def has_file_with_suffix(root: Path, suffix: str) -> bool:
    if not root.exists():
        return False
    with os.scandir(root) as entries:
        return any(entry.name.endswith(suffix) for entry in entries)


def list_date_files(root: Path, suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.name.endswith(suffix):
                continue
            init_date = ymd_from_text(entry.name)
            if init_date:
                out[init_date] = Path(entry.path)
    return out


def list_netcdf_stream_files(root: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    if not root.exists():
        return out
    pattern = re.compile(r"^(\d{8})_(cf|pf)\.nc$")
    with os.scandir(root) as entries:
        for entry in entries:
            match = pattern.match(entry.name)
            if match:
                init_date, stream = match.groups()
                out.setdefault(init_date, {})[stream] = Path(entry.path)
    return out


def summarize_dates(dates: Iterable[str]) -> dict[str, str | int]:
    unique = sorted(set(dates))
    if not unique:
        return {"count": 0, "first": "", "last": "", "frequency": ""}
    dts = [datetime.strptime(d, "%Y%m%d").date() for d in unique]
    diffs = [(b - a).days for a, b in zip(dts, dts[1:])]
    weekdays = sorted({d.weekday() for d in dts})
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if diffs and all(d == 1 for d in diffs):
        frequency = "daily"
    elif diffs and set(weekdays) == {0, 3} and all(d in (3, 4) for d in diffs):
        frequency = "Monday/Thursday"
    elif len(unique) == 1:
        frequency = "single init"
    else:
        frequency = "irregular"
        if diffs:
            frequency += f"; gaps={','.join(map(str, sorted(set(diffs))))}"
        frequency += f"; weekdays={','.join(weekday_names[i] for i in weekdays)}"
    return {"count": len(unique), "first": unique[0], "last": unique[-1], "frequency": frequency}


def values_1d(value: np.ndarray | object) -> np.ndarray:
    arr = np.array(value)
    return arr.reshape(-1)


def grid_summary(lat: np.ndarray, lon: np.ndarray) -> str:
    lat = values_1d(lat)
    lon = values_1d(lon)
    if lat.size == 0 or lon.size == 0:
        return ""
    dy = abs(float(lat[1] - lat[0])) if lat.size > 1 else np.nan
    dx = abs(float(lon[1] - lon[0])) if lon.size > 1 else np.nan
    return (
        f"{lat.size}x{lon.size}; "
        f"lat {float(np.nanmin(lat)):.2f}..{float(np.nanmax(lat)):.2f}; "
        f"lon {float(np.nanmin(lon)):.2f}..{float(np.nanmax(lon)):.2f}; "
        f"dx~{dx:g}, dy~{dy:g} deg"
    )


def lead_summary_hours(hours: np.ndarray) -> tuple[int, str, str]:
    vals = values_1d(hours).astype(float)
    if vals.size == 0:
        return 0, "", ""
    if vals.size > 1:
        diffs = np.unique(np.diff(vals))
        step = f"{diffs[0]:g} h" if diffs.size == 1 else "mixed"
    else:
        step = "single"
    return int(vals.size), f"{float(vals.min()):g}..{float(vals.max()):g} h", step


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_zarr_metadata(path: Path, model: str, season: str, product: str) -> list[dict]:
    rows: list[dict] = []
    group = zarr.open_group(str(path), mode="r")
    coord_names = {"ensemble", "member", "lat", "lon", "latitude", "longitude", "lead_time", "time"}
    arrays = list(group.array_keys())
    variables = sorted(k for k in arrays if k not in coord_names)
    lat_name = "lat" if "lat" in arrays else "latitude"
    lon_name = "lon" if "lon" in arrays else "longitude"
    lat = np.array(group[lat_name][:]) if lat_name in arrays else np.array([])
    lon = np.array(group[lon_name][:]) if lon_name in arrays else np.array([])
    grid = grid_summary(lat, lon)
    member_name = "ensemble" if "ensemble" in arrays else "member"
    member_count = group[member_name].shape[0] if member_name in arrays else ""
    lead_count = ""
    lead_range = ""
    lead_step = ""
    if "lead_time" in arrays:
        lead = np.array(group["lead_time"][:])
        if np.issubdtype(lead.dtype, np.timedelta64):
            lead = lead / np.timedelta64(1, "h")
        lead_count, lead_range, lead_step = lead_summary_hours(lead)
    for variable in variables:
        arr = group[variable]
        rows.append({
            "season": season,
            "model": model,
            "product": product,
            "variable": variable,
            "format": "zarr",
            "sample_path": str(path),
            "read_ok": True,
            "dims": "unknown",
            "shape": "x".join(map(str, arr.shape)),
            "units": "",
            "member_count": member_count,
            "lead_count": lead_count,
            "lead_range": lead_range,
            "lead_step": lead_step,
            "grid": grid,
            "notes": "Zarr metadata read only; full data arrays were not loaded.",
        })
    return rows


def read_netcdf_metadata(path: Path, model: str, season: str, product: str) -> list[dict]:
    rows: list[dict] = []
    with nc.Dataset(path) as ds:
        coord_names = {
            "number", "time", "step", "heightAboveGround", "meanSea", "surface",
            "latitude", "longitude", "lat", "lon", "valid_time", "isobaricInhPa",
            "pressure_level", "level", "member", "lead_time", "channel", "init_time",
        }
        lat_name = "latitude" if "latitude" in ds.variables else "lat"
        lon_name = "longitude" if "longitude" in ds.variables else "lon"
        lat = np.array(ds.variables[lat_name][:]) if lat_name in ds.variables else np.array([])
        lon = np.array(ds.variables[lon_name][:]) if lon_name in ds.variables else np.array([])
        grid = grid_summary(lat, lon)
        member_count = ""
        if "number" in ds.dimensions:
            member_count = len(ds.dimensions["number"])
        elif "member" in ds.dimensions:
            member_count = len(ds.dimensions["member"])
        lead_count = ""
        lead_range = ""
        lead_step = ""
        if "step" in ds.variables:
            lead_count, lead_range, lead_step = lead_summary_hours(np.array(ds.variables["step"][:]))
        elif "lead_time" in ds.variables:
            lead_vals = np.array(ds.variables["lead_time"][:])
            vals = values_1d(lead_vals).astype(float)
            lead_count = int(vals.size)
            if vals.size:
                lead_range = f"{float(vals.min()):g}..{float(vals.max()):g} days"
                lead_step = "1 day" if vals.size > 1 and np.all(np.diff(vals) == 1) else ""
        for name, var in ds.variables.items():
            if name in coord_names:
                continue
            rows.append({
                "season": season,
                "model": model,
                "product": product,
                "variable": name,
                "format": "netcdf",
                "sample_path": str(path),
                "read_ok": True,
                "dims": ",".join(var.dimensions),
                "shape": "x".join(map(str, var.shape)),
                "units": getattr(var, "units", ""),
                "member_count": member_count,
                "lead_count": lead_count,
                "lead_range": lead_range,
                "lead_step": lead_step,
                "grid": grid,
                "notes": "NetCDF header read only; full data arrays were not loaded.",
            })
    return rows


def safe_eccodes_get(gid, key: str):
    try:
        import eccodes
        return eccodes.codes_get(gid, key)
    except Exception:
        return None


def read_grib_metadata(path: Path, model: str, season: str, product: str, max_messages: int) -> list[dict]:
    try:
        import eccodes
    except Exception as exc:
        return [{
            "season": season, "model": model, "product": product, "variable": "",
            "format": "grib", "sample_path": str(path), "read_ok": False, "dims": "",
            "shape": "", "units": "", "member_count": "", "lead_count": "",
            "lead_range": "", "lead_step": "", "grid": "", "notes": f"eccodes import failed: {exc}",
        }]

    by_var: dict[str, dict] = {}
    message_count = 0
    with path.open("rb") as f:
        while message_count < max_messages:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            message_count += 1
            try:
                short = safe_eccodes_get(gid, "shortName") or "unknown"
                item = by_var.setdefault(short, {"steps": set(), "numbers": set(), "grid": ""})
                step = safe_eccodes_get(gid, "step")
                if step is not None:
                    item["steps"].add(float(step))
                number = (
                    safe_eccodes_get(gid, "perturbationNumber")
                    or safe_eccodes_get(gid, "number")
                    or safe_eccodes_get(gid, "ensembleMemberNumber")
                )
                if number is not None:
                    item["numbers"].add(int(number))
                if not item["grid"]:
                    ni = safe_eccodes_get(gid, "Ni")
                    nj = safe_eccodes_get(gid, "Nj")
                    lat1 = safe_eccodes_get(gid, "latitudeOfFirstGridPointInDegrees")
                    lat2 = safe_eccodes_get(gid, "latitudeOfLastGridPointInDegrees")
                    lon1 = safe_eccodes_get(gid, "longitudeOfFirstGridPointInDegrees")
                    lon2 = safe_eccodes_get(gid, "longitudeOfLastGridPointInDegrees")
                    if None not in (ni, nj, lat1, lat2, lon1, lon2):
                        item["grid"] = f"{nj}x{ni}; lat {min(lat1, lat2):.2f}..{max(lat1, lat2):.2f}; lon {min(lon1, lon2):.2f}..{max(lon1, lon2):.2f}"
            finally:
                eccodes.codes_release(gid)

    unit_map = {"mx2t6": "K", "mn2t6": "K", "tp": "kg m**-2", "gh": "gpm"}
    long_map = {
        "mx2t6": "Maximum 2m temperature in last 6 hours",
        "mn2t6": "Minimum 2m temperature in last 6 hours",
        "tp": "Total precipitation",
        "gh": "Geopotential height",
    }
    rows = []
    for variable, item in sorted(by_var.items()):
        steps = sorted(item["steps"])
        numbers = sorted(item["numbers"])
        lead_count, lead_range, lead_step = lead_summary_hours(np.array(steps)) if steps else ("", "", "")
        rows.append({
            "season": season,
            "model": model,
            "product": product,
            "variable": variable,
            "format": "grib",
            "sample_path": str(path),
            "read_ok": True,
            "dims": "number(optional),step,latitude,longitude",
            "shape": "",
            "units": unit_map.get(variable, ""),
            "member_count": len(numbers) if numbers else "",
            "lead_count": lead_count,
            "lead_range": lead_range,
            "lead_step": lead_step,
            "grid": item["grid"],
            "notes": f"{long_map.get(variable, '')}; scanned {message_count} GRIB messages max.",
        })
    return rows


def scan_delysm(data_root: Path, season: str, read_each: bool, read_metadata: bool) -> tuple[list[dict], list[dict]]:
    root = data_root / "delysm" / season
    rows: list[dict] = []
    metadata: list[dict] = []
    sample_read = False
    for init_date in date_entries(root):
        store = root / init_date / "india" / "forecast.zarr"
        chunk = store / "t2m" / "c" / "0" / "0" / "8" / "0" / "0"
        store_ok = store.exists()
        chunk_ok = chunk.exists() and chunk.stat().st_size > 0
        read_ok = store_ok
        notes = []
        if read_metadata and store_ok and (read_each or not sample_read):
            try:
                metadata.extend(read_zarr_metadata(store, "delysm", season, "india/forecast.zarr"))
                sample_read = True
            except Exception as exc:
                sample_read = True
                notes.append(f"metadata_read_error={exc}")
        elif store_ok:
            read_ok = True
        is_usable = store_ok and chunk_ok and read_ok
        if not store_ok:
            status = "missing_store"
        elif not chunk_ok:
            status = "missing_data_chunks"
        else:
            status = "usable"
        rows.append({
            "season": season,
            "model": "delysm",
            "init_date": init_date,
            "is_usable": is_usable,
            "status": status,
            "products_present": "india/forecast.zarr" if store_ok else "",
            "products_missing": "" if store_ok else "india/forecast.zarr",
            "file_or_store_count": int(store_ok),
            "notes": "; ".join(notes),
        })
    return rows, metadata


def discover_nc_products(root: Path) -> list[Product]:
    products: list[Product] = []
    if not root.exists():
        return products
    for child_entry in sorted(os.scandir(root), key=lambda entry: entry.name):
        child = Path(child_entry.path)
        if not child_entry.is_dir(follow_symlinks=False) or child_entry.name == "logs":
            continue
        if has_file_with_suffix(child, ".nc"):
            products.append(Product(child.name, child, "netcdf"))
        for sub_entry in sorted(os.scandir(child), key=lambda entry: entry.name):
            if not sub_entry.is_dir(follow_symlinks=False):
                continue
            sub = Path(sub_entry.path)
            if has_file_with_suffix(sub, ".nc"):
                products.append(Product(f"{child.name}/{sub.name}", sub, "netcdf"))
    return products


def scan_netcdf_model(data_root: Path, model: str, season: str, read_each: bool, read_metadata: bool) -> tuple[list[dict], list[dict]]:
    root = data_root / model / season
    products = discover_nc_products(root)
    product_files = {product.label: list_netcdf_stream_files(product.path) for product in products}
    dates = sorted({date for files_by_date in product_files.values() for date in files_by_date})
    rows: list[dict] = []
    metadata: list[dict] = []
    metadata_read_products: set[str] = set()
    for init_date in dates:
        present: list[str] = []
        missing: list[str] = []
        count = 0
        notes: list[str] = []
        for product in products:
            streams_for_date = product_files[product.label].get(init_date, {})
            cf = streams_for_date.get("cf")
            pf = streams_for_date.get("pf")
            cf_ok = cf is not None
            pf_ok = pf is not None
            if cf_ok:
                count += 1
            if pf_ok:
                count += 1
            if cf_ok or pf_ok:
                streams = ",".join(s for s, ok in (("cf", cf_ok), ("pf", pf_ok)) if ok)
                present.append(f"{product.label}:{streams}")
            if not (cf_ok and pf_ok):
                missing.append(product.label)
            sample = pf if pf_ok else cf
            if read_metadata and sample is not None and (read_each or product.label not in metadata_read_products):
                try:
                    metadata.extend(read_netcdf_metadata(sample, model, season, product.label))
                    metadata_read_products.add(product.label)
                except Exception as exc:
                    notes.append(f"{product.label}_metadata_read_error={exc}")
        is_usable = bool(products) and not missing
        rows.append({
            "season": season,
            "model": model,
            "init_date": init_date,
            "is_usable": is_usable,
            "status": "usable" if is_usable else "partial_or_missing_products",
            "products_present": ";".join(present),
            "products_missing": ";".join(missing),
            "file_or_store_count": count,
            "notes": "; ".join(notes),
        })
    return rows, metadata


def scan_fuxi(data_root: Path, season: str, read_each: bool, read_metadata: bool) -> tuple[list[dict], list[dict]]:
    if season == "jjas2019":
        root = data_root.parent / "s2s_final_data" / "jjas" / "fuxi_combined"
    else:
        root = data_root / "fuxi" / season / "combined"
    files_by_date = list_date_files(root, ".nc")
    if season == "jjas2019":
        files_by_date = {date: path for date, path in files_by_date.items() if date.startswith("2019")}
    rows: list[dict] = []
    metadata: list[dict] = []
    sample_read = False
    for init_date, path in sorted(files_by_date.items()):
        ok = True
        read_ok = ok
        notes: list[str] = []
        if read_metadata and ok and (read_each or not sample_read):
            try:
                metadata.extend(read_netcdf_metadata(path, "fuxi", season, "combined"))
                sample_read = True
            except Exception as exc:
                sample_read = True
                notes.append(f"metadata_read_error={exc}")
        elif ok:
            read_ok = True
        rows.append({
            "season": season,
            "model": "fuxi",
            "init_date": init_date,
            "is_usable": ok and read_ok,
            "status": "usable" if ok and read_ok else "metadata_read_failed" if ok else "missing_or_empty",
            "products_present": "combined" if ok else "",
            "products_missing": "" if ok else "combined",
            "file_or_store_count": int(ok),
            "notes": "; ".join(notes),
        })
    return rows, metadata


def scan_ncep(data_root: Path, season: str, read_grib_samples: bool) -> tuple[list[dict], list[dict]]:
    root = data_root / "ncep" / season
    streams = ("surface/cf", "surface/pf", "z/500/cf", "z/500/pf")
    stream_dates: dict[str, set[str]] = {}
    stream_files: dict[str, dict[str, Path]] = {}
    for stream in streams:
        files = list_date_files(root / stream, ".grib")
        stream_files[stream] = files
        stream_dates[stream] = set(files)
    dates = sorted(set().union(*stream_dates.values())) if stream_dates else []
    rows: list[dict] = []
    metadata: list[dict] = []
    for init_date in dates:
        present = [stream for stream in streams if init_date in stream_dates[stream]]
        missing = [stream for stream in streams if init_date not in stream_dates[stream]]
        rows.append({
            "season": season,
            "model": "ncep",
            "init_date": init_date,
            "is_usable": not missing,
            "status": "usable" if not missing else "partial_or_missing_streams",
            "products_present": ";".join(present),
            "products_missing": ";".join(missing),
            "file_or_store_count": len(present),
            "notes": "",
        })
    if read_grib_samples:
        for product in ("surface/pf", "z/500/pf"):
            sample_dates = sorted(stream_dates.get(product, set()))
            if not sample_dates:
                continue
            sample = stream_files[product][sample_dates[0]]
            metadata.extend(read_grib_metadata(sample, "ncep", season, product, max_messages=2500))
    return rows, metadata


def scan_all(
    data_root: Path,
    read_each: bool,
    read_metadata: bool,
    read_grib_samples: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    availability: list[dict] = []
    metadata: list[dict] = []
    for season in SEASONS:
        for scanner in (
            lambda: scan_delysm(data_root, season, read_each, read_metadata),
            lambda: scan_netcdf_model(data_root, "ecmwf", season, read_each, read_metadata),
            lambda: scan_netcdf_model(data_root, "ukmo", season, read_each, read_metadata),
            lambda: scan_ncep(data_root, season, read_grib_samples),
            lambda: scan_fuxi(data_root, season, read_each, read_metadata),
        ):
            rows, meta = scanner()
            availability.extend(rows)
            metadata.extend(meta)
    return pd.DataFrame(availability), pd.DataFrame(metadata)


def build_model_summary(availability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, model), group in availability.groupby(["season", "model"], sort=True):
        usable_dates = sorted(group.loc[group["is_usable"].astype(bool), "init_date"].astype(str).unique())
        all_dates = sorted(group["init_date"].astype(str).unique())
        summary = summarize_dates(usable_dates)
        rows.append({
            "season": season,
            "model": model,
            "scanned_init_count": len(all_dates),
            "usable_init_count": len(usable_dates),
            "first_usable_init": summary["first"],
            "last_usable_init": summary["last"],
            "usable_frequency": summary["frequency"],
            "usable_init_dates": ";".join(usable_dates),
            "problem_init_dates": ";".join(sorted(set(all_dates) - set(usable_dates))),
        })
    for season in SEASONS:
        present = set(availability.loc[availability["season"] == season, "model"].unique())
        for model in MODELS:
            if model not in present:
                rows.append({
                    "season": season,
                    "model": model,
                    "scanned_init_count": 0,
                    "usable_init_count": 0,
                    "first_usable_init": "",
                    "last_usable_init": "",
                    "usable_frequency": "",
                    "usable_init_dates": "",
                    "problem_init_dates": "",
                })
    return pd.DataFrame(rows).sort_values(["season", "model"]).reset_index(drop=True)


def common_dates_for_models(usable_by_model: dict[str, set[str]], models: list[str]) -> list[str]:
    if not models:
        return []
    sets = [usable_by_model.get(model, set()) for model in models]
    if not sets or any(not s for s in sets):
        return []
    return sorted(set.intersection(*sets))


def build_comparable_sets(availability: pd.DataFrame, output_root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    pairwise_tables: dict[str, pd.DataFrame] = {}
    for season in SEASONS:
        season_df = availability[availability["season"] == season]
        usable_by_model = {
            model: set(group.loc[group["is_usable"].astype(bool), "init_date"].astype(str))
            for model, group in season_df.groupby("model")
        }
        nonempty = [model for model in MODELS if usable_by_model.get(model)]
        groups = {
            "all_usable_models": nonempty,
            "operational_models": [m for m in OPERATIONAL_MODELS if usable_by_model.get(m)],
            "operational_plus_fuxi": [m for m in ("ecmwf", "ukmo", "ncep", "fuxi") if usable_by_model.get(m)],
            "delysm_operational": [m for m in ("delysm", "ecmwf", "ukmo", "ncep") if usable_by_model.get(m)],
            "ai_models_present": [m for m in AI_MODELS if usable_by_model.get(m)],
        }
        for group_name, models in groups.items():
            common = common_dates_for_models(usable_by_model, models)
            summary = summarize_dates(common)
            note = "" if len(models) >= 2 else "Only one usable model in this group; not a cross-model comparison."
            rows.append({
                "season": season,
                "set_name": group_name,
                "models": ";".join(models),
                "model_count": len(models),
                "common_init_count": len(common),
                "first_common_init": summary["first"],
                "last_common_init": summary["last"],
                "frequency": summary["frequency"],
                "common_init_dates": ";".join(common),
                "notes": note,
            })
            matched_dir = output_root / season / "02_processed" / "matched_init"
            matched_dir.mkdir(parents=True, exist_ok=True)
            matched_path = matched_dir / f"{group_name}_init_dates.csv"
            pd.DataFrame({"init_date": common}).to_csv(matched_path, index=False)
        matrix = pd.DataFrame(index=MODELS, columns=MODELS, dtype=int)
        for left in MODELS:
            for right in MODELS:
                matrix.loc[left, right] = len(usable_by_model.get(left, set()) & usable_by_model.get(right, set()))
        matrix.insert(0, "model", matrix.index)
        pairwise_tables[season] = matrix.reset_index(drop=True)
    return pd.DataFrame(rows), pairwise_tables


def write_outputs(
    availability: pd.DataFrame,
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
    comparable: pd.DataFrame,
    pairwise: dict[str, pd.DataFrame],
    output_root: Path,
) -> dict[str, str]:
    common_inventory = output_root / "common" / "inventory"
    common_inventory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "availability": common_inventory / "forecast_readability_by_init.csv",
        "metadata": common_inventory / "forecast_product_metadata.csv",
        "summary": common_inventory / "forecast_model_init_summary.csv",
        "comparable": common_inventory / "forecast_comparable_init_sets.csv",
    }
    availability.to_csv(outputs["availability"], index=False)
    metadata.to_csv(outputs["metadata"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    comparable.to_csv(outputs["comparable"], index=False)

    for season in SEASONS:
        coverage_dir = output_root / season / "01_qc" / "coverage"
        metadata_dir = output_root / season / "01_qc" / "metadata"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        season_avail = availability[availability["season"] == season]
        season_meta = metadata[metadata["season"] == season] if not metadata.empty else metadata
        season_summary = summary[summary["season"] == season]
        season_comparable = comparable[comparable["season"] == season]

        season_avail.to_csv(coverage_dir / "forecast_readability_by_init.csv", index=False)
        season_summary.to_csv(coverage_dir / "forecast_model_init_summary.csv", index=False)
        season_comparable.to_csv(coverage_dir / "forecast_comparable_init_sets.csv", index=False)
        pairwise[season].to_csv(coverage_dir / "model_init_overlap_matrix.csv", index=False)
        season_meta.to_csv(metadata_dir / "forecast_product_metadata.csv", index=False)

    manifest = {
        "generated_from": str(DEFAULT_DATA_ROOT),
        "output_root": str(output_root),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "season_outputs": {
            season: {
                "coverage": str(output_root / season / "01_qc" / "coverage"),
                "metadata": str(output_root / season / "01_qc" / "metadata"),
                "matched_init": str(output_root / season / "02_processed" / "matched_init"),
            }
            for season in SEASONS
        },
    }
    manifest_path = common_inventory / "forecast_scan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    outputs["manifest"] = manifest_path
    return {key: str(path) for key, path in outputs.items()}


def print_brief(summary: pd.DataFrame, comparable: pd.DataFrame) -> None:
    print("\nUsable init counts by model")
    for season in SEASONS:
        print(f"\n{season}")
        season_summary = summary[summary["season"] == season]
        for _, row in season_summary.sort_values("model").iterrows():
            print(
                f"  {row['model']:13s} "
                f"usable={int(row['usable_init_count']):3d} "
                f"first={row['first_usable_init']} last={row['last_usable_init']} "
                f"{row['usable_frequency']}"
            )
        season_sets = comparable[comparable["season"] == season]
        print("  comparable sets:")
        for _, row in season_sets.iterrows():
            print(
                f"    {row['set_name']:20s} n={int(row['common_init_count']):3d} "
                f"models={row['models']} first={row['first_common_init']} last={row['last_common_init']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--read-each",
        action="store_true",
        help="Open metadata for every forecast file/store. Default reads one sample per product.",
    )
    parser.add_argument(
        "--no-sample-metadata",
        action="store_true",
        help="Only scan paths/init dates. Do not open sample NetCDF/Zarr metadata.",
    )
    parser.add_argument(
        "--read-grib-metadata",
        action="store_true",
        help="Also scan sample NCEP GRIB metadata with eccodes. Slower on large GRIB files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    availability, metadata = scan_all(
        data_root=args.data_root,
        read_each=args.read_each,
        read_metadata=not args.no_sample_metadata,
        read_grib_samples=args.read_grib_metadata,
    )
    if availability.empty:
        raise SystemExit(f"No forecast init rows found under {args.data_root}")
    summary = build_model_summary(availability)
    comparable, pairwise = build_comparable_sets(availability, args.output_root)
    outputs = write_outputs(availability, metadata, summary, comparable, pairwise, args.output_root)
    print_brief(summary, comparable)
    print("\nWrote:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
