#!/usr/bin/env python
"""Audit which model files are usable together for the benchmark cases.

This is a lightweight inventory and sanity check. It does not run full
verification; it answers the foundation questions first:

- which models/variables exist for JFM2026 and JJAS2019
- which initialization dates overlap
- which variables have full 1-42 day lead coverage
- whether truth data currently overlaps those valid dates
- which unit conversions are required before scoring
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

try:
    import zarr
except ImportError:  # pragma: no cover - reported as a row-level failure.
    zarr = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2s_benchmark.paths import get_paths


G = 9.80665
STANDARD_LEADS = set(range(1, 43))
DATE_RE = re.compile(r"(\d{8})")

FUXI_CHANNELS = {"TP": "tp", "Z500": "z500", "T2M": "t2m"}

ECMWF_UKMO_VARIABLES = {
    "TP": ("tp", "tp"),
    "Z500": ("z/500", "gh"),
    "T2M": ("2t", "t2m"),
}


@dataclass(frozen=True)
class CaseSeason:
    name: str
    start: date
    end: date


CASE_SEASONS = {
    "JFM2026": CaseSeason("JFM2026", date(2026, 1, 1), date(2026, 3, 31)),
    "JJAS2019": CaseSeason("JJAS2019", date(2019, 6, 1), date(2019, 9, 30)),
}


def parse_date(text: str) -> date | None:
    match = DATE_RE.search(text)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def fmt_date(value: date | None) -> str:
    return value.isoformat() if value else ""


def fmt_dates(dates: Iterable[date], limit: int = 12) -> str:
    values = sorted(dates)
    if not values:
        return ""
    shown = [d.isoformat() for d in values[:limit]]
    if len(values) > limit:
        shown.append(f"...(+{len(values) - limit})")
    return ";".join(shown)


def date_range(start: date, end: date) -> set[date]:
    days = (end - start).days + 1
    return {start + timedelta(days=i) for i in range(days)}


def init_dates_from_files(files: Iterable[Path]) -> set[date]:
    return {d for d in (parse_date(p.name) for p in files) if d is not None}


def date_files(path: Path, suffix: str) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.glob(f"*_{suffix}.nc"))


def all_nc_dates(path: Path) -> set[date]:
    if not path.exists():
        return set()
    return init_dates_from_files(path.glob("*.nc"))


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def lead_days_from_values(values: np.ndarray) -> set[float]:
    arr = np.asarray(values)
    if arr.size == 0:
        return set()
    if np.issubdtype(arr.dtype, np.timedelta64):
        days = arr.astype("timedelta64[s]").astype(float) / 86400.0
    else:
        days = arr.astype(float)
    return {float(x) for x in np.ravel(days) if np.isfinite(x)}


def lead_summary(leads: set[float]) -> dict[str, str]:
    if not leads:
        return {
            "lead_count": "0",
            "lead_min": "",
            "lead_max": "",
            "standard_leads_1_42": "NO",
        }
    rounded = {int(round(x)) for x in leads if abs(x - round(x)) < 1e-6}
    return {
        "lead_count": str(len(leads)),
        "lead_min": scalar(min(leads)),
        "lead_max": scalar(max(leads)),
        "standard_leads_1_42": "YES" if STANDARD_LEADS.issubset(rounded) else "NO",
    }


def sample_indexers(da: xr.DataArray) -> dict[str, object]:
    indexers: dict[str, object] = {}
    for dim in da.dims:
        size = da.sizes[dim]
        if dim in {"member", "number", "ensemble"}:
            indexers[dim] = slice(0, min(size, 3))
        elif dim in {"lead_time", "step", "time"}:
            indexers[dim] = slice(0, min(size, 3))
        elif dim in {"lat", "latitude", "lon", "longitude"}:
            stride = max(1, size // 16)
            indexers[dim] = slice(None, None, stride)
        elif dim == "channel":
            indexers[dim] = 0
        else:
            indexers[dim] = 0
    return indexers


def value_stats(da: xr.DataArray) -> dict[str, str]:
    try:
        arr = np.asarray(da.isel(sample_indexers(da)).load().values, dtype=float)
        finite = np.isfinite(arr)
        if not finite.any():
            return {
                "sample_finite_fraction": "0",
                "sample_min": "",
                "sample_max": "",
                "sample_mean": "",
            }
        return {
            "sample_finite_fraction": f"{finite.mean():.3f}",
            "sample_min": f"{np.nanmin(arr):.6g}",
            "sample_max": f"{np.nanmax(arr):.6g}",
            "sample_mean": f"{np.nanmean(arr):.6g}",
        }
    except Exception as exc:
        return {
            "sample_finite_fraction": "",
            "sample_min": "",
            "sample_max": "",
            "sample_mean": "",
            "sample_error": f"{type(exc).__name__}: {exc}",
        }


def empty_stats() -> dict[str, str]:
    return {
        "sample_finite_fraction": "",
        "sample_min": "",
        "sample_max": "",
        "sample_mean": "",
    }


def nc_metadata(path: Path, data_var: str) -> dict[str, object]:
    with xr.open_dataset(path) as ds:
        if data_var not in ds:
            raise KeyError(f"{data_var!r} not in {list(ds.data_vars)}")
        da = ds[data_var]
        leads: set[float] = set()
        if "step" in ds.coords:
            leads = lead_days_from_values(np.atleast_1d(ds["step"].values))
        elif "lead_time" in ds.coords:
            leads = lead_days_from_values(np.atleast_1d(ds["lead_time"].values))
        members = ""
        for dim in ("number", "member", "ensemble"):
            if dim in da.dims:
                members = str(da.sizes[dim])
                break
        lat_dim = "lat" if "lat" in da.dims else "latitude" if "latitude" in da.dims else ""
        lon_dim = "lon" if "lon" in da.dims else "longitude" if "longitude" in da.dims else ""
        grid = ""
        if lat_dim and lon_dim:
            grid = f"{da.sizes[lat_dim]}x{da.sizes[lon_dim]}"
        out: dict[str, object] = {
            "members": members,
            "native_dims": "x".join(str(da.sizes[d]) for d in da.dims),
            "native_grid": grid,
            "native_units": da.attrs.get("units", ""),
            "lead_days": leads,
        }
        out.update(value_stats(da))
        return out


def fuxi_metadata(path: Path, variable: str) -> dict[str, object]:
    channel = FUXI_CHANNELS[variable]
    with xr.open_dataset(path) as ds:
        labels = [str(v) for v in ds["channel"].values]
        if channel not in labels:
            raise KeyError(f"channel {channel!r} not in {labels}")
        da = ds["forecast"].sel(channel=channel)
        leads = lead_days_from_values(ds["lead_time"].values)
        out: dict[str, object] = {
            "members": str(da.sizes.get("member", "")),
            "native_dims": "x".join(str(da.sizes[d]) for d in da.dims),
            "native_grid": f"{da.sizes.get('lat', '')}x{da.sizes.get('lon', '')}",
            "native_units": "native",
            "lead_days": leads,
        }
        out.update(value_stats(da))
        return out


def delysm_metadata(path: Path, variable: str) -> dict[str, object]:
    if zarr is None:
        raise RuntimeError("zarr is not installed")
    group = zarr.open_group(str(path), mode="r")
    zvar = {"Z500": "z500", "T2M": "t2m"}[variable]
    if zvar not in group:
        raise KeyError(f"{zvar!r} not in DELYSM store")
    leads = lead_days_from_values(np.asarray(group["lead_time"][:]))
    arr = group[zvar]
    lead_values = np.asarray(group["lead_time"][:])
    positive = np.where(lead_values >= np.timedelta64(0, "ns"))[0]
    lead_index = int(positive[0]) if len(positive) else 0
    sample = np.asarray(arr[0, 0, lead_index, :: max(1, arr.shape[-2] // 16), :: max(1, arr.shape[-1] // 16)])
    finite = np.isfinite(sample)
    stats = empty_stats()
    if finite.any():
        stats = {
            "sample_finite_fraction": f"{finite.mean():.3f}",
            "sample_min": f"{np.nanmin(sample):.6g}",
            "sample_max": f"{np.nanmax(sample):.6g}",
            "sample_mean": f"{np.nanmean(sample):.6g}",
        }
    return {
        "members": str(arr.shape[0]),
        "native_dims": "x".join(str(x) for x in arr.shape),
        "native_grid": f"{group['lat'].shape[0]}x{group['lon'].shape[0]}",
        "native_units": "native",
        "lead_days": leads,
        **stats,
    }


def conversion_note(model: str, variable: str) -> str:
    if variable == "Z500" and model.startswith(("FuXi", "DELYSM")):
        return "divide by 9.80665: geopotential -> gpm"
    if variable == "TP" and model.startswith("FuXi"):
        return "multiply by 24: mm/hour rate -> mm/day"
    if variable == "TP" and model in {"ECMWF", "UKMO"}:
        return "cumulative kg m-2 (=mm); difference to daily before scoring"
    if variable == "Z500" and model in {"ECMWF", "UKMO"}:
        return "already geopotential height, gpm"
    if variable == "T2M":
        return "already K"
    return ""


def model_row(
    *,
    model: str,
    season: str,
    variable: str,
    root: Path,
    init_dates: set[date],
    pf_count: int = 0,
    cf_count: int = 0,
    metadata: dict[str, object] | None = None,
    status: str = "OK",
    note: str = "",
) -> dict[str, object]:
    metadata = metadata or {}
    leads = metadata.pop("lead_days", set())
    lead_info = lead_summary(leads)
    return {
        "model": model,
        "season": season,
        "variable": variable,
        "status": status,
        "init_count": len(init_dates),
        "first_init": fmt_date(min(init_dates) if init_dates else None),
        "last_init": fmt_date(max(init_dates) if init_dates else None),
        "init_dates_sample": fmt_dates(init_dates),
        "pf_file_count": pf_count,
        "cf_file_count": cf_count,
        "members": metadata.get("members", ""),
        "native_dims": metadata.get("native_dims", ""),
        "native_grid": metadata.get("native_grid", ""),
        "native_units": metadata.get("native_units", ""),
        **lead_info,
        **empty_stats(),
        **{k: v for k, v in metadata.items() if k.startswith("sample_")},
        "conversion": conversion_note(model, variable),
        "note": note,
        "root": str(root),
    }


def audit_ecmwf_ukmo(model: str, root: Path, season: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variable, (subdir, data_var) in ECMWF_UKMO_VARIABLES.items():
        var_root = root / season.lower() / subdir
        pf_files = date_files(var_root, "pf")
        cf_files = date_files(var_root, "cf")
        init_dates = init_dates_from_files(pf_files)
        metadata: dict[str, object] = {}
        status = "OK" if pf_files else "MISSING"
        note = ""
        if pf_files:
            try:
                metadata = nc_metadata(pf_files[0], data_var)
            except Exception as exc:
                status = "OPEN_FAIL"
                note = f"{type(exc).__name__}: {exc}"
        rows.append(
            model_row(
                model=model,
                season=season,
                variable=variable,
                root=var_root,
                init_dates=init_dates,
                pf_count=len(pf_files),
                cf_count=len(cf_files),
                metadata=metadata,
                status=status,
                note=note,
            )
        )
    return rows


def audit_fuxi(paths, season: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs: list[tuple[str, Path, tuple[str, ...]]] = []
    if season == "JFM2026":
        specs = [
            ("FuXi-11", paths.all_model_data / "fuxi" / "jfm2026" / "combined", ("TP", "Z500", "T2M")),
            ("FuXi-50", paths.all_model_data / "fuxi" / "jfm2026_ens50" / "combined", ("TP", "Z500", "T2M")),
        ]
    elif season == "JJAS2019":
        specs = [("FuXi-50", paths.jjas_fuxi_compact, ("TP", "Z500", "T2M"))]

    for model, root, variables in specs:
        files = sorted(root.glob("*.nc")) if root.exists() else []
        if season == "JJAS2019":
            files = [p for p in files if p.name.startswith("2019")]
        init_dates = all_nc_dates(root)
        if season == "JJAS2019":
            init_dates = {d for d in init_dates if d.year == 2019}
        for variable in variables:
            metadata: dict[str, object] = {}
            status = "OK" if files else "MISSING"
            note = ""
            if files:
                try:
                    metadata = fuxi_metadata(files[0], variable)
                except Exception as exc:
                    status = "MISSING_VARIABLE"
                    note = f"{type(exc).__name__}: {exc}"
            rows.append(
                model_row(
                    model=model,
                    season=season,
                    variable=variable,
                    root=root,
                    init_dates=init_dates if status == "OK" else set(),
                    pf_count=len(files) if status == "OK" else 0,
                    metadata=metadata,
                    status=status,
                    note=note,
                )
            )
    return rows


def audit_delysm(paths, season: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = paths.all_model_data / "delysm" / season.lower()
    date_dirs = sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []
    stores = [p / "india" / "forecast.zarr" for p in date_dirs]
    existing = [p for p in stores if p.exists()]
    missing_count = len(stores) - len(existing)
    for variable in ("TP", "Z500", "T2M"):
        metadata: dict[str, object] = {}
        status = "OK" if existing else "MISSING"
        note = ""
        usable_dates: set[date] = set()
        usable_stores: list[Path] = []
        if variable == "TP":
            status = "MISSING_VARIABLE"
            note = "DELYSM store has no precipitation variable; use Z500/T2M only."
        elif existing:
            zvar = {"Z500": "z500", "T2M": "t2m"}[variable]
            bad_count = 0
            for store in existing:
                try:
                    if zarr is None:
                        raise RuntimeError("zarr is not installed")
                    group = zarr.open_group(str(store), mode="r")
                    if zvar not in group:
                        raise KeyError(zvar)
                    init = parse_date(store.parent.parent.name)
                    if init is not None:
                        usable_dates.add(init)
                        usable_stores.append(store)
                except Exception:
                    bad_count += 1
            if usable_stores:
                try:
                    metadata = delysm_metadata(usable_stores[0], variable)
                    notes = []
                    if missing_count:
                        notes.append(f"{missing_count} date directories lack forecast.zarr")
                    if bad_count:
                        notes.append(f"{bad_count} forecast.zarr stores are unreadable or lack {zvar}")
                    note = "; ".join(notes)
                except Exception as exc:
                    status = "OPEN_FAIL"
                    note = f"{type(exc).__name__}: {exc}"
                    usable_dates = set()
            else:
                status = "MISSING_VARIABLE"
                note = f"No readable DELYSM stores contain {zvar}"
        rows.append(
            model_row(
                model="DELYSM",
                season=season,
                variable=variable,
                root=root,
                init_dates=usable_dates,
                pf_count=len(existing) if status == "OK" else 0,
                metadata=metadata,
                status=status,
                note=note,
            )
        )
    return rows


def audit_ncep(paths, season: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = paths.all_model_data / "ncep" / season.lower()
    specs = {
        "TP": root / "surface",
        "Z500": root / "z" / "500",
    }
    for variable, var_root in specs.items():
        pf_files = sorted((var_root / "pf").glob("*.grib")) if (var_root / "pf").exists() else []
        cf_files = sorted((var_root / "cf").glob("*.grib")) if (var_root / "cf").exists() else []
        dates = init_dates_from_files(pf_files)
        metadata = {"lead_days": set(range(1, 45))} if pf_files and cf_files else None
        rows.append(
            model_row(
                model="NCEP",
                season=season,
                variable=variable,
                root=var_root,
                init_dates=dates,
                pf_count=len(pf_files),
                cf_count=len(cf_files),
                metadata=metadata,
                status="OK" if pf_files and cf_files else "MISSING",
                note="NCEP GRIB files are supported by scripts/06_open_comparable_forecasts.py.",
            )
        )
    return rows


def truth_inventory(paths) -> dict[tuple[str, str], dict[str, object]]:
    out: dict[tuple[str, str], dict[str, object]] = {}

    def dates_from_time_nc(path: Path) -> set[date]:
        if not path.exists():
            return set()
        with xr.open_dataset(path) as ds:
            values = {
                np.datetime_as_string(v, unit="D")
                for v in np.asarray(ds["time"].values)
            }
        return {datetime.strptime(v, "%Y-%m-%d").date() for v in values}

    jfm_files = sorted(paths.jfm_ground_truth_root.glob("*.nc"))
    jfm_dates = init_dates_from_files(jfm_files)
    jfm_tp_dates = dates_from_time_nc(paths.era5_prev_daily_tp_nc)
    jfm_t2m_dates = dates_from_time_nc(paths.era5_prev_daily_t2m_nc)
    out[("JFM2026", "TP")] = {
        "source": "ERA5 daily TP under s2s-forecast-data-prev/era5/daily",
        "dates": jfm_tp_dates,
        "note": "IMD 2026 rainfall NetCDF is not present; ERA5 TP covers through 2026-05-10.",
    }
    out[("JFM2026", "Z500")] = {
        "source": "ERA5 daily files under fuxi/jfm2026/ground_truth",
        "dates": jfm_dates,
        "note": "Only Jan-Mar Z500 truth is present; full 42-day verification needs Apr-May truth for later inits.",
    }
    out[("JFM2026", "T2M")] = {
        "source": "ERA5 daily T2M under s2s-forecast-data-prev/era5/daily",
        "dates": jfm_t2m_dates,
        "note": "ERA5 T2M covers through 2026-05-10.",
    }

    imd_2019_dates: set[date] = set()
    if paths.jjas2019_imd_rainfall_nc.exists():
        with xr.open_dataset(paths.jjas2019_imd_rainfall_nc) as ds:
            imd_2019_dates = {
                np.datetime_as_string(v, unit="D")
                for v in np.asarray(ds["time"].values)
            }
        imd_2019_dates = {datetime.strptime(v, "%Y-%m-%d").date() for v in imd_2019_dates}
    out[("JJAS2019", "TP")] = {
        "source": "IMD daily rainfall NetCDF",
        "dates": imd_2019_dates,
        "note": "Default TP truth is IMD with IMD 1991-2020 climatology; pipeline can also run ERA5 TP with --truth-source era5.",
    }
    for variable in ("Z500", "T2M"):
        dates = date_range(date(2019, 1, 1), date(2019, 12, 31)) if paths.weatherbench2_era5_zarr.exists() else set()
        out[("JJAS2019", variable)] = {
            "source": "WeatherBench2 ERA5 zarr",
            "dates": dates,
            "note": "Use WeatherBench2 ERA5 for JJAS2019 Z500/T2M truth.",
        }
    return out


def add_truth_columns(rows: list[dict[str, object]], truth: dict[tuple[str, str], dict[str, object]]) -> None:
    for row in rows:
        variable = str(row["variable"])
        info = truth.get((str(row["season"]), variable))
        if info is None:
            row.update(
                {
                    "truth_source": "",
                    "truth_date_count": "",
                    "truth_first": "",
                    "truth_last": "",
                    "truth_full_42_init_count": "",
                    "truth_partial_init_count": "",
                    "truth_note": "",
                    "score_readiness": row["status"],
                }
            )
            continue
        truth_dates = set(info["dates"])
        init_dates = {
            datetime.strptime(v, "%Y-%m-%d").date()
            for v in str(row["init_dates_sample"]).split(";")
            if re.match(r"^\d{4}-\d{2}-\d{2}$", v)
        }
        # init_dates_sample is truncated, so reconstructing from it is only a
        # fallback. The caller adds exact init dates through this private key.
        init_dates = set(row.get("_init_dates_exact", init_dates))
        full_42 = 0
        partial = 0
        full_42_dates: set[date] = set()
        partial_dates: set[date] = set()
        for init in init_dates:
            targets = {init + timedelta(days=lead) for lead in STANDARD_LEADS}
            present = len(targets & truth_dates)
            if present == len(STANDARD_LEADS):
                full_42 += 1
                full_42_dates.add(init)
            elif present:
                partial += 1
                partial_dates.add(init)
        full_leads = row.get("standard_leads_1_42") == "YES"
        if row["status"] != "OK":
            readiness = str(row["status"])
        elif not full_leads:
            readiness = "NO_FULL_1_42_LEADS"
        elif full_42:
            readiness = "SCORE_READY" if partial == 0 else "PARTIAL_TRUTH"
        else:
            readiness = "MISSING_TRUTH"
        row.update(
            {
                "truth_source": info["source"],
                "truth_date_count": len(truth_dates),
                "truth_first": fmt_date(min(truth_dates) if truth_dates else None),
                "truth_last": fmt_date(max(truth_dates) if truth_dates else None),
                "truth_full_42_init_count": full_42,
                "truth_partial_init_count": partial,
                "truth_note": info["note"],
                "score_readiness": readiness,
            }
        )
        row["_full_42_init_dates"] = full_42_dates
        row["_partial_truth_init_dates"] = partial_dates


def with_exact_dates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach exact date sets privately for overlap math and truth accounting."""

    for row in rows:
        dates = set()
        sample = str(row.get("init_dates_sample", ""))
        for token in sample.split(";"):
            if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
                dates.add(datetime.strptime(token, "%Y-%m-%d").date())
        row["_init_dates_exact"] = dates
    return rows


def attach_exact_dates_from_sources(rows: list[dict[str, object]], paths) -> None:
    """Rebuild exact date sets after rows have been serialized-friendly."""

    for row in rows:
        model = row["model"]
        season = row["season"]
        variable = row["variable"]
        exact: set[date] = set()
        if model in {"ECMWF", "UKMO"} and variable in ECMWF_UKMO_VARIABLES:
            subdir, _ = ECMWF_UKMO_VARIABLES[str(variable)]
            root = paths.all_model_data / str(model).lower() / str(season).lower() / subdir
            exact = init_dates_from_files(date_files(root, "pf"))
        elif str(model).startswith("FuXi"):
            if season == "JFM2026":
                root = (
                    paths.all_model_data
                    / "fuxi"
                    / ("jfm2026_ens50" if model == "FuXi-50" else "jfm2026")
                    / "combined"
                )
                exact = all_nc_dates(root)
            elif season == "JJAS2019":
                exact = {d for d in all_nc_dates(paths.jjas_fuxi_compact) if d.year == 2019}
        elif model == "DELYSM" and variable in {"Z500", "T2M"}:
            root = paths.all_model_data / "delysm" / str(season).lower()
            exact = {
                d
                for d in (parse_date(p.name) for p in root.glob("*") if (p / "india" / "forecast.zarr").exists())
                if d is not None
            }
        elif model == "NCEP":
            root = paths.all_model_data / "ncep" / str(season).lower()
            if variable == "TP":
                exact = init_dates_from_files((root / "surface" / "pf").glob("*.grib"))
            elif variable == "Z500":
                exact = init_dates_from_files((root / "z" / "500" / "pf").glob("*.grib"))
        row["_init_dates_exact"] = exact if row["status"] == "OK" else set()


def overlap_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seasons = sorted({str(row["season"]) for row in rows})
    variables = ["TP", "Z500", "T2M"]
    for season in seasons:
        for variable in variables:
            candidates = [
                row
                for row in rows
                if row["season"] == season
                and row["variable"] == variable
                and row["status"] == "OK"
                and row["standard_leads_1_42"] == "YES"
            ]
            if not candidates:
                continue
            forecast_sets = [set(row["_init_dates_exact"]) for row in candidates]
            forecast_common = set.intersection(*forecast_sets) if forecast_sets else set()
            score_ready = [
                row
                for row in candidates
                if row["score_readiness"] in {"SCORE_READY", "PARTIAL_TRUTH"}
                and row.get("_full_42_init_dates")
            ]
            score_sets = [set(row["_full_42_init_dates"]) for row in score_ready]
            score_common = set.intersection(*score_sets) if score_sets else set()
            out.append(
                {
                    "season": season,
                    "variable": variable,
                    "forecast_models": ",".join(str(row["model"]) for row in candidates),
                    "forecast_common_init_count": len(forecast_common),
                    "forecast_common_first": fmt_date(min(forecast_common) if forecast_common else None),
                    "forecast_common_last": fmt_date(max(forecast_common) if forecast_common else None),
                    "forecast_common_sample": fmt_dates(forecast_common),
                    "score_ready_models": ",".join(str(row["model"]) for row in score_ready),
                    "score_ready_common_init_count": len(score_common),
                    "score_ready_common_first": fmt_date(min(score_common) if score_common else None),
                    "score_ready_common_last": fmt_date(max(score_common) if score_common else None),
                    "score_ready_common_sample": fmt_dates(score_common),
                }
            )
    return out


def truth_rows(truth: dict[tuple[str, str], dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for (season, variable), info in sorted(truth.items()):
        dates = set(info["dates"])
        rows.append(
            {
                "season": season,
                "variable": variable,
                "source": info["source"],
                "date_count": len(dates),
                "first": fmt_date(min(dates) if dates else None),
                "last": fmt_date(max(dates) if dates else None),
                "note": info["note"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [key for key in rows[0] if not key.startswith("_")]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows: list[dict[str, object]], overlaps: list[dict[str, object]], truth: list[dict[str, object]]) -> str:
    lines = [
        "# Model Usability Audit",
        "",
        "## Bottom Line",
        "",
    ]
    ready = [row for row in rows if row.get("score_readiness") in {"SCORE_READY", "PARTIAL_TRUTH"}]
    lines.append(f"- Score-ready or partial-truth model-variable rows: {len(ready)}")
    lines.append("- Use rows marked `SCORE_READY` directly; rows marked `PARTIAL_TRUTH` need init-date restriction or more truth dates.")
    lines.append("- Rows marked `NO_FULL_1_42_LEADS`, `MISSING_TRUTH`, `MISSING_VARIABLE`, or `GRIB_UNSUPPORTED` should not enter the main metrics yet.")
    lines.append("")
    lines.append("## Main Warnings")
    lines.append("")
    warnings = []
    for row in rows:
        if row["status"] != "OK" or row["score_readiness"] not in {"SCORE_READY", "PARTIAL_TRUTH"}:
            warnings.append(
                f"- {row['season']} {row['model']} {row['variable']}: "
                f"{row['score_readiness']} ({row.get('note') or row.get('truth_note') or 'check row'})"
            )
    for warning in warnings[:40]:
        lines.append(warning)
    if len(warnings) > 40:
        lines.append(f"- ... {len(warnings) - 40} more warnings in model_usability_inventory.csv")
    lines.append("")
    lines.append("## Forecast Overlap")
    lines.append("")
    for row in overlaps:
        lines.append(
            f"- {row['season']} {row['variable']}: "
            f"{row['forecast_common_init_count']} common full-lead forecast inits across "
            f"{row['forecast_models'] or 'none'}; "
            f"{row['score_ready_common_init_count']} common score-ready inits across "
            f"{row['score_ready_models'] or 'none'}."
        )
    lines.append("")
    lines.append("## Truth Inventory")
    lines.append("")
    for row in truth:
        lines.append(
            f"- {row['season']} {row['variable']}: {row['date_count']} dates "
            f"({row['first']} to {row['last']}) from {row['source']}. {row['note']}"
        )
    lines.append("")
    lines.append("## Unit Rules Before Scoring")
    lines.append("")
    lines.append("- FuXi/DELYSM Z500: divide by 9.80665 before comparing with ERA5/ECMWF/UKMO gpm.")
    lines.append("- FuXi TP: multiply by 24 to convert mm/hour rate to mm/day.")
    lines.append("- ECMWF/UKMO TP: cumulative kg m-2 equals mm; difference along lead to daily increments before rainfall metrics.")
    lines.append("- T2M is K for the checked sources.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    paths = get_paths()
    rows: list[dict[str, object]] = []
    for season in CASE_SEASONS:
        rows.extend(audit_ecmwf_ukmo("ECMWF", paths.ecmwf_root, season))
        rows.extend(audit_ecmwf_ukmo("UKMO", paths.ukmo_root, season))
        rows.extend(audit_fuxi(paths, season))
        rows.extend(audit_delysm(paths, season))
        rows.extend(audit_ncep(paths, season))

    attach_exact_dates_from_sources(rows, paths)
    truth = truth_inventory(paths)
    add_truth_columns(rows, truth)
    overlaps = overlap_rows(rows)
    truth_table = truth_rows(truth)

    inventory_csv = args.outputs / "model_usability_inventory.csv"
    overlap_csv = args.outputs / "model_usability_overlap.csv"
    truth_csv = args.outputs / "truth_inventory.csv"
    summary_md = args.outputs / "model_usability_summary.md"
    write_csv(inventory_csv, rows)
    write_csv(overlap_csv, overlaps)
    write_csv(truth_csv, truth_table)
    summary_md.write_text(make_summary(rows, overlaps, truth_table), encoding="utf-8")

    print(f"wrote: {inventory_csv}")
    print(f"wrote: {overlap_csv}")
    print(f"wrote: {truth_csv}")
    print(f"wrote: {summary_md}")
    print("")
    for row in overlaps:
        print(
            f"{row['season']} {row['variable']:4s}: "
            f"forecast common={row['forecast_common_init_count']:>3} "
            f"score-ready common={row['score_ready_common_init_count']:>3} "
            f"models={row['forecast_models']}"
        )
    print("")
    bad = [row for row in rows if row["score_readiness"] not in {"SCORE_READY", "PARTIAL_TRUTH"}]
    print(f"rows needing action before scoring: {len(bad)}")
    for row in bad[:18]:
        print(f"- {row['season']} {row['model']} {row['variable']}: {row['score_readiness']}")
    if len(bad) > 18:
        print(f"- ... {len(bad) - 18} more in {inventory_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
