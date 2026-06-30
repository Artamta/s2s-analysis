#!/usr/bin/env python3
"""Download daily ERA5 verification fields for JFM2026 lead windows.

Source is the public ARCO-ERA5 hourly zarr on Google Cloud. Output files are
monthly NetCDFs with daily fields over the India box:

- ``tp``: daily total precipitation, mm/day
- ``t2m``: 2 m temperature at 00 UTC, K
- ``z500``: 500 hPa geopotential height at 00 UTC, gpm
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ARCO_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
G = 9.80665


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.period_range(start=start, end=end, freq="M")
    return [period.to_timestamp() for period in months]


def month_end(month_start: pd.Timestamp) -> pd.Timestamp:
    ndays = calendar.monthrange(month_start.year, month_start.month)[1]
    return month_start + pd.Timedelta(days=ndays - 1)


def existing_complete(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    try:
        with xr.open_dataset(path) as ds:
            if not {"tp", "t2m", "z500"}.issubset(ds.data_vars):
                return False
            wanted = pd.date_range(start, end, freq="D")
            times = pd.to_datetime(ds["time"].values).normalize()
            if not set(wanted).issubset(set(times)):
                return False
            for var in ("tp", "t2m", "z500"):
                sample = ds[var].sel(time=wanted[: min(3, len(wanted))]).mean(skipna=True)
                if not np.isfinite(float(sample.values)):
                    return False
    except Exception:
        return False
    return True


def open_arco() -> xr.Dataset:
    return xr.open_zarr(
        ARCO_PATH,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks=None,
    )


def daily_for_month(ds: xr.Dataset, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    hourly_start = start
    hourly_end = end + pd.Timedelta(hours=23)
    spatial = {
        "time": slice(hourly_start, hourly_end),
        "latitude": slice(40.0, 0.0),
        "longitude": slice(60.0, 100.0),
    }

    wanted_days = pd.date_range(start, end, freq="D")
    wanted_midnight = wanted_days

    tp_hourly = ds["total_precipitation"].sel(**spatial)
    if tp_hourly.sizes.get("time", 0) == 0:
        raise ValueError(f"no ARCO time records for {start.date()} to {end.date()}")
    tp = (tp_hourly.resample(time="1D").sum() * 1000.0).clip(min=0.0).rename("tp")
    tp = tp.reindex(time=wanted_days)
    tp.attrs.update(
        {
            "long_name": "Daily total precipitation",
            "short_name": "tp",
            "units": "mm/day",
            "source_note": "sum of hourly ERA5 total_precipitation multiplied by 1000",
        }
    )

    t2m = ds["2m_temperature"].sel(**spatial).sel(time=wanted_midnight).rename("t2m")
    t2m = t2m.assign_coords(time=wanted_days)
    t2m.attrs.update(
        {
            "long_name": "2 metre temperature at 00 UTC",
            "short_name": "t2m",
            "units": "K",
        }
    )

    z500_src = ds["geopotential"].sel(**spatial).sel(level=500)
    z500 = (z500_src.sel(time=wanted_midnight, drop=True) / G).rename("z500")
    z500 = z500.assign_coords(time=wanted_days)
    z500.attrs.update(
        {
            "long_name": "500 hPa geopotential height at 00 UTC",
            "short_name": "z500",
            "units": "gpm",
            "source_note": "ERA5 geopotential at 500 hPa divided by 9.80665",
        }
    )

    out = xr.Dataset({"tp": tp.astype("float32"), "t2m": t2m.astype("float32"), "z500": z500.astype("float32")})
    out = out.sortby("latitude", ascending=False).sortby("longitude")
    out.attrs.update(
        {
            "source": "ARCO-ERA5 public zarr on Google Cloud",
            "source_path": ARCO_PATH,
            "domain": "0-40N, 60-100E",
            "created_by": Path(__file__).name,
            "date_start": start.strftime("%Y-%m-%d"),
            "date_end": end.strftime("%Y-%m-%d"),
        }
    )
    return out


def write_month(ds: xr.Dataset, out_path: Path) -> None:
    tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in ds.data_vars
    }
    ds.to_netcdf(tmp_path, format="NETCDF4", encoding=encoding)
    with xr.open_dataset(tmp_path) as check:
        for var in ("tp", "t2m", "z500"):
            if var not in check:
                raise ValueError(f"{tmp_path} missing variable {var}")
            if check[var].sizes.get("time", 0) != ds.sizes["time"]:
                raise ValueError(f"{tmp_path} has incomplete time dimension for {var}")
            if not np.isfinite(float(check[var].mean(skipna=True).values)):
                raise ValueError(f"{tmp_path} has non-finite mean for {var}")
    tmp_path.replace(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/storage/raj.ayush/All_Model_Data/ground_truth/era5_daily/jfm2026"),
    )
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise ValueError("--end must be on or after --start")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Opening ARCO-ERA5: {ARCO_PATH}", flush=True)
    ds = open_arco()
    manifest: list[dict[str, object]] = []
    try:
        for month_start in month_starts(start, end):
            part_start = max(start, month_start)
            part_end = min(end, month_end(month_start))
            out_path = args.output_dir / f"era5_daily_{month_start.year}{month_start.month:02d}.nc"
            if not args.overwrite and existing_complete(out_path, part_start, part_end):
                print(f"{out_path.name}: complete, skipping", flush=True)
                manifest.append(
                    {
                        "path": str(out_path),
                        "start": part_start.strftime("%Y-%m-%d"),
                        "end": part_end.strftime("%Y-%m-%d"),
                        "status": "skipped_existing",
                    }
                )
                continue
            print(f"{out_path.name}: downloading {part_start.date()} to {part_end.date()}", flush=True)
            daily = daily_for_month(ds, part_start, part_end).load()
            write_month(daily, out_path)
            manifest.append(
                {
                    "path": str(out_path),
                    "start": part_start.strftime("%Y-%m-%d"),
                    "end": part_end.strftime("%Y-%m-%d"),
                    "status": "written",
                    "n_days": int(daily.sizes["time"]),
                    "sizes": {dim: int(size) for dim, size in daily.sizes.items()},
                }
            )
            print(f"{out_path.name}: wrote {daily.sizes['time']} days", flush=True)
    finally:
        ds.close()

    manifest_path = args.output_dir / "download_manifest_jfm2026_era5_daily.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
