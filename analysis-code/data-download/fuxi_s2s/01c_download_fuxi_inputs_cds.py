#!/usr/bin/env python3
"""
01c_download_fuxi_inputs_cds.py
================================
Download ERA5 initial conditions for FuXi-S2S inference via CDS API.
Use this for dates after 2023-01-10 (when ARCO-ERA5 ends).

For each init date, downloads (prev_day + init_day) at 00 UTC:
  - 5 pressure-level vars × 13 levels: z, t, u, v, q
  - 11 surface vars: t2m, d2m, sst, ttr, 10u, 10v, 100u, 100v, msl, tcwv, tp
Then combines into input.nc using FuXi data_util.make_input().

CDS endpoint : https://cds.climate.copernicus.eu/api
Output       : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs/{YYYYMMDD}/input.nc

Usage
-----
  python 01c_download_fuxi_inputs_cds.py           # all 90 JFM 2026 dates
  python 01c_download_fuxi_inputs_cds.py --date 20260102
  sbatch slurm/fuxi_01c_inputs_cds.sbatch
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent / "FuXi-S2S"))
from data_util import make_input

# ── PATHS ─────────────────────────────────────────────────────────────────────
INPUT_DIR = Path("/storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs")
LOG_DIR   = Path(__file__).parent / "logs"

# ── CDS CONFIG ────────────────────────────────────────────────────────────────
CDS_URL = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
CDS_KEY = os.environ.get("CDSAPI_KEY")

# Global 1.5° grid — exactly what FuXi expects
AREA = [90, 0, -90, 358.5]   # global
GRID = [1.5, 1.5]

PL_VARS = [
    "geopotential", "temperature",
    "u_component_of_wind", "v_component_of_wind", "specific_humidity",
]
PL_LEVELS = ["50", "100", "150", "200", "250", "300", "400",
             "500", "600", "700", "850", "925", "1000"]

SFC_VARS = [
    "2m_temperature", "2m_dewpoint_temperature", "sea_surface_temperature",
    "top_net_thermal_radiation",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "100m_u_component_of_wind", "100m_v_component_of_wind",
    "mean_sea_level_pressure", "total_column_water_vapour",
    "total_precipitation",
]

DATE_START = "2026-01-01"
DATE_END   = "2026-03-31"

MAX_RETRIES   = 5
RETRY_BACKOFF = [60, 120, 300, 600, 900]


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def is_done(date: pd.Timestamp) -> bool:
    return (INPUT_DIR / f"{date:%Y%m%d}" / "input.nc").exists()


def retrieve_with_retry(client, dataset, req, dest, label, log):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(dataset, req, str(dest))
            return
        except Exception as e:
            log.warning(f"  {label} attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                log.info(f"  waiting {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"{label} failed after {MAX_RETRIES} attempts: {e}")


def download_one(client: cdsapi.Client, init_date: pd.Timestamp,
                 log: logging.Logger) -> bool:
    date_str = f"{init_date:%Y%m%d}"
    out_file = INPUT_DIR / date_str / "input.nc"

    if out_file.exists():
        log.info(f"SKIP  {date_str}")
        return True

    prev_day = init_date - pd.Timedelta(days=1)
    # CDS time selection: both days at 00:00
    dates_str = f"{prev_day:%Y-%m-%d}/{init_date:%Y-%m-%d}"

    log.info(f"START {date_str}  (ERA5: {prev_day:%Y-%m-%d} + {init_date:%Y-%m-%d})")
    date_dir = INPUT_DIR / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Pressure levels ───────────────────────────────────────────────
        pl_file = tmp / "pl.nc"
        pl_req = {
            "product_type": "reanalysis",
            "variable":       PL_VARS,
            "pressure_level": PL_LEVELS,
            "date":           dates_str,
            "time":           "00:00",
            "area":           AREA,
            "grid":           GRID,
            "data_format":    "netcdf",
        }
        retrieve_with_retry(client, "reanalysis-era5-pressure-levels",
                            pl_req, pl_file, f"{date_str}/pl", log)

        # ── Surface ───────────────────────────────────────────────────────
        sfc_file = tmp / "sfc.nc"
        sfc_req = {
            "product_type": "reanalysis",
            "variable":     SFC_VARS,
            "date":         dates_str,
            "time":         "00:00",
            "area":         AREA,
            "grid":         GRID,
            "data_format":  "netcdf",
        }
        retrieve_with_retry(client, "reanalysis-era5-single-levels",
                            sfc_req, sfc_file, f"{date_str}/sfc", log)

        # ── Split into per-variable files that make_input expects ─────────
        pl_ds = xr.open_dataset(str(pl_file), engine="netcdf4")

        # CDS sometimes returns a ZIP of multiple netcdf files (instant + accum vars)
        import zipfile
        if zipfile.is_zipfile(str(sfc_file)):
            sfc_parts = []
            with zipfile.ZipFile(str(sfc_file)) as z:
                for name in z.namelist():
                    part_path = tmp / name
                    z.extract(name, str(tmp))
                    sfc_parts.append(xr.open_dataset(str(part_path), engine="netcdf4"))
            sfc_ds = xr.merge(sfc_parts)
        else:
            sfc_ds = xr.open_dataset(str(sfc_file), engine="netcdf4")

        # CDS returns: valid_time, pressure_level, latitude, longitude
        # make_input expects: time, level, latitude, longitude
        pl_rename = {}
        if "valid_time" in pl_ds.dims:  pl_rename["valid_time"]     = "time"
        if "pressure_level" in pl_ds.dims: pl_rename["pressure_level"] = "level"
        if pl_rename:
            pl_ds = pl_ds.rename(pl_rename)

        sfc_rename = {}
        if "valid_time" in sfc_ds.dims: sfc_rename["valid_time"] = "time"
        if sfc_rename:
            sfc_ds = sfc_ds.rename(sfc_rename)

        # Save each variable as its own .nc in date_dir
        var_map_pl = {
            "z": "geopotential", "t": "temperature",
            "u": "u_component_of_wind", "v": "v_component_of_wind",
            "q": "specific_humidity",
        }
        long_names_pl = {v: k for k, v in var_map_pl.items()}

        for long_name in PL_VARS:
            p = date_dir / f"{long_name}.nc"
            if p.exists():
                continue
            # CDS uses short name in netcdf (z, t, u, v, q)
            short = long_names_pl.get(long_name, long_name)
            var = short if short in pl_ds else long_name
            if var not in pl_ds:
                # try first data var
                var = list(pl_ds.data_vars)[0]
            da = pl_ds[var].astype("float32")
            da.name = "data"
            da.to_netcdf(str(p))

        var_map_sfc = {
            "t2m": "2m_temperature", "d2m": "2m_dewpoint_temperature",
            "sst": "sea_surface_temperature", "ttr": "top_net_thermal_radiation",
            "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
            "u100": "100m_u_component_of_wind", "v100": "100m_v_component_of_wind",
            "msl": "mean_sea_level_pressure", "tcwv": "total_column_water_vapour",
            "tp": "total_precipitation",
        }
        long_names_sfc = {v: k for k, v in var_map_sfc.items()}

        for long_name in SFC_VARS:
            p = date_dir / f"{long_name}.nc"
            if p.exists():
                continue
            short = long_names_sfc.get(long_name, long_name)
            var = short if short in sfc_ds else long_name
            if var not in sfc_ds:
                continue
            da = sfc_ds[var].astype("float32")
            da.name = "data"
            # make_input expects a level dim for surface vars
            if "level" not in da.dims:
                da = da.assign_coords(level=1000).expand_dims("level")
            da.to_netcdf(str(p))

        pl_ds.close()
        sfc_ds.close()

    # ── Combine into input.nc ─────────────────────────────────────────────
    log.info(f"  combining → input.nc")
    inp = make_input(str(date_dir))
    if "latitude" in inp.dims:
        inp = inp.rename({"latitude": "lat", "longitude": "lon"})
    # Strip .0 suffix from channel names (e.g. "z1000.0" → "z1000")
    clean_channels = [c.replace(".0", "") if c[-2:] == ".0" else c
                      for c in inp.channel.values.tolist()]
    inp = inp.assign_coords(channel=clean_channels)
    inp.to_netcdf(str(out_file))

    # Clean raw variable files
    for f in date_dir.iterdir():
        if f.suffix == ".nc" and f.name != "input.nc":
            f.unlink()

    mb = out_file.stat().st_size / 1024**2
    log.info(f"DONE  {date_str}  {mb:.0f} MB  → {out_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download FuXi ERA5 inputs from CDS (JFM 2026)")
    parser.add_argument("--date",  type=str, default=None, help="Single date YYYYMMDD")
    parser.add_argument("--start", type=str, default=DATE_START)
    parser.add_argument("--end",   type=str, default=DATE_END)
    args = parser.parse_args()

    log_file = LOG_DIR / f"fuxi_inputs_cds_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        dates = list(pd.date_range(args.start, args.end, freq="D"))

    pending = [d for d in dates if not is_done(d)]

    log.info("=" * 65)
    log.info("FuXi-S2S ERA5 Input Download via CDS (JFM 2026)")
    log.info(f"  Output dir : {INPUT_DIR}")
    log.info(f"  Dates      : {len(dates)}  ({dates[0].date()} → {dates[-1].date()})")
    log.info(f"  Done       : {len(dates) - len(pending)}")
    log.info(f"  Pending    : {len(pending)}")
    log.info(f"  Grid       : Global 1.5°  (121×240, matches FuXi input)")
    log.info(f"  PL vars    : {PL_VARS}")
    log.info(f"  SFC vars   : {len(SFC_VARS)} variables")
    log.info("=" * 65)

    if not pending:
        log.info("All input.nc files exist — nothing to do.")
        return

    client_kwargs = {"url": CDS_URL, "quiet": True}
    if CDS_KEY:
        client_kwargs["key"] = CDS_KEY
    client = cdsapi.Client(**client_kwargs)
    failed = []
    for i, date in enumerate(pending, 1):
        try:
            download_one(client, date, log)
        except Exception as e:
            log.error(f"FAIL  {date:%Y%m%d}: {e}")
            failed.append(f"{date:%Y%m%d}")
        log.info(f"Progress {i}/{len(pending)}  failed={len(failed)}")

    log.info("=" * 65)
    log.info(f"Done. Success: {len(pending)-len(failed)}  Failed: {len(failed)}")
    for f in failed:
        log.info(f"  FAILED: {f}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
