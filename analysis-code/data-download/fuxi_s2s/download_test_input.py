#!/usr/bin/env python3
"""
download_test_input.py
======================
Download ERA5 initial conditions for a single test forecast date via CDS.
Saves to /storage/raj.ayush/All_Model_Data/fuxi/test/inputs/{YYYYMMDD}/input.nc

This is a thin wrapper around the logic in 01c_download_fuxi_inputs_cds.py,
but pointing at the test output directory and accepting any date.

Usage
-----
  python download_test_input.py --date 20260620
"""

import argparse
import logging
import os
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent / "FuXi-S2S"))
from data_util import make_input

INPUT_DIR = Path("/storage/raj.ayush/All_Model_Data/fuxi/test/inputs")
LOG_DIR   = Path(__file__).parent / "logs"

CDS_URL = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
CDS_KEY = os.environ.get("CDSAPI_KEY")

AREA = [90, 0, -90, 358.5]
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


def retrieve_with_retry(client, dataset, req, dest, label, log):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(dataset, req, str(dest))
            return
        except Exception as e:
            log.warning(f"  {label} attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                log.info(f"  waiting {wait}s ...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"{label} failed after {MAX_RETRIES} attempts: {e}")


def download_one(client: cdsapi.Client, init_date: pd.Timestamp, log: logging.Logger):
    date_str = f"{init_date:%Y%m%d}"
    out_file = INPUT_DIR / date_str / "input.nc"

    if out_file.exists():
        log.info(f"SKIP  {date_str} — input.nc already exists")
        return

    prev_day   = init_date - pd.Timedelta(days=1)
    dates_str  = f"{prev_day:%Y-%m-%d}/{init_date:%Y-%m-%d}"
    date_dir   = INPUT_DIR / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading ERA5 for {prev_day:%Y-%m-%d} + {init_date:%Y-%m-%d} → {out_file}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Pressure levels ───────────────────────────────────────────────────
        pl_file = tmp / "pl.nc"
        retrieve_with_retry(client, "reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis",
            "variable":       PL_VARS,
            "pressure_level": PL_LEVELS,
            "date":           dates_str,
            "time":           "00:00",
            "area":           AREA,
            "grid":           GRID,
            "data_format":    "netcdf",
        }, pl_file, f"{date_str}/pl", log)

        # ── Surface ───────────────────────────────────────────────────────────
        sfc_file = tmp / "sfc.nc"
        retrieve_with_retry(client, "reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            "variable":     SFC_VARS,
            "date":         dates_str,
            "time":         "00:00",
            "area":         AREA,
            "grid":         GRID,
            "data_format":  "netcdf",
        }, sfc_file, f"{date_str}/sfc", log)

        # ── Open datasets ─────────────────────────────────────────────────────
        pl_ds = xr.open_dataset(str(pl_file), engine="netcdf4")

        if zipfile.is_zipfile(str(sfc_file)):
            parts = []
            with zipfile.ZipFile(str(sfc_file)) as z:
                for name in z.namelist():
                    part_path = tmp / name
                    z.extract(name, str(tmp))
                    parts.append(xr.open_dataset(str(part_path), engine="netcdf4"))
            sfc_ds = xr.merge(parts)
        else:
            sfc_ds = xr.open_dataset(str(sfc_file), engine="netcdf4")

        # Rename CDS dims → what make_input expects
        pl_rename = {}
        if "valid_time"     in pl_ds.dims: pl_rename["valid_time"]     = "time"
        if "pressure_level" in pl_ds.dims: pl_rename["pressure_level"] = "level"
        if pl_rename:
            pl_ds = pl_ds.rename(pl_rename)

        sfc_rename = {}
        if "valid_time" in sfc_ds.dims: sfc_rename["valid_time"] = "time"
        if sfc_rename:
            sfc_ds = sfc_ds.rename(sfc_rename)

        # ── Write per-variable files into date_dir ─────────────────────────
        var_map_pl  = {"z": "geopotential", "t": "temperature",
                       "u": "u_component_of_wind", "v": "v_component_of_wind",
                       "q": "specific_humidity"}
        long2short_pl = {v: k for k, v in var_map_pl.items()}

        for long_name in PL_VARS:
            p = date_dir / f"{long_name}.nc"
            if p.exists():
                continue
            short = long2short_pl.get(long_name, long_name)
            var   = short if short in pl_ds else long_name
            if var not in pl_ds:
                var = list(pl_ds.data_vars)[0]
            da      = pl_ds[var].astype("float32")
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
        long2short_sfc = {v: k for k, v in var_map_sfc.items()}

        for long_name in SFC_VARS:
            p = date_dir / f"{long_name}.nc"
            if p.exists():
                continue
            short = long2short_sfc.get(long_name, long_name)
            var   = short if short in sfc_ds else long_name
            if var not in sfc_ds:
                continue
            da = sfc_ds[var].astype("float32")
            da.name = "data"
            if "level" not in da.dims:
                da = da.assign_coords(level=1000).expand_dims("level")
            da.to_netcdf(str(p))

        pl_ds.close()
        sfc_ds.close()

    # ── Combine per-variable files into input.nc ──────────────────────────────
    log.info(f"  Combining into input.nc ...")
    inp = make_input(str(date_dir))
    if "latitude" in inp.dims:
        inp = inp.rename({"latitude": "lat", "longitude": "lon"})
    clean_channels = [c.replace(".0", "") if c.endswith(".0") else c
                      for c in inp.channel.values.tolist()]
    inp = inp.assign_coords(channel=clean_channels)
    inp.to_netcdf(str(out_file))

    # Remove raw per-variable files
    for f in date_dir.iterdir():
        if f.suffix == ".nc" and f.name != "input.nc":
            f.unlink()

    mb = out_file.stat().st_size / 1024**2
    log.info(f"DONE  {date_str}  {mb:.1f} MB  →  {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True,
                        help="Init date YYYYMMDD (ERA5 must be available on CDS, ~5 day lag)")
    args = parser.parse_args()

    init_date = pd.Timestamp(args.date)
    log_file  = LOG_DIR / f"test_dl_{datetime.now():%Y%m%d_%H%M%S}.log"
    log       = setup_logging(log_file)

    log.info("=" * 60)
    log.info("FuXi-S2S  Test Download")
    log.info(f"  Init date  : {init_date.date()}")
    log.info(f"  Output dir : {INPUT_DIR}")
    log.info("=" * 60)

    client_kwargs = {"url": CDS_URL, "quiet": True}
    if CDS_KEY:
        client_kwargs["key"] = CDS_KEY
    client = cdsapi.Client(**client_kwargs)
    download_one(client, init_date, log)


if __name__ == "__main__":
    main()
