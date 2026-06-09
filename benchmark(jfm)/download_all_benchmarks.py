#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_all_benchmarks.py
==========================
Unified download tool for S2S benchmarking in JFM 2026.
Downloads:
1. FuXi-S2S Initial Conditions (ERA5 from GCP ARCO-ERA5 public Zarr store)
   - Resolution: 1.5° (global)
   - Output path: /storage/raj.ayush/benchmark(jfm)/fuxi_s2sm/{YYYYMMDD}/
2. ECMWF Operational S2S Forecasts (from Copernicus/ECMWF MARS API)
   - Region: India domain (0-50°N, 55-105°E) at 1.5° resolution
   - Output path: /storage/raj.ayush/benchmark(jfm)/ecmwf/data/

Note: This script writes all data to the large /storage partition and 
automatically creates symbolic links in the workspace so local scripts continue 
to work seamlessly.
"""

import os
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

# --- PATH CONFIGURATIONS ---
# Actual storage paths on the large /storage partition
STORAGE_BASE_DIR = Path("/storage/raj.ayush/benchmark(jfm)")
FUXI_OUT_DIR = STORAGE_BASE_DIR / "fuxi_s2sm"
ECMWF_OUT_DIR = STORAGE_BASE_DIR / "ecmwf" / "data"

# Workspace paths (symlinks will be created here pointing to /storage)
WORKSPACE_BASE_DIR = Path("/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)")
WORKSPACE_FUXI_LINK = WORKSPACE_BASE_DIR / "fuxi_s2sm"
WORKSPACE_ECMWF_LINK = WORKSPACE_BASE_DIR / "ecmwf" / "data"

# Generate forecast initialization dates (Mondays & Thursdays in JFM 2026)
ALL_DAYS = pd.date_range("2026-01-01", "2026-03-31", freq="D")
INIT_DATES = [d for d in ALL_DAYS if d.weekday() in (0, 3)]  # Mon=0, Thu=3

# --- FUXI INITIAL CONDITIONS CONFIG ---
FUXI_PL_NAMES = [
    'geopotential', 'temperature', 'u_component_of_wind', 
    'v_component_of_wind', 'specific_humidity'
]
FUXI_SFC_NAMES = [
    '2m_temperature', '2m_dewpoint_temperature', 'sea_surface_temperature', 
    'top_net_thermal_radiation', '10m_u_component_of_wind', '10m_v_component_of_wind', 
    '100m_u_component_of_wind', '100m_v_component_of_wind', 'mean_sea_level_pressure', 
    'total_column_water_vapour', 'total_precipitation'
]
FUXI_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# --- ECMWF S2S CONFIG ---
ECMWF_AREA = [50, 55, 0, 105]       # [North, West, South, East] (India domain)
ECMWF_GRID = [1.5, 1.5]
ECMWF_STEPS = [str(h) for h in range(24, 1105, 24)]  # 1 to 46 days (24h intervals)


def ensure_symlink(target_path: Path, link_path: Path):
    """
    Creates target_path if it doesn't exist, and ensures link_path is a symlink 
    pointing to target_path.
    """
    target_path.mkdir(parents=True, exist_ok=True)
    
    if link_path.is_symlink():
        current_target = link_path.readlink().resolve()
        if current_target == target_path.resolve():
            return
        else:
            print(f"Updating existing symlink: {link_path} -> {target_path}")
            link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir() and not any(link_path.iterdir()):
            print(f"Removing empty workspace directory to make way for symlink: {link_path}")
            link_path.rmdir()
        else:
            print(f"Warning: {link_path} exists and is not empty. Cannot create symlink automatically.")
            return

    # Ensure parent of link path exists
    link_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        link_path.symlink_to(target_path, target_is_directory=True)
        print(f"Created symlink: {link_path} -> {target_path}")
    except Exception as e:
        print(f"Failed to create symlink {link_path} -> {target_path}: {e}")


def download_fuxi_init_conditions(dates):
    """
    Downloads FuXi S2S initial conditions from Google Cloud ARCO-ERA5 Zarr store.
    Slices variables to global 1.5 degree resolution (step of 6 from 0.25).
    """
    print("\n" + "=" * 65)
    print("DOWNLOAD: FuXi S2S Initial Conditions (ARCO-ERA5)")
    print(f"Storage Directory: {FUXI_OUT_DIR}")
    print(f"Workspace Link   : {WORKSPACE_FUXI_LINK}")
    print(f"Total Dates      : {len(dates)}")
    print("=" * 65)
    
    # Setup storage directory and symlink
    ensure_symlink(FUXI_OUT_DIR, WORKSPACE_FUXI_LINK)
    
    print("Connecting to Google Cloud ARCO-ERA5 dataset (Zarr)...")
    try:
        ds = xr.open_zarr(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", 
            storage_options={"token": "anon"}
        )
    except Exception as e:
        print(f"Error connecting to ARCO-ERA5: {e}")
        print("Please verify you have 'zarr' and 'gcsfs' installed in your environment.")
        return

    for idx, init_date in enumerate(dates):
        date_str = init_date.strftime("%Y-%m-%d")
        print(f"\n[{idx+1}/{len(dates)}] Extracting Initial Conditions for {date_str}...")
        
        # Output directory on /storage
        date_dir = FUXI_OUT_DIR / init_date.strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # FuXi S2S requires history step (init_date - 1 day) and current step (init_date)
        time0 = init_date - pd.Timedelta(days=1)
        time1 = init_date
        times = [time0.strftime("%Y-%m-%dT00:00:00"), time1.strftime("%Y-%m-%dT00:00:00")]
        
        # Slice the dataset: time, and every 6th pixel (0.25° * 6 = 1.5° resolution)
        # We also rename coordinate dimensions 'latitude'->'lat', 'longitude'->'lon'
        try:
            ds_sliced = ds.sel(
                time=times, 
                latitude=slice(None, None, 6), 
                longitude=slice(None, None, 6)
            ).rename({"latitude": "lat", "longitude": "lon"})
        except Exception as e:
            print(f"  [ERROR] Slicing dataset failed: {e}")
            continue

        # Extract Pressure Level variables
        for var in FUXI_PL_NAMES:
            outfile = date_dir / f"{var}.nc"
            if outfile.exists() and outfile.stat().st_size > 0:
                print(f"  [SKIPPED] {outfile.name} already exists.")
                continue
            
            print(f"  -> Extracting PL variable: {var}...")
            try:
                var_data = ds_sliced[var].sel(level=FUXI_LEVELS).compute()
                var_data.to_netcdf(outfile)
            except Exception as e:
                print(f"    [ERROR] Failed to extract {var}: {e}")

        # Extract Surface variables
        for var in FUXI_SFC_NAMES:
            outfile = date_dir / f"{var}.nc"
            if outfile.exists() and outfile.stat().st_size > 0:
                print(f"  [SKIPPED] {outfile.name} already exists.")
                continue
            
            print(f"  -> Extracting Surface variable: {var}...")
            try:
                var_data = ds_sliced[var].compute()
                
                # Check for accidental level coordinates
                if 'level' in var_data.dims:
                    var_data = var_data.isel(level=0)
                
                # Add dummy level coord expected by FuXi S2S
                var_data = var_data.assign_coords(level=1000).expand_dims("level")
                
                var_data.to_netcdf(outfile)
            except Exception as e:
                print(f"    [ERROR] Failed to extract {var}: {e}")
                
        print(f"Finished extracting FuXi initial conditions for {date_str}.")


def download_ecmwf_s2s(dates):
    """
    Downloads ECMWF Operational S2S Forecasts using the cdsapi client.
    Downloads control forecast (cf) and perturbed forecast (pf) for surface/pressure fields.
    """
    print("\n" + "=" * 65)
    print("DOWNLOAD: ECMWF Operational S2S Benchmark Forecasts")
    print(f"Storage Directory: {ECMWF_OUT_DIR}")
    print(f"Workspace Link   : {WORKSPACE_ECMWF_LINK}")
    print(f"Total Dates      : {len(dates)}")
    print(f"Grid Region      : {ECMWF_AREA} (India domain at 1.5°)")
    print("=" * 65)

    # Setup storage directory and symlink
    ensure_symlink(ECMWF_OUT_DIR, WORKSPACE_ECMWF_LINK)

    try:
        import cdsapi
    except ImportError:
        print("Error: 'cdsapi' python package is not installed.")
        print("Please install it or activate the appropriate conda environment (e.g. s2s-hind).")
        return

    # Check for ~/.ecmwfapirc API key
    api_rc = Path.home() / ".ecmwfapirc"
    if not api_rc.exists():
        print(f"Warning: ECMWF API credentials file (~/.ecmwfapirc) not found.")
        print(f"Please configure your ECMWF key at {api_rc} before downloading.")
        print(f"See: https://api.ecmwf.int/v1/key/ for details.")
        return

    client = cdsapi.Client()

    for idx, init_date in enumerate(dates):
        date_str = init_date.strftime("%Y-%m-%d")
        print(f"\n[{idx+1}/{len(dates)}] Requesting ECMWF forecasts for Init Date: {date_str}")
        
        # Loop through control forecast (cf) and perturbed forecast (pf)
        for ftype in ("cf", "pf"):
            # 1. Surface download
            sfc_file = ECMWF_OUT_DIR / f"sfc_{ftype}_{init_date:%Y%m%d}.grib"
            if sfc_file.exists() and sfc_file.stat().st_size > 0:
                print(f"  [SKIPPED] {sfc_file.name} already exists.")
            else:
                print(f"  [REQUESTING] {sfc_file.name} ...")
                sfc_request = {
                    "origin": "ecmwf",
                    "forecast_type": "control_forecast" if ftype == "cf" else "perturbed_forecast",
                    "level_type": "single_level",
                    "variable": ["2t", "mx2t6", "mn2t6", "tp"],
                    "year": str(init_date.year),
                    "month": f"{init_date.month:02d}",
                    "day": f"{init_date.day:02d}",
                    "time": "00:00",
                    "step": ECMWF_STEPS,
                    "area": ECMWF_AREA,
                    "grid": ECMWF_GRID,
                    "data_format": "grib"
                }
                try:
                    client.retrieve("s2s-forecasts", sfc_request, str(sfc_file))
                    print(f"  [SUCCESS] Saved {sfc_file.name}")
                except Exception as e:
                    print(f"  [ERROR] Surface retrieve failed for {ftype} on {date_str}: {e}")

            # 2. Pressure level download (gh @ 500 hPa)
            pl_file = ECMWF_OUT_DIR / f"pl_{ftype}_{init_date:%Y%m%d}.grib"
            if pl_file.exists() and pl_file.stat().st_size > 0:
                print(f"  [SKIPPED] {pl_file.name} already exists.")
            else:
                print(f"  [REQUESTING] {pl_file.name} ...")
                pl_request = {
                    "origin": "ecmwf",
                    "forecast_type": "control_forecast" if ftype == "cf" else "perturbed_forecast",
                    "level_type": "pressure",
                    "variable": "gh",
                    "level": "500",
                    "year": str(init_date.year),
                    "month": f"{init_date.month:02d}",
                    "day": f"{init_date.day:02d}",
                    "time": "00:00",
                    "step": ECMWF_STEPS,
                    "area": ECMWF_AREA,
                    "grid": ECMWF_GRID,
                    "data_format": "grib"
                }
                try:
                    client.retrieve("s2s-forecasts", pl_request, str(pl_file))
                    print(f"  [SUCCESS] Saved {pl_file.name}")
                except Exception as e:
                    print(f"  [ERROR] PL retrieve failed for {ftype} on {date_str}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download benchmarks (FuXi initial conditions and ECMWF S2S data).")
    parser.add_argument("--fuxi", action="store_true", help="Download FuXi-S2S initial conditions from ARCO-ERA5 Zarr.")
    parser.add_argument("--ecmwf", action="store_true", help="Download ECMWF operational S2S forecast benchmark data.")
    parser.add_argument("--all", action="store_true", help="Download both FuXi initial conditions and ECMWF data.")
    parser.add_argument("--date", type=str, default=None, help="Process a single init date (YYYYMMDD) instead of all JFM 2026.")
    
    args = parser.parse_args()
    
    if not (args.fuxi or args.ecmwf or args.all):
        parser.print_help()
        print("\nError: You must specify --fuxi, --ecmwf, or --all.")
        sys.exit(1)

    # Resolve dates
    if args.date:
        try:
            dates = [pd.to_datetime(args.date)]
        except Exception as e:
            print(f"Error parsing date '{args.date}': {e}")
            sys.exit(1)
    else:
        dates = INIT_DATES
        
    if args.fuxi or args.all:
        download_fuxi_init_conditions(dates)
        
    if args.ecmwf or args.all:
        download_ecmwf_s2s(dates)

    print("\nUnified download process complete.")


if __name__ == "__main__":
    main()
