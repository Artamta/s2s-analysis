#!/usr/bin/env python
import os
import argparse
import pandas as pd
import xarray as xr
import numpy as np

# Pl and Sfc names required by FuXi-S2S data_util.py
pl_names = [
    'geopotential', 'temperature', 'u_component_of_wind', 
    'v_component_of_wind', 'specific_humidity'
]

sfc_names = [
    '2m_temperature', '2m_dewpoint_temperature', 'sea_surface_temperature', 
    'top_net_thermal_radiation', '10m_u_component_of_wind', '10m_v_component_of_wind', 
    '100m_u_component_of_wind', '100m_v_component_of_wind', 'mean_sea_level_pressure', 
    'total_column_water_vapour', 'total_precipitation'
]

levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

def download_for_date(ds, init_date, out_dir):
    date_dir = os.path.join(out_dir, init_date.strftime("%Y%m%d"))
    os.makedirs(date_dir, exist_ok=True)
    
    time0 = init_date - pd.Timedelta(days=1)
    time1 = init_date
    times = [time0.strftime("%Y-%m-%dT00:00:00"), time1.strftime("%Y-%m-%dT00:00:00")]
    
    print(f"[{init_date.strftime('%Y-%m-%d')}] Extracting times: {times}")
    
    # Slice time, lat, lon (every 6th pixel for 1.5 degree resolution from 0.25 degree)
    # ARCO-ERA5 lat: 90 to -90, lon: 0 to 359.75
    # 721 lats -> 121 lats; 1440 lons -> 240 lons
    ds_sliced = ds.sel(
        time=times, 
        latitude=slice(None, None, 6), 
        longitude=slice(None, None, 6)
    )
    
    # Process pressure level variables
    for name in pl_names:
        outfile = os.path.join(date_dir, f"{name}.nc")
        if os.path.exists(outfile):
            continue
        print(f"  -> extracting {name}...")
        try:
            var_data = ds_sliced[name].sel(level=levels).compute()
            var_data.to_netcdf(outfile)
        except Exception as e:
            print(f"  Error on {name}: {e}")

    # Process surface variables
    for name in sfc_names:
        outfile = os.path.join(date_dir, f"{name}.nc")
        if os.path.exists(outfile):
            continue
        print(f"  -> extracting {name}...")
        try:
            var_data = ds_sliced[name].compute()
            # If variable has level dimension by accident (like in some datasets), drop it
            if 'level' in var_data.dims:
                var_data = var_data.isel(level=0)
            
            # Reformat dummy level coordinate as expected by data_util.py 
            var_data = var_data.assign_coords(level=1000).expand_dims("level")
            
            var_data.to_netcdf(outfile)
        except Exception as e:
            print(f"  Error on {name}: {e}")

    print(f"[{init_date.strftime('%Y-%m-%d')}] Finished!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Specific init date (YYYYMMDD). If not provided, runs for all JFM 2026.", type=str, default=None)
    parser.add_argument("--outdir", help="Output directory", type=str, default="data")
    args = parser.parse_args()
    
    print("Connecting to Google Cloud ARCO-ERA5 dataset...")
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", 
        storage_options={"token": "anon"}
    )
    
    if args.date:
        init_dates = [pd.to_datetime(args.date)]
    else:
        # All Mondays and Thursdays in JFM 2026
        all_days = pd.date_range("2026-01-01", "2026-03-31", freq="D")
        init_dates = [d for d in all_days if d.weekday() in (0, 3)]
    
    print(f"Found {len(init_dates)} initialization dates to process.")
    
    for init_date in init_dates:
        download_for_date(ds, init_date, args.outdir)

if __name__ == "__main__":
    main()
