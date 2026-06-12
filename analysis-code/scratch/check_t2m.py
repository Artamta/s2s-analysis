import xarray as xr
import numpy as np
import glob

print("--- Checking ERA5 t2m ---")
try:
    era_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface.grib', engine='cfgrib', filter_by_keys={'shortName': '2t'})
    t2m_era = era_ds['t2m']
    print(f"ERA5 shape: {t2m_era.shape}, NaNs: {np.isnan(t2m_era).any().values}, min: {t2m_era.min().values:.2f}, max: {t2m_era.max().values:.2f}")
except Exception as e:
    print(f"Error loading ERA5: {e}")

print("\n--- Checking ECMWF t2m ---")
ec_files = sorted(glob.glob('/storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_cf_*.grib'))
for f in ec_files:
    try:
        ds = xr.open_dataset(f, engine='cfgrib', filter_by_keys={'shortName': '2t'})
        v = ds['t2m']
        print(f"ECMWF {f.split('/')[-1]}: shape {v.shape}, NaNs {np.isnan(v).any().values}, min {v.min().values:.2f}, max {v.max().values:.2f}")
    except Exception as e:
        print(f"Error in {f.split('/')[-1]}: {e}")

print("\n--- Checking NCEP t2m ---")
nc_files = sorted(glob.glob('/storage/raj.ayush/s2s-forecast-data/ncep/data/sfc_cf_*.grib'))
for f in nc_files:
    try:
        ds = xr.open_dataset(f, engine='cfgrib')
        if 't2m' in ds: v = ds['t2m']
        elif 'mx2t6' in ds: v = ds['mx2t6']
        else:
            print(f"NCEP {f.split('/')[-1]}: No t2m or mx2t6 found!")
            continue
        print(f"NCEP {f.split('/')[-1]}: shape {v.shape}, NaNs {np.isnan(v).any().values}, min {v.min().values:.2f}, max {v.max().values:.2f}")
    except Exception as e:
        print(f"Error in {f.split('/')[-1]}: {e}")
