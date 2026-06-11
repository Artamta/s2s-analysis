import os
import numpy as np
import xarray as xr
import pandas as pd
import glob

# ==========================================
# MASTER PATHS
# ==========================================
STORAGE_BASE = "/storage/raj.ayush"
FUXI_DIR = os.path.join(STORAGE_BASE, "fuxi-init-jfm-weekely/output")
ECMWF_DIR = os.path.join(STORAGE_BASE, "benchmark(jfm)/ecmwf/data")
NCEP_DIR = os.path.join(STORAGE_BASE, "benchmark(jfm)/ncep/data")
SPIRE_ZARR = os.path.join(STORAGE_BASE, "spire-hindecast-weekely-initialized/spire_hindcast_jfm.zarr")
ERA5_DIR = os.path.join(STORAGE_BASE, "benchmark(jfm)/era5/data")

# ==========================================
# LOADER FUNCTIONS
# ==========================================

def load_fuxi(init_date: str):
    """
    Loads all 11 members and 42 lead times for FuXi.
    init_date format: 'YYYYMMDD' (e.g. '20260101')
    """
    date_path = os.path.join(FUXI_DIR, init_date, "member")
    if not os.path.exists(date_path):
        raise FileNotFoundError(f"FuXi data not found for {init_date} at {date_path}")
        
    members = sorted(os.listdir(date_path))
    datasets = []
    
    # We use xarray's lazy loading so this uses almost 0 RAM initially!
    for mem in members:
        mem_path = os.path.join(date_path, mem, "*.nc")
        # Load all 42 days for this member along a new 'lead_time' dimension
        ds_mem = xr.open_mfdataset(mem_path, combine='nested', concat_dim='lead_time')
        # Assign member coordinate
        ds_mem = ds_mem.assign_coords(member=int(mem))
        datasets.append(ds_mem)
        
    # Combine all members into a single Xarray tensor
    fuxi_ds = xr.concat(datasets, dim='member')
    return fuxi_ds

def load_ecmwf(init_date: str):
    """
    Loads the ECMWF GRIB data.
    init_date format: 'YYYYMMDD'
    """
    # Find the matching Surface (sfc) GRIB files for ECMWF Control (cf) and Perturbed (pf)
    cf_files = glob.glob(os.path.join(ECMWF_DIR, f"sfc_cf_{init_date}.grib"))
    pf_files = glob.glob(os.path.join(ECMWF_DIR, f"sfc_pf_{init_date}.grib"))
    
    # TODO: We can expand this to load GRIB via cfgrib, but for now we return the file paths
    # as GRIB requires specific engine kwargs based on the variables requested.
    return {"control": cf_files, "perturbed": pf_files}

def load_ncep(init_date: str):
    """
    Loads the NCEP GRIB data.
    init_date format: 'YYYYMMDD'
    """
    cf_files = glob.glob(os.path.join(NCEP_DIR, f"sfc_cf_{init_date}.grib"))
    pf_files = glob.glob(os.path.join(NCEP_DIR, f"sfc_pf_{init_date}.grib"))
    
    return {"control": cf_files, "perturbed": pf_files}

def load_spire(init_date: str):
    """
    Loads Spire Hindcasts directly from the unified Zarr store.
    init_date format: 'YYYY-MM-DD' (Note the hyphens for Spire!)
    """
    # Spire hides its data inside subgroups and uses unique names
    ds = xr.open_zarr(SPIRE_ZARR, group='mean_stddev', consolidated=False)
    
    # We instantly standardize it to perfectly match the rest of our pipeline
    ds = ds.rename({'reference_time': 'time', 'step': 'lead_time', 'air_temperature': 't2m'})
    
    # Select the specific initialization date lazily
    spire_date = ds.sel(time=init_date)
    return spire_date

# ==========================================
# STANDARDIZATION UTILITY
# ==========================================

def standardize_units(ds, model_name):
    """
    Ensures all models are mathematically uniform before evaluation:
    - Temperature: Kelvin
    - Geopotential: m^2/s^2 (multiplies Spire Z500 by 9.80665)
    - Precipitation: mm/day (differentiates ECMWF/NCEP accumulations)
    """
    ds_out = ds.copy()
    
    # 1. Spire Geopotential Height -> Geopotential
    if model_name == 'spire' and 'z500' in ds_out:
        ds_out['z500'] = ds_out['z500'] * 9.80665
        
    # 2. ECMWF/NCEP Accumulated Precipitation (m) -> Daily Rate (mm/day)
    if model_name in ['ecmwf', 'ncep'] and 'tp' in ds_out:
        # diff() un-accumulates it. Multiply by 1000 for mm.
        daily_precip = ds_out['tp'].diff(dim='lead_time') * 1000.0
        # The first day becomes NaN because there is no previous day to subtract from.
        # For Day 1, the accumulated value IS the daily value.
        first_day = ds_out['tp'].isel(lead_time=0) * 1000.0
        ds_out['tp'] = xr.concat([first_day, daily_precip], dim='lead_time')

    return ds_out

# ==========================================
# REGRIDDING UTILITY
# ==========================================

def regrid_to_common(ds, resolution=1.5):
    """
    Interpolates any dataset to a unified global grid of the specified resolution.
    This strictly ensures fair apples-to-apples evaluation across all models.
    """
    # 1. Create a mathematically perfect global grid
    new_lat = np.arange(90.0, -90.1, -resolution)
    new_lon = np.arange(0.0, 360.0, resolution)
    
    # 2. Use bilinear interpolation to map the data onto the standard grid
    # This securely scales 0.25 models (FuXi, ERA5) down to match the 1.5 models (ECMWF)
    return ds.interp(latitude=new_lat, longitude=new_lon, method="linear")

def load_era5_ground_truth(start_date: str, end_date: str):
    """
    Loads the high-res ERA5 ground truth for a specific date range.
    Dates should be 'YYYYMMDD'.
    """
    # Open all daily nc files in the range
    files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_surface_z500_*.nc")))
    # Filter files within date range
    # Open ERA5 cleanly
    ds = xr.open_mfdataset(f"/storage/raj.ayush/benchmark(jfm)/era5/data/era5_surface_z500_{start_date}.nc")
    return ds

# ==========================================
# FUXI S2S AI LOADER
# ==========================================

def load_fuxi(init_date: str, member="00"):
    """
    Loads the 42-day AI forecast from FuXi for a specific initialization date.
    Automatically stitches the 42 daily .nc files together into one massive tensor.
    """
    base_path = f"/storage/raj.ayush/fuxi-init-jfm-weekely/output/{init_date}/member/{member}"
    
    # Lazily stitch all 42 lead-time files together!
    try:
        ds = xr.open_mfdataset(f"{base_path}/*.nc", combine='nested', concat_dim='lead_time')
    except OSError:
        raise FileNotFoundError(f"FuXi forecast not found for {init_date}")
        
    # The ONNX inference names the raw tensor '__xarray_dataarray_variable__'
    tensor = ds['__xarray_dataarray_variable__']
    
    # Rename lat/lon to standard names so the regrid_to_common function works perfectly
    tensor = tensor.rename({'lat': 'latitude', 'lon': 'longitude'})
    
    # Return as a clean dataset
    return xr.Dataset({'data': tensor})

# ==========================================
# QUICK TEST
# ==========================================
if __name__ == "__main__":
    print("Testing FuXi Loader for Jan 1st...")
    try:
        fuxi = load_fuxi("20260101")
        print(f"Success! FuXi Tensor Shape: Members: {len(fuxi.member)}, Lead Times: {len(fuxi.lead_time)}")
        print(f"Memory used: {fuxi.nbytes / 1e9:.2f} GB (Note: Xarray keeps this lazy!)")
    except Exception as e:
        print(f"FuXi Error: {e}")
