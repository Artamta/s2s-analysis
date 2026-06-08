#!/usr/bin/env python
"""
Inspect and Visualize ECMWF and NCEP S2S Benchmark Datasets
This script prints metadata and plots a side-by-side comparison of 
surface variables for ECMWF and NCEP.
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Try importing cartopy for enhanced map features, but fall back to standard matplotlib if unavailable
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def find_latest_file(dataset_dir, pattern="sfc_cf*.grib"):
    """Finds the first available GRIB file matching the pattern, falling back to any .grib if not found."""
    grib_files = sorted(glob.glob(os.path.join(dataset_dir, pattern)))
    if not grib_files:
        grib_files = sorted(glob.glob(os.path.join(dataset_dir, "*.grib")))
    if not grib_files:
        raise FileNotFoundError(f"No GRIB files found in {dataset_dir}")
    return grib_files[0]

def inspect_dataset(file_path, name):
    """Opens a GRIB dataset and prints its core metadata."""
    print(f"\n==========================================")
    print(f" Inspecting {name} Dataset")
    print(f"==========================================")
    print(f"File Path: {file_path}")
    print(f"File Size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    try:
        # Load dataset using cfgrib
        ds = xr.open_dataset(file_path, engine='cfgrib')
        print("\n--- Dataset Info ---")
        print(ds)
        print("\n--- Coordinates ---")
        for coord in ds.coords:
            c_vals = ds[coord].values
            if hasattr(c_vals, 'shape') and c_vals.ndim > 0:
                print(f"  {coord}: shape={c_vals.shape}, min={c_vals.min()}, max={c_vals.max()}")
            else:
                print(f"  {coord}: scalar={c_vals}")
        
        print("\n--- Variables ---")
        for var in ds.data_vars:
            attrs = ds[var].attrs
            print(f"  {var}: dimensions={ds[var].dims}, units={attrs.get('units', 'N/A')}, long_name='{attrs.get('longName', 'N/A')}'")
        return ds
    except Exception as e:
        print(f"Error opening dataset with cfgrib: {e}")
        return None

def plot_comparison(ecmwf_ds, ncep_ds, var_name, step_index=6, save_path=None):
    """Plots a side-by-side comparison of ECMWF and NCEP for a specific forecast step."""
    if ecmwf_ds is None or ncep_ds is None:
        print("Cannot plot comparison because one or both datasets failed to load.")
        return

    if var_name not in ecmwf_ds.data_vars or var_name not in ncep_ds.data_vars:
        available_ec = list(ecmwf_ds.data_vars)
        available_nc = list(ncep_ds.data_vars)
        print(f"Variable '{var_name}' not available in both datasets.")
        print(f"ECMWF has: {available_ec}, NCEP has: {available_nc}")
        # Try to find a common variable
        common = list(set(available_ec).intersection(available_nc))
        if common:
            var_name = common[0]
            print(f"Falling back to common variable: '{var_name}'")
        else:
            print("No common variables found to plot.")
            return

    # Select variables at specific forecast step
    ecmwf_slice = ecmwf_ds[var_name].isel(step=step_index)
    ncep_slice = ncep_ds[var_name].isel(step=step_index)
    
    # Get metadata for titles
    ecmwf_time = np.datetime_as_string(ecmwf_slice.time.values, unit='D')
    ecmwf_step_days = ecmwf_slice.step.values / np.timedelta64(1, 'D')
    
    ncep_time = np.datetime_as_string(ncep_slice.time.values, unit='D')
    ncep_step_days = ncep_slice.step.values / np.timedelta64(1, 'D')

    # Create figure
    fig = plt.figure(figsize=(14, 6))
    
    # Setup subplots with or without Cartopy projection
    if HAS_CARTOPY:
        projection = ccrs.PlateCarree()
        ax1 = fig.add_subplot(1, 2, 1, projection=projection)
        ax2 = fig.add_subplot(1, 2, 2, projection=projection)
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
    # Colormap and levels setup
    cmap = 'coolwarm'
    if var_name == 'tp':
        cmap = 'Blues'
    elif var_name == 'gh':
        cmap = 'viridis'
        
    # Plot ECMWF
    lon, lat = ecmwf_ds.longitude.values, ecmwf_ds.latitude.values
    data1 = ecmwf_slice.values
    # Convert Kelvin to Celsius for temperatures if units are Kelvin
    units = ecmwf_ds[var_name].attrs.get('units', 'K')
    if units == 'K' and var_name in ['mx2t6', 'mn2t6', 't2m']:
        data1 = data1 - 273.15
        units_label = '°C'
    else:
        units_label = units
        
    if HAS_CARTOPY:
        ax1.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE, linewidth=1)
        gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl1.top_labels = False
        gl1.right_labels = False
        
        im1 = ax1.pcolormesh(lon, lat, data1, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')
    else:
        im1 = ax1.pcolormesh(lon, lat, data1, cmap=cmap, shading='auto')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True)
        
    ax1.set_title(f"ECMWF Control - {var_name.upper()}\nInit: {ecmwf_time} | Forecast: +{ecmwf_step_days:.0f} days")
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08)
    cbar1.set_label(f"{ecmwf_ds[var_name].attrs.get('longName', var_name)} ({units_label})")

    # Plot NCEP
    data2 = ncep_slice.values
    if units == 'K' and var_name in ['mx2t6', 'mn2t6', 't2m']:
        data2 = data2 - 273.15
        
    if HAS_CARTOPY:
        ax2.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.COASTLINE, linewidth=1)
        gl2 = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl2.top_labels = False
        gl2.right_labels = False
        
        im2 = ax2.pcolormesh(lon, lat, data2, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')
    else:
        im2 = ax2.pcolormesh(lon, lat, data2, cmap=cmap, shading='auto')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.grid(True)
        
    ax2.set_title(f"NCEP Control - {var_name.upper()}\nInit: {ncep_time} | Forecast: +{ncep_step_days:.0f} days")
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.08)
    cbar2.set_label(f"{ncep_ds[var_name].attrs.get('longName', var_name)} ({units_label})")

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")
    else:
        plt.show()

def upsample_dataset(ds, target_lat, target_lon, method='linear'):
    """Bilinearly interpolates the dataset's latitude and longitude to a 0.5 degree target grid."""
    if ds is None:
        return None
    try:
        # Interpolate the dataset onto the target lat/lon grid
        ds_upsampled = ds.interp(latitude=target_lat, longitude=target_lon, method=method)
        
        # Copy attributes from original variables to the new ones
        for var in ds.data_vars:
            if var in ds_upsampled.data_vars:
                ds_upsampled[var].attrs = ds[var].attrs.copy()
        ds_upsampled.attrs = ds.attrs.copy()
        return ds_upsampled
    except Exception as e:
        print(f"Error upsampling dataset: {e}")
        return ds

def main():
    # Since script is in benchmark(jfm)/analysis/
    # base_dir is benchmark(jfm)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ecmwf_dir = os.path.join(base_dir, "ecmwf", "data")
    ncep_dir = os.path.join(base_dir, "ncep", "data")
    
    try:
        # Prefer sfc_cf, fallback to pl_cf
        ecmwf_file = find_latest_file(ecmwf_dir, "sfc_cf*.grib")
        ncep_file = find_latest_file(ncep_dir, "sfc_cf*.grib")
    except FileNotFoundError as e:
        print(e)
        return

    # Inspect datasets
    ecmwf_ds = inspect_dataset(ecmwf_file, "ECMWF")
    ncep_ds = inspect_dataset(ncep_file, "NCEP")
    
    # Define target 0.5-degree grid matching Spire's coordinates
    target_lat = np.arange(0.0, 50.1, 0.5)
    target_lon = np.arange(55.0, 105.1, 0.5)
    
    print("\n==========================================")
    print(" Upsampling to 0.5° Grid (Spire Resolution)")
    print("==========================================")
    print(f"Interpolating ECMWF from {ecmwf_ds['mx2t6'].shape[1:]} to (101, 101)...")
    ecmwf_ds_upsampled = upsample_dataset(ecmwf_ds, target_lat, target_lon)
    print(f"Interpolating NCEP from {ncep_ds['mx2t6'].shape[1:]} to (101, 101)...")
    ncep_ds_upsampled = upsample_dataset(ncep_ds, target_lat, target_lon)
    
    # Decide which variable to plot (mx2t6 is default, fallback to gh or other)
    target_var = 'mx2t6'
    plot_path = os.path.join(base_dir, "analysis", "figures", "ecmwf_ncep_comparison_0.5deg.png")
    plot_comparison(ecmwf_ds_upsampled, ncep_ds_upsampled, var_name=target_var, step_index=6, save_path=plot_path)

if __name__ == "__main__":
    main()
