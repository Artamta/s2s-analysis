import os
import sys
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Add the parent directory to the path so we can import our data loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loaders.model_loader import load_ecmwf, load_ncep, load_spire

# Master config
INIT_DATE = "20260101"
SPIRE_DATE = "2026-01-01"
OUT_FIG = "../figures/verification/jan1_benchmark_comparison.png"

def main():
    print(f"Loading data for {INIT_DATE}...")
    
    # 1. Load File Paths from the Data Loader
    ecmwf_paths = load_ecmwf(INIT_DATE)
    ncep_paths = load_ncep(INIT_DATE)
    
    # Grab just the Control Forecasts (CF) for this plot to keep it simple
    ecmwf_cf_file = ecmwf_paths['control'][0] if ecmwf_paths['control'] else None
    ncep_cf_file = ncep_paths['control'][0] if ncep_paths['control'] else None
    
    # 2. Open Datasets
    print("Opening Spire Zarr...")
    spire_ds = load_spire(SPIRE_DATE)
    
    print("Opening ECMWF GRIB...")
    # ECMWF needs an explicit shortName filter to ignore precipitation
    ecmwf_ds = xr.open_dataset(ecmwf_cf_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
    
    print("Opening NCEP GRIB...")
    # NCEP defaults to loading mx2t6 and mn2t6 properly without filters
    ncep_ds = xr.open_dataset(ncep_cf_file, engine='cfgrib')
    
    # 3. Extract the 10-day Forecast (Lead Time) for Temperature
    # In ECMWF/NCEP GRIBs, 'step' is the lead time. Step 10 days = 10 days ahead.
    # In Spire, the coordinate might be 'lead_time' or similar. We will select the 10th index.
    
    print("Extracting Day 1 Temperature Forecasts...")
    # ECMWF only has a scalar step in this specific GRIB file, so we just take the 2D array
    e_temp = ecmwf_ds['t2m']
        
    try:
        # NCEP has 44 steps. We grab step 0 (Day 1)
        n_temp = (ncep_ds['mx2t6'].isel(step=0) + ncep_ds['mn2t6'].isel(step=0)) / 2.0
    except KeyError:
        n_temp = ncep_ds['mx2t6'].isel(step=0)

    # Spire has all steps. We grab lead_time=0 (Day 1)
    s_temp = spire_ds['t2m'].isel(lead_time=0)

    # 4. Create a beautiful 1x3 Plot using Cartopy
    print("Generating the spatial map...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0.1)

    models = [("Spire Hindcast", s_temp), ("ECMWF Control", e_temp), ("NCEP Control", n_temp)]
    
    for ax, (title, data) in zip(axes, models):
        # Add beautiful map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        # Plot the data
        # We use standard limits based on the India Domain (0-50N, 55-105E)
        ax.set_extent([55, 105, 0, 50], crs=ccrs.PlateCarree())
        
        # Plotting the temperature (Kelvin usually, we can plot raw values)
        im = data.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, cmap='coolwarm')
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add a shared colorbar at the bottom
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label("2m Temperature (K)", fontsize=12)
    
    fig.suptitle("S2S Benchmark Comparison: 10-Day Temperature Forecast (Jan 1, 2026)", fontsize=18, y=1.05)
    
    # 5. Save the Figure
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, bbox_inches='tight', dpi=300)
    print(f"Plot successfully saved to: {OUT_FIG}")
    
if __name__ == "__main__":
    main()
