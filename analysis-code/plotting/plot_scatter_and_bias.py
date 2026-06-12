import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loaders.model_loader import load_ecmwf, load_ncep, load_spire, load_fuxi, regrid_to_common
from verification.metrics import get_india_mainland

# Config
INIT_DATE = "20260101"
SPIRE_DATE = "2026-01-01"
ERA5_FILE = "/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface_z500_20260101.nc"
OUT_FIG = "../figures/verification/jan1_scatter_and_bias.png"

def main():
    print("Loading datasets...")
    # Load Models
    ecmwf_paths = load_ecmwf(INIT_DATE)
    ncep_paths = load_ncep(INIT_DATE)
    
    ecmwf_ds = xr.open_dataset(ecmwf_paths['control'][0], engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
    ncep_ds = xr.open_dataset(ncep_paths['control'][0], engine='cfgrib')
    spire_ds = load_spire(SPIRE_DATE)
    fuxi_ds = load_fuxi(INIT_DATE)
    era5_ds = xr.open_dataset(ERA5_FILE)
    
    # Extract Day 1 Temp
    print("Extracting Day 1...")
    e_temp = ecmwf_ds['t2m']
    try:
        n_temp = (ncep_ds['mx2t6'].isel(step=0) + ncep_ds['mn2t6'].isel(step=0)) / 2.0
    except KeyError:
        n_temp = ncep_ds['mx2t6'].isel(step=0)
    s_temp = spire_ds['t2m'].isel(lead_time=0)
    f_temp = fuxi_ds['data'].sel(channel='t2m').isel(lead_time=0)
    truth_temp = era5_ds['t2m'].isel(time=0)
    
    # Regrid & Crop to India
    print("Regridding and Cropping to India...")
    models = {"FuXi AI": f_temp, "Spire": s_temp, "ECMWF": e_temp, "NCEP": n_temp}
    
    truth_grid = get_india_mainland(regrid_to_common(truth_temp))
    
    processed_models = {}
    for name, data in models.items():
        regridded = regrid_to_common(data)
        cropped = get_india_mainland(regridded)
        processed_models[name] = cropped
        
    # --- PLOTTING ---
    print("Generating Professor Singh's requested plots...")
    fig = plt.figure(figsize=(24, 12))
    
    # We will make 2 rows, 4 columns.
    # Top Row: Spatial Bias Maps (Model - ERA5)
    # Bottom Row: Scatter Plots (ERA5 vs Model)
    
    truth_flat = truth_grid.values.flatten()
    # Remove NaNs if any (due to ocean masking or boundaries)
    valid_mask = ~np.isnan(truth_flat)
    truth_flat = truth_flat[valid_mask]
    
    for i, (name, model_data) in enumerate(processed_models.items()):
        # Calculate Bias (Model - Truth)
        bias = model_data - truth_grid
        
        # 1. SPATIAL BIAS MAP
        ax_map = fig.add_subplot(2, 4, i + 1, projection=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.COASTLINE, linewidth=1)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')
        ax_map.set_extent([68, 98, 8, 38], crs=ccrs.PlateCarree())
        
        # Plot bias (coolwarm: Blue=Cold bias, Red=Hot bias)
        im = bias.plot(ax=ax_map, transform=ccrs.PlateCarree(), add_colorbar=False, cmap='coolwarm', vmin=-10, vmax=10)
        ax_map.set_title(f"{name} Spatial Bias (Model - ERA5)", fontsize=14, fontweight='bold')
        
        # 2. SCATTER PLOT
        model_flat = model_data.values.flatten()[valid_mask]
        
        # Calculate Metrics
        rmse = np.sqrt(mean_squared_error(truth_flat, model_flat))
        mae = mean_absolute_error(truth_flat, model_flat)
        r2 = r2_score(truth_flat, model_flat)
        
        ax_scatter = fig.add_subplot(2, 4, i + 5)
        ax_scatter.scatter(truth_flat, model_flat, alpha=0.5, color='teal', edgecolors='k', linewidth=0.5)
        
        # 1-to-1 Perfect line
        min_val = min(np.nanmin(truth_flat), np.nanmin(model_flat))
        max_val = max(np.nanmax(truth_flat), np.nanmax(model_flat))
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Forecast')
        
        ax_scatter.set_title(f"{name} vs ERA5 Scatter", fontsize=14, fontweight='bold')
        ax_scatter.set_xlabel("ERA5 Temperature (K)", fontsize=12)
        ax_scatter.set_ylabel(f"{name} Temperature (K)", fontsize=12)
        
        # Annotate Metrics
        textstr = '\n'.join((
            f'$R^2$ = {r2:.3f}',
            f'RMSE = {rmse:.2f} K',
            f'MAE = {mae:.2f} K'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax_scatter.text(0.05, 0.95, textstr, transform=ax_scatter.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
        ax_scatter.grid(True, linestyle='--', alpha=0.6)

    # Add shared colorbar for the maps
    cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Temperature Bias (Kelvin)", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, bbox_inches='tight', dpi=300)
    print(f"Plot successfully saved to: {OUT_FIG}")

if __name__ == "__main__":
    main()
