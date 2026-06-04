import os
import sys
import gc
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Headless cluster environment execution
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

OUTPUT_DIR = "paper_figures_multivar_final_calibrated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# CONFIGURATION SETTINGS
# ----------------------------------------------------
TARGET_LEAD_DAYS = [7, 14, 21, 28]
init_idx = 40 

# Geographic Slicing Box adjusted for Full India + Ocean context
lon_min, lon_max = 65.0, 98.0
lat_min, lat_max = 5.0, 39.0

# ----------------------------------------------------
# UNIFIED INTERLOCKED VALIDATION MATRIX
# ----------------------------------------------------
VALIDATION_MATRIX = {
    "T2M": {
        "spire_var": "air_temperature",
        "spire_std": "air_temperature_stddev",
        "era5_var": "2m_temperature",
        "conversion": "kelvin_to_celsius",
        "units": "°C",
        "abs_cmap": "magma",
        "bias_cmap": "RdBu_r",
        # ADJUSTED: Extended from -15.0 to -30.0 to prevent the Tibetan Plateau from clipping into solid black blobs
        "vmin_abs": -30.0,
        "vmax_abs": 50.0,
        "vmax_bias": 8.0,    
        "vmax_spread": 4.0   
    },
    "Total_Precip": {
        "spire_var": "total_precipitation",
        "spire_std": "total_precipitation_stddev",
        "era5_var": "total_precipitation",
        "conversion": "era5_meters_to_mm_daily_sum",
        "units": "mm/day",
        "abs_cmap": "YlGnBu",
        "bias_cmap": "BrBG",  
        "vmin_abs": 0.0,
        "vmax_abs": 50.0,
        "vmax_bias": 20.0,   
        "vmax_spread": 15.0  
    },
    "Z500": {
        "spire_var": "geopotential_height_500hPa",
        "spire_std": "geopotential_height_500hPa_stddev",
        "era5_var": "geopotential",
        "conversion": "era5_w_to_gpm_500hPa",
        "units": "gpm",
        "abs_cmap": "viridis",
        "bias_cmap": "RdBu_r",
        "vmin_abs": 5400.0,
        "vmax_abs": 5900.0,  
        "vmax_bias": 100.0,  
        "vmax_spread": 50.0  
    }
}

# ----------------------------------------------------
# MAIN PROCESSING ENGINE
# ----------------------------------------------------
from arraylake import Client
print("Initializing network data layer connections...")
client = Client()
repo_spire = client.get_repo("artamta/s2s-research")

for var_key, cfg in VALIDATION_MATRIX.items():
    print(f"\n####################################################")
    print(f" STARTING VALIDATION PIPELINE FOR FIELD: {var_key}")
    print(f"####################################################")

    for lead_day in TARGET_LEAD_DAYS:
        print(f"\n--- Lead Day +{lead_day} ---")
        
        # ------------------------------------------------
        # PHASE A: ISOLATED SPIRE EXTRACTION
        # ------------------------------------------------
        print("Executing Phase A: Extracting Spire Absolute Forecast & Uncertainty...")
        session_spire = repo_spire.readonly_session("main")
        ds_spire = xr.open_zarr(session_spire.store, group="mean_stddev")
        
        if cfg["spire_var"] not in ds_spire:
            print(f"Variable '{cfg['spire_var']}' missing from Spire store dataset group. Skipping {var_key}.")
            del ds_spire
            gc.collect()
            break
            
        spire_lat = 'latitude' if 'latitude' in ds_spire.coords else 'lat'
        spire_lon = 'longitude' if 'longitude' in ds_spire.coords else 'lon'
        
        lat_slice_spire = slice(lat_max, lat_min) if ds_spire[spire_lat][0] > ds_spire[spire_lat][-1] else slice(lat_min, lat_max)
        lon_slice_spire = slice(lon_min, lon_max)
        
        init_time = ds_spire['reference_time'].isel(reference_time=init_idx).values
        init_date_str = str(init_time)[:10]
        target_step = np.timedelta64(lead_day, 'D')
        
        spire_raw = ds_spire[cfg["spire_var"]].sel({spire_lat: lat_slice_spire, spire_lon: lon_slice_spire})
        spire_day = spire_raw.isel(reference_time=init_idx).sel(step=target_step, method='nearest').compute()
        
        spread_raw = ds_spire[cfg["spire_std"]].sel({spire_lat: lat_slice_spire, spire_lon: lon_slice_spire})
        spread_day = spread_raw.isel(reference_time=init_idx).sel(step=target_step, method='nearest').compute()
        
        if cfg["conversion"] == "kelvin_to_celsius":
            fcst_absolute_values = spire_day.values - 273.15
        else:
            fcst_absolute_values = spire_day.values
            
        spread_absolute_values = spread_day.values
        grid_lat = spire_day[spire_lat].values
        grid_lon = spire_day[spire_lon].values
        
        valid_date = init_time + target_step
        valid_date_str = str(valid_date)[:10]
        
        del ds_spire, spire_raw, spire_day, spread_raw, spread_day
        gc.collect()
        
        # ------------------------------------------------
        # PHASE B: ISOLATED ARCO-ERA5 EXTRACTION
        # ------------------------------------------------
        print(f"Executing Phase B: Streaming independent ARCO-ERA5 validation for {valid_date_str}...")
        ds_era5 = xr.open_zarr(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            storage_options={"token": "anon"}
        )
        
        lat_slice_era5 = slice(lat_max, lat_min) if ds_era5['latitude'][0] > ds_era5['latitude'][-1] else slice(lat_min, lat_max)
        lon_slice_era5 = slice(lon_min, lon_max)
        
        if cfg["era5_var"] == "geopotential" and "level" in ds_era5[cfg["era5_var"]].coords:
            era5_raw = ds_era5[cfg["era5_var"]].sel(latitude=lat_slice_era5, longitude=lon_slice_era5, level=500)
        else:
            era5_raw = ds_era5[cfg["era5_var"]].sel(latitude=lat_slice_era5, longitude=lon_slice_era5)
            
        era5_day_hourly = era5_raw.sel(time=slice(f"{valid_date_str}T00:00:00", f"{valid_date_str}T23:00:00")).compute()
        
        if era5_day_hourly.size == 0:
            raise ValueError(f"Data stream gap encountered: ARCO-ERA5 returned empty coordinates for {valid_date_str}!")
            
        if cfg["conversion"] == "kelvin_to_celsius":
            era5_day_processed = era5_day_hourly.mean(dim='time') - 273.15
        elif cfg["conversion"] == "era5_meters_to_mm_daily_sum":
            era5_day_processed = era5_day_hourly.sum(dim='time') * 1000.0
        elif cfg["conversion"] == "era5_w_to_gpm_500hPa":
            era5_day_processed = era5_day_hourly.mean(dim='time') / 9.80665
        else:
            era5_day_processed = era5_day_hourly.mean(dim='time')
            
        era5_aligned_da = era5_day_processed.interp(
            latitude=grid_lat,
            longitude=grid_lon,
            method="linear"
        )
        era5_absolute_values = era5_aligned_da.values
        
        del ds_era5, era5_raw, era5_day_hourly, era5_day_processed, era5_aligned_da
        gc.collect()
        
        # ------------------------------------------------
        # PHASE C: HARD MATHEMATICAL VERIFICATION CHECK
        # ------------------------------------------------
        print("Executing Phase C: Auditing matrix separation guidelines...")
        data_similarity_check = np.max(np.abs(fcst_absolute_values - era5_absolute_values))
        if data_similarity_check < 1e-4:
            raise RuntimeError(
                f"CRITICAL BUG DETECTED: Spire Forecast and ERA5 Arrays are completely identical for {var_key}! "
                "Data separation rules violated. Halting script execution."
            )
            
        absolute_bias_field = fcst_absolute_values - era5_absolute_values
        
        # ------------------------------------------------
        # PHASE D: GENERATE 4-PANEL VERIFICATION PLOT
        # ------------------------------------------------
        print("Executing Phase D: Finalizing figure formatting layouts...")
        
        fig, axes = plt.subplots(1, 4, figsize=(26, 7.5), dpi=300, 
                                 subplot_kw={'projection': ccrs.PlateCarree()},
                                 gridspec_kw={'wspace': 0.15})
        
        ax1, ax2, ax3, ax4 = axes
        
        v_min_abs = cfg["vmin_abs"]
        v_max_abs = cfg["vmax_abs"]
        v_max_bias = cfg["vmax_bias"]
        v_max_spread = cfg["vmax_spread"]

        def render_panel(ax, title, data_array, cmap, vmin, vmax, label_str, is_first_panel=False):
            current_cmap = plt.get_cmap(cmap).copy()
            
            # FIXED: Set a smooth, deep dark-midnight tone for anything that drops below extreme ranges 
            # instead of solid clipped black blocks
            current_cmap.set_under('#1d1d2d') 
            
            # Draw underlying meteorological fields
            im = ax.pcolormesh(grid_lon, grid_lat, data_array, 
                               cmap=current_cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), shading='auto')
            
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            # Explicit high-resolution coastline rendering over ocean background
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='#1a252f')
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gainsboro', linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
            if not is_first_panel:
                gl.left_labels = False
                
            gl.xlabel_style = {'size': 8.5}
            gl.ylabel_style = {'size': 8.5}
            
            ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
            
            cb = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.048, pad=0.03, shrink=0.92, extend='both' if vmin < 0 else 'max')
            cb.set_label(label_str, fontsize=9.5, fontweight='bold')
            cb.ax.tick_params(labelsize=8)

        # Plot matching panels safely mapped to structural layout settings
        render_panel(ax1, f"(a) Spire Forecast Mean\nLead Horizon: Day +{lead_day}", fcst_absolute_values, cfg["abs_cmap"], v_min_abs, v_max_abs, cfg["units"], is_first_panel=True)
        render_panel(ax2, f"(b) ERA5 Reanalysis Obs\nValid Date: {valid_date_str}", era5_absolute_values, cfg["abs_cmap"], v_min_abs, v_max_abs, cfg["units"], is_first_panel=False)
        render_panel(ax3, f"(c) Forecast Systematic Bias\n(Spire - ERA5)", absolute_bias_field, cfg["bias_cmap"], -v_max_bias, v_max_bias, cfg["units"], is_first_panel=False)
        render_panel(ax4, f"(d) Forecast Ensemble Spread\n(Uncertainty $\sigma$)", spread_absolute_values, "YlOrRd", 0, v_max_spread, cfg["units"], is_first_panel=False)
        
        ax3.title.set_color('crimson')
        ax4.title.set_color('darkorange')

        fig.suptitle(f"Subseasonal Spatial Calibration Analysis: {var_key} Configuration Profile\n"
                     f"Model Initialization: {init_date_str} | Verification Target: {valid_date_str} (Lead Horizon: +{lead_day}d)", 
                     fontsize=14, fontweight='bold', y=1.04)
        
        output_filename = f"{OUTPUT_DIR}/spire_calibrated_{var_key}_day_{lead_day}.png"
        
        fig.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f" -> Output verified diagnostic sheet saved: {output_filename}")

print("\nSuccess! Multi-variable interlocked calibration pipeline has compiled all figures successfully.")