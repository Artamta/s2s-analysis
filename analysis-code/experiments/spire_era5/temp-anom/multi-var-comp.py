import os
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Cluster friendly headless mode
import matplotlib.pyplot as plt

# 1. Setup Output Directory
OUTPUT_DIR = "paper_figures_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Connect via Arraylake Session
print("Connecting to Arraylake [anomalies & mean_stddev groups]...")
try:
    from arraylake import Client
    client = Client()
    repo = client.get_repo("artamta/s2s-research")
    session = repo.readonly_session("main")
    
    ds_anom = xr.open_zarr(session.store, group="anomalies")
    ds_std  = xr.open_zarr(session.store, group="mean_stddev")
except Exception as e:
    print(f"Arraylake connection failed: {e}", file=sys.stderr)
    sys.exit(1)

# 3. Spatial Slicing for Exact India Bounding Box
lat_name = 'latitude' if 'latitude' in ds_anom.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds_anom.coords else 'lon'

if ds_anom[lat_name][0] > ds_anom[lat_name][-1]:
    lat_slice = slice(40.0, 5.0)  # North to South
else:
    lat_slice = slice(5.0, 40.0)  # South to North
lon_slice = slice(60.0, 100.0)

# Target strong initialization date: Feb 10, 2026 (index 40)
init_idx = 40
init_date = str(ds_anom['reference_time'].isel(reference_time=init_idx).values)[:10]
lead_days = ds_anom['step'].values / np.timedelta64(1, 'D')

# 4. Verified Variable Mapping from Manifest Audit
TARGET_FIELDS = {
    "T2M_Max": {
        "anom_var": "air_temperature_max",
        "std_var": "air_temperature_max_stddev",
        "has_level": False,
        "level_val": None,
        "title": "2m Maximum Temperature Anomaly",
        "units": "°C"
    },
    "T850": {
        "anom_var": "temperature",
        "std_var": "air_temperature_at_isobaric_levels_stddev",
        "has_level": True,
        "level_val": 850,
        "title": "850 hPa Temperature Anomaly",
        "units": "°C"
    },
    "Z500": {
        "anom_var": "geopotential_height",
        "std_var": "geopotential_height_at_isobaric_levels_stddev",
        "has_level": True,
        "level_val": 500,
        "title": "500 hPa Geopotential Height Anomaly",
        "units": "m"
    }
}

# 5. Execution Pipeline Loop
for field_key, config in TARGET_FIELDS.items():
    print(f"\nProcessing publication structures for: {field_key}")
    
    a_var = config["anom_var"]
    s_var = config["std_var"]
    
    if a_var not in ds_anom:
        print(f"Variable {a_var} missing from anomalies group. Skipping.")
        continue

    # Spatial subsets
    da_anom_full = ds_anom[a_var].sel({lat_name: lat_slice, lon_name: lon_slice})
    da_std_full = ds_std[s_var].sel({lat_name: lat_slice, lon_name: lon_slice})
    
    # Pressure level subset selection using nearest-match security
    if config["has_level"]:
        try:
            da_anom_slice = da_anom_full.sel(isobar=config["level_val"], method='nearest')
            da_std_slice = da_std_full.sel(isobar=config["level_val"], method='nearest')
            actual_level = int(da_anom_slice.isobar.values)
            print(f" -> Found and targeted pressure index level: {actual_level} hPa")
        except Exception as e:
            print(f"Could not slice level {config['level_val']} for {a_var}: {e}. Skipping.")
            continue
    else:
        da_anom_slice = da_anom_full
        da_std_slice = da_std_full

    # Core Calculations
    mean_series = da_anom_slice.isel(reference_time=init_idx).mean(dim=[lat_name, lon_name]).compute()
    spread_series = da_std_slice.isel(reference_time=init_idx).mean(dim=[lat_name, lon_name]).compute()
    
    hist_data = da_anom_slice.isel(reference_time=init_idx).compute().values.flatten()
    hist_data = hist_data[~np.isnan(hist_data)]

    # ==========================================================================
    # IMAGE 1: DISTINCT HISTOGRAM DISTRIBUTION
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax1.hist(hist_data, bins=50, color='#34495e', edgecolor='#2c3e50', alpha=0.75, rwidth=0.85)
    ax1.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Climate Baseline (0°C Anomaly)')
    
    ax1.set_title(f"Regional Grid Point Distribution: {config['title']}\nInitialized: {init_date} | India Domain Box", fontsize=11, fontweight='bold', pad=12)
    ax1.set_xlabel(f"Anomaly Value ({config['units']})", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Frequency (Grid Points)", fontsize=10, fontweight='bold')
    ax1.grid(color='gainsboro', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.legend(frameon=True, facecolor='white', edgecolor='none')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    hist_output = f"{OUTPUT_DIR}/spire_distribution_{field_key}.png"
    plt.savefig(hist_output, bbox_inches='tight', dpi=300)
    plt.close()
    print(f" -> Saved Image 1 (Distribution): {hist_output}")

    # ==========================================================================
    # IMAGE 2: DISTINCT FORECAST UNCERTAINTY PLUME
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=300)
    upper_95 = mean_series + 1.96 * spread_series
    lower_95 = mean_series - 1.96 * spread_series
    upper_68 = mean_series + spread_series
    lower_68 = mean_series - spread_series

    ax2.fill_between(lead_days, lower_95, upper_95, color='#e67e22' if 'T2M' in field_key else '#2980b9', alpha=0.15, label='95% Ensemble Confidence Interval')
    ax2.fill_between(lead_days, lower_68, upper_68, color='#e67e22' if 'T2M' in field_key else '#2980b9', alpha=0.32, label='68% Ensemble Confidence Interval')
    ax2.plot(lead_days, mean_series, color='#d35400' if 'T2M' in field_key else '#1f3a60', linewidth=2.5, marker='o', markersize=3.5, label='Ensemble Mean Forecast')
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.set_title(f"46-Day Regional Mean Trajectory & Spread Envelope: {config['title']}\nInitialized: {init_date} | India Domain Box", fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel("Forecast Lead Time (Days)", fontsize=10, fontweight='bold')
    ax2.set_ylabel(f"Domain-Average Anomaly ({config['units']})", fontsize=10, fontweight='bold')
    ax2.set_xlim(1, 46)
    ax2.grid(color='gainsboro', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plume_output = f"{OUTPUT_DIR}/spire_plume_{field_key}.png"
    plt.savefig(plume_output, bbox_inches='tight', dpi=300)
    plt.close()
    print(f" -> Saved Image 2 (Plume Envelope): {plume_output}")

print("\nSuccess! Review separate standalone PNG files within your paper_figures_2026/ folder.")