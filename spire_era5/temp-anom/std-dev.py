import os
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Directories
OUTPUT_DIR = "paper_figures_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Dual-Group Cloud Connection
print("Connecting to Arraylake Repository...")
try:
    from arraylake import Client
    client = Client()
    repo = client.get_repo("artamta/s2s-research")
    session = repo.readonly_session("main")
    
    # We pull BOTH groups to combine the ensemble mean anomaly and the spread
    ds_anom = xr.open_zarr(session.store, group="anomalies")
    ds_std  = xr.open_zarr(session.store, group="mean_stddev")
except Exception as e:
    print(f"Connection failed: {e}", file=sys.stderr)
    sys.exit(1)

# 3. Domain Slicing (India Standard Bounding Box)
lat_name = 'latitude' if 'latitude' in ds_anom.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds_anom.coords else 'lon'
lat_slice = slice(40.0, 5.0) if ds_anom[lat_name][0] > ds_anom[lat_name][-1] else slice(5.0, 40.0)
lon_slice = slice(60.0, 100.0)

# Slice both datasets identically
da_anom = ds_anom['air_temperature_max'].sel({lat_name: lat_slice, lon_name: lon_slice})
da_spread = ds_std['air_temperature_max_stddev'].sel({lat_name: lat_slice, lon_name: lon_slice})

# 4. Extract Trajectories & Uncertainty Plumes
# Target: February 10, 2026 (Index 40) - the strong anomaly run
init_idx = 40
init_date = str(da_anom['reference_time'].isel(reference_time=init_idx).values)[:10]

print(f"Computing spatial means and uncertainty bounds for initialization: {init_date}")
mean_trajectory = da_anom.isel(reference_time=init_idx).mean(dim=[lat_name, lon_name]).compute()
spread_trajectory = da_spread.isel(reference_time=init_idx).mean(dim=[lat_name, lon_name]).compute()

lead_days = da_anom['step'].values / np.timedelta64(1, 'D')

# 5. Plotting a Professional Forecast Uncertainty Plume
print("Generating publication figure...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Calculate 1-sigma and 2-sigma spread boundaries
upper_1sig = mean_trajectory + spread_trajectory
lower_1sig = mean_trajectory - spread_trajectory
upper_2sig = mean_trajectory + 1.96 * spread_trajectory
lower_2sig = mean_trajectory - 1.96 * spread_trajectory

# Plot Uncertainty Bands (Plumes)
ax.fill_between(lead_days, lower_2sig, upper_2sig, color='#d35400', alpha=0.15, label='95% Ensemble Confidence Interval')
ax.fill_between(lead_days, lower_1sig, upper_1sig, color='#d35400', alpha=0.30, label='68% Ensemble Confidence Interval')

# Plot the Ensemble Mean Trajectory
ax.plot(lead_days, mean_trajectory, color='#d35400', linewidth=2.5, marker='o', markersize=4, label='Ensemble Mean Forecast')

# Baseline reference
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6)

# Formatting Layout
ax.set_title(f"Spire S2S Forecast Plume over India\nInitialized: {init_date}", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Forecast Lead Time (Days)", fontsize=11, fontweight='bold')
ax.set_ylabel("Temperature Anomaly (°C deviation)", fontsize=11, fontweight='bold')
ax.set_xlim(1, 46)
ax.grid(color='gainsboro', linestyle=':', linewidth=0.5, alpha=0.7)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=10)

# Clean up axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

output_name = f"{OUTPUT_DIR}/figure2_spire_uncertainty_plume.png"
plt.savefig(output_name, bbox_inches='tight', dpi=300)
plt.close()

print(f"Success! Upgraded validation plume saved to: {output_name}")