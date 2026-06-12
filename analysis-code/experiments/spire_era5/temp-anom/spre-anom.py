import os
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Safe for headless cluster compute nodes
import matplotlib.pyplot as plt

# 1. Setup Output Directory
OUTPUT_DIR = "paper_figures_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Connect via Arraylake Session (Yesterday's Working Context)
print("Connecting to Arraylake [anomalies group]...")
try:
    from arraylake import Client
    client = Client()
    # Reconnecting to your actual, verified data repository
    repo = client.get_repo("artamta/s2s-research")
    session = repo.readonly_session("main")
    
    # Open the store, ensuring we target the 'anomalies' group hierarchy
    ds_anom = xr.open_zarr(session.store, group="anomalies")
except Exception as e:
    print(f"Arraylake connection failed: {e}", file=sys.stderr)
    print("\n[TIP] If this times out, check if your cluster internet connection is fully active.", file=sys.stderr)
    sys.exit(1)

# 3. Spatial Slicing for India Domain (Strictly matching IMD standard boundaries)
lat_name = 'latitude' if 'latitude' in ds_anom.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds_anom.coords else 'lon'

# Safely handle coordinate ordering (North-to-South vs South-to-North)
if ds_anom[lat_name][0] > ds_anom[lat_name][-1]:
    lat_slice = slice(40.0, 5.0)  # North to South
else:
    lat_slice = slice(5.0, 40.0)  # South to North
lon_slice = slice(60.0, 100.0)

ds_india = ds_anom.sel({lat_name: lat_slice, lon_name: lon_slice})
t_anom = ds_india['air_temperature_max']

print("Calculating data summaries for regional profile...")
# Select 3 initialization periods: Early Jan, Mid-Feb, Late March
init_indices = [0, 40, 75]
init_dates = [str(ds_india['reference_time'].isel(reference_time=idx).values)[:10] for idx in init_indices]

spatial_means = []
for idx in init_indices:
    # Compute the average specifically over the India polygon bounding box
    mean_series = t_anom.isel(reference_time=idx).mean(dim=[lat_name, lon_name]).compute()
    spatial_means.append(mean_series)

# Convert lead time step coordinates to integer days
lead_days = ds_anom['step'].values / np.timedelta64(1, 'D')

# Fetch the distribution block for Feb 10th initialization
hist_data = t_anom.isel(reference_time=40).compute().values.flatten()
hist_data = hist_data[~np.isnan(hist_data)]  # Clean missing data points/ocean masking

# 4. Create the 1x2 Profiling Canvas
print("Generating high-resolution publication profile...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# --- PANEL 1: THE HISTOGRAM ---
ax1.hist(hist_data, bins=50, color='#3498db', edgecolor='#2980b9', alpha=0.75, rwidth=0.85)
ax1.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Climate Baseline (0°C Anomaly)')
ax1.set_title("(a) Distribution of Temperature Anomalies over India", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Anomaly Value (°K or °C deviation)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Frequency (Grid Points)", fontsize=10, fontweight='bold')
ax1.grid(color='gainsboro', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.legend(frameon=True, facecolor='white', edgecolor='none')

# --- PANEL 2: THE FORECAST TRAJECTORIES ---
colors = ['#27ae60', '#d35400', '#8e44ad']
for i, mean_series in enumerate(spatial_means):
    ax2.plot(lead_days, mean_series, color=colors[i], linewidth=2, marker='o', markersize=3,
             label=f"Init: {init_dates[i]}")

ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.set_title("(b) 46-Day Forecast Trajectories (India Average)", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Forecast Lead Time (Days)", fontsize=10, fontweight='bold')
ax2.set_ylabel("Domain-Average Anomaly", fontsize=10, fontweight='bold')
ax2.set_xlim(1, 46)
ax2.grid(color='gainsboro', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.legend(frameon=True, facecolor='white', edgecolor='none')

# Clean frame aesthetics
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle("Spire S2S Dataset Profile: Regional Domain Audit (India & Neighbors)", 
             fontsize=14, fontweight='bold', y=1.02)

output_name = f"{OUTPUT_DIR}/figure1_spire_india_profile.png"
plt.savefig(output_name, bbox_inches='tight', dpi=300)
plt.close()

print(f"Success! Verification profile saved directly to: {output_name}")