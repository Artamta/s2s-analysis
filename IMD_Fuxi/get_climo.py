import os
import xarray as xr
import numpy as np

# ----------------------------------------------------------------------
# 1. PARAMETERS & GEOGRAPHIC FRAME CONFIGURATION
# ----------------------------------------------------------------------
START_YEAR = 1981
END_YEAR = 2020  # Complete 40-Year Climate Reference Baseline Epoch
OUTPUT_DIR = "./era5_master_climatology"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tight India regional framing coordinates (Keeps memory footprints light)
lats = slice(38.5, 5.0)   # North-to-South ordering
lons = slice(65.0, 98.5)  # West-to-East ordering

print("🔗 Handshaking with remote Google Cloud ARCO-ERA5 dataset cluster...")
ds_global = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"token": "anon"}
)

print(f"✂️ Slicing spatial and temporal domains: {START_YEAR} to {END_YEAR}...")
# Immediate sub-selection prevents global chunks from entering local RAM cache
ds_india = ds_global.sel(
    time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"),
    latitude=lats,
    longitude=lons
)

# ----------------------------------------------------------------------
# 2. SEPARATE LAZY COMPRESSION STEP ENGINES
# ----------------------------------------------------------------------
print("\n🔄 Step 1: Converting hourly streams to daily summaries (Lazy Processing)...")

# A. Precipitation Transformation (Hourly Meters -> Daily Accumulation mm)
print("   ↳ Aggregating cumulative daily precipitation...")
tp_daily = ds_india["total_precipitation"].resample(time="1D").sum(dim="time") * 1000.0

# B. Thermal Extremes Transformation (Hourly Kelvin -> Daily Celsius Max/Min)
print("   ↳ Isolating diurnal Tmax and Tmin thresholds...")
t2m_c = ds_india["2m_temperature"] - 273.15
tmax_daily = t2m_c.resample(time="1D").max(dim="time")
tmin_daily = t2m_c.resample(time="1D").min(dim="time")

# C. Pressure Transformation (Hourly Pascals -> Daily Mean hPa)
print("   ↳ Computing mean sea level pressure cycles...")
mslp_daily = (ds_india["mean_sea_level_pressure"].resample(time="1D").mean(dim="time")) / 100.0

# D. Surface Wind Vector Components Transformation (Hourly m/s -> Daily Mean Magnitude)
print("   ↳ Resolving 10m wind speed scalar aggregates...")
u10_d = ds_india["10m_u_component_of_wind"].resample(time="1D").mean(dim="time")
v10_d = ds_india["10m_v_component_of_wind"].resample(time="1D").mean(dim="time")
wind_speed_daily = np.sqrt(u10_d**2 + v10_d**2)

# ----------------------------------------------------------------------
# 3. INTERLOCKED DAY-OF-YEAR VECTOR GROUPING
# ----------------------------------------------------------------------
print("\n📊 Step 2: Compiling multi-decadal day-of-year matrix (1 to 366)...")

# Grouping across the 40-year timeline by Julian calendar days
tp_climo    = tp_daily.groupby("time.dayofyear").mean(dim="time")
tmax_climo  = tmax_daily.groupby("time.dayofyear").mean(dim="time")
tmin_climo  = tmin_daily.groupby("time.dayofyear").mean(dim="time")
mslp_climo  = mslp_daily.groupby("time.dayofyear").mean(dim="time")
wind_climo  = wind_speed_daily.groupby("time.dayofyear").mean(dim="time")

# ----------------------------------------------------------------------
# 4. UNIFIED CONSOLIDATION & NETCDF FLUSH
# ----------------------------------------------------------------------
print("\n📦 Step 3: Packing arrays into a single dataset container...")
master_climatology_ds = xr.Dataset({
    "tp": tp_climo,
    "tmax": tmax_climo,
    "tmin": tmin_climo,
    "mslp": mslp_climo,
    "wind_speed": wind_climo
})

# Assign explicit metadata to make the file self-documenting
master_climatology_ds.attrs["description"] = f"ERA5 Regional Day-of-Year Climatology Baseline for India Grid Boundary"
master_climatology_ds.attrs["reference_period"] = f"{START_YEAR}-{END_YEAR}"
master_climatology_ds.attrs["spatial_resolution"] = "0.25 deg x 0.25 deg"

output_filepath = os.path.join(OUTPUT_DIR, f"era5_india_master_climatology_{START_YEAR}_{END_YEAR}.nc")

print(f"💾 Step 4: Writing NetCDF file to disk: {output_filepath}")
print("          (This executes all lazy steps in parallel—please stand by...)")

# Trigger the actual cloud stream and save computation local
master_climatology_ds.to_netcdf(output_filepath, format="NETCDF4")

print(f"\n🎉 Success! Your year-round master climatology file is saved and ready at:\n   {output_filepath}")