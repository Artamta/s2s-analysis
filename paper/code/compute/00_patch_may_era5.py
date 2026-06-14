"""
Download missing May 11-14 2026 for era5_daily_tp.nc
"""
import warnings; warnings.filterwarnings('ignore')
import xarray as xr, pandas as pd, os

FILE = '/storage/raj.ayush/s2s-forecast-data/era5/daily/era5_daily_tp.nc'
print(f"Opening existing file {FILE}...")
old = xr.open_dataset(FILE)
print(f"Old dates: {old.time.values[0].astype('datetime64[D]')} to {old.time.values[-1].astype('datetime64[D]')}")

if pd.to_datetime(old.time.values[-1]) >= pd.to_datetime('2026-05-14'):
    print("Already goes up to May 14. Exiting.")
    exit(0)

print("Opening ARCO-ERA5 (anon)...", flush=True)
ds = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                  consolidated=True, storage_options={"token": "anon"})

DATES = pd.date_range('2026-05-11', '2026-05-14', freq='D')

# ARCO uses lat, lon. The existing file probably uses latitude, longitude
lat_name = 'latitude' if 'latitude' in old.dims else 'lat'
lon_name = 'longitude' if 'longitude' in old.dims else 'lon'

LAT = slice(float(old[lat_name].max()), float(old[lat_name].min()))  # ARCO descending
LON = slice(float(old[lon_name].min()), float(old[lon_name].max()))

print(f"Downloading TP for {DATES[0].strftime('%Y-%m-%d')} to {DATES[-1].strftime('%Y-%m-%d')}...", flush=True)
tp_raw = ds['total_precipitation'].sel(latitude=LAT, longitude=LON)

days = []
for d in DATES:
    day_str = d.strftime('%Y-%m-%d')
    val = tp_raw.sel(time=day_str).sum('time') * 1000.0  # m -> mm/day
    days.append(val.load())
    print(f"  {day_str}: mean={float(val.mean()):.4f}", flush=True)

new_tp = xr.concat(days, dim='time').assign_coords(time=DATES)
new_tp = new_tp.rename({'latitude': lat_name, 'longitude': lon_name})
new_ds = xr.Dataset({'tp': new_tp})

print("Concatenating...", flush=True)
merged = xr.concat([old, new_ds], dim='time')
old.close()

TMP = FILE + '.tmp'
merged.to_netcdf(TMP)
os.rename(TMP, FILE)

print(f"✓ SUCCESSFULLY UPDATED {FILE}")
print(f"New dates: {merged.time.values[0].astype('datetime64[D]')} to {merged.time.values[-1].astype('datetime64[D]')}")
