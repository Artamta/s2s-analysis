import os
import cdsapi

# Safely store the reanalysis data on the large storage drive
OUT_DIR = '/storage/raj.ayush/s2s-forecast-data/era5/data'
os.makedirs(OUT_DIR, exist_ok=True)

c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key="f628388c-5c81-44ae-a403-266655286ed0")

# Exact same domain and resolution as your S2S benchmark downloads!
# This means NO interpolation is needed later when doing RMSE/ACC math!
area = [50, 55, 0, 105] # N, W, S, E
grid = [1.5, 1.5]

# Covering Jan 1 to May 15 (a few extra padding days to fully cover the March 26 + 46 day forecast)
dates = "2026-01-01/2026-05-15"

print(f"Requesting ERA5 Surface Variables for {dates}...")
try:
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                '2m_temperature',
                'total_precipitation',
                # Matching the NCEP/ECMWF mx2t6 and mn2t6
                'maximum_2m_temperature_since_previous_post_processing',
                'minimum_2m_temperature_since_previous_post_processing',
            ],
            'date': dates,
            'time': '00:00',
            'area': area,
            'grid': grid,
        },
        os.path.join(OUT_DIR, 'era5_surface.grib')
    )
    print("Surface download success!")
except Exception as e:
    print(f"Failed to download surface data: {e}")

print(f"Requesting ERA5 Pressure Variables (Z500) for {dates}...")
try:
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                'geopotential',
            ],
            'pressure_level': '500',
            'date': dates,
            'time': '00:00',
            'area': area,
            'grid': grid,
        },
        os.path.join(OUT_DIR, 'era5_pressure_500hpa.grib')
    )
    print("Pressure download success!")
except Exception as e:
    print(f"Failed to download pressure data: {e}")

print("\nAll ERA5 Ground truth downloads are complete!")
