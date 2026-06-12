import os
import sys
import numpy as np
import xarray as xr

# Attempt to import CuPy for NVIDIA A100 GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================
START_YEAR = "1991-01-01"
END_YEAR = "2020-12-31"
TARGET_MONTHS = [1, 2, 3]  # January, February, March

# Standard Publication Domain Boundaries for India and Neighbors
LAT_MIN, LAT_MAX = 5.0, 40.0
LON_MIN, LON_MAX = 60.0, 100.0

# Native forecast resolution grid configuration (0.5 degree)
TARGET_LAT = np.arange(LAT_MAX, LAT_MIN - 0.5, -0.5)  # 40.0 down to 5.0
TARGET_LON = np.arange(LON_MIN, LON_MAX + 0.5, 0.5)  # 60.0 up to 100.0

VARIABLES_TO_PROCESS = {
    # --- PRESSURE LEVEL VARIABLES ---
    'geopotential':            {'type': 'pressure', 'levels': [200, 500, 850, 1000]},
    'temperature':           {'type': 'pressure', 'levels': [200, 500, 850, 1000]},
    'u_component_of_wind':   {'type': 'pressure', 'levels': [200, 500, 850, 1000]},
    'v_component_of_wind':   {'type': 'pressure', 'levels': [200, 500, 850, 1000]},
    
    # --- SURFACE / SINGLE LEVEL VARIABLES ---
    '2m_temperature':          {'type': 'single',   'levels': [None]},
    'mean_sea_level_pressure': {'type': 'single',   'levels': [None]},
    'total_precipitation':     {'type': 'single',   'levels': [None]},
    '10m_u_component_of_wind': {'type': 'single',   'levels': [None]},
    '10m_v_component_of_wind': {'type': 'single',   'levels': [None]}
}

OUTPUT_DIR = "/home/raj.ayush/s2s/s2s_anlysis/spire_era5/temp-anom/climatology_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def run_regional_climatology_pipeline():
    """
    Connects to the ARCO-ERA5 Zarr-v3 cloud store, isolates the spatial domain
    to the Indian subcontinent, interpolates to a 0.5-degree grid, offloads array 
    reductions to the GPU if available, and exports regional baseline datasets.
    """
    if HAS_GPU:
        print("NVIDIA GPU acceleration activated via CuPy backend.")
    else:
        print("NVIDIA GPU backend unavailable. Defaulting to standard CPU processing.")

    print("Connecting to verified ARCO ERA5 cloud repository...")
    try:
        ds = xr.open_zarr(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            storage_options={"token": "anon"}
        )
    except Exception as e:
        print(f"Error establishing cloud connection: {e}", file=sys.stderr)
        sys.exit(1)

    for var_name, info in VARIABLES_TO_PROCESS.items():
        print(f"\nProcessing regional field: {var_name.upper()}")
        
        if var_name not in ds:
            print(f"Variable identifier '{var_name}' not found in datastore. Skipping.", file=sys.stderr)
            continue
            
        # 1. Broad temporal and structural slicing to save cloud pipeline memory
        data_field = ds[var_name].sel(time=slice(START_YEAR, END_YEAR))
        data_jfm = data_field.where(data_field.time.dt.month.isin(TARGET_MONTHS), drop=True)
        
        # 2. Extract regional boundaries directly from the cloud slice (+1 degree buffer for interpolation integrity)
        regional_ds = data_jfm.sel(
            latitude=slice(LAT_MAX + 1.0, LAT_MIN - 1.0),
            longitude=slice(LON_MIN - 1.0, LON_MAX + 1.0)
        )
        
        for lvl in info['levels']:
            if info['type'] == 'pressure':
                try:
                    grid_subset = regional_ds.sel(level=lvl)
                except Exception as e:
                    print(f"Level {lvl} hPa unavailable for {var_name}: {e}. Skipping.", file=sys.stderr)
                    continue
                filename = f"era5_JFM_climatology_{var_name}_{lvl}hPa_India_0.5deg.nc"
            else:
                grid_subset = regional_ds
                filename = f"era5_JFM_climatology_{var_name}_surface_India_0.5deg.nc"
                
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(output_path):
                print(f"Existing regional file located at {output_path}. Skipping.")
                continue

            print(f"Interpolating to India 0.5-deg domain and calculating daily baseline...")
            try:
                # Spatial bilinear interpolation down to target 0.5-degree resolution over India
                regridded_subset = grid_subset.interp(latitude=TARGET_LAT, longitude=TARGET_LON, method='linear')
                
                # If GPU acceleration is enabled, convert underlying data arrays to CuPy format before reduction
                if HAS_GPU:
                    regridded_subset.data = cp.asarray(regridded_subset.data)
                
                # Math: Group by day of year and compute long-term regional mean
                climatology = regridded_subset.groupby('time.dayofyear').mean('time').compute()
                
                # Write individual clean regional asset to disk
                climatology.to_netcdf(output_path)
                print(f"Successfully serialized regional data structure to: {output_path}")
            except Exception as e:
                print(f"Failure processing variable {var_name} at level {lvl}: {e}", file=sys.stderr)

    print("\nRegional pipeline execution complete. All India-domain files compiled at 0.5 degrees.")

if __name__ == "__main__":
    run_regional_climatology_pipeline()