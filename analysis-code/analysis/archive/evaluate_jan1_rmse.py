import os
import sys
import pandas as pd
import xarray as xr

# Add parent path so we can import our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loaders.model_loader import load_ecmwf, load_ncep, load_spire, regrid_to_common
from verification.metrics import (
    calculate_rmse, 
    get_india_mainland, 
    get_northwest_india, 
    get_central_india, 
    get_south_peninsular_india, 
    get_northeast_india
)

# Configuration
INIT_DATE = "20260101"
SPIRE_DATE = "2026-01-01"
ERA5_FILE = "/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface_z500_20260101.nc"
RESULTS_CSV = "../results/jan1_rmse_evaluation.csv"

def extract_temperature(ecmwf_ds, ncep_ds, spire_ds, era5_ds):
    """Safely extracts the Day 1 Temperature from all models."""
    print("Extracting Day 1 Temperature...")
    e_temp = ecmwf_ds['t2m']
    
    try:
        n_temp = (ncep_ds['mx2t6'].isel(step=0) + ncep_ds['mn2t6'].isel(step=0)) / 2.0
    except KeyError:
        n_temp = ncep_ds['mx2t6'].isel(step=0)
        
    s_temp = spire_ds['t2m'].isel(lead_time=0)
    
    # ERA5 2m Temperature is 't2m'. We take the first timestamp (Day 1).
    truth_temp = era5_ds['t2m'].isel(time=0)
    
    return s_temp, e_temp, n_temp, truth_temp

def main():
    print(f"--- S2S RMSE EVALUATION PIPELINE: {INIT_DATE} ---")
    
    # 1. Load Data
    print("Loading Models...")
    ecmwf_paths = load_ecmwf(INIT_DATE)
    ncep_paths = load_ncep(INIT_DATE)
    
    ecmwf_ds = xr.open_dataset(ecmwf_paths['control'][0], engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
    ncep_ds = xr.open_dataset(ncep_paths['control'][0], engine='cfgrib')
    spire_ds = load_spire(SPIRE_DATE)
    era5_ds = xr.open_dataset(ERA5_FILE)
    
    # 2. Extract Variable
    s_temp, e_temp, n_temp, truth_temp = extract_temperature(ecmwf_ds, ncep_ds, spire_ds, era5_ds)
    
    # 3. Regrid EVERYTHING to the mathematically fair 1.5-degree grid
    print("Regridding all models to unified 1.5° grid...")
    s_temp = regrid_to_common(s_temp)
    e_temp = regrid_to_common(e_temp)
    n_temp = regrid_to_common(n_temp)
    truth_temp = regrid_to_common(truth_temp)
    
    # 4. Regional Evaluation Maps
    regions = {
        "Mainland India": get_india_mainland,
        "Northwest India (NWI)": get_northwest_india,
        "Central India (CI)": get_central_india,
        "South Peninsular (SPI)": get_south_peninsular_india,
        "Northeast India (ENEI)": get_northeast_india
    }
    
    models = {
        "Spire": s_temp,
        "ECMWF": e_temp,
        "NCEP": n_temp
    }
    
    results = []
    
    # 5. Calculate RMSE Loop
    print("Calculating RMSE across all IMD Regions...")
    for region_name, crop_function in regions.items():
        for model_name, model_data in models.items():
            # Crop both the forecast and the truth to the specific Indian region
            cropped_forecast = crop_function(model_data)
            cropped_truth = crop_function(truth_temp)
            
            # Calculate RMSE
            rmse_val = calculate_rmse(cropped_forecast, cropped_truth)
            
            # Store Result (convert xarray scalar to float)
            results.append({
                "Region": region_name,
                "Model": model_name,
                "RMSE (Kelvin)": float(rmse_val.values)
            })
            
    # 6. Save to CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    
    print("\n--- EVALUATION COMPLETE ---")
    print(df.to_string(index=False))
    print(f"\nSaved beautiful tabular results to: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
