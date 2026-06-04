import os
from datetime import datetime
from earth2studio.data import WB2Climatology

# Disable any unnecessary verbose parallel loading issues for the test
os.environ["MKL_NUM_THREADS"] = "1"

def test_online_climatology():
    print("🔗 Connecting to remote WeatherBench2 Climatology on Google Cloud...")
    
    # 1. Instantiate the WeatherBench 2 Climatology client (1990-2019 baseline)
    climo_source = WB2Climatology(
        climatology_zarr_store='1990-2019_6h_1440x721.zarr', # Full 0.25° Global Resolution
        cache=True,                                          # Saves a local copy of requested parts
        verbose=True
    )
    
    # 2. Set the target date matching your operational week start window
    # WeatherBench2 matches the Day of Year and Hour (00, 06, 12, 18 UTC) automatically.
    test_time = datetime(2021, 8, 26, 0, 0)
    
    # Earth2Studio Lexicon vocab: 't2m' = 2m Temperature, 'tp' = Total Precipitation
    test_variables = ["t2m", "tp"]
    
    print(f"📥 Querying global maps for date: {test_time.strftime('%B %d')} | Variables: {test_variables}...")
    
    # 3. Call the source to download/stream the filtered array chunk
    climo_data = climo_source(test_time, test_variables)
    
    # 4. Print Verification Diagnostics
    print("\n" + "="*60)
    print("🛰️  WEATHERBENCH2 CLIMATOLOGY API VERIFICATION")
    print("="*60)
    print(f"Return Object Type : {type(climo_data)}")
    print(f"Array Dimensions   : {climo_data.dims}")
    print(f"Array Shape        : {climo_data.shape}")
    
    print("\n📋 Coordinate Details:")
    for dim in climo_data.dims:
        print(f"  ↳ Dimension '{dim}': size {len(climo_data[dim])}")
        
    print("\n📊 Statistical Summary Check:")
    for var in test_variables:
        var_slice = climo_data.sel(variable=var)
        print(f"  ↳ [{var}] -> Global Mean: {float(var_slice.mean()):.4f} | Min: {float(var_slice.min()):.4f} | Max: {float(var_slice.max()):.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        test_online_climatology()
        print("🎉 Online check successful! Earth2Studio fetched global baselines cleanly.")
    except Exception as e:
        print(f"❌ Verification failed due to error: {e}")