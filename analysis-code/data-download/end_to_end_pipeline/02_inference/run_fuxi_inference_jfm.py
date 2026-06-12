import os
import glob
import subprocess
import pandas as pd
import xarray as xr
import sys

# Append FuXi-S2S dir to path so we can import data_util
FUXI_DIR = "/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)/fuxi_s2s/FuXi-S2S"
sys.path.append(FUXI_DIR)
from data_util import make_input

def main():
    data_base_dir = "/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)/fuxi_era5_initializations_1.5deg"
    model_path = os.path.join(FUXI_DIR, "model", "fuxi_s2s.onnx")
    mask_path = os.path.join(FUXI_DIR, "data", "mask.nc")
    output_base_dir = "/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)/fuxi_s2s_predictions"
    
    # Get all init dates from the downloaded folders
    init_dirs = sorted(glob.glob(os.path.join(data_base_dir, "2026*")))
    print(f"Found {len(init_dirs)} initialization dates to process.")
    
    for init_dir in init_dirs:
        date_str = os.path.basename(init_dir)
        print(f"\n=============================================")
        print(f"Processing Init Date: {date_str}")
        print(f"=============================================")
        
        input_nc_path = os.path.join(init_dir, "input.nc")
        
        # Step 1: Compile the input tensor if not already done
        if not os.path.exists(input_nc_path):
            print("Compiling input.nc tensor from downloaded ERA5 variables...")
            try:
                input_tensor = make_input(init_dir)
                input_tensor.to_netcdf(input_nc_path)
                print(f"Saved compiled tensor to {input_nc_path}")
            except Exception as e:
                print(f"Failed to compile input tensor for {date_str}: {e}")
                continue
        else:
            print(f"Tensor {input_nc_path} already exists.")
            
        # Step 2: Run inference for this date
        out_dir = os.path.join(output_base_dir, date_str)
        os.makedirs(out_dir, exist_ok=True)
        
        # Run the inference script via subprocess
        inference_cmd = [
            "python", os.path.join(FUXI_DIR, "inference.py"),
            "--model", model_path,
            "--input", input_nc_path,
            "--device", "cuda",
            "--total_step", "42",
            "--total_member", "1",  # Run control forecast first for speed
            "--save_dir", out_dir
        ]
        
        print(f"Running inference: {' '.join(inference_cmd)}")
        # We need to run this in the fuxi_s2s environment, but we will run this master script 
        # inside the fuxi_s2s environment so it can just use 'python'
        try:
            subprocess.run(inference_cmd, check=True)
            print(f"Inference for {date_str} completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Inference failed for {date_str} with error: {e}")

if __name__ == "__main__":
    main()
