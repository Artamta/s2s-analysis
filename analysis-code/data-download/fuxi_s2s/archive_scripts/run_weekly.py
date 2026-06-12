import os
import subprocess
import shutil
import sys

# Add FuXi-S2S to path to import data_util
sys.path.append(os.path.abspath('FuXi-S2S'))
from data_util import make_input

dates = [
    '2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
    '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
    '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26'
]

base_storage = '/storage/raj.ayush/s2s-forecast-data/fuxi'
data_dir = os.path.join(base_storage, 'data')
output_dir = os.path.join(base_storage, 'output')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for date in dates:
    date_str = date.replace('-', '')
    date_data_dir = os.path.join(data_dir, date_str)
    date_output_dir = os.path.join(output_dir, date_str)
    
    print(f"\n================ Processing {date} ================")
    
    if os.path.exists(date_output_dir) and len(os.listdir(date_output_dir)) > 0:
        print(f"Output for {date} already exists, skipping...")
        continue
    
    input_file = os.path.join(date_data_dir, 'input.nc')
    
    # 1. Download data (only if input.nc doesn't exist)
    if not os.path.exists(input_file):
        print("Downloading input data...")
        cmd = ['python', 'download_fuxi_input.py', '--date', date_str, '--outdir', data_dir]
        subprocess.run(cmd, check=True)
        
        # 2. Make input.nc
        print("Creating input.nc...")
        ds = make_input(date_data_dir)
        if 'latitude' in ds.dims:
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        ds.to_netcdf(input_file)
        
        # 3. Clean up raw files to save space
        print("Cleaning up raw variables to save space...")
        for f in os.listdir(date_data_dir):
            if f.endswith('.nc') and f != 'input.nc':
                os.remove(os.path.join(date_data_dir, f))
    else:
        print("input.nc already exists, skipping download and cleanup!")
            
    # 4. Run inference
    print("Running inference...")
    date_output_dir = os.path.join(output_dir, date_str)
    os.makedirs(date_output_dir, exist_ok=True)
    
    cmd_infer = [
        '/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python', 'inference.py', 
        '--model', 'model/fuxi_s2s.onnx',
        '--input', input_file,
        '--device', 'cuda',
        '--save_dir', date_output_dir
    ]
    
    # Needs to be run from FuXi-S2S to find mask.nc properly
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/home/raj.ayush/.conda/envs/fuxi_s2s/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"] = "false"
    subprocess.run(cmd_infer, cwd='FuXi-S2S', check=True, env=env)
    
    # 5. Clean up input.nc if really strict on storage
    # print("Cleaning up input.nc...")
    # os.remove(input_file)
    # shutil.rmdir(date_data_dir)
    
print("All dates processed successfully!")
