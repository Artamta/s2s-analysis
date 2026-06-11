import os
import subprocess
import sys

# Add FuXi-S2S to path to import data_util
sys.path.append(os.path.abspath('FuXi-S2S'))
from data_util import make_input

dates = [
    '2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
    '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
    '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26'
]

base_storage = '/storage/raj.ayush/fuxi-init-jfm-weekely'
data_dir = os.path.join(base_storage, 'data')
os.makedirs(data_dir, exist_ok=True)

for date in dates:
    date_str = date.replace('-', '')
    date_data_dir = os.path.join(data_dir, date_str)
    input_file = os.path.join(date_data_dir, 'input.nc')
    
    print(f"\n================ Downloading {date} ================")
    
    if os.path.exists(input_file):
        print(f"input.nc for {date} already exists, skipping!")
        continue
        
    print("Downloading raw ERA5 inputs...")
    cmd = ['python', 'download_fuxi_input.py', '--date', date_str, '--outdir', data_dir]
    subprocess.run(cmd, check=True)
    
    print("Consolidating into input.nc and fixing dimensions...")
    ds = make_input(date_data_dir)
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds.to_netcdf(input_file)
    
    print("Cleaning up raw variables to save storage...")
    for f in os.listdir(date_data_dir):
        if f.endswith('.nc') and f != 'input.nc':
            os.remove(os.path.join(date_data_dir, f))
            
print("\nAll FuXi initial conditions downloaded and formatted successfully!")
