import xarray as xr
import os

input_file = '/storage/raj.ayush/fuxi-init-jfm-weekely/data/20260101/input.nc'
if os.path.exists(input_file):
    print("Fixing dimensions for existing input.nc...")
    ds = xr.open_dataarray(input_file)
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        ds.to_netcdf('temp.nc')
        import shutil
        shutil.move('temp.nc', input_file)
        print("Fixed!")
    else:
        print("Already fixed.")
