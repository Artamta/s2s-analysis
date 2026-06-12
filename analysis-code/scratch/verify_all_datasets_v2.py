import xarray as xr
import glob
import numpy as np

print("=== ERA5 ===")
# Check GRIB files
try:
    sfc = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
    lat_res = abs(sfc.latitude.values[0] - sfc.latitude.values[1])
    print(f"era5_surface.grib (tp): shape {sfc['tp'].shape}, resolution {lat_res:.2f}°")
    
    sfc_t2m = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface.grib', engine='cfgrib', filter_by_keys={'shortName': 't2m'})
    print(f"era5_surface.grib (t2m): shape {sfc_t2m['t2m'].shape}")
except Exception as e:
    print(f"ERA5 Grib err: {e}")

# Check NC files
try:
    nc_files = glob.glob('/storage/raj.ayush/s2s-forecast-data/era5/data/*.nc')
    if nc_files:
        ds = xr.open_dataset(nc_files[0])
        lat_res = abs(ds.latitude.values[0] - ds.latitude.values[1])
        print(f"era5_surface_z500.nc: vars {list(ds.data_vars)}, shape {ds['t2m'].shape}, resolution {lat_res:.2f}°")
except Exception as e:
    print(f"ERA5 NC err: {e}")

print("\n=== FUXI ===")
try:
    fx_files = sorted(glob.glob('/storage/raj.ayush/s2s-forecast-data/fuxi/output/20260101/member/00/*.nc'))
    if fx_files:
        ds = xr.open_dataset(fx_files[0])
        lat_res = abs(ds.lat.values[0] - ds.lat.values[1])
        print(f"FuXi 00/01.nc: {len(list(ds.channel.values))} vars, shape {ds['__xarray_dataarray_variable__'].shape}, resolution {lat_res:.2f}°")
        
    members = glob.glob('/storage/raj.ayush/s2s-forecast-data/fuxi/output/20260101/member/*')
    print(f"FuXi Members count: {len(members)}")
except Exception as e:
    print(f"FuXi err: {e}")

print("\n=== SPIRE ===")
try:
    spire = xr.open_zarr('/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr', group='mean_stddev')
    lat_res = abs(spire.latitude.values[0] - spire.latitude.values[1])
    print(f"Spire Zarr: vars {list(spire.data_vars)}, shape {spire['air_temperature'].shape}, resolution {lat_res:.2f}°")
    print(f"Spire Steps: {len(spire.step)}")
except Exception as e:
    print(f"Spire err: {e}")

print("\n=== ECMWF ===")
try:
    cf_tp = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_new_cf_20260101.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
    cf_mx = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_new_cf_20260101.grib', engine='cfgrib', filter_by_keys={'shortName': 'mx2t6'})
    lat_res = abs(cf_tp.latitude.values[0] - cf_tp.latitude.values[1])
    print(f"ECMWF cf tp shape: {cf_tp['tp'].shape}, resolution {lat_res:.2f}°")
    print(f"ECMWF cf mx2t6 shape: {cf_mx['mx2t6'].shape}")
    
    pf = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_new_pf_20260101.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
    print(f"ECMWF pf members: {len(pf.number)}")
except Exception as e:
    print(f"ECMWF err: {e}")

print("\n=== NCEP ===")
try:
    cf_nc = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ncep/data/sfc_cf_20260101.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
    lat_res = abs(cf_nc.latitude.values[0] - cf_nc.latitude.values[1])
    print(f"NCEP cf tp shape: {cf_nc['tp'].shape}, resolution {lat_res:.2f}°")
    
    pf_nc = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ncep/data/sfc_pf_20260101.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
    print(f"NCEP pf members: {len(pf_nc.number)}")
except Exception as e:
    print(f"NCEP err: {e}")
