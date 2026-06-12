import xarray as xr
import glob

print("==========================================================")
print("             DATASET VERIFICATION SUMMARY                 ")
print("==========================================================\n")

# 1. ERA5 (Ground Truth)
print("1. ERA5 (Ground Truth)")
try:
    era_sfc = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface.grib', engine='cfgrib', backend_kwargs={'indexpath': ''})
    lat_res = abs(era_sfc.latitude.values[0] - era_sfc.latitude.values[1])
    lon_res = abs(era_sfc.longitude.values[0] - era_sfc.longitude.values[1])
    print(f"  - Location: /storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface.grib")
    print(f"  - Variables: {list(era_sfc.data_vars)}")
    print(f"  - Time range: {era_sfc.time.values[0]} to {era_sfc.time.values[-1]} ({len(era_sfc.time)} steps)")
    print(f"  - Resolution: {lat_res:.2f}° x {lon_res:.2f}°")
    print(f"  - Members: None (Reanalysis)")
except Exception as e:
    print(f"  - Error loading ERA5: {e}")

print("\n2. FuXi-S2S (AI Model)")
try:
    # Check one day
    fx_files = sorted(glob.glob('/storage/raj.ayush/s2s-forecast-data/fuxi/output/20260101/member/00/*.nc'))
    if fx_files:
        fx = xr.open_dataset(fx_files[-1])
        lat_res = abs(fx.lat.values[0] - fx.lat.values[1])
        lon_res = abs(fx.lon.values[0] - fx.lon.values[1])
        print(f"  - Location: /storage/raj.ayush/s2s-forecast-data/fuxi/output/[init_date]/member/00/[day].nc")
        print(f"  - Variables: {list(fx.channel.values)}")
        print(f"  - Lead Times: {len(fx_files)} continuous daily files (up to Day {len(fx_files)})")
        print(f"  - Resolution: {lat_res:.2f}° x {lon_res:.2f}°")
        print(f"  - Members: Deterministic (Member 00)")
except Exception as e:
    print(f"  - Error loading FuXi: {e}")

print("\n3. SPIRE (AI Model)")
try:
    spire = xr.open_zarr('/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr', group='mean_stddev')
    lat_res = abs(spire.latitude.values[0] - spire.latitude.values[1])
    lon_res = abs(spire.longitude.values[0] - spire.longitude.values[1])
    print(f"  - Location: /storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr")
    print(f"  - Variables: {list(spire.data_vars)}")
    print(f"  - Lead Times: {len(spire.step)} steps, up to {spire.step.values[-1]}")
    print(f"  - Resolution: {lat_res:.2f}° x {lon_res:.2f}°")
    print(f"  - Members: Ensemble mean/stddev stored directly in Zarr")
except Exception as e:
    print(f"  - Error loading Spire: {e}")

print("\n4. ECMWF (Operational)")
try:
    ec_cf = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_new_cf_20260101.grib', engine='cfgrib', backend_kwargs={'indexpath': ''})
    ec_pf = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_new_pf_20260101.grib', engine='cfgrib', backend_kwargs={'indexpath': ''})
    lat_res = abs(ec_cf.latitude.values[0] - ec_cf.latitude.values[1])
    lon_res = abs(ec_cf.longitude.values[0] - ec_cf.longitude.values[1])
    print(f"  - Location: /storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_new_[cf/pf]_[date].grib")
    print(f"  - Variables: {list(ec_cf.data_vars)}")
    print(f"  - Lead Times (CF): {len(ec_cf.step)} steps, {ec_cf.step.values[0]} to {ec_cf.step.values[-1]}")
    print(f"  - Resolution: {lat_res:.2f}° x {lon_res:.2f}°")
    print(f"  - Members: 1 Control (cf), {len(ec_pf.number)} Perturbed (pf)")
except Exception as e:
    print(f"  - Error loading ECMWF: {e}")

print("\n5. NCEP (Operational)")
try:
    nc_cf = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ncep/data/sfc_cf_20260101.grib', engine='cfgrib', backend_kwargs={'indexpath': ''})
    nc_pf = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/ncep/data/sfc_pf_20260101.grib', engine='cfgrib', backend_kwargs={'indexpath': ''})
    lat_res = abs(nc_cf.latitude.values[0] - nc_cf.latitude.values[1])
    lon_res = abs(nc_cf.longitude.values[0] - nc_cf.longitude.values[1])
    print(f"  - Location: /storage/raj.ayush/s2s-forecast-data/ncep/data/sfc_[cf/pf]_[date].grib")
    print(f"  - Variables: {list(nc_cf.data_vars)}")
    print(f"  - Lead Times (CF): {len(nc_cf.step)} steps, {nc_cf.step.values[0]} to {nc_cf.step.values[-1]}")
    print(f"  - Resolution: {lat_res:.2f}° x {lon_res:.2f}°")
    print(f"  - Members: 1 Control (cf), {len(nc_pf.number)} Perturbed (pf)")
except Exception as e:
    print(f"  - Error loading NCEP: {e}")

print("\n==========================================================")
