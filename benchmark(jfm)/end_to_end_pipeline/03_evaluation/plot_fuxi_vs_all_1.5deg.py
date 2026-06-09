#!/usr/bin/env python
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from arraylake import Client
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_grib_week1_t2m(file_path):
    """Loads ECMWF/NCEP 6-hourly GRIB, averages over Week 1 for T2M."""
    ds = xr.open_dataset(file_path, engine='cfgrib')
    # Use 2t (T2m) and calculate mean over days 1 to 7
    steps_hours = ds.step.values / np.timedelta64(1, 'h')
    day = (steps_hours - 0.1) // 24 + 1
    ds = ds.assign_coords(day=('step', day))
    
    daily_mean = ds['2t'].groupby('day').mean(dim='step')
    week1_mean = daily_mean.sel(day=slice(1, 7)).mean(dim='day')
    return week1_mean - 273.15

def get_fuxi_week1_t2m(fuxi_dir):
    """Loads FuXi-S2S daily predictions (01.nc to 07.nc) and averages over Week 1 for T2M."""
    files = [os.path.join(fuxi_dir, "member", "00", f"{d:02d}.nc") for d in range(1, 8)]
    datasets = [xr.open_dataset(f) for f in files]
    ds = xr.concat(datasets, dim='lead_time')
    
    # Extract t2m (channel index for t2m is after 69 pressure variables, typically index 69 or by channel name)
    # FuXi 'data' variable has dims: time, lead_time, channel, lat, lon
    t2m_idx = list(ds.channel.values).index('t2m')
    t2m_da = ds['data'].isel(channel=t2m_idx).mean(dim='lead_time').squeeze()
    return t2m_da - 273.15

def main():
    date_str = "20260101"
    
    analysis_dir = os.path.dirname(os.path.abspath(__file__)) 
    pipeline_dir = os.path.dirname(analysis_dir)
    benchmark_dir = os.path.dirname(pipeline_dir)
    
    ecmwf_file = os.path.join(benchmark_dir, "ecmwf", "data", f"sfc_cf_{date_str}_6h.grib")
    fuxi_dir = os.path.join(pipeline_dir, "fuxi_s2s_predictions", date_str)
    
    print("Loading ECMWF...")
    ecmwf_week1 = get_grib_week1_t2m(ecmwf_file)
    ecmwf_ds = xr.open_dataset(ecmwf_file, engine='cfgrib')
    
    print("Loading FuXi-S2S...")
    fuxi_week1_raw = get_fuxi_week1_t2m(fuxi_dir)
    
    lat_min, lat_max = 0.0, 50.0
    lon_min, lon_max = 55.0, 105.0
    
    print("Connecting to Arraylake API for Spire data...")
    client = Client()
    repo = client.get_repo('artamta/s2s-research')
    session = repo.readonly_session('main')
    ds_spire = xr.open_zarr(session.store, group='mean_stddev')
    
    steps = [np.timedelta64(d, 'D') for d in range(1, 8)]
    spire_week1_raw = ds_spire['air_temperature_max'].sel(
        reference_time='2026-01-01', step=steps
    ).mean('step').compute() - 273.15
    
    if spire_week1_raw.latitude.values[0] > spire_week1_raw.latitude.values[-1]:
        spire_week1_raw = spire_week1_raw.isel(latitude=slice(None, None, -1))
    spire_week1_raw = spire_week1_raw.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    
    print("Connecting to ARCO-ERA5 GCS Zarr Store for Ground Truth...")
    ds_e = xr.open_zarr('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3', storage_options={'token': 'anon'})
    
    da_era5 = ds_e['2m_temperature'].sel(
        latitude=slice(lat_max+1, lat_min-1), longitude=slice(lon_min-1, lon_max+1),
        time=slice('2026-01-02T00:00', '2026-01-08T23:00')
    )
    era5_week1_raw = da_era5.resample(time='1D').mean('time').mean('time').compute() - 273.15
    
    print("Interpolating all datasets to ECMWF 1.5° grid...")
    spire_week1 = spire_week1_raw.interp(latitude=ecmwf_ds.latitude, longitude=ecmwf_ds.longitude, method='linear')
    era5_week1 = era5_week1_raw.interp(latitude=ecmwf_ds.latitude, longitude=ecmwf_ds.longitude, method='linear')
    
    # FuXi is already 1.5 deg, but crop it
    fuxi_week1_raw = fuxi_week1_raw.assign_coords(lon=(((fuxi_week1_raw.lon + 180) % 360) - 180)) # if lon is 0-360
    fuxi_week1 = fuxi_week1_raw.interp(lat=ecmwf_ds.latitude, lon=ecmwf_ds.longitude, method='linear')
    
    fig = plt.figure(figsize=(14, 11))
    
    vmin = min(ecmwf_week1.min().item(), fuxi_week1.min().item(), spire_week1.min().item(), era5_week1.min().item())
    vmax = max(ecmwf_week1.max().item(), fuxi_week1.max().item(), spire_week1.max().item(), era5_week1.max().item())
    vmin, vmax = np.floor(vmin), np.ceil(vmax)
    
    panels = [
        ("ERA5 Ground Truth (T2M)", era5_week1, 1),
        ("Spire Forecast (T2M)", spire_week1, 2),
        ("ECMWF Control (T2M)", ecmwf_week1, 3),
        ("FuXi-S2S AI (T2M)", fuxi_week1, 4)
    ]
    
    lon, lat = ecmwf_ds.longitude.values, ecmwf_ds.latitude.values
    
    for name, data, idx in panels:
        ax = fig.add_subplot(2, 2, idx, projection=ccrs.PlateCarree())
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False; gl.right_labels = False
        
        # Note: FuXi uses 'lat', 'lon' while others use 'latitude', 'longitude'
        # We interpolated so values are aligned, but just use numpy array
        im = ax.pcolormesh(lon, lat, data.values, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(name, fontsize=12, fontweight='bold')
    
    fig.subplots_adjust(bottom=0.15, hspace=0.15)
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Week 1 Mean T2M (°C)', fontsize=12)
    
    fig.suptitle('FuXi vs Spire vs ECMWF (Week 1, Init: 2026-01-01)', fontsize=15, fontweight='bold', y=0.95)
    
    plot_path = os.path.join(analysis_dir, "fuxi_vs_ecmwf_spire_era5_week1_t2m.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")

if __name__ == "__main__":
    main()
