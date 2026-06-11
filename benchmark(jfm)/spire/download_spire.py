#!/usr/bin/env python
import os
import pandas as pd
import xarray as xr
from arraylake import Client

dates = [
    '2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
    '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
    '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26'
]

out_dir = '/storage/raj.ayush/spire-hindecast-weekely-initialized'
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, 'spire_hindcast_jfm.zarr')

print("Connecting to Spire Arraylake repo...")
client = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")

init_ts = pd.to_datetime(dates)

print("Fetching mean_stddev variables...")
ds_ms = xr.open_zarr(session.store, group="mean_stddev")
# Select only specific dates to avoid huge download
ds_ms_sub = ds_ms.sel(reference_time=init_ts)
# Select key variables to save space (user requested hindcast, but didn't specify vars. These are the most common)
vars_to_keep_ms = [
    'air_temperature', 'air_temperature_stddev', 
    'precipitation_amount', 'precipitation_amount_stddev',
    'geopotential_height_at_isobaric_levels', 'geopotential_height_at_isobaric_levels_stddev'
]
# Only keep vars that exist
vars_to_keep_ms = [v for v in vars_to_keep_ms if v in ds_ms_sub.data_vars]
ds_ms_sub = ds_ms_sub[vars_to_keep_ms]

print(f"Saving mean_stddev to {out_path}...")
ds_ms_sub.to_zarr(out_path, group="mean_stddev", mode="w")

print("Fetching percentiles variables...")
ds_pctl = xr.open_zarr(session.store, group="percentiles")
ds_pctl_sub = ds_pctl.sel(reference_time=init_ts)
vars_to_keep_pctl = [
    'air_temperature_pctl', 'precipitation_amount_pctl'
]
vars_to_keep_pctl = [v for v in vars_to_keep_pctl if v in ds_pctl_sub.data_vars]
ds_pctl_sub = ds_pctl_sub[vars_to_keep_pctl]

print(f"Saving percentiles to {out_path}...")
ds_pctl_sub.to_zarr(out_path, group="percentiles", mode="a")

print("Spire download completed!")
