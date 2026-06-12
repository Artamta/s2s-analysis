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

out_dir = '/storage/raj.ayush/s2s-forecast-data/spire'
out_path = os.path.join(out_dir, 'spire_hindcast_jfm.zarr')

client = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")
init_ts = pd.to_datetime(dates)

groups_to_fetch = ['anomalies', 'probabilities', 'regimes']

for group in groups_to_fetch:
    print(f"Fetching {group} variables...")
    try:
        ds = xr.open_zarr(session.store, group=group)
        ds_sub = ds.sel(reference_time=init_ts)
        print(f"Saving {group} to {out_path}...")
        ds_sub.to_zarr(out_path, group=group, mode="a")
    except Exception as e:
        print(f"Failed to fetch {group}: {e}")

print("Remaining groups downloaded successfully!")
