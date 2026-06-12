from arraylake import Client
import xarray as xr

client = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")

ds_ms = xr.open_zarr(session.store, group="mean_stddev")
print("MEAN_STDDEV variables:", list(ds_ms.data_vars))
print("MEAN_STDDEV coords:", list(ds_ms.coords))

ds_pctl = xr.open_zarr(session.store, group="percentiles")
print("PERCENTILES variables:", list(ds_pctl.data_vars))
print("PERCENTILES coords:", list(ds_pctl.coords))
