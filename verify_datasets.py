# %% [markdown]
# # S2S Dataset Verification Notebook
# Open and inspect all 5 datasets: ERA5, ECMWF, NCEP, FuXi, SPIRE

# %% CELL 1 — Imports
import numpy as np
import pandas as pd
import xarray as xr
import os, glob

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("xarray:", xr.__version__)
print("All imports OK")

# %% CELL 2 — Define paths and init dates
BASE = "/storage/raj.ayush/s2s-forecast-data"

PATHS = {
    "era5":  f"{BASE}/era5/data/",
    "ecmwf": f"{BASE}/ecmwf/data/",
    "ncep":  f"{BASE}/ncep/data/",
    "fuxi":  f"{BASE}/fuxi/output/",
    "spire": f"{BASE}/spire/",
}

INIT_DATES = [
    "20260101","20260108","20260115","20260122","20260129",
    "20260205","20260212","20260219","20260226",
    "20260305","20260312","20260319","20260326",
]

print("Checking paths...")
for name, path in PATHS.items():
    exists = os.path.exists(path)
    print(f"  {'OK' if exists else 'MISSING'} {name}: {path}")

# %% CELL 3 — ERA5: list files and open one
print("=== ERA5 ===")
era5_files = sorted(glob.glob(PATHS["era5"] + "*.nc"))
print(f"Files found: {len(era5_files)}")
for f in era5_files[:5]:
    print(" ", os.path.basename(f))

# Open first file
ds_era5 = xr.open_dataset(era5_files[0])
print("\nFirst file:", os.path.basename(era5_files[0]))
print(ds_era5)

# %% CELL 4 — ERA5: quick sanity check
print("=== ERA5 Variables ===")
for var in ds_era5.data_vars:
    da = ds_era5[var]
    print(f"  {var}: shape={da.shape}, units={da.attrs.get('units','?')}, "
          f"min={float(da.min()):.3f}, max={float(da.max()):.3f}, mean={float(da.mean()):.3f}")

# %% CELL 5 — ECMWF: list files and open one
print("=== ECMWF ===")
ecmwf_sfc = sorted(glob.glob(PATHS["ecmwf"] + "sfc_pf_*.grib"))
ecmwf_pl  = sorted(glob.glob(PATHS["ecmwf"] + "pl_pf_*.grib"))
print(f"Surface files: {len(ecmwf_sfc)}")
print(f"Pressure-level files: {len(ecmwf_pl)}")
for f in ecmwf_sfc[:3]:
    print(" ", os.path.basename(f))

# Open one surface file
import cfgrib
ds_ecmwf_sfc = xr.open_dataset(ecmwf_sfc[0], engine="cfgrib",
                                 backend_kwargs={"indexpath": ""})
print("\nFirst sfc file:", os.path.basename(ecmwf_sfc[0]))
print(ds_ecmwf_sfc)

# %% CELL 6 — ECMWF: sanity check
print("=== ECMWF Surface Variables ===")
for var in ds_ecmwf_sfc.data_vars:
    da = ds_ecmwf_sfc[var]
    print(f"  {var}: shape={da.shape}, units={da.attrs.get('units','?')}, "
          f"min={float(da.min()):.4f}, max={float(da.max()):.4f}")

# %% CELL 7 — NCEP: list files and open one
print("=== NCEP ===")
ncep_sfc = sorted(glob.glob(PATHS["ncep"] + "sfc_pf_*.grib"))
ncep_pl  = sorted(glob.glob(PATHS["ncep"] + "pl_pf_*.grib"))
print(f"Surface files: {len(ncep_sfc)}")
print(f"Pressure-level files: {len(ncep_pl)}")

ds_ncep_sfc = xr.open_dataset(ncep_sfc[0], engine="cfgrib",
                                backend_kwargs={"indexpath": ""})
print("\nFirst sfc file:", os.path.basename(ncep_sfc[0]))
print(ds_ncep_sfc)

# %% CELL 8 — NCEP: sanity check
print("=== NCEP Surface Variables ===")
for var in ds_ncep_sfc.data_vars:
    da = ds_ncep_sfc[var]
    print(f"  {var}: shape={da.shape}, units={da.attrs.get('units','?')}, "
          f"min={float(da.min()):.4f}, max={float(da.max()):.4f}")

# %% CELL 9 — FuXi: structure check
print("=== FuXi ===")
# Structure: fuxi/output/YYYYMMDD/member/MM/DD.nc
date = INIT_DATES[0]   # 20260101
fuxi_date_dir = os.path.join(PATHS["fuxi"], date)
print(f"Date dir: {fuxi_date_dir}")
print(f"Exists: {os.path.exists(fuxi_date_dir)}")

members = sorted(os.listdir(fuxi_date_dir))
print(f"Members: {members}")

# Open member 00, day 01
fuxi_file = os.path.join(fuxi_date_dir, "member", "00", "01.nc")
print(f"\nOpening: {fuxi_file}")
ds_fuxi = xr.open_dataset(fuxi_file)
print(ds_fuxi)

# %% CELL 10 — FuXi: sanity check
print("=== FuXi Variables ===")
for var in ds_fuxi.data_vars:
    da = ds_fuxi[var]
    print(f"  {var}: shape={da.shape}, dtype={da.dtype}, "
          f"min={float(da.min()):.4f}, max={float(da.max()):.4f}, mean={float(da.mean()):.4f}")

# Channel index for z500 (should be channel 5)
if "channel" in ds_fuxi.dims:
    print(f"\nChannels: {list(ds_fuxi.channel.values)}")

# %% CELL 11 — SPIRE: open zarr
print("=== SPIRE ===")
import zarr

spire_files = sorted(glob.glob(PATHS["spire"] + "*.zarr"))
print(f"Zarr stores found: {len(spire_files)}")
for f in spire_files:
    print(" ", os.path.basename(f))

# Open the main store
spire_store = zarr.open(spire_files[0], mode="r")
print("\nGroups in zarr:")
print(list(spire_store.keys()))

# %% CELL 12 — SPIRE: open with xarray
ds_spire = xr.open_zarr(spire_files[0], group="mean_stddev")
print("=== SPIRE mean_stddev group ===")
print(ds_spire)

# %% CELL 13 — SPIRE: sanity check
print("=== SPIRE Variables ===")
for var in ds_spire.data_vars:
    da = ds_spire[var]
    try:
        india = da.sel(latitude=slice(38, 5), longitude=slice(65, 100))
        mean_val = float(india.mean())
        print(f"  {var}: shape={da.shape}, India mean={mean_val:.4f}")
    except Exception as e:
        print(f"  {var}: shape={da.shape}, (could not compute India mean: {e})")

# %% CELL 14 — Summary table
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
print(f"ERA5  : {len(era5_files)} files, vars={list(ds_era5.data_vars)}")
print(f"ECMWF : {len(ecmwf_sfc)} sfc + {len(ecmwf_pl)} pl files")
print(f"NCEP  : {len(ncep_sfc)} sfc + {len(ncep_pl)} pl files")
print(f"FuXi  : {len(members)} members found for {date}")
print(f"SPIRE : {len(spire_files)} zarr store(s), groups={list(spire_store.keys())}")
print("="*60)
