"""
S2S Dataset Verification Script
================================
Opens every dataset, prints a full summary:
  - Variables, units, shape, resolution
  - Time range, domain coverage
  - Sanity-check values (India-domain mean)

Run: python verify_all_datasets.py
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import xarray as xr
import os, glob

SEP = "=" * 65

# ── Paths ────────────────────────────────────────────────────────────
BASE   = "/storage/raj.ayush/s2s-forecast-data"
ERA5_DAILY = f"{BASE}/era5/daily"
ERA5_RAW   = f"{BASE}/era5/data"
ECMWF_DIR  = f"{BASE}/ecmwf/data"
NCEP_DIR   = f"{BASE}/ncep/data"
FUXI_DIR   = f"{BASE}/fuxi/output"
SPIRE_DIR  = f"{BASE}/spire"

# India bounding box
LAT_S, LAT_N = 5, 38
LON_W, LON_E = 65, 100


def india_mean(da):
    """Cosine-weighted India-domain mean (handles both ascending & descending lat)."""
    try:
        lat_name = [d for d in da.dims if "lat" in d.lower()][0]
        lon_name = [d for d in da.dims if "lon" in d.lower()][0]
        lat = da[lat_name]
        # India slice
        if float(lat[0]) > float(lat[-1]):   # descending
            da_india = da.sel({lat_name: slice(LAT_N, LAT_S),
                               lon_name: slice(LON_W, LON_E)})
        else:                                 # ascending
            da_india = da.sel({lat_name: slice(LAT_S, LAT_N),
                               lon_name: slice(LON_W, LON_E)})
        weights = np.cos(np.deg2rad(da_india[lat_name]))
        return float(da_india.weighted(weights).mean())
    except Exception as e:
        return f"N/A ({e})"


def resolution_deg(coord):
    """Estimate grid spacing from a coordinate array."""
    vals = np.array(coord)
    if len(vals) < 2:
        return "?"
    return round(abs(float(vals[1]) - float(vals[0])), 4)


def print_var_table(ds, extra_dims=None):
    """Print a neat variable table for an xarray Dataset."""
    print(f"  {'Variable':<45} {'Shape':<25} {'Units':<15} {'Min':>10} {'Max':>10} {'India mean':>12}")
    print(f"  {'-'*45} {'-'*25} {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
    for var in ds.data_vars:
        da = ds[var]
        units = da.attrs.get("units", da.attrs.get("unit", "?"))
        try:
            mn  = float(da.min())
            mx  = float(da.max())
            ind = india_mean(da)
            ind_str = f"{ind:.4f}" if isinstance(ind, float) else str(ind)
            mn_str  = f"{mn:.4f}"
            mx_str  = f"{mx:.4f}"
        except Exception:
            mn_str, mx_str, ind_str = "?", "?", "?"
        print(f"  {var:<45} {str(da.shape):<25} {str(units):<15} {mn_str:>10} {mx_str:>10} {ind_str:>12}")


# ═══════════════════════════════════════════════════════════════════
print(SEP)
print("1. ERA5 DAILY TRUTH  (precomputed, ARCO-ERA5 hourly → daily)")
print(SEP)
for fname in ["era5_daily_tp.nc", "era5_daily_t2m.nc"]:
    fpath = os.path.join(ERA5_DAILY, fname)
    print(f"\n  File : {fpath}")
    if not os.path.exists(fpath):
        print("  ❌  NOT FOUND"); continue
    ds = xr.open_dataset(fpath)
    lat = ds["latitude"] if "latitude" in ds.coords else ds["lat"]
    lon = ds["longitude"] if "longitude" in ds.coords else ds["lon"]
    print(f"  Dims : {dict(ds.dims)}")
    print(f"  Res  : {resolution_deg(lat)}° lat × {resolution_deg(lon)}° lon")
    print(f"  Time : {str(ds.time.values[0])[:10]}  →  {str(ds.time.values[-1])[:10]}  ({len(ds.time)} days)")
    print(f"  Lat  : {float(lat.min()):.2f} – {float(lat.max()):.2f}")
    print(f"  Lon  : {float(lon.min()):.2f} – {float(lon.max()):.2f}")
    print_var_table(ds)
    ds.close()


# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("2. ERA5 RAW HOURLY  (per-day files in /storage/era5/data/)")
print(SEP)
raw_files = sorted(glob.glob(os.path.join(ERA5_RAW, "*.nc")))
print(f"  Files : {len(raw_files)}  ({os.path.basename(raw_files[0])} … {os.path.basename(raw_files[-1])})")
ds = xr.open_dataset(raw_files[0])
lat = ds["latitude"] if "latitude" in ds.coords else ds["lat"]
lon = ds["longitude"] if "longitude" in ds.coords else ds["lon"]
print(f"  Dims  : {dict(ds.dims)}")
print(f"  Res   : {resolution_deg(lat)}° lat × {resolution_deg(lon)}° lon  (0.25° ARCO-ERA5)")
print(f"  Time  : {str(ds.time.values[0])[:19]}  →  {str(ds.time.values[-1])[:19]}  (hourly, 1 file/day)")
print(f"  Lat   : {float(lat.min()):.2f} – {float(lat.max()):.2f}")
print(f"  Lon   : {float(lon.min()):.2f} – {float(lon.max()):.2f}")
print_var_table(ds)
ds.close()


# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("3. ECMWF  (1.5°, 100 PF members, 46 days)")
print(SEP)
import cfgrib
sfc_files = sorted(glob.glob(os.path.join(ECMWF_DIR, "sfc_pf_*.grib")))
pl_files  = sorted(glob.glob(os.path.join(ECMWF_DIR, "pl_pf_*.grib")))
print(f"  Surface files : {len(sfc_files)}")
print(f"  Pres-lev files: {len(pl_files)}")
print(f"\n  --- Surface (tp, mx2t6, mn2t6) ---")
ds_sfc = xr.open_dataset(sfc_files[0], engine="cfgrib",
                          backend_kwargs={"indexpath": ""})
lat = ds_sfc["latitude"]
lon = ds_sfc["longitude"]
print(f"  Dims : {dict(ds_sfc.dims)}")
print(f"  Res  : {resolution_deg(lat)}° lat × {resolution_deg(lon)}° lon")
step0 = int(ds_sfc.step.values[0] / 1e9 / 3600 / 24)
step1 = int(ds_sfc.step.values[-1] / 1e9 / 3600 / 24)
print(f"  Steps: day {step0} … day {step1}  ({len(ds_sfc.step)} steps)")
print(f"  Lat  : {float(lat.min()):.2f} – {float(lat.max()):.2f}")
print(f"  Lon  : {float(lon.min()):.2f} – {float(lon.max()):.2f}")
print_var_table(ds_sfc)

print(f"\n  --- Pressure levels (gh @ 500 hPa) ---")
ds_pl = xr.open_dataset(pl_files[0], engine="cfgrib",
                         backend_kwargs={"indexpath": "",
                                         "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": 500}})
print(f"  Dims : {dict(ds_pl.dims)}")
print_var_table(ds_pl)
ds_sfc.close(); ds_pl.close()


# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("4. NCEP  (1.5°, 15 PF members, 44 days)")
print(SEP)
sfc_files = sorted(glob.glob(os.path.join(NCEP_DIR, "sfc_pf_*.grib")))
pl_files  = sorted(glob.glob(os.path.join(NCEP_DIR, "pl_pf_*.grib")))
print(f"  Surface files : {len(sfc_files)}")
print(f"  Pres-lev files: {len(pl_files)}")
print(f"\n  --- Surface (tp, mx2t6, mn2t6) ---")
ds_sfc = xr.open_dataset(sfc_files[0], engine="cfgrib",
                          backend_kwargs={"indexpath": ""})
lat = ds_sfc["latitude"]
lon = ds_sfc["longitude"]
print(f"  Dims : {dict(ds_sfc.dims)}")
print(f"  Res  : {resolution_deg(lat)}° lat × {resolution_deg(lon)}° lon")
step0 = int(ds_sfc.step.values[0] / 1e9 / 3600 / 24)
step1 = int(ds_sfc.step.values[-1] / 1e9 / 3600 / 24)
print(f"  Steps: day {step0} … day {step1}  ({len(ds_sfc.step)} steps)")
print_var_table(ds_sfc)

print(f"\n  --- Pressure levels (gh @ 500 hPa) ---")
ds_pl = xr.open_dataset(pl_files[0], engine="cfgrib",
                         backend_kwargs={"indexpath": "",
                                         "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": 500}})
print(f"  Dims : {dict(ds_pl.dims)}")
print_var_table(ds_pl)
ds_sfc.close(); ds_pl.close()


# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("5. FuXi-S2S  (1.5°, 11 members, 42 days, 76 channels)")
print(SEP)
date       = "20260101"
member_dir = os.path.join(FUXI_DIR, date, "member")
members    = sorted(os.listdir(member_dir))
days       = sorted(os.listdir(os.path.join(member_dir, members[0])))
print(f"  Init date : {date}")
print(f"  Members   : {len(members)}  ({members[0]} … {members[-1]})")
print(f"  Days/files: {len(days)}  ({days[0]} … {days[-1]})")

# Open one file and inspect
ds = xr.open_dataset(os.path.join(member_dir, members[0], days[0]))
da = ds["__xarray_dataarray_variable__"]
channels = list(da.channel.values)
lat = da["lat"]
lon = da["lon"]
print(f"  Dims      : {dict(da.sizes)}")
print(f"  Res       : {resolution_deg(lat)}° lat × {resolution_deg(lon)}° lon  (global)")
print(f"  Lat range : {float(lat.min()):.1f} – {float(lat.max()):.1f}")
print(f"  Channels  : {len(channels)} total")

# Key channels
key_channels = {"z500": None, "t2m": None, "tp": None}
for ch in key_channels:
    if ch in channels:
        key_channels[ch] = channels.index(ch)
print(f"\n  Key channel indices:")
for ch, idx in key_channels.items():
    if idx is not None:
        val = float(da.sel(channel=ch).mean())
        print(f"    channel[{idx:>2}] = '{ch}'  →  global mean = {val:.4f}")

g = 9.80665
z500_raw = float(da.sel(channel="z500").mean())
print(f"\n  z500 raw mean  = {z500_raw:.2f} m²/s²")
print(f"  z500 as height = {z500_raw/g:.2f} m  (÷g={g})")

tp_raw = float(da.sel(channel="tp").mean())
print(f"  tp raw mean    = {tp_raw:.6f}  (rate, mm/h or mm/day?)")
print(f"  tp × 24        = {tp_raw*24:.4f}  (if rate → mm/day)")
ds.close()


# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("5. SPIRE AI-S2S  (0.5°, ens mean+std, 46 days, global)")
print(SEP)
import zarr
spire_files = sorted(glob.glob(os.path.join(SPIRE_DIR, "*.zarr")))
print(f"  Store : {os.path.basename(spire_files[0])}")
store = zarr.open(spire_files[0], mode="r")
groups = list(store.keys())
print(f"  Groups: {groups}")

print(f"\n  --- mean_stddev group ---")
ds = xr.open_zarr(spire_files[0], group="mean_stddev")
lat = ds["latitude"]
lon = ds["longitude"]
print(f"  Dims  : {dict(ds.dims)}")
print(f"  Res   : {resolution_deg(lat)}° lat × {resolution_deg(lon)}° lon  (global 0.5°)")
print(f"  Inits : {len(ds.reference_time)}  ({str(ds.reference_time.values[0])[:10]} … {str(ds.reference_time.values[-1])[:10]})")
print(f"  Steps : {len(ds.step)}  ({str(ds.step.values[0])} … {str(ds.step.values[-1])})")
if "isobar" in ds.dims:
    print(f"  Isobars (Pa): {[float(x) for x in ds.isobar.values]}  → 500 hPa = 50000 Pa")

print_var_table(ds)
ds.close()


# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FINAL SUMMARY")
print(SEP)
print("""
  Dataset   | Res   | Members | Steps | Key Variables                  | Units
  ----------|-------|---------|-------|--------------------------------|-------------------------------
  ERA5 daily| 0.25° | truth   | 130d  | tp, t2m                        | m/day, K
  ECMWF     | 1.5°  | 100 PF  | 46d   | mx2t6, mn2t6, tp, gh@500       | K, K, kg/m², gpm
  NCEP      | 1.5°  | 15 PF   | 44d   | mx2t6, mn2t6, tp, gh@500       | K, K, kg/m², gpm
  FuXi      | 1.5°  | 11      | 42d   | z500 (m²/s²), t2m (K), tp×24  | m²/s², K, mm/day
  SPIRE     | 0.5°  | mean+σ  | 46d   | air_temp, precip, gh@500hPa    | K, mm/day, m (×g for m²/s²)
""")
print(SEP)
