"""
01_download_data.py

Download Spire S2S forecast and ERA5 observed daily T2m for a single init date.
All four downloads (Spire global, Spire India, ERA5 global, ERA5 India) run
in parallel threads.

Init date : 2026-01-01
Lead days : 1–46  (valid dates: 2026-01-02 → 2026-02-16)

Domains:
  India  : lat  5–40°N,  lon  65–100°E
  Global : full global grid

Spire outputs (from arraylake):
  spire_forecast_jan1_india.zarr   — groups: mean_stddev/, percentiles/
  spire_forecast_jan1_global.zarr  — groups: mean_stddev/, percentiles/

ERA5 outputs (from ARCO-ERA5 GCS):
  era5_observed_jan2_feb16_india.zarr   — daily mean 2m_temperature (°C)
  era5_observed_jan2_feb16_global.zarr  — daily mean 2m_temperature (°C), 0.5° grid

NOTE: Spire has NO raw ensemble members — only pre-computed mean, stddev,
      and percentiles. Percentiles (p1,5,10,20,50,80,90,95,99) represent the
      forecast uncertainty / spread.
"""

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import os
import threading

INIT_DATE = "2026-01-01"
LEAD_DAYS = 46
OUT_DIR   = "."

# India box
IND_LAT_MIN, IND_LAT_MAX =  5.0, 40.0
IND_LON_MIN, IND_LON_MAX = 65.0, 100.0

SPIRE_INDIA_OUT  = os.path.join(OUT_DIR, "spire_forecast_jan1_india.zarr")
SPIRE_GLOBAL_OUT = os.path.join(OUT_DIR, "spire_forecast_jan1_global.zarr")
ERA5_INDIA_OUT   = os.path.join(OUT_DIR, "era5_observed_jan2_feb16_india.zarr")
ERA5_GLOBAL_OUT  = os.path.join(OUT_DIR, "era5_observed_jan2_feb16_global.zarr")

errors = {}
errors_lock = threading.Lock()

# ── Spire helper ──────────────────────────────────────────────────────────────
def _spire_download(label, out_path, lat_min, lat_max, lon_min, lon_max):
    """Fetch Spire mean+stddev+percentiles for one domain and save to zarr."""
    try:
        from arraylake import Client
        print(f"[Spire-{label}] Connecting …")
        client  = Client()
        session = client.get_repo("artamta/s2s-research").readonly_session("main")
        steps   = [np.timedelta64(d, "D") for d in range(1, LEAD_DAYS + 1)]
        init_ts = pd.Timestamp(INIT_DATE)

        # ── mean + stddev ──────────────────────────────────────────────────────
        print(f"[Spire-{label}] mean_stddev …")
        ds_ms = xr.open_zarr(session.store, group="mean_stddev")
        ds_ms = ds_ms.isel(latitude=slice(None, None, -1))   # flip to S→N
        if lat_min is not None:
            ds_ms = ds_ms.sel(latitude=slice(lat_min, lat_max),
                              longitude=slice(lon_min, lon_max))
        ds_ms_init = ds_ms.sel(reference_time=init_ts, step=steps)

        t_mean   = ds_ms_init["air_temperature"].compute() - 273.15
        t_stddev = ds_ms_init["air_temperature_stddev"].compute()
        t_mean.attrs.update({"units": "°C",  "long_name": "Spire ensemble mean 2m temperature"})
        t_stddev.attrs.update({"units": "K", "long_name": "Spire ensemble stddev 2m temperature"})

        ds_out_ms = xr.Dataset({"t2m_mean": t_mean, "t2m_stddev": t_stddev})
        ds_out_ms.attrs.update({
            "init_date": INIT_DATE,
            "domain": label,
            "description": f"Spire S2S daily forecast — ensemble mean & stddev ({label})",
            "note": "No raw ensemble members. Mean/stddev pre-computed by Spire.",
        })

        # ── percentiles ────────────────────────────────────────────────────────
        print(f"[Spire-{label}] percentiles …")
        ds_pctl = xr.open_zarr(session.store, group="percentiles")
        ds_pctl = ds_pctl.isel(latitude=slice(None, None, -1))
        if lat_min is not None:
            ds_pctl = ds_pctl.sel(latitude=slice(lat_min, lat_max),
                                  longitude=slice(lon_min, lon_max))
        ds_pctl_init = ds_pctl.sel(reference_time=init_ts, step=steps)

        t_pctl = ds_pctl_init["air_temperature_pctl"].compute() - 273.15
        t_pctl.attrs.update({
            "units": "°C",
            "long_name": "Spire 2m temperature percentiles",
            "percentile_values": "1,5,10,20,50,80,90,95,99",
        })
        ds_out_pctl = xr.Dataset({"t2m_pctl": t_pctl})
        ds_out_pctl.attrs.update({
            "init_date": INIT_DATE,
            "domain": label,
            "description": f"Spire S2S daily forecast — temperature percentiles ({label})",
        })

        print(f"[Spire-{label}] Saving → {out_path} …")
        ds_out_ms.to_zarr(out_path,   group="mean_stddev", mode="w")
        ds_out_pctl.to_zarr(out_path, group="percentiles", mode="a")
        print(f"[Spire-{label}] Done. Shape: {t_mean.shape}")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with errors_lock:
            errors[f"spire_{label}"] = tb
        print(f"[Spire-{label}] ERROR: {e}")


# ── ERA5 helper ───────────────────────────────────────────────────────────────
def _era5_download(label, out_path, lat_min, lat_max, lon_min, lon_max,
                   new_lat, new_lon):
    """Fetch ERA5 daily mean T2m for one domain and save to zarr."""
    try:
        print(f"[ERA5-{label}] Opening ARCO-ERA5 …")
        ds_era5 = xr.open_zarr(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            storage_options={"token": "anon"},
        )
        init_ts = pd.Timestamp(INIT_DATE)
        t_start = init_ts + pd.Timedelta(1, "D")
        t_end   = init_ts + pd.Timedelta(LEAD_DAYS, "D")

        print(f"[ERA5-{label}] Fetching {t_start.date()} → {t_end.date()} …")
        if lat_min is not None:
            da = ds_era5["2m_temperature"].sel(
                latitude=slice(lat_max + 1, lat_min - 1),   # N→S in ERA5
                longitude=slice(lon_min - 1, lon_max + 1),
                time=slice(f"{t_start.date()}T00:00", f"{t_end.date()}T23:00"),
            )
        else:
            da = ds_era5["2m_temperature"].sel(
                time=slice(f"{t_start.date()}T00:00", f"{t_end.date()}T23:00"),
            )

        print(f"[ERA5-{label}] Resampling to daily mean …")
        daily = da.resample(time="1D").mean("time").compute() - 273.15

        # Regrid to target grid (0.5° to match Spire)
        print(f"[ERA5-{label}] Regridding to 0.5° …")
        daily = daily.interp(latitude=new_lat, longitude=new_lon, method="linear")

        daily.attrs.update({"units": "°C", "long_name": "ERA5 daily mean 2m temperature"})

        ds_out = xr.Dataset({"t2m": daily})
        ds_out.attrs.update({
            "description": (f"ERA5 observed daily mean T2m ({label}), "
                            f"valid {t_start.date()} – {t_end.date()}, 0.5° grid"),
            "source": "ARCO-ERA5 gs://gcp-public-data-arco-era5",
            "domain": label,
        })

        print(f"[ERA5-{label}] Saving → {out_path} …")
        ds_out.to_zarr(out_path, mode="w")
        print(f"[ERA5-{label}] Done. Shape: {daily.shape}")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with errors_lock:
            errors[f"era5_{label}"] = tb
        print(f"[ERA5-{label}] ERROR: {e}")


# ── grids ─────────────────────────────────────────────────────────────────────
india_lat  = np.arange(IND_LAT_MIN, IND_LAT_MAX + 0.1, 0.5)
india_lon  = np.arange(IND_LON_MIN, IND_LON_MAX + 0.1, 0.5)
global_lat = np.arange(-90, 90.1, 0.5)
global_lon = np.arange(0, 360.0, 0.5)

# ── launch 4 threads in parallel ──────────────────────────────────────────────
print("=" * 65)
print(f"Init date : {INIT_DATE}   Lead days : {LEAD_DAYS}")
print("Launching 4 parallel downloads:")
print("  1. Spire India   2. Spire Global")
print("  3. ERA5  India   4. ERA5  Global")
print("=" * 65)

threads = [
    threading.Thread(target=_spire_download,
                     args=("India", SPIRE_INDIA_OUT,
                           IND_LAT_MIN, IND_LAT_MAX, IND_LON_MIN, IND_LON_MAX)),
    threading.Thread(target=_spire_download,
                     args=("Global", SPIRE_GLOBAL_OUT,
                           None, None, None, None)),
    threading.Thread(target=_era5_download,
                     args=("India", ERA5_INDIA_OUT,
                           IND_LAT_MIN, IND_LAT_MAX, IND_LON_MIN, IND_LON_MAX,
                           india_lat, india_lon)),
    threading.Thread(target=_era5_download,
                     args=("Global", ERA5_GLOBAL_OUT,
                           None, None, None, None,
                           global_lat, global_lon)),
]

for t in threads:
    t.start()
for t in threads:
    t.join()

print("\n" + "=" * 65)
if errors:
    for k, v in errors.items():
        print(f"\n[{k.upper()}] FAILED:\n{v}")
else:
    print("All 4 downloads completed successfully.")
    for path, grp in [(SPIRE_INDIA_OUT, "mean_stddev"),
                      (SPIRE_GLOBAL_OUT, "mean_stddev"),
                      (ERA5_INDIA_OUT,   None),
                      (ERA5_GLOBAL_OUT,  None)]:
        ds = xr.open_zarr(path, group=grp) if grp else xr.open_zarr(path)
        dims = dict(ds.dims)
        print(f"  {os.path.basename(path)}/{grp or ''}: {dims}")

