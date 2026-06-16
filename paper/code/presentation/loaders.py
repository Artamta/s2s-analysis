#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
loaders.py  —  Configuration + all data access for the S2S benchmark.
================================================================================
This module isolates EVERYTHING period/data-specific so the driver (verify_s2s)
and the maths (metrics) stay clean. To run a DIFFERENT period (e.g. JJAS 2019)
you edit only the CONFIG dataclass below; nothing else changes.

Contents
--------
  CONFIG (dataclass)     every path, date, grid, unit constant, threshold
  build_grid_context()   verification grid + land mask + IMD region masks
  to_grid / region_da    interpolation + land masking + region selection
  open_clim/clim_field   30-yr WMO climatology access (TP/Z500/T2M)
  open_truth + truth_*    ERA5 daily truth (TP mm/day, T2M K, Z500 gpm)
  load_spire/fuxi/op      per-system raw loaders (return ensemble containers)
  persistence_field       observed pre-init week (Dec-2025 patch aware)
  weekly/daily aggregation helpers (cumulative-precip differencing, ens mean/std)

Units (raw -> verification)
---------------------------
  Z500 : geopotential [m^2 s^-2] / 9.80665 -> gpm  (ERA5 z, FuXi z500, clim z500)
         ECMWF/NCEP 'gh' already gpm.
  TP   : truth = TRUE 24-h ERA5 daily total [mm/day]; ECMWF/NCEP 'tp' CUMULATIVE
         -> differenced to mm/day; FuXi 'tp' mm/hour x24 -> mm/day; clim tp m x1000.
  T2M  : truth = ERA5 daily-MEAN [K]; SPIRE air_temperature & FuXi t2m are
         INSTANTANEOUS; ECMWF/NCEP only archive 6-h max/min -> (mx2t6+mn2t6)/2.
         All 4 are kept and scored as ANOMALIES vs the ERA5 daily-mean climatology
         (the constant instantaneous-vs-daily-mean offset cancels in anomalies;
         raw bias is reported with a documented caveat). NOTE: ECMWF instantaneous
         '2t' is analysis-only (no forecast steps), so (mx+mn)/2 is the only
         ECMWF forecast T2M available — same definition as NCEP.

SPIRE mean vs stddev
--------------------
  SPIRE is delivered as an ensemble SUMMARY (per-step mean + per-step stddev,
  zarr group 'mean_stddev'), not raw members. Deterministic track uses the mean;
  probabilistic track uses Gaussian(mean, stddev). This makes SPIRE comparable
  to the member-based systems (FuXi 11, ECMWF 100, NCEP 15).
================================================================================
"""
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

# utils/ (validated land mask) ------------------------------------------------
CODE_ROOT = '/home/raj.ayush/s2s/s2s_anlysis/paper/code'
sys.path.append(CODE_ROOT)
from utils.verification_extra import get_land_mask, mask_land   # noqa: E402

OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})


# ==============================================================================
# CONFIG  —  EDIT THIS BLOCK ONLY to retarget a different period (e.g. JJAS2019)
# ==============================================================================
@dataclass
class Config:
    # --- period identity ------------------------------------------------------
    season_label: str = 'JFM2026'
    init_dates: tuple = (
        '2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
        '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
        '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26',
    )
    valid_end: str = '2026-05-10'       # last ERA5 daily-truth date available
    valid_end_clim: str = '2026-05-15'  # last date used for clim-anomaly window
    grib_start: str = '2026-01-01'      # ERA5 GRIB start (before -> use dec25 patch)

    # --- data roots / files ---------------------------------------------------
    data_root: str = '/storage/raj.ayush/s2s-forecast-data'
    clim_path: str = '/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc'
    spire_zarr: str = '/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr'
    fuxi_root: str = '/storage/raj.ayush/s2s-forecast-data/fuxi/output'
    region_mask_nc: str = '/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc'
    era5_daily_tp: str = '/storage/raj.ayush/s2s-forecast-data/era5/daily/era5_daily_tp.nc'
    era5_daily_t2m: str = '/storage/raj.ayush/s2s-forecast-data/era5/daily/era5_daily_t2m.nc'
    era5_z500_grib: str = '/storage/raj.ayush/s2s-forecast-data/era5/data/era5_pressure_500hpa.grib'
    era5_surface_grib: str = '/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface.grib'
    dec25_patch: str = '/storage/raj.ayush/s2s-forecast-data/era5/data/era5_dec2025_persistence.nc'

    # --- verification grid ----------------------------------------------------
    lat0: float = 38.0
    lat1: float = 5.0
    lon0: float = 65.0
    lon1: float = 100.0
    dgrid: float = 1.5

    # --- physics / units ------------------------------------------------------
    G: float = 9.80665
    fuxi_tp_factor: float = 24.0
    sig_floor: dict = field(default_factory=lambda: {'TP': 0.05, 'Z500': 1.0, 'T2M': 0.1})

    # --- Brier event thresholds ----------------------------------------------
    tp_thresholds: tuple = (1.0, 10.0)  # mm/day exceedance events
    use_terciles: bool = True           # above-/below-normal tercile events per var

    # --- ensembles ------------------------------------------------------------
    fuxi_members: int = 11

    # how each system's T2M is defined (recorded in metadata; all scored as anomaly)
    t2m_definition: dict = field(default_factory=lambda: {
        'SPIRE': 'instantaneous', 'FuXi': 'instantaneous',
        'ECMWF': '(mx2t6+mn2t6)/2', 'NCEP': '(mx2t6+mn2t6)/2'})

    def grid(self):
        lat = np.arange(self.lat0, self.lat1, -self.dgrid)
        lon = np.arange(self.lon0, self.lon1, self.dgrid)
        return lat, lon


CFG = Config()

# ---- pipeline-wide constants (period-agnostic) ------------------------------
WEEKS = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
         ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
REGIONS = ['All India', 'northwest_india', 'central_india',
           'south_peninsula', 'east_northeast_india']
BASE_SYSTEMS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
DET_MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME', 'Persistence']
VARS = ['TP', 'Z500', 'T2M']

CLIM_VAR = {'TP': 'tp', 'Z500': 'z500', 'T2M': 't2m'}
CLIM_SCALE = {'TP': 1000.0, 'Z500': 1.0 / CFG.G, 'T2M': 1.0}


# ==============================================================================
# GRID + MASK CONTEXT  (built once per worker process)
# ==============================================================================
def build_grid_context():
    lat, lon = CFG.grid()
    land = get_land_mask(lat, lon)
    mds = xr.open_dataset(CFG.region_mask_nc)
    region_masks = {k: mds[k].values.astype(bool) for k in mds.data_vars}
    all_india = np.zeros((len(lat), len(lon)), dtype=bool)
    for m in region_masks.values():
        all_india |= m
    return dict(lat=lat, lon=lon, land=land,
                region_masks=region_masks, all_india=all_india)


def to_grid(da, GC):
    ren = {}
    if 'latitude' in da.dims:
        ren['latitude'] = 'lat'
    if 'longitude' in da.dims:
        ren['longitude'] = 'lon'
    if ren:
        da = da.rename(ren)
    out = da.interp(lat=GC['lat'], lon=GC['lon'], method='linear').squeeze()
    return mask_land(out, GC['land'])


def region_da(rg, GC):
    arr = GC['all_india'] if rg == 'All India' else GC['region_masks'][rg]
    return xr.DataArray(arr, dims=['lat', 'lon'],
                        coords={'lat': GC['lat'], 'lon': GC['lon']})


def crop_box(da):
    """Pre-crop a global field to a generous India box (speeds interpolation)."""
    la = 'latitude' if 'latitude' in da.dims else 'lat'
    lo = 'longitude' if 'longitude' in da.dims else 'lon'
    lasl = slice(40, 3) if float(da[la][0]) > float(da[la][-1]) else slice(3, 40)
    losl = slice(60, 102) if float(da[lo][0]) < float(da[lo][-1]) else slice(102, 60)
    return da.sel({la: lasl, lo: losl})


# ==============================================================================
# CLIMATOLOGY  (30-yr WMO, day-of-year)
# ==============================================================================
def open_clim():
    return xr.open_dataset(CFG.clim_path)


def clim_field(clim_ds, var, doys, GC):
    """Climatological-mean field for a list of day-of-year values, on the
       verification grid, in verification units."""
    c = clim_ds[CLIM_VAR[var]].sel(dayofyear=doys).mean('dayofyear') * CLIM_SCALE[var]
    return to_grid(c, GC)


def clim_spread_field(var, truth, GC):
    """Per-grid-point CLIMATOLOGICAL SPREAD = temporal std of the ERA5 truth over
       the available verification period. Used as the spread of the climatology
       REFERENCE forecast for CRPSS, and to set Gaussian tercile boundaries for
       the Brier events. This reproduces the validated compute/06 approach
       (in-sample temporal std); it is the climatological uncertainty at each
       location, which is more principled than a single domain-wide scalar."""
    src = (truth['tp_daily'] if var == 'TP'
           else truth['t2m_daily'] if var == 'T2M' else truth['z_raw'])
    return to_grid(src.std('time'), GC)


# ==============================================================================
# ERA5 TRUTH  (opened once per worker)
# ==============================================================================
def open_truth():
    t = {}
    t['tp_daily'] = xr.open_dataset(CFG.era5_daily_tp)['tp']             # mm/day
    t['t2m_daily'] = xr.open_dataset(CFG.era5_daily_t2m)['t2m']          # K (daily mean)
    t['z_raw'] = crop_box(xr.open_dataset(CFG.era5_z500_grib, **OPEN)['z'] / CFG.G)  # gpm
    t['dec25'] = xr.open_dataset(CFG.dec25_patch) if os.path.exists(CFG.dec25_patch) else None
    return t


def truth_period_mean(var, truth, valid, GC):
    """ERA5 truth field averaged over the `valid` date list (weekly mean)."""
    try:
        if var == 'TP':
            src = truth['tp_daily']
        elif var == 'T2M':
            src = truth['t2m_daily']
        else:
            src = truth['z_raw']
        return to_grid(src.sel(time=slice(valid[0], valid[-1])).mean('time'), GC)
    except Exception:
        return None


def truth_day(var, truth, date, GC):
    """ERA5 truth field for a single day."""
    try:
        if var == 'TP':
            src = truth['tp_daily']
        elif var == 'T2M':
            src = truth['t2m_daily']
        else:
            src = truth['z_raw']
        o = to_grid(src.sel(time=slice(date, date)).mean('time'), GC)
        return o if not bool(np.isnan(o).all()) else None
    except Exception:
        return None


# ==============================================================================
# PERSISTENCE  (observed mean over the 7 days immediately BEFORE init)
# ==============================================================================
def persistence_field(var, truth, init, GC):
    pre = pd.date_range(end=pd.to_datetime(init) - pd.Timedelta(days=1), periods=7)
    valid = [d.strftime('%Y-%m-%d') for d in pre]
    pieces = []
    for d in valid:
        try:
            if d < CFG.grib_start and truth['dec25'] is not None:
                if var == 'TP':
                    pieces.append(to_grid(truth['dec25']['tp'].sel(time=d), GC))
                elif var == 'Z500':
                    pieces.append(to_grid(truth['dec25']['z500'].sel(time=d), GC))
                else:
                    # no dec25 t2m patch -> skip pre-grib days for T2M persistence
                    continue
            else:
                src = (truth['tp_daily'] if var == 'TP'
                       else truth['t2m_daily'] if var == 'T2M' else truth['z_raw'])
                pieces.append(to_grid(src.sel(time=d), GC))
        except Exception:
            pass
    return xr.concat(pieces, 't').mean('t') if pieces else None


# ==============================================================================
# MODEL LOADERS
# ==============================================================================
def load_spire(init):
    """{var: (mean_da, stddev_da)} with a `step` dim; None if unavailable."""
    try:
        s = xr.open_zarr(CFG.spire_zarr, group='mean_stddev').sel(reference_time=init)
    except Exception as e:
        print(f"  [SPIRE] open fail {init}: {e}", flush=True)
        return None
    out = {}
    try:
        out['TP'] = (crop_box(s['precipitation_amount']),
                     crop_box(s['precipitation_amount_stddev']))
    except Exception:
        pass
    try:
        z = s['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0)
        zs = s['geopotential_height_at_isobaric_levels_stddev'].sel(isobar=50000.0)
        out['Z500'] = (crop_box(z), crop_box(zs))
    except Exception:
        pass
    try:
        out['T2M'] = (crop_box(s['air_temperature']),
                      crop_box(s['air_temperature_stddev']))
    except Exception:
        pass
    return out or None


def _fuxi_member_day(init_str, mem, day, var, G):
    p = f"{CFG.fuxi_root}/{init_str}/member/{mem:02d}/{day:02d}.nc"
    if not os.path.exists(p):
        return None
    try:
        da = xr.open_dataset(p)['__xarray_dataarray_variable__']
    except Exception:
        return None
    if var == 'TP':
        da = da.sel(channel='tp')
    elif var == 'T2M':
        da = da.sel(channel='t2m')
    else:
        da = da.sel(channel='z500') / G
    for d in list(da.dims):
        if d not in ('lat', 'lon', 'latitude', 'longitude'):
            da = da.mean(d)
    return crop_box(da)


def load_fuxi(init_str, var, G):
    """{day: DataArray(member, lat, lon)} for one variable; {} if none."""
    per_day = {}
    for day in range(1, 43):
        members = []
        for mem in range(CFG.fuxi_members):
            f = _fuxi_member_day(init_str, mem, day, var, G)
            if f is not None:
                members.append(f.assign_coords(member=mem))
        if members:
            per_day[day] = xr.concat(members, 'member')
    return per_day


def load_op(model, init_str, var, G):
    """ECMWF/NCEP DataArray with dims (number, step, lat, lon); None if missing.
       TP stays CUMULATIVE (differenced later). T2M = (mx2t6+mn2t6)/2 proxy."""
    base = f'{CFG.data_root}/{model.lower()}/data'
    try:
        if var == 'TP':
            d = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib',
                                filter_by_keys={'shortName': 'tp'}, **OPEN)['tp']
            return crop_box(d)
        if var == 'Z500':
            d = xr.open_dataset(f'{base}/pl_pf_{init_str}.grib',
                                filter_by_keys={'shortName': 'gh'}, **OPEN)['gh']
            if 'isobaricInhPa' in d.dims:
                d = d.sel(isobaricInhPa=500)
            return crop_box(d)
        if var == 'T2M':
            mx = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib',
                                 filter_by_keys={'shortName': 'mx2t6'}, **OPEN)['mx2t6']
            mn = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib',
                                 filter_by_keys={'shortName': 'mn2t6'}, **OPEN)['mn2t6']
            return crop_box((mx + mn) / 2.0)
    except Exception as e:
        print(f"  [{model}] {var} load fail {init_str}: {e}", flush=True)
    return None


# ==============================================================================
# AGGREGATION HELPERS
# ==============================================================================
def weekly_mean_cumulative(cum, ds, de):
    """Weekly-mean RATE from a CUMULATIVE total (ECMWF/NCEP tp)."""
    days = de - ds + 1
    if ds == 1:
        return cum.isel(step=de - 1) / days
    return (cum.isel(step=de - 1) - cum.isel(step=ds - 2)) / days


def daily_from_cumulative(cum, di):
    """Single-day RATE from a CUMULATIVE total at lead index di (0-based)."""
    return cum.isel(step=di) - cum.isel(step=di - 1) if di > 0 else cum.isel(step=di)


def ens_mean_std(da, mdim, GC):
    """Gridded ensemble mean & spread (ddof=1) over member dim `mdim`."""
    mu = to_grid(da.mean(mdim), GC)
    sig = to_grid(da.std(mdim, ddof=1), GC)
    return mu, sig


def valid_dates_for(init, ds, de, end):
    dates = pd.date_range(start=init, periods=42)[ds - 1:de]
    return [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= end]
