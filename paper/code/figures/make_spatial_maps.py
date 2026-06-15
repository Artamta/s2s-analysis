"""
make_spatial_maps.py  –  Publication-quality spatial maps
Rows: ERA5 | SPIRE | FuXi | ECMWF | NCEP | MME
Cols: TP  |  Z500
Produces one absolute-value figure and one anomaly figure.
Change INIT / WEEK at the top to inspect different cases.
"""
import warnings; warnings.filterwarnings('ignore')
from scipy.ndimage import gaussian_filter
import sys, os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')

# ── USER CONFIG ────────────────────────────────────────────────────────────────
ADIR      = '/home/raj.ayush/s2s/s2s_anlysis/paper/results'
FIGDIR    = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
CLIM_PATH = '/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc'
INIT      = '2026-01-08'   # change to inspect other cases
WEEK      = 'Week 2'       # Week 1 … Week 6
G         = 9.80665
PROJ      = ccrs.PlateCarree()
EXTENT    = [63, 100, 5, 38]

MODELS    = ['ERA5', 'SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME']
MLAB      = {'ERA5':'ERA5\n(Observed)','SPIRE':'SPIRE','FuXi':'FuXi-S2S',
             'ECMWF':'ECMWF','NCEP':'NCEP','MME':'MME'}
MCOL      = {'ERA5':'#222222','SPIRE':'#E06C00','FuXi':'#1A6FBF',
             'ECMWF':'#2A9D5C','NCEP':'#9B59B6','MME':'#1C1C1C'}

WEEK_MAP  = {'Week 1':(1,7),'Week 2':(8,14),'Week 3':(15,21),
             'Week 4':(22,28),'Week 5':(29,35),'Week 6':(36,42)}

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading weekly_anom_fields.nc ...", flush=True)
ds   = xr.open_dataset(f'{ADIR}/weekly_anom_fields.nc')
lats = ds.lat.values
lons = ds.lon.values
clim = xr.open_dataset(CLIM_PATH)

# ── India political boundary mask (union of 4 IMD regions) ────────────────────
print("Loading India mask ...", flush=True)
_mask_path = '/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc'
_mask_ds   = xr.open_dataset(_mask_path)
ALL_INDIA_MASK = np.zeros((len(lats), len(lons)), dtype=bool)
for _v in _mask_ds.data_vars:
    ALL_INDIA_MASK |= _mask_ds[_v].values.astype(bool)
print(f'  India mask: {ALL_INDIA_MASK.sum()} valid cells out of {ALL_INDIA_MASK.size}', flush=True)

init_list  = [str(x)[:10] for x in ds.init.values]
weeks_list = [str(x)      for x in ds.week.values]
ii         = init_list.index(INIT)

ds0, de0 = WEEK_MAP[WEEK]
dates    = pd.date_range(start=INIT, periods=42)[ds0-1:de0]
doys     = [d.dayofyear for d in dates]
date_str = f"{dates[0].strftime('%b %d')}–{dates[-1].strftime('%b %d, %Y')}"

def clim_on_grid(var, scale):
    raw = clim[var].sel(dayofyear=doys).mean('dayofyear').values * scale
    da  = xr.DataArray(raw, dims=['latitude','longitude'],
                       coords={'latitude':clim.latitude.values,
                               'longitude':clim.longitude.values})
    return da.interp(latitude=lats, longitude=lons).values

print("Computing climatologies ...", flush=True)
clim_tp = clim_on_grid('tp',   1000.0)
clim_z  = clim_on_grid('z500', 1.0/G)

def get_tp(m):
    if m == 'ERA5':
        return ds['tp_obs'].isel(init=ii).sel(week=WEEK).values + clim_tp
    a = ds['tp_fcst'].sel(model=m).isel(init=ii).sel(week=WEEK).values
    return None if np.all(np.isnan(a)) else a + clim_tp

def get_z(m):
    if m == 'ERA5':
        return ds['z_obs'].isel(init=ii).sel(week=WEEK).values + clim_z
    a = ds['z_fcst'].sel(model=m).isel(init=ii).sel(week=WEEK).values
    return None if np.all(np.isnan(a)) else a + clim_z

print("Assembling fields ...", flush=True)
tp_abs  = {m: get_tp(m) for m in MODELS}
z_abs   = {m: get_z(m)  for m in MODELS}
tp_anom = {m: (tp_abs[m]-clim_tp) if tp_abs[m] is not None else None for m in MODELS}
z_anom  = {m: (z_abs[m] -clim_z)  if z_abs[m]  is not None else None for m in MODELS}

# Apply India political boundary mask to all fields
def apply_india_mask(arr):
    if arr is None: return None
    out = arr.copy().astype(float)
    out[~ALL_INDIA_MASK] = np.nan
    return out

def smooth(arr, sigma=0.9):
    """Light Gaussian smooth for display only — NaN-aware."""
    if arr is None: return None
    filled = np.where(np.isnan(arr), 0.0, arr)
    weights = np.where(np.isnan(arr), 0.0, 1.0)
    s = gaussian_filter(filled,  sigma=sigma)
    w = gaussian_filter(weights, sigma=sigma)
    out = np.where(w > 0.1, s / w, np.nan)
    out[~ALL_INDIA_MASK] = np.nan   # re-apply mask after smoothing
    return out

for m in MODELS:
    tp_abs[m]  = smooth(apply_india_mask(tp_abs[m]))
    z_abs[m]   = smooth(apply_india_mask(z_abs[m]))
    tp_anom[m] = smooth(apply_india_mask(tp_anom[m]))
    z_anom[m]  = smooth(apply_india_mask(z_anom[m]))

tp_vmax  = max(3.0, float(np.nanpercentile(tp_abs['ERA5'], 97)))
tp_alim  = max(1.5, float(np.nanpercentile(np.abs(tp_anom['ERA5']), 95)))
zv       = z_abs['ERA5'][~np.isnan(z_abs['ERA5'])]
z_vmin, z_vmax, z_vctr = zv.min(), zv.max(), float(np.nanmean(zv))
z_alim   = max(25.0, float(np.nanpercentile(np.abs(z_anom['ERA5']), 95)))

# ── Map drawing helper ─────────────────────────────────────────────────────────
def draw_panel(ax, data, cmap, vmin, vmax, norm=None,
               ll=False, lb=False, strip=None):
    ax.set_extent(EXTENT, crs=PROJ)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#CCDDF5', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'),  facecolor='#F2F0EB', zorder=0)
    im = None
    if data is not None and not np.all(np.isnan(data)):
        kw = dict(transform=PROJ, cmap=cmap, zorder=2,
                  shading='auto', rasterized=True)
        kw['norm'] = norm if norm is not None else plt.Normalize(vmin, vmax)
        im = ax.pcolormesh(lons, lats, data, **kw)
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=8.5, color='#888')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.55, zorder=5, edgecolor='#111')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),   lw=0.35, zorder=5,
                   edgecolor='#333', linestyle='--')
    ax.add_feature(cfeature.STATES.with_scale('50m'),    lw=0.22, zorder=4, edgecolor='#999')
    gl = ax.gridlines(crs=PROJ, draw_labels=False, lw=0.3,
                      color='#AAAAAA', alpha=0.5, linestyle=':')
    gl.xlocator = mticker.FixedLocator(range(65,100,10))
    gl.ylocator = mticker.FixedLocator(range(10,40,10))
    if ll:
        gl.left_labels = True
        gl.yformatter  = LATITUDE_FORMATTER
        gl.ylabel_style = {'size':7.5,'color':'#444'}
    if lb:
        gl.bottom_labels = True
        gl.xformatter   = LONGITUDE_FORMATTER
        gl.xlabel_style = {'size':7.5,'color':'#444'}
    if strip:
        for spine in ax.spines.values():
            spine.set_edgecolor(strip); spine.set_linewidth(2.5)
    return im

# ── Figure builder ─────────────────────────────────────────────────────────────
def build_figure(mode):
    assert mode in ('abs','anom')
    nm = len(MODELS)
    fig = plt.figure(figsize=(14, 2.8*nm + 1.4), facecolor='white')
    gs  = GridSpec(nm, 2, figure=fig,
                   hspace=0.08, wspace=0.05,
                   top=0.93, bottom=0.06, left=0.11, right=0.95)

    if mode == 'abs':
        cmap_tp, vmin_tp, vmax_tp, norm_tp = 'YlGnBu',  0, tp_vmax, None
        cmap_z,  vmin_z,  vmax_z,  norm_z  = 'RdYlBu_r', z_vmin, z_vmax, \
            TwoSlopeNorm(vcenter=z_vctr, vmin=z_vmin, vmax=z_vmax)
        col_tp_cb = 'mm day⁻¹';  col_z_cb  = 'gpm'
        tp_data = tp_abs;  z_data = z_abs
        ttl_tp = 'Total Precipitation  (mm day⁻¹)'
        ttl_z  = 'Z500 Geopotential Height  (gpm)'
        suptitle = (f'Weekly Mean Forecasts vs ERA5  ·  Init: {INIT}  ·  '
                    f'{WEEK}  ({date_str})')
        tag = 'abs'
    else:
        cmap_tp, vmin_tp, vmax_tp = 'BrBG', -tp_alim, tp_alim
        norm_tp = TwoSlopeNorm(0, vmin=-tp_alim, vmax=tp_alim)
        cmap_z,  vmin_z,  vmax_z  = 'RdBu_r', -z_alim, z_alim
        norm_z  = TwoSlopeNorm(0, vmin=-z_alim,  vmax=z_alim)
        col_tp_cb = 'mm day⁻¹ anomaly';  col_z_cb = 'gpm anomaly'
        tp_data = tp_anom; z_data = z_anom
        ttl_tp = 'TP Anomaly  (mm day⁻¹,  rel. 1990–2019 climatology)'
        ttl_z  = 'Z500 Anomaly  (gpm,  rel. 1990–2019 climatology)'
        suptitle = (f'Weekly Mean Anomalies vs ERA5  ·  Init: {INIT}  ·  '
                    f'{WEEK}  ({date_str})\n'
                    'Anomaly = Forecast − 1990–2019 WMO 30-yr climatology')
        tag = 'anom'

    ims_tp, ims_z = [], []
    for ri, m in enumerate(MODELS):
        lb = (ri == nm-1)
        ax_tp = fig.add_subplot(gs[ri, 0], projection=PROJ)
        ax_z  = fig.add_subplot(gs[ri, 1], projection=PROJ)
        im_tp = draw_panel(ax_tp, tp_data[m], cmap_tp, vmin_tp, vmax_tp,
                           norm=norm_tp, ll=(ri==0), lb=lb, strip=MCOL[m])
        im_z  = draw_panel(ax_z,  z_data[m],  cmap_z,  vmin_z,  vmax_z,
                           norm=norm_z,  lb=lb, strip=MCOL[m])
        ax_tp.text(-0.02, 0.5, MLAB[m], transform=ax_tp.transAxes,
                   fontsize=9.5, fontweight='bold', color=MCOL[m],
                   ha='right', va='center')
        if im_tp is not None: ims_tp.append(im_tp)
        if im_z  is not None: ims_z.append(im_z)

    # column headers  (first row axes)
    row0_tp = fig.axes[0];  row0_z = fig.axes[1]
    row0_tp.set_title(ttl_tp, fontsize=11, fontweight='bold', pad=7, color='#111')
    row0_z.set_title( ttl_z,  fontsize=11, fontweight='bold', pad=7, color='#111')

    # shared colorbars
    cax_tp = fig.add_axes([0.11, 0.025, 0.38, 0.016])
    cax_z  = fig.add_axes([0.57, 0.025, 0.38, 0.016])
    kw_cb  = dict(orientation='horizontal', extend='both' if mode=='anom' else 'max')
    if ims_tp:
        cb1 = fig.colorbar(ims_tp[0], cax=cax_tp, **kw_cb)
        cb1.set_label(col_tp_cb, fontsize=9); cb1.ax.tick_params(labelsize=8)
    if ims_z:
        cb2 = fig.colorbar(ims_z[0],  cax=cax_z,  **kw_cb)
        cb2.set_label(col_z_cb,  fontsize=9); cb2.ax.tick_params(labelsize=8)

    fig.suptitle(suptitle, fontsize=12.5, fontweight='bold', y=0.972)

    fname = f'{FIGDIR}/spatial_{tag}_{INIT.replace("-","")}_{WEEK.replace(" ","")}.png'
    fig.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {os.path.basename(fname)}', flush=True)
    return fname

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'\nSpatial maps: INIT={INIT}  {WEEK}  ({date_str})', flush=True)
    os.makedirs(FIGDIR, exist_ok=True)
    f1 = build_figure('abs')
    f2 = build_figure('anom')
    print(f'\nDONE\n  {f1}\n  {f2}', flush=True)
