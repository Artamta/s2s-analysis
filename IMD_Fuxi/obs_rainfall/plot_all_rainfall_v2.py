"""
Three figures, matching the IMD ERF layout exactly:

Layout per figure: 2 rows × 4 columns
  Row 1: Week 1 | Week 2 | Week 3 | Week 4   (all ACTUAL)
  Row 2: Week 1 | Week 2 | Week 3 | Week 4   (all ANOMALY)

Figures:
  fig_obs_IC0707_v2.png    — ERA5 observed, IC 20210707
  fig_obs_IC0825_v2.png    — ERA5 observed, IC 20210825
  fig_fuxi_raw_IC0707.png  — FuXi forecast (raw mm), IC 20210707
  fig_fuxi_raw_IC0825.png  — FuXi forecast (raw mm), IC 20210825
  fig_fuxi_norm_IC0707.png — FuXi forecast (normalised to ERA5 India mean), IC 20210707
  fig_fuxi_norm_IC0825.png — FuXi forecast (normalised to ERA5 India mean), IC 20210825

Anomaly colormap: RdYlBu  (blue = excess/surplus, orange/red = deficit)
  This matches IMD ERF convention.

Actual colormap: YlGn with IMD-style levels [0,2,5,10,20,40,60]
"""

import os, warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

CACHE   = "/home/raj.ayush/s2s/s2s_anlysis/IMD_Fuxi/obs_rainfall/era5_obs_clim_cache.nc"
FIG_DIR = "/home/raj.ayush/s2s/s2s_anlysis/IMD_Fuxi/obs_rainfall/figures"
ZARR    = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
FUXI0707 = "/tmp/fuxi_IC0707"
FUXI0825 = "/tmp/fuxi_IC0825"
os.makedirs(FIG_DIR, exist_ok=True)

# IMD domain (tight India)
LON0, LON1 = 66.5, 100.0
LAT0, LAT1 =  6.0,  38.0

# ── Colormaps ──────────────────────────────────────────────────────────────
act_levels  = [0, 2, 5, 10, 20, 40, 60]
act_cmap    = plt.get_cmap('YlGn', len(act_levels))
act_norm    = mcolors.BoundaryNorm(act_levels, act_cmap.N, extend='max')

anom_levels = [-20, -15, -10, -5, -2, 2, 5, 10, 15, 20]
# RdYlBu (NOT _r): blue=positive/surplus, red=negative/deficit  → matches IMD ERF
anom_cmap   = plt.get_cmap('RdYlBu', len(anom_levels) + 2)
anom_norm   = mcolors.BoundaryNorm(anom_levels, anom_cmap.N, extend='both')


# ── Map decoration ─────────────────────────────────────────────────────────
def decorate(ax, title, row):
    ax.set_extent([LON0, LON1, LAT0, LAT1], crs=ccrs.PlateCarree())
    # Coastline only — no political borders, no disputed regions
    ax.add_feature(
        cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                     edgecolor='black', facecolor='none'),
        linewidth=0.8, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.6, linestyle=':')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7, 'weight': 'bold'}
    gl.ylabel_style = {'size': 7, 'weight': 'bold'}
    color = 'darkblue' if row == 0 else 'darkred'
    ax.set_title(title, fontsize=8.5, color=color, fontweight='bold', pad=3)


# ── Generic 2-row × 4-col figure maker ────────────────────────────────────
def make_2x4_figure(act_arrays, anom_arrays, lats, lons,
                    week_labels, suptitle, outfile,
                    act_label="ERA5 Actual Rainfall (mm/day)",
                    anom_label="ERA5 Rainfall Anomaly (mm/day)",
                    act_lvls=None, act_c=None, act_n=None,
                    anom_lvls=None, anom_c=None, anom_n=None):
    """
    act_arrays  : list of 4 2-D numpy arrays (actual, one per week)
    anom_arrays : list of 4 2-D numpy arrays (anomaly, one per week)
    week_labels : list of 4 strings
    """
    if act_lvls  is None: act_lvls  = act_levels
    if act_c     is None: act_c     = act_cmap
    if act_n     is None: act_n     = act_norm
    if anom_lvls is None: anom_lvls = anom_levels
    if anom_c    is None: anom_c    = anom_cmap
    if anom_n    is None: anom_n    = anom_norm

    fig, axes = plt.subplots(
        2, 4, figsize=(18, 9),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    fig.subplots_adjust(hspace=0.25, wspace=0.08, top=0.90, bottom=0.10,
                        left=0.04, right=0.97)

    im_act = im_anom = None
    for col, (act, anom, lbl) in enumerate(zip(act_arrays, anom_arrays, week_labels)):
        ax_a = axes[0, col]
        im_act  = ax_a.pcolormesh(lons, lats, act,
                                  cmap=act_c, norm=act_n,
                                  transform=ccrs.PlateCarree(), shading='auto')
        decorate(ax_a, f"({lbl})", row=0)

        ax_n = axes[1, col]
        im_anom = ax_n.pcolormesh(lons, lats, anom,
                                  cmap=anom_c, norm=anom_n,
                                  transform=ccrs.PlateCarree(), shading='auto')
        decorate(ax_n, f"({lbl})", row=1)

    # Row labels on left edge
    axes[0, 0].text(-0.18, 0.5, act_label,
                    transform=axes[0, 0].transAxes,
                    fontsize=10, color='darkred', fontweight='bold',
                    va='center', rotation=90)
    axes[1, 0].text(-0.18, 0.5, anom_label,
                    transform=axes[1, 0].transAxes,
                    fontsize=10, color='darkblue', fontweight='bold',
                    va='center', rotation=90)

    fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=0.96)

    # Colorbars
    # Actual — below row 0
    p00 = axes[0, 0].get_position(); p03 = axes[0, 3].get_position()
    cax_act = fig.add_axes([p00.x0, p00.y0 - 0.055,
                            p03.x1 - p00.x0, 0.018])
    cb_act = fig.colorbar(im_act, cax=cax_act, orientation='horizontal', extend='max')
    cb_act.set_label('mm/day', fontsize=9)
    cb_act.set_ticks(act_lvls)
    cb_act.ax.tick_params(labelsize=8)

    # Anomaly — below row 1
    p10 = axes[1, 0].get_position(); p13 = axes[1, 3].get_position()
    cax_anom = fig.add_axes([p10.x0, p10.y0 - 0.055,
                             p13.x1 - p10.x0, 0.018])
    cb_anom = fig.colorbar(im_anom, cax=cax_anom, orientation='horizontal', extend='both')
    cb_anom.set_label('mm/day', fontsize=9)
    cb_anom.set_ticks(anom_lvls)
    cb_anom.ax.tick_params(labelsize=8)

    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {outfile}")


# ── Load ERA5 observed data ────────────────────────────────────────────────
print("Loading ERA5 2021 daily tp from zarr …")
ds_era5   = xr.open_zarr(ZARR, storage_options={"token": "anon"})
tp_all    = ds_era5["total_precipitation"]
lat_slc   = slice(LAT1, LAT0)   # descending
lon_slc   = slice(LON0, LON1)

tp_h_2021 = tp_all.sel(
    time=slice("2021-07-08T00:00:00", "2021-09-22T23:00:00"),
    latitude=lat_slc, longitude=lon_slc)
tp_d_2021 = (tp_h_2021.resample(time="1D").sum("time") * 1000.0).compute()
lats_obs  = tp_d_2021.latitude.values
lons_obs  = tp_d_2021.longitude.values
print(f"  2021 daily tp: {tp_d_2021.shape}  "
      f"lat {lats_obs[0]:.1f}→{lats_obs[-1]:.1f}  lon {lons_obs[0]:.1f}→{lons_obs[-1]:.1f}")

def obs_weekly(start, end):
    return tp_d_2021.sel(time=slice(start, end)).mean("time").values


# ── Load climatology cache ─────────────────────────────────────────────────
print("Loading clim cache …")
ds_clim = xr.open_dataset(CACHE)
def get_clim(md_s, md_e):
    var = f"clim_{md_s.replace('-','')}_to_{md_e.replace('-','')}"
    return ds_clim[var].values if var in ds_clim else None


# ── Case definitions ───────────────────────────────────────────────────────
CASES = {
    "IC0707": {
        "suptitle_obs":  "ERA5 Observed Rainfall (mm/day)  |  IC: 20210707",
        "weeks": [
            {"label": "Week 1: 08–14Jul",  "start": "2021-07-08", "end": "2021-07-14",
             "clim": ("07-08", "07-14")},
            {"label": "Week 2: 15–21Jul",  "start": "2021-07-15", "end": "2021-07-21",
             "clim": ("07-15", "07-21")},
            {"label": "Week 3: 22–28Jul",  "start": "2021-07-22", "end": "2021-07-28",
             "clim": ("07-22", "07-28")},
            {"label": "Week 4: 29Jul–04Aug", "start": "2021-07-29", "end": "2021-08-04",
             "clim": ("07-29", "08-04")},
        ],
        "fuxi_dir": FUXI0707,
        "fuxi_days": [(1,7), (8,14), (15,21), (22,28)],   # (start_day, end_day) per week
    },
    "IC0825": {
        "suptitle_obs":  "ERA5 Observed Rainfall (mm/day)  |  IC: 20210825",
        "weeks": [
            {"label": "Week 1: 26Aug–01Sep", "start": "2021-08-26", "end": "2021-09-01",
             "clim": ("08-26", "09-01")},
            {"label": "Week 2: 02–08Sep",   "start": "2021-09-02", "end": "2021-09-08",
             "clim": ("09-02", "09-08")},
            {"label": "Week 3: 09–15Sep",   "start": "2021-09-09", "end": "2021-09-15",
             "clim": ("09-09", "09-15")},
            {"label": "Week 4: 16–22Sep",   "start": "2021-09-16", "end": "2021-09-22",
             "clim": ("09-16", "09-22")},
        ],
        "fuxi_dir": FUXI0825,
        "fuxi_days": [(1,7), (8,14), (15,21), (22,28)],
    },
}


# ── FuXi loader ───────────────────────────────────────────────────────────
def load_fuxi_weekly(fuxi_dir, day_start, day_end, lat0, lat1, lon0, lon1):
    """
    Load FuXi tp for days [day_start..day_end], crop to domain, return weekly mean mm/day.
    FuXi tp output is in mm (ERA5 tp was scaled ×1000 on input, so output is ~mm).
    """
    parts = []
    for d in range(day_start, day_end + 1):
        fn = os.path.join(fuxi_dir, f"{d:02d}.nc")
        if not os.path.exists(fn):
            continue
        ds  = xr.open_dataset(fn)
        da  = ds['__xarray_dataarray_variable__']
        arr = da.sel(channel='tp').squeeze().values   # (lat, lon)
        # crop to India domain
        f_lats = da.lat.values; f_lons = da.lon.values
        lat_m  = (f_lats >= lat0) & (f_lats <= lat1)
        lon_m  = (f_lons >= lon0) & (f_lons <= lon1)
        parts.append(arr[np.ix_(lat_m, lon_m)])
    mean = np.nanmean(np.stack(parts, axis=0), axis=0)
    # coords for plotting
    f_lats_c = da.lat.values[(da.lat.values >= lat0) & (da.lat.values <= lat1)]
    f_lons_c = da.lon.values[(da.lon.values >= lon0) & (da.lon.values <= lon1)]
    return mean, f_lats_c, f_lons_c


# FuXi anomaly clim: use ERA5 clim interpolated to FuXi 1.5° grid
def interp_clim_to_fuxi(clim_arr, obs_lats, obs_lons, fuxi_lats, fuxi_lons):
    """Bilinear interpolation of ERA5 clim (0.25°) to FuXi grid (1.5°)."""
    from scipy.interpolate import RegularGridInterpolator
    # obs_lats may be descending; RegularGridInterpolator needs ascending
    if obs_lats[0] > obs_lats[-1]:
        obs_lats_a = obs_lats[::-1]
        clim_a     = clim_arr[::-1, :]
    else:
        obs_lats_a = obs_lats
        clim_a     = clim_arr
    interp = RegularGridInterpolator(
        (obs_lats_a, obs_lons), clim_a,
        method='linear', bounds_error=False, fill_value=np.nan)
    gg_lat, gg_lon = np.meshgrid(fuxi_lats, fuxi_lons, indexing='ij')
    return interp((gg_lat, gg_lon))


# ── MAIN LOOP ──────────────────────────────────────────────────────────────
for ic, cfg in CASES.items():
    weeks     = cfg["weeks"]
    fuxi_dir  = cfg["fuxi_dir"]
    week_lbls = [w["label"] for w in weeks]

    # ── 1. ERA5 obs arrays ──────────────────────────────────────────────
    act_obs  = [obs_weekly(w["start"], w["end"]) for w in weeks]
    clims_raw = [get_clim(*w["clim"]) for w in weeks]
    # Clim was cached on a slightly different grid (old domain). Regrid to obs grid.
    clims = []
    for c in clims_raw:
        if c is None:
            clims.append(None)
            continue
        if c.shape == act_obs[0].shape:
            clims.append(c)
        else:
            # clim grid: load from cache coords
            clim_lats = ds_clim.latitude.values
            clim_lons = ds_clim.longitude.values
            clims.append(interp_clim_to_fuxi(c, clim_lats, clim_lons, lats_obs, lons_obs))
    anom_obs = [a - (c if c is not None else np.zeros_like(a))
                for a, c in zip(act_obs, clims)]

    print(f"\n=== {ic}: obs figure ===")
    make_2x4_figure(
        act_obs, anom_obs, lats_obs, lons_obs, week_lbls,
        suptitle = cfg["suptitle_obs"],
        outfile  = os.path.join(FIG_DIR, f"fig_obs_{ic}_v2.png"),
        act_label  = "Actual Rainfall (mm/day)",
        anom_label = "Rainfall Anomaly (mm/day)",
    )

    # ── 2. Load FuXi weekly arrays (needed for normalised figure) ─────
    fuxi_acts   = []
    fuxi_lats_c = fuxi_lons_c = None

    for d_s, d_e in cfg["fuxi_days"]:
        act_f, f_lats, f_lons = load_fuxi_weekly(
            fuxi_dir, d_s, d_e, LAT0, LAT1, LON0, LON1)
        fuxi_lats_c = f_lats
        fuxi_lons_c = f_lons
        fuxi_acts.append(act_f)

    # ── 3. FuXi normalised figure ─────────────────────────────────────
    print(f"=== {ic}: FuXi normalised figure ===")
    # Scale factor per week: ERA5 India mean / FuXi India mean
    fuxi_norm_acts  = []
    fuxi_norm_anoms = []
    lat_m_obs = (lats_obs >= LAT0) & (lats_obs <= LAT1)
    lon_m_obs = (lons_obs >= LON0) & (lons_obs <= LON1)

    for wk_i, (act_f, act_e) in enumerate(zip(fuxi_acts, act_obs)):
        era5_india_mean = act_e[np.ix_(lat_m_obs, lon_m_obs)].mean()
        fuxi_india_mean = act_f.mean()
        scale = era5_india_mean / fuxi_india_mean if fuxi_india_mean > 0 else 1.0
        print(f"  Week {wk_i+1}: ERA5_mean={era5_india_mean:.2f}  FuXi_mean={fuxi_india_mean:.4f}  scale={scale:.1f}×")
        act_scaled  = act_f * scale
        fuxi_norm_acts.append(act_scaled)

        clim_f = clims[wk_i]
        if clim_f is not None:
            clim_interp = interp_clim_to_fuxi(clim_f, lats_obs, lons_obs, fuxi_lats_c, fuxi_lons_c)
            fuxi_norm_anoms.append(act_scaled - clim_interp)
        else:
            fuxi_norm_anoms.append(np.zeros_like(act_scaled))

    make_2x4_figure(
        fuxi_norm_acts, fuxi_norm_anoms, fuxi_lats_c, fuxi_lons_c, week_lbls,
        suptitle   = f"FuXi-S2S Forecast Rainfall (mm/day)  |  IC: 2021{ic[2:]}  [normalised to ERA5 India mean]",
        outfile    = os.path.join(FIG_DIR, f"fig_fuxi_norm_{ic}.png"),
        act_label  = "FuXi Actual Rainfall (mm/day) — normalised",
        anom_label = "FuXi Rainfall Anomaly (mm/day) — normalised",
    )

print("\nAll 4 figures done.")
