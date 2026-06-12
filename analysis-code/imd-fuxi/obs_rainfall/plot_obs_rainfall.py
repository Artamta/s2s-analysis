"""
Observed ERA5 rainfall figures matching IMD ERF plot style, for two ICs.

IC 2021-07-07:  Wk1=08-14Jul  Wk2=15-21Jul  Wk3=22-28Jul  Wk4=29Jul-04Aug
IC 2021-08-25:  Wk1=26Aug-01Sep  Wk2=02-08Sep  Wk3=09-15Sep  Wk4=16-22Sep

Saves: figures/fig_obs_IC0707.png  figures/fig_obs_IC0825.png
"""

import os, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

FIG_DIR  = "/home/raj.ayush/s2s/s2s_anlysis/figures"
ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
os.makedirs(FIG_DIR, exist_ok=True)

LAT0, LAT1     = 5.0, 38.0   # domain
LON0, LON1     = 66.0, 100.0
CLIM_YEARS     = list(range(2001, 2021))

# ── All unique calendar-week windows needed ───────────────────────────────────
# (md_start, md_end) — day-of-year windows shared across both ICs
WEEK_DEFS = [
    ("07-08", "07-14"),
    ("07-15", "07-21"),
    ("07-22", "07-28"),
    ("07-29", "08-04"),
    ("08-26", "09-01"),
    ("09-02", "09-08"),
    ("09-09", "09-15"),
    ("09-16", "09-22"),
]

# ── Open ERA5 store ────────────────────────────────────────────────────────────
CACHE_FILE = "/home/raj.ayush/s2s/s2s_anlysis/IMD_Fuxi/obs_rainfall/era5_obs_clim_cache.nc"

print("Opening ERA5 zarr store …")
ds_era5  = xr.open_zarr(ZARR_URL, storage_options={"token": "anon"})
tp_all   = ds_era5["total_precipitation"]
lat_slc  = slice(LAT1, LAT0)   # ERA5 lat descending
lon_slc  = slice(LON0, LON1)


# ── Build the full list of date ranges we need to pull ────────────────────────
def make_date_ranges(md_s, md_e, years):
    """Return list of (start_str, end_str) for each year, handling Aug→Sep cross."""
    ranges = []
    ms, ds_ = int(md_s[:2]), int(md_s[3:])
    me, de  = int(md_e[:2]), int(md_e[3:])
    for yr in years:
        s = f"{yr}-{md_s}"
        # year-wrap only for 07-29→08-04 case (month_end < month_start)
        e_yr = yr + 1 if me < ms else yr
        e = f"{e_yr}-{md_e}"
        ranges.append((s, e))
    return ranges


def pull_daily_mmday(start, end):
    """Hourly ERA5 tp [start, end] → daily mm/day, spatially subsetted."""
    ts  = slice(f"{start}T00:00:00", f"{end}T23:00:00")
    tp_h = tp_all.sel(time=ts, latitude=lat_slc, longitude=lon_slc)
    tp_d = tp_h.resample(time="1D").sum("time") * 1000.0   # m→mm
    return tp_d.compute()


# ── Pre-fetch 2021 daily data for all weeks used across both ICs ─────────────
# Just pull the whole JJAS 2021 at once — it's 4 months of hourly India domain
print("Fetching JJAS 2021 ERA5 tp (India domain) …")
ts_2021 = slice("2021-07-08T00:00:00", "2021-09-22T23:00:00")
tp_h_2021 = tp_all.sel(time=ts_2021, latitude=lat_slc, longitude=lon_slc)
tp_d_2021 = (tp_h_2021.resample(time="1D").sum("time") * 1000.0).compute()
print(f"  2021 daily tp: {tp_d_2021.shape}")

lats = tp_d_2021.latitude.values
lons = tp_d_2021.longitude.values


def weekly_act(start, end):
    return tp_d_2021.sel(time=slice(start, end)).mean("time").values


# ── Pre-fetch climatology for each unique week window ─────────────────────────
# Pull each clim year in one hourly block per year to avoid 20 separate connections
# Strategy: for each of the 8 week windows, pull all 20 years as one batch.
# We fetch per-year because the time coord is continuous in the zarr.

clim_cache = {}   # key = (md_s, md_e) → 2-D numpy array

if os.path.exists(CACHE_FILE):
    print(f"\nLoading clim from cache: {CACHE_FILE}")
    ds_cache = xr.open_dataset(CACHE_FILE)
    for md_s, md_e in WEEK_DEFS:
        var = f"clim_{md_s.replace('-','')}_to_{md_e.replace('-','')}"
        if var in ds_cache:
            clim_cache[(md_s, md_e)] = ds_cache[var].values
    ds_cache.close()
    print("  cache loaded.")
else:
    print("\nFetching 2001-2020 climatology for all week windows …")
    clim_daily = {}
    for yr in CLIM_YEARS:
        s_str = f"{yr}-07-08T00:00:00"
        e_str = f"{yr}-09-22T23:00:00"
        print(f"  clim {yr} …", end=" ", flush=True)
        tp_h_yr = tp_all.sel(time=slice(s_str, e_str), latitude=lat_slc, longitude=lon_slc)
        tp_d_yr = (tp_h_yr.resample(time="1D").sum("time") * 1000.0).compute()
        clim_daily[yr] = tp_d_yr
        print(f"{tp_d_yr.shape}")
    print("  clim download complete.")

    ref_da = list(clim_daily.values())[0]
    lats_c = ref_da.latitude.values
    lons_c = ref_da.longitude.values

    for md_s, md_e in WEEK_DEFS:
        ms, ds_ = int(md_s[:2]), int(md_s[3:])
        me, de  = int(md_e[:2]), int(md_e[3:])
        parts = []
        for yr in CLIM_YEARS:
            s = f"{yr}-{md_s}"
            e_yr = yr + 1 if me < ms else yr
            e = f"{e_yr}-{md_e}"
            try:
                w = clim_daily[yr].sel(time=slice(s, e))
                if len(w.time) > 0:
                    parts.append(w.mean("time").values)
            except Exception as ex:
                print(f"  skip {s}→{e}: {ex}")
        clim_cache[(md_s, md_e)] = np.nanmean(np.stack(parts, axis=0), axis=0) if parts else None

    print("  clim means computed. Saving cache …")
    cache_vars = {}
    for (md_s, md_e), arr in clim_cache.items():
        if arr is not None:
            var = f"clim_{md_s.replace('-','')}_to_{md_e.replace('-','')}"
            cache_vars[var] = xr.DataArray(arr, dims=["latitude", "longitude"],
                                           coords={"latitude": lats_c, "longitude": lons_c})
    xr.Dataset(cache_vars).to_netcdf(CACHE_FILE)
    print(f"  cache saved: {CACHE_FILE}")


# ── Case definitions ───────────────────────────────────────────────────────────
CASES = {
    "IC0707": {
        "suptitle": "Observed Actual & Anomaly Rainfall (mm/day)  |  IC: 20210707",
        "weeks": [
            {"label": "Week 1: 08Jul–14Jul",  "start": "2021-07-08", "end": "2021-07-14",
             "clim": ("07-08", "07-14")},
            {"label": "Week 2: 15Jul–21Jul",  "start": "2021-07-15", "end": "2021-07-21",
             "clim": ("07-15", "07-21")},
            {"label": "Week 3: 22Jul–28Jul",  "start": "2021-07-22", "end": "2021-07-28",
             "clim": ("07-22", "07-28")},
            {"label": "Week 4: 29Jul–04Aug",  "start": "2021-07-29", "end": "2021-08-04",
             "clim": ("07-29", "08-04")},
        ],
        "outfile": os.path.join(FIG_DIR, "fig_obs_IC0707.png"),
    },
    "IC0825": {
        "suptitle": "Observed Actual & Anomaly Rainfall (mm/day)  |  IC: 20210825",
        "weeks": [
            {"label": "Week 1: 26Aug–01Sep", "start": "2021-08-26", "end": "2021-09-01",
             "clim": ("08-26", "09-01")},
            {"label": "Week 2: 02Sep–08Sep", "start": "2021-09-02", "end": "2021-09-08",
             "clim": ("09-02", "09-08")},
            {"label": "Week 3: 09Sep–15Sep", "start": "2021-09-09", "end": "2021-09-15",
             "clim": ("09-09", "09-15")},
            {"label": "Week 4: 16Sep–22Sep", "start": "2021-09-16", "end": "2021-09-22",
             "clim": ("09-16", "09-22")},
        ],
        "outfile": os.path.join(FIG_DIR, "fig_obs_IC0825.png"),
    },
}


# ── Colormaps ─────────────────────────────────────────────────────────────────
act_levels  = [0, 2, 5, 10, 20, 40, 60]
act_cmap    = plt.get_cmap('YlGn', len(act_levels))      # 7 bins (6 intervals + 1 over)
act_norm    = mcolors.BoundaryNorm(act_levels, act_cmap.N, extend='max')

anom_levels = [-20, -15, -10, -5, -2, 2, 5, 10, 15, 20]
anom_cmap   = plt.get_cmap('RdYlBu_r', len(anom_levels) + 2)  # 9 intervals + 2 extensions
anom_norm   = mcolors.BoundaryNorm(anom_levels, anom_cmap.N, extend='both')


# ── Map decoration ─────────────────────────────────────────────────────────────
def decorate(ax, title):
    ax.set_extent([LON0, LON1, LAT0, LAT1], crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '10m',
                                     edgecolor='black', facecolor='none'),
        linewidth=0.8, zorder=4)
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m',
                                     edgecolor='black', facecolor='none'),
        linewidth=0.4, zorder=4)
    ax.set_title(title, fontsize=9, color='blue', fontweight='bold', pad=4)


# ── Plot ──────────────────────────────────────────────────────────────────────
def make_figure(case_key):
    cfg   = CASES[case_key]
    weeks = cfg["weeks"]
    n     = len(weeks)

    fig, axes = plt.subplots(
        n, 2, figsize=(14, 18),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )

    im_act = im_anom = None

    for row, wk in enumerate(weeks):
        act  = weekly_act(wk['start'], wk['end'])
        clim = clim_cache.get(wk['clim'])
        if clim is None:
            print(f"  WARNING: no clim for {wk['clim']}, anomaly=0")
            clim = np.zeros_like(act)
        anom = act - clim

        label = wk['label']

        ax_a = axes[row, 0]
        im_act = ax_a.pcolormesh(lons, lats, act,
                                 cmap=act_cmap, norm=act_norm,
                                 transform=ccrs.PlateCarree(), shading='auto')
        decorate(ax_a, label)

        ax_n = axes[row, 1]
        im_anom = ax_n.pcolormesh(lons, lats, anom,
                                  cmap=anom_cmap, norm=anom_norm,
                                  transform=ccrs.PlateCarree(), shading='auto')
        decorate(ax_n, label)

    # column headers — override row-0 titles
    axes[0, 0].set_title(f"Actual Rainfall  —  {weeks[0]['label']}",
                         fontsize=9, color='blue', fontweight='bold', pad=4)
    axes[0, 1].set_title(f"Rainfall Anomaly  —  {weeks[0]['label']}",
                         fontsize=9, color='blue', fontweight='bold', pad=4)

    fig.suptitle(cfg["suptitle"], fontsize=12, fontweight='bold', y=1.005)
    plt.tight_layout(pad=1.5, h_pad=0.8, w_pad=0.5)

    p0 = axes[-1, 0].get_position()
    cax_act = fig.add_axes([p0.x0, p0.y0 - 0.045, p0.width, 0.018])
    cb_act  = fig.colorbar(im_act, cax=cax_act, orientation='horizontal', extend='max')
    cb_act.set_label('mm/day', fontsize=9)
    cb_act.set_ticks(act_levels)
    cb_act.ax.tick_params(labelsize=8)

    p1 = axes[-1, 1].get_position()
    cax_anom = fig.add_axes([p1.x0, p1.y0 - 0.045, p1.width, 0.018])
    cb_anom  = fig.colorbar(im_anom, cax=cax_anom, orientation='horizontal', extend='both')
    cb_anom.set_label('mm/day', fontsize=9)
    cb_anom.set_ticks(anom_levels)
    cb_anom.ax.tick_params(labelsize=8)

    out = cfg["outfile"]
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {out}")


print("\n=== Figure 1: IC 2021-07-07 ===")
make_figure("IC0707")

print("\n=== Figure 2: IC 2021-08-25 ===")
make_figure("IC0825")

print("\nAll done.")
