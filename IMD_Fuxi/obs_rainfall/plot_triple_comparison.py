"""
Triple comparison figures: FuXi | ERA5 | IMD ERFS
3 rows × 4 columns (anomaly only)
  Row 1: FuXi anomaly (normalised, 1.5°)
  Row 2: ERA5 observed anomaly (regridded to 1.5°)
  Row 3: IMD ERFS anomaly panel cropped from operational bulletin PNG

Saves:
  figures/fig_triple_comparison_IC0707.png
  figures/fig_triple_comparison_IC0825.png

Run IC0707 first — stops after saving so it can be reviewed before IC0825.
"""

import os, sys, warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
CACHE    = "/home/raj.ayush/s2s/s2s_anlysis/IMD_Fuxi/obs_rainfall/era5_obs_clim_cache.nc"
FIG_DIR  = "/home/raj.ayush/s2s/s2s_anlysis/IMD_Fuxi/obs_rainfall/figures"
ZARR     = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DATA_DIR = "/home/raj.ayush/s2s/s2s_anlysis/IMD_Fuxi/obs_rainfall/data"
FUXI0707 = "/tmp/fuxi_IC0707"
FUXI0825 = "/tmp/fuxi_IC0825"
os.makedirs(FIG_DIR, exist_ok=True)

LON0, LON1 = 66.5, 100.0
LAT0, LAT1 =  6.0,  38.0

# ── Colormap (same as existing figures) ────────────────────────────────────
anom_levels = [-20, -15, -10, -5, -2, 2, 5, 10, 15, 20]
anom_cmap   = plt.get_cmap('RdYlBu', len(anom_levels) + 2)
anom_norm   = mcolors.BoundaryNorm(anom_levels, anom_cmap.N, extend='both')

# ── Load ERA5 2021 ─────────────────────────────────────────────────────────
print("Loading ERA5 2021 daily tp …")
ds_era5   = xr.open_zarr(ZARR, storage_options={"token": "anon"})
tp_all    = ds_era5["total_precipitation"]
tp_h_2021 = tp_all.sel(
    time=slice("2021-07-08T00:00:00", "2021-09-22T23:00:00"),
    latitude=slice(LAT1, LAT0), longitude=slice(LON0, LON1))
tp_d_2021 = (tp_h_2021.resample(time="1D").sum("time") * 1000.0).compute()
lats_obs  = tp_d_2021.latitude.values
lons_obs  = tp_d_2021.longitude.values
print(f"  shape {tp_d_2021.shape}  lat {lats_obs[0]:.1f}→{lats_obs[-1]:.1f}")

def obs_weekly(start, end):
    return tp_d_2021.sel(time=slice(start, end)).mean("time").values

# ── Load clim cache ────────────────────────────────────────────────────────
print("Loading clim cache …")
ds_clim = xr.open_dataset(CACHE)
clim_lats = ds_clim.latitude.values
clim_lons = ds_clim.longitude.values

def get_clim(md_s, md_e):
    var = f"clim_{md_s.replace('-','')}_to_{md_e.replace('-','')}"
    return ds_clim[var].values if var in ds_clim else None

# ── Interpolation helper ───────────────────────────────────────────────────
def regrid(arr, src_lats, src_lons, dst_lats, dst_lons):
    """Bilinear regrid; src_lats may be descending."""
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        arr      = arr[::-1, :]
    fn = RegularGridInterpolator(
        (src_lats, src_lons), arr,
        method='linear', bounds_error=False, fill_value=np.nan)
    gg_lat, gg_lon = np.meshgrid(dst_lats, dst_lons, indexing='ij')
    return fn((gg_lat, gg_lon))

# ── FuXi loader ────────────────────────────────────────────────────────────
def load_fuxi_weekly(fuxi_dir, d_s, d_e):
    parts = []
    for d in range(d_s, d_e + 1):
        fn = os.path.join(fuxi_dir, f"{d:02d}.nc")
        if not os.path.exists(fn):
            continue
        da  = xr.open_dataset(fn)['__xarray_dataarray_variable__']
        arr = da.sel(channel='tp').squeeze().values
        f_lats = da.lat.values; f_lons = da.lon.values
        lm = (f_lats >= LAT0) & (f_lats <= LAT1)
        rm = (f_lons >= LON0) & (f_lons <= LON1)
        parts.append(arr[np.ix_(lm, rm)])
    mean     = np.nanmean(np.stack(parts, axis=0), axis=0)
    f_lats_c = da.lat.values[(da.lat.values >= LAT0) & (da.lat.values <= LAT1)]
    f_lons_c = da.lon.values[(da.lon.values >= LON0) & (da.lon.values <= LON1)]
    return mean, f_lats_c, f_lons_c

# ── Case definitions ───────────────────────────────────────────────────────
CASES = {
    "IC0707": {
        "title":    "Rainfall Anomaly: FuXi vs ERA5 vs IMD ERFS  |  IC: 20210707",
        "imd_png":  os.path.join(DATA_DIR, "rfactual_rfanom_MME2021070700.png"),
        "outfile":  os.path.join(FIG_DIR,  "fig_triple_comparison_IC0707.png"),
        "weeks": [
            {"label": "Week 1: 08–14Jul",    "start": "2021-07-08", "end": "2021-07-14",
             "clim": ("07-08","07-14"), "fuxi_days": (1,  7)},
            {"label": "Week 2: 15–21Jul",    "start": "2021-07-15", "end": "2021-07-21",
             "clim": ("07-15","07-21"), "fuxi_days": (8, 14)},
            {"label": "Week 3: 22–28Jul",    "start": "2021-07-22", "end": "2021-07-28",
             "clim": ("07-22","07-28"), "fuxi_days": (15,21)},
            {"label": "Week 4: 29Jul–04Aug", "start": "2021-07-29", "end": "2021-08-04",
             "clim": ("07-29","08-04"), "fuxi_days": (22,28)},
        ],
        "fuxi_dir": FUXI0707,
    },
    "IC0825": {
        "title":    "Rainfall Anomaly: FuXi vs ERA5 vs IMD ERFS  |  IC: 20210825",
        "imd_png":  os.path.join(DATA_DIR, "rfactual_rfanom_MME2021082500.png"),
        "outfile":  os.path.join(FIG_DIR,  "fig_triple_comparison_IC0825.png"),
        "weeks": [
            {"label": "Week 1: 26Aug–01Sep", "start": "2021-08-26", "end": "2021-09-01",
             "clim": ("08-26","09-01"), "fuxi_days": (1,  7)},
            {"label": "Week 2: 02–08Sep",    "start": "2021-09-02", "end": "2021-09-08",
             "clim": ("09-02","09-08"), "fuxi_days": (8, 14)},
            {"label": "Week 3: 09–15Sep",    "start": "2021-09-09", "end": "2021-09-15",
             "clim": ("09-09","09-15"), "fuxi_days": (15,21)},
            {"label": "Week 4: 16–22Sep",    "start": "2021-09-16", "end": "2021-09-22",
             "clim": ("09-16","09-22"), "fuxi_days": (22,28)},
        ],
        "fuxi_dir": FUXI0825,
    },
}

# ── Map decoration (rows 1-2) ──────────────────────────────────────────────
def decorate(ax, title, title_color):
    ax.set_extent([LON0, LON1, LAT0, LAT1], crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                     edgecolor='black', facecolor='none'),
        linewidth=0.8, zorder=4)
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m',
                                     edgecolor='black', facecolor='none'),
        linewidth=0.4, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.5, linestyle=':')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 7, 'weight': 'bold'}
    gl.ylabel_style = {'size': 7, 'weight': 'bold'}
    ax.set_title(title, fontsize=8.5, color=title_color, fontweight='bold', pad=3)

# ── Figure maker ───────────────────────────────────────────────────────────
def make_triple(ic):
    cfg   = CASES[ic]
    weeks = cfg["weeks"]

    # ── compute arrays ────────────────────────────────────────────────
    fuxi_anoms = []
    era5_anoms = []
    fuxi_lats = fuxi_lons = None

    for wk in weeks:
        # ERA5 obs anomaly
        act_e = obs_weekly(wk["start"], wk["end"])
        c_raw = get_clim(*wk["clim"])
        if c_raw is not None:
            if c_raw.shape == act_e.shape:
                clim_e = c_raw
            else:
                clim_e = regrid(c_raw, clim_lats, clim_lons, lats_obs, lons_obs)
        else:
            clim_e = np.zeros_like(act_e)
        anom_e = act_e - clim_e

        # FuXi anomaly (normalised)
        act_f, f_lats, f_lons = load_fuxi_weekly(cfg["fuxi_dir"], *wk["fuxi_days"])
        fuxi_lats = f_lats; fuxi_lons = f_lons

        lat_m = (lats_obs >= LAT0) & (lats_obs <= LAT1)
        lon_m = (lons_obs >= LON0) & (lons_obs <= LON1)
        scale = act_e[np.ix_(lat_m, lon_m)].mean() / act_f.mean() if act_f.mean() > 0 else 1.0
        act_f_sc = act_f * scale

        clim_f = regrid(clim_e, lats_obs, lons_obs, f_lats, f_lons) if c_raw is not None \
                 else np.zeros((len(f_lats), len(f_lons)))
        anom_f = act_f_sc - clim_f

        # ERA5 anom regridded to FuXi 1.5° grid
        anom_e_on_fuxi = regrid(anom_e, lats_obs, lons_obs, f_lats, f_lons)

        fuxi_anoms.append(anom_f)
        era5_anoms.append(anom_e_on_fuxi)

    # ── load & crop IMD PNG ───────────────────────────────────────────
    imd_path = cfg["imd_png"]
    img      = imread(imd_path)
    h, w     = img.shape[:2]
    print(f"\n{ic} IMD PNG: shape={img.shape}  h={h} w={w}")
    img_anom = img[:, w//2:]               # right half = anomaly
    img_anom = img_anom[int(h*0.07):, :]  # trim title bar
    print(f"  cropped anomaly panel: {img_anom.shape}")

    # ── build figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(cfg["title"], fontsize=13, fontweight='bold', y=0.975)
    fig.text(0.5, 0.955,
             "ERA5 regridded from native 0.25° to 1.5° for direct comparison with FuXi-S2S",
             ha='center', fontsize=9, style='italic', color='dimgray')

    # GridSpec for rows 0-1 only (maps); row 2 (IMD image) placed manually below
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.28, wspace=0.08,
                           top=0.93, bottom=0.46,
                           left=0.07, right=0.97)

    row_meta = [
        ("FuXi Anomaly\n(normalised, 1.5°)",       fuxi_anoms, 'darkgreen'),
        ("ERA5 Observed\n(regridded to 1.5°)",      era5_anoms, 'darkblue'),
    ]

    im = None
    row_axes = []   # row_axes[row][col]

    for row, (row_lbl, arrays, col) in enumerate(row_meta):
        this_row = []
        for c, (arr, wk) in enumerate(zip(arrays, weeks)):
            ax = fig.add_subplot(gs[row, c], projection=ccrs.PlateCarree())
            this_row.append(ax)
            im = ax.pcolormesh(fuxi_lons, fuxi_lats, arr,
                               cmap=anom_cmap, norm=anom_norm,
                               transform=ccrs.PlateCarree(), shading='auto')
            decorate(ax, f"({wk['label']})", col)
        row_axes.append(this_row)

        # row label: attach to the first-column axis (already created, no duplicate)
        this_row[0].text(-0.22, 0.5, row_lbl,
                         transform=this_row[0].transAxes,
                         fontsize=9, color=col, fontweight='bold',
                         va='center', ha='center', rotation=90)

    # ── Row 3: IMD image — left-aligned with grid, natural aspect ────
    # Image is nearly square (279×306). At figsize(20,20) we can show it large.
    # Left-align with gs.left=0.07 so it lines up with the map columns.
    ih, iw   = img_anom.shape[:2]
    img_disp_h = 0.36                        # 0.36 × 20" = 7.2" tall
    img_disp_w = img_disp_h * (iw / ih)     # preserve aspect → 0.328 × 20" = 6.56" wide
    img_left   = 0.07                        # left-aligned with the map grid
    img_bottom = 0.05

    ax_imd = fig.add_axes([img_left, img_bottom, img_disp_w, img_disp_h])
    ax_imd.imshow(img_anom, aspect='auto', interpolation='lanczos')
    ax_imd.axis('off')

    # Row label to the left of the image axis
    ax_imd.text(-0.12, 0.5,
                "IMD ERFS\n(operational bulletin)",
                transform=ax_imd.transAxes,
                fontsize=9, color='darkred', fontweight='bold',
                va='center', ha='center', rotation=90)

    # Caption below the image
    ax_imd.text(0.5, -0.07,
                "IMD ERFS from operational extended range forecast bulletin "
                "(Pattanaik et al., 2019);  "
                "gridded forecast data unavailable.  "
                "Colorbar scale: −20 to +20 mm/day.",
                transform=ax_imd.transAxes,
                fontsize=7.5, style='italic', color='dimgray',
                ha='center', va='top')

    # ── Shared colourbar below row 1 (maps), above IMD image ─────────
    r0c0 = row_axes[0][0].get_position()
    r1c3 = row_axes[1][3].get_position()
    cbar_left   = r0c0.x0
    cbar_width  = r1c3.x1 - r0c0.x0
    cbar_bottom = r1c3.y0 - 0.045
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, 0.017])
    cb  = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    cb.set_label('mm/day', fontsize=9)
    cb.set_ticks(anom_levels)
    cb.ax.tick_params(labelsize=8)

    fig.savefig(cfg["outfile"], dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {cfg['outfile']}")

# ── Run IC0707 first, then stop ────────────────────────────────────────────
make_triple("IC0707")
make_triple("IC0825")
print("\nBoth figures done.")
