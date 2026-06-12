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

# ── Colormap — matched to IMD ERFS operational bulletin ────────────────────
# Levels: [-20, -15, -10, -5, -2, 2, 5, 10, 15, 20]  →  9 interior bins
# With extend='both' → 11 total color bins (1 under + 9 interior + 1 over)
anom_levels = [-20, -15, -10, -5, -2, 2, 5, 10, 15, 20]
_imd_bin_colors = [
    # 11 bins total: under, 9 interior, over
    (0.600, 0.000, 0.000),   # < −20   very dark red  (under)
    (0.839, 0.047, 0.000),   # −20→−15 dark red
    (1.000, 0.322, 0.000),   # −15→−10 red-orange
    (1.000, 0.557, 0.114),   # −10→ −5 orange
    (1.000, 0.792, 0.349),   #  −5→ −2 light orange
    (1.000, 0.957, 0.647),   #  −2→ +2 pale yellow / near-white
    (0.784, 0.784, 0.914),   #  +2→ +5 lavender
    (0.549, 0.549, 0.749),   #  +5→+10 medium blue-lavender
    (0.396, 0.396, 0.647),   # +10→+15 medium blue
    (0.235, 0.235, 0.529),   # +15→+20 dark blue
    (0.000, 0.000, 0.180),   # > +20   near-black navy (over)
]
anom_cmap = mcolors.ListedColormap(_imd_bin_colors, name='imd_erfs')
anom_norm = mcolors.BoundaryNorm(anom_levels, anom_cmap.N, extend='both')

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

# (IMD-matched colormap is defined above with anom_levels, anom_cmap, anom_norm)

# ── Map decoration ─────────────────────────────────────────────────────────
def decorate(ax, title, title_color, row=0, col=0, n_map_rows=2):
    """Decorate a map subplot. Gridline labels shown only on edges."""
    ax.set_extent([LON0, LON1, LAT0, LAT1], crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                     edgecolor='black', facecolor='none'),
        linewidth=1.0, zorder=4)
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m',
                                     edgecolor='black', facecolor='none'),
        linewidth=0.5, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.5, linestyle=':')
    gl.top_labels   = False
    gl.right_labels = False
    gl.left_labels  = (col == 0)
    gl.bottom_labels = (row == n_map_rows - 1)
    gl.xlabel_style = {'size': 8, 'weight': 'bold'}
    gl.ylabel_style = {'size': 8, 'weight': 'bold'}
    ax.set_title(title, fontsize=10, color=title_color, fontweight='bold', pad=4)


# ── IMD panel cropping helper ──────────────────────────────────────────────
def crop_imd_quadrants(img_full):
    """
    Given the full IMD ERFS PNG, return a list of 4 individual week panels
    (Wk1, Wk2, Wk3, Wk4) cropped from the anomaly (right) half.

    IMD bulletin layout (right half, after title trim):
      rows   0–129 : Wk1 (left) | Wk2 (right)   ← top row
      rows 130–140 : white gap / week label strip
      rows 141–276 : Wk3 (left) | Wk4 (right)   ← bottom row
      rows 277–280 : white gap
      rows 281–295 : internal colorbar (discard)
      rows 296+    : white padding (discard)

    Vertical split at col ~140.
    """
    h, w = img_full.shape[:2]
    anom  = img_full[:, w // 2:]               # right half
    body  = anom[int(h * 0.07):, :]            # trim title bar
    bh, bw = body.shape[:2]

    # Find the horizontal midpoint by looking for white gap rows
    row_mean = np.mean(body, axis=(1, 2))
    mid_col  = bw // 2

    # Top row panels: rows 0 to first white gap
    top_end = 0
    for r in range(bh // 2 - 5, bh // 2 + 20):
        if row_mean[r] > 0.99:
            top_end = r
            break
    if top_end == 0:
        top_end = bh // 2 - 5

    # Bottom row panels: after the white gap to before the colorbar
    bot_start = top_end
    for r in range(top_end, min(top_end + 30, bh)):
        if row_mean[r] < 0.97:
            bot_start = r
            break

    # Bottom row ends before the colorbar/white footer
    bot_end = bh
    for r in range(bh - 1, bh // 2, -1):
        if row_mean[r] < 0.97:
            bot_end = r + 1
            break
    # Exclude internal colorbar — find where bottom maps end
    # The maps end ~row 276, colorbar starts ~281
    map_bot_end = bot_end
    for r in range(bot_start + 10, bot_end):
        if row_mean[r] > 0.99:
            map_bot_end = r
            break

    panels = [
        body[:top_end,          :mid_col],   # Wk1: top-left
        body[:top_end,          mid_col:],   # Wk2: top-right
        body[bot_start:map_bot_end, :mid_col],   # Wk3: bottom-left
        body[bot_start:map_bot_end, mid_col:],   # Wk4: bottom-right
    ]
    return panels


# ── Figure maker ───────────────────────────────────────────────────────────
def make_triple(ic):
    cfg   = CASES[ic]
    weeks = cfg["weeks"]

    # ── compute arrays (UNCHANGED) ────────────────────────────────────
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

    # ── Crop individual IMD panels ────────────────────────────────────
    imd_panels = crop_imd_quadrants(imread(cfg["imd_png"]))
    print(f"\n{ic} IMD: cropped {len(imd_panels)} individual week panels")
    for i, p in enumerate(imd_panels):
        print(f"  Wk{i+1}: {p.shape}")

    # ── Build 3×4 figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 15))
    fig.suptitle(cfg["title"], fontsize=14, fontweight='bold', y=0.985)
    fig.text(0.5, 0.965,
             "ERA5 regridded from native 0.25° to 1.5° for direct comparison with FuXi-S2S",
             ha='center', fontsize=9.5, style='italic', color='dimgray')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.25, wspace=0.08,
                           top=0.94, bottom=0.08,
                           left=0.08, right=0.97)

    row_meta = [
        ("FuXi Anomaly\n(normalised, 1.5°)",  fuxi_anoms, 'darkgreen'),
        ("ERA5 Observed\n(regridded to 1.5°)", era5_anoms, 'darkblue'),
    ]

    im = None
    all_axes = []     # flat list for panel-letter labelling
    panel_idx = 0

    # ── Rows 0–1: FuXi and ERA5 map panels ───────────────────────────
    for row, (row_lbl, arrays, color) in enumerate(row_meta):
        first_ax = None
        for c, (arr, wk) in enumerate(zip(arrays, weeks)):
            ax = fig.add_subplot(gs[row, c], projection=ccrs.PlateCarree())
            all_axes.append(ax)
            im = ax.pcolormesh(fuxi_lons, fuxi_lats, arr,
                               cmap=anom_cmap, norm=anom_norm,
                               transform=ccrs.PlateCarree(), shading='auto')
            decorate(ax, f"({wk['label']})", color, row=row, col=c, n_map_rows=2)

            # Panel letter
            ax.text(0.03, 0.95, f"({chr(97 + panel_idx)})",
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
            panel_idx += 1
            if first_ax is None:
                first_ax = ax

        # Row label on the left
        first_ax.text(-0.20, 0.5, row_lbl,
                      transform=first_ax.transAxes,
                      fontsize=10, color=color, fontweight='bold',
                      va='center', ha='center', rotation=90)

    # ── Row 2: IMD ERFS panels (cropped from bulletin PNG) ────────────
    imd_first_ax = None
    for c in range(4):
        ax = fig.add_subplot(gs[2, c])
        all_axes.append(ax)
        ax.imshow(imd_panels[c], aspect='equal', interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('gray')
        ax.set_title(f"({weeks[c]['label']})", fontsize=10, color='darkred',
                     fontweight='bold', pad=4)

        # Panel letter
        ax.text(0.03, 0.95, f"({chr(97 + panel_idx)})",
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
        panel_idx += 1
        if imd_first_ax is None:
            imd_first_ax = ax

    # Row label
    imd_first_ax.text(-0.20, 0.5, "IMD ERFS\n(operational bulletin)",
                      transform=imd_first_ax.transAxes,
                      fontsize=10, color='darkred', fontweight='bold',
                      va='center', ha='center', rotation=90)

    # ── Shared colorbar below all three rows ──────────────────────────
    cax = fig.add_axes([0.15, 0.035, 0.70, 0.018])
    cb  = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    cb.set_label('mm/day', fontsize=10, labelpad=3)
    cb.set_ticks(anom_levels)
    cb.ax.tick_params(labelsize=9)

    # ── Caption ───────────────────────────────────────────────────────
    fig.text(0.5, 0.005,
             "IMD ERFS panels cropped from operational extended range forecast bulletin "
             "(Pattanaik et al., 2019); gridded forecast data unavailable.",
             ha='center', fontsize=8, style='italic', color='dimgray')

    fig.savefig(cfg["outfile"], dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {cfg['outfile']}")

# ── Run both cases ─────────────────────────────────────────────────────────
make_triple("IC0707")
make_triple("IC0825")
print("\nBoth figures done.")
