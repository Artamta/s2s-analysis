"""
plot_imd_regions.py  —  v4, fully offline
==========================================
Uses ONLY already-cached shapefiles:
  - ne_10m_admin_1_states_provinces  (used in mask build → definitely cached)
  - ne_50m_admin_0_countries         (cached in previous run)
  - ne_50m_physical_land             (cached in previous run)
NO cfeature.* calls that trigger downloads. No internet needed.
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

MASK_FILE = '/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc'
OUT       = '/home/raj.ayush/s2s/s2s_anlysis/paper/code/figures/imd_region_mask_verification.png'

STATE_GROUPS = {
    'northwest_india': [
        'Jammu and Kashmir', 'Ladakh', 'Himachal Pradesh',
        'Punjab', 'Haryana', 'Delhi', 'Chandigarh',
        'Uttarakhand', 'Rajasthan', 'Uttar Pradesh',
    ],
    'central_india': [
        'Gujarat', 'Dadra and Nagar Haveli and Daman and Diu',
        'Dadra and Nagar Haveli', 'Daman and Diu',
        'Madhya Pradesh', 'Chhattisgarh',
        'Maharashtra', 'Goa', 'Odisha',
    ],
    'south_peninsula': [
        'Andhra Pradesh', 'Telangana', 'Tamil Nadu',
        'Puducherry', 'Karnataka', 'Kerala', 'Lakshadweep',
    ],
    'east_northeast_india': [
        'Bihar', 'West Bengal', 'Sikkim', 'Jharkhand',
        'Assam', 'Meghalaya', 'Nagaland', 'Manipur',
        'Mizoram', 'Tripura', 'Arunachal Pradesh',
        'Andaman and Nicobar',
    ],
}
REGION_STYLE = {
    'northwest_india':      {'label': 'Northwest India',        'color': '#C0392B'},
    'central_india':        {'label': 'Central India',          'color': '#E67E22'},
    'south_peninsula':      {'label': 'South Peninsula',        'color': '#16A085'},
    'east_northeast_india': {'label': 'East & Northeast India', 'color': '#2471A3'},
}
state_to_region = {n.lower(): k for k, ns in STATE_GROUPS.items() for n in ns}

# ── masks ────────────────────────────────────────────────────────────────────
ds = xr.open_dataset(MASK_FILE)

# ── shapefiles (all already cached) ─────────────────────────────────────────
shp_states    = shpreader.natural_earth('10m', 'cultural', 'admin_1_states_provinces')
shp_countries = shpreader.natural_earth('50m', 'cultural', 'admin_0_countries')
shp_land      = shpreader.natural_earth('50m', 'physical', 'land')

# ── figure ───────────────────────────────────────────────────────────────────
PC = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(11, 10), subplot_kw={'projection': PC})
ax.set_extent([60, 100, 2, 40], crs=PC)
ax.set_facecolor('#AED6F1')                     # ocean — no download needed

# 1. land background (all countries, no borders)
for geom in shpreader.Reader(shp_land).geometries():
    ax.add_geometries([geom], PC,
                      facecolor='#EAECEE', edgecolor='none', zorder=1)

# 2. colored Indian state polygons by IMD region
for rec in shpreader.Reader(shp_states).records():
    if rec.attributes.get('admin', '') != 'India':
        continue
    name = rec.attributes.get('name', '').lower()
    rkey = state_to_region.get(name)
    fc   = REGION_STYLE[rkey]['color'] if rkey else '#BDC3C7'
    ax.add_geometries([rec.geometry], PC,
                      facecolor=fc, alpha=0.85,
                      edgecolor='white', linewidth=0.5, zorder=3)

# 3. country outlines ONLY (no fill) — drawn thin so they act as coastline
#    No country borders = no Kashmir dispute shown
for rec in shpreader.Reader(shp_countries).records():
    ax.add_geometries([rec.geometry], PC,
                      facecolor='none',
                      edgecolor='#777', linewidth=0.5, zorder=4)

# 4. pipeline scoring domain box
import matplotlib.patches as patches
rect = patches.Rectangle(
    (65, 5), 35, 33,
    linewidth=2.0, edgecolor='#1C2833',
    facecolor='none', linestyle='--',
    transform=PC, zorder=6
)
ax.add_patch(rect)
ax.text(65.3, 5.6, 'Pipeline scoring domain\n5–38°N, 65–100°E',
        transform=PC, fontsize=7.5, color='#1C2833', va='bottom', zorder=7,
        bbox=dict(fc='white', alpha=0.75, ec='none', pad=1.5))

# 5. gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                  alpha=0.5, linestyle='--', zorder=5)
gl.top_labels   = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator(range(60, 105, 5))
gl.ylocator = mticker.FixedLocator(range(0,  45,  5))
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}

# 6. legend
legend_patches = [
    mpatches.Patch(facecolor=s['color'], edgecolor='white',
                   linewidth=0.8, label=s['label'], alpha=0.85)
    for s in REGION_STYLE.values()
]
legend_patches.append(
    mpatches.Patch(facecolor='#BDC3C7', edgecolor='white',
                   linewidth=0.8, label='Unassigned territory', alpha=0.7)
)
ax.legend(handles=legend_patches, loc='lower left', fontsize=9.5,
          framealpha=0.93, title='IMD Homogeneous Region',
          title_fontsize=10, edgecolor='#ccc',
          borderpad=0.9, labelspacing=0.55)

# 7. title + disclaimer
ax.set_title(
    'IMD Four Homogeneous Rainfall Regions\n'
    '(Pai et al., 2014  |  Natural Earth 10m state boundaries)',
    fontsize=12, fontweight='bold', pad=10
)
fig.text(
    0.5, 0.005,
    '* The external boundaries and coastlines of India as shown in this figure are neither authentic nor correct.\n'
    '  Boundaries are based on Natural Earth 10m data and are for representation purposes only.',
    ha='center', fontsize=6.5, color='#666', style='italic'
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved -> {OUT}')
