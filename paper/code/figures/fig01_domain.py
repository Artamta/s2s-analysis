"""
fig01_domain.py  — v10 FINAL (publication quality)
===================================================
Survey of India STATE_BOUNDARY.shp — full Kashmir.
600 DPI, professional cartography style.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from pyproj import Transformer
from shapely.ops import transform as shp_transform

plt.rcParams.update({'font.family': 'DejaVu Sans'})

OUT     = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs/fig01_domain.png'
SOI_SHP = '/storage/raj.ayush/s2s-forecast-/STATE_BOUNDARY.shp'

# ── Reproject SOI LCC → WGS84 ──────────────────────────────────────
_T = Transformer.from_crs(
    '+proj=lcc +lat_0=24 +lon_0=80 +lat_1=12.472944 +lat_2=35.172806'
    ' +x_0=4000000 +y_0=4000000 +datum=WGS84 +units=m +no_defs',
    '+proj=longlat +datum=WGS84 +no_defs', always_xy=True)

def _to_wgs84(geom):
    return shp_transform(_T.transform, geom)

print('Reprojecting SOI shapefile…', flush=True)
SOI_STATES = [{'name': rec.attributes['STATE'], 'geom': _to_wgs84(rec.geometry)}
              for rec in shpreader.Reader(SOI_SHP).records()]
print(f'  {len(SOI_STATES)} states loaded.', flush=True)

# ── IMD region definitions ──────────────────────────────────────────
STATE_GROUPS = {
    'northwest_india': {
        'JAMMU AND KASHMIR','LADAKH','HIMACHAL PRADESH','PUNJAB','HARYANA',
        'DELHI','CHANDIGARH','UTTARAKHAND','RAJASTHAN','UTTAR PRADESH',
        'DISPUTED (MADHYA PRADESH & RAJASTHAN)',
        'DISPUTED (RAJATHAN & GUJARAT)',
    },
    'central_india': {
        'GUJARAT','DADRA & NAGAR HAVELI & DAMAN & DIU','MADHYA PRADESH',
        'CHHATTISGARH','MAHARASHTRA','GOA','ODISHA',
        'DISPUTED (MADHYA PRADESH & GUJARAT)',
    },
    'south_peninsula': {
        'ANDHRA PRADESH','TELANGANA','TAMIL NADU','PUDUCHERRY',
        'KARNATAKA','KERALA','LAKSHADWEEP',
    },
    'east_northeast_india': {
        'BIHAR','WEST BENGAL','SIKKIM','JHARKHAND','ASSAM','MEGHALAYA',
        'NAGALAND','MANIPUR','MIZORAM','TRIPURA','ARUNACHAL PRADESH',
        'ANDAMAN & NICOBAR',
        'DISPUTED (WEST BENGAL , BIHAR & JHARKHAND)',
    },
}

# Professional, formal color palette
REGION_STYLE = {
    'northwest_india':      {'label': 'North West India',       'color': '#A93226'},
    'central_india':        {'label': 'Central India',          'color': '#1D8348'},
    'south_peninsula':      {'label': 'South Peninsular India', 'color': '#1A4F72'},
    'east_northeast_india': {'label': 'North East India',       'color': '#2E86C1'},
}
DOMAIN_COLOR = '#1C5E8A'   # clean steel blue
OCEAN        = '#D0E8F5'   # soft academic blue
NBRS         = '#DEDEDE'   # neutral neighbor land
LAND_SHP     = shpreader.natural_earth('50m', 'physical', 'land')
PC           = ccrs.PlateCarree()


def setup_bg(ax, extent):
    ax.set_extent(list(extent), crs=PC)
    ax.set_facecolor(OCEAN)
    for g in shpreader.Reader(LAND_SHP).geometries():
        ax.add_geometries([g], PC, facecolor=NBRS,
                          edgecolor='#AAAAAA', linewidth=0.4, zorder=1)


def draw_states(ax, color_map, lw=0.5):
    for st in SOI_STATES:
        fc = color_map.get(st['name'], '#CCCCCC')
        ax.add_geometries([st['geom']], PC, facecolor=fc, alpha=0.93,
                          edgecolor='white', linewidth=lw, zorder=3)


def ocean_labels(ax, labels):
    """labels: list of (lon, lat, text, fontsize)"""
    shadow = [pe.withStroke(linewidth=2, foreground=OCEAN)]
    for lon, lat, txt, fs in labels:
        ax.text(lon, lat, txt, transform=PC,
                color='#1A5276', fontstyle='italic', fontweight='bold',
                fontsize=fs, ha='center', va='center',
                path_effects=shadow, zorder=6)


def compass_rose(ax, x=0.073, y=0.88, sz=0.055):
    """Elegant compass: filled N arrow + hollow S + N label."""
    from matplotlib.patches import FancyArrowPatch, Wedge, Circle
    from matplotlib.lines import Line2D

    # Outer circle
    circ = plt.Circle((x, y), sz * 0.95, transform=ax.transAxes,
                       fc='white', ec='#444', lw=0.8, zorder=10, clip_on=False)
    ax.add_patch(circ)

    # North (filled dark)
    ax.annotate('', xy=(x, y + sz * 0.88), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='#222',
                                lw=1.2, mutation_scale=10), zorder=11)
    # South (thin outline)
    ax.annotate('', xy=(x, y - sz * 0.88), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='#888',
                                lw=0.8, mutation_scale=7), zorder=11)
    # N label
    ax.text(x, y + sz * 1.15, 'N', transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='center', va='center',
            color='#111', zorder=12, clip_on=False)


def gridlines(ax, xlocs, ylocs):
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color='#888',
                      alpha=0.5, linestyle='--', zorder=5)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.FixedLocator(xlocs)
    gl.ylocator     = mticker.FixedLocator(ylocs)
    gl.xlabel_style = {'size': 8, 'color': '#444'}
    gl.ylabel_style = {'size': 8, 'color': '#444'}


# ── Figure layout ───────────────────────────────────────────────────
print('Plotting…', flush=True)
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(16, 8),
    subplot_kw={'projection': PC},
    gridspec_kw={'wspace': 0.06}
)

EXT = (63, 100, 5, 39)   # same for both panels

# ── Panel (a) ────────────────────────────────────────────────────────
setup_bg(ax1, EXT)
domain_map = {s: DOMAIN_COLOR for grp in STATE_GROUPS.values() for s in grp}
draw_states(ax1, domain_map, lw=0.3)
ocean_labels(ax1, [
    (66.5, 17.0, 'Arabian\nSea',  8.5),
    (89.0, 13.0, 'Bay of\nBengal', 8.5),
    (77.0,  6.8, 'Indian Ocean',   8.0),
])
gridlines(ax1, xlocs=range(65, 105, 10), ylocs=range(10, 40, 10))
ax1.set_title('(a)  Study Domain — India\n'
              'Grid: 1.5° resolution  |  Land points only',
              fontsize=11, fontweight='bold', pad=10, color='#222')

# ── Panel (b) ────────────────────────────────────────────────────────
setup_bg(ax2, EXT)
region_map = {s: REGION_STYLE[k]['color']
              for k, names in STATE_GROUPS.items() for s in names}
draw_states(ax2, region_map, lw=0.5)
ocean_labels(ax2, [
    (66.5, 17.0, 'Arabian\nSea',  8.5),
    (90.5, 13.0, 'Bay of\nBengal', 8.5),
    (77.0,  6.8, 'Indian Ocean',   8.0),
])
gridlines(ax2, xlocs=range(65, 105, 10), ylocs=range(10, 40, 10))

# Region labels — carefully positioned, path effects for legibility
shadow_w = [pe.withStroke(linewidth=2.5, foreground='#000000AA')]
shadow_b = [pe.withStroke(linewidth=2.5, foreground='#FFFFFFAA')]
RLABELS = [
    # lon,  lat,  text,                     color,   fs,   effects
    (77.0, 28.5, 'North West\nIndia',       'white', 8.5,  shadow_w),
    (79.5, 21.5, 'Central\nIndia',          'white', 8.5,  shadow_w),
    (78.0, 13.2, 'South\nPeninsular India', 'white', 8.0,  shadow_w),
    (91.5, 26.0, 'North East\nIndia',       'white', 8.0,  shadow_w),
]
for lx, ly, ltxt, lcol, lfs, eff in RLABELS:
    ax2.text(lx, ly, ltxt, transform=PC, ha='center', va='center',
             fontsize=lfs, fontweight='bold', color=lcol,
             path_effects=eff, zorder=8)

ax2.set_title('(b)  IMD Four Homogeneous Rainfall Regions\n'
              'Pai et al. (2014)  |  Survey of India state boundaries',
              fontsize=11, fontweight='bold', pad=10, color='#222')

# ── Legend ───────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=s['color'], edgecolor='#555',
                   linewidth=0.6, label=s['label'], alpha=0.93)
    for s in REGION_STYLE.values()
]
leg = fig.legend(handles=handles, loc='lower center', ncol=4,
                 fontsize=10, frameon=True, framealpha=1.0,
                 edgecolor='#BBBBBB', fancybox=False,
                 bbox_to_anchor=(0.55, 0.0),
                 handlelength=1.8, handletextpad=0.6, columnspacing=1.6)
leg.get_frame().set_linewidth(0.8)

plt.savefig(OUT, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved → {OUT}')
