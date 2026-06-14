"""
fig01_domain.py — v9 FINAL
===========================
1. Reprojects SOI STATE_BOUNDARY.shp (LCC) → WGS84 using pyproj
2. Draws reprojected geometries in PlateCarree (fast, no cartopy CRS issues)
3. Full Kashmir/Ladakh from official SOI data
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from pyproj import Transformer
from shapely.ops import transform as shp_transform

OUT     = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs/fig01_domain.png'
SOI_SHP = '/storage/raj.ayush/s2s-forecast-/STATE_BOUNDARY.shp'

# Reproject SOI LCC → WGS84 (lat/lon)
_T = Transformer.from_crs(
    '+proj=lcc +lat_0=24 +lon_0=80 +lat_1=12.472944 +lat_2=35.172806'
    ' +x_0=4000000 +y_0=4000000 +datum=WGS84 +units=m +no_defs',
    '+proj=longlat +datum=WGS84 +no_defs',
    always_xy=True)

def to_wgs84(geom):
    return shp_transform(_T.transform, geom)

# Load & reproject all SOI state polygons once
print('Reprojecting SOI shapefile...', flush=True)
SOI_STATES = []
for rec in shpreader.Reader(SOI_SHP).records():
    SOI_STATES.append({
        'name': rec.attributes['STATE'],
        'geom': to_wgs84(rec.geometry)
    })
print(f'  {len(SOI_STATES)} states loaded.', flush=True)

# IMD region mapping (SOI state names, uppercase)
STATE_GROUPS = {
    'northwest_india': {
        'JAMMU AND KASHMIR', 'LADAKH', 'HIMACHAL PRADESH',
        'PUNJAB', 'HARYANA', 'DELHI', 'CHANDIGARH',
        'UTTARAKHAND', 'RAJASTHAN', 'UTTAR PRADESH',
        'DISPUTED (MADHYA PRADESH & RAJASTHAN)',
        'DISPUTED (RAJATHAN & GUJARAT)',
    },
    'central_india': {
        'GUJARAT', 'DADRA & NAGAR HAVELI & DAMAN & DIU',
        'MADHYA PRADESH', 'CHHATTISGARH',
        'MAHARASHTRA', 'GOA', 'ODISHA',
        'DISPUTED (MADHYA PRADESH & GUJARAT)',
    },
    'south_peninsula': {
        'ANDHRA PRADESH', 'TELANGANA', 'TAMIL NADU',
        'PUDUCHERRY', 'KARNATAKA', 'KERALA', 'LAKSHADWEEP',
    },
    'east_northeast_india': {
        'BIHAR', 'WEST BENGAL', 'SIKKIM', 'JHARKHAND',
        'ASSAM', 'MEGHALAYA', 'NAGALAND', 'MANIPUR',
        'MIZORAM', 'TRIPURA', 'ARUNACHAL PRADESH',
        'ANDAMAN & NICOBAR',
        'DISPUTED (WEST BENGAL , BIHAR & JHARKHAND)',
    },
}
REGION_STYLE = {
    'northwest_india':      {'label': 'North West India',       'color': '#A93226'},
    'central_india':        {'label': 'Central India',          'color': '#28B463'},
    'south_peninsula':      {'label': 'South Peninsular India', 'color': '#1A5276'},
    'east_northeast_india': {'label': 'North East India',       'color': '#5DADE2'},
}
DOMAIN_COLOR = '#2471A3'
state_to_rg  = {s: k for k, names in STATE_GROUPS.items() for s in names}

PC    = ccrs.PlateCarree()
OCEAN = '#AED6F1'
NBRS  = '#D5D8DC'
LAND_SHP = shpreader.natural_earth('50m', 'physical', 'land')


def setup_bg(ax, extent):
    ax.set_extent(list(extent), crs=PC)
    ax.set_facecolor(OCEAN)
    for g in shpreader.Reader(LAND_SHP).geometries():
        ax.add_geometries([g], PC, facecolor=NBRS,
                          edgecolor='#555', linewidth=0.5, zorder=1)


def draw_states(ax, color_map, lw=0.45):
    """Draw reprojected (WGS84) SOI states — passed as PC geometries."""
    for st in SOI_STATES:
        fc = color_map.get(st['name'], '#BFC9CA')
        ax.add_geometries([st['geom']], PC,
                          facecolor=fc, alpha=0.92,
                          edgecolor='white', linewidth=lw, zorder=3)


def ocean_labels(ax, x0, x1, y0, y1):
    kw = dict(transform=PC, color='#1F618D', fontstyle='italic',
              ha='center', fontweight='bold', zorder=6)
    def t(x, y, s, fs=9):
        if x0 < x < x1 and y0 < y < y1:
            ax.text(x, y, s, fontsize=fs, **kw)
    t(64.5, 16.5, 'Arabian\nSea')
    t(89.0, 13.0, 'Bay of\nBengal')
    t(76.0,  6.5, 'Indian Ocean', fs=8.5)


def north_arrow(ax):
    ax.annotate('N', xy=(0.065, 0.92), xytext=(0.065, 0.84),
                xycoords='axes fraction', fontsize=11,
                fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.8, color='k'),
                zorder=9)


def gridlines(ax, xlocs, ylocs):
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.4, linestyle='--', zorder=5)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.FixedLocator(xlocs)
    gl.ylocator     = mticker.FixedLocator(ylocs)
    gl.xlabel_style = {'size': 8.5, 'color': '#333'}
    gl.ylabel_style = {'size': 8.5, 'color': '#333'}


# ════════════════════════════════════════════
print('Plotting...', flush=True)
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(16, 8),
    subplot_kw={'projection': PC},
    gridspec_kw={'wspace': 0.05}
)

EXT_A = (63, 100, 5, 39)
EXT_B = (63, 100, 5, 39)

# Panel (a)
setup_bg(ax1, EXT_A)
domain_map = {s: DOMAIN_COLOR for grp in STATE_GROUPS.values() for s in grp}
draw_states(ax1, domain_map, lw=0.3)
ocean_labels(ax1, *EXT_A)
north_arrow(ax1)
gridlines(ax1, xlocs=range(60, 105, 10), ylocs=range(10, 40, 10))
ax1.text(81, 21.5, 'Verification Domain', transform=PC,
         ha='center', fontsize=9.5, fontweight='bold', color='white', zorder=7)
ax1.text(81, 19.0, '6.5°–38°N  |  65°–99.5°E', transform=PC,
         ha='center', fontsize=8.5, color='white', zorder=7)
ax1.set_title('(a)  Study Domain — Indian Subcontinent\n'
              'Grid: 1.5° resolution, land points only',
              fontsize=11, fontweight='bold', pad=9)

# Panel (b)
setup_bg(ax2, EXT_B)
region_map = {s: REGION_STYLE[k]['color']
              for k, names in STATE_GROUPS.items() for s in names}
draw_states(ax2, region_map, lw=0.45)
ocean_labels(ax2, *EXT_B)
north_arrow(ax2)
gridlines(ax2, xlocs=range(65, 105, 5), ylocs=range(10, 40, 5))
ax2.set_title('(b)  IMD Four Homogeneous Rainfall Regions\n'
              'Pai et al. (2014)  |  Survey of India state boundaries',
              fontsize=11, fontweight='bold', pad=9)

# Legend
handles = [
    mpatches.Patch(facecolor=s['color'], edgecolor='white',
                   linewidth=0.8, label=s['label'], alpha=0.92)
    for s in REGION_STYLE.values()
]
fig.legend(handles=handles, loc='lower center', ncol=4,
           fontsize=10.5, frameon=False,
           bbox_to_anchor=(0.55, 0.0),
           handlelength=1.8, handletextpad=0.5, columnspacing=1.5)

fig.text(0.5, -0.04,
         '* The boundaries and coastlines shown are for representation '
         'purposes only and are neither authentic nor correct.',
         ha='center', fontsize=7, color='#888', style='italic')

plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved → {OUT}')
