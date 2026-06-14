"""
imd_region_masks.py
====================
Builds boolean masks for the 4 IMD homogeneous rainfall regions using
the official Survey of India STATE_BOUNDARY.shp (LCC projection),
reprojected to WGS84 via pyproj + shapely.

IMD 4-region grouping (Pai et al., 2014):
  - Northwest India   : J&K, Ladakh, HP, Punjab, Haryana, Delhi, Chandigarh,
                        Uttarakhand, Rajasthan, UP
  - Central India     : Gujarat, DNHDD, MP, Chhattisgarh, Maharashtra, Goa, Odisha
  - South Peninsula   : AP, Telangana, TN, Karnataka, Kerala, Puducherry, Lakshadweep
  - East & NE India   : Bihar, WB, Sikkim, Jharkhand, Assam, Meghalaya,
                        Nagaland, Manipur, Mizoram, Tripura, Arunachal Pradesh,
                        Andaman & Nicobar

Usage:
    python imd_region_masks.py          # regenerate imd_region_masks.nc
    from utils.imd_region_masks import get_imd_masks
    masks = get_imd_masks(lat_array, lon_array)
"""

import numpy as np
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer

SOI_SHP = '/storage/raj.ayush/s2s-forecast-/STATE_BOUNDARY.shp'

# Transformer: SOI LCC → WGS84 (proj4 strings only — no EPSG DB needed)
_T = Transformer.from_crs(
    '+proj=lcc +lat_0=24 +lon_0=80 +lat_1=12.472944 +lat_2=35.172806'
    ' +x_0=4000000 +y_0=4000000 +datum=WGS84 +units=m +no_defs',
    '+proj=longlat +datum=WGS84 +no_defs',
    always_xy=True
)

def _to_wgs84(geom):
    return shp_transform(_T.transform, geom)


# SOI state names → IMD region (uppercase, exactly as in STATE attribute)
IMD_STATE_GROUPS = {
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

REGION_DISPLAY = {
    'northwest_india':      'Northwest India',
    'central_india':        'Central India',
    'south_peninsula':      'South Peninsula',
    'east_northeast_india': 'East & Northeast India',
}


def _load_soi_state_polygons(verbose=True):
    """Load SOI state polygons, reproject LCC → WGS84."""
    if verbose:
        print(f'  Loading SOI shapefile: {SOI_SHP}')
    state_polys = {}
    for rec in shpreader.Reader(SOI_SHP).records():
        name = rec.attributes['STATE']
        state_polys[name] = _to_wgs84(rec.geometry)
    if verbose:
        print(f'  {len(state_polys)} states loaded and reprojected to WGS84')
    return state_polys


def _build_region_polygons(state_polys, verbose=True):
    """Union SOI state polygons into 4 IMD region polygons."""
    region_polys = {}
    for key, wanted in IMD_STATE_GROUPS.items():
        matched, geoms = [], []
        for sname in wanted:
            if sname in state_polys:
                geoms.append(state_polys[sname])
                matched.append(sname)
        if not geoms:
            print(f'  WARNING: no states matched for {key}')
            continue
        region_polys[key] = unary_union(geoms)
        if verbose:
            print(f'  {REGION_DISPLAY[key]:<30} -> {len(matched)} states')
    return region_polys


def get_imd_masks(lat_array, lon_array, verbose=True):
    """
    Build boolean masks for the 4 IMD homogeneous rainfall regions
    using official SOI state boundaries.

    Parameters
    ----------
    lat_array : (nlat,) degrees North
    lon_array : (nlon,) degrees East

    Returns
    -------
    dict { region_key : np.ndarray(bool, shape=(nlat, nlon)) }
    """
    lat = np.asarray(lat_array, dtype=float)
    lon = np.asarray(lon_array, dtype=float)
    lon = np.where(lon > 180, lon - 360, lon)

    if verbose:
        print('Loading SOI state polygons (Survey of India STATE_BOUNDARY.shp)...')
    state_polys  = _load_soi_state_polygons(verbose=verbose)

    if verbose:
        print('Building IMD region polygons (shapely union)...')
    region_polys = _build_region_polygons(state_polys, verbose=verbose)

    nlat, nlon = len(lat), len(lon)
    LON, LAT   = np.meshgrid(lon, lat)
    pts        = list(zip(LON.ravel(), LAT.ravel()))

    if verbose:
        print(f'\nRunning point-in-polygon on {nlat}x{nlon}={nlat*nlon} grid points ...')

    masks = {}
    for key, poly in region_polys.items():
        if verbose:
            print(f'  {REGION_DISPLAY[key]}...', flush=True)
        inside       = np.array([poly.contains(Point(x, y)) for x, y in pts], dtype=bool)
        masks[key]   = inside.reshape(nlat, nlon)

    if verbose:
        overlap = sum(m.astype(int) for m in masks.values())
        print('\n  Coverage summary:')
        for key, m in masks.items():
            pct = 100 * m.sum() / (nlat * nlon)
            print(f'    {REGION_DISPLAY[key]:<30}  {m.sum():>5} pts  ({pct:.1f}% of box)')
        print(f'\n    Points in 0 regions (gap):       {(overlap == 0).sum()}')
        print(f'    Points in exactly 1 region:      {(overlap == 1).sum()}')
        print(f'    Points in 2+ regions (overlap):  {(overlap >= 2).sum()} <- should be 0!')

    return masks


def save_masks(masks, lat_array, lon_array, outpath):
    """Save masks to NetCDF for fast pipeline reuse."""
    import xarray as xr
    ds = xr.Dataset(
        {k: (['lat', 'lon'], v.astype(np.int8)) for k, v in masks.items()},
        coords={'lat': lat_array, 'lon': lon_array}
    )
    ds.attrs['description'] = (
        'IMD 4-region boolean masks (1=in region) built from Survey of India '
        'STATE_BOUNDARY.shp (official claimed boundaries, LCC reprojected to WGS84).'
    )
    ds.attrs['citation'] = 'Pai et al. (2014); boundaries: Survey of India'
    ds.attrs['regions']  = ', '.join(REGION_DISPLAY.values())
    ds.to_netcdf(outpath)
    print(f'Saved -> {outpath}')


if __name__ == '__main__':
    target_lat = np.arange(38, 5, -1.5)
    target_lon = np.arange(65, 100, 1.5)

    print('=' * 60)
    print('Building IMD 4-region masks on 1.5° grid  [SOI boundaries]')
    print('=' * 60)
    masks = get_imd_masks(target_lat, target_lon, verbose=True)

    out = '/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc'
    save_masks(masks, target_lat, target_lon, out)
    print('\nDone!')
