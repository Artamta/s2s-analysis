#!/usr/bin/env python3
"""
make_spatial_maps.py — Gridded spatial verification maps (22×24, 1.5° India grid).

For each (variable, week) pair, produces one figure with a row per model showing:
  col 1: Observed mean (ERA5, all 13 inits)
  col 2: Forecast mean (ensemble mean, all 13 inits)
  col 3: Mean Bias (fcst − obs)
  col 4: Point-wise PCC (correlation across 13 inits)

IMD homogeneous-region boundaries are drawn on every panel.

Outputs  →  plots/spatial_<VAR>_W<N>.png   (TP W1..W6, Z500 W1..W6 = 12 figs)

Usage:
  python make_spatial_maps.py              # all vars, all weeks
  python make_spatial_maps.py --var TP     # TP only
  python make_spatial_maps.py --week 1 2   # weeks 1 and 2 only
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loaders as L
import metrics as M
from loaders import CFG, WEEKS

ODIR = os.path.join(HERE, 'plots')
os.makedirs(ODIR, exist_ok=True)

# ── display config ─────────────────────────────────────────────────────────────
VAR_UNIT   = {'TP': 'mm/day', 'Z500': 'gpm'}
VAR_CLABEL = {'TP': 'TP (mm/day)', 'Z500': 'Z500 (gpm)'}

MODEL_LABEL = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi', 'ECMWF': 'ECMWF'}
BASE_MODELS  = ['SPIRE', 'FuXi', 'ECMWF']

# colormaps
CMAP_MEAN = {'TP': 'YlGnBu', 'Z500': 'RdBu_r'}
CMAP_BIAS = 'RdBu_r'
CMAP_PCC  = 'RdYlGn'

# IMD region colours for boundary overlay
REGION_COLORS = {
    'northwest_india':      '#e6194b',
    'central_india':        '#3cb44b',
    'south_peninsula':      '#4363d8',
    'east_northeast_india': '#f58231',
}
REGION_LABEL = {
    'northwest_india':      'NW India',
    'central_india':        'Central India',
    'south_peninsula':      'South Peninsula',
    'east_northeast_india': 'East & NE India',
}


def _proj():
    return ccrs.PlateCarree()


def _add_regions(ax, GC, alpha=0.25):
    """Draw filled IMD region overlays on a cartopy axis."""
    lat, lon = GC['lat'], GC['lon']
    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    for rname, color in REGION_COLORS.items():
        mask = GC['region_masks'].get(rname)
        if mask is None:
            continue
        ys, xs = np.where(mask)
        for yi, xi in zip(ys, xs):
            rect = plt.Rectangle(
                (lon[xi] - dlon / 2, lat[yi] - dlat / 2),
                dlon, dlat,
                linewidth=0.4, edgecolor=color, facecolor='none',
                transform=_proj(), zorder=3)
            ax.add_patch(rect)


def _pcolormesh(ax, lat, lon, data, cmap, vmin, vmax, title='', unit=''):
    dlat = abs(lat[1] - lat[0]) / 2
    dlon = abs(lon[1] - lon[0]) / 2
    lat_e = np.concatenate([[lat[0] + dlat], lat - dlat])
    lon_e = np.concatenate([[lon[0] - dlon], lon + dlon])
    d = np.where(np.isnan(data), np.nan, data)
    im = ax.pcolormesh(lon_e, lat_e, d, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=_proj(), zorder=2)
    ax.set_extent([lon[0] - dlon, lon[-1] + dlon,
                   lat[-1] - dlat, lat[0] + dlat], crs=_proj())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.5, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.4, zorder=4)
    ax.set_title(title, fontsize=8, pad=3)
    return im


def compute_gridded(var, week_idx, GC):
    """
    Compute per-grid-point arrays across all 13 inits for (var, week).

    Returns dict keyed by model name:
        'obs_mean'  (lat, lon)  — ERA5 mean over inits
        'fcst_mean' (lat, lon)  — ensemble-mean forecast mean over inits
        'bias'      (lat, lon)  — mean(fcst_mean - obs_mean)
        'pcc'       (lat, lon)  — Pearson r of (fcst_anom, obs_anom) across inits

    obs_mean / pcc / bias use the same ERA5 clim as the verification pipeline.
    Also returns 'obs_mean' under key 'OBS' (single shared field).
    """
    wn, ds, de = WEEKS[week_idx]
    clim_ds = L.open_clim()
    truth   = L.open_truth()
    lat, lon = GC['lat'], GC['lon']
    nY, nX   = len(lat), len(lon)
    land     = np.array(GC['land']).astype(float)  # 1=land, 0=ocean (NaN mask)
    land_mask = np.where(land == 0, np.nan, 1.0)

    # accumulators: list over inits
    obs_fields  = []   # (nY, nX) per init
    clim_fields = []
    fcst_fields = {m: [] for m in BASE_MODELS}  # ensemble mean per init

    for init in CFG.init_dates:
        init_str = pd.to_datetime(init).strftime('%Y%m%d')

        # ── observations ──────────────────────────────────────────────────────
        valid = L.valid_dates_for(init, ds, de, CFG.valid_end)
        if not valid:
            continue
        o = L.truth_period_mean(var, truth, valid, GC)
        if o is None or bool(np.isnan(o).all()):
            continue
        doys  = [pd.to_datetime(d).dayofyear for d in valid]
        clim_o = L.clim_field(clim_ds, var, doys, GC)

        obs_fields.append(np.array(o))
        clim_fields.append(np.array(clim_o))

        # ── SPIRE ──────────────────────────────────────────────────────────────
        spire = L.load_spire(init)
        if spire and var in spire:
            mu_da = spire[var][0]
            f = L.to_grid(mu_da.isel(step=slice(ds - 1, de)).mean('step'), GC)
            fcst_fields['SPIRE'].append(np.array(f))
        else:
            fcst_fields['SPIRE'].append(None)

        # ── FuXi ───────────────────────────────────────────────────────────────
        fuxi = L.load_fuxi_all(init_str, [var], CFG.G)
        fd = [fuxi[var][d] for d in range(ds, de + 1) if d in fuxi[var]]
        if fd:
            ens  = xr.concat(fd, 't').mean('t')
            if var == 'TP':
                ens = ens * CFG.fuxi_tp_factor
            mu, _ = L.ens_mean_std(ens, 'member', GC)
            fcst_fields['FuXi'].append(np.array(mu))
        else:
            fcst_fields['FuXi'].append(None)

        # ── ECMWF ──────────────────────────────────────────────────────────────
        arr = L.load_op('ECMWF', init_str, var, CFG.G)
        if arr is not None and arr.sizes.get('step', 0) >= de:
            field = (L.weekly_mean_cumulative(arr, ds, de) if var == 'TP'
                     else arr.isel(step=slice(ds - 1, de)).mean('step'))
            mu, _ = L.ens_mean_std(field, 'number', GC)
            fcst_fields['ECMWF'].append(np.array(mu))
        else:
            fcst_fields['ECMWF'].append(None)

    # ── stack to arrays ────────────────────────────────────────────────────────
    obs_arr  = np.stack(obs_fields,   axis=0)   # (N, nY, nX)
    clim_arr = np.stack(clim_fields,  axis=0)

    out = {}
    obs_mean = np.nanmean(obs_arr, axis=0) * land_mask
    out['OBS'] = {'obs_mean': obs_mean}

    for m in BASE_MODELS:
        flist = fcst_fields[m]
        valid_idx = [i for i, f in enumerate(flist) if f is not None]
        if not valid_idx:
            out[m] = None
            continue

        f_arr = np.stack([flist[i] for i in valid_idx], axis=0)
        o_sub = obs_arr[valid_idx]
        c_sub = clim_arr[valid_idx]

        fcst_mean = np.nanmean(f_arr, axis=0) * land_mask
        bias      = np.nanmean(f_arr - o_sub, axis=0) * land_mask

        # point-wise PCC across inits (anomaly correlation)
        fa = f_arr - c_sub   # fcst anomaly (vs ERA5 clim)
        oa = o_sub - c_sub   # obs anomaly
        pcc = np.full((nY, nX), np.nan)
        for yi in range(nY):
            for xi in range(nX):
                if land_mask[yi, xi] != 1.0:
                    continue
                f_v = fa[:, yi, xi]
                o_v = oa[:, yi, xi]
                ok  = np.isfinite(f_v) & np.isfinite(o_v)
                if ok.sum() < 3:
                    continue
                r = np.corrcoef(f_v[ok], o_v[ok])[0, 1]
                pcc[yi, xi] = r

        out[m] = dict(obs_mean=obs_mean, fcst_mean=fcst_mean,
                      bias=bias, pcc=pcc)
    return out


def make_figure(var, week_idx, GC):
    wn, ds, de = WEEKS[week_idx]
    wk  = week_idx + 1
    print(f'  computing {var} {wn} ...', flush=True)
    data = compute_gridded(var, week_idx, GC)

    lat, lon = GC['lat'], GC['lon']
    models_avail = [m for m in BASE_MODELS if data.get(m) is not None]
    nrows = len(models_avail)
    ncols = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.8),
                             subplot_kw={'projection': _proj()})
    if nrows == 1:
        axes = axes[None, :]

    # global colour limits
    obs_vals = data['OBS']['obs_mean']
    all_means = [obs_vals] + [data[m]['fcst_mean'] for m in models_avail]
    vmin_mean = np.nanpercentile(np.concatenate([v.ravel() for v in all_means
                                                  if v is not None]), 2)
    vmax_mean = np.nanpercentile(np.concatenate([v.ravel() for v in all_means
                                                  if v is not None]), 98)
    all_bias = [data[m]['bias'] for m in models_avail if data[m] is not None]
    bmax = max(np.nanpercentile(np.abs(v), 95) for v in all_bias if v is not None)
    bmax = max(bmax, 0.1)

    col_titles = ['Observed Mean', 'Forecast Mean', 'Bias (Fcst − Obs)', 'Point-wise PCC\n(across 13 inits)']
    unit = VAR_UNIT[var]

    for ri, m in enumerate(models_avail):
        d = data[m]
        row_label = MODEL_LABEL[m]

        # col 0: observed mean (same for all rows but plot for reference)
        im0 = _pcolormesh(axes[ri, 0], lat, lon, d['obs_mean'],
                          CMAP_MEAN[var], vmin_mean, vmax_mean,
                          title=col_titles[0] if ri == 0 else '',
                          unit=unit)
        _add_regions(axes[ri, 0], GC)

        # col 1: forecast mean
        im1 = _pcolormesh(axes[ri, 1], lat, lon, d['fcst_mean'],
                          CMAP_MEAN[var], vmin_mean, vmax_mean,
                          title=col_titles[1] if ri == 0 else '')
        _add_regions(axes[ri, 1], GC)

        # col 2: bias
        im2 = _pcolormesh(axes[ri, 2], lat, lon, d['bias'],
                          CMAP_BIAS, -bmax, bmax,
                          title=col_titles[2] if ri == 0 else '')
        _add_regions(axes[ri, 2], GC)

        # col 3: PCC
        im3 = _pcolormesh(axes[ri, 3], lat, lon, d['pcc'],
                          CMAP_PCC, -1, 1,
                          title=col_titles[3] if ri == 0 else '')
        _add_regions(axes[ri, 3], GC)

        # row label
        axes[ri, 0].text(-0.12, 0.5, row_label, va='center', ha='right',
                         transform=axes[ri, 0].transAxes, fontsize=9,
                         fontweight='bold', rotation=90)

    # colorbars
    fig.subplots_adjust(left=0.10, right=0.92, top=0.88, bottom=0.12,
                        wspace=0.08, hspace=0.18)
    # mean colorbar
    cax1 = fig.add_axes([0.10, 0.06, 0.52, 0.025])
    cb1  = fig.colorbar(im1, cax=cax1, orientation='horizontal')
    cb1.set_label(f'{VAR_CLABEL[var]}', fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    # bias colorbar
    cax2 = fig.add_axes([0.66, 0.06, 0.12, 0.025])
    cb2  = fig.colorbar(im2, cax=cax2, orientation='horizontal')
    cb2.set_label(f'Bias ({unit})', fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    # pcc colorbar
    cax3 = fig.add_axes([0.82, 0.06, 0.09, 0.025])
    cb3  = fig.colorbar(im3, cax=cax3, orientation='horizontal')
    cb3.set_label('PCC', fontsize=8)
    cb3.ax.tick_params(labelsize=7)

    # IMD region legend
    legend_patches = [Patch(edgecolor=c, facecolor='none', lw=1.2,
                            label=REGION_LABEL[r])
                      for r, c in REGION_COLORS.items()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.00), title='IMD Homogeneous Regions',
               title_fontsize=8)

    fig.suptitle(f'{var} Spatial Maps — {wn}  (JFM 2026, 13 inits)',
                 fontsize=12, y=0.93)

    out = os.path.join(ODIR, f'spatial_{var}_W{wk}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  WROTE {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--var',  nargs='+', default=['TP', 'Z500'],
                    choices=['TP', 'Z500'])
    ap.add_argument('--week', nargs='+', type=int, default=list(range(1, 7)),
                    choices=range(1, 7), metavar='W')
    args = ap.parse_args()

    print('Building grid context ...')
    GC = L.build_grid_context()

    for var in args.var:
        for wk in args.week:
            make_figure(var, wk - 1, GC)

    print('ALL DONE')


if __name__ == '__main__':
    main()
