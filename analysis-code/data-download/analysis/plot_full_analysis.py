#!/usr/bin/env python
"""
FULL S2S BENCHMARK ANALYSIS SUITE
Organized into 3 sections:
  01_skill/     – RMSE & ACC skill vs lead time line plots
  02_bias_maps/ – Spatial mean bias maps per model per week
  03_summary/   – Taylor diagram, skill drop-off bar chart, India RMSE
"""
import os, warnings
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from arraylake import Client
import pandas as pd

warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.titlepad': 12,
    'axes.labelsize': 12,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.1,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.grid': True,
    'grid.color': '#eeeeee',
    'grid.linewidth': 0.8,
    'grid.linestyle': '-',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'xtick.minor.visible': True,
    'legend.frameon': False,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

PALETTE = {'ECMWF': '#3B82F6', 'NCEP': '#10B981', 'Spire AI': '#F97316'}
LS = {'ECMWF': '--', 'NCEP': '-.', 'Spire AI': '-'}
LW = {'ECMWF': 2.2, 'NCEP': 2.2, 'Spire AI': 3.0}
MODELS = ['ECMWF', 'NCEP', 'Spire AI']

# India bounding box
IND = dict(lat_min=5, lat_max=40, lon_min=65, lon_max=100)

# ─── Metrics ─────────────────────────────────────────────────────────────────
def calc_rmse(f, o):
    return float(np.sqrt(((f - o)**2).mean(dim=['latitude','longitude'])))

def calc_rmse_ts(f, o):
    return np.sqrt(((f - o)**2).mean(dim=['latitude','longitude']))

def calc_acc(f, o):
    fp = f - f.mean(dim=['latitude','longitude'])
    op = o - o.mean(dim=['latitude','longitude'])
    cov = (fp * op).mean(dim=['latitude','longitude'])
    return cov / (np.sqrt((fp**2).mean(dim=['latitude','longitude'])) *
                  np.sqrt((op**2).mean(dim=['latitude','longitude'])))

def calc_bias(f, o):
    return (f - o).mean(dim='day')

def area_avg(da):
    w = np.cos(np.deg2rad(da.latitude))
    return da.weighted(w).mean(dim=['latitude','longitude'])

# ─── Data loaders ────────────────────────────────────────────────────────────
def load_grib(path, var, level=None):
    filt = {'shortName': var}
    if level: filt['level'] = level
    try:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'filter_by_keys': filt})
        return ds[var]
    except Exception as e:
        print(f'  ⚠ GRIB {var}: {e}')
        return None

def to_daily(da, agg):
    hrs  = da.step.values / np.timedelta64(1, 'h')
    days = (hrs - 0.1) // 24 + 1
    da   = da.assign_coords(day=('step', days))
    return getattr(da.groupby('day'), {'max':'max','min':'min','mean':'mean'}[agg])('step')

def load_and_align(cfg, benchmark_dir, ds_spire, ds_era5, max_days=42):
    """Returns (ecmwf_d, ncep_d, spire_d, era5_d) all on same grid, day coord 1..N"""
    ecmwf_path = os.path.join(benchmark_dir, 'ecmwf', 'data', cfg['grib_file'])
    ncep_path  = os.path.join(benchmark_dir, 'ncep',  'data', cfg['grib_file'])

    da_e = load_grib(ecmwf_path, cfg['grib_var'], cfg.get('grib_level'))
    da_n = load_grib(ncep_path,  cfg['grib_var'], cfg.get('grib_level'))
    if da_e is None or da_n is None:
        return None

    ecmwf_d = to_daily(da_e, cfg['agg']) + cfg.get('offset', 0)
    ncep_d  = to_daily(da_n, cfg['agg']) + cfg.get('offset', 0)
    avail   = min(max_days, int(ecmwf_d.day.max()), int(ncep_d.day.max()))
    ecmwf_d = ecmwf_d.sel(day=slice(1, avail))
    ncep_d  = ncep_d.sel(day=slice(1, avail))

    steps     = [np.timedelta64(d, 'D') for d in range(1, avail+1)]
    spire_raw = ds_spire[cfg['spire_var']].sel(
        reference_time='2026-01-01', step=steps).compute()
    if cfg.get('spire_level'):
        dim = 'isobar' if 'isobar' in spire_raw.dims else 'isobaricInhPa'
        spire_raw = spire_raw.sel({dim: cfg['spire_level']}, method='nearest')
    if spire_raw.latitude.values[0] > spire_raw.latitude.values[-1]:
        spire_raw = spire_raw.isel(latitude=slice(None, None, -1))
    spire_raw = spire_raw.sel(latitude=slice(0,50), longitude=slice(55,105))
    spire_raw = spire_raw * cfg.get('spire_scale', 1) + cfg.get('offset', 0)

    end_dt  = pd.Timestamp('2026-01-01') + pd.Timedelta(days=avail)
    era_raw = ds_era5[cfg['era5_var']].sel(
        latitude=slice(51,-1), longitude=slice(54,106),
        time=slice('2026-01-02T00:00', end_dt.strftime('%Y-%m-%dT23:00')))
    if cfg.get('era5_level'):
        era_raw = era_raw.sel(level=cfg['era5_level'])
    era_d = getattr(era_raw.resample(time='1D'), cfg.get('era5_agg','mean'))('time').compute()
    era_d = era_d.isel(time=slice(0, avail))
    era_d = era_d * cfg.get('era5_scale', 1) + cfg.get('offset', 0)

    spire_d = spire_raw.rename({'step':'day'}).assign_coords(day=np.arange(1, avail+1))
    era_d   = era_d.assign_coords(time=np.arange(1, avail+1)).rename({'time':'day'})
    ref_lat, ref_lon = ecmwf_d.latitude, ecmwf_d.longitude
    spire_d = spire_d.interp(latitude=ref_lat, longitude=ref_lon)
    era_d   = era_d.interp(latitude=ref_lat,   longitude=ref_lon)

    return ecmwf_d, ncep_d, spire_d, era_d


# ─── Plot helpers ─────────────────────────────────────────────────────────────
def skill_plot(out_path, series, y_label, title, metric='rmse', unit=''):
    fig, ax = plt.subplots(figsize=(9, 5))
    if metric == 'acc':
        ax.axhline(0.6, color='#aaa', lw=1.2, ls=':', zorder=1)
        ax.text(41, 0.615, 'Skill threshold', ha='right', va='bottom',
                color='#999', fontsize=10)
    for m, da in series.items():
        x, y = da.day.values, da.values
        ax.plot(x, y, color=PALETTE[m], lw=LW[m], ls=LS[m], label=m, zorder=3)
        ax.scatter(x[6::7], y[6::7], color=PALETTE[m], s=28, zorder=4, linewidths=0)

    ax.set_xlim(1, int(list(series.values())[0].day.max()))
    ax.set_xlabel('Forecast Lead Time (days)')
    ax.set_ylabel(y_label + (f' ({unit})' if unit else ''))
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    handles = [Line2D([0],[0],color=PALETTE[m],lw=LW[m],ls=LS[m],label=m) for m in series]
    ax.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ {os.path.basename(out_path)}')


def bias_map(out_path, bias_da, title, unit, cmap='RdBu_r', vmax=None):
    """Single bias map with colorbar."""
    fig, ax = plt.subplots(figsize=(9, 4.5),
                           subplot_kw={'projection': None})
    ax.set_facecolor('#eef2f7')
    lons = bias_da.longitude.values
    lats = bias_da.latitude.values
    data = bias_da.values

    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(data), 95))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(unit, fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # India bounding box
    from matplotlib.patches import Rectangle
    rect = Rectangle((IND['lon_min'], IND['lat_min']),
                      IND['lon_max'] - IND['lon_min'],
                      IND['lat_max'] - IND['lat_min'],
                      linewidth=1.5, edgecolor='#222', facecolor='none', zorder=5)
    ax.add_patch(rect)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ {os.path.basename(out_path)}')


def taylor_diagram(out_path, ref, forecasts, title, unit):
    """
    Taylor diagram on a polar axes.
    forecasts: dict of {model: da (day x lat x lon)}
    ref: da (day x lat x lon)
    """
    # Flatten to (time, space)
    def flatten(da):
        return da.values.reshape(da.shape[0], -1)

    r_flat = flatten(ref)
    ref_std = float(np.std(r_flat))

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Reference std arc
    t = np.linspace(0, np.pi, 200)
    ax.plot(t, np.full_like(t, ref_std), 'k--', lw=1.2, alpha=0.4)
    ax.plot(np.pi/2, ref_std, 'k*', ms=12, label='ERA5 (Ref)', zorder=5)

    # RMSE arcs
    for rmse_v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        scale = rmse_v * ref_std
        theta_arc = np.linspace(0, np.pi, 200)
        r_arc = []
        for th in theta_arc:
            # r² + ref² - 2r·ref·cos(θ) = rmse²
            a = 1
            b = -2 * ref_std * np.cos(th)
            c = ref_std**2 - scale**2
            disc = b**2 - 4*a*c
            if disc < 0: r_arc.append(np.nan); continue
            r1 = (-b + np.sqrt(disc)) / 2
            r_arc.append(r1 if r1 >= 0 else np.nan)
        ax.plot(theta_arc, r_arc, color='#ccc', lw=0.8, zorder=0)

    for model, da in forecasts.items():
        f_flat = flatten(da)
        std_f  = float(np.std(f_flat))
        # correlation
        r      = float(np.corrcoef(r_flat.ravel(), f_flat.ravel())[0,1])
        theta  = np.arccos(np.clip(r, -1, 1))
        ax.scatter(theta, std_f, color=PALETTE[model], s=120, zorder=6, label=model)
        ax.annotate(model, xy=(theta, std_f), xytext=(5, 5),
                    textcoords='offset points', fontsize=9,
                    color=PALETTE[model])

    # Axes labels
    ax.set_rlabel_position(135)
    ax.set_xlabel('Standard Deviation', labelpad=20)
    corr_ticks = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    theta_ticks = np.arccos(corr_ticks)
    ax.set_thetagrids(np.degrees(theta_ticks),
                      [str(c) for c in corr_ticks], fontsize=9)

    ax.set_title(title, pad=18)
    ax.legend(loc='lower left', bbox_to_anchor=(0.85, 0.02))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ {os.path.basename(out_path)}')


def skill_dropoff_bar(out_path, dropoff_dict):
    """Bar chart: day when ACC first drops below 0.6."""
    vars_list  = list(list(dropoff_dict.values())[0].keys())
    x = np.arange(len(vars_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(MODELS):
        days = [dropoff_dict[model].get(v, 42) for v in vars_list]
        bars = ax.bar(x + i*width - width, days, width,
                      label=model, color=PALETTE[model], alpha=0.88)
        for bar, d in zip(bars, days):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'D{int(d)}', ha='center', va='bottom', fontsize=10,
                    color='#333')

    ax.set_xlabel('Variable')
    ax.set_ylabel('Lead Day (ACC drops below 0.6)')
    ax.set_title('Forecast Skill Drop-off Day by Variable & Model', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(vars_list)
    ax.set_ylim(0, 47)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ {os.path.basename(out_path)}')


def india_rmse_plot(out_path, india_rmse_dict):
    """Line plot of India-domain RMSE per variable."""
    fig, axes = plt.subplots(1, len(india_rmse_dict), figsize=(5*len(india_rmse_dict), 5),
                             sharey=False)
    if len(india_rmse_dict) == 1:
        axes = [axes]
    for ax, (var_name, series) in zip(axes, india_rmse_dict.items()):
        for model, da in series.items():
            x, y = da.day.values, da.values
            ax.plot(x, y, color=PALETTE[model], lw=LW[model], ls=LS[model], label=model)
        ax.set_title(var_name)
        ax.set_xlabel('Lead Day')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color='#eeeeee', lw=0.8)

    axes[0].set_ylabel('India RMSE')
    handles = [Line2D([0],[0],color=PALETTE[m],lw=2,ls=LS[m],label=m) for m in MODELS]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Indian Subcontinent RMSE by Variable', fontsize=14,
                 fontweight='bold', y=1.07)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ {os.path.basename(out_path)}')


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    analysis_dir  = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.dirname(analysis_dir)
    base_out = os.path.join(analysis_dir, 'figures', 'full_analysis')

    dirs = {
        'skill':    os.path.join(base_out, '01_skill'),
        'bias':     os.path.join(base_out, '02_bias_maps'),
        'summary':  os.path.join(base_out, '03_summary'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print('Connecting to data stores …')
    client   = Client()
    repo     = client.get_repo('artamta/s2s-research')
    session  = repo.readonly_session('main')
    ds_spire = xr.open_zarr(session.store, group='mean_stddev')
    ds_era5  = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
        storage_options={'token': 'anon'})

    VARS = [
        dict(name='Tmax', unit='°C',
             grib_file='sfc_cf_20260101_6h.grib', grib_var='mx2t6',
             spire_var='air_temperature_max',
             era5_var='2m_temperature', era5_agg='max',
             offset=-273.15, agg='max'),
        dict(name='Tmin', unit='°C',
             grib_file='sfc_cf_20260101.grib', grib_var='mn2t6',
             spire_var='air_temperature_min',
             era5_var='2m_temperature', era5_agg='min',
             offset=-273.15, agg='min'),
        dict(name='Z500', unit='m',
             grib_file='pl_cf_20260101.grib', grib_var='gh', grib_level=500,
             spire_var='geopotential_height_at_isobaric_levels', spire_level=500,
             era5_var='geopotential', era5_level=500, era5_scale=1/9.80665,
             era5_agg='mean', agg='mean'),
    ]

    all_data   = {}   # {var_name: (e_d, n_d, s_d, era_d)}
    dropoff    = {m: {} for m in MODELS}
    india_rmse = {}

    for cfg in VARS:
        vname = cfg['name']
        print(f'\n══ {vname} ══')
        result = load_and_align(cfg, benchmark_dir, ds_spire, ds_era5)
        if result is None:
            print(f'  Skipping {vname}'); continue

        e_d, n_d, s_d, era_d = result
        all_data[vname] = result

        # ── 1. Skill line plots ──────────────────────────────────────────────
        rmse_s = {m: calc_rmse_ts(d, era_d)
                  for m, d in zip(MODELS, [e_d, n_d, s_d])}
        acc_s  = {m: calc_acc(d, era_d)
                  for m, d in zip(MODELS, [e_d, n_d, s_d])}

        skill_plot(os.path.join(dirs['skill'], f'{vname.lower()}_rmse.png'),
                   rmse_s, 'RMSE', f'{vname} – Forecast Error vs Lead Time',
                   'rmse', cfg['unit'])
        skill_plot(os.path.join(dirs['skill'], f'{vname.lower()}_acc.png'),
                   acc_s, 'Spatial Correlation (ACC)',
                   f'{vname} – Forecast Skill vs Lead Time', 'acc')

        # ── Skill drop-off day ───────────────────────────────────────────────
        for m, acc_da in acc_s.items():
            below = acc_da.where(acc_da < 0.6, drop=True)
            dropoff[m][vname] = int(below.day.min().item()) if len(below) else 42

        # ── 2. Bias maps (weekly windows) ────────────────────────────────────
        windows = [('Week1', 1, 7), ('Week2', 8, 14),
                   ('Week3-4', 15, 28), ('Week5-6', 29, 42)]
        max_day = int(e_d.day.max())
        bias_vmaxes = {'Tmax': 4, 'Tmin': 4, 'Z500': 80}
        vmax = bias_vmaxes.get(vname, None)

        for model, da in zip(MODELS, [e_d, n_d, s_d]):
            for wlabel, d1, d2 in windows:
                d2 = min(d2, max_day)
                if d1 > max_day: continue
                fcst_w = da.sel(day=slice(d1, d2)).mean('day')
                era_w  = era_d.sel(day=slice(d1, d2)).mean('day')
                bias_w = fcst_w - era_w
                fname  = f'{vname.lower()}_{model.lower().replace(" ","_")}_{wlabel}.png'
                bias_map(os.path.join(dirs['bias'], fname),
                         bias_w,
                         f'{vname} Mean Bias — {model} {wlabel}',
                         cfg['unit'], vmax=vmax)

        # ── 3a. India RMSE ───────────────────────────────────────────────────
        def india_clip(da):
            return da.sel(latitude=slice(IND['lat_max'], IND['lat_min']),
                          longitude=slice(IND['lon_min'], IND['lon_max']))

        india_rmse[vname] = {}
        for model, da in zip(MODELS, [e_d, n_d, s_d]):
            fi = india_clip(da)
            oi = india_clip(era_d)
            india_rmse[vname][model] = calc_rmse_ts(fi, oi)

    # ── 3b. Taylor diagram (use all avail lead times combined) ───────────────
    print('\n── Taylor Diagram ──')
    for vname, (e_d, n_d, s_d, era_d) in all_data.items():
        taylor_diagram(
            os.path.join(dirs['summary'], f'taylor_{vname.lower()}.png'),
            era_d,
            {'ECMWF': e_d, 'NCEP': n_d, 'Spire AI': s_d},
            f'Taylor Diagram – {vname}',
            VARS[[c['name'] for c in VARS].index(vname)]['unit']
        )

    # ── 3c. Skill drop-off bar chart ─────────────────────────────────────────
    print('\n── Skill Drop-off Chart ──')
    if any(dropoff[m] for m in MODELS):
        skill_dropoff_bar(os.path.join(dirs['summary'], 'skill_dropoff.png'), dropoff)

    # ── 3d. India RMSE multi-panel ───────────────────────────────────────────
    print('\n── India RMSE Panel ──')
    if india_rmse:
        india_rmse_plot(os.path.join(dirs['summary'], 'india_rmse.png'), india_rmse)

    print('\n\n✅  All done! Results saved to:', base_out)


if __name__ == '__main__':
    main()
