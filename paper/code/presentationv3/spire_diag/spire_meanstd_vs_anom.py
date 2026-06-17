#!/usr/bin/env python3
"""
spire_meanstd_vs_anom.py
Diagnostic: SPIRE scored TWO ways, to show why we use the `mean_stddev` group
and NOT the pre-computed `anomalies` group.

SPIRE's zarr has two relevant groups:
  mean_stddev : absolute ensemble mean.  We subtract OUR ERA5 climatology.
                  f_abs  = SPIRE_mean
                  f_anom = SPIRE_mean - ERA5_clim(doy)
  anomalies   : SPIRE's OWN pre-computed anomaly (they used their own ERA5
                1991-2020 reference, which differs from ours).
                  f_anom = SPIRE_anom            (used directly)
                  f_abs  = SPIRE_anom + ERA5_clim(doy)   (reconstituted)

Observation anomaly is ALWAYS  o - ERA5_clim  (our climatology), so the only
difference between the two paths is which climatology SPIRE's anomaly is
referenced to.

Expected result (documented in README):
  * ACC   ~ identical          (a constant clim offset cancels in correlation)
  * RMSE  : anomalies path INFLATED (the ~17 gpm / 0.2 mm-day offset does NOT
            cancel in RMSE / bias once you reconstitute absolute fields).

Output (this folder):  spire_acc.png   spire_rmse.png
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.path.dirname(HERE)            # presentationv3/
sys.path.insert(0, PKG)

import loaders as L                      # reuse the real pipeline
from metrics import acc, rmse, bias, cos_latitude_weights

WEEKS = L.WEEKS                          # [('Week 1',1,7), ...]
WLAB  = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
VARS  = ['TP', 'Z500']
VARLONG = {'TP': 'Total Precipitation (mm/day)', 'Z500': 'Z500 (gpm)'}

# anomalies-group variable names (differ from mean_stddev group)
ANOM_VAR = {'TP': 'precipitation_amount', 'Z500': 'geopotential_height'}


def load_spire_anom(init):
    """{var: anomaly DataArray with step dim} from the `anomalies` group."""
    try:
        ds = xr.open_zarr(L.CFG.spire_zarr, group='anomalies').sel(reference_time=init)
    except Exception as e:
        print(f"  [SPIRE-anom] open fail {init}: {e}")
        return None
    out = {}
    try:
        out['TP'] = L.crop_box(ds[ANOM_VAR['TP']])
    except Exception:
        pass
    try:
        z = ds[ANOM_VAR['Z500']].sel(isobar=50000.0)
        out['Z500'] = L.crop_box(z)
    except Exception:
        pass
    return out or None


def week_field(da, ds, de, GC):
    """SPIRE step-slice weekly mean onto the verification grid (matches verify_s2s)."""
    return L.to_grid(da.isel(step=slice(ds - 1, de)).mean('step'), GC)


def main():
    GC = L.build_grid_context()
    w  = cos_latitude_weights(GC['lat'], xr)
    clim = L.open_clim()
    truth = L.open_truth()
    inits = L.CFG.init_dates

    # results[path][var][metric] = list over weeks of per-init means
    rows = []
    for init in inits:
        sp_mean = L.load_spire(init)          # mean_stddev group
        sp_anom = load_spire_anom(init)       # anomalies group
        if sp_mean is None or sp_anom is None:
            print(f"  skip {init} (missing group)")
            continue
        for var in VARS:
            if var not in sp_mean or var not in sp_anom:
                continue
            mu_da, _ = sp_mean[var]
            an_da    = sp_anom[var]
            for (wn, ds, de) in WEEKS:
                valid = L.valid_dates_for(init, ds, de, L.CFG.valid_end)
                if not valid:
                    continue
                doys = [pd.to_datetime(d).dayofyear for d in valid]
                o      = L.truth_period_mean(var, truth, valid, GC)
                if o is None or bool(np.isnan(o).all()):
                    continue
                clim_o = L.clim_field(clim, var, doys, GC)

                # Path A: mean_stddev
                fA_abs  = week_field(mu_da, ds, de, GC)
                # Path B: anomalies (reconstitute absolute = anom + our clim)
                fB_anom = week_field(an_da, ds, de, GC)
                fB_abs  = fB_anom + clim_o

                rows.append(dict(init=init, var=var, week=wn,
                    accA=acc(fA_abs, o, clim_o, w),
                    accB=acc(fB_anom + clim_o, o, clim_o, w),  # same clim baseline
                    rmseA=rmse(fA_abs, o, w),
                    rmseB=rmse(fB_abs, o, w),
                    biasA=bias(fA_abs, o, w),
                    biasB=bias(fB_abs, o, w)))
        print(f"  done {init}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, 'spire_paths.csv'), index=False)
    print(f"\nrows: {len(df)}  ->  spire_paths.csv")
    _plot(df)


def _series(df, var, col):
    s = df[df['var'] == var]
    return [s[s.week == wn][col].mean() for (wn, _, _) in WEEKS]


def _plot(df):
    # ACC
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, var in zip(axes, VARS):
        ax.plot(range(1, 7), _series(df, var, 'accA'), 'o-', color='#1a6faf',
                lw=2.5, ms=9, label='mean_stddev (− our ERA5 clim)')
        ax.plot(range(1, 7), _series(df, var, 'accB'), 's--', color='#e05c2a',
                lw=2.5, ms=8, label='anomalies group (SPIRE own clim)')
        ax.axhline(0.5, color='gray', ls=':', lw=1.2)
        ax.set_title(VARLONG[var]); ax.set_xticks(range(1, 7))
        ax.set_xticklabels(WLAB); ax.set_xlabel('Forecast Week')
        ax.set_ylabel('ACC'); ax.grid(alpha=0.3)
    fig.suptitle('SPIRE ACC — mean_stddev vs anomalies group  (≈ identical)',
                 fontsize=15, fontweight='bold')
    axes[0].legend(loc='lower left', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, 'spire_acc.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved spire_acc.png')

    # RMSE
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, var in zip(axes, VARS):
        ax.plot(range(1, 7), _series(df, var, 'rmseA'), 'o-', color='#1a6faf',
                lw=2.5, ms=9, label='mean_stddev (− our ERA5 clim)')
        ax.plot(range(1, 7), _series(df, var, 'rmseB'), 's--', color='#e05c2a',
                lw=2.5, ms=8, label='anomalies group (SPIRE own clim)')
        ax.set_title(VARLONG[var]); ax.set_xticks(range(1, 7))
        ax.set_xticklabels(WLAB); ax.set_xlabel('Forecast Week')
        ax.set_ylabel('RMSE'); ax.grid(alpha=0.3)
    fig.suptitle('SPIRE RMSE — mean_stddev vs anomalies group  '
                 '(anomalies path inflated by clim offset)',
                 fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, 'spire_rmse.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved spire_rmse.png')


if __name__ == '__main__':
    main()
