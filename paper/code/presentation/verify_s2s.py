#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_s2s.py  —  Unified S2S verification DRIVER (deterministic + probabilistic).
================================================================================
Consolidates the old compute/03+04 (deterministic) and compute/06 (probabilistic)
into ONE parallel pass. No plotting — writes tidy CSVs into this folder for
downstream figures. Maths live in metrics.py; data access + CONFIG live in
loaders.py. Edit loaders.Config to retarget a new period (e.g. JJAS 2019).

OUTPUTS (this folder)
  skill_deterministic.csv  per (variable, model, region, init, scale, lead):
                           pcc, rmse, bias, msss_clim, msss_pers,
                           fcst_std, obs_std, std_ratio, fcst_mean, obs_mean
  skill_probabilistic.csv  per (variable, model[base 4], region, init, scale, lead):
                           crps, crps_clim, crps_pers, crpss_clim, crpss_pers,
                           spread, rmse, ssr
  skill_brier.csv          per (variable, model, region, init, scale=weekly, lead,
                           event): brier, brier_clim, briss_clim, base_rate
  reliability.npz          reliability bins per (event, model): obs_freq, fcst_p, count
  RUN_METADATA.txt         provenance (config, units, row counts, timestamp)

  scale='weekly' -> Week 1..6 means (HEADLINE S2S pattern skill)
  scale='daily'  -> forecast lead day 1..42 (finer DIAGNOSTIC; ACC noisier)
  lead = week index (weekly) OR forecast day (daily)

RUN
  python verify_s2s.py --test                 # 1 init smoke test (login-node OK)
  python verify_s2s.py --workers 13           # full run (use SLURM, not login)
  python verify_s2s.py --vars TP Z500         # subset of variables
  sbatch run_slurm.sh                          # production submission
================================================================================
"""
import os
import sys
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import metrics as M
import loaders as L
from loaders import (CFG, WEEKS, REGIONS, BASE_SYSTEMS, DET_MODELS, VARS)

NB = 10  # reliability bins


# ==============================================================================
# Brier event definitions  (Gaussian forecast probabilities)
# ==============================================================================
def brier_events(var, mu, sig, o, cm, cs):
    """Yield (event_name, P_fcst array, binary outcome array, clim base-rate array).
       cs = per-grid-point CLIMATOLOGICAL spread array (same shape as the grid).
       All arrays are numpy on the grid (NaN where masked)."""
    out = []
    muv, sigv, ov, cmv = mu.values, sig.values, o.values, cm.values
    csv = np.asarray(cs)
    if var == 'TP':
        for thr in CFG.tp_thresholds:
            p = M.prob_exceed(thr, muv, sigv)
            y = (ov > thr).astype(float)
            base = M.prob_exceed(thr, cmv, csv)
            out.append((f'tp_gt_{int(thr)}mm', p, y, base))
    if CFG.use_terciles:
        # Gaussian tercile boundaries about the climatological mean, per grid point:
        #   +/- 0.4307 * clim_sigma  (norm.ppf(2/3) = 0.4307)
        lo = cmv - 0.4307 * csv
        hi = cmv + 0.4307 * csv
        p_above = M.prob_exceed(0.0, muv - hi, sigv)
        p_below = M.prob_below(0.0, muv - lo, sigv)
        y_above = (ov > hi).astype(float)
        y_below = (ov < lo).astype(float)
        third = np.full_like(cmv, 1.0 / 3.0)
        out.append((f'{var.lower()}_above_normal', p_above, y_above, third))
        out.append((f'{var.lower()}_below_normal', p_below, y_below, third))
    return out


# ==============================================================================
# Build per-system fields for ONE (init, scale, lead) slice
#   Returns det_field {model: mean DataArray} and mu_sig {base model: (mu, sig)}
# ==============================================================================
def assemble_fields(var, kind, sl, spire, fuxi, op, GC):
    """kind='weekly' -> sl=(ds, de); kind='daily' -> sl=di (0-based lead index)."""
    det_field, mu_sig = {}, {}

    # ---- SPIRE (provider mean + stddev) ----
    if spire is not None and var in spire:
        mu_da, sg_da = spire[var]
        if kind == 'weekly':
            ds, de = sl
            m = L.to_grid(mu_da.isel(step=slice(ds - 1, de)).mean('step'), GC)
            s = L.to_grid(sg_da.isel(step=slice(ds - 1, de)).mean('step'), GC)
        else:
            m = L.to_grid(mu_da.isel(step=sl), GC)
            s = L.to_grid(sg_da.isel(step=sl), GC)
        det_field['SPIRE'] = m
        mu_sig['SPIRE'] = (m, s)

    # ---- FuXi (11 members) ----
    if kind == 'weekly':
        ds, de = sl
        fd = [fuxi[var][d] for d in range(ds, de + 1) if d in fuxi[var]]
        ens = xr.concat(fd, 't').mean('t') if fd else None
    else:
        day = sl + 1
        ens = fuxi[var].get(day)
    if ens is not None:
        if var == 'TP':
            ens = ens * CFG.fuxi_tp_factor
        mu, sig = L.ens_mean_std(ens, 'member', GC)
        det_field['FuXi'] = mu
        mu_sig['FuXi'] = (mu, sig)

    # ---- ECMWF / NCEP (members) ----
    for m in ('ECMWF', 'NCEP'):
        arr = op[m][var]
        if arr is None:
            continue
        if kind == 'weekly':
            ds, de = sl
            if arr.sizes.get('step', 0) < de:
                continue
            field = (L.weekly_mean_cumulative(arr, ds, de) if var == 'TP'
                     else arr.isel(step=slice(ds - 1, de)).mean('step'))
        else:
            if arr.sizes.get('step', 0) <= sl:
                continue
            field = (L.daily_from_cumulative(arr, sl) if var == 'TP' else arr.isel(step=sl))
        mu, sig = L.ens_mean_std(field, 'number', GC)
        det_field[m] = mu
        mu_sig[m] = (mu, sig)

    return det_field, mu_sig


# ==============================================================================
# CORE WORKER  —  all rows for ONE init date (runs in its own process)
# ==============================================================================
def process_init(init, want_vars):
    GC = L.build_grid_context()
    w = M.cos_latitude_weights(GC['lat'], xr)
    clim_ds = L.open_clim()
    truth = L.open_truth()
    init_str = pd.to_datetime(init).strftime('%Y%m%d')
    print(f"=== INIT {init} (pid {os.getpid()}) ===", flush=True)

    det_rows, prob_rows, brier_rows = [], [], []
    rel_accum = {}  # (event, model) -> (3, NB)

    spire = L.load_spire(init)
    # single-pass FuXi load: open each member-day file ONCE for all variables
    fuxi = L.load_fuxi_all(init_str, want_vars, CFG.G)
    op = {m: {v: L.load_op(m, init_str, v, CFG.G) for v in want_vars}
          for m in ('ECMWF', 'NCEP')}
    pers = {v: L.persistence_field(v, truth, init, GC) for v in want_vars}
    # per-grid-point climatological spread (temporal std) -> CRPSS clim reference
    clim_sigma = {v: L.clim_spread_field(v, truth, GC) for v in want_vars}

    # ---------------- helpers bound to this init ----------------
    def det_row(var, m, f, o, clim, persf, scale, lead, week):
        for rg in REGIONS:
            rda = L.region_da(rg, GC)
            fr, orr, cr = f.where(rda), o.where(rda), clim.where(rda)
            pr = persf.where(rda) if persf is not None else None
            row = dict(season=CFG.season_label, variable=var, model=m, region=rg,
                       init_date=init, scale=scale, lead=lead, week=week,
                       pcc=M.acc(fr, orr, cr, w), rmse=M.rmse(fr, orr, w),
                       bias=M.bias(fr, orr, w),
                       msss_clim=M.msss(fr, orr, cr, w),
                       msss_pers=(M.msss(fr, orr, pr, w) if pr is not None else np.nan),
                       fcst_std=M.wstd_anom(fr - cr, w), obs_std=M.wstd_anom(orr - cr, w),
                       fcst_mean=M._f(M.wmean(fr, w)), obs_mean=M._f(M.wmean(orr, w)))
            row['std_ratio'] = M.std_ratio(fr, orr, cr, w)
            det_rows.append(row)

    def prob_brier_rows(var, m, mu, sig, o, cm, cs, persf, scale, lead, week, do_brier):
        sig = sig.clip(min=CFG.sig_floor[var])
        cs = cs.clip(min=CFG.sig_floor[var])
        for rg in REGIONS:
            rda = L.region_da(rg, GC)
            muR, sgR, oR, cmR = mu.where(rda), sig.where(rda), o.where(rda), cm.where(rda)
            csR = cs.where(rda)
            prR = persf.where(rda) if persf is not None else None
            # CRPS (model)
            crps_g = muR.copy(data=M.crps_gauss(muR.values, sgR.values, oR.values))
            crps_m = M._f(M.wmean(crps_g, w))
            # CRPS (climatology ref): Gaussian(clim mean, per-gridpoint clim spread)
            crps_c_g = muR.copy(data=M.crps_gauss(cmR.values, csR.values, oR.values))
            crps_cl = M._f(M.wmean(crps_c_g, w))
            # CRPS (persistence ref): deterministic field, floor spread
            if prR is not None:
                crps_p_g = muR.copy(data=M.crps_gauss(
                    prR.values, np.full_like(prR.values, CFG.sig_floor[var]), oR.values))
                crps_p = M._f(M.wmean(crps_p_g, w))
            else:
                crps_p = np.nan
            rmse_m = M.rmse(muR, oR, w)
            spread_m = M._f(M.wmean(sgR, w))
            prob_rows.append(dict(
                season=CFG.season_label, variable=var, model=m, region=rg,
                init_date=init, scale=scale, lead=lead, week=week,
                crps=crps_m, crps_clim=crps_cl, crps_pers=crps_p,
                crpss_clim=M.crpss(crps_m, crps_cl), crpss_pers=M.crpss(crps_m, crps_p),
                spread=spread_m, rmse=rmse_m, ssr=M.ssr(spread_m, rmse_m)))
            # Brier events (weekly only)
            if do_brier:
                for ev, pf, yy, base in brier_events(var, muR, sgR, oR, cmR, csR.values):
                    bs = M.brier_score(pf, yy, w, muR)
                    bs_c = M.brier_clim(base, yy, w, muR)
                    brier_rows.append(dict(
                        season=CFG.season_label, variable=var, model=m, region=rg,
                        init_date=init, scale=scale, lead=lead, week=week, event=ev,
                        brier=bs, brier_clim=bs_c, briss_clim=M.briss(bs, bs_c),
                        base_rate=M._f(M.wmean(muR.copy(data=yy), w))))
                    if rg == 'All India':
                        key = (ev, m)
                        rel_accum.setdefault(key, np.zeros((3, NB)))
                        M.accumulate_reliability(rel_accum[key], pf, yy, NB)

    # ============ WEEKLY ============
    for wi, (wn, ds, de) in enumerate(WEEKS):
        wk = wi + 1
        valid = L.valid_dates_for(init, ds, de, CFG.valid_end)
        valid_c = L.valid_dates_for(init, ds, de, CFG.valid_end_clim)
        if not valid or not valid_c:
            continue
        doys = [pd.to_datetime(d).dayofyear for d in valid]
        for var in want_vars:
            o = L.truth_period_mean(var, truth, valid, GC)
            if o is None or bool(np.isnan(o).all()):
                continue
            clim = L.clim_field(clim_ds, var, doys, GC)
            persf = pers[var]
            det_field, mu_sig = assemble_fields(var, 'weekly', (ds, de), spire, fuxi, op, GC)
            # MME (>=3 base systems) + Persistence -> deterministic only
            base = [det_field[m] for m in BASE_SYSTEMS if m in det_field]
            if len(base) >= 3:
                det_field['MME'] = xr.concat(base, 'mm').mean('mm')
            if persf is not None and not bool(np.isnan(persf).all()):
                det_field['Persistence'] = persf
            for m in DET_MODELS:
                if m in det_field:
                    det_row(var, m, det_field[m], o, clim, persf, 'weekly', wk, wn)
            for m in BASE_SYSTEMS:
                if m in mu_sig:
                    mu, sig = mu_sig[m]
                    prob_brier_rows(var, m, mu, sig, o, clim, clim_sigma[var], persf,
                                    'weekly', wk, wn, True)

    # ============ DAILY ============
    for di in range(42):
        day = di + 1
        date = (pd.to_datetime(init) + pd.Timedelta(days=di)).strftime('%Y-%m-%d')
        if date > CFG.valid_end:
            break
        wn = f'Week {min(di // 7 + 1, 6)}'
        doy = [pd.to_datetime(date).dayofyear]
        for var in want_vars:
            o = L.truth_day(var, truth, date, GC)
            if o is None:
                continue
            clim = L.clim_field(clim_ds, var, doy, GC)
            persf = pers[var]
            det_field, mu_sig = assemble_fields(var, 'daily', di, spire, fuxi, op, GC)
            base = [det_field[m] for m in BASE_SYSTEMS if m in det_field]
            if len(base) >= 3:
                det_field['MME'] = xr.concat(base, 'mm').mean('mm')
            if persf is not None and not bool(np.isnan(persf).all()):
                det_field['Persistence'] = persf
            for m in DET_MODELS:
                if m in det_field:
                    det_row(var, m, det_field[m], o, clim, persf, 'daily', day, wn)
            for m in BASE_SYSTEMS:
                if m in mu_sig:
                    mu, sig = mu_sig[m]
                    prob_brier_rows(var, m, mu, sig, o, clim, clim_sigma[var], persf,
                                    'daily', day, wn, False)

    rel_out = [(ev, m, arr) for (ev, m), arr in rel_accum.items()]
    return det_rows, prob_rows, brier_rows, rel_out


# ==============================================================================
# DRIVER
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description='Unified S2S verification pipeline.')
    ap.add_argument('--test', action='store_true', help='first init only (smoke test)')
    ap.add_argument('--workers', type=int, default=8, help='parallel worker processes')
    ap.add_argument('--vars', nargs='+', default=VARS, choices=VARS, help='variables')
    args = ap.parse_args()

    inits = list(CFG.init_dates[:1] if args.test else CFG.init_dates)
    want_vars = list(args.vars)
    n_workers = max(1, min(args.workers, len(inits)))
    print(f"\nUnified S2S verification — season={CFG.season_label}", flush=True)
    print(f"  inits={len(inits)}  vars={want_vars}  workers={n_workers}", flush=True)
    print(f"  out_dir={HERE}\n", flush=True)

    det_all, prob_all, brier_all = [], [], []
    rel_merge = {}

    if n_workers <= 1:
        results = [process_init(init, want_vars) for init in inits]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_init, init, want_vars): init for init in inits}
            for fut in as_completed(futs):
                init = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"  !! init {init} FAILED: {e}", flush=True)

    for det_rows, prob_rows, brier_rows, rel_out in results:
        det_all.extend(det_rows)
        prob_all.extend(prob_rows)
        brier_all.extend(brier_rows)
        for ev, m, arr in rel_out:
            rel_merge.setdefault((ev, m), np.zeros((3, NB)))
            rel_merge[(ev, m)] += arr

    _write_csv(det_all, ['variable', 'model', 'region', 'scale', 'lead', 'init_date'],
               'skill_deterministic.csv')
    _write_csv(prob_all, ['variable', 'model', 'region', 'scale', 'lead', 'init_date'],
               'skill_probabilistic.csv')
    _write_csv(brier_all, ['variable', 'event', 'model', 'region', 'scale', 'lead', 'init_date'],
               'skill_brier.csv')

    # reliability
    rel_npz = {}
    for (ev, m), arr in rel_merge.items():
        cnt = arr[1]
        ok = cnt > 0
        rel_npz[f'{ev}__{m}__obs_freq'] = np.where(ok, arr[0] / np.maximum(cnt, 1), np.nan)
        rel_npz[f'{ev}__{m}__fcst_p'] = np.where(ok, arr[2] / np.maximum(cnt, 1), np.nan)
        rel_npz[f'{ev}__{m}__count'] = cnt
    np.savez_compressed(os.path.join(HERE, 'reliability.npz'), nbins=NB, **rel_npz)
    print(f"WROTE reliability.npz ({len(rel_merge)} event-model curves)", flush=True)

    det_df = pd.DataFrame(det_all)
    prob_df = pd.DataFrame(prob_all)
    brier_df = pd.DataFrame(brier_all)
    _write_metadata(det_df, prob_df, brier_df, inits, want_vars)
    _headline(det_df, prob_df)
    print("\nVERIFY_DONE", flush=True)


def _write_csv(rows, sort_cols, name):
    df = pd.DataFrame(rows)
    if len(df):
        keep = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(keep).reset_index(drop=True)
    df.to_csv(os.path.join(HERE, name), index=False)
    print(f"WROTE {name} ({len(df)} rows)", flush=True)


def _write_metadata(det_df, prob_df, brier_df, inits, want_vars):
    lines = [
        "S2S verification run metadata", "=" * 60,
        f"timestamp        : {datetime.now().isoformat(timespec='seconds')}",
        f"season_label     : {CFG.season_label}",
        f"n_inits          : {len(inits)}",
        f"init_dates       : {', '.join(inits)}",
        f"variables        : {', '.join(want_vars)}",
        f"regions          : {', '.join(REGIONS)}",
        f"grid             : {CFG.dgrid} deg, lat {CFG.lat0}->{CFG.lat1}, lon {CFG.lon0}->{CFG.lon1}",
        f"climatology      : {CFG.clim_path} (30-yr WMO, day-of-year)",
        f"FuXi tp factor   : x{CFG.fuxi_tp_factor} (mm/h -> mm/day)",
        f"sigma floors     : {CFG.sig_floor}",
        f"tp thresholds    : {CFG.tp_thresholds} mm/day; terciles={CFG.use_terciles}",
        "",
        "Units (verification): TP mm/day, Z500 gpm, T2M K",
        "Z500 = geopotential/9.80665; TP truth = true 24-h ERA5 daily total;",
        "ECMWF/NCEP TP cumulative -> differenced to mm/day.",
        "",
        "T2M definitions per system (all scored as ANOMALIES vs ERA5 daily-mean clim;",
        "constant instantaneous-vs-daily-mean offset cancels in anomalies; raw bias",
        "should be read with this caveat):",
    ]
    for m, d in CFG.t2m_definition.items():
        lines.append(f"  {m:6s}: {d}")
    lines += [
        "  ERA5  : daily mean of hourly 2t (truth)",
        "  (ECMWF instantaneous '2t' is analysis-only, no forecast steps -> not usable.)",
        "",
        "SPIRE: provider mean (deterministic) + stddev (Gaussian spread).",
        "Probabilistic framework: Gaussian(ensemble mean, spread), fair across",
        "SPIRE summary / FuXi 11 / ECMWF 100 / NCEP 15 members.",
        "",
        f"rows: deterministic={len(det_df)}  probabilistic={len(prob_df)}  brier={len(brier_df)}",
    ]
    with open(os.path.join(HERE, 'RUN_METADATA.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print("WROTE RUN_METADATA.txt", flush=True)


def _headline(det_df, prob_df):
    if not len(det_df):
        return
    print("\n=== HEADLINE (weekly, All India) ===", flush=True)
    d = det_df[(det_df.scale == 'weekly') & (det_df.region == 'All India')]
    p = prob_df[(prob_df.scale == 'weekly') & (prob_df.region == 'All India')] if len(prob_df) else prob_df
    for var in sorted(d.variable.unique()):
        print(f"\n{var}  PCC by week:")
        for m in DET_MODELS:
            sub = d[(d.variable == var) & (d.model == m)]
            if not len(sub):
                continue
            cells = "".join(f"  {sub[sub.lead == wk]['pcc'].mean():5.2f}" for wk in range(1, 7))
            print(f"  {m:11s}{cells}")
        if len(p):
            print(f"{var}  CRPSS-vs-clim by week:")
            for m in BASE_SYSTEMS:
                sub = p[(p.variable == var) & (p.model == m)]
                if not len(sub):
                    continue
                cells = "".join(f"  {sub[sub.lead == wk]['crpss_clim'].mean():5.2f}" for wk in range(1, 7))
                print(f"  {m:11s}{cells}")


if __name__ == '__main__':
    main()
