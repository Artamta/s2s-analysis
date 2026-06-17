#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_s2s.py  —  Unified S2S verification DRIVER, V3 (dual climatology).
================================================================================
V3 scores every model under TWO anomaly baselines and records both in the CSV
via a `clim_basis` column:

  clim_basis='era5'       — all models anomalised against the ERA5 30-yr WMO
                            DOY climatology (same baseline → fair inter-model
                            comparison).  SPIRE, FuXi, ECMWF, MME, Persistence.

  clim_basis='model_own'  — FuXi and ECMWF anomalised against their own lead-
                            dependent hindcast climatology (removes their own
                            systematic bias from the PCC / MSSS signal).
                            SPIRE excluded (no model-own clima file).
                            MME here = mean of FuXi + ECMWF model-own anomalies.

Probabilistic scores (CRPS/Brier) are anomaly-baseline-neutral (raw forecast
distribution vs raw obs), so they appear once under clim_basis='era5'.

SPIRE: mean_stddev group (absolute ensemble mean + spread).  The anomaly group
is NOT used — its ERA5 clim reference differs from ours by ~17 gpm / 0.2 mm/day,
which would inflate RMSE/MSSS.

OUTPUTS (this folder)
  skill_deterministic.csv   clim_basis added; ~2× rows vs V2
  skill_probabilistic.csv   clim_basis='era5' for all rows
  skill_brier.csv           clim_basis='era5' for all rows
  reliability.npz
  RUN_METADATA.txt

RUN
  python verify_s2s.py --test              # 1 init smoke test
  python verify_s2s.py --workers 13        # full run (use SLURM)
  sbatch run_slurm.sh
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
from loaders import (CFG, WEEKS, REGIONS, BASE_SYSTEMS, DET_MODELS, VARS,
                     MODEL_OWN_CLIM)

NB = 10  # reliability bins


# ==============================================================================
# Brier event definitions  (unchanged from V2)
# ==============================================================================
def brier_events(var, mu, sig, o, cm, cs):
    out = []
    muv, sigv, ov, cmv = mu.values, sig.values, o.values, cm.values
    csv = np.asarray(cs)
    if var == 'TP':
        for thr in CFG.tp_thresholds:
            p    = M.prob_exceed(thr, muv, sigv)
            y    = (ov > thr).astype(float)
            base = M.prob_exceed(thr, cmv, csv)
            out.append((f'tp_gt_{int(thr)}mm', p, y, base))
    if CFG.use_terciles:
        lo = cmv - 0.4307 * csv
        hi = cmv + 0.4307 * csv
        p_above = M.prob_exceed(0.0, muv - hi, sigv)
        p_below = M.prob_below(0.0, muv - lo, sigv)
        y_above = (ov > hi).astype(float)
        y_below = (ov < lo).astype(float)
        third   = np.full_like(cmv, 1.0 / 3.0)
        out.append((f'{var.lower()}_above_normal', p_above, y_above, third))
        out.append((f'{var.lower()}_below_normal', p_below, y_below, third))
    return out


# ==============================================================================
# Build per-system fields for ONE (init, scale, lead) slice
# ==============================================================================
def assemble_fields(var, kind, sl, spire, fuxi, op, GC):
    """kind='weekly' -> sl=(ds,de); kind='daily' -> sl=di (0-based)."""
    det_field, mu_sig = {}, {}

    # ---- SPIRE (mean_stddev, absolute) ----
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
        mu_sig['SPIRE']    = (m, s)

    # ---- FuXi (11 members) ----
    if kind == 'weekly':
        ds, de = sl
        fd  = [fuxi[var][d] for d in range(ds, de + 1) if d in fuxi[var]]
        ens = xr.concat(fd, 't').mean('t') if fd else None
    else:
        ens = fuxi[var].get(sl + 1)
    if ens is not None:
        if var == 'TP':
            ens = ens * CFG.fuxi_tp_factor
        mu, sig = L.ens_mean_std(ens, 'member', GC)
        det_field['FuXi'] = mu
        mu_sig['FuXi']    = (mu, sig)

    # ---- ECMWF ----
    for m in ('ECMWF',):
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
            field = (L.daily_from_cumulative(arr, sl) if var == 'TP'
                     else arr.isel(step=sl))
        mu, sig = L.ens_mean_std(field, 'number', GC)
        det_field[m] = mu
        mu_sig[m]    = (mu, sig)

    return det_field, mu_sig


# ==============================================================================
# CORE WORKER  —  all rows for ONE init date
# ==============================================================================
def process_init(init, want_vars):
    from time import time
    t0  = time()
    pid = os.getpid()
    def log(msg):
        print(f"[{init} pid{pid} +{time()-t0:5.0f}s] {msg}", flush=True)

    GC       = L.build_grid_context()
    w        = M.cos_latitude_weights(GC['lat'], xr)
    clim_ds  = L.open_clim()
    truth    = L.open_truth()
    init_str = pd.to_datetime(init).strftime('%Y%m%d')
    log(f"START vars={want_vars}  grid={GC['lat'].size}x{GC['lon'].size}")

    det_rows, prob_rows, brier_rows = [], [], []
    rel_accum = {}

    log("loading SPIRE (mean_stddev)...")
    spire = L.load_spire(init)
    log(f"  SPIRE: {'ok ' + ','.join(spire) if spire else 'MISSING'}")
    log("loading FuXi members...")
    fuxi  = L.load_fuxi_all(init_str, want_vars, CFG.G)
    log(f"  FuXi: " + ", ".join(f"{v}={len(fuxi.get(v,{}))}d" for v in want_vars))
    log("loading ECMWF...")
    op    = {m: {v: L.load_op(m, init_str, v, CFG.G) for v in want_vars}
             for m in ('ECMWF',)}
    log(f"  ECMWF: " + ", ".join(
        f"{v}={'ok' if op['ECMWF'][v] is not None else '--'}" for v in want_vars))
    pers       = {v: L.persistence_field(v, truth, init, GC) for v in want_vars}
    clim_sigma = {v: L.clim_spread_field(v, truth, GC) for v in want_vars}

    log("loading model-own climatologies (FuXi, ECMWF)...")
    model_clim = {m: {v: L.load_model_clim(m, v, init) for v in want_vars}
                  for m in MODEL_OWN_CLIM}
    for m in MODEL_OWN_CLIM:
        log(f"  {m} clima: " + ", ".join(
            f"{v}={'ok' if model_clim[m][v] is not None else 'MISS'}" for v in want_vars))
    log("data loaded -> scoring weekly + daily ...")

    # ── helpers ──────────────────────────────────────────────────────────────
    def det_row(var, m, f, o, clim_f, clim_o, persf, scale, lead, week, clim_basis):
        for rg in REGIONS:
            rda = L.region_da(rg, GC)
            fr, orr = f.where(rda), o.where(rda)
            cf, co  = clim_f.where(rda), clim_o.where(rda)
            pr = persf.where(rda) if persf is not None else None
            row = dict(
                season=CFG.season_label, variable=var, model=m, region=rg,
                init_date=init, scale=scale, lead=lead, week=week,
                clim_basis=clim_basis,
                pcc       = M.acc(fr, orr, cf, w, clim_o=co),
                rmse      = M.rmse(fr, orr, w),
                bias      = M.bias(fr, orr, w),
                msss_clim = M.msss(fr, orr, co, w),
                msss_pers = (M.msss(fr, orr, pr, w) if pr is not None else np.nan),
                fcst_std  = M.wstd_anom(fr - cf, w),
                obs_std   = M.wstd_anom(orr - co, w),
                fcst_mean = M._f(M.wmean(fr, w)),
                obs_mean  = M._f(M.wmean(orr, w)),
            )
            row['std_ratio'] = M.std_ratio(fr, orr, cf, w, clim_o=co)
            det_rows.append(row)

    def prob_brier_rows(var, m, mu, sig, o, cm_o, cs, persf,
                        scale, lead, week, do_brier):
        sig = sig.clip(min=CFG.sig_floor[var])
        cs  = cs.clip(min=CFG.sig_floor[var])
        for rg in REGIONS:
            rda = L.region_da(rg, GC)
            muR, sgR, oR = mu.where(rda), sig.where(rda), o.where(rda)
            cmR = cm_o.where(rda)
            csR = cs.where(rda)
            prR = persf.where(rda) if persf is not None else None
            crps_g  = muR.copy(data=M.crps_gauss(muR.values, sgR.values, oR.values))
            crps_m  = M._f(M.wmean(crps_g, w))
            crps_c_g = muR.copy(data=M.crps_gauss(cmR.values, csR.values, oR.values))
            crps_cl = M._f(M.wmean(crps_c_g, w))
            if prR is not None:
                crps_p_g = muR.copy(data=M.crps_gauss(
                    prR.values, np.full_like(prR.values, CFG.sig_floor[var]), oR.values))
                crps_p = M._f(M.wmean(crps_p_g, w))
            else:
                crps_p = np.nan
            rmse_m   = M.rmse(muR, oR, w)
            spread_m = M._f(M.wmean(sgR, w))
            prob_rows.append(dict(
                season=CFG.season_label, variable=var, model=m, region=rg,
                init_date=init, scale=scale, lead=lead, week=week,
                clim_basis='era5',
                crps=crps_m, crps_clim=crps_cl, crps_pers=crps_p,
                crpss_clim=M.crpss(crps_m, crps_cl),
                crpss_pers=M.crpss(crps_m, crps_p),
                spread=spread_m, rmse=rmse_m, ssr=M.ssr(spread_m, rmse_m)))
            if do_brier:
                for ev, pf, yy, base in brier_events(var, muR, sgR, oR, cmR, csR.values):
                    bs   = M.brier_score(pf, yy, w, muR)
                    bs_c = M.brier_clim(base, yy, w, muR)
                    brier_rows.append(dict(
                        season=CFG.season_label, variable=var, model=m, region=rg,
                        init_date=init, scale=scale, lead=lead, week=week,
                        clim_basis='era5', event=ev,
                        brier=bs, brier_clim=bs_c, briss_clim=M.briss(bs, bs_c),
                        base_rate=M._f(M.wmean(muR.copy(data=yy), w))))
                    if rg == 'All India':
                        key = (ev, m)
                        rel_accum.setdefault(key, np.zeros((3, NB)))
                        M.accumulate_reliability(rel_accum[key], pf, yy, NB)

    def _score_det_pass(var, det_field, o, clim_o, cf_field, persf,
                        scale, lead, wn, clim_basis):
        """One deterministic pass for a given clim_basis.

        clim_basis='era5'      : every model uses ERA5 clim as forecast baseline.
        clim_basis='model_own' : FuXi/ECMWF use their own clim; SPIRE skipped.
        MME is always included when ≥2 component anomalies are available.
        Persistence is only added in the 'era5' pass (it IS an observation).
        """
        anoms = []
        for m in BASE_SYSTEMS:
            if m not in det_field:
                continue
            if clim_basis == 'era5':
                cf = clim_o
            else:  # model_own
                if m not in MODEL_OWN_CLIM:
                    continue          # SPIRE has no model-own clim
                cf = cf_field.get(m)
                if cf is None:
                    continue          # model-own clim missing for this lead
            det_row(var, m, det_field[m], o, cf, clim_o, persf,
                    scale, lead, wn, clim_basis)
            a = det_field[m] - cf
            a = a.drop_vars([c for c in a.coords if c not in ('lat', 'lon')])
            anoms.append(a)

        # MME: mean anomaly + ERA5 clim, scored with clim_o as forecast baseline
        if len(anoms) >= 2:
            mme_anom = xr.concat(anoms, 'mm').mean('mm')
            det_row(var, 'MME', mme_anom + clim_o, o, clim_o, clim_o, persf,
                    scale, lead, wn, clim_basis)

        # Persistence (era5 pass only, once)
        if (clim_basis == 'era5' and persf is not None
                and not bool(np.isnan(persf).all())):
            det_row(var, 'Persistence', persf, o, clim_o, clim_o, persf,
                    scale, lead, wn, clim_basis)

    # ============ WEEKLY ============
    for wi, (wn, ds, de) in enumerate(WEEKS):
        wk    = wi + 1
        valid = L.valid_dates_for(init, ds, de, CFG.valid_end)
        valid_c = L.valid_dates_for(init, ds, de, CFG.valid_end_clim)
        if not valid or not valid_c:
            continue
        doys = [pd.to_datetime(d).dayofyear for d in valid]
        for var in want_vars:
            o      = L.truth_period_mean(var, truth, valid, GC)
            if o is None or bool(np.isnan(o).all()):
                continue
            clim_o = L.clim_field(clim_ds, var, doys, GC)
            persf  = pers[var]
            det_field, mu_sig = assemble_fields(
                var, 'weekly', (ds, de), spire, fuxi, op, GC)
            cf_field = {m: L.model_clim_aggregate(
                            m, var, model_clim[m][var], 'weekly', (ds, de), GC)
                        for m in MODEL_OWN_CLIM}
            # deterministic: two passes
            _score_det_pass(var, det_field, o, clim_o, cf_field, persf,
                            'weekly', wk, wn, 'era5')
            _score_det_pass(var, det_field, o, clim_o, cf_field, persf,
                            'weekly', wk, wn, 'model_own')
            # probabilistic: once (baseline-neutral)
            for m in BASE_SYSTEMS:
                if m in mu_sig:
                    mu, sig = mu_sig[m]
                    prob_brier_rows(var, m, mu, sig, o, clim_o,
                                    clim_sigma[var], persf,
                                    'weekly', wk, wn, do_brier=True)
        log(f"  weekly {wn} done ({len(det_rows)} det rows so far)")
    log(f"WEEKLY complete -> {len(det_rows)} det / {len(prob_rows)} prob / {len(brier_rows)} brier")

    # ============ DAILY ============
    for di in range(42):
        day  = di + 1
        date = (pd.to_datetime(init) + pd.Timedelta(days=di + 1)).strftime('%Y-%m-%d')
        if date > CFG.valid_end:
            break
        wn  = f'Week {min(di // 7 + 1, 6)}'
        doy = [pd.to_datetime(date).dayofyear]
        for var in want_vars:
            o = L.truth_day(var, truth, date, GC)
            if o is None:
                continue
            clim_o    = L.clim_field(clim_ds, var, doy, GC)
            persf     = pers[var]
            det_field, mu_sig = assemble_fields(
                var, 'daily', di, spire, fuxi, op, GC)
            cf_field = {m: L.model_clim_aggregate(
                            m, var, model_clim[m][var], 'daily', di, GC)
                        for m in MODEL_OWN_CLIM}
            _score_det_pass(var, det_field, o, clim_o, cf_field, persf,
                            'daily', day, wn, 'era5')
            _score_det_pass(var, det_field, o, clim_o, cf_field, persf,
                            'daily', day, wn, 'model_own')
            for m in BASE_SYSTEMS:
                if m in mu_sig:
                    mu, sig = mu_sig[m]
                    prob_brier_rows(var, m, mu, sig, o, clim_o,
                                    clim_sigma[var], persf,
                                    'daily', day, wn, do_brier=False)
        if day % 7 == 0:
            log(f"  daily lead {day}/42 done")
    log(f"DONE init {init}: {len(det_rows)} det / {len(prob_rows)} prob / {len(brier_rows)} brier rows")

    rel_out = [(ev, m, arr) for (ev, m), arr in rel_accum.items()]
    return det_rows, prob_rows, brier_rows, rel_out


# ==============================================================================
# DRIVER
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test',    action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--vars',    nargs='+', default=VARS, choices=VARS)
    args = ap.parse_args()

    inits      = list(CFG.init_dates[:1] if args.test else CFG.init_dates)
    want_vars  = list(args.vars)
    n_workers  = max(1, min(args.workers, len(inits)))
    from time import time
    t_start = time()
    print(f"\nUnified S2S verification V3 (dual clim basis) — season={CFG.season_label}",
          flush=True)
    print(f"  models={BASE_SYSTEMS} (+MME,Persistence)  vars={want_vars}", flush=True)
    print(f"  clim_basis: era5 (all models) + model_own (FuXi/ECMWF)", flush=True)
    print(f"  inits={len(inits)}  workers={n_workers}  out_dir={HERE}\n", flush=True)

    det_all, prob_all, brier_all = [], [], []
    rel_merge = {}
    n_total   = len(inits)

    if n_workers <= 1:
        results = []
        for i, init in enumerate(inits, 1):
            results.append(process_init(init, want_vars))
            el  = time() - t_start
            eta = el / i * (n_total - i)
            print(f">>> PROGRESS {i}/{n_total}  elapsed={el/60:.1f}m  ETA={eta/60:.1f}m",
                  flush=True)
    else:
        results = []
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_init, init, want_vars): init for init in inits}
            for fut in as_completed(futs):
                init = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"  !! init {init} FAILED: {e}", flush=True)
                done += 1
                el  = time() - t_start
                eta = el / done * (n_total - done)
                print(f">>> PROGRESS {done}/{n_total}  elapsed={el/60:.1f}m  ETA={eta/60:.1f}m",
                      flush=True)

    for det_rows, prob_rows, brier_rows, rel_out in results:
        det_all.extend(det_rows)
        prob_all.extend(prob_rows)
        brier_all.extend(brier_rows)
        for ev, m, arr in rel_out:
            rel_merge.setdefault((ev, m), np.zeros((3, NB)))
            rel_merge[(ev, m)] += arr

    _write_csv(det_all,
               ['variable', 'model', 'clim_basis', 'region', 'scale', 'lead', 'init_date'],
               'skill_deterministic.csv')
    _write_csv(prob_all,
               ['variable', 'model', 'region', 'scale', 'lead', 'init_date'],
               'skill_probabilistic.csv')
    _write_csv(brier_all,
               ['variable', 'event', 'model', 'region', 'scale', 'lead', 'init_date'],
               'skill_brier.csv')

    rel_npz = {}
    for (ev, m), arr in rel_merge.items():
        cnt = arr[1]; ok = cnt > 0
        rel_npz[f'{ev}__{m}__obs_freq'] = np.where(ok, arr[0] / np.maximum(cnt,1), np.nan)
        rel_npz[f'{ev}__{m}__fcst_p']   = np.where(ok, arr[2] / np.maximum(cnt,1), np.nan)
        rel_npz[f'{ev}__{m}__count']    = cnt
    np.savez_compressed(os.path.join(HERE, 'reliability.npz'), nbins=NB, **rel_npz)
    print(f"WROTE reliability.npz ({len(rel_merge)} event-model curves)", flush=True)

    det_df   = pd.DataFrame(det_all)
    prob_df  = pd.DataFrame(prob_all)
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
        "S2S verification run metadata — V3 (dual clim basis)", "=" * 60,
        f"timestamp        : {datetime.now().isoformat(timespec='seconds')}",
        f"season_label     : {CFG.season_label}",
        f"n_inits          : {len(inits)}",
        f"init_dates       : {', '.join(inits)}",
        f"variables        : {', '.join(want_vars)}",
        f"regions          : {', '.join(REGIONS)}",
        f"grid             : {CFG.dgrid} deg, lat {CFG.lat0}->{CFG.lat1}, lon {CFG.lon0}->{CFG.lon1}",
        f"models           : {', '.join(BASE_SYSTEMS)} (+ MME, Persistence)",
        f"clim_basis=era5      : all models vs ERA5 30-yr WMO DOY clim (fair comparison)",
        f"clim_basis=model_own : FuXi, ECMWF vs their own lead-dependent hindcast clim",
        f"SPIRE            : mean_stddev group only (no model-own clim file)",
        f"ERA5 climatology : {CFG.clim_path}",
        f"model clima root : {CFG.model_clim_root}",
        f"FuXi tp factor   : x{CFG.fuxi_tp_factor}",
        f"sigma floors     : {CFG.sig_floor}",
        f"tp thresholds    : {CFG.tp_thresholds}; terciles={CFG.use_terciles}",
        "",
        "Probabilistic scores (CRPS/Brier/SSR) are anomaly-baseline-neutral;",
        "they appear once, tagged clim_basis='era5'.",
        "",
        f"rows: deterministic={len(det_df)}  probabilistic={len(prob_df)}  brier={len(brier_df)}",
    ]
    with open(os.path.join(HERE, 'RUN_METADATA.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print("WROTE RUN_METADATA.txt", flush=True)


def _headline(det_df, prob_df):
    if not len(det_df):
        return
    d = det_df[(det_df.scale == 'weekly') & (det_df.region == 'All India')]
    p = prob_df[(prob_df.scale == 'weekly') & (prob_df.region == 'All India')] if len(prob_df) else prob_df
    for basis in ['era5', 'model_own']:
        db = d[d.clim_basis == basis]
        if not len(db):
            continue
        print(f"\n=== HEADLINE (weekly, All India, clim_basis={basis}) ===", flush=True)
        for var in sorted(db.variable.unique()):
            print(f"\n{var}  PCC by week:")
            for m in DET_MODELS:
                sub = db[(db.variable == var) & (db.model == m)]
                if not len(sub):
                    continue
                cells = "".join(f"  {sub[sub.lead==wk]['pcc'].mean():5.2f}"
                                for wk in range(1, 7))
                print(f"  {m:11s}{cells}")
            if basis == 'era5' and len(p):
                print(f"{var}  CRPSS-vs-clim by week:")
                for m in BASE_SYSTEMS:
                    sub = p[(p.variable == var) & (p.model == m)]
                    if not len(sub):
                        continue
                    cells = "".join(f"  {sub[sub.lead==wk]['crpss_clim'].mean():5.2f}"
                                    for wk in range(1, 7))
                    print(f"  {m:11s}{cells}")


if __name__ == '__main__':
    main()
