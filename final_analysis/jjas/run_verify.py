#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jjas/run_verify.py  —  JJAS monsoon verification DRIVER (dual climatology basis).
================================================================================
Same engine as jfm2026/run_verify.py but:
  * truth comes from the WeatherBench2 ERA5 zarr (multi-year), via open_truth_wb2;
  * every model that carries a model-own hindcast climatology (clim_adapter) is
    ALSO scored under clim_basis='model_own' (its own lead-dependent clim), in
    addition to clim_basis='era5' (the shared ERA5 DOY clim).

PILOT: one hindcast year at a time (config.build_config(year=...)). FuXi is added
once its .7z archives are extracted; nothing else here changes.

RUN
  python run_verify.py --test               # 2-init smoke test (year from config)
  python run_verify.py --year 2019 --workers 13
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
sys.path.append(os.path.dirname(HERE))                 # final_analysis/ on path

from core import metrics as M
from core import grid as G
from core import climatology as C
from core import truth as T
from core.config import WEEKS, REGIONS, N_RELIABILITY_BINS
from core.adapters import get_adapter
from core.aggregate import valid_dates_for

import adapters_jjas        # noqa: F401  (registers ecmwf_reforecast[_clim])
import adapters_fuxi        # noqa: F401  (registers fuxi_reforecast)
import config as jcfg
from config import CFG

NB = N_RELIABILITY_BINS
BASES = ("era5", "model_own")


# ==============================================================================
def brier_events(var, mu, sig, o, cm, cs, physics):
    out = []
    muv, sigv, ov, cmv = mu.values, sig.values, o.values, cm.values
    csv = np.asarray(cs)
    if var == "TP":
        for thr in physics.tp_thresholds:
            out.append((f"tp_gt_{int(thr)}mm", M.prob_exceed(thr, muv, sigv),
                        (ov > thr).astype(float), M.prob_exceed(thr, cmv, csv)))
    if physics.use_terciles:
        lo, hi = cmv - 0.4307 * csv, cmv + 0.4307 * csv
        third = np.full_like(cmv, 1.0 / 3.0)
        out.append((f"{var.lower()}_above_normal", M.prob_exceed(0.0, muv - hi, sigv),
                    (ov > hi).astype(float), third))
        out.append((f"{var.lower()}_below_normal", M.prob_below(0.0, muv - lo, sigv),
                    (ov < lo).astype(float), third))
    fin = np.isfinite(muv) & np.isfinite(ov)
    return [(n, np.where(fin, p, np.nan), np.where(fin, y, np.nan),
             np.where(fin, b, np.nan)) for n, p, y, b in out]


# ==============================================================================
def process_init(init, want_vars):
    from time import time
    t0, pid = time(), os.getpid()
    def log(m): print(f"[{init} pid{pid} +{time()-t0:5.0f}s] {m}", flush=True)

    phys = CFG.physics
    GC = G.build_grid_context(CFG.grid, CFG.paths.region_mask_nc)
    w = GC["weights"]
    clim_ds = C.open_clim(CFG.paths.clim_nc)
    # WB2 truth window: from a week before the first init to valid_end
    first_init = min(CFG.init_dates)
    start = (pd.to_datetime(first_init) - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    truth = T.open_truth_wb2(CFG.paths.wb2_zarr, phys, start, CFG.valid_end)
    log(f"START vars={want_vars} grid={GC['lat'].size}x{GC['lon'].size}")

    det_rows, prob_rows, brier_rows = [], [], []
    rel_accum = {}

    model_names = CFG.model_names
    own_clim = CFG.model_own_names                 # models with a model-own clima

    cubes, clim_cubes = {}, {}
    for spec in CFG.models:
        adapt = get_adapter(spec.adapter)
        cubes[spec.name] = {v: adapt(init, v, spec, phys) for v in want_vars}
        if spec.clim_adapter:
            cadapt = get_adapter(spec.clim_adapter)
            clim_cubes[spec.name] = {v: cadapt(init, v, spec, phys) for v in want_vars}
        else:
            clim_cubes[spec.name] = {v: None for v in want_vars}
        log(f"  {spec.name}: " + ", ".join(
            f"{v}={'ok' if cubes[spec.name][v] is not None else '--'}" for v in want_vars))

    pers = {v: T.persistence_field(v, truth, init, GC) for v in want_vars}
    clim_sigma = {}
    for v in want_vars:
        series = G.to_grid(T._src(v, truth), GC)
        clim_sigma[v] = C.clim_spread_field(v, series, clim_ds, GC, phys)
    log("data loaded -> scoring")

    # ── deterministic row ────────────────────────────────────────────────────
    def det_row(var, m, f, o, clim_f, clim_o, persf, scale, lead, week, basis, n_models=1):
        for rg in REGIONS:
            rda = G.region_da(rg, GC)
            fr, orr = f.where(rda), o.where(rda)
            cf, co = clim_f.where(rda), clim_o.where(rda)
            pr = persf.where(rda) if persf is not None else None
            det_rows.append(dict(
                season=CFG.season_label, variable=var, model=m, region=rg,
                init_date=init, scale=scale, lead=lead, week=week,
                clim_basis=basis, grid_res=CFG.grid.dgrid, n_models=n_models,
                pcc=M.acc(fr, orr, cf, w, clim_o=co), rmse=M.rmse(fr, orr, w),
                bias=M.bias(fr, orr, w), msss_clim=M.msss(fr, orr, co, w),
                msss_pers=(M.msss(fr, orr, pr, w) if pr is not None else np.nan),
                fcst_std=M.wstd_anom(fr - cf, w), obs_std=M.wstd_anom(orr - co, w),
                fcst_mean=M._f(M.wmean(fr, w)), obs_mean=M._f(M.wmean(orr, w)),
                std_ratio=M.std_ratio(fr, orr, cf, w, clim_o=co)))

    def score_det(var, det_field, model_clim_field, o, clim_o, persf, scale, lead, week):
        """Two passes: era5 (all vs ERA5 clim) and model_own (models with own clim)."""
        for basis in BASES:
            anoms = []
            for m in model_names:
                if m not in det_field:
                    continue
                if basis == "era5":
                    cf = clim_o
                else:
                    if m not in own_clim:
                        continue
                    cf = model_clim_field.get(m)
                    if cf is None:
                        continue
                det_row(var, m, det_field[m], o, cf, clim_o, persf, scale, lead, week, basis)
                a = det_field[m] - cf
                anoms.append(a.drop_vars([c for c in a.coords if c not in ("lat", "lon")]))
            if len(anoms) >= 2:
                mme = xr.concat(anoms, "mm").mean("mm") + clim_o
                det_row(var, "MME", mme, o, clim_o, clim_o, persf, scale, lead, week,
                        basis, n_models=len(anoms))
            if basis == "era5" and persf is not None and not bool(np.isnan(persf).all()):
                det_row(var, "Persistence", persf, o, clim_o, clim_o, persf,
                        scale, lead, week, "era5")

    # ── probabilistic + Brier (era5-neutral) ─────────────────────────────────
    def score_prob(var, m, mu, sig, o, cm_o, cs, persf, scale, lead, week, do_brier):
        sig, cs = sig.clip(min=phys.sig_floor[var]), cs.clip(min=phys.sig_floor[var])
        for rg in REGIONS:
            rda = G.region_da(rg, GC)
            muR, sgR, oR = mu.where(rda), sig.where(rda), o.where(rda)
            cmR, csR = cm_o.where(rda), cs.where(rda)
            prR = persf.where(rda) if persf is not None else None
            crps_m = M._f(M.wmean(muR.copy(data=M.crps_gauss(muR.values, sgR.values, oR.values)), w))
            crps_cl = M._f(M.wmean(muR.copy(data=M.crps_gauss(cmR.values, csR.values, oR.values)), w))
            crps_p = (M._f(M.wmean(muR.copy(data=M.crps_gauss(
                prR.values, np.full_like(prR.values, phys.sig_floor[var]), oR.values)), w))
                if prR is not None else np.nan)
            rmse_m, spread_m = M.rmse(muR, oR, w), M._f(M.wmean(sgR, w))
            prob_rows.append(dict(
                season=CFG.season_label, variable=var, model=m, region=rg,
                init_date=init, scale=scale, lead=lead, week=week, clim_basis="era5",
                grid_res=CFG.grid.dgrid, crps=crps_m, crps_clim=crps_cl, crps_pers=crps_p,
                crpss_clim=M.crpss(crps_m, crps_cl), crpss_pers=M.crpss(crps_m, crps_p),
                spread=spread_m, rmse=rmse_m, ssr=M.ssr(spread_m, rmse_m)))
            if do_brier:
                for ev, pf, yy, base in brier_events(var, muR, sgR, oR, cmR, csR.values, phys):
                    bs, bs_c = M.brier_score(pf, yy, w, muR), M.brier_clim(base, yy, w, muR)
                    brier_rows.append(dict(
                        season=CFG.season_label, variable=var, model=m, region=rg,
                        init_date=init, scale=scale, lead=lead, week=week,
                        clim_basis="era5", grid_res=CFG.grid.dgrid, event=ev,
                        brier=bs, brier_clim=bs_c, briss_clim=M.briss(bs, bs_c),
                        base_rate=M._f(M.wmean(muR.copy(data=np.where(np.isfinite(muR.values), yy, np.nan)), w))))
                    if rg == "All India":
                        rel_accum.setdefault((ev, m), np.zeros((3, NB)))
                        M.accumulate_reliability(rel_accum[(ev, m)], pf, yy, NB)

    def collapse(cube, kind, sl):
        return cube.weekly(*sl, GC) if kind == "weekly" else cube.daily(sl, GC)

    # ============================== WEEKLY ====================================
    for wi, (wn, ds, de) in enumerate(WEEKS):
        wk = wi + 1
        valid = valid_dates_for(init, ds, de, CFG.valid_end)
        if not valid:
            continue
        doys = [pd.to_datetime(d).dayofyear for d in valid]
        for var in want_vars:
            o = T.truth_period_mean(var, truth, valid, GC)
            if o is None or bool(np.isnan(o).all()):
                continue
            clim_o, persf = C.clim_field(clim_ds, var, doys, GC, phys), pers[var]
            det_field, mu_sig, mclim = {}, {}, {}
            for m in model_names:
                cube = cubes[m].get(var)
                if cube is None or not cube.has_week(de):
                    continue
                mu, sig = cube.weekly(ds, de, GC)
                det_field[m] = mu
                if sig is not None:
                    mu_sig[m] = (mu, sig)
                cc = clim_cubes[m].get(var)
                mclim[m] = cc.weekly(ds, de, GC)[0] if (cc is not None and cc.has_week(de)) else None
            score_det(var, det_field, mclim, o, clim_o, persf, "weekly", wk, wn)
            for m, (mu, sig) in mu_sig.items():
                score_prob(var, m, mu, sig, o, clim_o, clim_sigma[var], persf,
                           "weekly", wk, wn, do_brier=True)
        log(f"  weekly {wn} done ({len(det_rows)} det rows)")

    # ============================== DAILY =====================================
    for di in range(42):
        day = di + 1
        date = (pd.to_datetime(init) + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        if date > CFG.valid_end:
            break
        wn = f"Week {min(di // 7 + 1, 6)}"
        doy = [pd.to_datetime(date).dayofyear]
        for var in want_vars:
            o = T.truth_day(var, truth, date, GC)
            if o is None:
                continue
            clim_o, persf = C.clim_field(clim_ds, var, doy, GC, phys), pers[var]
            det_field, mu_sig, mclim = {}, {}, {}
            for m in model_names:
                cube = cubes[m].get(var)
                if cube is None or not cube.has_day(di):
                    continue
                mu, sig = cube.daily(di, GC)
                det_field[m] = mu
                if sig is not None:
                    mu_sig[m] = (mu, sig)
                cc = clim_cubes[m].get(var)
                mclim[m] = cc.daily(di, GC)[0] if (cc is not None and cc.has_day(di)) else None
            score_det(var, det_field, mclim, o, clim_o, persf, "daily", day, wn)
            for m, (mu, sig) in mu_sig.items():
                score_prob(var, m, mu, sig, o, clim_o, clim_sigma[var], persf,
                           "daily", day, wn, do_brier=False)
        if day % 7 == 0:
            log(f"  daily lead {day}/42 done")

    log(f"DONE {init}: {len(det_rows)} det / {len(prob_rows)} prob / {len(brier_rows)} brier")
    return det_rows, prob_rows, brier_rows, [(ev, m, a) for (ev, m), a in rel_accum.items()]


# ==============================================================================
def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="2-init smoke test")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--dgrid", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--vars", nargs="+", default=None)
    args = ap.parse_args()

    if args.year is not None or args.dgrid is not None:
        CFG = jcfg.build_config(year=args.year or CFG.init_dates and int(min(CFG.init_dates)[:4]),
                                dgrid=args.dgrid or CFG.grid.dgrid)
    os.makedirs(CFG.out_dir, exist_ok=True)
    want_vars = args.vars if args.vars else list(CFG.variables)

    inits = list(CFG.init_dates[:2] if args.test else CFG.init_dates)
    n_workers = max(1, min(args.workers, len(inits)))
    from time import time
    t0 = time()
    print(f"\nJJAS verification — {CFG.season_label} (grid={CFG.grid.dgrid}°)", flush=True)
    print(f"  models={CFG.model_names}  vars={want_vars}  bases={BASES}", flush=True)
    print(f"  inits={len(inits)} workers={n_workers} out={CFG.out_dir}\n", flush=True)

    det_all, prob_all, brier_all, rel = [], [], [], {}
    def collect(r):
        d, p, b, rl = r
        det_all.extend(d); prob_all.extend(p); brier_all.extend(b)
        for ev, m, a in rl:
            rel.setdefault((ev, m), np.zeros((3, NB))); rel[(ev, m)] += a

    if n_workers <= 1:
        for i, init in enumerate(inits, 1):
            collect(process_init(init, want_vars))
            print(f">>> {i}/{len(inits)} {(time()-t0)/60:.1f}m", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_init, i, want_vars): i for i in inits}
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    collect(fut.result())
                except Exception as e:
                    print(f"  !! init {futs[fut]} FAILED: {e}", flush=True)
                print(f">>> {k}/{len(inits)} {(time()-t0)/60:.1f}m", flush=True)

    _write(det_all, ["variable", "model", "clim_basis", "region", "scale", "lead", "init_date"],
           "skill_deterministic.csv")
    _write(prob_all, ["variable", "model", "region", "scale", "lead", "init_date"],
           "skill_probabilistic.csv")
    _write(brier_all, ["variable", "event", "model", "region", "scale", "lead", "init_date"],
           "skill_brier.csv")
    rel_npz = {}
    for (ev, m), a in rel.items():
        cnt = a[1]; ok = cnt > 0
        rel_npz[f"{ev}__{m}__obs_freq"] = np.where(ok, a[0] / np.maximum(cnt, 1), np.nan)
        rel_npz[f"{ev}__{m}__fcst_p"] = np.where(ok, a[2] / np.maximum(cnt, 1), np.nan)
    np.savez_compressed(os.path.join(CFG.out_dir, "reliability.npz"), nbins=NB, **rel_npz)
    _headline(pd.DataFrame(det_all))
    print("\nVERIFY_DONE", flush=True)


def _write(rows, sort, name):
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values([c for c in sort if c in df.columns]).reset_index(drop=True)
    df.to_csv(os.path.join(CFG.out_dir, name), index=False)
    print(f"WROTE {name} ({len(df)} rows)", flush=True)


def _headline(det):
    if not len(det):
        return
    d = det[(det.scale == "weekly") & (det.region == "All India")]
    for basis in BASES:
        db = d[d.clim_basis == basis]
        if not len(db):
            continue
        print(f"\n=== {CFG.season_label} weekly All-India PCC (basis={basis}) ===", flush=True)
        for var in sorted(db.variable.unique()):
            print(f"{var}:")
            for m in CFG.model_names + ["MME", "Persistence"]:
                s = db[(db.variable == var) & (db.model == m)]
                if len(s):
                    cells = "".join(f"  {s[s.lead == wk]['pcc'].mean():5.2f}" for wk in range(1, 7))
                    print(f"  {m:11s}{cells}")


if __name__ == "__main__":
    main()
