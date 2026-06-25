#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jfm2026/run_verify.py  —  The verification DRIVER (model-agnostic).
================================================================================
For each init date (in parallel) this:
  1. loads every model's ForecastCube via its adapter,
  2. for each week-window (W1..W6) and each daily lead, collapses each cube to a
     (mean, spread) field on the common grid,
  3. scores deterministic + probabilistic + Brier metrics over All-India and the
     4 IMD regions,
  4. writes three tidy CSVs + a reliability .npz + a metadata file.

Because every model speaks the SAME ForecastCube interface, there is ONE scoring
path here — no per-model branching. Adding a model never touches this file.

Anomaly baseline
----------------
  clim_basis='era5'  : every model anomalised against the ERA5 30-yr WMO DOY
                       climatology (fair inter-model comparison). ALWAYS run.
  clim_basis='model_own' : a model with a lead-dependent hindcast clima (loaded
                       as a mean-only ForecastCube) is ALSO scored against it.
                       Auto-skipped when no model-own clima is configured
                       (current JFM2026 case — those files are not on disk).

RUN
  python run_verify.py --test            # 1 init smoke test
  python run_verify.py --workers 13      # full run (use SLURM)
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

import adapters_jfm          # noqa: F401  (registers SPIRE/FuXi/ECMWF adapters)
from config import CFG, build_config

NB = N_RELIABILITY_BINS


# ==============================================================================
# Brier event definitions (TP exceedances + above/below-normal terciles)
# ==============================================================================
def brier_events(var, mu, sig, o, cm, cs, physics):
    out = []
    muv, sigv, ov, cmv = mu.values, sig.values, o.values, cm.values
    csv = np.asarray(cs)
    if var == "TP":
        for thr in physics.tp_thresholds:
            p    = M.prob_exceed(thr, muv, sigv)
            y    = (ov > thr).astype(float)
            base = M.prob_exceed(thr, cmv, csv)
            out.append((f"tp_gt_{int(thr)}mm", p, y, base))
    if physics.use_terciles:
        lo = cmv - 0.4307 * csv          # +-0.4307 sigma = tercile boundaries (Gaussian)
        hi = cmv + 0.4307 * csv
        p_above = M.prob_exceed(0.0, muv - hi, sigv)
        p_below = M.prob_below(0.0, muv - lo, sigv)
        y_above = (ov > hi).astype(float)
        y_below = (ov < lo).astype(float)
        third   = np.full_like(cmv, 1.0 / 3.0)
        out.append((f"{var.lower()}_above_normal", p_above, y_above, third))
        out.append((f"{var.lower()}_below_normal", p_below, y_below, third))
    # Mask every event's (prob, outcome, base-rate) to the SAME support: points
    # where both the forecast and the obs are valid (land-in-region). Without this
    # the tercile climatology reference (a constant 1/3) would be scored over ocean/
    # off-region points that the model term drops as NaN -> biased Brier skill.
    fin = np.isfinite(muv) & np.isfinite(ov)
    out = [(name, np.where(fin, p, np.nan), np.where(fin, y, np.nan),
            np.where(fin, base, np.nan)) for name, p, y, base in out]
    return out


# ==============================================================================
# CORE WORKER — all rows for ONE init date
# ==============================================================================
def process_init(init, want_vars):
    from time import time
    t0, pid = time(), os.getpid()
    def log(m): print(f"[{init} pid{pid} +{time()-t0:5.0f}s] {m}", flush=True)

    phys    = CFG.physics
    GC      = G.build_grid_context(CFG.grid, CFG.paths.region_mask_nc)
    w       = GC["weights"]
    clim_ds = C.open_clim(CFG.paths.clim_nc)
    truth   = T.open_truth(CFG.paths, phys)
    log(f"START vars={want_vars} grid={GC['lat'].size}x{GC['lon'].size}")

    det_rows, prob_rows, brier_rows = [], [], []
    rel_accum = {}

    # ---- load every model's cube: cubes[model][var] = ForecastCube|None -------
    cubes = {}
    for spec in CFG.models:
        adapt = get_adapter(spec.adapter)
        cubes[spec.name] = {v: adapt(init, v, spec, phys) for v in want_vars}
        log(f"  {spec.name}: " + ", ".join(
            f"{v}={'ok' if cubes[spec.name][v] is not None else '--'}" for v in want_vars))

    # ---- persistence + climatological spread (per variable) -------------------
    pers = {v: T.persistence_field(v, truth, init, GC) for v in want_vars}
    clim_sigma = {}
    for v in want_vars:
        series = _full_series_on_grid(v, truth, GC)
        clim_sigma[v] = C.clim_spread_field(v, series, clim_ds, GC, phys)
    log("data loaded -> scoring")

    model_names = CFG.model_names

    # ── deterministic scoring for one (var, scale, lead) slice ────────────────
    def det_row(var, m, f, o, clim_f, clim_o, persf, scale, lead, week, basis, n_models=1):
        for rg in REGIONS:
            rda = G.region_da(rg, GC)
            fr, orr = f.where(rda), o.where(rda)
            cf, co  = clim_f.where(rda), clim_o.where(rda)
            pr = persf.where(rda) if persf is not None else None
            row = dict(season=CFG.season_label, variable=var, model=m, region=rg,
                       init_date=init, scale=scale, lead=lead, week=week,
                       clim_basis=basis, grid_res=CFG.grid.dgrid, n_models=n_models,
                       pcc       = M.acc(fr, orr, cf, w, clim_o=co),
                       rmse      = M.rmse(fr, orr, w),
                       bias      = M.bias(fr, orr, w),
                       msss_clim = M.msss(fr, orr, co, w),
                       msss_pers = (M.msss(fr, orr, pr, w) if pr is not None else np.nan),
                       fcst_std  = M.wstd_anom(fr - cf, w),
                       obs_std   = M.wstd_anom(orr - co, w),
                       fcst_mean = M._f(M.wmean(fr, w)),
                       obs_mean  = M._f(M.wmean(orr, w)),
                       std_ratio = M.std_ratio(fr, orr, cf, w, clim_o=co))
            det_rows.append(row)

    def score_det(var, det_field, o, clim_o, persf, scale, lead, week):
        """era5-basis deterministic pass: each model vs ERA5 clim, + MME + persistence."""
        anoms = []
        for m in model_names:
            if m not in det_field:
                continue
            det_row(var, m, det_field[m], o, clim_o, clim_o, persf, scale, lead, week, "era5")
            a = det_field[m] - clim_o
            a = a.drop_vars([c for c in a.coords if c not in ("lat", "lon")])
            anoms.append(a)
        if len(anoms) >= 2:
            mme = xr.concat(anoms, "mm").mean("mm") + clim_o
            det_row(var, "MME", mme, o, clim_o, clim_o, persf, scale, lead, week, "era5",
                    n_models=len(anoms))   # record how many models the MME averaged
        if persf is not None and not bool(np.isnan(persf).all()):
            det_row(var, "Persistence", persf, o, clim_o, clim_o, persf, scale, lead, week, "era5")

    # ── probabilistic + Brier scoring for one slice ──────────────────────────
    def score_prob(var, m, mu, sig, o, cm_o, cs, persf, scale, lead, week, do_brier):
        sig = sig.clip(min=phys.sig_floor[var])
        cs  = cs.clip(min=phys.sig_floor[var])
        for rg in REGIONS:
            rda = G.region_da(rg, GC)
            muR, sgR, oR = mu.where(rda), sig.where(rda), o.where(rda)
            cmR, csR = cm_o.where(rda), cs.where(rda)
            prR = persf.where(rda) if persf is not None else None
            crps_m = M._f(M.wmean(muR.copy(data=M.crps_gauss(muR.values, sgR.values, oR.values)), w))
            crps_cl = M._f(M.wmean(muR.copy(data=M.crps_gauss(cmR.values, csR.values, oR.values)), w))
            if prR is not None:
                crps_p = M._f(M.wmean(muR.copy(data=M.crps_gauss(
                    prR.values, np.full_like(prR.values, phys.sig_floor[var]), oR.values)), w))
            else:
                crps_p = np.nan
            rmse_m   = M.rmse(muR, oR, w)
            spread_m = M._f(M.wmean(sgR, w))
            prob_rows.append(dict(
                season=CFG.season_label, variable=var, model=m, region=rg,
                init_date=init, scale=scale, lead=lead, week=week, clim_basis="era5",
                grid_res=CFG.grid.dgrid,
                crps=crps_m, crps_clim=crps_cl, crps_pers=crps_p,
                crpss_clim=M.crpss(crps_m, crps_cl), crpss_pers=M.crpss(crps_m, crps_p),
                spread=spread_m, rmse=rmse_m, ssr=M.ssr(spread_m, rmse_m)))
            if do_brier:
                for ev, pf, yy, base in brier_events(var, muR, sgR, oR, cmR, csR.values, phys):
                    bs   = M.brier_score(pf, yy, w, muR)
                    bs_c = M.brier_clim(base, yy, w, muR)
                    brier_rows.append(dict(
                        season=CFG.season_label, variable=var, model=m, region=rg,
                        init_date=init, scale=scale, lead=lead, week=week,
                        clim_basis="era5", grid_res=CFG.grid.dgrid, event=ev,
                        brier=bs, brier_clim=bs_c, briss_clim=M.briss(bs, bs_c),
                        base_rate=M._f(M.wmean(muR.copy(data=yy), w))))
                    if rg == "All India":
                        key = (ev, m)
                        rel_accum.setdefault(key, np.zeros((3, NB)))
                        M.accumulate_reliability(rel_accum[key], pf, yy, NB)

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
            det_field, mu_sig = {}, {}
            for m in model_names:
                cube = cubes[m].get(var)
                if cube is None or not cube.has_week(de):
                    continue
                mu, sig = cube.weekly(ds, de, GC)
                det_field[m] = mu
                if sig is not None:
                    mu_sig[m] = (mu, sig)
            score_det(var, det_field, o, clim_o, persf, "weekly", wk, wn)
            for m, (mu, sig) in mu_sig.items():
                score_prob(var, m, mu, sig, o, clim_o, clim_sigma[var], persf,
                           "weekly", wk, wn, do_brier=True)
        log(f"  weekly {wn} done ({len(det_rows)} det rows)")

    # ============================== DAILY =====================================
    for di in range(42):
        day  = di + 1
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
            det_field, mu_sig = {}, {}
            for m in model_names:
                cube = cubes[m].get(var)
                if cube is None or not cube.has_day(di):
                    continue
                mu, sig = cube.daily(di, GC)
                det_field[m] = mu
                if sig is not None:
                    mu_sig[m] = (mu, sig)
            score_det(var, det_field, o, clim_o, persf, "daily", day, wn)
            for m, (mu, sig) in mu_sig.items():
                score_prob(var, m, mu, sig, o, clim_o, clim_sigma[var], persf,
                           "daily", day, wn, do_brier=False)
        if day % 7 == 0:
            log(f"  daily lead {day}/42 done")

    log(f"DONE {init}: {len(det_rows)} det / {len(prob_rows)} prob / {len(brier_rows)} brier")
    rel_out = [(ev, m, arr) for (ev, m), arr in rel_accum.items()]
    return det_rows, prob_rows, brier_rows, rel_out


def _full_series_on_grid(var, truth, GC):
    """Whole continuous ERA5 truth series on the verification grid (for clim spread)."""
    src = (truth["tp_daily"] if var == "TP"
           else truth["t2m_daily"] if var == "T2M" else truth["z_raw"])
    return G.to_grid(src, GC)


# ==============================================================================
# DRIVER
# ==============================================================================
def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="1 init smoke test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dgrid", type=float, default=None,
                    help="verification resolution in deg (e.g. 1.5 common, 0.5 SPIRE-native)")
    ap.add_argument("--fuxi-root", default=None,
                    help="FuXi root containing combined/<YYYYMMDD>.nc; use jfm2026_ens50 for 50 members")
    ap.add_argument("--fuxi-members", type=int, default=None,
                    help="select the first N FuXi members from each combined file")
    ap.add_argument("--out-suffix", default=None,
                    help="append suffix to results_<dgrid>deg, e.g. ens50")
    ap.add_argument("--vars", nargs="+", default=None)
    args = ap.parse_args()

    # rebuild config at the requested resolution (forked workers inherit this CFG)
    if any(x is not None for x in (args.dgrid, args.fuxi_root, args.fuxi_members, args.out_suffix)):
        CFG = build_config(
            args.dgrid if args.dgrid is not None else CFG.grid.dgrid,
            fuxi_root=args.fuxi_root,
            fuxi_members=args.fuxi_members,
            out_suffix=args.out_suffix,
        )
    os.makedirs(CFG.out_dir, exist_ok=True)

    want_vars = args.vars if args.vars else list(CFG.variables)
    bad = [v for v in want_vars if v not in CFG.variables]
    if bad:
        ap.error(f"unknown vars {bad}; choices: {list(CFG.variables)}")

    inits = list(CFG.init_dates[:1] if args.test else CFG.init_dates)
    n_workers = max(1, min(args.workers, len(inits)))

    from time import time
    t_start = time()
    print(f"\nS2S verification — {CFG.season_label}  (grid={CFG.grid.dgrid}°)", flush=True)
    print(f"  models={CFG.model_names} (+MME,Persistence)  vars={want_vars}", flush=True)
    print(f"  inits={len(inits)} workers={n_workers} out={CFG.out_dir}\n", flush=True)

    det_all, prob_all, brier_all, rel_merge = [], [], [], {}

    def collect(res):
        d, p, b, rel = res
        det_all.extend(d); prob_all.extend(p); brier_all.extend(b)
        for ev, m, arr in rel:
            rel_merge.setdefault((ev, m), np.zeros((3, NB)))
            rel_merge[(ev, m)] += arr

    if n_workers <= 1:
        for i, init in enumerate(inits, 1):
            collect(process_init(init, want_vars))
            print(f">>> {i}/{len(inits)} elapsed={(time()-t_start)/60:.1f}m", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_init, init, want_vars): init for init in inits}
            done = 0
            for fut in as_completed(futs):
                try:
                    collect(fut.result())
                except Exception as e:
                    print(f"  !! init {futs[fut]} FAILED: {e}", flush=True)
                done += 1
                print(f">>> {done}/{len(inits)} elapsed={(time()-t_start)/60:.1f}m", flush=True)

    _write_csv(det_all, ["variable", "model", "clim_basis", "region", "scale", "lead", "init_date"],
               "skill_deterministic.csv")
    _write_csv(prob_all, ["variable", "model", "region", "scale", "lead", "init_date"],
               "skill_probabilistic.csv")
    _write_csv(brier_all, ["variable", "event", "model", "region", "scale", "lead", "init_date"],
               "skill_brier.csv")

    rel_npz = {}
    for (ev, m), arr in rel_merge.items():
        cnt = arr[1]; ok = cnt > 0
        rel_npz[f"{ev}__{m}__obs_freq"] = np.where(ok, arr[0] / np.maximum(cnt, 1), np.nan)
        rel_npz[f"{ev}__{m}__fcst_p"]   = np.where(ok, arr[2] / np.maximum(cnt, 1), np.nan)
        rel_npz[f"{ev}__{m}__count"]    = cnt
    np.savez_compressed(os.path.join(CFG.out_dir, "reliability.npz"), nbins=NB, **rel_npz)
    print(f"WROTE reliability.npz ({len(rel_merge)} curves)", flush=True)

    _write_metadata(det_all, prob_all, brier_all, inits, want_vars)
    _headline(pd.DataFrame(det_all), pd.DataFrame(prob_all))
    print("\nVERIFY_DONE", flush=True)


def _write_csv(rows, sort_cols, name):
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values([c for c in sort_cols if c in df.columns]).reset_index(drop=True)
    df.to_csv(os.path.join(CFG.out_dir, name), index=False)
    print(f"WROTE {name} ({len(df)} rows)", flush=True)


def _write_metadata(det, prob, brier, inits, want_vars):
    lines = [
        "S2S verification run metadata", "=" * 60,
        f"timestamp     : {datetime.now().isoformat(timespec='seconds')}",
        f"season_label  : {CFG.season_label}",
        f"n_inits       : {len(inits)}",
        f"init_dates    : {', '.join(inits)}",
        f"variables     : {', '.join(want_vars)}",
        f"regions       : {', '.join(REGIONS)}",
        f"grid          : {CFG.grid.dgrid} deg, lat {CFG.grid.lat0}->{CFG.grid.lat1}, "
        f"lon {CFG.grid.lon0}->{CFG.grid.lon1}",
        f"models        : {', '.join(CFG.model_names)} (+ MME, Persistence)",
        "model paths    :",
        *[f"  {m.name}: adapter={m.adapter}, kwargs={m.kwargs}" for m in CFG.models],
        f"clim_basis    : era5 (all models vs ERA5 30-yr WMO DOY clim)",
        f"ERA5 clim     : {CFG.paths.clim_nc}",
        f"FuXi tp factor: x{CFG.physics.fuxi_tp_factor}",
        f"sigma floors  : {CFG.physics.sig_floor}",
        f"tp thresholds : {CFG.physics.tp_thresholds}; terciles={CFG.physics.use_terciles}",
        "",
        f"rows: det={len(det)} prob={len(prob)} brier={len(brier)}",
    ]
    with open(os.path.join(CFG.out_dir, "RUN_METADATA.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("WROTE RUN_METADATA.txt", flush=True)


def _headline(det_df, prob_df):
    if not len(det_df):
        return
    d = det_df[(det_df.scale == "weekly") & (det_df.region == "All India")]
    p = prob_df[(prob_df.scale == "weekly") & (prob_df.region == "All India")] if len(prob_df) else prob_df
    print("\n=== HEADLINE (weekly, All India, clim_basis=era5) ===", flush=True)
    for var in sorted(d.variable.unique()):
        print(f"\n{var}  PCC by week:")
        for m in CFG.model_names + ["MME", "Persistence"]:
            sub = d[(d.variable == var) & (d.model == m)]
            if not len(sub):
                continue
            cells = "".join(f"  {sub[sub.lead == wk]['pcc'].mean():5.2f}" for wk in range(1, 7))
            print(f"  {m:11s}{cells}")
        if len(p):
            print(f"{var}  CRPSS-vs-clim by week:")
            for m in CFG.model_names:
                sub = p[(p.variable == var) & (p.model == m)]
                if not len(sub):
                    continue
                cells = "".join(f"  {sub[sub.lead == wk]['crpss_clim'].mean():5.2f}" for wk in range(1, 7))
                print(f"  {m:11s}{cells}")


if __name__ == "__main__":
    main()
