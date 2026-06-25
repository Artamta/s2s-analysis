#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jjas/plots/_run_subset.py — FAST presentation-subset JJAS-2019 verification.
============================================================================
Runs run_verify.process_init for a hand-picked subset of ~8 inits (spread
across Jun/Jul/Aug for clean lead curves) PLUS 2019-06-20 (the one FuXi init
with an extracted compact file), then writes the standard result CSVs into
the SAME out_dir the full driver uses. No core/ or run_verify.py edits.

  python plots/_run_subset.py                 # default 8+1 inits, 8 workers
  python plots/_run_subset.py --workers 4
"""
import os
import sys
import argparse
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
JJAS = os.path.dirname(HERE)
sys.path.append(JJAS)                                  # jjas/ on path
sys.path.append(os.path.dirname(JJAS))                 # final_analysis/ on path

import config as jcfg
import run_verify as RV

# Subset spanning the monsoon: early/mid/late Jun, Jul, Aug + the FuXi init.
SUBSET = [
    "2019-06-04", "2019-06-18", "2019-06-20",   # 06-20 = FuXi compact file
    "2019-07-02", "2019-07-16", "2019-07-30",
    "2019-08-13", "2019-08-20", "2019-08-31",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--dgrid", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--vars", nargs="+", default=["TP", "Z500"])
    args = ap.parse_args()

    cfg = jcfg.build_config(year=args.year, dgrid=args.dgrid)
    # Override the global the worker function reads, and restrict inits to subset.
    inits = [d for d in SUBSET if d.startswith(str(args.year))]
    # Keep only inits that exist in the full config window (calendar-valid).
    inits = [d for d in inits if d in set(cfg.init_dates) or d == "2019-06-20"]
    # Build a config whose init_dates is the subset so truth windows are correct.
    cfg.init_dates = tuple(sorted(inits))
    RV.CFG = cfg
    os.makedirs(cfg.out_dir, exist_ok=True)

    want_vars = args.vars
    n_workers = max(1, min(args.workers, len(inits)))
    t0 = time()
    print(f"\nJJAS SUBSET verify — {cfg.season_label} grid={cfg.grid.dgrid}deg", flush=True)
    print(f"  inits({len(inits)})={list(cfg.init_dates)}", flush=True)
    print(f"  models={cfg.model_names} vars={want_vars} workers={n_workers}", flush=True)
    print(f"  out={cfg.out_dir}\n", flush=True)

    det_all, prob_all, brier_all, rel = [], [], [], {}
    NB = RV.NB

    def collect(r):
        d, p, b, rl = r
        det_all.extend(d); prob_all.extend(p); brier_all.extend(b)
        for ev, m, a in rl:
            rel.setdefault((ev, m), np.zeros((3, NB))); rel[(ev, m)] += a

    if n_workers <= 1:
        for i, init in enumerate(inits, 1):
            collect(RV.process_init(init, want_vars))
            print(f">>> {i}/{len(inits)} {(time()-t0)/60:.1f}m", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker,
                                 initargs=(cfg,)) as ex:
            futs = {ex.submit(RV.process_init, i, want_vars): i for i in inits}
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    collect(fut.result())
                except Exception as e:
                    import traceback
                    print(f"  !! init {futs[fut]} FAILED: {e}", flush=True)
                    traceback.print_exc()
                print(f">>> {k}/{len(inits)} {(time()-t0)/60:.1f}m", flush=True)

    RV._write(det_all, ["variable", "model", "clim_basis", "region", "scale", "lead", "init_date"],
              "skill_deterministic.csv")
    RV._write(prob_all, ["variable", "model", "region", "scale", "lead", "init_date"],
              "skill_probabilistic.csv")
    RV._write(brier_all, ["variable", "event", "model", "region", "scale", "lead", "init_date"],
              "skill_brier.csv")
    rel_npz = {}
    for (ev, m), a in rel.items():
        cnt = a[1]; ok = cnt > 0
        rel_npz[f"{ev}__{m}__obs_freq"] = np.where(ok, a[0] / np.maximum(cnt, 1), np.nan)
        rel_npz[f"{ev}__{m}__fcst_p"] = np.where(ok, a[2] / np.maximum(cnt, 1), np.nan)
    np.savez_compressed(os.path.join(cfg.out_dir, "reliability.npz"), nbins=NB, **rel_npz)
    RV._headline(pd.DataFrame(det_all))
    print(f"\nVERIFY_DONE ({(time()-t0)/60:.1f}m)", flush=True)


def _init_worker(cfg):
    """Make each worker process see the subset config as RV.CFG."""
    RV.CFG = cfg


if __name__ == "__main__":
    main()
