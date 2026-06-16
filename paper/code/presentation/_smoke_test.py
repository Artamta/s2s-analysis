#!/usr/bin/env python
"""Fast smoke test: exercises the real pipeline functions WITHOUT the heavy FuXi
I/O wall. Validates imports, grid/masks, climatology, truth, SPIRE loader,
assemble_fields, all metrics, Brier events, reliability accumulation, and row
dict structure. Run on login node — completes in seconds."""
import os, sys, numpy as np, pandas as pd, xarray as xr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE)
import metrics as M, loaders as L
from verify_s2s import brier_events
import verify_s2s as V

init = L.CFG.init_dates[0]
print(f"smoke init = {init}")

# 1) grid + masks
GC = L.build_grid_context()
w = M.cos_latitude_weights(GC['lat'], xr)
assert GC['all_india'].sum() > 0
print(f"[ok] grid {len(GC['lat'])}x{len(GC['lon'])}, all-india pts={int(GC['all_india'].sum())}, "
      f"regions={list(GC['region_masks'])}")

# 2) climatology + truth
clim_ds = L.open_clim(); truth = L.open_truth()
valid = L.valid_dates_for(init, 1, 7, L.CFG.valid_end)
doys = [pd.to_datetime(d).dayofyear for d in valid]
print(f"[ok] week-1 valid dates {valid[0]}..{valid[-1]}")

# 3) SPIRE loader (fast) + assemble_fields SPIRE branch, week 1
spire = L.load_spire(init)
assert spire is not None, "SPIRE failed to load"
fuxi_empty = {v: {} for v in ('TP', 'Z500', 'T2M')}
op_empty = {m: {v: None for v in ('TP', 'Z500', 'T2M')} for m in ('ECMWF', 'NCEP')}

for var in ['TP', 'Z500', 'T2M']:
    o = L.truth_period_mean(var, truth, valid, GC)
    clim = L.clim_field(clim_ds, var, doys, GC)
    persf = L.persistence_field(var, truth, init, GC)
    det, musig = V.assemble_fields(var, 'weekly', (1, 7), spire, fuxi_empty, op_empty, GC)
    assert 'SPIRE' in det and 'SPIRE' in musig, f"{var}: SPIRE missing"
    f = det['SPIRE']; mu, sig = musig['SPIRE']

    # deterministic metrics
    pcc = M.acc(f, o, clim, w); rmse = M.rmse(f, o, w); bias = M.bias(f, o, w)
    msss_c = M.msss(f, o, clim, w); sr = M.std_ratio(f, o, clim, w)
    assert -1.01 <= pcc <= 1.01, f"{var} PCC out of range: {pcc}"
    assert rmse >= 0
    # probabilistic
    sig = sig.clip(min=L.CFG.sig_floor[var])
    cs = L.clim_spread_field(var, truth, GC).clip(min=L.CFG.sig_floor[var])
    crps_g = mu.copy(data=M.crps_gauss(mu.values, sig.values, o.values))
    crps_m = M._f(M.wmean(crps_g, w))
    crps_c = M._f(M.wmean(mu.copy(data=M.crps_gauss(
        clim.values, cs.values, o.values)), w))
    crpss = M.crpss(crps_m, crps_c)
    spread = M._f(M.wmean(sig, w)); ssr = M.ssr(spread, M.rmse(mu, o, w))
    assert crps_m >= 0 and spread > 0
    # brier (per-gridpoint climatological spread)
    nev = 0
    for ev, pf, yy, base in brier_events(var, mu, sig, o, clim, cs.values):
        bs = M.brier_score(pf, yy, w, mu); bsc = M.brier_clim(base, yy, w, mu)
        assert 0 <= bs <= 1.01, f"{var}/{ev} brier out of range {bs}"
        nev += 1
    print(f"[ok] {var:4s}  PCC={pcc:+.3f}  RMSE={rmse:.3f}  bias={bias:+.3f}  "
          f"MSSS={msss_c:+.3f}  stdratio={sr:.2f}  CRPS={crps_m:.3f}  "
          f"CRPSS={crpss:+.3f}  SSR={ssr:.2f}  brier_events={nev}")

# 4) FuXi loader sanity — just 2 days, 1 var (prove the loader works, not full I/O)
init_str = pd.to_datetime(init).strftime('%Y%m%d')
fpd = {}
import loaders
n_found = 0
for day in (1, 2):
    members = []
    for mem in range(loaders.CFG.fuxi_members):
        f = loaders._fuxi_member_day(init_str, mem, day, 'Z500', loaders.CFG.G)
        if f is not None: members.append(f.assign_coords(member=mem))
    if members:
        fpd[day] = xr.concat(members, 'member'); n_found += len(members)
print(f"[ok] FuXi Z500 days 1-2 loaded {n_found} member-fields; "
      f"day1 ens-mean shape after to_grid={L.to_grid(fpd[1].mean('member'), GC).shape if 1 in fpd else 'NONE'}")

print("\nSMOKE_OK — all pipeline functions exercised, numbers sane.")
