# STATUS — live progress tracker (continuity / handoff)

**Purpose:** single source of truth so ANY AI model / collaborator can continue
without losing context. Pairs with: `README.md` (architecture), `RUNBOOK.md` (how
to run), `jjas/PLAN.md` (JJAS design + paths), `PAPER_PLAN.md` (the paper).
Env: always `conda activate s2s-hind`. Code in home; **DATA in
`/storage/raj.ayush/s2s_final_data/`**; figures in `final_analysis/**/plots/figs/`.
GitHub: `Artamta/s2s-analysis` (main) — commit/push after each working change.

Legend: ✅ done · 🔄 in progress · ☐ todo · ⚠️ known issue

---

## 0. RIGHT NOW — presentation plots (10-12 figures)
- 🔄 **Agent A** — ERA5 long-term obs analysis → `analysis/era5_monsoon.py` +
  figs in `analysis/figs/era5_monsoon/` (JJAS climatology maps, annual cycle,
  interannual+trend, trend maps, MISO Hovmöller, variability). PARALLELIZED, `--quick`.
- 🔄 **Agent B** — JJAS model eval (ECMWF vs ERA5, +FuXi where available) →
  `jjas/plots/figs/model_eval/` (skill vs lead, dual-basis era5 vs model_own,
  CRPSS/SSR, ECMWF bias map). Runs a SMALL ECMWF JJAS verify first.
- 🔄 confirm-test: `jjas/run_verify.py --year 2019 --test` validating the to_grid fix.
- ☐ collect the best ~12 into one slide folder once both agents land.

---

## 1. JFM2026 study — ✅ COMPLETE
- ✅ Pipeline: SPIRE/FuXi/ECMWF (+MME,+Persistence), TP+Z500+T2M, era5 basis.
- ✅ Common 1.5° + native 0.5° runs; results `jfm2026/results_{1.5,0.5}deg/`.
- ✅ Figures: full suite `jfm2026/plots/figs/{1.5,0.5}deg/`; 4 meeting figs
  `jfm2026/plots/figs/meeting/`; month×region `jfm2026/results_1.5deg/monthwise/`.
- ✅ Validated vs old V3 (PCC tables match). Tercile-Brier masking bug fixed.
- ☐ (optional) migrate JFM results to /storage on next run.

## 2. JJAS monsoon study — 🔄 core built, scaling pending
- ✅ ECMWF reforecast adapter + free model-own clim (`jjas/adapters_jjas.py`).
- ✅ FuXi extract→compact pipeline + adapter (`preprocess_fuxi.py`, `adapters_fuxi.py`).
- ✅ WB2 ERA5 truth (`core/truth.open_truth_wb2`); dual era5+model_own driver.
- ✅ Validated JJAS2019 ECMWF (smoke): TP PCC era5 0.82→own 0.50; Z500 0.98→0.97.
- ✅ to_grid (lon,lat) bug fixed → full JJAS verify now runs.
- 🔄 FuXi extraction: only init `20190620` compact so far (1 of ~26 for 2019).
- ☐ Full JJAS-2019 run (FuXi+ECMWF, all 27 inits).
- ☐ **Scale to 2002-2019** (18 yr) — the robust benchmark. See RUNBOOK Step 2.
- ☐ FuXi model-own climatology (mean over years per MMDD).

## 3. Shared tools — ✅
- ✅ `core/` engine (metrics, grid+masks, climatology, truth, aggregate, adapters,
  config, regions, plotting). Season-agnostic.
- ✅ `analysis/monthwise.py` (month×IMD-region). ✅ `jjas/plots/monsoon_maps.py`
  (per-year India maps). ✅ `jfm2026/plots/{make_plots,meeting_figs}.py`.
- ✅ IMD masks at 1.5° & 0.5° (`masks/`), rebuildable via `python -m core.regions`.

## 4. NOVEL science for the paper — ☐ NOT STARTED (highest value)
- ☐ **MISO/BSISO index** module (project TP/U850 anomalies on BSISO EOFs; amplitude
  + phase error vs lead) → headline figure. *(self-contained, lives in `analysis/`)*
- ☐ **Active/break spell** skill (region rainfall anomaly → active>+1σ/break<−1σ →
  hit-rate / lead-time of spell onset).
- ☐ **Intraseasonal variance ratio** vs lead (AI damping). 
- ☐ needs winds (U850): WB2 has them; FuXi has them; add to truth + FuXi adapter.
> To start: ask "build MISO module" — testable on ECMWF pilot + ERA5, no FuXi needed.

## 5. Paper — ☐
- ✅ Plan written (`PAPER_PLAN.md`): AI-vs-dynamical S2S monsoon; dual-basis
  genuine-vs-climatological skill; MISO/active-break process verdict; SPIRE snapshot.
- ☐ Draft once §2 scaled + §4 built. Target npj Clim Atmos Sci / GRL.

---

## ⚠️ Known issues & fixes (so they don't bite again)
- ⚠️ **`7z` not on SLURM compute nodes** → FuXi extraction must run on the LOGIN
  node (7z + py7zr both available there), NOT inside the sbatch. Run
  `preprocess_fuxi.py --start … --end …` interactively/nohup on login, THEN sbatch
  the verify. (This is why job 56226's FuXi stage failed.)
- ✅ **to_grid dim order** — fixed (was crashing JJAS verify on the numpy/Brier path).
- ⚠️ **ECMWF cfgrib is slow** (~100 s to open a 20-yr×46-step reforecast grib) →
  use parallel workers; `_open_all` is lru_cached per worker. Keep test runs small.
- ⚠️ **Weekly CRPSS uses a daily-scale clim spread** (inherited from V3) → weekly
  probabilistic skill is uniformly optimistic; document or switch to weekly spread.
- ⚠️ **No SPIRE hindcast / no ECMWF reforecast T2M** → JJAS core = FuXi vs ECMWF,
  TP+Z500; SPIRE only in JFM2026 box.

## Key paths
| | path |
|---|---|
| code | `final_analysis/` (home, git) |
| FuXi hindcast .7z | `/storage/raj.ayush/archive/All_Model_Data/models/fuxi/data/<YYYYMMDD>.7z` |
| ECMWF reforecast | `/storage/raj.ayush/archive/All_Model_Data/models/ecmwf/data/<var>_<cf\|pf>_<MMDD>.grib` |
| WB2 ERA5 truth | `/storage/bedartha/public/datasets/.../1959-2023_01_10-6h-240x121_…zarr` |
| ERA5 DOY clim | `/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc` |
| generated data | `/storage/raj.ayush/s2s_final_data/` |
| SOI shapefile | `/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp` |

## Next 3 actions (if continuing)
1. Finish the presentation plots (agents A+B landing) → curate ~12.
2. Extract FuXi JJAS-2019 on the LOGIN node, then full JJAS-2019 verify.
3. Build the **MISO module** (§4) — the paper's novelty.
