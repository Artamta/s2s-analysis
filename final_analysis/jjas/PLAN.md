# JJAS Monsoon S2S Verification — Plan & Handoff

> Context doc for any AI model / collaborator picking up the JJAS work. Pairs
> with the season-agnostic engine in `final_analysis/core/` and the completed
> `final_analysis/jfm2026/` study. Read `final_analysis/README.md` first for the
> architecture (core + per-season config/adapters; every model = one adapter
> returning a `ForecastCube`).

Date started: 2026-06-25.

---

## 1. The paper (two complementary parts)

1. **JFM2026 (DONE)** — operational snapshot: **SPIRE vs FuXi vs ECMWF**, TP+Z500
   (+T2M SPIRE/FuXi), 13 weekly inits, 1 season. The novelty is SPIRE competing
   operationally. Code: `final_analysis/jfm2026/`.
2. **JJAS multi-year (THIS) — the scientific core**: **FuXi vs ECMWF**, TP+Z500,
   **2002–2019** (18-yr reforecast overlap), Indian summer monsoon. Statistically
   robust → publishable. No SPIRE (no hindcast), no T2M (ECMWF reforecast lacks it).

Why JJAS is the stronger result: 18 years over the monsoon vs 1 season/1 year.
Enables interannual analysis (active/break, ENSO years), monsoon onset, and the
proper **dual anomaly basis** (era5 + model_own) that JFM2026 couldn't do.

---

## 2. Data paths (authoritative)

| Thing | Path | Structure |
|---|---|---|
| **ECMWF reforecast** | `/storage/raj.ayush/archive/All_Model_Data/models/ecmwf/data/` | `<stem>_<cf\|pf>_<MMDD>.grib`, stem∈{`tp`,`z500`}; dims `(number, time=20yr 2000-2019, step=46, 34×34 @1.5°)`. tp=`tp` kg/m² CUMULATIVE; z500=`gh` gpm. pf=10 members + cf control → 11. `mean('time')` = model-own clim. |
| **FuXi hindcast** | `/storage/raj.ayush/archive/All_Model_Data/models/fuxi/data/<YYYYMMDD>.7z` | ~2080 archives, 2002-2021 Mon/Thu. Each → `<yr>/<YYYYMMDD>/member/<1..50>/<lead>.nc`, 50 members, 42 leads, ~76 channels (incl z500/t2m/tp). z500=geopotential (÷G); tp=mm/h (×24). **Needs extraction.** |
| **ERA5 truth (WB2)** | `/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr` | global 1.5°, 6-hourly, 1959-2023. Use `total_precipitation_6hr`→daily SUM×1000 (mm/day); `geopotential@500`÷G→daily mean (gpm). Native-res variant available: `...1440x721_with_derived_variables.zarr` (0.25°). |
| **ERA5 DOY climatology** | `/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc` | dayofyear×721×1440, tp/z500/t2m. Covers ALL days incl JJAS — reused as-is. |
| **IMD region masks** | `final_analysis/masks/imd_region_masks_<res>deg.nc` (+ existing 1.5° at `s2s-forecast-data-prev/era5/daily/imd_region_masks.nc`) | 4 IMD regions; rebuild any res via `python -m core.regions --dgrid <r> --out ...`. |
| **SOI shapefile** | `/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp` | for region masks. |

Overlap years for FuXi×ECMWF: **2002–2019** (ECMWF 2000-2019 ∩ FuXi 2002-2021).

---

## 3. What's built (`final_analysis/jjas/`)

- `config.py` — `build_config(year, dgrid)`. JJAS inits = ECMWF reforecast MMDD
  files in Jun–Aug as `<year>-MM-DD` (27 for 2019). Truth=WB2, clim=reused ERA5
  DOY, models=[ECMWF] (FuXi added next). Output → `results_<year>_<res>deg/`.
- `adapters_jjas.py` — `ecmwf_reforecast` (forecast for the init's year, 11
  members) and `ecmwf_reforecast_clim` (mean over 20 yr × members = lead-dependent
  model-own clim, a mean-only ForecastCube). tp accum='cumulative', z500 'instant'.
- `run_verify.py` — driver with the **dual basis** (`era5` + `model_own`). Per init,
  parallel; weekly W1–W6 + daily 1–42; All-India + 4 IMD regions; det + prob + Brier.
- `core/truth.py::open_truth_wb2` — WB2 → daily tp/z500 truth (same dict interface).
- `core/config.py` — `ModelSpec.clim_adapter` (registry key for model-own clim).

**FuXi (in progress, separate agent):** `jjas/preprocess_fuxi.py` (extract .7z →
select tp/z500 → compact `fuxi_combined/<init>.nc` → delete raw, incrementally)
and `jjas/adapters_fuxi.py` (`fuxi_reforecast` adapter). FuXi model-own clim =
mean of the per-year compact files for the same MMDD (follow-up).

---

## 4. How to run

```bash
conda activate s2s-hind
cd final_analysis/jjas
python run_verify.py --test --workers 1            # 2-init smoke (year from config)
python run_verify.py --year 2019 --workers 13      # one full JJAS year
# plots reuse the JFM suite:
cd plots 2>/dev/null || true
python ../../jfm2026/plots/make_plots.py --results ../results_2019_1.5deg
```

Outputs per run (in `results_<year>_<res>deg/`): `skill_deterministic.csv`
(has `clim_basis` ∈ {era5, model_own} and `grid_res`), `skill_probabilistic.csv`,
`skill_brier.csv`, `reliability.npz`.

---

## 5. Roadmap

- [x] ECMWF reforecast adapter + model-own clim + WB2 truth + dual-basis driver.
- [~] Validate single year (2019) ECMWF-vs-ERA5 end-to-end (smoke passing).
- [ ] FuXi: extract+combine JJAS inits, `fuxi_reforecast` adapter (agent).
- [ ] FuXi model-own climatology (multi-year mean per MMDD).
- [ ] Add FuXi to `config.models`; run FuXi-vs-ECMWF for 2019.
- [ ] **Scale to 2002–2019** (init_dates span all years; more workers; FuXi
      extraction is the cost — do incrementally).
- [ ] Monsoon analyses: onset skill, active/break spells, ISMR sub-seasonal,
      interannual (ENSO/IOD years), regional IMD breakdown, native 0.25° vs 1.5°.
- [ ] (optional) re-download ECMWF `2t`/`mn2t6`/`mx2t6` reforecasts with full
      step axis to add a T2M track.

## 6. Key facts / gotchas

- ECMWF reforecast files stack 20 years in ONE grib → select `time.dt.year==Y`;
  model-own clim is FREE (`mean('time')`). z500 var is `gh` (already gpm).
- WB2 truth daily build is the slow step on first call (~a few s once warm);
  each worker loads its own window. tp uses `total_precipitation_6hr` daily-summed.
- Reuse: `core/` is season-agnostic; `region_mask_path`, the plotting suite, and
  the `ForecastCube`/registry all carry over from JFM2026 unchanged.
- The dual-basis MME needs ≥2 models, so MME appears only once FuXi is added.
