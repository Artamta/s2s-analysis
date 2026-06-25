# High-Impact Paper Plan — AI vs Dynamical S2S Prediction of the Indian Monsoon

Working draft, 2026-06-25. Built on the `final_analysis/` verification engine.

---

## 1. The one-line pitch

**Do AI weather models actually forecast the Indian monsoon's subseasonal
active–break cycle — or just reproduce a (biased) climatology?** An 18-year,
multi-model, process-level evaluation over the Indian summer monsoon, with a
method to separate *genuine anomaly skill* from *climatological pattern-matching*.

## 2. Why this is high-impact (the gap)

- **AI S2S is hot but under-validated over the monsoon.** FuXi-S2S (Nat. Commun.
  2024) leads global AI subseasonal forecasting and extends MJO skill 30→36 d, but
  explicitly leaves **BSISO / monsoon active–break as future work**. Independent,
  *regional, process-level* evaluation over the Indian monsoon is largely missing.
- **The stakes are proven.** In 2025 an AI+statistical system delivered subseasonal
  monsoon-onset forecasts to ~38 M Indian farmers; a 2024 PNAS study used
  data-driven oscillatory-mode forecasts to improve South-Asian monsoon rainfall.
  The monsoon intraseasonal oscillation (MISO/BSISO) sets the *active/break* spatial
  rainfall structure — the variable farmers and water managers actually need.
- **A known AI failure mode is untested here.** AI models damp variability at long
  leads (regression-to-mean). If they kill the northward-propagating MISO, their
  aggregate "skill" can look good while the *useful* signal (active/break timing) is
  gone. Nobody has cleanly shown this for the Indian monsoon.
- **Methodological novelty:** our **dual climatology basis** (score vs shared ERA5
  clim *and* vs each model's own hindcast clim) cleanly separates genuine anomaly
  skill from matching a biased model climatology — a transferable diagnostic.

## 3. Central questions / hypotheses

- **Q1 (benchmark).** How skilful are AI (FuXi) vs dynamical (ECMWF) S2S forecasts
  of monsoon rainfall (TP) and circulation (Z500) over India and the 4 IMD regions,
  week 1–6, across 18 seasons (2002–2019)?
- **Q2 (genuine vs climatological).** How much of each model's apparent skill is real
  anomaly prediction vs matching its own climatology? *Hypothesis:* AI/dynamical
  rainfall skill collapses on the model-own basis (climatological mimicry), while
  circulation skill is genuine. (Pilot already shows ECMWF TP 0.82→0.50 vs Z500
  0.98→0.97.)
- **Q3 (process — the headline).** Do the models reproduce the **MISO/BSISO**
  northward propagation and **active/break spell** timing? *Hypothesis:* skill for
  the intraseasonal envelope decays much faster than aggregate skill suggests; AI
  damps the oscillation amplitude more than ECMWF at weeks 3–6.
- **Q4 (operational frontier).** As an operational complement: how does the newest
  AI model **SPIRE** compare against FuXi/ECMWF in the JFM2026 real-time window?

## 4. Data (all in hand)

| Role | Source | Coverage |
|---|---|---|
| AI model | **FuXi-S2S** 50-member hindcast | 2002–2021, Mon/Thu, TP+Z500 (+t2m,winds) |
| Dynamical | **ECMWF** reforecast 11-member | 2000–2019, TP+Z500 |
| Newest AI (operational) | **SPIRE** ens mean/std | JFM2026, TP+Z500(+T2M) |
| Truth | **WeatherBench2 ERA5** | 1959–2023, 1.5° & 0.25° |
| Climatology | ERA5 30-yr DOY + model-own hindcast clims | — |
| Regions | IMD 4 homogeneous regions (SOI shapefile) | — |

Overlap for the core study: **FuXi × ECMWF, JJAS 2002–2019 (18 yr)**.

## 5. Methods (engine already built — `final_analysis/`)

Config-driven verification: common-grid regridding (1.5° + native 0.5°),
cosine-weighted WMO metrics (ACC/PCC, RMSE, bias, MSSS), probabilistic
(CRPS/CRPSS, spread-skill ratio, Brier/BSS, reliability), All-India + 4 IMD
regions, weekly (W1–6) + daily leads, **dual climatology basis**, and month-wise
(Jun/Jul/Aug/Sep) stratification. Each model is one pluggable adapter; truth from
WB2. *This is done and validated.*

**New analysis modules to add (the novel science):**
1. **MISO/BSISO index** — project forecast & obs daily anomalies (OLR-proxy: TP;
   and U850 if added) onto the BSISO EOFs (Lee et al. 2013 / Kikuchi) → bivariate
   index; score amplitude error + phase error vs lead (like an MJO bivariate skill
   ACC, target 0.5 crossing). *Headline figure.*
2. **Active/break spell detection** — standardized IMD-region rainfall anomaly;
   define active (>+1σ) / break (<−1σ) spells; score hit rate / ETS / lead-time of
   spell onset prediction.
3. **Monsoon onset** (optional) — onset date over Kerala/All-India from forecast vs
   obs; lead-dependent onset-date error.
4. **Variance/ amplitude diagnostics** — ratio of forecast to obs intraseasonal
   (20–60 d band-pass) variance vs lead → quantify AI damping.

## 6. Figures (paper)

1. Skill scorecard: PCC/CRPSS by model × week × variable (All-India). *(built)*
2. Skill vs lead, TP & Z500, with W1–6 horizon. *(built)*
3. **Dual-basis** era5 vs model_own — the genuine-vs-climatological-skill panel.
   *(built; the key methodological figure)*
4. Month-wise × IMD-region heatmaps (Jun/Jul/Aug/Sep evolution). *(built)*
5. Per-year JJAS rainfall + anomaly maps on India; forecast bias maps. *(maps tool
   in progress)*
6. **MISO/BSISO bivariate skill** (amplitude + phase error vs lead) — *headline.* *(to build)*
7. **Active/break spell** hit-rate / lead-time skill by region. *(to build)*
8. Intraseasonal variance ratio vs lead (AI damping). *(to build)*
9. Spread-skill / reliability (calibration). *(built)*
10. (Box) SPIRE operational JFM2026 snapshot vs FuXi/ECMWF. *(built)*

## 7. Novel contributions (what referees will cite)

1. First **18-yr, multi-model, IMD-regional** AI-vs-dynamical S2S benchmark for the
   Indian monsoon.
2. **Genuine-vs-climatological skill** diagnosis via the dual basis — shows
   rainfall "skill" is often climatological mimicry of a biased model clim.
3. **Process verdict**: whether AI models capture the **MISO/active-break** cycle,
   and quantification of AI variance damping at S2S leads.
4. Operational frontier: independent look at the newest commercial AI model (SPIRE).

## 8. Target & timeline (fast track)

- **Target:** npj Climate & Atmospheric Science or GRL (process story → possibly
  Nature Communications). Backup: QJRMS, JAMES, Climate Dynamics.
- **Fast path:** Phases — (A) finish FuXi extraction + full JJAS 2002–2019 run
  [pipeline ready, ~compute]; (B) build MISO + active/break modules [the new
  science]; (C) draft around figures 3, 6, 7 as the core. The benchmark + dual-basis
  + month/region figures already exist → a complete results section is days away
  once the multi-year run lands.

## 9. Risks / caveats (pre-empt referees)

- **No SPIRE hindcast** → SPIRE only in the JFM2026 operational box, not the 18-yr
  core. Frame honestly as "newest-model snapshot."
- **No T2M for ECMWF reforecast** → core is TP+Z500 (fine for monsoon).
- **Instantaneous vs daily-mean** caveats already handled/documented.
- **MISO index choice** — use an established definition (Lee et al. 2013 BSISO) and
  report sensitivity; need U850 (add FuXi/ERA5 winds — FuXi has them).
- **Weekly CRPSS uses a daily-scale clim spread** (V3-inherited) → document, or
  switch to a weekly-aggregated reference for the probabilistic numbers.
- **Reforecast ensemble sizes differ** (FuXi 50 vs ECMWF 11) → note; optionally
  sub-sample FuXi to 11 for a fair probabilistic comparison.

---

### Sources (positioning)
- FuXi-S2S — Nature Communications (2024): https://www.nature.com/articles/s41467-024-50714-1
- FuXi-S2S preprint: https://arxiv.org/abs/2312.09926
- Data-driven oscillatory-mode S2S monsoon rainfall — PNAS (2024): https://www.pnas.org/doi/10.1073/pnas.2312573121
- Probabilistic AI monsoon forecasts for agriculture (2026): https://arxiv.org/abs/2603.07893
- Deep learning reveals moisture as MJO predictability source — npj Clim Atmos Sci: https://www.nature.com/articles/s41612-023-00561-6
