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
5. Per-year JJAS rainfall + anomaly maps on India; model-vs-obs maps (TP & Z500,
   FuXi & ECMWF) and forecast bias maps. *(built — polished publication-grade)*
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

## 10. Research & figure roadmap — what MORE to do (prioritized)

*Added 2026-06-26. Ranked by scientific leverage ÷ cost. The WB2 ERA5 store is
far richer than we've used: winds @13 levels, OLR-proxy (top net LW flux),
vertical velocity, moisture divergence, MSLP, SST, IVT, soil moisture, RH/q.
This unlocks process diagnostics no current AI-S2S monsoon paper has shown.*

### Tier 1 — the headline novelty (build next; high impact, moderate cost)
- **MISO/BSISO bivariate skill** *(§5.1, the headline)*. Project obs+forecast daily
  anomalies onto BSISO1/BSISO2 EOFs (Lee et al. 2013; OLR-proxy = top-net-LW or
  −TP, plus U850 — both in WB2 & FuXi). → bivariate amplitude error + phase error
  vs lead; "useful-skill horizon" where bivariate ACC crosses 0.5. *Figure: ACC &
  RMSE-amplitude & phase-error vs lead, FuXi vs ECMWF.* This is the figure referees
  will cite — it tests FuXi's own stated future-work gap.
- **Active/break spell skill** *(§5.2)*. Standardize area-mean rainfall over the
  Monsoon-Core-Zone (MCZ, ~18–28°N, 73–82°E) and the 4 IMD regions; define
  active(>+1σ)/break(<−1σ) spells; score **hit-rate / false-alarm / ETS / Heidke**
  and **lead-time of onset** of each spell. *Figure: spell-onset hit-rate vs lead +
  a 2019 case-study timeline (obs vs FuXi vs ECMWF rainfall-anomaly index).*
- **Intraseasonal variance ratio vs lead** *(§5.4 — the "AI damping" smoking gun)*.
  Band-pass (20–60 d) the forecast & obs rainfall/U850; plot
  var(forecast)/var(obs) vs lead per model. *Hypothesis: AI damps faster.* Cheap
  once daily fields are in hand; pairs with the MISO figure.

### Tier 2 — cheap, high-value figures from data ALREADY computed
- **Multi-panel skill scorecard (one figure).** A single heatmap/scorecard: rows =
  {TP, Z500} × {era5, model_own} × {FuXi, ECMWF}, cols = W1–6, cell = PCC (and a
  twin for CRPSS). Replaces several line plots with one citable "table-figure."
- **Taylor diagram** per variable/lead — correlation + normalized σ + centred-RMSE
  in one glance; classic, compact, reviewers love it. *(matplotlib, from existing CSVs.)*
- **Skill-horizon bar** — the lead (in days) at which PCC drops below 0.5, per
  model × variable × IMD region. One bar chart = the paper's "how far out is it
  useful" answer.
- **Forecast-vs-obs scatter / Q–Q of area-mean rainfall** per week — shows the wet
  bias & variance compression directly (complements the dual-basis story).
- **Bias-vs-lead growth curves** (signed area-mean bias W1–6) — does the wet bias
  grow, saturate, or drift? Cheap from the deterministic CSV.
- **Reliability + sharpness** for active/break (tercile) events — already have the
  reliability machinery; extend to the monsoon events.

### Tier 3 — richer process & teleconnection science (WB2 enables; medium cost)
- **OLR / convection verification.** Use top-net-LW flux as OLR proxy → verify the
  convective envelope (the variable MISO is classically defined on), not just TP.
- **Vertical structure of bias.** Z & T & RH bias profiles (50–1000 hPa) over the
  MCZ vs lead — does the AI model drift in the mid-troposphere? *(WB2 has all levels.)*
- **Moisture-budget / IVT skill.** Verify forecast IVT (computed from q·V, not the
  WB2 `integrated_vapor_transport` which is in bad units — see dynamics module) and
  vertically-integrated moisture divergence → the monsoon's moisture supply.
- **Low-Level Jet (Somali/Findlater) & TEJ skill.** U850 over the Arabian Sea LLJ
  box and U200 TEJ index vs lead — circulation drivers of active/break.
- **Teleconnection conditioning.** Stratify skill by ENSO/IOD phase (we already have
  Niño3.4↔ISMR r=−0.40, IOD links): is monsoon skill higher in ENSO-active years?
- **Onset/withdrawal date error** (optional) — Kerala onset & all-India advance from
  forecast vs obs; lead-dependent onset-date error in days.

### Tier 4 — robustness, fairness, framing (referee-proofing)
- **Fair-ensemble comparison** — sub-sample FuXi 50→11 members to match ECMWF for an
  apples-to-apples CRPSS/spread-skill; report both raw and matched.
- **Bootstrap CIs** on all pooled-year skill (resample inits/years) → error bars on
  every skill-vs-lead curve; significance of FuXi−ECMWF differences.
- **Native-resolution check** — repeat the core skill at 0.25°/0.5° to show
  conclusions aren't a regridding artefact (engine already supports `--dgrid`).
- **Weekly-spread CRPSS fix** — switch the probabilistic reference to a
  weekly-aggregated clim spread (current daily-scale spread makes weekly CRPSS
  uniformly optimistic — §9 caveat). Re-report probabilistic numbers.
- **Persistence & climatology baselines on every panel** — already in JFM; carry the
  same skill-floor lines into JJAS so "beats persistence" is explicit.

### ERA5 long-term context figures (finish the stuck Agent-A set)
Only `1_clim_maps_jjas.png` landed. Re-run `analysis/era5_monsoon.py` (it was slow)
to complete: JJAS annual cycle, interannual ISMR + trend, trend maps, MISO
Hovmöller (lat–time northward propagation), and intraseasonal-variance maps —
these are the obs "stage-setting" figures (1–2 in the paper) and feed the MISO module.

### Suggested final figure set (≈10, mix of built + to-build)
1 scorecard (T2) · 2 skill-vs-lead+horizon (built) · 3 **dual-basis** (built, key) ·
4 month×region (built) · 5 model-vs-obs maps TP+Z500 (built, just polished) ·
6 **MISO bivariate skill** (T1, headline) · 7 **active/break onset skill + 2019
case** (T1) · 8 **intraseasonal variance ratio** (T1) · 9 Taylor + reliability
(T2) · 10 SPIRE JFM2026 box + ERA5 long-term context inset (built).

---

### Sources (positioning)
- FuXi-S2S — Nature Communications (2024): https://www.nature.com/articles/s41467-024-50714-1
- FuXi-S2S preprint: https://arxiv.org/abs/2312.09926
- Data-driven oscillatory-mode S2S monsoon rainfall — PNAS (2024): https://www.pnas.org/doi/10.1073/pnas.2312573121
- Probabilistic AI monsoon forecasts for agriculture (2026): https://arxiv.org/abs/2603.07893
- Deep learning reveals moisture as MJO predictability source — npj Clim Atmos Sci: https://www.nature.com/articles/s41612-023-00561-6
