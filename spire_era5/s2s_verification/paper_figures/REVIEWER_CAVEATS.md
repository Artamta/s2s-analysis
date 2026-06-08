# Reviewer Caveats & Defensive Framing

Issues a referee *will* raise about the Spire JFM 2026 verification, and how to
handle each. Address these in the paper proactively — they are about framing and
statistics, not bugs.

---

## 1. The 90 initialisations are NOT independent  ⚠️ most important

The 90 inits are **consecutive daily** starts. Their verification windows overlap
heavily (W1 of init *d* and W1 of init *d+1* share 6 of 7 valid days; the verifying
atmospheric state is almost the same). So the **effective sample size is far smaller
than 90** — closer to the number of independent ~weekly/synoptic events in JFM,
i.e. roughly **8–15**, not 90.

**Consequences**
- ACC is computed across these 90 inits per grid point. The correlation is real but
  its **degrees of freedom are inflated**.
- The stippling threshold |ACC| < 0.3 (a rough 95% level for n=90) is **too lenient**.
  With effective N ≈ 12, the 95% significance threshold is roughly **|r| ≳ 0.5**.

**How to handle**
- State explicitly that inits are daily and overlapping; report an **effective
  sample size** (e.g. estimate via lag-1 autocorrelation of the init-mean series,
  N_eff = N·(1−ρ)/(1+ρ), or by block-bootstrapping whole weeks).
- Re-draw significance stippling at the N_eff-based threshold (≈0.5), or bootstrap
  confidence intervals on the India-mean ACC curve (`P10`).
- Alternatively subsample to ~weekly inits (every 7th) for an independence check and
  show the skill curve is robust.

## 2. Single season → "case study", not "hindcast skill"

All 90 inits are within **one JFM season (2026)**. The anomalies are departures from
a 30-yr DOY climatology, so they capture **one season's sub-seasonal evolution**, not
a multi-year skill distribution. A reader may over-interpret the ACC as Spire's
general S2S skill.

**How to handle:** frame as a **case study of JFM 2026** (an anomalously warm season
over India). Skill numbers describe how well Spire tracks *this* season's evolution.
If multi-year hindcasts are available, a follow-up across seasons would generalise.

## 3. Climatology period & product consistency

- T2m mean/max climo: ERA5 **1991–2020** (computed here from ARCO hourly).
- Z500/precip climo: WB2 **1990–2019**.
- The two differ by one year — negligible, but **state both periods**.
- **Precip** especially: Spire precip anomaly comes from Spire's own `anomalies`
  group while ERA5 precip uses WB2. Both are ~30-yr observational baselines, but the
  products differ. The Spire≈0 vs ERA5≈+1 mm/day gap is **indicative of a model dry
  bias**, but do not quote it as an exact calibrated number.

## 4. Verification grid / interpolation

ERA5 0.25° is linearly interpolated to Spire 0.5° (truth → forecast grid). Linear
interpolation smooths ERA5 slightly, which can marginally *inflate* ACC and *reduce*
RMSE at small scales. Standard practice; just disclose it. (Conservative alternative:
conservative/area-weighted regridding.)

## 5. ACC formulation

ACC here = **temporal** Pearson r across inits at each grid point (a "local" /
grid-point ACC), then area-averaged. This differs from the **centred pattern ACC**
(spatial correlation of anomaly maps per init) used in some S2S papers. Both are
valid — **state which one** and cite accordingly. Consider also reporting the
pattern-ACC per init as a complementary metric.

## 6. Ensemble mean vs spread

We verify the **ensemble mean** only. Ensemble-mean ACC/RMSE say nothing about
**reliability/calibration**. If Spire provides ensemble members or tercile
probabilities (it does — `probabilities` group), add at least one **probabilistic**
score (RPSS or reliability diagram) — referees of S2S papers expect it.

## 7. Area averaging includes ocean

The India box (0–50°N, 55–105°E) includes Arabian Sea / Bay of Bengal. SST-driven
grid points behave differently from land. Consider a **land-only mask** for the
India-mean curves, or show land and ocean separately.

---

### Minimal checklist to satisfy a referee
- [ ] Report effective sample size; redo significance at N_eff threshold (or bootstrap).
- [ ] Frame as JFM-2026 case study.
- [ ] State both climatology periods and the precip product caveat.
- [ ] Disclose interpolation direction and ACC definition.
- [ ] Add one probabilistic score from the `probabilities` group.
- [ ] (Optional) land-only India mean.
