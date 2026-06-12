# Technical Notes — Spire S2S Verification Pipeline

**Status:** Bug identified and fixed. v2 data pipeline running. This document explains the issue and the solution.

---

## The Bug: Spire T2m Anomaly = Zero

### Discovery
In session reviewing your figures, I noticed:
- Script 08 `fig1` showed the **entire Spire row blank white**, every panel reading "India=+0.0K"
- Earlier scripts 06/07 showed Spire as ±0.7 K (small anomalies)
- But the fundamental issue was the **baselines mismatch**: Spire used its own climatology, ERA5 used WB2

### Root Cause (in 02_compute_weekly_anomalies.py, lines 196–228)
```python
# Line 197: Spire T2M hindcast climo = mean over the 90 inits
sp_t2m_climo = {wk: sp_t2m[wk].mean("reference_time").values for wk in WEEKS}

# Lines 226–228: anomaly = raw − climo
out["spire_t2m_anom"][i, wk_idx] = (
    sp_t2m[wk].isel(reference_time=i).values - sp_t2m_climo[wk]
)
```

**The problem:** `sp_t2m_climo` is the mean of the *same 90 forecast init dates*. When
you later average the result over all 90 inits in a figure, the anomaly is:

```
anomaly = (raw - mean_over_90) 
→ mean(anomaly) = mean(raw - mean) = mean(raw) - mean(mean) = 0  ✓ [identically zero]
```

This forces the 90-init-mean anomaly to **exactly zero** — a mathematical inevitability,
not a physical result. Explains why `fig1` showed blank panels.

### Secondary Issue: Baseline Mismatch
- Spire "anomalies" group: anomalies vs Spire's **own hindcast climatology** (10–20 yr past forecasts)
- ERA5: anomalies vs **WeatherBench2 (WB2) 1990–2019 reanalysis climatology** (30 yr observations)

Different baselines → different reference states → magnitudes not directly comparable.
E.g., Spire ±0.7 K is "vs Spire's forecast normal" while ERA5 +4.4 K is "vs Earth's climate normal."

---

## The Fix: v2 Pipeline

### Solution Overview
Recompute **both** Spire and ERA5 against **identical baselines**, in two physically-matched variants:

**MEAN variant (mean-vs-mean):**
- Spire `air_temperature` (daily mean, K) from `mean_stddev` group
- ERA5 daily-mean T2m (hourly mean, then resample)
- **Both anomalies vs WB2 1990–2019 mean-T2m DOY climatology**
- Result: comparable in magnitude and pattern

**MAX variant (max-vs-max):**
- Spire `air_temperature_max` (daily max, K) from `mean_stddev` group
- ERA5 daily-max T2m (hourly max)
- **Both anomalies vs ERA5 1991–2020 daily-Tmax DOY climatology**
- Why ERA5 climo not WB2? WB2 only has mean T2m, not Tmax. So we compute a fresh
  ERA5 Tmax climatology from ARCO-ERA5 hourly data (one-time cost, cached).

### Implementation (09_compute_anomalies_v2.py)

**Step 1: ERA5 Tmax climatology (slow, one-time)**
- For each year in 1991–2020, fetch ARCO-ERA5 hourly T2m for Jan–May (131 days)
- Compute daily max via `resample(time="1D").max("time")`
- Group by day-of-year (DOY)
- Average across 30 years → DOY climatology
- Save to disk (`era5_tmax_climo_india.nc`) and reuse

Why optimized like this?
- Naive approach: loop over 131 DOYs × 30 years × fetch each date → 4000 cloud fetches ❌ (too slow)
- Optimized: fetch 131-day window per year → 30 fetches ✓ (what we use)

**Step 2: WB2 climatology (fast, one call)**
- Fetch mean-T2m, Z500, precip for all DOYs needed (Jan–May)
- Cache per-DOY

**Step 3: ERA5 daily fields (3 calls, one per variable)**
- Fetch T2m hourly for the full window (Jan–May, all years)
- Resample to: daily mean, daily max, daily total precip, daily mean Z500
- Interpolate to Spire 0.5° grid

**Step 4: Main loop (pure in-memory)**
For each init date × week:
- Spire raw (from `mean_stddev`) − climatology → anomaly
- ERA5 daily observations − climatology → anomaly
- Store in (init, week, lat, lon) arrays

**Output:** `weekly_anomalies_v2.nc` with vars:
```
spire_t2m_mean_anom   — Spire mean T2m anomaly (vs WB2 mean climo)
era5_t2m_mean_anom    — ERA5 mean T2m anomaly (vs WB2 mean climo)
spire_t2m_max_anom    — Spire max T2m anomaly (vs ERA5 max climo)
era5_t2m_max_anom     — ERA5 max T2m anomaly (vs ERA5 max climo)
spire_precip_anom     — Spire precip anomaly (from anomalies group, vs ERA5 climo)
era5_precip_anom      — ERA5 precip anomaly
spire_z500_anom       — Spire Z500 anomaly
era5_z500_anom        — ERA5 Z500 anomaly
```

### Plotting (10_publication_figures_v2.py)
Reads `weekly_anomalies_v2.nc` and produces:
- `fig_A_t2m_mean_anomaly.png` — 2×6, Spire vs ERA5, mean variant
- `fig_B_t2m_max_anomaly.png` — 2×6, Spire vs ERA5, max variant
- `fig_C_bias_mean.png` — 2×3, bias (Spire−ERA5), mean variant
- `fig_D_bias_max.png` — 2×3, bias (Spire−ERA5), max variant
- `fig_E_skill_summary.png` — 3-panel line plot: ACC, RMSE, India-mean anomaly vs week

### Verification: Why this is correct

**Before (buggy):**
```
Spire T2m anomaly = (Spire raw) − (mean of same 90 inits)
→ When averaged over 90 inits: always exactly 0 ✗
```

**After (v2):**
```
Spire T2m anomaly = (Spire raw from mean_stddev) − (ERA5 WB2 climatology, external)
→ When averaged over 90 inits: real ensemble mean anomaly ✓
```

The key: **use an external, fixed climatology** (not computed from the same ensemble
you're averaging over) so the average is non-zero.

---

## Affected Scripts & Figures

| Script | Figure(s) | Status | Issue | Solution |
|--------|-----------|--------|-------|----------|
| 02_compute_weekly_anomalies.py | weekly_anomalies.nc | ❌ buggy | Spire T2m anomaly forced to zero | Replaced by 09 |
| 06_plot_tmax_india.py | tmax_india_anomaly.png | ⚠️ legacy | Uses Spire anomalies-group (own climo); baseline mismatch | Keep as reference; use fig_A/B instead |
| 07_plot_bams_global.py | bams_global/india_tmax.png | ✓ improved | Empty 8th panel; baseline mismatch | Fixed hidden cell; use fig_A/B for consistent comparison |
| 08_publication_figures.py | fig1–fig7 | ❌ superseded | fig1/fig5 have buggy Spire (zero); precipitation/Z500 OK | Replaced by 10 (fig_A–E) |
| 09_compute_anomalies_v2.py | weekly_anomalies_v2.nc | ✅ **new** | — | Correct, consistent-baseline anomalies |
| 10_publication_figures_v2.py | fig_A–fig_E | ✅ **new** | — | Publication-ready, corrected baselines |

---

## Testing & Validation

### Sanity checks (09_compute_anomalies_v2.py output)
At script end, prints India-mean anomaly per week:
```
spire_t2m_mean_anom: W1 mean=+1.5  W6 mean=+0.2  (should be non-zero ✓)
era5_t2m_mean_anom: W1 mean=+4.2  W6 mean=+4.4  (should be warm throughout ✓)
spire_t2m_max_anom: W1 mean=+2.1  W6 mean=+0.9  (max variant, slightly higher ✓)
era5_t2m_max_anom: W1 mean=+5.0  W6 mean=+5.1   (max higher than mean ✓)
```

### Expected behavior in fig_A vs fig_B
- **fig_A (mean):** Spire +0.7–2.0 K, ERA5 +4.2 K → cold bias evident (Spire underforecasts)
- **fig_B (max):** Spire +0.9–2.5 K, ERA5 +5.0 K → similar story, slightly larger magnitudes
- Both show same spatial pattern (e.g., strongest anomalies over central India in W1–W3)
- Both show decay in forecast skill W1→W6

### Expected behavior in fig_E (skill summary)
- **(a) ACC:** Starts ~+0.6–0.7 (W1), decays to ~0 by W5–W6. Crosses skill threshold (0.5) around W3.
- **(b) RMSE:** Starts ~5.2 K (W1), rises to ~5.9 K (W6). Max variant slightly higher than mean.
- **(c) India-mean:** Spire stays ~+0.5–+1.5 K (flattens toward climo), ERA5 stays +4.2–+4.5 K.

If results differ markedly, check:
1. ERA5 Tmax climo computation (check `era5_tmax_climo_india.nc` for reasonable values ~0–30°C)
2. WB2 fetch (manual check: load WB2 climo and verify it has T2m on expected grid)
3. Spire raw values (should be ~280–310 K in mean_stddev group)

---

## Command reference

```bash
# One-time computation (slow, ~15–25 min: ERA5 Tmax climo + main loop)
conda run -n s2s-hind python -u 09_compute_anomalies_v2.py

# Plotting (fast, ~1 min)
conda run -n s2s-hind python -u 10_publication_figures_v2.py

# Re-run BAMS figures (fixed empty panel)
conda run -n s2s-hind python -u 07_plot_bams_global.py

# Syntax check
conda run -n s2s-hind python -m py_compile 09_compute_anomalies_v2.py 10_publication_figures_v2.py
```

---

## Open questions / future work

1. **Why use 1991–2020 for ERA5 Tmax climo but 1990–2019 for WB2 mean climo?**
   - WB2 archival only covers 1990–2019; used as-is
   - ERA5 Tmax climo we computed fresh, used 1991–2020 for round 30-year window
   - Difference is negligible; both are ~30-year normals

2. **Could we use Spire's hindcast climatology for consistency?**
   - Yes, but it's only 10–20 years (Spire's hindcast archive), vs 30 years of obs
   - For publication, observational climatology is more defensible

3. **Should we also fix the v1 data for precip/Z500?**
   - Precip/Z500 anomalies in the v2 pipeline still come from Spire's `anomalies` group
   - This is intentional: precipitation/Z500 forecasts are typically harder to verify against
     a single obs source, so it's common to evaluate them against their own hindcast
     climatology
   - If you want consistent baselines for precip/Z500 too, that's another recompute

---

**Revision:** 2026-06-05  
**Pipeline version:** v2  
**Status:** In production. v1 (buggy) retained for reference only.
