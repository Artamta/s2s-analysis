# Spire JFM 2026 S2S Hindcast Verification — Master Summary

**Author working notes / methods reference for the paper.**
Last updated: 2026-06-08.

This document is the single source of truth for: *what each script does, what each
figure shows, how it is computed, which figures are publication-ready, and the
caveats a reviewer will raise.* Read this before touching any figure.

---

## 1. The study in one paragraph

We verify the **Spire S2S hindcast** (90 daily initialisations spanning JFM 2026,
2026-01-01 → 2026-03-31; lead 1–46 days, global 0.5° grid) against **ERA5
reanalysis as ground truth**, over the **India domain** (0–50°N, 55–105°E). We
evaluate weekly-mean anomalies at lead weeks **W1 (days 1–7) … W6 (days 36–42)**
for four fields: 2 m **mean** temperature, 2 m **max** temperature,
**precipitation**, and **500 hPa geopotential height (Z500)**. Skill is measured by
the **Anomaly Correlation Coefficient (ACC)**, **RMSE**, and **bias**, computed per
grid point across the 90 initialisations and summarised over India.

## 2. Data sources

| Source | What | Access |
|---|---|---|
| **Spire** | Hindcast ensemble mean (`mean_stddev` group: `air_temperature`, `air_temperature_max`) and pre-computed anomalies (`anomalies` group: `precipitation_amount`, `geopotential_height`) | arraylake `artamta/s2s-research`, readonly |
| **ERA5 (ARCO)** | Ground-truth `2m_temperature`, `total_precipitation`, `geopotential@500` | `gs://gcp-public-data-arco-era5/...` (anon), 0.25° → interpolated to Spire 0.5° |
| **ERA5 climatology** | Daily-mean & daily-max T2m DOY climatology, 1991–2020 (computed here from ARCO hourly) | cached: `era5_tmean_climo_india.nc`, `era5_tmax_climo_india.nc` |
| **WB2 climatology** | Z500 & precip DOY climatology, 1990–2019 | earth2studio `WB2Climatology` |

## 3. The pipeline (run order)

```
09_compute_anomalies_v2.py   →  weekly_anomalies_v2.nc      # the corrected data file
12_master_publication_plots.py → figures/pub/P01..P15.png    # all publication figures
make_main_figure.py          →  figures/paper/Fig1_main.png  # the consolidated main figure
```

`weekly_anomalies_v2.nc` has dims `(init_time=90, week=6, lat=101, lon=101)` and 8
variables: `{spire,era5}_t2m_mean_anom`, `{spire,era5}_t2m_max_anom`,
`{spire,era5}_precip_anom`, `{spire,era5}_z500_anom`.

### How anomalies are computed (the critical part)

An anomaly = value − climatology. **The comparison is only valid if forecast and
observation use the *same* climatology.** Each variant does:

- **Mean T2m:** Spire `air_temperature` and ERA5 daily-mean T2m, **both** minus the
  **ERA5 1991–2020 daily-mean DOY climatology**.
- **Max T2m:** Spire `air_temperature_max` and ERA5 daily-max T2m, **both** minus
  the **ERA5 1991–2020 daily-max DOY climatology**. *(This is Spire's native
  product — the headline variant.)*
- **Z500 / Precip:** Spire from its `anomalies` group; ERA5 minus WB2 1990–2019.
  Both reference ~30-yr observational climo (see caveat §6).

ACC at each grid point = Pearson correlation between the Spire and ERA5 anomaly
**across the 90 init dates**; then averaged over India. RMSE/bias likewise.

## 4. ⚠️ The bug that was fixed (read this)

The original mean-T2m variant subtracted the **WB2 climatology sampled at 00 UTC
only**. 00 UTC ≈ 05:30 IST = near the daily *minimum*. Verified: WB2 Feb-1 central
India 00 UTC = 10.8 °C vs true daily mean = 16.5 °C — the baseline was **~5.7 K too
cold**, which inflated the mean-T2m anomaly to a spurious **uniform +4 K** in *both*
Spire and ERA5. The tell-tale: mean anomaly (+4 K) exceeded max anomaly (+1.3 K)
for the same data, which is physically impossible.

**Fix:** an ERA5 daily-mean DOY climatology computed from ARCO hourly data (same
fetch as the Tmax climo), used for both sides. Mean and max variants are now
methodologically identical. *Earlier figures showing +4 K mean anomalies are
superseded — do not cite those magnitudes.*

## 5. Figure guide — what to use in the paper

All in `figures/pub/`. ✅ = publication-ready, ⚠️ = use with caveat, ❌ = superseded.

### Spatial anomaly maps
- **`P02_anomaly_max.png`** ✅ **— HEADLINE.** 2×6, Spire vs ERA5 daily-max T2m
  anomaly, W1→W6. Spire's native product, correct baseline. *Story: how well Spire
  reproduces the observed warm pattern and how it decays with lead.*
- **`P01_anomaly_mean.png`** ✅ (after re-run) — same for daily-mean T2m. Companion
  to P02. Both vs ERA5 1991–2020 climo.

### Bias maps (forecast − obs)
- **`P03_bias_mean.png`**, **`P04_bias_max.png`** ✅ — 2×3, where/how much Spire is
  too warm/cold vs ERA5. Robust (any climo offset cancels). Good supplementary.

### Scatter / quantitative agreement
- **`P06_scatter_max.png`** ✅ — grid-cell Spire vs ERA5, per week, with r, R², RMSE,
  MAE, bias, 1:1 and OLS lines. **The quantitative agreement figure.** (`P05` = mean.)
- **`P08_scatter_initmean_max.png`** ✅ — each dot = 1 init date (India-mean). Shows
  how well Spire tracks the sub-seasonal evolution init-by-init. (`P07` = mean.)

### Skill summaries
- **`P09_acc_skill_maps.png`** ✅ — 3×6 per-grid-point ACC maps (T2m/Precip/Z500),
  stippled where |ACC|<0.3. **The spatial skill figure.**
- **`P10_acc_vs_lead.png`** ✅ — India-mean ACC vs lead for all 4 fields with the
  0.5 skill line. **The single most important summary plot** — defines the useful
  forecast horizon.
- **`P11_rmse_vs_lead.png`**, **`P12_bias_vs_lead.png`**, **`P13_anomaly_vs_lead.png`**
  ✅ — supporting error-growth curves.
- **`P14_skill_dashboard.png`** ✅ — 6-panel overview (ACC, RMSE, bias, anomaly,
  scatter, ACC heatmap). Great for a talk; pick panels for the paper.
- **`P15_acc_rmse_heatmap.png`** ✅ — variable × week ACC & RMSE tables as heatmaps.
  Use this to fill the results table.

### Superseded / legacy (do NOT use in paper)
- `fig1`–`fig7` (script 08), `tmax_india_anomaly.png` (06), `bams_*` (07) — earlier
  baselines / the zero-anomaly bug. Kept only for provenance.

## 6. Per-variable verdict

| Variable | Verdict | Note |
|---|---|---|
| **T2m max** | ✅ Trust fully | Native Spire product; consistent ERA5 Tmax climo |
| **T2m mean** | ✅ Trust (post-fix) | Consistent ERA5 daily-mean climo |
| **Z500** | ✅ Trust | Weak diurnal cycle; Spire ≈ ERA5 (~+25 gpm) |
| **Precip** | ⚠️ Pattern/ACC OK; magnitude caveat | Spire ≈ 0 vs ERA5 +1 mm/day. Both use ~30-yr obs climo but different products; treat the dry-vs-wet gap as indicative, not exact. |

## 7. Reviewer caveats — see `REVIEWER_CAVEATS.md`
The 90 daily inits **overlap** (serial correlation → effective N ≪ 90); this is a
**single-season case study**, not a multi-year hindcast skill estimate. Frame
accordingly and adjust significance thresholds. Full discussion in that file.
