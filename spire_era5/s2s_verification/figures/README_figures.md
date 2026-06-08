# Spire JFM 2026 S2S Verification — Figure Guide

**Domain:** India subcontinent (lat 0–50°N, lon 55–105°E), 0.5° grid.
**Forecast:** Spire S2S hindcast, 90 init dates spanning JFM 2026 (2026-01-01 → 2026-03-31).
**Verification:** ERA5 reanalysis.
**Lead weeks:** W1 = days 1–7, W2 = 8–14, W3 = 15–21, W4 = 22–28, W5 = 29–35, W6 = 36–42.
**Borders:** coastlines only (no political/disputed borders).

---

## ⚠️ Important note on anomaly baselines (read first)

An *anomaly* = (value) − (climatology). The comparison is only meaningful if the
forecast and the observations are measured against the **same climatology**. Earlier
figures mixed baselines and one had a bug, so they are superseded:

| Figure set | Status | Problem |
|------------|--------|---------|
| `fig1`–`fig7` (script 08) | ❌ superseded | Spire T2m anomaly was computed as *raw − mean over the same 90 inits*, which is **identically zero** when re-averaged. That's why the Spire row is blank/white and every panel reads "India=+0.0K". |
| `tmax_india_anomaly.png` (script 06) | ⚠️ legacy | Spire uses its **own** hindcast climatology (anomalies group, ±0.7 K); ERA5 uses WB2. Different baselines → magnitudes not directly comparable. |
| `bams_global_tmax.png`, `bams_india_tmax.png` (script 07) | ⚠️ legacy | Same baseline mismatch as above; also had an empty 8th panel (now hidden). |
| **`fig_A`–`fig_E` (script 10, v2)** | ✅ **use these** | Spire and ERA5 share an **identical** climatology in each variant. Real, comparable anomalies. |

---

## ✅ Primary figures (v2 — consistent baselines)

These come from `weekly_anomalies_v2.nc` (built by `09_compute_anomalies_v2.py`).
Two physically-matched variants are provided because Spire forecasts *daily-max*
temperature while many analyses use *daily-mean*:

- **mean-vs-mean:** Spire `air_temperature` and ERA5 daily-mean T2m, both vs the
  WeatherBench-2 1990–2019 mean-T2m day-of-year climatology.
- **max-vs-max:** Spire `air_temperature_max` and ERA5 daily-max T2m, both vs an
  ERA5 1991–2020 daily-Tmax day-of-year climatology (computed for this study,
  cached in `era5_tmax_climo_india.nc`).

### `fig_A_t2m_mean_anomaly.png` — Mean-T2m anomaly, Spire vs ERA5
2×6 grid. **Top row:** Spire forecast anomaly (W1→W6). **Bottom row:** ERA5 observed
anomaly. Both vs WB2 1990–2019. Red = warmer than normal, blue = cooler.
Panel corner `μ=` is the India-domain mean.
- *What to look for:* In W1 the forecast pattern should resemble the observed pattern;
  by W4–W6 the forecast skill decays and the anomaly weakens toward climatology.
- *Story:* JFM 2026 was anomalously warm in the observations; how much of that warmth
  the forecast captures, and how fast it loses it with lead time.

### `fig_B_t2m_max_anomaly.png` — Max-T2m anomaly, Spire vs ERA5
Same layout as `fig_A` but for **daily-maximum** temperature, both vs the ERA5
1991–2020 Tmax climatology. This is the physically correct match for Spire's native
`air_temperature_max` product. Use this one when discussing heat extremes / Tmax.

### `fig_C_bias_mean.png` — Mean-T2m forecast bias (Spire − ERA5)
2×3 grid, one panel per week. Shows where and by how much the forecast is too warm
(red) or too cold (blue) relative to the verifying observations.
- *What to look for:* A spatially coherent cold/warm bias and how it grows with lead.
- `μ=` corner is the India-mean bias.

### `fig_D_bias_max.png` — Max-T2m forecast bias (Spire − ERA5)
Same as `fig_C` for daily-max temperature.

### `fig_E_skill_summary.png` — India-mean skill vs lead week (3 panels)
- **(a) ACC** — Anomaly Correlation Coefficient: Pearson correlation between forecast
  and observed anomaly across the 90 init dates, computed per grid point then averaged
  over India. Plotted for both mean and max variants. The shaded band marks the
  conventional skill threshold ACC = 0.5. Higher = better; expect a monotonic decay
  W1→W6 and a crossing of 0.5 that defines the useful forecast horizon.
- **(b) RMSE** — root-mean-square error (K) vs ERA5, both variants. Lower = better;
  rises with lead time.
- **(c) India-mean anomaly** — forecast vs observed mean anomaly per week, all four
  series (Spire/ERA5 × mean/max). Shows whether the forecast captures the overall
  magnitude of the seasonal warm anomaly or damps it toward zero.

---

## Legacy / reference figures

### `tmax_india_anomaly.png` (script 06)
2×6 India layout, Spire Tmax (own anomalies group) vs ERA5 (WB2). Colorbar ±8 K.
Kept for reference; baselines differ (see table above).

### `bams_global_tmax.png` (script 07)
2×4 global Robinson projection, BAMS-multimodel style. Panel (a) = ERA5 verification,
(b)–(g) = Spire W1–W6. Discrete ±12 K colormap. The unused 8th cell is now hidden.
Good for showing the *global* context of the forecast anomaly field.

### `bams_india_tmax.png` (script 07)
Same BAMS style zoomed to India (PlateCarree). 7 panels, 8th cell hidden.

### `fig3_acc_skill_maps.png`, `fig4_rmse_skill_maps.png` (script 08)
Per-grid-point ACC and RMSE maps for T2m, precipitation, and Z500 (3×6). The *map*
skill metrics here are valid (they don't depend on the Spire-T2m-anomaly bug, since
ACC/RMSE come from `skill_metrics.nc`). Useful as spatial skill diagnostics.
Note: the precipitation and Z500 anomalies still use Spire's anomalies-group baseline.

---

## Data provenance

| File | Built by | Contents |
|------|----------|----------|
| `weekly_anomalies_v2.nc` | `09_compute_anomalies_v2.py` | Consistent-baseline weekly anomalies (mean & max variants) |
| `era5_tmax_climo_india.nc` | `09_compute_anomalies_v2.py` | ERA5 1991–2020 daily-Tmax DOY climatology (cache) |
| `weekly_anomalies.nc` | `02_compute_weekly_anomalies.py` | ⚠️ legacy — Spire T2m anomaly is buggy (zero) |
| `skill_metrics.nc` | `03_skill_maps.py` | ACC/RMSE/bias maps for T2m, precip, Z500 |

**Reproduce primary figures:**
```bash
conda run -n s2s-hind python -u 09_compute_anomalies_v2.py   # builds weekly_anomalies_v2.nc (one-time, slow: ERA5 Tmax climo)
conda run -n s2s-hind python -u 10_publication_figures_v2.py # builds fig_A … fig_E
```
