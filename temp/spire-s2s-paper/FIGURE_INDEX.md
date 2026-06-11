# Spire AI-S2S Paper — Master Figure Index

Single source of truth for every figure: what it shows, the script that builds it,
and its status. Domain: India (0–50°N, 55–105°E), 0.5°. 90 daily inits, JFM 2026
(2026-01-01 → 2026-03-31). Verification: ERA5 (and IMD, in progress). **Coastlines
only — no political borders.** All figures in `spire-s2s-paper/figures/`.

Status key: ✅ publication-ready · ⚠️ use with caveat · 🔬 descriptive (not verification)

---

## §1 — Basic: absolute model vs ground truth  (Prof. Singh's steer — LEAD WITH THESE)
| Fig | Shows | Script | Status |
|-----|-------|--------|--------|
| `fig21_basic_spatial_tmax` | Spire \| ERA5 \| (Spire−ERA5), daily-max T, W1/W3/W6 | `14_basic_verification.py` | ✅ |
| `fig22_basic_scatter_tmax` | Spire vs ERA5 Tmax scatter per week (R², RMSE, MAE, bias) | `14_basic_verification.py` | ✅ R²=0.99→0.99 |
| `fig23_basic_spatial_tmean` | Same as fig21 for daily-mean T | `14_basic_verification.py` | ✅ |
| `fig24_basic_scatter_tmean` | Same as fig22 for daily-mean T | `14_basic_verification.py` | ✅ |

## §2 — Deterministic skill (anomaly-based)
| Fig | Shows | Script | Status |
|-----|-------|--------|--------|
| `fig01_acc_vs_lead` | India-mean ACC vs lead, all variables | `fig_deterministic.py` | ✅ |
| `fig02_rmse_vs_lead` | RMSE vs lead | `fig_deterministic.py` | ✅ |
| `fig03_bias_vs_lead` | Bias (Spire−ERA5) vs lead | `fig_deterministic.py` | ✅ |
| `fig04_acc_skill_maps` | Per-gridpoint ACC maps (var × week) | `fig_deterministic.py` | ✅ |
| `fig05_bias_maps` | Per-gridpoint bias maps | `fig_deterministic.py` | ✅ |
| `fig06_scatter_gridcell` | Grid-cell scatter Spire vs ERA5 | `fig_synthesis.py` | ✅ |
| `fig07_scatter_initmean` | Init-mean scatter (1 dot = 1 init) | `fig_synthesis.py` | ✅ |
| `fig08_scorecard_heatmap` | ACC scorecard variable × week | `fig_synthesis.py` | ✅ |
| `fig09_anomaly_timeseries` | India-mean anomaly vs lead, Spire vs ERA5 | `fig_synthesis.py` | ✅ (climo bug fixed) |
| `fig10_verification_dashboard` | Combined dashboard | `fig_synthesis.py` | ✅ |

## §3 — MJO  (see `MJO_AUDIT.md` for the full audit)
| Fig | Shows | Script | Status |
|-----|-------|--------|--------|
| `fig20_mjo_rmm_skill` | **RMM bivariate COR & RMSE vs lead** (headline MJO skill, ~16-day horizon) | `fig_mjo_rmm_skill.py` | ✅ NEW |
| `fig18_mjo_phase_diagram` | RMM phase-space trajectories, 2 events | `fig_mjo_phase_diagram.py` | ✅ FIXED (WH04 norm) |
| `fig19_z500_mjo_composite` | Z500 anomaly composite by MJO phase, Spire vs ERA5 | `fig_z500_mjo_composite.py` | ✅ (small-N caveat) |
| `fig19b_z500_mjo_pattern_corr` | Pattern-correlation bar chart | `fig_z500_mjo_composite.py` | ✅ |
| `mjo/fig11_mjo_climatology_maps` | Mean OLR/U850/U200 forecast fields | `fig_mjo_extended.py` | 🔬 |
| `mjo/fig12_mjo_spread_vs_lead` | Init-to-init spread vs lead | `fig_mjo_extended.py` | 🔬 (relabel "spread") |
| `mjo/fig13_mjo_variance_maps` | Spatial variance maps | `fig_mjo_extended.py` | 🔬 |
| `mjo/fig14_hovmoller_olr` | Equatorial OLR Hovmöller | `fig_mjo_extended.py` | 🔬 |
| `mjo/fig15_hovmoller_u850` | Equatorial U850 Hovmöller | `fig_mjo_extended.py` | 🔬 |
| `mjo/fig16_mjo_snr` | Signal-to-noise vs lead | `fig_mjo_extended.py` | 🔬 |
| `mjo/fig17_extended_scorecard` | (a) ACC vs ERA5 + (b) internal-consistency proxy | `fig_mjo_extended.py` | ✅ FIXED (split) |

## §4 — Probabilistic & baselines  (ensemble verification + references)
| Fig | Shows | Script | Status |
|-----|-------|--------|--------|
| `fig25_rpss_vs_lead` | Tercile RPSS vs lead (RPSS>0 ⇒ beats climatology) | `15_probabilistic_verification.py` | ✅ skill to ~W2 |
| `fig26_reliability_tercile` | Reliability diagram, upper tercile | `15_probabilistic_verification.py` | ✅ |
| `fig27_baseline_acc` | Spire vs **persistence** ACC vs lead | `16_baselines.py` | ✅ Spire wins all leads |
| `fig28_baseline_rmse` | Spire vs persistence vs **climatology** RMSE | `16_baselines.py` | ✅ Spire lowest |

**Headline numbers (India, JFM 2026):**
- Deterministic: Spire ACC 0.82→0.1 (W1→W6), crosses 0.5 near W2–W3; **beats
  persistence at every lead**; RMSE below climatology at every lead.
- Probabilistic: tercile RPSS +0.47 (W1), +0.12 (W2), <0 from W3 — **useful
  probabilistic skill to ~2 weeks**, overconfident beyond (honest caveat).

## §5 — Benchmark & in progress
- **ECMWF S2S benchmark** (`download_ecmwf_s2s.py`, `ECMWF_BENCHMARK_SETUP.md`) — ready;
  needs user's ECMWF API key, then `verify_ecmwf_s2s.py` → fig29+.
- **Domain/variable expansion** — global/tropics skill, winds & MSLP (queued).
- **IMD verification** — 🔴 blocked: IMD 2026 gridded data not published yet (upstream lag).

---

## Data files
| File | Built by | Contents |
|------|----------|----------|
| `data/weekly_anomalies_v2.nc` | `09_compute_anomalies_v2.py` | Weekly anomalies (T2m mean/max, precip, Z500), consistent baselines |
| `s2s_verification/weekly_absolute_v2.nc` | `14_basic_verification.py` | Absolute weekly-mean T2m (Spire & ERA5) |
| `data/equatorial_hovmoller.nc`, `data/eofs_MJO.nc`, `data/rmm.74toRealtime.txt` | `extract_mjo_extended.py` / NCAR / BoM | MJO inputs |
| `s2s_verification/era5_tmean_climo_india.nc`, `era5_tmax_climo_india.nc` | `09` | ERA5 DOY climatologies |
| `s2s_verification/era5_30yr_daily_india.nc` | `15` | ERA5 30-yr daily (for tercile thresholds) |

## Reproduce
```bash
cd spire_era5/s2s_verification
conda run -n s2s-hind python -u 09_compute_anomalies_v2.py   # data (slow, cached)
conda run -n s2s-hind python -u 14_basic_verification.py     # fig21-24
conda run -n s2s-hind python -u 15_probabilistic_verification.py  # fig25-26 (slow, cached)
cd ../../spire-s2s-paper
for s in fig_deterministic fig_synthesis fig_mjo_extended fig_mjo_phase_diagram \
         fig_mjo_rmm_skill fig_z500_mjo_composite; do
  conda run -n s2s-hind python -u scripts/$s.py
done
```

See also: `MJO_AUDIT.md`, and `../spire_era5/s2s_verification/{SUMMARY,REVIEWER_CAVEATS}.md`.
