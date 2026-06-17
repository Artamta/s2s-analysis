# S2S Hindcast Verification — JFM 2026 Summary

**Season:** Jan–Mar 2026 (JFM) | **Init dates:** 13 (weekly, Jan 1 – Mar 26)  
**Models:** SPIRE · FuXi · ECMWF · NCEP · MME  
**Baselines:** ERA5 30-yr WMO DOY climatology · Persistence  
**Variables:** TP (mm/day), Z500 (gpm), T2M (K)  
**Regions:** All India + 4 IMD homogeneous zones (NW, Central, South Peninsula, East/NE)  
**Truth:** ERA5 (1.5° interpolated)  
**Lead weeks:** W1=days 1–7 … W6=days 36–42

---

## Key Headline Numbers — All India, Weekly

### Pattern Correlation (PCC) — TP

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPIRE** | **0.84** | **0.69** | **0.56** | **0.43** | **0.41** | **0.36** |
| ECMWF | 0.81 | 0.64 | 0.50 | 0.37 | 0.37 | 0.31 |
| FuXi | 0.79 | 0.67 | 0.48 | 0.40 | 0.38 | 0.35 |
| MME | 0.82 | 0.67 | 0.51 | 0.40 | 0.40 | 0.35 |
| Persistence | 0.49 | 0.41 | 0.36 | 0.17 | 0.30 | 0.27 |

> Useful skill (PCC > 0.5) through **Week 3** for TP. SPIRE leads all weeks.

### Pattern Correlation (PCC) — Z500

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPIRE** | **0.94** | **0.87** | 0.75 | **0.59** | 0.47 | **0.44** |
| ECMWF | 0.94 | 0.84 | **0.77** | 0.50 | 0.32 | 0.15 |
| FuXi | 0.82 | 0.50 | 0.04 | 0.07 | −0.28 | −0.11 |
| MME | 0.94 | 0.84 | 0.68 | 0.53 | **0.49** | 0.39 |
| Persistence | 0.49 | 0.44 | 0.38 | 0.12 | −0.02 | 0.27 |

> Z500 skill extends to **Week 4** (SPIRE/ECMWF). FuXi collapses after W2 — large cold drift.

### Pattern Correlation (PCC) — T2M

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPIRE** | **0.74** | **0.55** | **0.41** | **0.31** | 0.24 | 0.23 |
| NCEP | 0.39 | 0.35 | 0.28 | 0.25 | **0.25** | **0.24** |
| FuXi | 0.36 | 0.29 | 0.17 | 0.09 | 0.06 | 0.08 |
| ECMWF | 0.35 | 0.30 | 0.19 | 0.09 | 0.04 | 0.03 |
| Persistence | 0.58 | 0.45 | 0.28 | 0.12 | −0.05 | 0.09 |

> SPIRE strongly leads T2M due to near-unbiased initialization. Others 3–5 K cold.

---

## Probabilistic Skill — All India

### CRPSS vs ERA5 Climatology (TP)

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SPIRE | +0.58 | +0.36 | +0.14 | +0.05 | +0.03 | −0.01 |
| ECMWF | +0.57 | +0.39 | +0.22 | +0.11 | +0.07 | +0.03 |
| FuXi | +0.42 | +0.27 | +0.14 | +0.15 | +0.12 | +0.09 |

> Probabilistic TP skill (CRPSS > 0) extends to **Week 4–5** for SPIRE/ECMWF.

### CRPSS vs ERA5 Climatology (Z500)

| Model | W1 | W2 | W3 | W4 |
|:---|:---:|:---:|:---:|:---:|
| ECMWF | +0.73 | +0.43 | +0.13 | −0.06 |
| SPIRE | +0.44 | +0.23 | +0.10 | 0.00 |
| FuXi | +0.61 | −0.04 | −0.81 | −0.51 |

---

## MSSS vs Climatology (All India, TP)

| Model | W1 | W2 | W3 | W4 |
|:---|:---:|:---:|:---:|:---:|
| SPIRE | +0.69 | +0.44 | +0.26 | +0.05 |
| ECMWF | +0.54 | +0.17 | −0.03 | −0.27 |
| FuXi | +0.45 | 0.00 | −0.20 | −0.27 |
| MME | +0.60 | +0.31 | +0.13 | −0.10 |

---

## Systematic Biases

| Variable | SPIRE | FuXi | ECMWF | NCEP |
|:---|:---:|:---:|:---:|:---:|
| TP (mm/day) | ~0.0 | ~−0.1 | ~−0.1 | ~−0.1 |
| Z500 (gpm) | **+12 to +14** | −2 → −28 (drifts cold W1→W6) | +5 → −3 (small) | −14 to −27 |
| T2M (K) | **~0 (near-unbiased)** | −4.2 | −4.6 | −3.5 |

> SPIRE Z500 has a persistent warm bias (~12 gpm) across all weeks but its **spatial pattern skill remains high** (PCC ~0.9 W1). FuXi/NCEP show large cold Z500 drift leading to negative MSSS by W3.

---

## Regional Highlights

| Region | Best TP model | Skill up to | Notes |
|:---|:---:|:---:|:---|
| All India | SPIRE | **W3** | Consistent SPIRE lead |
| NW India | SPIRE/ECMWF | **W3** | Dry season; high interannual variance |
| Central India | ECMWF/NCEP | **W5** | NCEP surprisingly competitive Wk3–5 |
| South Peninsula | None dominant | **W1** | No model > 0.5 PCC for TP — hardest region |
| East/NE India | SPIRE | **W4** | Highest TP values; all models dry-biased |

---

## Key Findings

1. **SPIRE leads TP skill** in nearly all regions and weeks, with ECMWF close behind.
2. **Z500 skill is strong through Week 4** (SPIRE/ECMWF). FuXi shows catastrophic Z500 drift after W2, likely a model-climatology mismatch in this study period.
3. **T2M: SPIRE dominates** due to near-zero initialization bias. FuXi/ECMWF/NCEP carry a persistent ~4–5 K cold bias throughout.
4. **South Peninsula** is uniformly the hardest region — no model achieves useful skill for TP (PCC > 0.5) in any week.
5. **All models beat climatology** for TP through Week 2–3. Beyond Week 3, only SPIRE (and occasionally MME) maintains positive MSSS.
6. **SPIRE ensemble spread** is over-dispersive (SSR > 1 from W2 onward for TP), indicating its probabilistic TP forecasts are wider than needed. ECMWF is better calibrated.
7. **MME** is competitive on RMSE and Z500 at short leads (reduces noise from individual model errors) but offers no gain on PCC.

---

## Files in This Directory

| File | Description |
|:---|:---|
| `skill_deterministic.csv` | PCC, RMSE, Bias, MSSS — all models, regions, leads, weekly+daily |
| `skill_probabilistic.csv` | CRPSS, CRPS, SSR — probabilistic metrics |
| `skill_brier.csv` | BSS — above/below normal tercile events |
| `s2s_skill_summary.md` | Full detailed metric tables (all regions × variables × models) |
| `verify_s2s.py` | Driver script for verification |
| `loaders.py` | Data loaders (SPIRE zarr, FuXi netCDF, ECMWF GRIB) |

---

*Generated: June 2026 | ERA5 truth | 13 init dates JFM 2026 | 1.5° India grid*
