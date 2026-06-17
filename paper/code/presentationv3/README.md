# S2S Hindcast Verification V3 — Dual Climatology Basis

**Season:** JFM 2026 | **Init dates:** 13 (Jan 1 – Mar 26, weekly)  
**Models:** SPIRE · FuXi · ECMWF · MME | **Truth:** ERA5 1.5°  
**Variables:** TP (mm/day), Z500 (gpm) | **Leads:** W1–W6 (days 1–42)  
**SLURM job:** 55272 | **Completed:** Wed 17 Jun 2026 13:51 IST

---

## What's New in V3 — Dual Climatology Basis

Every metric is scored **twice**:

| `clim_basis` | Anomaly baseline for forecast | Models included |
|:---|:---|:---|
| `era5` | ERA5 30-yr WMO DOY climatology (same for all) | SPIRE, FuXi, ECMWF, MME, Persistence |
| `model_own` | Each model's own lead-dependent hindcast climatology | FuXi, ECMWF, MME only |

- **`era5` basis** → fair inter-model comparison (same baseline for all)
- **`model_own` basis** → removes each model's systematic climatological bias; tests if skill comes from true anomaly prediction vs matching climatological patterns

> **SPIRE has no multi-year hindcast archive**, so it is only scored under `era5` basis.

---

## All India — ERA5 Basis

### Pattern Correlation (PCC) — TP

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPIRE** | **0.81** | **0.68** | **0.51** | **0.41** | **0.39** | **0.34** |
| FuXi | 0.76 | 0.65 | 0.45 | 0.39 | 0.37 | 0.33 |
| ECMWF | 0.79 | 0.61 | 0.47 | 0.37 | 0.35 | 0.29 |
| MME | 0.80 | 0.66 | 0.49 | 0.40 | 0.37 | 0.33 |
| Persistence | 0.44 | 0.42 | 0.32 | 0.14 | 0.29 | 0.30 |

### Pattern Correlation (PCC) — Z500

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ECMWF | **0.96** | 0.85 | **0.72** | 0.45 | 0.27 | 0.10 |
| **SPIRE** | 0.93 | **0.84** | 0.71 | **0.53** | **0.44** | **0.43** |
| MME | 0.94 | 0.84 | 0.55 | 0.47 | 0.42 | 0.18 |
| FuXi | 0.82 | 0.54 | 0.03 | 0.07 | −0.27 | −0.16 |
| Persistence | 0.45 | 0.42 | 0.38 | 0.08 | 0.03 | 0.27 |

> FuXi Z500 collapses after W2 — large cold drift (−16 to −29 gpm by W3–W4).

### MSSS vs ERA5 Climatology — TP

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPIRE** | **+0.63** | **+0.44** | **+0.23** | **+0.08** | +0.03 | −0.05 |
| MME | +0.61 | +0.33 | +0.14 | −0.03 | −0.12 | −0.18 |
| ECMWF | +0.54 | +0.17 | 0.00 | −0.19 | −0.29 | −0.43 |
| FuXi | +0.47 | +0.05 | −0.14 | −0.22 | −0.25 | −0.27 |

### MSSS vs ERA5 Climatology — Z500

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ECMWF | **+0.91** | **+0.67** | **+0.19** | −0.29 | −0.45 | −0.26 |
| MME | +0.91 | +0.66 | −0.04 | −0.20 | −0.17 | −0.09 |
| SPIRE | +0.65 | +0.27 | +0.15 | −0.02 | −0.06 | −0.15 |
| FuXi | +0.83 | −0.04 | −2.77 | −1.85 | −1.42 | −0.64 |

---

## All India — Model-Own Basis (FuXi / ECMWF only)

Each model scored against its own lead-dependent hindcast climatology.

### TP PCC — model_own vs era5

| Model | W1 (era5→own) | W2 | W3 | W4 |
|:---|:---:|:---:|:---:|:---:|
| FuXi | 0.76 → **0.69** | 0.65 → 0.61 | 0.45 → 0.39 | 0.39 → 0.23 |
| ECMWF | 0.79 → **0.73** | 0.61 → 0.60 | 0.47 → 0.29 | 0.37 → 0.18 |
| MME | 0.80 → **0.73** | 0.66 → 0.62 | 0.49 → 0.40 | 0.40 → 0.21 |

### Z500 PCC — model_own vs era5

| Model | W1 (era5→own) | W2 | W3 | W4 |
|:---|:---:|:---:|:---:|:---:|
| FuXi | 0.82 → **0.73** | 0.54 → 0.54 | 0.03 → 0.05 | 0.07 → 0.01 |
| ECMWF | 0.96 → **0.87** | 0.85 → 0.84 | 0.72 → 0.71 | 0.45 → 0.55 |
| MME | 0.94 → **0.81** | 0.84 → 0.74 | 0.55 → 0.35 | 0.47 → 0.29 |

### Interpretation

- **ECMWF Z500 is robust** — PCC barely drops (0.96→0.87) when using its own climatology. Its skill is genuine anomaly prediction.
- **FuXi TP drops ~0.07 at W1** when model-own clim used — some skill was from matching ERA5 climatological patterns rather than true anomaly forecasting.
- **ECMWF TP** shows a larger drop (0.79→0.73 at W1, 0.47→0.29 at W3) — its W3 skill is mostly climatological pattern matching.
- **Model-own MSSS = era5 MSSS** for FuXi/ECMWF TP (the bias cancels in anomaly space; RMSE doesn't change significantly). The difference shows up in PCC at longer leads where climatological drift matters.

---

## Probabilistic — All India (ERA5 Basis)

### CRPSS vs Climatology — TP

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPIRE** | **+0.58** | +0.36 | +0.14 | +0.05 | +0.03 | −0.01 |
| ECMWF | +0.57 | **+0.39** | **+0.22** | **+0.11** | **+0.07** | **+0.03** |
| FuXi | +0.42 | +0.27 | +0.14 | +0.15 | +0.12 | +0.09 |

### CRPSS vs Climatology — Z500

| Model | W1 | W2 | W3 | W4 |
|:---|:---:|:---:|:---:|:---:|
| ECMWF | **+0.73** | **+0.43** | **+0.13** | −0.06 |
| SPIRE | +0.44 | +0.23 | +0.10 | 0.00 |
| FuXi | +0.61 | −0.04 | −0.81 | −0.51 |

### Spread-Skill Ratio (SSR) — TP  *(ideal = 1.0)*

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SPIRE | 0.71 | 1.28 | 1.39 | 1.36 | 1.29 | 1.22 |
| ECMWF | 0.36 | 0.65 | 0.71 | 0.69 | 0.66 | 0.58 |
| FuXi | 0.12 | 0.30 | 0.42 | 0.43 | 0.47 | 0.42 |

> SPIRE: under-dispersive at W1, over-dispersive W2+. ECMWF and FuXi: consistently under-dispersive (overconfident). 

---

## Brier Skill Score — All India

### BSS — TP above-normal tercile

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ECMWF | **+0.44** | **+0.21** | −0.04 | −0.15 | −0.14 | −0.16 |
| SPIRE | +0.34 | +0.07 | −0.33 | −0.48 | −0.52 | −0.60 |
| FuXi | 0.00 | 0.00 | −0.29 | −0.06 | −0.02 | −0.11 |

### BSS — TP > 10 mm/day (extreme precip)

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPIRE** | **+0.66** | **+0.50** | **+0.33** | **+0.23** | **+0.18** | **+0.12** |
| FuXi | +0.63 | +0.25 | +0.23 | +0.11 | +0.07 | +0.06 |
| ECMWF | +0.57 | +0.40 | +0.25 | +0.15 | +0.02 | −0.07 |

> All models skillful for heavy rain (>10 mm/day) through **Week 4**. SPIRE leads across all weeks.

### BSS — Z500 above-normal

| Model | W1 | W2 | W3 |
|:---|:---:|:---:|:---:|
| ECMWF | **+0.72** | **+0.35** | −0.02 |
| FuXi | +0.54 | −0.08 | −0.55 |
| SPIRE | +0.21 | −0.04 | −0.24 |

---

## Systematic Biases — All India, W1

| Variable | SPIRE | FuXi | ECMWF |
|:---|:---:|:---:|:---:|
| TP (mm/day) | **~0.00** | −0.12 | −0.08 |
| Z500 (gpm) | **+12.0** | −2.4 | +4.0 |

- **SPIRE Z500** carries a persistent +12 gpm warm bias but its spatial pattern skill remains high (PCC 0.93 W1). Bias grows smaller at longer leads.
- **FuXi Z500** starts near-zero but drifts strongly cold (−29 gpm by W3), destroying its MSSS/CRPSS.
- **TP bias** is near-zero for all at short leads; FuXi/ECMWF go dry by W4–W6 (−0.3 to −0.5 mm/day).

---

## Summary Takeaways

| Finding | Detail |
|:---|:---|
| **SPIRE best TP pattern skill** | Leads PCC W1–W6; MSSS positive through W3 |
| **ECMWF best Z500 skill** | Highest PCC+MSSS+CRPSS for Z500; most reliable |
| **FuXi Z500 fails after W2** | Cold drift −29 gpm by W3; MSSS = −2.77 |
| **Skill horizon TP** | ~W3 deterministic, ~W4 probabilistic (CRPSS>0) |
| **Skill horizon Z500** | ~W2–W3 for most; ECMWF holds to W3 |
| **Heavy rain BSS** | All skillful to W4; SPIRE leads (+0.66 → +0.12) |
| **Model-own basis** | Reduces apparent skill ~0.05–0.09 PCC at W1; larger at W3+ |
| **ECMWF most genuine** | Smallest drop era5→own basis; skill is real anomaly prediction |
| **SPIRE spread** | Over-dispersive from W2 (SSR>1); calibration needed |
| **ECMWF/FuXi spread** | Under-dispersive throughout (SSR 0.3–0.7); too confident |

---

## Files

| File | Description |
|:---|:---|
| `skill_deterministic.csv` | PCC, RMSE, Bias, MSSS — columns include `clim_basis` |
| `skill_probabilistic.csv` | CRPSS, CRPS, SSR — scored under `era5` basis |
| `skill_brier.csv` | BSS for 6 events (above/below normal, >1mm, >10mm) |
| `reliability.npz` | Reliability diagram data (attribute + resolution) |
| `verify_s2s.py` | V3 driver — dual-pass scoring loop |
| `loaders.py` | SPIRE zarr (`mean_stddev` group), FuXi netCDF, ECMWF GRIB |
| `plots_scr/plots_v3.py` | 14-figure plotting script (`--basis era5|model_own|both`) |
| `run_slurm.sh` | SLURM submission (GPU-AI_prio, 64 CPUs, 13 workers) |
| `verify_55272.out` | Full run log + headline tables |

---

*Run: SLURM job 55272, gpu1, ~14 min wall-time | June 2026*
