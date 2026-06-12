# MJO Figures — Correctness Audit (2026-06-08)

Audit of the MJO figure set in `spire-s2s-paper/`. ✅ correct, ⚠️ caveat, 🔴 was wrong.

| Fig | Script | Verdict | Notes |
|---|---|---|---|
| **fig18** phase diagram | `fig_mjo_phase_diagram.py` | 🔴→✅ **FIXED** | Was non-WH04: per-gridpoint normalisation + division by the forecast's own std → forecast RMM amplitude inflated ~2× and phase sometimes wrong (day-1 ≠ analysis). **Fixed:** WH04 scalar field norms (15.1/1.81/4.81) + day-1 calibration to observed RMM. Day-1 now corr 0.93/0.94 with the analysis; amplitudes match observed scale. |
| **fig20** RMM skill (NEW) | `fig_mjo_rmm_skill.py` | ✅ (with caveat) | **The canonical MJO skill metric** — bivariate COR & RMSE vs lead. COR=0.83 at d1, crosses 0.5 at **~16 days** (useful MJO horizon), min 0.41 at d21. ⚠️ The long-lead COR rebound (>0.5 after ~d35) is **low-frequency/seasonal contamination**, not skill — report only the first crossing. |
| **fig11** climatology maps | `fig_mjo_extended.py` | ✅ descriptive | 90-init-mean OLR/U850/U200 forecast fields. Not a verification — label as "mean forecast fields". |
| **fig12** spread vs lead | `fig_mjo_extended.py` | ⚠️ label | "Spread" here = init-to-init std of the ensemble mean, NOT within-forecast ensemble spread. Rename to avoid confusion. |
| **fig13** variance maps | `fig_mjo_extended.py` | ⚠️ label | Same as fig12 (spatial). Descriptive only. |
| **fig14/15** Hovmöller OLR/U850 | `fig_mjo_extended.py` | ✅ | Legitimate equatorial Hovmöller (composite + active/quiet init). Descriptive (not vs ERA5). |
| **fig16** SNR vs lead | `fig_mjo_extended.py` | ✅ diagnostic | Signal/noise proxy; fine if labelled as such. |
| **fig17** extended scorecard | `fig_mjo_extended.py` | ⚠️ **misleading** | T2m/Precip/Z500 = real ACC vs ERA5 ✅. But OLR/U850/U200 rows use **init-pair correlation** (forecast self-similarity), which is **NOT skill**. They sit in the same heatmap (asterisked). Either compute real ACC vs ERA5 OLR/winds, or split into a separate clearly-labelled panel. |
| **fig19 / 19b** Z500 MJO composite | `fig_z500_mjo_composite.py` | ✅ (small-N caveat) | Proper Spire-vs-ERA5 Z500 composites by observed MJO phase at init + pattern correlation. Uses correct Z500 data. ⚠️ Few inits per phase — report n and treat noisy phases cautiously. |

## Big-picture gap
The MJO section is mostly **descriptive forecast diagnostics**; the only true
verification-vs-observation pieces are **fig18** (fixed), **fig19** (Z500 composite),
and the **new fig20** (RMM skill). For the paper, fig20 is the headline MJO result
(Spire predicts the MJO with COR>0.5 to ~16 days). fig17's OLR/U850/U200 "skill"
should be fixed or relabelled.

## Recommended MJO figure order for the paper
1. **fig20** — RMM bivariate COR & RMSE vs lead (headline skill).
2. **fig18** — RMM phase-space trajectories for 2 events (illustrative).
3. **fig19** — Z500 teleconnection composite by MJO phase (impact over India).
4. fig14/15 Hovmöller + fig11 climatology as supporting/descriptive.
