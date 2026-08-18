# Captions: frozen FuXi–IMD anchored neural adapter

Global scope for all figures: development generalization audit, not an untouched independent test. The frozen model was selected on 2018–2019 and evaluated here on 100 operational JJAS starts (35/35/30 in 2022/2023/2024), with 171 IMD-supported cells. W1 is initialization day through +6; W6 is +35 through +41. ACC uses one common fixed training-only 2002–2017 IMD climatology.

## Figure 1 — Lead-wise skill and correction anatomy

Across the six leads, the anchored neural adapter has lower mean case-wise RMSE and higher common-reference spatial ACC than raw FuXi. Pooled W1–W6 RMSE is 5.275 versus 5.723 mm day⁻¹ (7.82% lower; descriptive paired block interval [6.42, 9.20]%). ACC is 0.358 versus 0.276 (Δ +0.082; 95% interval [+0.054, +0.109]). Most RMSE reduction is supplied by the training-only log-bias anchor (5.365 mm day⁻¹); the neural residual provides the smaller increment to 5.275. Signed bias becomes more negative (-0.225 to -0.842 mm day⁻¹).

## Figure 2 — Year and region robustness

Raw-to-adapter RMSE and ACC point improvements occur in all three audit years and all four reported regional masks. The smaller neural increment over log-bias is heterogeneous. Year rows use paired within-year circular block-13 percentile intervals. Regional rows use a paired two-stage bootstrap that first resamples the three audit years and then samples circular 13-start blocks within the selected years (2,000 draws). Intervals are descriptive, not p-values or population-level significance. Regional masks overlap and must not be summed.

## Figure 3 — Paired case gains and failures

The adapter improves RMSE in 480/600 (80.0%) initialization–lead cases and ACC in 439/600 (73.2%). Absolute bias improves in only 241/600 (40.2%). The 600 points are serially dependent and are displayed descriptively rather than as independent replicates.

## Figure 4 — Native-grid spatial footprint

Pooled local RMSE is lower over 73.8% of static weighted area versus raw FuXi and 80.7% versus log-bias. Maps show native 1.5° cells without interpolation. Area fractions use the frozen static cell-area × IMD-support weights and common finite support; the prediction cube did not save the audit scorer's fractional weekly-coverage field. Across the 600 case-leads, the static weight sum is 0.0049% higher than the saved audit effective-area sum, with differences in 8/600 case-leads. Local fields and area fractions are descriptive; no pixel-wise significance or multiplicity-adjusted inference is claimed.

## Figure 5 — Intensity and extremes stress test

The adapter improves ETS through the 10 mm day⁻¹ weekly-mean threshold, but at 20 mm day⁻¹ its ETS is 0.167 versus 0.169 for raw FuXi. For verifying weekly-mean rainfall ≥20 mm day⁻¹, adapter RMSE is 17.554 versus 17.593 mm day⁻¹, while adapter MAE is 14.884 versus 14.642 and signed bias is -14.388 mm day⁻¹. Threshold and stratum metrics use the same static cell-area × IMD-support weights noted for Figure 4, not the unavailable fractional weekly-coverage field. Aggregate error reduction therefore does not establish calibrated wet-tail prediction. This is a post-hoc weekly-mean cell–lead diagnostic, not a daily extreme-event analysis or a selection result.
