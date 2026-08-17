# Experiment register

This is the implementation queue for the paper. It is not a second manuscript
and it contains no placeholder scores. The frozen 2025 benchmark has already
been opened; every new analysis below must therefore be labelled
**retrospective/exploratory**, even when its parameters are fitted using only
pre-2025 data. Nothing here may be added to or overwrite
artifacts/confirmatory_2025/.

## What is already sufficient

- JJAS-initialized precipitation, Weeks 1--6, on 35 common 2025 dates.
- Seven individual systems, equal weighting, inverse-2024-RMSE weighting, and
  a per-lead 2024-RMSE-selected individual baseline.
- Forecast-only, full, and location/calendar-only PiggyCast variants.
- All-India and four IMD monsoon-region scores.
- ACC, RMSE, MAE, signed bias, wet-area error, paired moving-block intervals,
  block-length sensitivity, exact coverage, source hashes, and trained models.

These results support the current bounded paper without temperature,
multi-season evaluation, IMERG, or a neural adapter.

## P0: simple baselines before submission

### A. IMD climatology error baseline

Purpose: show the absolute-error reference that any forecast should beat.

Protocol:

1. Predict the stored weekly IMD climatology for every 2025 case.
2. Use the identical valid dates, grid, IMD support, and case aggregation.
3. Report RMSE, MAE, signed bias, and wet-area error.
4. Report ACC as an em dash/undefined. A zero-anomaly climatology has zero
   spatial variance, so assigning ACC = 0 would be mathematically wrong.
5. Save case-level and aggregate tables under
   artifacts/exploratory_baselines/; do not edit the frozen artifact tree.

This is an error reference, not a competing spatial-pattern forecast.

### B. Additive bias-corrected equal-weight mean

Purpose: test whether XGBoost adds anything beyond the simplest spatial
calibration.

Freeze one method only:

\[
  \hat y^{\,bc}_{w,g} =
  \hat y^{\,eq}_{w,g} -
  \operatorname{mean}_{t \in \mathrm{fit}}
  \left(\hat y^{\,eq}_{t,w,g} - y_{t,w,g}\right),
\]

where \(w\) is lead week and \(g\) is grid cell.

Protocol:

1. Implement the per-lead, per-grid additive correction using 2020--2023.
2. Use 2024 only to check the implementation and freeze the exact method.
3. Refit the unchanged mean-error field on 2020--2024.
4. Apply it once to the same 35 dates in 2025 with no clipping.
5. Evaluate the same five domains and all existing metrics.
6. Bootstrap paired differences against equal weighting and full PiggyCast
   with the same initialization blocks and seed family.
7. Record a protocol JSON, case metrics, aggregate metrics, intervals, source
   hashes, and an audit in artifacts/exploratory_baselines/.

Do not try affine, multiplicative, quantile, clipped, and several spatial
variants and report only the winner.

Paper decision:

- If full PiggyCast remains better in ACC without worse RMSE/bias, say it adds
  forecast-conditioned structure beyond additive calibration.
- If the bias baseline matches it, change the lesson to: much of the apparent
  gain is obtainable with simple calibration.
- If the bias baseline is better, lead with that negative result. It makes the
  benchmark more credible, not less.

## P1: IMD--IMERG observational-reference sensitivity

Purpose: determine whether conclusions depend on the rainfall reference.

Use frozen absolute predictions; do not retrain or reselect PiggyCast on
IMERG. For a clean comparison:

1. Freeze the same 35 initializations, six leads, 1.5-degree grid, and common
   finite IMD--IMERG support.
2. Build 2001--2019 normals for both IMD and IMERG so the reference datasets
   share a climatology period. Keep the headline IMD 1991--2019 result
   unchanged.
3. Recompute ACC, RMSE, MAE, and bias separately against IMD and IMERG.
4. Also report IMD--IMERG observation agreement by lead. This distinguishes
   forecast failure from observational disagreement.
5. Save the run under artifacts/reference_sensitivity/ with its own protocol,
   case table, intervals, hashes, and audit.
6. Use a four-panel appendix heatmap: IMD ACC, IMERG ACC, IMD RMSE, and IMERG
   RMSE, with methods as rows and Weeks 1--6 as columns.

Mention this in the main text only if the direction of the full-minus-equal
result is stable across references. If it changes, report that sensitivity
prominently rather than choosing the favorable reference.

## P2: optional persistence diagnostic

If time remains, repeat the observed IMD mean from initialization-7 through
initialization-1 as a future-week forecast. Evaluate it on the same dates and
support, using each future week's climatology for ACC. Label it diagnostic:
real-time availability of the gauge product may not match the experiment.
This is lower priority than additive bias correction and IMERG sensitivity.

## Defer from this short paper

### Temperature

Temperature creates a second scientific question and a different model cohort.
NeuralGCM has no standardized 2-m temperature product here, and the available
NCEP value is a min/max midpoint proxy rather than the same daily-mean
statistic. A later temperature benchmark should use ERA5 truth and a
pre-2020 ERA5 climatology, with the proxy separated as a sensitivity.

### Multiple seasons

The current split, climatology, interpretation, and coverage contract is
JJAS-specific. A multi-season study needs a new frozen protocol and complete
valid-day truth. Late-OND 2025 Week-6 windows extend into 2026, and the current
truth loader assumes valid dates remain in the initialization year.

### FuXi neural correction

The saved evaluator uses +0...+6 through +35...+41 day windows; this benchmark
uses +1...+7 through +36...+42. Inclusion requires regenerated labels,
pre-2025 selection, retraining, and inference on the identical 35 dates. It is
a complementary alternative, never a stage after PiggyCast.

## Completion gate for every new experiment

- one written protocol before the run;
- identical valid dates, units, grid, support, and metric implementation;
- no 2025 tuning or method shopping;
- saved case-level metrics, not only aggregate plots;
- paired initialization-block uncertainty;
- source/data/artifact hashes and an independent aggregate audit;
- explicit exploratory label in artifact metadata and paper text.
