# CCAI neural-adapter figure package

This directory contains exactly five conference-ready PNG/PDF figure pairs for the frozen FuXi-to-IMD anchored neural postprocessor.

## Scope

- Development generalization audit; not an untouched independent test.
- Train: 2002–2017; selection: 2018–2019; plotted audit: 100 JJAS starts in 2022–2024.
- No 2025 initialization is used here.
- W1 = init+0..6 through W6 = init+35..41.
- 171 native 1.5° IMD-supported cells; IMD is the verification reference.
- Keep separate from INDIA-S2S-BENCH, whose W1 begins at init+1.
- Figures 4–5 use saved static cell-area × IMD-support weights because fractional weekly coverage was not stored in the prediction cube. Their pooled weight is 0.0049% above the audit effective-area sum (8/600 case-leads differ).

## Main result

Pooled mean case-wise RMSE is 5.275 versus 5.723 mm day⁻¹ (7.82% lower), and common-reference ACC is 0.358 versus 0.276 (Δ +0.082). The adapter is best described as an RMSE/pattern-skill postprocessor: overall signed bias worsens from -0.225 to -0.842 mm day⁻¹, and the ≥20 mm day⁻¹ weekly-mean cell–lead diagnostic remains weak. It is not a daily extreme-event analysis.

## Figures

1. `01_lead_skill_and_correction_decomposition` — W1–W6 RMSE, ACC, paired RMSE reduction, and signed bias.
2. `02_year_region_robustness` — year and regional-mask effects versus raw FuXi and log-bias, with descriptive paired block intervals.
3. `03_paired_case_gains_and_failures` — dependent case-lead effect distributions and win fractions.
4. `04_native_grid_spatial_footprint` — pooled local RMSE-reduction maps and lead-wise improved-area fractions.
5. `05_intensity_and_extremes_stress_test` — weekly-mean threshold ETS/frequency bias and truth-stratified RMSE/bias; not daily extreme-event verification.

See `CAPTIONS.md` for conference-safe wording and `MANIFEST.json` for hashes, provenance, definitions, and status. Numerical values behind every plot are in `tables/`.

## Regenerate

From `clean/studies/fuxi_imd_adapter_benchmark_v1`:

```bash
conda run -n weather_forecast python make_ccai_neural_adapter_figures.py --output results/full_context_jjas_2022_2024_ccai_figures_v1
```

The default refuses to overwrite an existing output. Use `--overwrite` only when deliberately replacing this generated package.
