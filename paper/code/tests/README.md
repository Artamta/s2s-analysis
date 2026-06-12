# Test suite

Rigorous checks on the verification code, the precomputed skill tables, the
headline scientific results, and the manuscript figures.

```bash
# from the repository root
python -m pytest paper/code/tests -v
# or
bash paper/code/tests/run_tests.sh
```

Requires `pytest` (plus the runtime deps: numpy, pandas, xarray, scipy,
`global-land-mask`). Tests that need a precomputed table **skip** (rather than
fail) if that table is absent, so the suite is safe to run on a fresh checkout;
run the pipeline first (`bash paper/code/run_all.sh`) for full coverage.

| File | What it verifies |
|------|------------------|
| `test_compile.py`            | every compute/figure script parses; expected scripts present |
| `test_utils_metrics.py`      | known-answer maths for ACC/RMSE/bias, cosine weights, bootstrap CIs |
| `test_land_mask.py`          | India land fraction ~0.6; known land/ocean points; ocean→NaN |
| `test_data_integrity.py`     | skill tables: schema, 13 inits × 6 weeks × 4 systems × 5 regions, value ranges; ERA5 references sane; scatter points are anomalies |
| `test_results_invariants.py` | **headline results**: SPIRE best on TP/T2M and top-2 on Z500 (wk 1); FuXi precip unit-harmonised; SPIRE better calibrated than the overconfident FuXi |
| `test_figures_manifest.py`   | every figure the `.tex` includes exists in `paper/figs`; Taylor (removed) is not cited |
