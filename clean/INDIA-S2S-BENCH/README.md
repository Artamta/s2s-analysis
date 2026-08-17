# INDIA-S2S-BENCH

Reproducible workspace for the workshop study **Lightweight Post-processing of
Multi-model Subseasonal Rainfall Forecasts over India**.

## Confirmatory outcome

The frozen 2025 run is complete and audited. Full PiggyCast improves mean
spatial ACC over equal weighting by 0.040 (95% moving-block interval 0.011 to
0.070) and lowers RMSE by 0.186 mm day-1. It does **not** clear the stronger
predeclared ACC gate against the validation-selected individual system because
that interval is -0.002 to 0.093. The manuscript therefore reports a bounded,
lead- and region-dependent gain rather than a universal headline improvement.

The compiled five-page draft is `paper/submission_draft.pdf`; its sources,
generated tables, and submission hashes are under `paper/`.

The confirmatory contract is frozen in `protocol.json`. The runner uses seven
forecast systems on the common 1.5 degree India grid, IMD rainfall as truth, a
single IMD 1991--2019 climatology for every anomaly, and a 2025 JJAS
initialization test set that is never used for fitting or selection.

## Commands

Use the environment that already contains the compatible Zarr 2 and XGBoost
runtime:

```bash
cd /home/raj.ayush/s2s/s2s_anlysis/clean/INDIA-S2S-BENCH
PYTHONPATH=src /home/raj.ayush/.conda/envs/weather_forecast/bin/python run.py preflight
PYTHONPATH=src /home/raj.ayush/.conda/envs/weather_forecast/bin/python run.py run
PYTHONPATH=src /home/raj.ayush/.conda/envs/weather_forecast/bin/python run.py audit
```

The first command is metadata/QC only and emits no forecast score. `run`
trains the frozen 2020--2024 post-processing methods and evaluates the 2025
test once. `audit` independently regenerates aggregate tables from saved
case-level metrics and checks every artifact hash.

Focused tests do not require forecast data:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/raj.ayush/.conda/envs/weather_forecast/bin/python -m pytest -q tests
```

Regenerate manuscript numbers only after a passing audit:

```bash
PYTHONPATH=src /home/raj.ayush/.conda/envs/weather_forecast/bin/python make_paper.py
```

## Output contract

`artifacts/confirmatory_2025/` contains:

- exact date coverage and source provenance;
- case-level India and regional metrics;
- aggregate scores and paired moving-block intervals;
- validation weights and selected model metadata;
- gate decisions controlling the manuscript claim;
- paper figures/tables and SHA256 hashes.

The runner refuses an existing output directory. Re-running the test therefore
requires an explicit new output path and produces a separately auditable run.

The neural FuXi correction is a complementary method, not a stage after the
multi-model mixer. Its existing frozen evaluator uses a different valid-day
alignment (+0...+6), so it is deliberately excluded from the common-date
headline result until retrained with the +1...+7 contract.
