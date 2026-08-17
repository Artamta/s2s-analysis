# Evidence ledger

This ledger separates verified repository evidence from proposed manuscript
claims. A paper statement may move to **supported** only when it cites a saved
artifact and its hash appears in the confirmatory manifest.

| ID | Evidence or proposed claim | Status | Repository evidence | Paper use |
|---|---|---|---|---|
| E01 | Seven systems have common 2020--2024 standardized precipitation forecasts on the 27 x 27, 1.5 degree India grid. | verified | `../studies/india_s2s_benchmark_v1/METHODS_AND_DATA.md`; archive catalogs | Methods |
| E02 | CMA, ECMWF, FuXi-S2S, NCEP, and UKMO have standardized 2025 stores. | verified | preflight coverage manifest | Methods/coverage |
| E03 | DLESyM v0 and NeuralGCM have 104 complete native 2025 forecasts and manifests but are not yet appended to the standardized catalog. | verified | `../model-runs/configs/dlesym_2025.json`; `../model-runs/configs/neuralgcm_production_tp_2025_ens10.json`; preflight | Coverage limitation |
| E04 | The two native 2025 extensions use the same 27 x 27 coordinates, 42 daily leads, and mm day-1 precipitation units as the benchmark. | verified by metadata/QC, not skill | preflight report | Adapter justification |
| E05 | IMD 1991--2019 climatology and exact fractional-area support are available as immutable Zarr stores. | verified | benchmark observation and spatial stores; preflight hashes | Methods |
| E06 | The previous PiggyCast result used 2020--2022 training, 2023 validation, and 2024 test, and used model-specific forecast climatologies. | verified; retrospective only | `/storage/raj.ayush/ashoka_storage/piggycast_s2s/runs/slurm_84636/manifest.json` | Related experiment/limitation |
| E07 | The previous common-sample PiggyCast comparison is not the confirmatory result because its split and anomaly contract differ from this protocol. | verified | `../neural_adapter/common_sample_multimodel_benchmark_manifest.json`; `protocol.json` | Limitation |
| E08 | The existing FuXi neural-control evaluator uses W1 init+0...+6, whereas the benchmark forecast periods end on init+1...+7. | verified | `../bias-correction/evaluate_independent_2025_control.py`; benchmark methods | Appendix exclusion |
| E09 | IMERG truth and a pre-2020 climatology exist on the common grid, but no audited common-support reference-sensitivity result is part of this paper. | data availability verified; result pending | standardized observation stores | Do not claim IMD--IMERG robustness |
| C01 | Full PiggyCast improves 2025 JJAS ACC over equal weighting. | **supported, bounded** | `artifacts/confirmatory_2025/paired_intervals.csv`: +0.040, 95% MBB interval [+0.011, +0.070] | State with interval and Weeks 5--6 limitation |
| C02 | Full PiggyCast improves ACC over the validation-selected individual system without material RMSE/bias failure. | **headline gate failed** | ACC +0.052, 95% MBB interval [-0.002, +0.093]; RMSE and bias guards pass | Do not claim robust superiority |
| C03 | Forecast-conditioned features add skill beyond location/calendar alone. | **supported** | forecast-only minus location/calendar ACC +0.238 [+0.209, +0.266] | Ablation result |
| C04 | Improvements are not confined to one IMD region or a few initializations. | **partially supported** | block-length sensitivity preserves equal-weight result; 3/4 regional deltas positive, east/northeast negative (-0.097) | Report heterogeneity, not universal gain |
| C05 | The neural correction improves FuXi on the identical 2025 cases. | not testable under current alignment | corrected +1...+7 retraining required | Do not claim; diagnostic appendix only |
| C06 | Full PiggyCast improves upon an additive bias-corrected equal-weight mean. | not tested; any new 2025 comparison is exploratory | `EXPERIMENT_REGISTER.md` | Do not claim until a separately audited run exists |
| C07 | The result is robust to IMERG as the observational reference. | not tested | `EXPERIMENT_REGISTER.md` | Do not claim |

## Interpretation rules

- Spatial ACC measures anomaly-pattern agreement and cannot establish rainfall
  amount calibration by itself.
- Every ACC statement must be accompanied by RMSE and signed bias.
- A confidence interval is described as a paired moving-block percentile
  interval, not as a p-value or a formal null-hypothesis test.
- Failed gates and negative results remain in the artifact bundle and appendix.
- Representative maps cannot be selected using the largest 2025 improvement;
  if maps are included, use a predeclared date or a climatological composite.

## Required citations to finalize before submission

The manuscript must cite the official papers or provider documentation for
FuXi-S2S, NeuralGCM, DLESyM, ECMWF/UKMO/NCEP/CMA S2S data, the IMD gridded
rainfall dataset, the S2S database, XGBoost, and the PiggyCast tutorial or paper.
Bibliographic metadata must be checked against primary sources before the PDF
is submitted. No novelty claim of “first ML correction of Indian monsoon S2S
rainfall” is permitted.

## Frozen-workflow outcome (2026-08-17)

The frozen workflow evaluated 35 common 2025 JJAS initializations without using
2025 data for fitting or selection. The
artifact audit independently regenerated aggregate metrics from 13 methods x
35 cases x 6 weeks x 5 spatial domains and verified all manifest hashes. The
overall headline gate is **closed** because the primary block-length interval
against the validation-selected individual model overlaps zero. The manuscript
therefore presents an auditable benchmark and a bounded positive result over
equal weighting, not a general claim that adaptive mixing is best. Because the
2025 scores are now known, all newly proposed baselines or reference
sensitivities are explicitly exploratory.
