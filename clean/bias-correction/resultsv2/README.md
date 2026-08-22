# Version-2 experiment results

This directory is reserved for new immutable experiment outputs that follow
the post-cleanup provenance contract. Generated run directories are not edited
in place.

The no-fitted-log-bias neural ablation is stored under
`fuxi_imd_no_log_bias_ablation/`. Smoke runs prove execution only. Full runs
are validation-selected development evidence, and any 2020–2024 diagnostics
remain exploratory. No run here may access 2025 unless a separate one-time
final-test protocol explicitly authorizes it.

Canonical follow-up artifacts:

- `fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z`:
  complete 51-member, all-season India-box probabilistic calibration run using
  2002--2017 training, 2018--2019 checkpoint selection, and the reused
  2020--2021 development test. All 224 non-manifest artifacts were independently
  verified; manifest SHA-256
  `94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91`.
  Its manuscript-scale derived bundle is at
  `../presentation/deliverables/fuxi_allseason_ensemble_calibration_20260822`
  (manifest SHA-256
  `b1e72076a28f3abc2b7b8b89c3fdb7245968a7e960e87d4f4bcde019ccb9786c`).
- `fuxi_imd_raw_identity_2022_2024_audit/canonical_circular_20260822T0225Z`:
  fixed five-method 2022--2024 development audit, 10,000 paired circular-block
  draws, no retraining; manifest SHA-256
  `bc9fa96182906f736dc542000ed62f1ddd70460448f07e679943efcffceeeeec`.
- `fuxi_imd_adapter_station_external_target/canonical_five_method_20260822T0230Z`:
  frozen external 2024 station-target sensitivity over 30 starts and six
  leads; manifest SHA-256
  `5404867e63f0fd6d3b09799c32727f64c36a62489b5e1c27310b8ca33463d249`.
- `raw_identity_independent_2025_sealed/selection.json`: the frozen
  raw-identity-versus-raw final-test contract; selection SHA-256
  `cad4af2a7443ee57ccec29f45ce812fb08f7e78ab135e6fe6f4871245b4dd6b6`.
  Synthetic CUDA preflight job 109981 passed with receipt SHA-256
  `c2484d8d9e5a93c782e23b4419b5363e04e254a3f8e88e2d37dd37ba9f43db3b`.
  It created no storage access, access ledger, or result.

The first two are immutable development evidence. The third is only the
sealed contract and synthetic proof for the still-unopened 2025 final test.
