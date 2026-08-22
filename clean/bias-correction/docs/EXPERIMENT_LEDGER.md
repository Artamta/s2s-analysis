# FuXi–IMD experiment ledger

Updated: 22 August 2026

This is the authoritative human-readable index for model status and evidence
level. A low metric in a run directory does not override the promotion
decision recorded here and in that run's selection manifest.

## Current decision

The frozen raw-identity TP/T2M adapter is now the strongest deterministic
neural candidate. It was selected on 2018--2019 without a fitted log-bias
reconstruction anchor, improved on the legacy anchored adapter in the fixed
2022--2024 audit, and retained favorable error and ACC effects against a
separate 2024 rain-gauge target. This is still development evidence rather
than an untouched temporal final.

The raw-mean-preserving diagnostic is not the primary method. It repaired the
large dry-bias trade-off under IMD verification and slightly improved RMSE,
but worsened MAE there and degraded RMSE, MAE, ACC, and absolute bias against
the station target. This target dependence is scientifically useful, but it
rules out describing the constraint as a universal calibration fix.

No later loss, target-transform, affine-calibration, or global-initialization
candidate passed all predeclared guards. The matched global-pretraining
experiment completed and selected scratch: global initialization was worse on
pooled RMSE, MAE, signed bias, and ACC, and zero of three pretrained seeds
achieved a lower best India validation composite loss than matched scratch.
This negative result does not change the completed E2/E3 method sets.

The 2025 final initialization year remains sealed. Its raw-identity-versus-raw
selection is frozen and its storage-incapable synthetic CUDA preflight passed;
no access ledger or final result exists.

### Parallel all-season probabilistic branch

The member-preserving location-and-spread adapter is the strongest current
probabilistic development result. It uses the whole-world FuXi archive as its
forecast source but trains and scores only the 39N--0N, 60E--99E India box
against IMD. On 208 reused 2020--2021 starts it improves raw-FuXi CRPS by
16.37% (95% initialization-block interval 14.28% to 18.16%), with positive
CRPS intervals at all six leads. Its central contribution is uncertainty
calibration: RMS spread / pooled RMS error changes from 0.562 to 0.967.

This branch is not independent confirmation. Pooled signed bias does not
improve, weeks 1--2 remain underdispersed, and the summary-only neural
ablation nearly ties the set encoder. The paper claim should therefore be a
lightweight distributional calibration result, not proof that member-set
features are essential and not a universal bias-correction result.

## Evidence inventory

| Experiment | Canonical artifact | Evidence role | Status |
|---|---|---|---|
| All-season 51-member location-and-spread calibration | `../resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z` | 2002--2017 training, 2018--2019 selection, reused 2020--2021 development evaluation | Complete promising probabilistic development result; 224 artifacts verified; not independent confirmation |
| Full-context compact TP/T2M adapter | `../results/fuxi_imd_full_context_compact_allweeks/full_20260811T152024Z` | 2018–2019 selection | Frozen current control |
| Compact physical-variable bank | `../results/fuxi_imd_compact_validation_sweep/physical_confirm3_a100_20260811T233520Z` | 2018–2019 validation, three seeds | Qualified, not independently confirmed |
| Physical-variable locked hindcast | `../results/fuxi_imd_locked_hindcast_evaluation/physical_full_compact_exploratory_2020_2021_20260812T010224Z` | Reused 2020–2021 | Exploratory only |
| Frozen anchored-only source audit | `../../studies/fuxi_imd_adapter_benchmark_v1/results/full_context_jjas_2022_2024_job91439` | Prior frozen 2022–2024 development audit and source for the matched E2 evaluation | Legacy source evidence; superseded as the headline comparison by canonical E2 |
| Raw-identity/no-fitted-log-bias adapter | `../resultsv2/fuxi_imd_no_log_bias_ablation/full_20260822T010749Z` | 2018–2019 three-seed selection; reused 2020–2021 evaluation | Qualified and carried into the completed fixed audit |
| Canonical raw-identity matched audit | `../resultsv2/fuxi_imd_raw_identity_2022_2024_audit/canonical_circular_20260822T0225Z` | Fixed 2022--2024, 100 starts, no retraining | Complete canonical development audit; all 16 artifact hashes verified |
| Frozen station-target sensitivity | `../resultsv2/fuxi_imd_adapter_station_external_target/canonical_five_method_20260822T0230Z` | Fixed 2024 gauge-derived cell target, 30 starts × 6 leads | Complete frozen secondary external-target sensitivity; all 14 artifact hashes verified |
| Global patch pretraining versus scratch | `../results/fuxi_imd_global_pretraining_comparison/execution_full_parallel_20260822a/full_20260822T040953Z` | 2002–2015 global fit, 2016–2017 global validation; matched 2018–2019 India comparison | Complete negative component result; scratch selected, manifest SHA-256 `37b6dac32a9fc2be799651b8e8577cac247d144caec379d1a3fdfce0417c1f5d` |
| Global-comparison mean-preservation follow-up | `../results/fuxi_imd_global_pretraining_followups/full_parallel_20260822a` | Artifact-only post-hoc 2018–2019 diagnostic | Both scratch and pretrained projections rejected; scientifically ineligible for promotion, manifest SHA-256 `e4748243bf8a57b34dc4460af97ce57cbdcfd63d39daebe36e9458356b0510bc` |
| Bias-aware anchor/loss 2×2 | `../results/fuxi_imd_bias_aware_validation_sweep/full_20260813T135600Z` | Reused 2018–2019 validation | No candidate qualified; reference retained |
| Heavy-tail weighting | `../results/fuxi_imd_tail_weight_validation_sweep/screen_seed42_cn14_20260813T152500Z` | Seed-42 screen | Rejected |
| Intensity/regime losses | `../results/fuxi_imd_intensity_loss_validation_sweep/screen_seed42_cn13_20260814T_loss_v1` | Seed-42 screen | Rejected |
| Fixed Box–Cox targets | `../results/fuxi_imd_target_transform_validation_sweep/screen_seed42_cn14_20260814T_transform_v1` | Seed-42 screen | Exact log target retained |
| Learnable Box–Cox target | No result directory | Implemented only | Not run; no evidence |
| Train-only affine calibration | `../results/fuxi_imd_train_affine_calibration/full_cn14_20260813T155500Z` | Development diagnostic | Not promoted |
| Blocked-OOF affine calibration | `../archive/blocked_oof_failure_20260813.tar.gz` | Intended leakage-safe development | Failed entering fold 2; not evidence |
| Independent 2025 evaluation | `../resultsv2/raw_identity_independent_2025_sealed/selection.json` | Final untouched test; selection SHA-256 `cad4af2a7443ee57ccec29f45ce812fb08f7e78ab135e6fe6f4871245b4dd6b6` | Frozen and synthetic CUDA preflight passed; no access ledger/result; not opened |

The physical-variable model is not the default for 2025 because its nine
additional FuXi variables are absent from the current operational archive.
Do not fill missing variables with zeros.

## Best defensible result

Five frozen forecasts were evaluated on the exact same 100 JJAS starts from
2022--2024, with no retraining or retuning:

| Forecast | RMSE (mm/day) | MAE (mm/day) | Bias (mm/day) | ACC |
|---|---:|---:|---:|---:|
| Raw FuXi | 5.723 | 3.822 | -0.225 | 0.276 |
| Training-only log-bias anchor | 5.365 | 3.543 | -0.824 | 0.337 |
| Frozen anchored adapter | 5.275 | 3.478 | -0.842 | 0.358 |
| Frozen raw-identity adapter | 5.239 | **3.453** | -0.858 | **0.364** |
| Raw-mean-preserving raw identity | **5.227** | 3.538 | **-0.216** | 0.360 |

Relative to raw FuXi, raw identity reduces RMSE by 8.45%, reduces MAE by
9.65%, and increases ACC by 0.0889. The paired year-stratified circular-block
interval for its RMSE improvement is 0.383 to 0.587 mm/day. Relative to the
legacy anchored adapter, it improves RMSE by 0.036 mm/day (interval 0.014 to
0.059) and MAE by 0.024 mm/day (interval 0.008 to 0.041).

The raw-identity bias claim is negative: its pooled signed bias changes from
-0.225 to -0.858 mm/day. Raw-mean preservation restores bias to -0.216 and
has the best IMD RMSE, but its MAE is 2.46% worse than unprojected raw identity.
Describe the model as an error- and spatial-skill-improving post-processor,
not as a uniformly successful bias correction.

The independent-target sensitivity strengthens and sharpens that conclusion.
Across 180 paired 2024 station-verification cases, raw identity improves raw
FuXi RMSE by 0.449 mm/day (95% circular-block interval 0.389 to 0.507), MAE by
0.391 mm/day, and ACC by 0.0276. Its pooled station RMSE/MAE/ACC are
7.493/4.893/0.431 versus 7.942/5.284/0.403 for raw FuXi. The station target is
wetter-biased for raw FuXi rather than dry-biased; consequently raw-mean
preservation degrades RMSE to 7.825 and is rejected as the main method.
Raw identity versus raw was a frozen secondary E3 comparison; the sole
predeclared E3 primary estimand was the legacy selected adapter versus raw.
The raw-identity station result is therefore consistent external-target
sensitivity, not a second confirmatory test.

Rainfall-intensity results bound the claim further. Raw identity improves
pooled dry and moderate errors, but for >=20 mm/day rainfall its RMSE
improvement interval crosses zero, its MAE point effect is unfavorable, and
its signed dry bias remains large. The current evidence does not establish an
extreme-rainfall improvement.

## Later-screen decisions

- `physical_full_compact` passed the declared validation guards, changing
  matched-control RMSE from 5.4563 to 5.4358 and ACC from 0.3271 to 0.3302.
  The gain is small and lacks untouched confirmation.
- The bias-aware/recentered candidate improved pooled RMSE and signed bias but
  worsened MAE and did not beat raw FuXi absolute bias.
- Heavy-rain weighting reduced tail errors at the cost of much larger
  dry/light errors, pooled MAE, and ACC.
- Equal-regime and wet-occurrence objectives moved gradients in the requested
  direction but worsened heavy-rain and pooled skill.
- Fixed Box–Cox power 0.25 was a weak Pareto direction, not a promoted model.
  Its apparent pooled bias gain partly came from cancellation between
  low-intensity overprediction and heavy-rain underprediction.
- Train-only affine recalibration failed its promotion rules.
- The full blocked-OOF run stopped before fold 2 training with
  `weekly IMD truth changed between folds`. The one-fold smoke result
  was also not promoted.
- The raw-identity ablation used raw FuXi, rather than fitted log-bias, as the
  neural reconstruction anchor. Its frozen three-seed `normal_climo_model`
  improved raw-FuXi RMSE by 10.67% in 2018 and 6.22% in 2019. On the reused
  2020--2021 cohort it reached RMSE 5.1503, MAE 3.3452, bias -0.7526 mm/day,
  and ACC 0.3634. It slightly exceeded the anchored adapter on the same cases
  but retained a large dry bias; these are exploratory development results,
  not independent confirmation.
- Global patch pretraining failed its frozen matched gate. On 2018--2019 the
  scratch ensemble had RMSE/MAE/bias/ACC
  5.4809/3.5466/-1.2120/0.3207, while the pretrained ensemble had
  5.5080/3.5555/-1.2425/0.3119. Global RMSE was 0.493% worse, both validation
  years regressed, only three of six leads improved the composite score, and
  zero of three pretrained seeds achieved a lower best India validation
  composite loss than its matched scratch seed. This tests a
  log-bias-anchored transfer initialization; it does not prove that every
  possible global or raw-identity pretraining scheme must fail.
- The artifact-only regional-mean follow-up also failed: projection increased
  scratch RMSE by 0.0392 mm/day and pretrained RMSE by 0.0355, with zero
  improving leads, years, or seeds. It remains a post-hoc diagnostic.

Detailed loss and transform protocols are preserved under
`experiments/`.

## Publication route

1. Retain the completed global-pretraining-versus-scratch comparison as a
   negative component result; do not tune a replacement from 2018--2019.
2. Retain the completed no-retraining 2022--2024 matched audit and frozen
   station-target sensitivity as immutable evidence; do not retune from them.
3. Treat raw identity as the primary model and raw-mean preservation as a
   target-dependent Pareto diagnostic, not a promoted universal fix.
4. Keep the failed blocked-OOF affine branch out of the main evidence line.
   Repair it only if a later calibration study explicitly requires it.
5. Freeze the raw-identity-versus-raw 2025 hierarchy using
   `RAW_IDENTITY_2025_SEALED_WORKFLOW.md`. The older
   `INDEPENDENT_2025_CONTROL_WORKFLOW.md` is a superseded physical-control
   design and must not be used for the selected raw-identity model.
6. Perform the GPU preflight, then open 2025 exactly once after explicit user
   approval.
7. Report case-level paired effects with initialization-block uncertainty,
   all six leads, years/regions, rainfall regimes, and the negative bias result.

Repeatedly tuning against 2018–2019 is now a larger publication risk than
retaining the current model. A larger network is not the missing evidence.

## Separate AI-Quest result

`../../ai-quest-global` is a global probabilistic ERA5-category project
scored with RPS/RPSS, not this deterministic IMD experiment. Its current
validation result is also not publication-ready: the calibrated neural model
improves the calibrated Debias++ control by only about 0.33%, three-seed
confirmation is incomplete, and its untouched 2021 test is unopened.

Use its provenance and test-access patterns, not its targets, metrics, or
claims.
