# Presentation figure audit

Audit scope: the existing architecture diagram, three-seed training-curves
figure, and four `publication_validation` figures from the completed physical
predictor validation run. This was a visual and metadata audit only. No target,
observation, prediction, residual, or forecast array was opened.

Remediation status: both presentation-critical findings below were corrected
in v2 renderers after this audit. The packager prefers
`physical_temporal_unet_architecture_v2` and
`physical_full_compact_training_curves_v2` whenever their complete render
families are present. It falls back to the original assets only for backward
compatibility and records the applicable caveat in the slide index.

## Evidence labels that must remain explicit

- Training: 2002–2017.
- Architecture/checkpoint/variable selection: blocked validation, 2018–2019.
- 2020–2021: previously reused hindcast test. It can support only an
  exploratory locked evaluation, never independent confirmation.
- Genuine independent 2025 physical-model evaluation: unavailable.

## Presentation-critical issues

1. The original architecture diagram labels 2020–2021 simply as `TEST` and says it was
   quarantined until final evaluation. That describes the current selection
   run but can falsely imply that the period is still a pristine independent
   test for the overall project. Relabel it verbally or in the slide caption as
   `exploratory/reused test; not independent confirmation`.
2. The original training figure says the mean/shading spans three seeds. All three seeds
   are present only through display epoch 20. Two remain from epochs 21–26 and
   one remains from epoch 27 onward. The late mean is therefore not a
   three-seed comparison. Use this figure to show early stopping and growing
   train/validation separation, or regenerate it with a seed-count cue.

## Figure-by-figure assessment

### Architecture

- Technically coherent and visually polished; it correctly shows a shared
  residual head, a nine-slot fixed-capacity adapter, joint W1–W6 processing,
  the log-bias anchor, and IMD-supported loss.
- It is too information-dense for comfortable reading when embedded as half a
  slide. Use it full-slide, or split the architecture and leakage-safe contract
  into two slides.
- The bottom equations, channel inventory, and footer are too small for a
  normal conference room projection.

### Training and blocked-validation curves

- Split labels are honest and prominent.
- Train loss continues downward while blocked-validation loss bottoms early
  and then rises, visibly supporting checkpoint-level early stopping.
- Besides the changing seed count noted above, the two panels use different
  vertical scales. This is legitimate but should not be read as a direct
  amplitude comparison.

### Spatial mean bias by lead

- The common symmetric color scale and six-week/three-method layout aid direct
  comparison.
- Twenty-four maps are too dense for a single talk slide; reserve this for the
  paper/supplement or split it.
- It is descriptive and contains no spatial significance inference.
- Administrative boundaries require the existing presentation-reference
  disclaimer and are not legal/cadastral boundaries.

### Spatial RMSE skill versus compact control

- The six-week layout, sign convention, and `no cell-wise significance claim`
  footer are clear.
- Area-improved fractions around 67–80% look visually strong, but the reported
  area-weighted mean improvement is only about 0.18–0.54%. Lead with the small
  mean magnitude and avoid `large improvement` wording.
- Local red/blue patches are descriptive point estimates, not individually
  significant cells.

### India-area-weighted metrics by lead

- This is the cleanest main validation summary.
- Physical and compact-control curves overlap closely. The plot has no
  uncertainty intervals, so it cannot by itself establish statistical
  significance or a practically large effect.
- Bias remains negative at every lead; bias correction is incomplete even
  where RMSE and ACC improve.

### Temporal and lead-wise validation behavior

- The 2018/2019 divider is clear, and the five-initialization smoothing is
  correctly performed separately within each year.
- The heatmap honestly shows heterogeneous case/lead behavior, including
  degradations. It should be described as a stability/heterogeneity diagnostic,
  not as proof of universal improvement.
- Fine date labels and the dense heatmap are better suited to a full slide or
  appendix.

## Recommended talk order

1. Architecture, simplified or full-slide.
2. Training/validation separation and early stopping, with the seed-count
   caveat fixed or stated.
3. India-area-weighted metrics by lead.
4. Spatial RMSE skill, emphasizing sub-1% mean effects and descriptive scope.
5. Temporal/lead heterogeneity.
6. Spatial mean-bias atlas in appendix/supplement.
7. Explicit limitations slide: reused 2020–2021 exploratory test and no genuine
   independent 2025 result.
