# India S2S Benchmark v1: Pilot Visual-QC Review

## Decision

The preprocessing visual gate **passes with documented cautions** for pilot
`pilot_20230628_29_week1_v4`, catalog `d3ce84adc9f1a7ca`. This decision permits
the same adapters and data model to proceed to full-archive preprocessing. It
does not establish forecast skill, select a best model, or validate an ACC,
MAE, climatology, or probabilistic-score implementation.

The generated review suite is stored at:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/standardized/
india_s2s_benchmark_v1/pilots/pilot_20230628_29_week1_v4/qc_plots/
```

Open `QC_PLOT_INDEX.md` there for links to all figures. The suite contains 18
PNG files and 12 matching publication-resolution PDF files. Six of the PNGs
are valid-time-matched daily precipitation sheets.

## Evidence reviewed

1. Spatial support, India fractions, and area weights have coherent geography
   and no visible longitude/latitude reversal.
2. The standard precipitation products have plausible magnitudes and coherent
   spatial structure on the common grid. The primary cross-model visual check
   is the exactly matched six-day total for 2023-06-30 through 2023-07-05.
3. Daily-mean T2M fields have plausible magnitudes and consistent large-scale
   gradients. NCEP `t2m_proxy` and ERPAS instantaneous `tsfc` are shown in a
   separate non-comparable figure.
4. ERPAS native-grid and common-grid precipitation retain the same large-scale
   structure. The global and India 0.5-degree sensitivity products remain
   distinguishable and are not silently pooled.
5. Ensemble-member inventories agree with the catalog. Spread is present for
   genuine member archives; ERPAS remains labelled as a provider mean with its
   source-member count, never as a deterministic one-member ensemble.
6. The ERPAS geopotential-height panels contain the expected seven pressure
   levels and plausible vertical/spatial structure.
7. The valid-time plot correctly exposes the one-day initialization offset:
   dense systems cover 2023-06-30 through 2023-07-06, while ERPAS covers
   2023-06-29 through 2023-07-05. Daily comparison sheets join by valid time,
   producing the six-date intersection above rather than joining equal lead
   indices.
8. India-area-weighted precipitation time series are finite and physically
   plausible for this integrity test.

## Cautions retained

- The seven-day weekly-total overview panels cover different absolute periods
  for dense systems and ERPAS because the pilot initializations differ by one
  day. The dates are printed on each panel. Use the matched six-day total for
  direct cross-model visual comparison.
- Small negative precipitation increments from differencing accumulated
  physics-model fields are real archive diagnostics. They are visible in the
  dedicated plot, preserved in the processed data, and must not be clipped
  silently.
- ERPAS raw members are unavailable, so ERPAS must be excluded from
  member-based probabilistic scores unless those members are obtained.
- Visual plausibility is necessary but not sufficient. Full-production stores
  must still pass the existing numerical read-back, unit, checksum, temporal,
  and catalog validation before evaluation begins.

## Reproduction

From the repository root, with the pilot archive available:

```bash
PYTHONPATH=/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1/_deps \
  python studies/india_s2s_benchmark_v1/benchmark.py plot-pilot-qc
```

The command reads only finalized catalog records; it does not discover inputs
by recursively listing the large storage tree. The explicit `PYTHONPATH`
selects the archive-pinned Zarr 2 runtime used by the validated pilot.
