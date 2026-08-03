# India S2S Benchmark v1

This directory contains the publication preprocessing layer for the India S2S
benchmark.  Raw forecast archives are read-only.  Standardized outputs are
model/experiment/variable/grid/year Zarr v2 stores containing the member field
and derived ensemble statistics.

The canonical archive root is:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1
```

The first execution gate is a seven-lead-day adapter pilot using 2023-06-29
for the dense/CNRM systems and 2023-06-28 for ERPAS.  The pilot is a processing
test, not a model comparison.

## Environment

The system Earth-2 environment currently exposes Zarr 3 with an incompatible
`typing_extensions`.  Use a clean environment with the pinned requirements:

```bash
python -m pip install -r studies/india_s2s_benchmark_v1/requirements.txt
```

## Commands

```bash
python studies/india_s2s_benchmark_v1/benchmark.py inventory
python studies/india_s2s_benchmark_v1/benchmark.py build-spatial
python studies/india_s2s_benchmark_v1/benchmark.py pilot
python studies/india_s2s_benchmark_v1/benchmark.py validate --pilot
python studies/india_s2s_benchmark_v1/benchmark.py build-index --pilot
PYTHONPATH=/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1/_deps \
  python studies/india_s2s_benchmark_v1/benchmark.py plot-pilot-qc
python studies/india_s2s_benchmark_v1/benchmark.py status --pilot
```

Full processing is deliberately model-variable-year scoped:

```bash
python studies/india_s2s_benchmark_v1/benchmark.py preprocess \
  --model fuxi_s2s --variable tp --year 2023 --grid common_1p5
```

Completed stores are immutable.  An interrupted `.incomplete-*` store may be
removed after inspection and rerun; a completed destination is never replaced.
Paper scripts must pin the generated catalog filename and SHA256.

`spatial/spatial_support.zarr` contains the four IMD-region fractional masks,
their union (`india_fraction`), exact spherical cell areas, and the canonical
`india_area_weight_km2` weights. ACC/MAE scripts should consume those weights
directly instead of rasterizing the India mask again.

The visual-QC command reads the finalized pilot catalog and writes maps,
valid-time diagnostics, ensemble spread/member plots, ERPAS native/common
comparisons, and negative-TP diagnostics below the pilot `qc_plots/` directory.
Those plots are preprocessing checks, not skill or ranking figures.
The explicit `PYTHONPATH` selects the same pinned Zarr 2 runtime used by the
validated SLURM pilot and avoids incompatible user-level Zarr installations.

## SLURM parallelization

The CPU-only preprocessing arrays use the `gpu_prio` partition with no `--gres`
request, so they allocate zero GPUs. They use one one-CPU task per
model-variable-grid-year store, with one writer per store. The default
concurrency cap is 24 jobs (24 independent workers). Override full-production
concurrency with `S2S_ARRAY_CONCURRENCY`, bounded to 1--32. A runtime preflight
records dependency versions on every assigned node before any data are opened.

The seven-day pilot uses 25 independent tasks and an `afterok` finalizer:

```bash
bash studies/india_s2s_benchmark_v1/slurm/submit_pilot.sh
```

Full production is generated and submitted with:

```bash
bash studies/india_s2s_benchmark_v1/slurm/submit_full.sh
```

The archive-local `_deps` directory must contain the pinned Zarr 2 runtime
before submitting compute jobs, including `numcodecs==0.15.1`.

Submission first creates or reuses a content-addressed runtime snapshot below
`runtime/<sha256-prefix>/`. Compute nodes execute that storage-resident copy,
because `/home` is not mounted consistently across all `gpu_prio` nodes.
