# India S2S Forecast Benchmark

Code companion for the preprint:

**Machine-Learning and Operational Subseasonal Forecasts over India: An Early
Two-Season Benchmark across Winter and Monsoon Regimes**

This repository contains the verification code, workflow scripts, small region
masks, scheduler launchers, and reproducibility notes for the India S2S
benchmark. Manuscript source, arXiv bundles, generated figures, generated
tables, raw forecasts, and provider datasets are intentionally not tracked here.

The scientific scope is intentionally limited: this is an early two-season
benchmark, not a climatological ranking of all forecast systems.

## Repository Layout

```text
src/s2s_benchmark/
  Reusable Python package: path registry, grid utilities, region masks, metrics,
  and weekly verification pipeline.

scripts/
  Command-line checks, data inventory tools, forecast openers, pipeline runners,
  validation scripts, and result builders.

docs/
  Methodology notes, data-flow notes, study decisions, and output conventions.

masks/
  Small IMD homogeneous-region masks used by the verification code.

slurm/
  Optional HPC launchers for smoke tests and full verification runs.

outputs/
  Local/generated verification outputs. Only `.gitkeep` is tracked.
```

## Environment

```bash
conda env create -f environment.yml
conda activate s2s-analysis
```

Useful path overrides:

```bash
export S2S_STORAGE_ROOT=/path/to/storage-root
export S2S_DATA_ROOT=/path/to/All_Model_Data
export S2S_OUTPUT_ROOT=/path/to/verification-outputs
```

## Quick Checks

```bash
python scripts/01_check_core.py
python scripts/02_check_metrics_formulas.py
```

Foundation checks and full pipeline runs require the local/provider datasets
described in [`DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md):

```bash
python scripts/00_check_foundation.py
python scripts/04_audit_model_usability.py
python scripts/07_run_weekly_metrics_pipeline.py --season jfm2026 --include-spire --run-label full_daily_spire
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`docs/PIPELINE.md`](docs/PIPELINE.md)
for the full workflow.

## Repository Boundary

Tracked:

- Verification source code and CLI scripts.
- Small, derived IMD region masks required by the code.
- Documentation, licensing, citation, and security metadata.
- Empty output-directory placeholder.

Not tracked:

- Manuscript source/PDF, arXiv bundles, generated figures, generated tables.
- Raw or provider-delivered forecasts and truth datasets.
- Intermediate pipeline outputs, scheduler logs, notebooks, scratch analyses,
  credentials, and local environment files.

## Data And Licenses

Code is MIT licensed. Project-authored documentation is CC BY 4.0. Raw/provider
data are not redistributed or relicensed by this repository. See
[`DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md).

## Citation

Citation metadata is in [`CITATION.cff`](CITATION.cff). After the arXiv
identifier or any archival DOI is available, update that file.

## Security

Do not commit API keys, `.cdsapirc`, provider tokens, private data paths, or
restricted datasets. Use environment variables or provider-specific credential
files outside the repository.
