# India S2S Forecast Benchmark

Python code for verifying subseasonal forecast skill over India.

This repository contains the core analysis package, workflow scripts, and small
IMD region masks used for the India S2S benchmark. It does not include raw
forecasts, provider datasets, generated figures/tables, manuscript files, or
arXiv bundles.

## Layout

```text
src/s2s_benchmark/   core package
scripts/             command-line workflow scripts
masks/               small IMD region masks
environment.yml      conda environment
```

## Setup

```bash
conda env create -f environment.yml
conda activate s2s-analysis
```

Optional path overrides:

```bash
export S2S_STORAGE_ROOT=/path/to/storage-root
export S2S_DATA_ROOT=/path/to/All_Model_Data
export S2S_OUTPUT_ROOT=/path/to/outputs
```

## Quick Checks

```bash
python scripts/01_check_core.py
python scripts/02_check_metrics_formulas.py
```

Full runs require local access to the forecast and verification datasets:

```bash
python scripts/00_check_foundation.py
python scripts/07_run_weekly_metrics_pipeline.py --season jfm2026 --include-spire --run-label full_daily_spire
```

## License

MIT. See [LICENSE](LICENSE).
