# Reproducibility Guide

This repository is a paper and code companion, not a mirror of the restricted
forecast archives. It tracks the manuscript, paper-build scripts, generated
publication figures/tables, reusable verification code, small region masks, and
metadata. Raw forecasts, truth datasets, provider downloads, intermediate
pipeline outputs, scheduler logs, local notebooks, and credentials stay outside
Git.

## Environment

Create the analysis environment from the repository root:

```bash
conda env create -f environment.yml
conda activate s2s-analysis
```

The paper scripts read generated verification outputs from:

```text
final_paper/outputs/s2s_paper_outputs/
```

That path can be overridden without editing code:

```bash
export S2S_PAPER_OUTPUT_ROOT=/path/to/s2s_paper_outputs
```

The verification pipeline uses local/provider data roots. Important overrides:

```bash
export S2S_STORAGE_ROOT=/path/to/storage-root
export S2S_DATA_ROOT=/path/to/All_Model_Data
export S2S_ERA5_CLIMATOLOGY=/path/to/era5_climatology.nc
export S2S_WEATHERBENCH2_ERA5_ZARR=/path/to/weatherbench2_era5.zarr
```

## Verification Pipeline

Run lightweight checks first:

```bash
cd final_paper
python scripts/00_check_foundation.py
python scripts/01_check_core.py
python scripts/02_check_metrics_formulas.py
python scripts/03_check_common_grid.py
python scripts/04_audit_model_usability.py
```

Then run the weekly metrics workflow for the desired season/run labels. Full
publication runs require the restricted/local forecast and truth archives named
in the paper and `DATA_AND_LICENSES.md`.

## Paper Rebuild

From the repository root:

```bash
python paper_v2/scripts/make_bootstrap.py
python paper_v2/scripts/make_spatial_cache.py
python paper_v2/scripts/make_tables.py
python paper_v2/scripts/make_figures.py
python paper_v2/scripts/make_case_study.py
python paper_v2/scripts/make_scatter.py
cd paper_v2
tectonic s2s_india_benchmark.tex
python scripts/make_arxiv_bundle.py
```

The generated arXiv tarball is intentionally ignored by Git:

```text
paper_v2/arxiv_submission.tar.gz
```

## What To Archive

For a permanent release, archive this GitHub repository plus the derived CSVs
needed to rebuild the paper tables/figures. Do not redistribute raw
provider-delivered datasets unless their licenses explicitly allow it. After an
arXiv identifier or Zenodo DOI is available, update `CITATION.cff` and the
paper data-availability statement.
