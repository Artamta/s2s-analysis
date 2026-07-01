# India S2S AI and Operational Forecast Benchmark

This repository supports the preprint:

**Machine-Learning and Operational Subseasonal Forecasts over India: An Early
Two-Season Benchmark across Winter and Monsoon Regimes**

The current paper lives in [`paper_v2/`](paper_v2/). It benchmarks AI and
operational subseasonal-to-seasonal forecast systems over India using a common
verification framework for:

- JFM 2026 winter forecasts, including Spire AI-S2S, FuXi-S2S, DLESyM, ECMWF,
  UKMO, and NCEP where available.
- JJAS 2019 monsoon forecasts for the systems with overlapping hindcast
  availability.
- Deterministic metrics: ACC, RMSE, bias.
- Probabilistic metrics: CRPSS and spread-skill ratio.
- India-wide and IMD homogeneous-region verification.

The key framing is intentionally limited: this is an early two-season benchmark,
not a climatological ranking of all systems.

## Repository Map

```text
paper_v2/
  arXiv-ready manuscript, generated tables, figures, and build scripts.

final_paper/
  Current verification package: reusable analysis code, pipeline scripts,
  checks, masks, documentation, and SLURM launchers.

REPRODUCIBILITY.md
  Environment, data-root assumptions, rebuild commands, and archive guidance.

DATA_AND_LICENSES.md
  Dataset access boundaries and license notes.
```

## Environment

The portable environment is defined in [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate s2s-analysis
```

The analysis assumes local/provider data are available outside the Git checkout.
Set these when your paths differ from the defaults:

```bash
export S2S_STORAGE_ROOT=/path/to/storage-root
export S2S_DATA_ROOT=/path/to/All_Model_Data
export S2S_PAPER_OUTPUT_ROOT=/path/to/s2s_paper_outputs
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full rebuild workflow.

## Reproduce The Current Preprint

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

The generated arXiv upload bundle is:

```text
paper_v2/arxiv_submission.tar.gz
```

The paper scripts expect the processed result products under
`final_paper/outputs/s2s_paper_outputs/`, or the path specified by
`S2S_PAPER_OUTPUT_ROOT`. Raw forecast and truth data are not redistributed in
this repository.

## What Is Intentionally Not Tracked

The GitHub repository intentionally excludes exploratory notebooks, old paper
drafts, legacy analysis directories, raw forecasts, model weights, provider
downloads, scratch figures, scheduler logs, and local storage products. Keep
those locally or archive them separately if needed; this repository is scoped to
the current paper and the code needed to rebuild its generated artifacts.

`.gitignore` prevents new local/generated files from entering Git; it does not
remove files that are already tracked. The current tracked tree is intentionally
small: manuscript, reusable code, scripts, small masks, generated publication
figures/tables, and metadata.

## Data And Licenses

License boundaries are explicit:

- Code in this repository is licensed under the MIT License; see
  [`LICENSE`](LICENSE).
- Manuscript text, generated figures, generated tables, and documentation that
  are authored for this project are licensed under CC BY 4.0; see
  [`LICENSE-DOCS.md`](LICENSE-DOCS.md).
- Raw or provider-delivered datasets are not relicensed here. Users must obtain
  Spire, FuXi-S2S, DLESyM, ECMWF, UKMO, NCEP, ERA5, and IMD data under their own
  applicable access terms.

More detail is in [`DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). After the arXiv
identifier and any Zenodo DOI are available, update that file and the manuscript
data-availability statement.

## Security Note

Do not commit API keys, `.cdsapirc`, provider tokens, raw restricted data, or
private storage paths. CDS credentials should be supplied through the standard
CDS configuration file or environment variables such as `CDSAPI_KEY`.
