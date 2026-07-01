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
  Current analysis pipeline and result products used to generate paper_v2.

analysis-code/
  Data-download utilities, exploratory analysis, earlier experiments, and
  supporting scripts. Some paths point to local or restricted storage.

paper/
  Legacy JFM-only paper generation workflow retained for provenance.

final_analysis/
  Earlier analysis framework and diagnostics retained for provenance.
```

## Reproduce The Current Preprint

From `paper_v2/`:

```bash
python scripts/make_bootstrap.py
python scripts/make_spatial_cache.py
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/make_case_study.py
python scripts/make_scatter.py
conda run -n tectonic_env tectonic s2s_india_benchmark.tex
python scripts/make_arxiv_bundle.py
```

The generated arXiv upload bundle is:

```text
paper_v2/arxiv_submission.tar.gz
```

The paper scripts expect the processed result products under
`final_paper/outputs/s2s_paper_outputs/`. Raw forecast and truth data are not
redistributed in this repository.

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
