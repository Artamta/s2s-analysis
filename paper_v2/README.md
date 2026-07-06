# paper_v2 — Two-Season India S2S Benchmark (clean rebuild)

arXiv-style single-column preprint. **Current story: two asymmetric case
studies, not a symmetric six-model/two-season hindcast.** JFM 2026 is the
Spire-inclusive winter case; JJAS 2019 is the 35-date monsoon benchmark for
ECMWF/UKMO/NCEP/FuXi, with DLESyM retained only as a smaller sensitivity.

## The story (precipitation-first)

1. **Winter (JFM 2026): ML can win.** Spire AI-S2S leads individual precipitation systems at every
   lead (W1 ACC 0.78 vs 0.73 ECMWF), with the strongest block-bootstrap support at short lead.
2. **Monsoon (JJAS 2019): the great equalizer.** In the 35-date benchmark,
   ECMWF/UKMO/NCEP/FuXi all lose useful precipitation skill by week 3-4. Spire
   is not available for this season; DLESyM has no precipitation channel. This
   TP result belongs in the main Results because JJAS rainfall is the most
   operationally important target.
3. **Regional structure matters.** The main winter figure now shows all-India
   precipitation ACC first, then the four IMD homogeneous rainfall regions below it.
4. **Deterministic != probabilistic.** ECMWF leads JFM precipitation CRPSS in
   weeks 2-5 even where Spire leads ACC; calibration is a separate axis.
5. **Z500 is appendix context.** Z500 is retained as an auxiliary circulation
   diagnostic, but it is no longer a headline target variable in the main text.

## How to work on it

```bash
# 1. paired-bootstrap significance over initializations (writes the CSVs the
#    tables/figures consume; run before make_tables/make_figures)
python paper_v2/scripts/make_bootstrap.py     # -> paper_v2/tables/bootstrap_ci.csv, bootstrap_pairwise.csv

# 2. spatial cache: reduce the big grid-scatter CSVs to per-cell diagnostics
python paper_v2/scripts/make_spatial_cache.py # -> paper_v2/cache/spatial_cells_*.csv

# 3. regenerate tables and figures (idempotent)
python paper_v2/scripts/make_tables.py        # -> paper_v2/tables/*.tex
python paper_v2/scripts/make_figures.py       # -> paper_v2/figs/*.pdf
python paper_v2/scripts/make_case_study.py    # -> paper_v2/figs/fig_case_study_{jfm,jjas}.pdf
python paper_v2/scripts/make_scatter.py       # -> paper_v2/figs/fig_scatter_{tp,z500}.pdf

# 4. compile, then assemble the arXiv upload tarball
cd paper_v2
tectonic s2s_india_benchmark.tex
python scripts/make_arxiv_bundle.py  # -> arxiv_submission.tar.gz
```

The portable environment is defined in `../environment.yml`. The scripts read
processed verification products from `../final_paper/outputs/s2s_paper_outputs`
by default. Override this with `S2S_PAPER_OUTPUT_ROOT=/path/to/s2s_paper_outputs`
when rebuilding from archived outputs.

**Never hand-type a score.** All numbers live in `tables/*.tex` and `figs/*.pdf`,
generated from:
- JFM: `final_paper/.../jfm2026/05_tables/full_jfm2026_daily_spire/`
- JJAS TP main: `final_paper/.../jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/`
- JJAS Z500 main: `final_paper/.../jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/`
- JJAS DLESyM sensitivity/T2M: `final_paper/.../jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/`
To change a number, edit the script and rerun — not the `.tex`.

Main Figure 2 is generated as `figs/fig_acc_lead.pdf`: all-India JFM 2026
precipitation ACC in a compact centered panel and the four IMD regional ACC
panels below. JJAS TP reference sensitivity remains in the main Results. The
appendix Z500 all-India ACC figure is `figs/fig_z500_acc_appendix.pdf`.

Spatial/scatter appendix PDFs are included in the current paper draft. The
current appendix figures are regenerated from the cached per-cell diagnostics
and the full grid-level scatter CSVs listed above.

## Writing status

All prose placeholders have been resolved. Remaining polish before submission is
limited to rebuilding the PDF/arXiv bundle after any manuscript or figure change.

## Files

- `s2s_india_benchmark.tex` — main paper (the scaffold)
- `tables/tab_models.tex` — hand-written model table
- `tables/tab_*.tex` — auto-generated metric tables (do not edit by hand)
- `figs/*.pdf` — auto-generated figures
- `bibliography_block.tex` — 32 citations carried from the prior draft
- `scripts/make_tables.py`, `scripts/make_figures.py` — generators
