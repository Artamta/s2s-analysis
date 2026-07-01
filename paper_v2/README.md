# paper_v2 — Two-Season India S2S Benchmark (clean rebuild)

arXiv-style single-column preprint. **Current story: two asymmetric case
studies, not a symmetric six-model/two-season hindcast.** JFM 2026 is the
Spire-inclusive winter case; JJAS 2019 is the 35-date monsoon benchmark for
ECMWF/UKMO/NCEP/FuXi, with DLESyM retained only as a smaller sensitivity.

## The story (3 + 1 findings)

1. **Winter (JFM 2026): ML can win.** Spire AI-S2S leads precipitation at every
   lead (W1 ACC 0.78 vs 0.73 ECMWF) and uniquely holds Z500 skill to week 6,
   while FuXi-S2S and DLESyM Z500 collapse to *negative* ACC by week 3–4.
2. **Monsoon (JJAS 2019): the great equalizer.** In the 35-date benchmark,
   ECMWF/UKMO/NCEP/FuXi all lose useful precipitation skill by week 3-4. Spire
   is not available for this season; DLESyM has no precipitation channel.
3. **ML systems diverge in JFM.** Long-lead stability differs sharply across
   Spire/FuXi/DLESyM; week-1 leaderboards are misleading.
4. **Deterministic != probabilistic.** ECMWF leads JFM precipitation CRPSS in
   weeks 2-5 even where Spire leads ACC; calibration is a separate axis.

## How to work on it

```bash
# 1. paired-bootstrap significance over initializations (writes the CSVs the
#    tables/figures consume; run before make_tables/make_figures)
python scripts/make_bootstrap.py   # -> tables/bootstrap_ci.csv, bootstrap_pairwise.csv

# 2. regenerate tables and metric figures from the result CSVs (idempotent)
python scripts/make_tables.py      # -> tables/*.tex (incl. tab_jfm_sig_*.tex)
python scripts/make_figures.py     # -> figs/*.pdf (ACC fig gets CI bands)

# 3. compile
conda run -n tectonic_env tectonic s2s_india_benchmark.tex
```

Note: analysis scripts run under the `s2s-hind` conda env (pandas/numpy/
matplotlib); `tectonic_env` is LaTeX-only. Concretely:
`/home/raj.ayush/.conda/envs/s2s-hind/bin/python3 scripts/make_bootstrap.py`.

**Never hand-type a score.** All numbers live in `tables/*.tex` and `figs/*.pdf`,
generated from:
- JFM: `final_paper/.../jfm2026/05_tables/full_jfm2026_daily_spire/`
- JJAS TP main: `final_paper/.../jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/`
- JJAS Z500 main: `final_paper/.../jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/`
- JJAS DLESyM sensitivity/T2M: `final_paper/.../jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/`
To change a number, edit the script and rerun — not the `.tex`.

Spatial/scatter appendix PDFs are not included in the current paper draft
because the current full JFM grid-level scatter CSV is empty. Regenerate
grid-level diagnostics before reintroducing those figures.

## Writing status

All prose placeholders have been resolved. Remaining polish:
- Add bootstrap confidence intervals or avoid language that implies formal
  significance.
- Rebuild grid-level spatial diagnostics if those appendix figures are wanted.
- Consider trimming Limitations if it runs long relative to journal norms.

## Files

- `s2s_india_benchmark.tex` — main paper (the scaffold)
- `tables/tab_models.tex` — hand-written model table
- `tables/tab_*.tex` — auto-generated metric tables (do not edit by hand)
- `figs/*.pdf` — auto-generated figures
- `bibliography_block.tex` — 32 citations carried from the prior draft
- `scripts/make_tables.py`, `scripts/make_figures.py` — generators
