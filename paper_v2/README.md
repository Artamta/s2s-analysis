# paper_v2 — Two-Season India S2S Benchmark (clean rebuild)

arXiv-style single-column preprint. Structure built with Opus; full prose
written with Sonnet. **Status: complete first draft, 19 pages, compiles
clean, zero undefined refs/citations.** Next pass: human read-through,
domain-map figure (Sec. 2.1 currently has no figure), and tightening.

## The story (3 + 1 findings)

1. **Winter (JFM 2026): ML can win.** Spire AI-S2S leads precipitation at every
   lead (W1 ACC 0.78 vs 0.73 ECMWF) and uniquely holds Z500 skill to week 6,
   while FuXi-S2S and DLESyM Z500 collapse to *negative* ACC by week 3–4.
2. **Monsoon (JJAS 2019): the great equalizer.** Every system — ML and
   dynamical — loses precipitation skill by week 3 (all-India ACC ≤ 0.06). No
   ML advantage survives.
3. **ML systems diverge.** Long-lead stability differs sharply across the three
   ML systems; week-1 leaderboards are misleading.
4. **Deterministic ≠ probabilistic.** ECMWF leads CRPSS even where Spire leads
   ACC — calibration is a separate axis.

## How to work on it

```bash
# 1. regenerate tables and figures from the result CSVs (idempotent)
python scripts/make_tables.py      # -> tables/*.tex
python scripts/make_figures.py     # -> figs/*.pdf

# 2. compile
conda run -n tectonic_env tectonic s2s_india_benchmark.tex
```

**Never hand-type a score.** All numbers live in `tables/*.tex` and `figs/*.pdf`,
generated from:
- JFM: `final_paper/.../jfm2026/05_tables/full_jfm2026_daily_spire/`
- JJAS: `final_paper/.../jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/`
To change a number, edit the script and rerun — not the `.tex`.

## Writing status

All `\TODO` prose markers have been resolved — every section has full text,
verified against the result CSVs (no hand-typed numbers; every quoted score
was cross-checked against `tables/*.tex` or pulled fresh from the summary
CSVs). Remaining polish for a Sonnet pass:
- Add the domain/region map figure referenced conceptually in §2.1 (not yet
  drawn — would reuse the old paper's `fig01_domain` style)
- Read-through for sentence-level tightening and flow
- Verify the `\citep`/`\citealt` mix reads naturally throughout
- Consider trimming Limitations if it runs long relative to journal norms

## Files

- `s2s_india_benchmark.tex` — main paper (the scaffold)
- `tables/tab_models.tex` — hand-written model table
- `tables/tab_*.tex` — auto-generated metric tables (do not edit by hand)
- `figs/*.pdf` — auto-generated figures
- `bibliography_block.tex` — 32 citations carried from the prior draft
- `scripts/make_tables.py`, `scripts/make_figures.py` — generators
