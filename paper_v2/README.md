# paper_v2 — Two-Season India S2S Benchmark (clean rebuild)

arXiv-style single-column preprint. **Structure built with Opus; prose to be
expanded/polished with Sonnet.**

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

## Writing checklist (search the .tex for `\TODO`)

Each `\TODO{...}` is one prose task. Recommended order:
1. Intro ¶1–4 (motivation → ML rise → two-season framing → questions → roadmap)
2. Data/Methods (domain, truth sources, metrics, init sets) — mostly reusable
   from `paper/jfm2026_india_s2s_benchmark.tex`
3. Results §5.2–5.7 (the heart — numbers already in the tables, just narrate)
4. Discussion / Limitations / Conclusions
5. Abstract final pass (last)

## Files

- `s2s_india_benchmark.tex` — main paper (the scaffold)
- `tables/tab_models.tex` — hand-written model table
- `tables/tab_*.tex` — auto-generated metric tables (do not edit by hand)
- `figs/*.pdf` — auto-generated figures
- `bibliography_block.tex` — 32 citations carried from the prior draft
- `scripts/make_tables.py`, `scripts/make_figures.py` — generators
