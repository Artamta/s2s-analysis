# Manuscript — *Subseasonal forecast benchmarking over India (JFM 2026)*

Self-contained paper: the LaTeX source, the compiled report, all figures, **and
the code that generates the figures**.

```
paper/
├── jfm2026_india_s2s_benchmark.tex   manuscript source
├── jfm2026_india_s2s_benchmark.pdf   compiled report
├── figs/                             all figures (fig01…fig27), referenced by the .tex
└── code/                             the figure-generation pipeline
    ├── README.md                       maps every figure → its script
    ├── run_all.sh                      reproduce all figures (compute → figures)
    ├── compute/  figures/  utils/
```

## Compile the manuscript

```bash
cd paper
tectonic jfm2026_india_s2s_benchmark.tex      # or pdflatex (run twice for refs)
```

## Regenerate / verify the figures

```bash
# from the repository root — replot from the precomputed skill tables:
for f in paper/code/figures/make_*.py; do python "$f"; done
```

Figure scripts read the precomputed skill tables from
`../analysis-code/analysis/` and write the PDFs into `paper/figs/`. See
`code/README.md` for the figure-to-script map and the full
recompute-from-raw pipeline.
