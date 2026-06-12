# Subseasonal (S2S) AI vs Physics benchmarking over India — JFM 2026

Real-time evaluation of four subseasonal forecast systems — **SPIRE AI-S2S**,
**FuXi-S2S**, **ECMWF** extended-range, and **NCEP CFSv2** — over India for
January–March 2026, verified against ERA5 for precipitation, 500 hPa
geopotential height (Z500), and 2-m temperature.

```
.
├── paper/                ★ SELF-CONTAINED PAPER: manuscript + figures + the code that makes them
│   ├── jfm2026_india_s2s_benchmark.tex / .pdf   manuscript source + compiled report
│   ├── figs/                                     all manuscript figures (fig01…fig27)
│   ├── code/                                     pipeline that produces every figure
│   │   ├── README.md   maps each figure → its script
│   │   ├── run_all.sh  reproduces all figures (compute → figures)
│   │   ├── compute/    raw forecasts → skill tables
│   │   ├── figures/    skill tables → figure PDFs (written to ../figs)
│   │   └── utils/      shared verification helpers
│   └── README.md
│
└── analysis-code/        working / supporting code + intermediate data
    ├── analysis/         development scripts AND the precomputed skill tables (CSV/NC/NPZ)
    │                       that paper/code reads
    ├── data-download/    download scripts for ERA5, ECMWF, NCEP, FuXi, SPIRE, IMD
    ├── imd-fuxi/         IMD gauge-rainfall vs FuXi comparison work
    └── experiments/      earlier prototypes and scratch outputs
```

## For reviewers

- **The paper and its code are together in `paper/`.** Read
  `paper/jfm2026_india_s2s_benchmark.pdf`; review the code in `paper/code/`
  (start with `paper/code/README.md`, which maps every figure to the exact
  script that makes it).
- `analysis-code/analysis/` is the development history plus the **precomputed
  skill tables** (`*.csv`, `*.nc`, `*.npz`) that the figure scripts read.

## Reproduce / rerun / verify

```bash
# regenerate every figure from the precomputed skill tables:
for f in paper/code/figures/make_*.py; do python "$f"; done

# or the full pipeline (recompute the skill tables from raw forecasts, then plot):
bash paper/code/run_all.sh
```

- **Figure step** needs only the precomputed tables in `analysis-code/analysis/`
  (paths are set at the top of each script).
- **Compute step** additionally reads raw forecasts from
  `/storage/raj.ayush/s2s-forecast-data` and the public ARCO-ERA5 archive.
