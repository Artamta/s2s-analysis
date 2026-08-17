# Manuscript build

`main.tex` is the only workshop manuscript and `main.pdf` is its only compiled
PDF. `appendix.tex` is included in the same document. All numerical macros,
result tables, and figures are generated from audited case-level artifacts:

```bash
MPLCONFIGDIR=/tmp/india-s2s-mpl PYTHONPATH=src \
  /home/raj.ayush/.conda/envs/weather_forecast/bin/python make_paper.py
latexmk -pdf -halt-on-error -interaction=nonstopmode -cd paper/main.tex
```

Use Git commits rather than filenames such as draft-v2 or final-final. Do not
change generated result macros by hand.

The source intentionally uses a generic article layout. The active submission
form requires the official CCAI 2026 style and a four-content-page paper-track
limit, but current author policy requires the author to retrieve and apply the
CCAI template personally rather than providing it to an LLM. After manual
template migration, recheck page count, anonymity, references, and every
figure in the rendered PDF. See `WRITING_HANDOFF.md`.
