# Manuscript build

`main.tex` is the workshop manuscript and `appendix.tex` is its supplement.
All numerical macros and result tables are generated from the audited
case-level artifacts:

```bash
PYTHONPATH=src /home/raj.ayush/.conda/envs/weather_forecast/bin/python make_paper.py
latexmk -pdf -halt-on-error -interaction=nonstopmode -cd paper/main.tex
```

The current five-page `main.pdf` was successfully compiled with Tectonic 0.15.0;
there were no missing references, figures, or fatal errors. Replace `Anonymous
authors` and the generic article layout with the official CCAI--NeurIPS 2026
style once that template is available; do not change the generated result
macros by hand.
