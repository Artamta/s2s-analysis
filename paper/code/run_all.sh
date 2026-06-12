#!/usr/bin/env bash
# Reproduce every manuscript figure and table, end to end.
#   compute/ : raw forecasts -> skill tables  (analysis-code/analysis/*.csv,*.nc,*.npz)
#   figures/ : skill tables  -> manuscript PDFs (paper/figs/)
# The scripts use absolute paths, so this can be launched from anywhere, e.g.:
#   bash paper/code/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"            # -> paper/code/
export PYTHONUNBUFFERED=1

echo "=================  COMPUTE PIPELINE  ================="
for s in compute/0*.py; do
    echo ">>> $s"
    python "$s"
done

echo "=================  FIGURES  ========================="
for s in figures/make_*.py; do
    echo ">>> $s"
    python "$s"
done

echo "All manuscript figures and tables regenerated -> paper/figs/ and analysis-code/analysis/tables_regional.tex"
