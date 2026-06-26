#!/usr/bin/env bash
# Regenerate every new_anal figure. Run from new_anal/.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] A1 spatial bias maps ..."
python a1_spatial_bias.py

echo "[2/4] A2 grid-point skill maps ..."
python a2_skill_maps.py

echo "[3/4] A3 IMD-region skill profiles ..."
python a3_region_profiles.py

echo "[4/4] A4 SST verification (FuXi vs ERA5) ..."
python a4_sst.py

echo "DONE -> figs/"
ls -1 figs/
