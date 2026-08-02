#!/bin/bash
# Build the manifest, validate task zero first, then submit the full resumable array.

set -euo pipefail

REPO="${FUXI_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}"
PYTHON="${FUXI_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}"
MANIFEST_SCRIPT="${REPO}/clean/model-runs/fuxi/scripts/build_native_reforecast_manifest.py"
SBATCH_SCRIPT="${REPO}/clean/model-runs/fuxi/slurm/standardize_native_reforecast_jjas_2002_2021.sbatch"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_jjas_2002_2021"

"${PYTHON}" "${MANIFEST_SCRIPT}"
mkdir -p "${RUN_ROOT}/logs"
pilot_job=$(sbatch --parsable --array=0 "${SBATCH_SCRIPT}")
full_job=$(sbatch --parsable --array=1-699%8 --dependency="afterok:${pilot_job}_0" "${SBATCH_SCRIPT}")
printf 'native_reforecast_pilot=%s\n' "${pilot_job}"
printf 'native_reforecast_full=%s\n' "${full_job}"
