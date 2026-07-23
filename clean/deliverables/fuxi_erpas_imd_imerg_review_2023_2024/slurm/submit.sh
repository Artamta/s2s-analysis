#!/usr/bin/env bash
set -euo pipefail

REVIEW_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024
mkdir -p "${REVIEW_ROOT}/logs"

STAGE_JOB=$(sbatch --parsable "${REVIEW_ROOT}/slurm/stage_imerg_years.sbatch")
AGGREGATE_JOB=$(sbatch --parsable --dependency="afterok:${STAGE_JOB}" "${REVIEW_ROOT}/slurm/aggregate_imerg.sbatch")
REVIEW_JOB=$(sbatch --parsable --dependency="afterok:${AGGREGATE_JOB}" "${REVIEW_ROOT}/slurm/calculate_and_plot.sbatch")

printf 'IMERG_YEAR_ARRAY=%s\nIMERG_AGGREGATE=%s\nREVIEW_AND_FIGURES=%s\n' \
  "${STAGE_JOB}" "${AGGREGATE_JOB}" "${REVIEW_JOB}"
