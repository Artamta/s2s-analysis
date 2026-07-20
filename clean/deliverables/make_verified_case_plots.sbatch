#!/bin/bash
#SBATCH --job-name=verified_plots
#SBATCH --partition=GPU-AI_prio
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/verified_plots_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/verified_plots_%j.err

set -euo pipefail

PLOT_CACHE=/tmp/verified_plots_${SLURM_JOB_ID}
mkdir -p "${PLOT_CACHE}/mpl" "${PLOT_CACHE}/xdg"
export MPLCONFIGDIR=${PLOT_CACHE}/mpl
export XDG_CACHE_HOME=${PLOT_CACHE}/xdg

/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  /home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/make_verified_case_plots.py \
  --init 20260601 \
  --init 20260623
