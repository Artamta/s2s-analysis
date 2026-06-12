#!/bin/bash
#SBATCH --job-name=FuXi_Download
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=fuxi_download_%j.log
#SBATCH --time=12:00:00

# Activate the correct environment
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate s2s-hind

# Move to the correct directory and run the downloader
cd "/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)/fuxi_s2s/"

# We are giving it all 32 threads so it downloads incredibly fast!
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

python download_all_fuxi_inputs.py
