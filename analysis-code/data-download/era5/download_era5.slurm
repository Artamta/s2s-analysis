#!/bin/bash
#SBATCH --job-name=ERA5_Download
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=era5_download_%j.log
#SBATCH --time=12:00:00

# Activate the correct environment
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate s2s-hind

# Move to the correct directory and run the downloader
cd "/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)/era5/"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python download_era5_arco.py
