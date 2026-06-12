#!/bin/bash
#SBATCH --job-name=IMD_Download
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=imd_download_%j.log
#SBATCH --time=02:00:00

# Activate the correct environment
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate s2s-hind

# Ensure imdlib is installed in the environment before running
pip install imdlib

# Move to the correct directory and run the downloader
cd "/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)/imd/"

python download_imd_rainfall.py
