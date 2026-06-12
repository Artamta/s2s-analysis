#!/bin/bash
#SBATCH --job-name=FuXi_Infer
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=fuxi_inference_%j.log
#SBATCH --time=12:00:00

# Activate the dedicated FuXi conda environment
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate fuxi_s2s

# Move into the inference directory and run your script
cd "/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)/fuxi_s2s"

python run_inference_loop.py
