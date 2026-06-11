#!/bin/bash
#SBATCH --job-name=fuxi_s2s
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/fuxi_s2s_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/fuxi_s2s_%j.err

# Activate conda env
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate fuxi_s2s

FUXI_DIR="/home/raj.ayush/s2s/s2s_anlysis/benchmark(jfm)/fuxi_s2s/FuXi-S2S"
cd "$FUXI_DIR"

echo "=== GPU info ==="
nvidia-smi
echo "=== Starting FuXi-S2S inference ==="
echo "Time: $(date)"

python inference.py \
    --model model/fuxi_s2s.onnx \
    --input data/input.nc \
    --device cuda \
    --total_step 42 \
    --total_member 11 \
    --save_dir output

echo "=== Done: $(date) ==="
