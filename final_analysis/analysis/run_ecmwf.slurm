#!/bin/bash
#SBATCH --job-name=s2s_ecmwf
#SBATCH --partition=gpu
#SBATCH --account=iiser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/ecmwf_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/ecmwf_%j.err

echo "═══ ECMWF S2S Hindcast TP Analysis ═══"
echo "Node: $(hostname) | Time: $(date)"
source ~/.bashrc; conda activate s2s-hind 2>/dev/null || true
cd /home/raj.ayush/s2s/s2s_anlysis/final_analysis
mkdir -p analysis/slurm_logs
python analysis/explore_ecmwf_tp.py 2>&1 || echo "⚠ Some figures failed"
echo "═══ Done! $(date) ═══"
ls -lhS analysis/ecmwf_tp_exploration/*.png 2>/dev/null
