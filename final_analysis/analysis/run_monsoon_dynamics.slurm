#!/bin/bash
#SBATCH --job-name=s2s_monsoon_dyn
#SBATCH --partition=gpu
#SBATCH --account=iiser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/monsoon_dyn_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/monsoon_dyn_%j.err

echo "═══ Advanced Monsoon Dynamics Analysis ═══"
echo "Node: $(hostname) | Time: $(date)"
source ~/.bashrc; conda activate s2s-hind 2>/dev/null || true
cd /home/raj.ayush/s2s/s2s_anlysis/final_analysis
mkdir -p analysis/slurm_logs
python analysis/explore_monsoon_dynamics.py 2>&1 || echo "⚠ Some figures failed"
echo "═══ Done! $(date) ═══"
ls -lhS analysis/monsoon_dynamics/*.png 2>/dev/null
