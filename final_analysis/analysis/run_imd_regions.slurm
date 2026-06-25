#!/bin/bash
#SBATCH --job-name=s2s_imd_regions
#SBATCH --partition=gpu
#SBATCH --account=iiser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/imd_regions_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/imd_regions_%j.err

echo "═══ IMD Homogeneous Regions + Ground Truth Analysis ═══"
echo "Node: $(hostname) | Time: $(date)"
source ~/.bashrc; conda activate s2s-hind 2>/dev/null || true
cd /home/raj.ayush/s2s/s2s_anlysis/final_analysis
mkdir -p analysis/slurm_logs
python analysis/explore_imd_regions.py 2>&1 || echo "⚠ Some figures failed"
echo "═══ Done! $(date) ═══"
ls -lhS analysis/imd_regions_exploration/*.png 2>/dev/null
