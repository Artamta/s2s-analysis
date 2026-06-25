#!/bin/bash
#SBATCH --job-name=s2s_hires_025
#SBATCH --partition=gpu
#SBATCH --account=iiser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/hires_025_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/hires_025_%j.err

###############################################################################
#  HIGH-RES 0.25° ANALYSIS JOB
#  Needs more memory (1.8 GB data + processing headroom)
###############################################################################

set -e
echo "═══ 0.25° High-Resolution ERA5 Analysis ═══"
echo "Node: $(hostname) | CPUs: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "Time: $(date)"

source ~/.bashrc
conda activate s2s-hind 2>/dev/null || true
cd /home/raj.ayush/s2s/s2s_anlysis/final_analysis
mkdir -p analysis/slurm_logs

echo ""
echo "━━━ Running 16 figures at 0.25° resolution ━━━"
python analysis/explore_tp_025.py 2>&1 || echo "⚠ Some figures failed"

echo ""
echo "═══ Done! $(date) ═══"
echo "Figures:"
ls -lhS analysis/tp_0.25/*.png 2>/dev/null || echo "  (no output found)"
