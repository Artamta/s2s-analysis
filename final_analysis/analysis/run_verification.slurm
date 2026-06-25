#!/bin/bash
#SBATCH --job-name=s2s_verify
#SBATCH --partition=gpu
#SBATCH --account=iiser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/verify_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/verify_%j.err

###############################################################################
#  ECMWF S2S VERIFICATION + NOVEL ANALYSIS
#  15 figures: ACC, RMSE, reliability, rank histogram, conditional skill,
#  forecast busts, scale-dependent skill, Murphy decomposition, ROC, etc.
###############################################################################

echo "═══ ECMWF S2S Verification Pipeline ═══"
echo "Node: $(hostname) | CPUs: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "Time: $(date)"

source ~/.bashrc
conda activate s2s-hind 2>/dev/null || true
cd /home/raj.ayush/s2s/s2s_anlysis/final_analysis
mkdir -p analysis/slurm_logs

# Wait for script to appear (subagent may still be writing it)
for i in $(seq 1 60); do
    if [ -f analysis/ecmwf_verification.py ]; then
        echo "Script found! Starting..."
        break
    fi
    echo "  Waiting for script... ($i/60)"
    sleep 10
done

if [ ! -f analysis/ecmwf_verification.py ]; then
    echo "ERROR: Script not found after 10 minutes. Exiting."
    exit 1
fi

python analysis/ecmwf_verification.py 2>&1 || echo "⚠ Some figures may have failed"

echo ""
echo "═══ Done! $(date) ═══"
echo "Figures:"
ls -lhS analysis/ecmwf_verification/*.png 2>/dev/null || echo "  (no output)"
