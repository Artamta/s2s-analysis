#!/bin/bash
#SBATCH --job-name=s2s_tp_analysis
#SBATCH --partition=gpu
#SBATCH --account=iiser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/s2s_analysis_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis/slurm_logs/s2s_analysis_%j.err

###############################################################################
#  MASTER S2S ANALYSIS BATCH JOB
#  Runs ALL precipitation, temperature, ECMWF, and monsoon dynamics analyses
#  Submit: sbatch run_all_analysis.slurm
###############################################################################

set -e
echo "═══════════════════════════════════════════════════════════════"
echo "  S2S Analysis Master Batch Job"
echo "  Node: $(hostname)"
echo "  CPUs: $(nproc)"
echo "  RAM:  $(free -h | awk '/^Mem:/{print $2}')"
echo "  Time: $(date)"
echo "  Conda: $CONDA_DEFAULT_ENV"
echo "═══════════════════════════════════════════════════════════════"

# Activate conda environment
source ~/.bashrc
conda activate s2s-hind 2>/dev/null || true

cd /home/raj.ayush/s2s/s2s_anlysis/final_analysis

# Create log dir
mkdir -p analysis/slurm_logs

###############################################################################
echo ""
echo "━━━ [1/5] ERA5 TP Publication Figures (1.5°) ━━━"
echo "  Script: analysis/explore_tp_publication.py"
echo "  Output: analysis/tp_exploration_pub/"
python analysis/explore_tp_publication.py 2>&1 || echo "  ⚠ Some figures may have failed"

###############################################################################
echo ""
echo "━━━ [2/5] ERA5 T2M Temperature Figures ━━━"
echo "  Script: analysis/explore_t2m_patterns.py"
echo "  Output: analysis/t2m_exploration/"
python analysis/explore_t2m_patterns.py 2>&1 || echo "  ⚠ Some figures may have failed"

###############################################################################
echo ""
echo "━━━ [3/5] ECMWF S2S Hindcast TP Figures ━━━"
echo "  Script: analysis/explore_ecmwf_tp.py"
echo "  Output: analysis/ecmwf_tp_exploration/"
python analysis/explore_ecmwf_tp.py 2>&1 || echo "  ⚠ Some figures may have failed"

###############################################################################
echo ""
echo "━━━ [4/5] IMD Regions & Ground Truth ━━━"
echo "  Script: analysis/explore_imd_regions.py"
echo "  Output: analysis/imd_regions_exploration/"
if [ -f analysis/explore_imd_regions.py ]; then
    python analysis/explore_imd_regions.py 2>&1 || echo "  ⚠ Some figures may have failed"
else
    echo "  ⏭ Script not yet created, skipping"
fi

###############################################################################
echo ""
echo "━━━ [5/5] Advanced Monsoon Dynamics ━━━"
echo "  Script: analysis/explore_monsoon_dynamics.py"
echo "  Output: analysis/monsoon_dynamics/"
if [ -f analysis/explore_monsoon_dynamics.py ]; then
    python analysis/explore_monsoon_dynamics.py 2>&1 || echo "  ⚠ Some figures may have failed"
else
    echo "  ⏭ Script not yet created, skipping"
fi

###############################################################################
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL ANALYSES COMPLETE"
echo "  Time: $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Output directories:"
for d in tp_exploration_pub tp_exploration t2m_exploration ecmwf_tp_exploration imd_regions_exploration monsoon_dynamics tp_0.25; do
    if [ -d "analysis/$d" ]; then
        count=$(ls analysis/$d/*.png 2>/dev/null | wc -l)
        size=$(du -sh analysis/$d 2>/dev/null | cut -f1)
        echo "  analysis/$d/: $count figures ($size)"
    fi
done
