#!/bin/bash
# Wait for the s2s_jfm2026 SLURM jobs to LEAVE the queue (race-free: does not
# depend on whether stale CSVs exist), then render all figures for both
# resolutions + the common-vs-native comparison.
set -uo pipefail
J=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026
echo "waiting for SLURM jobs named 's2s_jfm2' to finish ..."
# small grace so the just-submitted jobs are registered before we poll
sleep 30
until ! squeue -u raj.ayush -h -n s2s_jfm2026 2>/dev/null | grep -q .; do
    sleep 20
done
echo "jobs done; letting CSV writes flush ..."
sleep 15
source ~/.bashrc 2>/dev/null; conda activate s2s-hind 2>/dev/null
cd "$J/plots"
for res in 1.5deg 0.5deg; do
    if [ -s "$J/results_${res}/skill_deterministic.csv" ]; then
        echo "=== plotting ${res} ==="
        python make_plots.py --results "$J/results_${res}" 2>&1 | grep -v "Warning\|warn"
    else
        echo "!! missing results_${res}/skill_deterministic.csv — skipped"
    fi
done
echo "=== common-vs-native comparison ==="
python make_plots.py --results "$J/results_1.5deg" --compare "$J/results_0.5deg" 2>&1 | grep -v "Warning\|warn"
echo "ALL_FIGURES_DONE"
