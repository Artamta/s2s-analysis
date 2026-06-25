# RUNBOOK — what to run, in order (self-serve)

Practical execution steps. Research framing is in `PAPER_PLAN.md`; JJAS design in
`jjas/PLAN.md`. Data → `/storage/raj.ayush/s2s_final_data`; figures → home.
Always: `conda activate s2s-hind`.

---

## ▶ STEP 0 — running now (unattended, SLURM job 56226)
JJAS-2019 end-to-end: FuXi extract → verify (FuXi+ECMWF, dual basis) → all plots.
```bash
squeue -u raj.ayush | grep jjas2019                  # is it running?
tail -f final_analysis/jjas/logs/jjas2019_56226.out  # live progress
```
When done (~2-3 h), outputs:
- CSVs → `/storage/raj.ayush/s2s_final_data/jjas/results_2019_1.5deg/`
- figs → `final_analysis/jjas/plots/figs/` and `final_analysis/jfm2026/plots/figs/`

---

## ▶ STEP 1 — eyeball the JJAS-2019 result
```bash
RES=/storage/raj.ayush/s2s_final_data/jjas/results_2019_1.5deg
python -c "import pandas as pd; d=pd.read_csv('$RES/skill_deterministic.csv'); \
d=d[(d.scale=='weekly')&(d.region=='All India')]; \
print(d.pivot_table(index=['variable','model','clim_basis'],columns='lead',values='pcc').round(2))"
```
Look for: FuXi vs ECMWF TP/Z500 PCC by week, and the **era5 vs model_own** gap.

---

## ▶ STEP 2 — scale to all years (2002–2019)  [the robust benchmark]
First extract FuXi for the other years (slow; run per year, disk-safe):
```bash
cd final_analysis/jjas
for Y in 2002 2003 ... 2019; do                      # any subset to start
  python preprocess_fuxi.py --start ${Y}0601 --end ${Y}0831
done
```
Then run each year (config takes --year), or add a multi-year init list. Quick loop:
```bash
for Y in 2002 ... 2019; do python run_verify.py --year $Y --workers 13; done
```
Concatenate the per-year CSVs for pooled 18-yr skill (each row carries init_date →
year). Tip: start with 3 years (e.g. 2017-2019) before committing the full extract.

---

## ▶ STEP 3 — make the paper / meeting figures
```bash
cd final_analysis/jfm2026/plots
python make_plots.py   --results <RESULTS_DIR>        # full suite
python meeting_figs.py --results <RESULTS_DIR>        # 4 slide-ready
cd ../../analysis && python monthwise.py --results <RESULTS_DIR> --out <fig_dir>
cd ../jjas/plots && python monsoon_maps.py --years 2002-2019   # per-year India maps
```

---

## ▶ STEP 4 — the NOVEL science to build (makes it high-impact)
Not yet coded — these are the differentiators (see PAPER_PLAN §5):
1. **MISO/BSISO index** module: project daily TP (and U850) anomalies onto BSISO
   EOFs → bivariate index; score amplitude + phase error vs lead. *(headline fig)*
2. **Active/break spell** skill: standardized IMD-region rainfall anomaly →
   active(>+1σ)/break(<−1σ) spells → hit-rate / lead-time of spell onset.
3. **Intraseasonal variance ratio** vs lead (quantify AI damping).
> Ask me (in a fresh turn) to build module 1 or 2; each is self-contained, lives in
> `final_analysis/analysis/`, and tests on the ECMWF pilot + ERA5 (no FuXi needed).

---

## ▶ STEP 5 — write-up
Target npj Clim Atmos Sci / GRL. Results section already has: scorecard,
skill-vs-lead, dual-basis, month×region, monsoon maps. Add MISO + active/break →
draft. Caveats list is in PAPER_PLAN §9.

---

### Cheat-sheet: where things live
| | path |
|---|---|
| code | `final_analysis/` (home, git) |
| data (CSVs, FuXi compact) | `/storage/raj.ayush/s2s_final_data/` |
| figures | `final_analysis/**/plots/figs/` (home) |
| JFM results | `final_analysis/jfm2026/results_{1.5,0.5}deg/` |
| pipeline docs | `README.md`, `jjas/PLAN.md`, `PAPER_PLAN.md` |
