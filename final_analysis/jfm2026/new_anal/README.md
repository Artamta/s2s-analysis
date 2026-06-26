# new_anal — extra spatial + SST analysis for the JFM2026 meeting

The `jfm2026/` pipeline scores SPIRE / FuXi / ECMWF vs ERA5 and writes **scalar
skill CSVs** (PCC, CRPSS, Brier). Those answer *how much* skill but not *where*.
This package adds the spatial and ocean views that make the comparison legible in
a meeting. Everything reuses the verified `core/` + `adapters_jfm.py` machinery, so
the numbers match the pipeline — only the rendering is new.

All figures land in `new_anal/figs/`.

## Figures

| script | figure(s) | what it shows |
|---|---|---|
| `a1_spatial_bias.py` | `A1_spatial_bias_<VAR>_W{1,4}.png` | Composite mean-state map (ERA5 truth vs each model) **and** the bias map (model − ERA5), per variable, at Week 1 and Week 4. Reveals *where* each model is wet/dry, warm/cold. e.g. FuXi runs ~4.7 K **cold** over India in week 1; SPIRE is near-unbiased. |
| `a2_skill_maps.py` | `A2_skill_map_<VAR>.png` | Grid-point anomaly correlation (ACC) across the 13 inits, one row per model × columns for Weeks 1,2,3,4,6. Shows *where* and *how far out* each model is skilful (area-mean ACC annotated per panel). |
| `a3_region_profiles.py` | `A3_region_profiles_<VAR>.png`, `A3_region_scorecard_<VAR>_MME.png` | Skill (PCC vs lead) split across the 4 IMD homogeneous regions, plus a region × week scorecard. South Peninsula is the hardest region for rainfall; East/NE India the most predictable. |
| `a4_sst.py` | `A4_S1_sst_bias_W{1,4}.png`, `A4_S2_sst_basin_skill.png`, `A4_S3_sst_skill_map.png` | **Sea-surface temperature**: FuXi SST forecast vs ERA5 SST (ARCO-ERA5 truth) over the north Indian Ocean. Spatial bias, basin-mean ACC/RMSE vs lead (Arabian Sea, Bay of Bengal, Equatorial Indian Ocean), and a grid-point SST skill map. Only FuXi carries an SST channel. |

## Run

```bash
cd new_anal
python a1_spatial_bias.py            # TP, T2M, Z500 ; Weeks 1 & 4
python a2_skill_maps.py              # grid-point ACC, Weeks 1,2,3,4,6
python a3_region_profiles.py         # IMD-region skill from results_1.5deg CSVs
python a4_sst.py                     # SST (first run fetches ERA5 SST from ARCO)
```

Useful flags: `--weeks 1 3 6`, `--vars TP T2M`, `--dgrid 1.5` (maps default to a
finer 0.5° grid for clarity; the skill is identical at the 1.5° fair grid).

## Data notes

- **Maps** are drawn on a 0.5° grid (`masks/imd_region_masks_0.5deg.nc`); the
  verified skill numbers use the 1.5° fair grid.
- **SST truth** = ERA5 hourly SST from the public ARCO-ERA5 zarr, reduced to a
  daily mean from 4 synoptic hours over exactly the days the forecasts need, then
  cached to `era5_sst_jfm2026.nc` (re-download with `--no-cache`).
- **SST anomalies** are taken against a smooth 31-day rolling-mean seasonal cycle
  (a day-of-year climatology is degenerate with a single season of data).
- **ECMWF has no T2M** forecast on disk, so it is absent from the T2M panels (by
  design — same as the pipeline).
