# India S2S Benchmark v1: Publication-Pilot Audit

## Final gate

The final preprocessing pilot is `pilot_20230628_29_week1_v4`. It uses
2023-06-29 for the dense systems and 2023-06-28 for ERPAS, with seven daily
leads. It is an adapter and archive-integrity test, not a model-skill result.

- SLURM partition: `gpu_prio`
- GPU request: none
- Array job: `83635`
- Finalizer job: `83636`
- Array concurrency: 24 one-CPU workers
- Content-addressed runtime: `runtime/89e306a5078070bf`
- Array result: 25 of 25 tasks completed with exit code 0
- Finalizer result: completed with exit code 0
- Completed stores: 25
- Incomplete stores: 0
- Catalog records / initialization-index rows: 25 / 25

The 25 stores comprise 21 common-grid products and four ERPAS native-India
products. Model store counts are ECMWF 2, UKMO 2, CMA 2, NCEP 2, CNRM 2,
FuXi-S2S 2, DLESyM-v0 2, DLESyM-v1 1, NeuralGCM 1, FCN3 1, and ERPAS 8.
Variable store counts are TP 12, T2M 8, NCEP T2M proxy 1, ERPAS surface
temperature 2, and ERPAS geopotential height 2.

## Pinned artifacts

- Catalog ID: `d3ce84adc9f1a7ca`
- Catalog SHA256:
  `bd04daff51e9e3ce1f7f8845fa7d4eb755df2541900c2e7e97037b580c88f639`
- Initialization-index SHA256:
  `482adb83ea74be84330206af81ae032860cd23df5b12e281565f8700197caab5`
- Spatial-support Zarr metadata SHA256:
  `07bb0e60a396a6056df0cea9c3b96861aeb5fe0f1db9640173b9e166306cfbe4`
- IMD mask source SHA256:
  `4bd16e7c6f8a2d3009ac294f6bd4be83a802b290691c272faa2d79403dcb0596`

Re-running catalog construction returned the same catalog ID and reused the
immutable JSON/Parquet pair.

## Scientific and metadata checks

Every store passed read-back validation of ensemble mean, population standard
deviation, member counts, complete-week aggregation, source-path existence,
and consolidated metadata. Derived-field unit contracts were independently
checked:

- TP daily and weekly mean: mm day-1;
- TP weekly total: mm;
- T2M, T2M proxy, and surface temperature: degrees Celsius;
- pressure-level geopotential height: gpm.

All ERPAS pilot fields record `source_ensemble_size(init)=20`, read directly
from the pilot GRIBs. ERPAS is marked `mean_only`, not as a probabilistic
one-member ensemble. A scan of all 148 ERPAS initializations found global-field
source counts of 4 (9 dates), 8 (1), 12 (1), 19 (1), and 20 (136). The India
0.5° TP sensitivity product differs only on 2023-01-25, when it represents four
sources while the global fields represent eight.

Small negative daily TP increments were present after differencing accumulated
physics-model fields: ECMWF minimum -0.015625 mm day-1, CMA -0.0078125, CNRM
-0.00390625, NCEP -0.0078125, and UKMO -0.0000152588. They were preserved and
counted, never clipped. ERPAS, FuXi-S2S, DLESyM-v0, and NeuralGCM pilot TP had
no negative values.

The spatial-support store contains 174 common-grid cells with nonzero fractional
India support and a total fractional area weight of 3,358,619.805 km2. This is
the canonical weight store for later ACC/MAE calculations.

## Problems found and resolved

1. Some `gpu_prio` compute nodes do not mount `/home`. The solution was to run
   from a content-addressed code-and-inventory snapshot under shared storage.
2. Some nodes had no system `numcodecs`. Zarr 2.18.7 and numcodecs 0.15.1 are
   now pinned in the archive-local `_deps`; every task logs an import/version
   preflight before reading data.
3. Four allocated CPUs per array element did not create four Python workers.
   The array was changed to 24 simultaneous one-CPU elements, producing 24
   effective independent store workers without BLAS oversubscription.
4. A fixed ERPAS source-member assumption was invalid. `GRIB_totalNumber` is
   now read per variable and initialization and stored along the `init` axis.
5. Xarray append would otherwise leave only the first source path in yearly
   global metadata. The writer now finalizes the complete source-path set after
   all initializations are appended; a two-initialization regression test covers
   this behavior.
6. Catalog generation time originally affected catalog identity. Catalog IDs
   now hash scientific content only, so identical content has a stable ID.

Earlier `v1`–`v3` pilot directories are development evidence only. Downstream
evaluation must use the final v4 catalog above. Full multi-year production was
not submitted as part of this pilot gate.
