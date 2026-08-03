# India S2S Benchmark v1: Data and Preprocessing Methods

## Scope and analysis boundary

This workflow constructs a versioned, analysis-ready forecast archive. It does
not compute forecast skill, select a best model, or define the final
climatology. Those steps must consume a pinned catalog produced here. Keeping
preprocessing separate from verification prevents changes in file discovery,
units, grids, ensemble membership, or lead definitions from silently changing
ACC, MAE, RMSE, or probabilistic scores.

The frozen source inventory is
`deliverables/s2s_data_inventory_20260803/inventory.json`. The canonical output
root is
`/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1`.
Raw inputs are read-only.

## Forecast experiments currently registered

Counts below describe the inventory frozen on 2026-08-03. All registered dense
forecasts have 42 daily leads. ERPAS files have 33 daily leads. The varying
ECMWF membership (51 or 101 total members, depending on date) is retained, not
subsampled.

| Archive label | Years | Initializations | Nominal cadence | Members per initialization | Standardized fields |
|---|---:|---:|---|---:|---|
| ECMWF operational | 2020–2025 | 621 | Monday/Thursday | 51 or 101 | TP, T2M |
| UKMO operational | 2020–2025 | 621 | Monday/Thursday | 4 | TP, T2M |
| CMA operational | 2020–2025 | 621 | Monday/Thursday | 4 | TP, T2M |
| NCEP operational | 2020–2025 | 621 | Monday/Thursday | 16 | TP, T2M proxy |
| CNRM operational | 2020–2025 | 269 | Thursday | 25 | TP, T2M |
| FuXi-S2S strict-00Z | 2020–2025 | 621 | Monday/Thursday | 50 | TP, T2M |
| DLESyM v0 | 2020–2024 | 517 | Monday/Thursday | 1 | TP, T2M |
| DLESyM v1 | 2020–2024 | 517 | Monday/Thursday | 4 | T2M |
| NeuralGCM v1 | 2020–2024 | 517 | Monday/Thursday | 10 | TP |
| FCN3 v1 | 2020–2024 | 516 | Monday/Thursday | 3 | T2M |
| ERPAS provider mean | 2023–2025 | 148 | Wednesday | one stored mean; source count 4–20 depending on initialization | TP, India 0.5° TP sensitivity, surface temperature, pressure-level GH |

FCN3 is missing the 2021-03-08 initialization because its ERA5 initial
condition was non-finite; 516 of 517 expected dates are available. ERPAS 0.5°
India precipitation is absent for 2024-12-18, while global ERPAS precipitation
is present. DLESyM, NeuralGCM, and FCN3 2025 are not yet included in this frozen
inventory and can be appended under new experiment IDs after generation.

The NCEP temperature field is the daily mean of four six-hourly
`(mx2t6 + mn2t6) / 2` estimates. It is stored as `t2m_proxy` and must never be
pooled with native daily-mean T2M without an explicitly reported sensitivity
test. ERPAS surface temperature is an instantaneous daily sample and is stored
as `tsfc`, not T2M. ERPAS geopotential height is stored at its seven source
pressure levels using the `pressure_hpa` coordinate.

## Standard data model

Each immutable Zarr v2 store represents one
`model/experiment/variable/grid/year` combination. Its principal array is
`forecast(init, member, lead_day, [pressure_hpa], latitude, longitude)`. It also
contains:

- `ensemble_mean`, `ensemble_std` (`ddof=0`), and
  `ensemble_member_count`;
- `member_available(init, member)`, allowing membership to change with date;
- complete-block `forecast_weekly_mean`, `ensemble_mean_weekly`, and associated
  spread/count fields;
- for precipitation, `forecast_weekly_total` and
  `ensemble_mean_weekly_total`;
- `valid_time`, `forecast_period_start`, and `forecast_period_end`.

Only complete, non-overlapping seven-day blocks are emitted. Lead day 1 is the
24-hour interval ending one day after initialization. Weekly lead 1 therefore
contains lead days 1–7. Daily precipitation is in mm day-1; a seven-day mean
remains in mm day-1 and a seven-day total is in mm. Temperature is in degrees
Celsius and geopotential height in geopotential metres (gpm). Units and
temporal-statistic attributes are written on every primary and derived field.

Raw member fields are retained whenever available. Deterministic models retain
a member axis of length one. ERPAS is explicitly marked `mean_only` because the
provider supplied an unweighted mean and the underlying members are unavailable;
it is not treated as a one-member ensemble for probabilistic verification. The
source count is read from `GRIB_totalNumber` for every initialization and stored
as `source_ensemble_size(init)`, rather than hard-coded. Across the 148 current
ERPAS initializations, the counts are 4 (9 dates), 8 (1), 12 (1), 19 (1), and
20 (136) for the global fields. The India 0.5° TP sensitivity product differs
on 2023-01-25 (4 rather than 8); storing the count per variable/init preserves
that distinction.

The archive layout is:

```text
india_s2s_benchmark_v1/
├── inventory/                         # frozen availability table and known gaps
├── spatial/spatial_support.zarr       # IMD fractions and canonical area weights
├── forecasts/<model>/<experiment>/<variable>/<grid>/<year>.zarr
├── manifests/<model>/<experiment>/<variable>/<grid>/<year>.json
├── indexes/catalog_full_<content-id>.json
├── indexes/init_index_full_<content-id>.parquet
├── pilots/<pilot-id>/                 # isolated one-week processing gates
├── runtime/<content-id>/              # code + inventory used on compute nodes
└── logs/                              # SLURM output and dependency preflight
```

The JSON catalog is the machine-readable entry point for evaluation. Each
record supplies model, experiment, variable, grid, year, store path, manifest
path, distribution representation, units, temporal statistic, initialization
count, QC status, and Zarr metadata checksum. The Parquet initialization index
provides one row per available initialization so evaluation code can join
models and observations by date without scanning storage directories.

## Unit and temporal transformations

- Physics-model precipitation supplied as accumulation since initialization is
  converted to disjoint daily increments by first difference. NCEP six-hourly
  products are sampled at 24-hour endpoints before differencing.
- FuXi-S2S precipitation is converted from mm h-1 to mm day-1 by multiplication
  by 24.
- Kelvin temperature fields are converted to degrees Celsius by subtracting
  273.15.
- ERPAS precipitation is supplied as disjoint 24-hour kg m-2 accumulations;
  1 kg m-2 is treated as 1 mm, reported as the equivalent daily mean rate.
- No suspicious finite value is clipped or silently replaced. In particular,
  small negative precipitation increments created by accumulated-field
  differencing are preserved and counted in QC metadata.

## Spatial grids and remapping

The common verification grid is `india_1p5_27x27_v1`: 27 latitudes from 39°N
to 0° at 1.5° spacing and 27 longitudes from 60°E to 99°E at 1.5° spacing.
Physics-model and retained AI outputs already on this grid are not interpolated.

ERPAS is retained both on a native India-plus-halo subset (2° beyond the
verification box) and on the common grid. Precipitation uses first-order
spherical cell-overlap conservative averaging with finite-value normalization.
Instantaneous surface temperature and geopotential height use bilinear
interpolation. The invariant remapping method is stored in Zarr global
attributes, while initialization-specific audit values are stored in the JSON
manifest. The optional ERPAS India 0.5° precipitation product is a sensitivity
dataset and has a distinct experiment ID.

## India support and spatial weights

`spatial/spatial_support.zarr` is built once from the existing Survey of
India-derived four-region IMD mask dataset (Pai et al., 2014 metadata). Each
binary source mask is mapped to the common grid as a spherical overlap fraction.
The store contains each regional fraction, their union (`india_fraction`), exact
spherical `cell_area_km2`, and the canonical
`india_area_weight_km2 = cell_area_km2 × india_fraction`. The source file SHA256
and output metadata SHA256 are recorded. Verification scripts must use these
stored weights rather than independently re-rasterizing India.

## Quality control and immutability

For every initialization, the manifest records source paths, shape, members,
source ensemble size where applicable, lead count, finite/non-finite counts,
minimum, maximum, negative-TP count, and initialization-specific preprocessing
audit attributes. The yearly Zarr global metadata is finalized with the complete
set of source paths only after every initialization has been appended.
Each completed store is reopened and checked for:

1. required variables and consolidated Zarr metadata;
2. ensemble mean and population standard deviation reproduced from members;
3. the precipitation identity `weekly_total = 7 × weekly_mean`;
4. existence of every recorded source file; and
5. a SHA256 checksum of `.zmetadata`.

Writes occur in a same-filesystem `.incomplete-<id>` directory and are renamed
to the final store only after successful read-back. Completed destinations and
manifests are immutable. Interrupted directories remain visible for diagnosis.
Catalog IDs are hashes of catalog content (excluding generation time), so an
identical rerun returns the same catalog. Paper analyses must pin both catalog
filename and SHA256.

## Parallel execution and failure recovery

SLURM uses the `gpu_prio` partition but requests no GPU (`--gres` is absent).
Parallelism is across independent archive stores, not within one file:

`one array element = one model × variable × grid × year`.

The pilot permits 24 simultaneous one-CPU elements, giving 24 effective Python
workers. Full production defaults to the same 24-worker cap and accepts
`S2S_ARRAY_CONCURRENCY=1..32`. One writer owns each destination, preventing
concurrent Zarr corruption. BLAS/OpenMP thread counts are fixed to one so a
worker cannot oversubscribe a node. A final validation/indexing job runs only
after every array element succeeds.

Code and the frozen inventory are copied into a content-addressed runtime below
the storage archive because `/home` is not mounted on every compute node. Every
task runs an import/version preflight. The node-portability pilot revealed that
some nodes had no system `numcodecs`; the fix was to pin `numcodecs==0.15.1`
inside the archive-local dependency bundle. Reruns are idempotent: completed
stores are validated and reused, while only missing tasks are built.

## Pilot gate and subsequent evaluation

The seven-day pilot uses 2023-06-29 for dense systems and 2023-06-28 for ERPAS.
Its purpose is adapter, metadata, grid, member, storage, and scheduler validation;
it is not a forecast ranking. Full production should begin only after all 25
pilot stores, the spatial support, the validation report, and the deterministic
catalog pass.

The later verification stage should be a separate versioned workflow. The
planned primary ACC comparison uses a five-year model climatology and a 30-year
observational climatology, with those exact periods and leave-one-out policy to
be fixed before calculation. Deterministic metrics (ACC, MAE, bias, RMSE) should
use ensemble means where real members exist. Probabilistic metrics should only
use experiments retaining member fields; ERPAS `mean_only` output is excluded
from CRPS, spread-skill, rank-histogram, and reliability calculations unless raw
members become available.
