# FCN3, DLESyM, and NeuralGCM Six-Year Inference Runbook

Status: the broader six-year multi-model plan below remains a planning record.
The currently approved NeuralGCM production run is the separately frozen
2020-2024 experiment documented in Section 3.1.

## 1. Objective

Run public, pretrained AI weather models from the same ERA5 analysis source on
the exact six-year physics-model initialization calendar:

- calendar: `config/all_season_dates_2020_2025.csv`
- checked calendar SHA256:
  `2b5a305a6cde6ef86bd60b99884dc87fd929fbc22eeb67798fbd78d0cf2713f3`
- years: 2020-2025 inclusive
- cases: 621 total (`105 + 104 + 104 + 104 + 100 + 104`)
- forecast reference time: 00 UTC on initialization date D
- forecast length: 42 complete UTC days
- requested verification fields: total precipitation (`tp`) and 2 m
  temperature (`t2m`)
- common verification domain: latitude 39 to 0 deg north and longitude 60 to
  99 deg east, on the existing 1.5 deg 27 x 27 grid
- common final units: `tp` in `mm day-1`; `t2m` in `degC`

The same *source analysis* must be used, but each checkpoint must receive its
own documented input variables, history, grid, normalization, and forcing
policy. Feeding an identical tensor to all three models would be incorrect.

## 2. Critical Feasibility and Naming Result

The released systems do not all provide both requested fields in the same way.
In particular, FCN3 has no native precipitation channel, but NVIDIA publishes a
compatible precipitation diagnostic. NeuralGCM's public precipitation
checkpoint has no 2 m temperature decoder.

| System to run | `t2m` path | `tp` path | Production use in this project |
|---|---|---|---|
| FCN3 v1 plus AFNO_DX_TP-V1-ERA5 | Native FCN3 channel | Official separate AFNOv2 diagnostic | T2M and diagnostic TP, 10 native stochastic samples |
| DLESyM v0 ISCCP-ERA5 | Native prognostic channel | Official paired DLESyM diagnostic | T2M and diagnostic TP, one deterministic member |
| DLESyM V1 ERA5, optional sensitivity | Native prognostic channel | Unavailable | T2M-only 16-checkpoint ensemble |
| NeuralGCM v1 stochastic precipitation 2.8 deg | Unavailable | Checkpoint precipitation diagnostic | TP only, 10 native stochastic samples |

The scientifically clean comparison is therefore field-wise:

- T2M panel: FCN3, DLESyM, FuXi, and physics models. NeuralGCM is unavailable.
- TP panel: `FCN3+AFNOv2`, `DLESyM-v0+TP-diagnostic`, NeuralGCM, FuXi, and
  physics models.
- Any figure, table, directory, or manifest must preserve those compound names.
  Calling the AFNOv2 output simply "FCN3 precipitation" would wrongly imply
  that precipitation is part of the FCN3 checkpoint.

Do not write an all-model file filled with NaNs for unavailable fields. Omit
the field and record `field_availability` in the per-date manifest.

### 2.1 FCN3 TP contract

FCN3 has 72 state channels and no precipitation channel. Do not use the older
`PrecipitationAFNO` checkpoint, whose relative-humidity input contract does not
match FCN3. Use `PrecipitationAFNOv2` instead:

- AFNOv2 predicts `tp06` in meters for the preceding interval `[t-6 h, t]` on
  the 720 x 1440 south-pole-excluded 0.25 deg grid.
- It needs 20 atmospheric variables. FCN3 predicts 19 directly and lacks only
  surface pressure (`sp`).
- Earth2Studio's `DerivedSurfacePressure` computes the missing `sp` in Pa from
  FCN3 geopotential and temperature at all 13 pressure levels, using the FCN3
  package's `orography.nc` surface geopotential.
- Earth2Studio 0.16.0 contains an FCN3 diagnostic-wrapper contract test that
  composes exactly `FCN3 -> DerivedSurfacePressure -> PrecipitationAFNOv2`.
  This establishes software/coordinate compatibility; it does not remove the
  need to validate diagnostic skill on long-lead FCN3 states.

The released diagnostic was trained on ERA5 atmospheric states, not jointly
with FCN3. Distribution shift can therefore grow with lead time. Save and score
the result as `FCN3+AFNOv2`, record both checkpoint hashes, and report native
FCN3 T2M separately from diagnostic TP. The production pilot must compare
actual ERA5 `sp` with derived `sp`, compare AFNOv2 output using those two `sp`
sources at initialization, and inspect TP behavior by lead week.

### 2.2 Why NeuralGCM T2M is unavailable

The public NeuralGCM precipitation checkpoint decodes pressure-level
temperature and precipitation diagnostics. It does not decode 2 m temperature.
Neither 1000 hPa temperature nor the lowest model level is a valid substitute
for T2M. A learned near-surface diagnostic would be a separate project and
must be evaluated and labeled separately.

### 2.3 Which DLESyM release to use

There are two materially different public DLESyM releases:

- DLESyM V1 ERA5 is the S2S ensemble release. It has four atmosphere and four
  ocean checkpoints, giving 16 checkpoint pairs, but it does not predict TP.
- DLESyM v0 ISCCP-ERA5 is the climate-model release. Its package includes the
  official two-state `tp06` diagnostic. Use this exact pair when both TP and
  T2M are required.

Do not combine V1 T2M and v0 TP into one unlabeled "DLESyM" forecast. They are
different checkpoints, variables, and training configurations.

If the project later requires both fields from every model, stop this workflow
and create a separate diagnostic-training protocol. That protocol needs a
training period, held-out years, calibration, uncertainty propagation, and new
model names. It cannot be added as an undocumented post-processing shortcut.

## 3. Current Local Status

- `model-runs/fcn3` and `model-runs/Deylsm` do not yet contain production
  implementations. The NeuralGCM TP production runner is implemented and
  validated separately under the five-year contract in Section 3.1.
- The final storage model-run tree contains FuXi plus the NeuralGCM checkpoint,
  launch-gate output, and one-case 10-member benchmark.
- `/home/raj.ayush/.conda/envs/fcn3run` must not be reused. Its previous logs
  show incompatible PhysicsNeMo/PyTorch custom operators, Zarr modifications,
  and a missing torch-harmonics attention CUDA operator.
- `/home/raj.ayush/.conda/envs/earth2` can import the current Earth2Studio source
  only when its Conda `lib` directory is placed first in `LD_LIBRARY_PATH`. It
  is useful for a pilot, but it is not a frozen production environment.
- `/home/raj.ayush/.conda/envs/neuralgcm` has NeuralGCM 1.2.2, JAX/JAXLIB 0.10.2,
  and the CUDA 12 plugin. The 42-day one-member production gate and 10-member
  timing benchmark both passed on an A100 before the five-year launch.
- The existing local WeatherBench2 store under `era5-daily` is daily data. It
  cannot initialize FCN3, DLESyM, or NeuralGCM, which require instantaneous
  states and/or subdaily history. Use the hourly ARCO-ERA5 source.

Before implementation, standardize the empty typo directory `Deylsm` to the
lowercase name `dlesym`. Do not maintain two spellings in code or storage.

### 3.1 Current NeuralGCM production run: 2020-2024

The executable NeuralGCM production contract is narrower than the legacy
six-year plan in this document:

- calendar: `config/all_season_dates_2020_2024.csv`
- calendar SHA256:
  `ab1b82b215f50ba3c52654e242675905dce2b4f2aa5bf93c2e0d08f5b9131b5a`
- cases: 517, with year counts `105/104/104/104/100`
- product: TP only, 42 daily periods in `mm day-1`
- ensemble: 10 deterministic, model-native stochastic members per date
- partition: `GPU-AI_prio`, with at most four array tasks running concurrently
- output root:
  `/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2020_2024_ens10`

Submit the frozen array with:

```bash
sbatch model-runs/neural-gcm/slurm/run_neuralgcm_tp_2020_2024_ens10.sbatch
```

Each array index is materialized through the frozen CSV. IC and forecast files
are written atomically and accompanied by SHA256 manifests. Re-running the
same array is the supported resume operation: a date is skipped only when both
its final file and manifest validate. Partial or invalid final products fail
for inspection rather than being overwritten.

Monitor a submitted array and count completed manifests with:

```bash
squeue -j JOB_ID
find /storage/raj.ayush/s2s_final_data/final_iteration/model-runs/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2020_2024_ens10/manifests -name '*.json' -type f | wc -l
```

Logs are written under `model-runs/neural-gcm/logs` as
`production_ens10_JOBID_ARRAYINDEX.out` and `.err`. The measured one-case A100
inference time was 161.6 seconds through member 5 and 220.6 seconds through
member 10. The expected full run is about 47 GPU-hours before queue delays.

## 4. Frozen Model and Source Versions

Every production manifest must contain the complete URI or code commit, not
only a friendly model name.

### 4.1 Earth2Studio

Pin Earth2Studio `0.16.0`. Its current official installation guide supplies the
dependency commits below:

- Earth2Studio: `https://github.com/NVIDIA/earth2studio.git@0.16.0`
- torch-harmonics:
  `a632ca748a12bd9f74dbc1e00653317810991f74`
- Makani: `b38fcb2799d7dbc146fa60459f3f9823394a8bf1`
- earth2grid: `11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3`

### 4.2 Model packages

- FCN3 package used by the pinned wrapper:
  `hf://nvidia/fourcastnet3@76ef0c60237e458b33196ba027134e27f3fc4538`
- AFNOv2 precipitation diagnostic used with FCN3:
  `ngc://models/nvidia/earth-2/afno_dx_tp-v1-era5@v0.1.0`
- DLESyM v0 ISCCP-ERA5 package used by the pinned wrapper:
  `hf://nvidia/dlesym-v0-isccp-era5@924b2d62644ef61289dd960e018f60d6e067bfca`
- Optional DLESyM V1 package:
  `hf://nvidia/dlesym-v1-era5@9dbcdb83706702ac3b7d93f5dad5e535abc2fb72`
- NeuralGCM package: `neuralgcm==1.2.2`
- NeuralGCM checkpoint:
  `gs://neuralgcm/models/v1_precip/stochastic_precip_2_8_deg.pkl`

After downloading, write `sha256sum` output for every checkpoint artifact to a
model-package manifest. The URI pin and local content hash must both be kept.

### 4.3 ERA5 source

Use Google's hourly ARCO-ERA5 store:

```text
gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
```

Pin the source URL, the requested variables/times, Earth2Studio or Dinosaur
adapter version, and a deterministic request hash for every initialization.

## 5. Common Scientific Contract

### 5.1 Initialization and information cutoff

- Forecast case D is always labeled `forecast_reference_time = D 00:00 UTC`.
- FCN3 uses the ERA5 state at D 00 UTC.
- DLESyM consumes the official unified 6-hourly history from D-48 h through D
  00 UTC. Its atmosphere uses D-18, D-12, D-6, and D; its ocean component uses
  D-48 and D. The wrapper's full input coordinate includes all 6-hourly states
  between D-48 and D.
- NeuralGCM uses the ERA5 atmospheric state at D 00 UTC. Use the official
  NeuralGCM 24-hour backward shift to select the initialization SST and
  sea-ice forcing. Then apply this project's no-future-information policy by
  persisting that D-1 forcing for the 42-day forecast. The persistence step is
  a comparison-policy choice, not a claim that the official example prescribes
  fixed 42-day boundary conditions.
- FuXi's D-2/D-1 daily-state rule remains FuXi-specific. Do not apply that daily
  input rule to these instantaneous-state models.

For every primary run, set `information_cutoff_time = D 00:00 UTC`. A
NeuralGCM experiment using observed future SST/sea ice is allowed only as a
separately named prescribed-boundary sensitivity, never as the primary
forecast.

### 5.2 Lead periods

- Produce six-hourly model states through exactly +1008 h (42 days).
- Lead day 1 is the half-open interval `[D, D+1 day)`.
- Lead day `k` is `[D+k-1 days, D+k days)`.
- `valid_time` labels the period end, matching the strict FuXi/physics
  contract; the explicit bounds, not that point label, define the statistic.
- Store lead days 1-42 and their start/end bounds.
- Form weeks 1-6 only after daily products exist. Each week is seven complete,
  non-overlapping UTC days.

### 5.3 Daily fields

- FCN3 and DLESyM T2M are six-hourly instantaneous fields, whereas the physics
  providers and FuXi supply interval daily means. A six-hourly forecast cannot
  reconstruct the exact continuous daily mean. Use one declared approximation:
  trapezoidal integration of the five boundary/synoptic states at +0, +6, +12,
  +18, and +24 h for day 1. Equivalently, use weights
  `[1/8, 1/4, 1/4, 1/4, 1/8]`, then shift the window by 24 h for each later
  day. Record `temporal_statistic = 6h_piecewise_linear_trapezoidal_mean`.
  This aligns the integration bounds exactly and avoids arbitrarily assigning
  the +24 h endpoint wholly to either adjacent day.
- FCN3+AFNOv2 `tp06` is precipitation accumulated over the six hours ending at
  the diagnostic valid time. Discard the diagnostic evaluated on the initial
  D 00 state because it covers `[D-6 h, D]`; day 1 is the sum of diagnostics at
  +6, +12, +18, and +24 h, multiplied from meters to millimeters.
- DLESyM `tp06` is precipitation accumulated over the six hours ending at the
  diagnostic valid time. Sum four consecutive `tp06` fields and multiply
  meters by 1000 to obtain `mm day-1`.
- NeuralGCM `precipitation_cumulative_mean` is meters accumulated from the
  simulation start. Difference the values at each pair of daily boundaries
  and multiply by 1000. Keep the +0 h value so day 1 is `cum(+24)-cum(0)`.
- Convert T2M from K to degC only after the daily mean.
- In the pilot, apply the same six-hour trapezoidal sampler to hourly ERA5 and
  compare it with ERA5's exact 24-hour mean. Report this temporal-sampling
  error alongside model skill; do not tune the weights by forecast score.
- Never clip TP silently. Fail if negative increments exceed the numerical
  tolerance established in the pilot; record any tolerance-based correction.

### 5.4 Grid and remapping

Keep the model-native grid through rollout and temporal aggregation. Regrid the
daily fields once, after aggregation:

- T2M: bilinear interpolation to the canonical 1.5 deg grid.
- TP: first-order conservative remapping to the canonical 1.5 deg grid.

Generate and freeze remapping weights once per native grid. Save the weight
file hash and test area-integrated precipitation before and after remapping.
Earth2Grid's default HEALPix-to-lat/lon operation is bilinear; do not call that
operation "conservative". If it is used for a pilot, retain native-grid TP and
replace it with a verified conservative HEALPix remap before production.

NeuralGCM's 2.8 deg Gaussian output is coarser than the 1.5 deg target. The
target grid is a comparison grid, not an increase in physical resolution.

### 5.5 Ensemble policy

- FCN3: 10 model-native stochastic samples in the primary run.
- Apply AFNOv2 independently to every FCN3 member and valid time. Do not
  diagnose only the ensemble mean; that would erase nonlinear member spread.
- NeuralGCM: 10 independent model-native stochastic samples in the primary
  run.
- DLESyM v0 plus precipitation diagnostic: one deterministic member.
- Optional DLESyM V1: all 4 x 4 = 16 atmosphere/ocean checkpoint pairs, with no
  added noise or external perturbation wrapper.

Ten stochastic members match the existing benchmark's matched-member view and
keep the first production pass tractable. A 50-member FCN3/NeuralGCM
sensitivity can be added later without discarding the first ten members.

Generate seeds from a stable cryptographic hash of
`run_label/init_date/member`; never use Python's process-randomized `hash()`.
Store the integer seed for every member. A rerun with the same environment,
checkpoint, input, and seed must reproduce the pilot within the documented
numeric tolerance.

## 6. Recommended Repository and Storage Layout

```text
model-runs/
  common/
    calendar.py
    output_contract.py
    regrid.py
    audit.py
    tests/
  configs/
    common_2020_2025.json
    fcn3_v1_afnov2_tpdiag_2020_2025_ens10.json
    dlesym_v0_isccp_2020_2025.json
    neuralgcm_precip_2p8_2020_2025_ens10.json
  fcn3/
    scripts/
    slurm/
    tests/
  dlesym/
    scripts/
    slurm/
    tests/
  neural-gcm/
    scripts/
    slurm/
    tests/
```

Use this storage root:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/
```

Recommended immutable run labels:

```text
fcn3/fcn3_v1_afnov2_tpdiag_era5_00z_2020_2025_ens10/
dlesym/dlesym_v0_isccp_era5_tpdiag_00z_2020_2025/
neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2020_2025_ens10/
```

Each run contains:

```text
config/                 frozen JSON and environment lock
weights/                checkpoint manifests and hashes, not duplicate weights
inputs/                 staged-input manifests and completion markers
forecasts/YYYY/         one compact NetCDF per initialization
manifests/YYYY/         one provenance JSON per initialization
logs/prefetch/
logs/inference/
logs/slurm/
audit/
```

Do not overwrite a run in place after changing a model pin, forcing policy,
member count, remapping weights, or temporal definition. Create a new run
label.

## 7. Environment Build

Create separate environments. This prevents FCN3's compiled torch-harmonics
operators from being destabilized by DLESyM/PhysicsNeMo or JAX changes.

### 7.1 FCN3 environment

The following is the pinned form of NVIDIA's current installation recipe. Run
it in a fresh environment with a CUDA compiler compatible with the PyTorch
wheel and the A100 architecture.

```bash
conda create -n fcn3-prod python=3.13 -y
conda activate fcn3-prod
python -m pip install --upgrade pip
export FORCE_CUDA_EXTENSION=1
export MAX_JOBS=8
python -m pip install --no-build-isolation \
  "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@a632ca748a12bd9f74dbc1e00653317810991f74"
python -m pip install \
  "makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"
python -m pip install \
  "earth2studio[fcn3,precip-afno-v2] @ git+https://github.com/NVIDIA/earth2studio.git@0.16.0"
```

The install is not accepted until all of these pass inside an A100 allocation:

```python
import torch
from torch_harmonics.disco import cuda_kernels_is_available
from earth2studio.models.dx import DerivedSurfacePressure, PrecipitationAFNOv2
from earth2studio.models.px import FCN3

assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 79_000_000_000
assert cuda_kernels_is_available()
```

Also load both checkpoint packages, build the nested
`FCN3 -> DerivedSurfacePressure -> PrecipitationAFNOv2` wrapper, and execute
two autoregressive 6-hour steps. Confirm finite T2M and `tp06`, exact variable
order, and the 721-to-720 latitude mapping. An import-only check is
insufficient. Record peak allocated/reserved GPU memory because the combined
pipeline, not FCN3 alone, determines the production request.

### 7.2 DLESyM environment

```bash
conda create -n dlesym-prod python=3.13 -y
conda activate dlesym-prod
python -m pip install --upgrade pip
python -m pip install --no-build-isolation \
  "earth2grid @ git+https://github.com/NVlabs/earth2grid@11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3"
python -m pip install \
  "earth2studio[dlesym] @ git+https://github.com/NVIDIA/earth2studio.git@0.16.0"
```

Inside the job, put the Conda library directory first if `pygrib` reports a
`GLIBCXX` mismatch:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

The smoke test must load `DLESyMv0_ISCCP_ERA5`, its LatLon reference wrapper,
and `DLESyMv0_ISCCP_ERA5Precip`; fetch one initialization; and produce +6 and
+12 h T2M/TP outputs. The native and LatLon paths must also pass the first-block
parity check described below.

### 7.3 NeuralGCM environment

The currently installed combination is a good reproducible starting pin:

```bash
conda create -n neuralgcm-prod python=3.11 -y
conda activate neuralgcm-prod
python -m pip install --upgrade pip
python -m pip install \
  neuralgcm==1.2.2 dinosaur==1.3.6 gcsfs==2026.6.0 xarray==2026.4.0
python -m pip install --upgrade "jax[cuda12]==0.10.2"
```

JAX's pip CUDA wheels should supply their own CUDA libraries. Do not inherit an
unrelated `LD_LIBRARY_PATH` from the Earth2Studio jobs. The GPU gate is:

```python
import jax
import neuralgcm

assert any(device.platform == "gpu" for device in jax.devices())
assert neuralgcm.__version__ == "1.2.2"
```

Load the precipitation checkpoint, encode one state, and unroll two 6-hour
steps. Confirm that `precipitation_cumulative_mean` exists and that
`2m_temperature` does not.

### 7.4 Freeze environments

After a successful GPU smoke test, save:

```bash
python -m pip freeze --all > environment.freeze.txt
conda list --explicit > conda-explicit.txt
```

Record hashes of both files in the run config. Do not upgrade any package
inside a run that already has output.

## 8. Download Checkpoints Once

Before downloading, complete an access and license preflight. Confirm that the
pinned Hugging Face revisions, the NGC AFNOv2 package, and the public NeuralGCM
GCS object are all reachable non-interactively from a compute job. Archive the
governing terms or model-card metadata with the run config: FCN3 is Apache-2.0,
DLESyM v0 is OpenMDW-1.1, and the public NeuralGCM checkpoints are CC BY-SA 4.0.
The checked AFNOv2 NGC card states that the model is ready for commercial use
but does not name a license in its `License/Terms of Use` field, so preserve the
terms presented by NGC at download time and obtain any required institutional
approval. Do not redistribute checkpoint files as part of the forecast output.

Use shared read-only model caches after a single controlled download:

```bash
export MODEL_RUN_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs
export EARTH2STUDIO_MODEL_CACHE="${MODEL_RUN_ROOT}/_cache/earth2studio-models"
export EARTH2STUDIO_DATA_CACHE="${MODEL_RUN_ROOT}/_cache/earth2studio-data"
```

Implement `prefetch_weights.py` so it resolves every file in the pinned FCN3,
AFNOv2, and DLESyM packages, downloads the NeuralGCM checkpoint, computes
SHA256, and writes a JSON inventory with byte sizes. Run it once. Inference
tasks must never race to download the same package. The AFNOv2 NGC package is
about 495 MB compressed, but record the resolved files rather than relying on
that catalog size.

For NeuralGCM, the public checkpoint URL is:

```text
https://storage.googleapis.com/neuralgcm/models/v1_precip/stochastic_precip_2_8_deg.pkl
```

The object was 44,898,125 bytes when checked. The local SHA256 is still the
authoritative content check.

## 9. Stage ERA5 Initial Conditions

Input staging is a separate, resumable phase. It must finish before model
arrays start.

### 9.1 Common rules

1. Read dates only from `config/all_season_dates_2020_2025.csv`.
2. Assert exactly 621 unique, sorted 00 UTC cases and the per-year counts.
3. Query ARCO at the model-requested times and variables.
4. Use the pinned model library's variable adapter and normalization. Do not
   manually guess aliases or units.
5. Write to a temporary path, validate shape/finiteness/times, then atomically
   rename and create a completion marker.
6. Store source URL, request hash, adapter version, exact times, variables,
   dimensions, and file hash.
7. Run prefetch with low concurrency (`%1` initially) so hundreds of workers do
   not hammer ARCO or corrupt a shared cache.

### 9.2 FCN3 inputs

At D 00 UTC stage the exact 72 channels returned by
`FCN3.input_coords()` on the 721 x 1440 0.25 deg grid. The channels are:

- `u10m`, `v10m`, `u100m`, `v100m`, `t2m`, `msl`, `tcwv`
- `u`, `v`, `z`, `t`, and `q` at 50, 100, 150, 200, 250, 300, 400, 500,
  600, 700, 850, 925, and 1000 hPa

Do not add TP, surface pressure, or relative humidity to the FCN3 prognostic
input tensor. The TP diagnostic derives `sp` from each predicted FCN3 state;
using future ERA5 surface pressure would leak verification data. Load static
surface geopotential from the pinned FCN3 package, not from a second mutable
source. One raw float32 state is about 285 MiB before compression; the full
six-year stage is therefore of order 180 GiB before compression.

At pilot time only, also fetch actual ERA5 `sp` at D 00. It is a validation
reference for `DerivedSurfacePressure`, never an input to the production
forecast after initialization.

### 9.3 DLESyM inputs

Use `DLESyMv0_ISCCP_ERA5LatLon.input_coords()` as the source of truth. Fetch its
base ERA5 fields at all nine 6-hourly times from D-48 h through D. The pinned
wrapper must perform:

- `ws10m = sqrt(u10m**2 + v10m**2)`
- `tau300-700 = z300 - z700`
- SST missing-value handling
- lat/lon to HEALPix nside=64 regridding
- the bundled day-of-year ERA5 TTR to ISCCP OLR transformation
- checkpoint normalization

Cache the fetched ARCO chunks first. Use the unmodified LatLon wrapper as the
reference pilot, then make the production stage native HEALPix to avoid a
lat/lon conversion on every 96-hour block:

1. On the 721 x 1440 source grid, compute the derived variables and fill SST
   NaNs exactly as the pinned LatLon wrapper does.
2. Regrid the nine physical, unnormalized history states once with that
   wrapper's pinned Earth2Grid lat/lon-to-HEALPix operator.
3. Store `ttr` rather than precomputed `rlut`; the base
   `DLESyMv0_ISCCP_ERA5(use_ttr=True)` model must apply its bundled day-of-year
   transform once at initialization.
4. Preserve the exact `input_coords()` order and hashes of the package,
   Earth2Grid build, and preprocessing code.

The native float32 stage is roughly 18 MiB per date before compression, versus
hundreds of MiB for the source lat/lon history. Accept this optimization only
after all first-block atmospheric and ocean outputs agree with the public
LatLon path within a frozen numeric tolerance. The reference and optimized
paths must use the same checkpoint, transforms, and initial data.

### 9.4 NeuralGCM inputs

Read `model.input_variables`, `model.forcing_variables`, and
`model.data_coords` from the checkpoint at runtime. For the pinned
precipitation checkpoint they are:

- pressure-level inputs: geopotential, specific humidity, temperature, u wind,
  v wind, specific cloud ice water content, and specific cloud liquid water
  content
- levels: all 37 pressure levels from 1 to 1000 hPa
- forcings: sea-ice cover and sea-surface temperature
- native data grid: 128 x 64 Gaussian grid

Follow the official preprocessing exactly:

1. Select the D 00 UTC atmospheric state.
2. Shift SST and sea ice backward by 24 hours.
3. Conservatively regrid ERA5 to `model.data_coords.horizontal` with Dinosaur's
   `ConservativeRegridder`.
4. Fill remaining NaNs with the official nearest-value utility.
5. Save the compact native-grid atmospheric state and persisted forcing.

The staged file must contain no future SST or sea-ice values.

## 10. Model Runner Requirements

### 10.1 FCN3 plus AFNOv2 runner

1. Load the pinned FCN3 and AFNOv2 packages once per Slurm task and keep them
   resident.
2. Read `Z[0]` from FCN3's pinned `orography.nc` and construct
   `DerivedSurfacePressure` with pressure levels `[50, 100, 150, 200, 250,
   300, 400, 500, 600, 700, 850, 925, 1000]`.
3. Construct nested Earth2Studio wrappers in this order:
   `DiagnosticWrapper(FCN3, DerivedSurfacePressure)`, then
   `DiagnosticWrapper(previous_wrapper, PrecipitationAFNOv2)`. Do not use the
   older `PrecipitationAFNO` class.
4. Assert that AFNOv2 receives exactly `u10m, v10m, t2m, sp, msl, tcwv, u500,
   u850, u1000, v500, v850, v1000, z50, z500, z850, z1000, t500, t850,
   q500, q850`, in the order advertised by `input_coords()`.
5. Load one staged D 00 FCN3 state. For member 0-9, set the recorded stable
   seed, reset FCN3's internal RNG, and create a fresh iterator.
6. The iterator yields the D 00 state first. Retain its diagnostic only for the
   initialization validation and exclude it from forecast totals because it
   represents `[D-6 h, D]`.
7. Run 168 autoregressive 6-hour steps to +1008 h. At every step extract native
   FCN3 `t2m` and AFNOv2 `tp06`; do not retain all global channels after the
   diagnostic finishes.
8. For each day, apply the five-state trapezoidal T2M mean and sum the four
   `tp06` states.
   Convert TP from m to mm. AFNOv2 already applies its checkpoint's exponential
   inverse transform and clips negative raw diagnostic output to zero; record
   that model behavior and do not apply a second undocumented clip.
9. Bilinearly regrid daily T2M and conservatively regrid daily TP to the
   canonical India grid, once per daily field.
10. Write both fields and the two-checkpoint provenance manifest atomically.

FCN3 must still evolve the full global 72-channel state internally. Cropping
the state before inference is invalid. The diagnostic consumes the 720 x 1440
subset from 90 to -89.75 deg; assert that only FCN3's -90 deg row is removed.
Keep the checkpoint-selected precision and do not call `.half()` on the full
wrapper. Process members serially at first because batching may exceed 80 GB.
The model card reports a 60-day one-member FCN3 forecast in under four minutes
on a suitable single GPU, but the local combined-pipeline pilot is the resource
authority. If FCN3+AFNOv2 does not fit one A100, stop and profile a two-GPU
streaming design; do not spill every 72-channel forecast state to storage.

### 10.2 DLESyM v0 plus precipitation runner

1. Load the native-HEALPix `DLESyMv0_ISCCP_ERA5` prognostic and
   `DLESyMv0_ISCCP_ERA5Precip` from the same pinned package. Keep the LatLon
   wrapper only as a pilot/reference path.
2. Load the staged D-48 h to D native history with `use_ttr=True`. The first
   iterator yield must be in model-space `rlut` after the one-time bundled
   TTR-to-OLR transformation.
3. Keep the rollout in model space (`rlut`, not raw `ttr`) after initialization.
4. One coupled step advances 96 hours and returns 16 atmosphere times at 6-hour
   spacing. Run 11 coupled steps, retain through +1008 h, and discard +1014 to
   +1056 h.
5. Use both `retrieve_valid_atmos_outputs` and `retrieve_valid_ocean_outputs`.
   The dense coupled tensor contains valid SST only at the 48-hour ocean output
   times; its intervening ocean slots must never be read.
6. Construct six-hourly SST by linear interpolation between valid ocean states
   at 0, +48, +96 h, and so on. This reproduces the upstream DLESyM
   `data_processing/precip_forecast/prep_precip_inputs.py` procedure. Retain
   enough look-ahead ocean output to interpolate every atmosphere time through
   +1008 h.
7. Combine each valid six-hourly native-HEALPix atmosphere state with its
   interpolated SST.
   Maintain the previous combined state and pass each previous/current pair to
   the precipitation diagnostic in its exact variable order.
8. The +6 h TP diagnostic must use the D 00 and +6 h states. Do not lose the
   initial state at the block boundary.
9. Load the diagnostic with `use_ttr=False` while chaining model output, since
   the prognostic output is already in OLR space.
10. Sum four `tp06` fields per day; apply the five-state trapezoidal T2M mean.
11. Regrid daily TP conservatively and T2M bilinearly, then write atomically.

The official example diagnoses only the final displayed state and converts a
lat/lon output back to HEALPix for that display calculation. A production
runner must remain on native HEALPix, diagnose every six-hourly state,
interpolate only between valid ocean SST forecasts, and preserve pairs across
all 96-hour block boundaries. Convert only the 42 aggregated daily fields to
the common output grid.

### 10.3 Optional DLESyM V1 runner

Run the 16 atmosphere/ocean checkpoint pairs as members 0-15. Save T2M only.
Do not use the v0 precipitation head on V1 output: V1 lacks the OLR channel and
does not share the diagnostic's input/scaling contract.

### 10.4 NeuralGCM runner

1. Load the 2.8 deg stochastic precipitation checkpoint once per process.
2. Load one preprocessed D state and D-1 persisted SST/sea-ice forcing.
3. Encode each member with an independent stored `jax.random.key`.
4. Use a six-hour output interval. The model's internal step is one hour.
5. Call `unroll(..., steps=169, timedelta=np.timedelta64(6, "h"),
   start_with_input=True)`. NeuralGCM defines those 169 decoded frames at +0,
   +6, ..., +1008 h; the separately returned advanced state is at +1014 h and
   must not be written.
6. Decode `precipitation_cumulative_mean` and difference +24 h boundaries.
7. Do not request or synthesize T2M.
8. Conservatively remap the 42 daily TP fields to the canonical India grid.
9. Write the file and manifest atomically.

JAX compilation is expensive. Keep one worker alive across a chunk of dates
with fixed array shapes so `encode`/`unroll` compile once. Pilot serial members
first; then test `vmap` or a fixed member batch only if memory and reproducible
seed handling are verified.

## 11. Output Contract

Write one NetCDF4 file per model and initialization, for example:

```text
forecasts/2020/20200601.nc
```

Required dimensions and coordinates:

```text
member
lead_day = 1..42
latitude = 39, 37.5, ..., 0
longitude = 60, 61.5, ..., 99
bounds = 0, 1
valid_time
forecast_period_start
forecast_period_end
forecast_period_bounds(lead_day, bounds)
init_time
forecast_reference_time
information_cutoff_time
```

Variables by run:

- FCN3+AFNOv2: `t2m(member, lead_day, latitude, longitude)` and `tp(...)`
- DLESyM v0: `tp(...)`, `t2m(...)`
- NeuralGCM precipitation: `tp(...)`

Required variable units:

- `tp`: `mm day-1`
- `t2m`: `degC`

Required global attributes include:

- run label and model display name
- exact checkpoint URI and SHA256
- Earth2Studio/NeuralGCM code version and commit
- environment lock hashes
- input source URL and per-case request hash
- native grid and native temporal resolution
- per-variable provenance (`t2m = FCN3 native`, `tp = AFNOv2 diagnostic`)
- forcing policy and information cutoff
- temporal aggregation definition
- remapping method and weight hash
- native and output member counts
- member seed mapping or DLESyM checkpoint-pair mapping
- available and unavailable requested fields, with reasons
- creation time, host, Slurm job ID, and repository Git commit

Write a matching JSON manifest containing source paths and file SHA256. Create
the final `.nc` only after all checks pass; otherwise leave a `.failed.json`
with the exception and no completion marker.

Uncompressed output is small compared with staged inputs. Across all 621 dates,
one float32 field is about 0.59 GiB at 10 members, 42 days, and 27 x 27 points.
Use moderate compression and chunks aligned to one member and one or seven lead
days. Avoid thousands of tiny per-lead files.

## 12. Slurm Execution Strategy

The checked cluster has A100-SXM4 80 GB GPUs in `GPU-AI`; older logs also show
A30 GPUs in the generic `gpu` partition and a 6 GB MIG node. Use `GPU-AI` for
FCN3 and DLESyM. NeuralGCM may use A30 after a successful test, but exclude the
6 GB MIG node (`cn3`) unless a measured memory profile proves it sufficient.

Initial requests, to be revised from pilot measurements:

| Runner | Partition | GPU | CPU | RAM | Initial task unit |
|---|---|---:|---:|---:|---|
| FCN3+AFNOv2 | `GPU-AI` | 1 x 80 GB, subject to combined smoke test | 8 | 64 GB | 1 date, 10 serial members |
| DLESyM v0 | `GPU-AI` | 1 x 80 GB | 8 | 64 GB | 4-8 dates per process |
| NeuralGCM | `gpu`, excluding `cn3`, or `GPU-AI` | 1 | 8 | 64 GB | 8-16 dates per process |

Start arrays with a concurrency cap of two. Increase only after observing GPU
availability, ARCO cache behavior, storage metadata load, and scheduler policy.
The GPU-AI smoke test was pending because nodes were reserved at the time of
this review, so queue access itself is a preflight gate.

Every job must:

- use `set -euo pipefail`
- print hostname, GPU name/memory, package versions, run label, case range, and
  start/end timestamps
- set private writable `XDG_CACHE_HOME` and `MPLCONFIGDIR` under job-local `/tmp`
- use shared model/data cache paths only for completed read-only artifacts
- trap termination and remove only that job's temporary directory
- skip a date only when both its final file and valid manifest pass audit
- never infer a date from filesystem ordering; map array indices through the
  frozen CSV/chunk manifest

Do not launch one JAX process per date if it causes 621 identical compilations.
Do not let hundreds of model jobs fetch ARCO concurrently.

## 13. Launch Gates and Pilot Sequence

Use `2020-06-01` for the first pilot because it is an actual calendar row.

### Gate A: static contract tests

- 621 unique dates and exact year counts
- expected model variables, input histories, levels, and grids
- 168 six-hour forecast steps or 169 NeuralGCM frames including +0
- exact daily and weekly period bounds
- no output claim for an unavailable field
- exact FCN3 -> derived `sp` -> AFNOv2 variable order and 721-to-720 grid map
- stable seed generation
- atomic/resume behavior

### Gate B: two-step GPU smoke

For each environment and model:

- GPU is visible and has sufficient memory
- checkpoint loads from the pinned cache
- one initialization fetch/adaptation succeeds
- +6 and +12 h outputs have expected coordinates and finite values
- rerunning the same seed reproduces the output

### Gate C: short two-day pilot

- FCN3+AFNOv2 and NeuralGCM: two members
- DLESyM: one member
- verify daily aggregation by recomputing from retained six-hour pilot values
- compare six-hour trapezoidal ERA5 T2M with the exact hourly ERA5 daily mean
  over the pilot dates and record the sampling error
- verify FCN3 and DLESyM `tp06` sums, NeuralGCM cumulative differencing, and
  all TP units
- for FCN3 at D 00, compare derived `sp` against actual ERA5 `sp` and compare
  AFNOv2 `tp06` using each; save bias/RMSE maps and distribution summaries
- confirm the FCN3 initial diagnostic is excluded and day 1 uses only +6,
  +12, +18, and +24 h
- verify DLESyM +6 h pairing, SST interpolation around +48/+96 h, and the
  +96/+102 h block boundary

### Gate D: one complete 42-day pilot

- exact final valid time +1008 h
- 42 daily fields, six seven-day means
- no missing or duplicate valid times
- physically plausible ranges and no unexplained negative TP
- FCN3+AFNOv2 TP remains finite and nonnegative by lead week, with no collapse
  to all-zero or explosive distributions
- conservative-remap area-integral check
- native-grid and canonical-grid maps inspected for orientation/seams
- peak GPU/CPU memory and wall time recorded

### Gate E: cross-year pilot

Run at least one case in each year, including leap-day handling around
`2024-02-29`. Confirm day-of-year OLR climatology indexing in DLESyM and the
same calendar/output behavior across years.

### Gate F: limited production batch

Run ten dates per model. Kill and fix the workflow if any audit rule fails.
Use measured P95 wall time and memory, plus a safety margin, to set final Slurm
resources and chunk sizes.

### Gate G: full launch

Launch the remaining dates only after Gates A-F are recorded in a signed-off
`PILOT_REPORT.md`. This is the first point at which a 621-date array is allowed.

## 14. Validation and Audit Rules

Per-date validation:

- filename initialization equals internal `init_time`
- finite values everywhere on the India grid
- exact dimensions, coordinate order, 42 lead days, and expected member count
- exact period bounds with no gaps or overlaps
- T2M uses the frozen five-state trapezoidal weights for each exact UTC period
  and is converted from K to degC exactly once
- FCN3+AFNOv2 TP is four `tp06` sums per day, meters to millimeters exactly
  once, and excludes the initial `[D-6 h, D]` field
- DLESyM TP is four `tp06` sums per day and meters to millimeters exactly once
- NeuralGCM TP is a daily difference of cumulative meters and converted once
- cumulative NeuralGCM precipitation is nondecreasing within numeric tolerance
- ensemble members differ for stochastic models and repeat for fixed seeds
- remapping weight hash matches the run config
- unavailable fields are absent and documented
- every FCN3 file identifies T2M as native and TP as AFNOv2 diagnostic output

Whole-run audit:

- 621 valid forecast files and 621 valid manifests per run
- per-year counts 105/104/104/104/100
- no orphan temporary files or duplicate initializations
- one checkpoint hash, environment hash, input policy, and regrid hash per run
- member/seed/checkpoint mapping complete for every date
- file SHA256 agrees with manifest
- summary CSV lists success/failure, wall time, peak memory, value ranges, and
  output bytes for all cases

Compare a sample of daily products against ERA5 truth and simple persistence
before calculating skill. This catches orientation, unit, and lead-offset bugs
that can survive shape checks.

## 15. Efficient Order of Work

1. Freeze the common config and copy the 621-row calendar into each immutable
   run config with a hash.
2. Build and GPU-smoke-test the three isolated environments.
3. Download and hash all checkpoints once.
4. Implement/test the common output, date, seed, remap, and audit modules.
5. Stage/prefetch ARCO inputs with low concurrency.
6. Implement the FCN3 native-T2M plus AFNOv2-diagnostic-TP runner and pass
   Gates A-D.
7. Implement DLESyM v0 TP/T2M runner, paying special attention to diagnostic
   state pairs and 96-hour boundaries.
8. Implement NeuralGCM TP runner with D-1 persisted SST/sea ice and JIT reuse.
9. Run cross-year and ten-date pilots.
10. Launch production arrays at conservative concurrency.
11. Audit all outputs before weekly aggregation or verification.
12. Add optional DLESyM V1 T2M and 50-member stochastic sensitivities only
    after the primary runs are complete.

## 16. Official References Checked

- FCN3 model card: <https://huggingface.co/nvidia/fourcastnet3>
- AFNOv2 precipitation diagnostic API:
  <https://nvidia.github.io/earth2studio/modules/generated/models/dx/earth2studio.models.dx.PrecipitationAFNOv2.html>
- AFNO_DX_TP-V1-ERA5 model card and exact 20-variable contract:
  <https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/afno_dx_tp-v1-era5/>
- Surface-pressure derivation method:
  <https://nvidia.github.io/earth2studio/userguide/notes/surface_pressure.html>
- Earth2Studio 0.16.0 FCN3/derived-SP/AFNOv2 wrapper contract test:
  <https://github.com/NVIDIA/earth2studio/blob/0.16.0/test/models/px/test_dxwrapper.py>
- Earth2Studio installation and pinned optional dependencies:
  <https://nvidia.github.io/earth2studio/userguide/about/install.html>
- Earth2Studio source: <https://github.com/NVIDIA/earth2studio>
- Makani source: <https://github.com/NVIDIA/makani>
- DLESyM V1 model card: <https://huggingface.co/nvidia/dlesym-v1-era5>
- DLESyM v0 model card:
  <https://huggingface.co/nvidia/dlesym-v0-isccp-era5>
- DLESyM v0 precipitation diagnostic API:
  <https://nvidia.github.io/earth2studio/modules/generated/models/dx/earth2studio.models.dx.DLESyMv0_ISCCP_ERA5Precip.html>
- Upstream DLESyM repository: <https://github.com/AtmosSci-DLESM/DLESyM>
- NeuralGCM source: <https://github.com/neuralgcm/neuralgcm>
- NeuralGCM checkpoint guide:
  <https://neuralgcm.readthedocs.io/en/stable/checkpoints.html>
- NeuralGCM data preparation and forcing shift:
  <https://neuralgcm.readthedocs.io/en/latest/data_preparation.html>
- NeuralGCM precipitation dataset semantics:
  <https://neuralgcm.readthedocs.io/en/latest/neuralgcm_datasets.html>
- JAX GPU installation: <https://docs.jax.dev/en/latest/installation.html>
