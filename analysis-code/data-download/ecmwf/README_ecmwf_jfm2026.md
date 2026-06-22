# ECMWF S2S JFM 2026 — Operational Forecast Data

## Overview

Real-time ECMWF S2S operational forecasts for **January–March 2026**, downloaded
from ECDS (`s2s-forecasts` dataset) via cdsapi v0.7.7.

**NOT hindcasts/reforecasts** — this is the actual forecast data for verification
against observations (ERA5, IMD, etc.).

---

## Data Location

```
/storage/raj.ayush/All_Model_Data/ecmwf/jfm2026/
├── tp/          # Total precipitation
├── 2t/          # 2m temperature        ← IN PROGRESS (downloading now)
├── msl/         # Mean sea level pressure
└── z/
    ├── 200/     # Geopotential height at 200 hPa
    ├── 500/     # Geopotential height at 500 hPa
    ├── 850/     # Geopotential height at 850 hPa
    └── 1000/    # Geopotential height at 1000 hPa
```

---

## Download Status (as of 2026-06-22)

| Variable | Description | Files | Status |
|----------|-------------|-------|--------|
| `tp` | Total precipitation | 180 / 180 | ✅ Complete |
| `msl` | Mean sea level pressure | 180 / 180 | ✅ Complete |
| `z/200` | Geopotential height 200 hPa | 180 / 180 | ✅ Complete |
| `z/500` | Geopotential height 500 hPa | 180 / 180 | ✅ Complete |
| `z/850` | Geopotential height 850 hPa | 180 / 180 | ✅ Complete |
| `z/1000` | Geopotential height 1000 hPa | 180 / 180 | ✅ Complete |
| `2t` | 2m temperature | 0 / 180 | 🔄 Downloading (SLURM job 55755) |

**Total: 1080 / 1260 files (86%)**

> `2t` requires a per-step workaround (46 separate API calls per file, then
> concatenated) because ECDS collapses multi-step 2m temperature requests into
> a single timestep in netcdf output.

### Not available in operational S2S stream
- `mx2t` (max 2m temperature) — only exists in reforecasts
- `mn2t` (min 2m temperature) — only exists in reforecasts

---

## Data Specifications

| Property | Value |
|----------|-------|
| **Source** | ECDS `s2s-forecasts`, ECMWF origin |
| **Init dates** | Every calendar day: 2026-01-01 → 2026-03-31 (90 dates) |
| **Forecast types** | `cf` = control (member 0) · `pf` = 100 perturbed members |
| **Total ensemble** | 101 members (1 cf + 100 pf) |
| **Lead times** | Day 1–46 (steps 24, 48, …, 1104 h at 24 h intervals) |
| **Domain** | India: 0–40°N, 60–100°E |
| **Resolution** | 1.5° × 1.5° (27 lat × 27 lon grid points) |
| **Format** | NetCDF4 |
| **Units** | tp: m (accumulated) · 2t: K · msl: Pa · z: gpm |

---

## File Naming Convention

```
{var}/{YYYYMMDD}_{type}.nc
```

Examples:
- `tp/20260101_cf.nc` — tp control forecast, init 2026-01-01
- `tp/20260101_pf.nc` — tp 100-member perturbed forecast, init 2026-01-01
- `z/500/20260115_cf.nc` — Z500 control forecast, init 2026-01-15

---

## File Dimensions

| File type | Dimensions | Approx size |
|-----------|------------|-------------|
| `*_cf.nc` | `(step=46, lat=27, lon=27)` | ~65–100 KB |
| `*_pf.nc` | `(number=100, step=46, lat=27, lon=27)` | ~4–5 MB |
| `2t_cf.nc` | `(step=46, lat=27, lon=27)` | ~23 KB |
| `2t_pf.nc` | `(number=100, step=46, lat=27, lon=27)` | ~4.6 MB |

---

## Reading the Data

```python
import xarray as xr

# Control forecast — tp for Jan 1 2026
ds = xr.open_dataset(
    "/storage/raj.ayush/All_Model_Data/ecmwf/jfm2026/tp/20260101_cf.nc"
)
# dims: (step=46, latitude=27, longitude=27)
# step values: 1 day, 2 days, ..., 46 days

# Perturbed forecast — all 100 members
ds_pf = xr.open_dataset(
    "/storage/raj.ayush/All_Model_Data/ecmwf/jfm2026/tp/20260101_pf.nc"
)
# dims: (number=100, step=46, latitude=27, longitude=27)

# Combine cf + pf into one 101-member dataset
import numpy as np
cf = ds["tp"].expand_dims(number=[0])
pf = ds_pf["tp"]
all_members = xr.concat([cf, pf], dim="number")
# shape: (101, 46, 27, 27)
```

---

## Download Scripts

| Script | Purpose |
|--------|---------|
| [download_ecmwf_jfm2026.py](download_ecmwf_jfm2026.py) | Main download script — JFM 2026 operational |
| [download_ecmwf_reforecasts_fullyr.py](download_ecmwf_reforecasts_fullyr.py) | Hindcast download — all Mon+Thu, 2000–2019 |

SLURM scripts: `/storage/raj.ayush/s2s-data-pipeline/slurm/ecmwf_jfm2026.sbatch`

Logs: `/storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/jfm2026/`

---

## Monitor Progress

```bash
# Live log
tail -f /storage/raj.ayush/s2s-data-pipeline/logs/ecmwf/jfm2026/slurm_55755.log

# File count by variable
find /storage/raj.ayush/All_Model_Data/ecmwf/jfm2026/ -name "*.nc" \
  | sed 's|.*/jfm2026/||' | cut -d'/' -f1-2 | sort | uniq -c

# Job status
squeue -u raj.ayush
```
