# `final_analysis/` — Reproducible S2S verification pipeline

A clean, config-driven rewrite of the S2S hindcast verification. One shared
**core** (the maths + grid + masks + truth) is reused by every season; each
season (JFM2026, JJAS, …) supplies only an `ExperimentConfig` and a handful of
model **adapters**.

> Goal: *inject a forecast (path / format) and get ACC, RMSE, CRPSS, BSS, …* —
> at a common resolution, over All-India + the 4 IMD homogeneous regions —
> **without editing the maths or the driver.**

---

## Layout

```
final_analysis/
├── core/                  season-agnostic base (shared by ALL seasons)
│   ├── metrics.py         pure math: acc, rmse, bias, msss, crps/crpss, ssr, brier/bss, reliability
│   ├── grid.py            common-grid regridding, land mask, IMD region masks, cos weights
│   ├── climatology.py     ERA5 WMO day-of-year clim mean + climatological spread
│   ├── truth.py           ERA5 truth (TP/T2M/Z500), period-mean / daily / persistence
│   ├── aggregate.py       week/day collapse, cumulative differencing, ensemble mean/std
│   ├── adapters.py        ForecastCube canonical container + model-adapter REGISTRY
│   └── config.py          typed ExperimentConfig / ModelSpec  (the injectable contract)
├── jfm2026/
│   ├── config.py          JFM2026 init dates, resolved paths, the 3 model specs
│   ├── adapters_jfm.py    SPIRE / FuXi / ECMWF loaders for the current file layout
│   ├── run_verify.py      the driver (per-init parallel, weekly + daily, era5 basis)
│   └── plots/             figure scripts (read the CSVs)
└── jjas20xx/              later: a new config + adapters, reuses ALL of core/
```

## The one idea that makes it pluggable: `ForecastCube`

Every model is loaded by an **adapter** that returns the same object, already in
verification units (Z500 gpm, T2M K, TP mm/day-equivalent):

```python
ForecastCube(
    name="ECMWF", var="TP", accum="cumulative",
    members=DataArray(member, step, lat, lon),   # ensemble systems
)
# …or for a summary system (SPIRE: only mean+spread are archived):
ForecastCube(name="SPIRE", var="Z500", accum="instant",
             mean=DataArray(step, lat, lon), std=DataArray(step, lat, lon))
```

`accum` tells the cube how to collapse a lead-window:

| accum | used by | week collapse |
|:--|:--|:--|
| `instant` | Z500, T2M | mean over the window's steps |
| `rate` | FuXi tp (×24), SPIRE tp | mean over the window's steps |
| `cumulative` | ECMWF/NCEP tp | (end − start) / n_days |

The cube's `.weekly(ds, de, GC)` / `.daily(di, GC)` return `(mean, spread)` on
the common grid. **The driver and metrics never branch on model identity.**

## Add a new forecast in 2 steps

1. Write an adapter in the season's `adapters_jfm.py`:
   ```python
   @register("mymodel")
   def load_mymodel(init, var, spec, physics):
       da = xr.open_dataset(f"{spec.kwargs['root']}/{init}.nc")[var]
       return ForecastCube("MyModel", var, accum="instant", members=da)
   ```
2. Add one `ModelSpec(name="MyModel", adapter="mymodel", kwargs={...})` to the
   config's `models` list. Done — it now appears in every CSV and figure.

## Run

```bash
conda activate s2s-hind
cd jfm2026
python run_verify.py --test               # 1-init smoke test
python run_verify.py --workers 13         # full run (or sbatch)
```

Outputs (in the season folder): `skill_deterministic.csv`,
`skill_probabilistic.csv`, `skill_brier.csv`, `reliability.npz`,
`RUN_METADATA.txt`.

## Conventions (locked across the pipeline)

- **Common grid**: everything is interpolated to the `GridSpec` grid (default
  1.5° India box) and land-masked before any metric — so resolutions never mix.
- **Regions**: All-India + 4 IMD homogeneous rainfall regions, masks built from
  the Survey-of-India shapefile (see `core/regions.py`, prebuilt to NetCDF).
- **Anomaly baseline**: `clim_basis='era5'` — every model vs the ERA5 30-yr WMO
  day-of-year climatology (fair inter-model comparison). A `model_own` basis
  hook exists for when per-model hindcast climatologies are available.
- **Weighting**: cosine-latitude, normalised to mean 1.0 (WMO).

## Status / notes

- JFM2026 models: **SPIRE, FuXi, ECMWF** (+ MME, + Persistence), vars **TP, Z500**.
- `model_own` climatology files are not currently on disk, so the first JFM2026
  run is **era5-basis only** (the headline fair comparison). Regenerate them via
  `/storage/raj.ayush/s2s-data-pipeline/climatology/` to enable the second basis.
