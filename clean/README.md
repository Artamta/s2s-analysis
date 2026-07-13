# Clean S2S Workspace

This folder is the version-controlled control plane for downloading, organizing,
running, and verifying the final S2S benchmark. Heavy data stay on `/storage`.

## Purpose

Keep raw provider downloads, model-run products, and analysis outputs separated
so every later result can be traced back to a provider, initialization date,
forecast type, variable, lead range, and download request.

## Layout

```text
clean/
  config/
    datasets.json
    archive_policy.json
    comparable_dates_2019_2026.csv
  data-download/
    ecmwf/
    ukmo/
    ncep/
  model-runs/
    configs/
    logs/
    outputs/
    manifests/
```

Provider folders contain code and documentation only:

```text
scripts/      download and QC scripts
slurm/        reproducible cluster launchers
manifests/    lightweight plans and summaries suitable for Git
```

The canonical heavy-data root is:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/
  raw/<provider>/<forecast|reforecast>/<season>/
  standardized/<provider>/<experiment>/
  truth/<dataset>/
  manifests/<provider>/
  logs/<provider>/
```

Legacy datasets are not copied. Their physical locations and roles are recorded
in `config/datasets.json`.

## Rules

- Raw downloads are immutable after validation.
- Every download script must be resumable and skip existing non-empty files.
- Every attempted request must write a JSONL manifest row.
- Keep operational forecasts and reforecasts/hindcasts separate on disk.
- Keep this repo for code, configuration, lightweight inventories, and run
  summaries only.
- Never compare models by lead number alone when their initialization dates
  differ; align the valid dates first.

## Locked Acquisition Order

- Primary physics providers: ECMWF, UKMO, NCEP, and CMA operational S2S.
- Secondary physics provider: CNRM on the common weekly subset.
- Missing years: `2020-2024`.
- Target season: 35 FuXi starts from June 2 through September 29; a paired
  ECMWF initialization may fall in late May.
- Cadence: the 35 fixed FuXi JJAS starts, paired to ECMWF's native schedule
  using `config/comparable_dates_2019_2026.csv`.
- Variables: `tp`; ECMWF/UKMO daily-mean `t2m`; labeled NCEP `t2m_proxy`.
- Valid window: FuXi lead days `1-42`; ECMWF lead endpoints extend to day
  `42 - init_offset_days` where starts differ.
- Ensemble: control plus every native perturbed member.
- Comparison: same valid-date windows on the common 1.5 degree India grid.
- Gate: finish and validate 2020-2024 forecasts before reforecast downloads.
- Reforecasts: native archives first; `2002-2010` only as a common-period
  sensitivity. There is no honest 20-year common archive across all models.
