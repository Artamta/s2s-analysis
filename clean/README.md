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

## Locked First Phase

- Provider: ECMWF operational S2S.
- Missing years: `2020-2024`.
- Season: JJAS (`June 1` through `September 30`).
- Cadence: twice weekly, ECMWF Monday/Thursday initialization cycle.
- Variables: `tp` and `z500`.
- Lead range: daily lead days `1-42`.
- Ensemble: control plus every perturbed member returned by ECMWF.
- Comparison: same valid-date windows on the common 1.5 degree India grid.
