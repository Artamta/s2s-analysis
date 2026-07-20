# DLESyM 2020-2024 inference

This directory implements two separately labelled DLESyM products on the frozen
517-date Monday/Thursday calendar in `config/all_season_dates_2020_2024.csv`:

- `DLESyM-V1-ERA5`: four members using matched atmosphere/ocean checkpoint
  indices `(0,0)` through `(3,3)`, T2M only.
- `DLESyM-v0-ISCCP-ERA5 + TP diagnostic`: one deterministic member, native T2M
  plus the official two-state `tp06` precipitation diagnostic.

The products must not be merged or labelled as one checkpoint configuration.
Every task stages the official ERA5 history on native HEALPix, runs 42 complete
UTC days, writes a compact 27 x 27 India-domain NetCDF, and writes a matching
SHA256 manifest. Existing results are skipped only after their manifest and file
hash validate.

## Pilot order

Submit the V1 T2M pilot first. Submit the V0 T2M/TP pilot with an `afterok`
dependency on V1. Both use one A100 in `GPU-AI_prio` and initialization
2020-06-01.
The jobs download missing package files with visible progress and write a hashed
byte inventory under each run's `provenance/` directory.

Each production script is capped at one concurrent A100 task. V1 and V0 run
independently in parallel, for a combined DLESyM limit of two GPUs. Do not submit
them until both plots, ranges, lead times, GPU memory, and wall times in the pilot
manifests have been reviewed.

The V1 package exposes four atmosphere and four ocean files but does not
document their Cartesian product as a 16-member protocol. A rejected pilot of
all 16 cross-pairs made `(atmosphere 0, ocean 1)` numerically unstable at lead
days 40-42. The benchmark therefore pairs corresponding component indices and
enforces a strict `-100 < t2m < 70 degC` output gate. The rejected pilot remains
in storage under the `ens16` label and must not be used for scores.

Official documentation:

- <https://nvidia.github.io/earth2studio/examples/06_seasonal/02_dlesym_example.html>
- <https://nvidia.github.io/earth2studio/examples/06_seasonal/03_dlesym_climate.html>
- <https://huggingface.co/nvidia/dlesym-v1-era5>
- <https://huggingface.co/nvidia/dlesym-v0-isccp-era5>
