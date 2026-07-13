# FuXi-S2S Launch Status

## Contract

- Calendar: 621 exact physics-model dates, 2020-2025.
- Ensemble: 50 model-native stochastic members from official `inference.py`.
- No artificial input perturbations and no assumed control member.
- Leads: days 1-42.
- Final fields: native FuXi TP and T2M on the exact 27 x 27 India grid.

## Preflight

Commit `e770262` passed:

- full SHA256 verification of the official ONNX graph, 2 GB external checkpoint,
  official inference script, and mask;
- unique mapping of all 621 calendar rows to 621 output paths;
- Python, JSON, Slurm, and whitespace checks;
- full 50-member x 42-lead compaction/QC against an existing official run.

## Pilot

Slurm task `67911_0` ran initialization `2020-01-02` on `gpu2` and completed in
`00:12:37` with exit code `0:0` and peak RAM of about 3.3 GB.

The retained product has:

```text
member=50, lead_day=42, latitude=27, longitude=27
valid_time=2020-01-03 through 2020-02-13
TP range=0.0 to 4.4377 mm h-1
T2M range=233.4311 to 303.3557 K
member 0/1 maximum difference=16.5919
```

The output is 7,819,309 bytes, its manifest status is `generated_valid`, and
the raw 2,100-file work directory was deleted after QC. The official child
returned `255` at shutdown after all files were complete; the wrapper recorded
that code, validated all raw files and the final NetCDF, and Slurm completed the
task successfully.

## Production

Production tasks `1-620` are pending submission. Task `0` is already complete
and will not be duplicated.
