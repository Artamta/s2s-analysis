# Model Runs

This folder is for derived model-run products and verification-ready outputs,
not raw provider downloads.

Recommended layout:

```text
configs/      run configs copied from the exact download/verification setup
logs/         model-run and post-processing logs
outputs/      compact NetCDF/Zarr products used by verification
manifests/    product manifests linking outputs back to raw downloads
```

Every run should record:

- run label
- provider/model
- source manifest path
- variables
- initialization dates
- lead range
- grid/resolution
- units after conversion
- output files

The FuXi-S2S operational benchmark is implemented under `fuxi/`. It uses the
same 621-date 2020-2025 initialization calendar as the physics providers and
writes compact India TP/T2M products with per-date provenance manifests. Its 50
members come only from the official model's native stochastic inference; no
external perturbation wrapper is used.

The checked implementation plan for the 2020-2025 FCN3, DLESyM, and NeuralGCM
runs is in `MULTI_MODEL_INFERENCE_RUNBOOK.md`. Read its field-availability
matrix and launch gates before writing runners or submitting production arrays.
