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
