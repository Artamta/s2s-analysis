# ERPAS Data

ERPAS data from Google Drive is downloaded directly into the same canonical
storage tree used by the other forecast providers.

## Source Audit

- Drive folder: `1S58Swev_M33lAK6Aap7FJuFvG5gtv3Jf`
- Dated forecasts: 2023-2025, weekly Wednesday initializations
- Forecast length: 33 daily leads, initialized at 00 UTC
- Ensemble metadata: no member dimension; treat these files as deterministic
- Global grid: regular 1 degree, 360 x 181
- India precipitation grid: regular 0.5 degree, 241 x 161

The dated source products are:

- `APCP_YYYYMMDD.grb`: 24-hour total precipitation, `kg m-2`;
- `tsfc_YYYYMMDD.grb`: instantaneous surface temperature at daily endpoints,
  not 2 m temperature and not a daily mean;
- `gpot_YYYYMMDD.grb`: geopotential height at 1000, 925, 850, 700, 500, 300,
  and 200 hPa;
- `Ind_0.5_APCP_YYYYMMDD.grb`: India-domain 0.5-degree precipitation.

The source also has duplicate Drive folders named `z_HINDCAST_CLIMATOLOGY`.
Together they contain 432 files in 144 `MMDD` directories. The downloader
merges every duplicate branch into one climatology tree.

Provider utility scripts are excluded from canonical storage. One Drive script
contains a plaintext provider credential and must not be copied into Git or
the dataset tree.

## Storage

```text
/storage/raj.ayush/s2s_final_data/final_iteration/
  raw/erpas/
    forecast/
      annual2023/
        tp/
        surface_temperature/
        geopotential_height/
        tp_india_0p5/
      annual2024/...
      annual2025/...
    reforecast/
      climatology/<MMDD>/{APCP,gpot,tsfc}.grb
  manifests/erpas/google_drive_20260714/
  logs/erpas/
```

`surface_temperature` must not be used as the `t2m` field in the primary
ECMWF/NCEP/UKMO/FuXi comparison. ERPAS precipitation can be evaluated through
lead day 33 after grid, date, and deterministic-versus-ensemble alignment.

## Transfer

```bash
sbatch clean/data-download/erpas/slurm/download_erpas_gdrive.sbatch
```

The job uses a shared rclone executable, copies with two concurrent transfers,
verifies each source subset, and writes JSON and SHA-256 inventories.

An interrupted climatology transfer can reuse its existing staging directory:

```bash
sbatch --export=ALL,ERPAS_STAGING_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/staging/erpas/<staging-directory> \
  clean/data-download/erpas/slurm/download_erpas_gdrive.sbatch
```

Rclone skips matching destination files, verifies every source subset, and
removes the selected staging directory only after all manifests are complete.
The bundled rclone does not implement SHA-256, so the downloader uses the
system `sha256sum` command over a sorted relative-path inventory.
