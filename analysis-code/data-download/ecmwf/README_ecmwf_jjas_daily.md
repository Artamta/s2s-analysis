# ECMWF Daily Operational JJAS Downloads

This uses the CDS/ECDS `s2s-forecasts` route, matching the successful JFM 2026
ECMWF downloader. For historical 2019 ECMWF S2S forecasts, available
initialization dates are Mon/Thu only. For 2025 the daily operational cycle is
used. It writes to:

```text
/storage/raj.ayush/All_Model_Data/ecmwf/jjas2019
/storage/raj.ayush/All_Model_Data/ecmwf/jjas2025
```

Layout mirrors `ecmwf/jfm2026`:

```text
2t/YYYYMMDD_{cf,pf}.nc
msl/YYYYMMDD_{cf,pf}.nc
tp/YYYYMMDD_{cf,pf}.nc
z/{1000,850,500,200}/YYYYMMDD_{cf,pf}.nc
```

Note: as in the JFM downloader, ECMWF operational S2S `2t` is requested only at
step 24. Multi-lead temperature verification should use available max/min fields
or another agreed temperature definition.

Run:

```bash
cd /home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/ecmwf
sbatch slurm_ecmwf_jjas_2019_2025_daily.sbatch
```

The downloader is resumable and skips existing non-empty files.
