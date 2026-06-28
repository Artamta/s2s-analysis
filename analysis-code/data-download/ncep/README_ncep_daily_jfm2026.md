# NCEP Daily JFM 2026 Download

This downloader writes NCEP S2S forecasts into:

`/storage/raj.ayush/All_Model_Data/ncep/jfm2026`

Layout:

```text
ncep/jfm2026/
  surface/cf/YYYYMMDD.grib
  surface/pf/YYYYMMDD.grib
  z/500/cf/YYYYMMDD.grib
  z/500/pf/YYYYMMDD.grib
  logs/
```

Coverage:

- Initialization dates: daily, `2026-01-01` to `2026-03-31`
- Leads: daily forecast steps from 24 to 1056 hours, i.e. days 1 to 44
- Surface variables: `2t`, `mx2t6`, `mn2t6`, `tp`
- Pressure variable: `gh` at 500 hPa
- Domain/grid: India box `[50, 55, 0, 105]`, 1.5 degree grid
- Ensemble: NCEP native `cf` plus `pf` members

Run:

```bash
cd /home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/ncep
sbatch slurm_ncep_jfm2026_daily.sbatch
```

The script is resumable and skips existing non-empty files.

Additional seasons can use the same downloader with `--out-root`, for example:

```bash
python download_ncep_jfm2026_daily.py \
  --out-root /storage/raj.ayush/All_Model_Data/ncep/jjas2019 \
  --start 20190601 \
  --end 20190930
```

The JJAS 2019 and JJAS 2025 sequential Slurm job is:

```bash
sbatch slurm_ncep_jjas_2019_2025_daily.sbatch
```
