# Physics-Model Day-1 Smoke Test

Initialization: `2024-06-02 00Z`. Domain: `0-40 N, 60-100 E`. The test
downloaded one lead day of TP and temperature for ECMWF, UKMO, and NCEP,
including each control and every native perturbed member.

## Result

All 14 ECDS requests downloaded, opened with ecCodes/cfgrib, and passed basic
variable, member, step, coordinate, and nonempty-file checks.

| provider | members | TP | temperature |
|---|---:|---|---|
| ECMWF | 1 control + 100 perturbed | 24-hour accumulation | `0_24` daily mean |
| UKMO | 1 control + 3 perturbed | 24-hour accumulation | `0_24` daily mean |
| NCEP | 1 control + 15 perturbed | 24-hour accumulation | mean of four six-hour `(Tmax + Tmin) / 2` proxies |

The plotted fields are native-ensemble means. Both precipitation panels and
temperature panels use a single cross-model scale.

| provider | mean TP (mm) | maximum TP (mm) | mean temperature (degC) | minimum temperature (degC) | maximum temperature (degC) |
|---|---:|---:|---:|---:|---:|
| ECMWF | 4.51 | 66.68 | 25.14 | -8.28 | 40.18 |
| UKMO | 5.16 | 89.40 | 25.05 | -6.46 | 36.77 |
| NCEP | 4.74 | 69.98 | 25.80 | -6.38 | 40.84 |

The NCEP temperature panel is deliberately labeled `6-hour extrema proxy`.
It is useful for weekly comparison but is not the same archived statistic as
ECMWF/UKMO/FuXi daily-mean T2M.

## Files

```text
/storage/raj.ayush/s2s_final_data/final_iteration/smoke_tests/physics_models/20240602/
  ecmwf/*.grib
  ukmo/*.grib
  ncep/*.grib
  manifest.json
  field_summary.json
  physics_models_day1_comparison.png
```

Reproduce or resume:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/data-download/scripts/smoke_test_physics_models.py \
  --init-date 2024-06-02
```

Existing valid files are reused unless `--overwrite` is supplied. Plot again
without contacting ECDS by adding `--skip-download`.
