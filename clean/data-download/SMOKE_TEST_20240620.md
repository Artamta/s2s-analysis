# Physics-Model Day-1 Smoke Test

Initialization: `2024-06-20 00Z`. Domain: `0-40 N, 60-100 E`. This is one of
50 exact 2024 initialization dates shared by ECMWF, UKMO, NCEP, CMA, and CNRM
for both TP and temperature in the live ECDS catalogue.

## Result

All 22 ECDS requests downloaded, opened with ecCodes/cfgrib, and passed basic
variable, member, step, coordinate, and nonempty-file checks. Every field is on
the requested 27 x 27 regional grid. No empty files or `.part` files remain.

| provider | members | cadence relevant to study | TP | temperature |
|---|---:|---|---|---|
| ECMWF | 1 control + 100 perturbed | daily in 2024 | 24-hour accumulation | `0_24` daily mean |
| UKMO | 1 control + 3 perturbed | daily, with archive gaps | 24-hour accumulation | `0_24` daily mean |
| NCEP | 1 control + 15 perturbed | daily | 24-hour accumulation | mean of four six-hour `(Tmax + Tmin) / 2` proxies |
| CMA | 1 control + 3 perturbed | Monday/Thursday | 24-hour accumulation | `0_24` daily mean |
| CNRM | 1 control + 24 perturbed | weekly Thursday | 24-hour accumulation | `0_24` daily mean |

The plotted fields are native-ensemble means. All precipitation panels share
the full `0-199.37 mm` scale and all temperature panels share the full
`-4.97-41.36 degC` scale. The shared scale deliberately exposes CMA's localized
day-1 TP maximum instead of normalizing it away.

| provider | mean TP (mm) | maximum TP (mm) | mean temperature (degC) | minimum temperature (degC) | maximum temperature (degC) |
|---|---:|---:|---:|---:|---:|
| ECMWF | 4.88 | 40.79 | 25.01 | -4.17 | 40.98 |
| UKMO | 6.35 | 106.04 | 25.32 | -2.61 | 37.26 |
| NCEP | 6.82 | 69.95 | 25.55 | -2.59 | 41.36 |
| CMA | 4.59 | 199.37 | 25.43 | -4.39 | 37.49 |
| CNRM | 3.09 | 29.53 | 25.51 | -4.97 | 39.03 |

The NCEP temperature panel is deliberately labeled `6-hour extrema proxy`.
It is useful for weekly comparison but is not the same archived statistic as
the four providers' or FuXi's daily-mean T2M.

An attempted `2024-06-06` test correctly failed on UKMO with `MarsNoDataError`.
The live catalogue shows a UKMO gap from June 5-19. The final test therefore
uses June 20 rather than silently comparing different initialization dates.

## Files

```text
/storage/raj.ayush/s2s_final_data/final_iteration/smoke_tests/physics_models/20240620/
  ecmwf/*.grib
  ukmo/*.grib
  ncep/*.grib
  cma/*.grib
  cnrm/*.grib
  manifest.json
  field_summary.json
  physics_models_day1_comparison.png
```

Reproduce or resume:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/data-download/scripts/smoke_test_physics_models.py
```

Existing valid files are reused unless `--overwrite` is supplied. Plot again
without contacting ECDS by adding `--skip-download`.
