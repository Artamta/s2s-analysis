# Current Inventory: JJAS 2019-2025

Generated with:

```bash
python clean/data-download/scripts/inventory_s2s_data.py \
  --providers ecmwf,ukmo,ncep \
  --start-year 2019 --end-year 2025 --months 6,7,8,9
```

Outputs:

- `manifests/existing_file_inventory.csv`
- `manifests/coverage_summary.csv`
- `manifests/reforecast_mmdd_summary.csv`

## Operational Forecast Files Found

Coverage below is for JJAS Mon/Thu initializations only. JJAS has 35 Mon/Thu
init dates in both 2019 and 2025.

| provider | year | variables found | Mon/Thu coverage | notes |
|---|---:|---|---:|---|
| ECMWF | 2019 | msl, t2m, tp, z1000, z850, z500, z200 | 35/35 | cf + pf present |
| ECMWF | 2025 | msl, t2m, tp, z1000, z850, z500, z200 | 35/35 | cf + pf present; many non-Mon/Thu dates too |
| UKMO | 2019 | t2m, tp, z500 | 35/35 | cf + pf present |
| UKMO | 2025 | t2m, tp, z500 | 35/35 | cf + pf present |
| NCEP | 2019 | surface_multi, z500 | 35/35 | cf + pf present; surface file contains multiple vars |
| NCEP | 2025 | surface_multi, z500 | 35/35 | cf + pf present |

## Missing Operational Years

No operational JJAS folders/files were found in the scanned storage roots for:

```text
ECMWF: 2020, 2021, 2022, 2023, 2024
UKMO : 2020, 2021, 2022, 2023, 2024
NCEP : 2020, 2021, 2022, 2023, 2024
```

## ECMWF Reforecast Files Found

Separate from operational forecast folders, ECMWF reforecast files exist at:

```text
/storage/raj.ayush/archive/All_Model_Data/models/ecmwf/data
```

For JJAS MMDDs, found complete 35-slot Mon/Thu-style coverage for:

```text
tp   cf + pf
z500 cf + pf
```

These files are named by calendar initialization day, for example:

```text
tp_pf_0601.grib
z500_pf_0601.grib
```

They should be treated as reforecast/hindcast products, not as single-year
operational forecasts.

## Spot-Checked Ensemble Metadata

Quick sample opens:

| file type | sample | dimensions |
|---|---|---|
| ECMWF 2019 tp pf | `jjas2019/tp/20190603_pf.nc` | number=50, step=46, 27x27 |
| ECMWF 2025 tp pf | `jjas2025/tp/20250602_pf.nc` | number=100, step=46, 27x27 |
| UKMO 2019 tp pf | `jjas2019/tp/20190603_pf.nc` | number=3, step=60, 27x27 |
| UKMO 2025 tp pf | `jjas2025/tp/20250602_pf.nc` | number=3, step=60, 27x27 |
| NCEP 2019 surface pf | `jjas2019/surface/pf/20190603.grib` | number=15, step=44, 34x34; vars mx2t6, mn2t6, tp |
| NCEP 2019 z500 pf | `jjas2019/z/500/pf/20190603.grib` | number=15, step=44, 34x34; var gh |

## Practical Conclusion

For a 2019-2025 operational comparison over India:

1. Existing operational data are enough for a 2019 vs 2025 pilot across ECMWF,
   UKMO, and NCEP.
2. Operational years 2020-2024 need download for ECMWF, UKMO, and NCEP.
3. ECMWF reforecast `tp/z500` already exists separately and may be enough for a
   hindcast-style ECMWF climatology/skill track, but it is not a replacement for
   UKMO/NCEP operational files unless the study design switches to provider
   reforecast products.
