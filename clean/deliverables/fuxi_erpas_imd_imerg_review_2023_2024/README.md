# FuXi-S2S versus ERPAS: IMD and IMERG review package

## What this analysis answers

This package tests whether FuXi-S2S represents the observed weekly Indian
rainfall-anomaly pattern better than ERPAS, and whether the result is robust to
using IMD gauges or IMERG Final V07B as the verifying reference.

The comparison contains 31 paired JJAS starts across 2023--2024: 17 in 2023
and 14 in 2024. Each ERPAS Wednesday initialization is paired with the
preceding-Monday FuXi initialization. Both models verify over the same four
non-overlapping Thursday--Wednesday weeks on the same native-limited 1.5-degree
grid and fixed 169-cell India support.

## Scientific method

- FuXi forecast anomalies use its native 2002--2021 lead/init-aware reforecast
  climatology.
- ERPAS forecast anomalies use its provider reforecast climatology.
- IMD observed anomalies use the IMD 1991--2020 daily climatology.
- IMERG observed anomalies use a fixed 2001--2022 Final V07B daily
  climatology. Verification years 2023--2024 are excluded.
- ACC is the area-weighted spatial Pearson correlation calculated separately
  for each initialization and week. The headline bar height is the arithmetic
  mean across the 31 cases.
- MAE, RMSE and bias use raw weekly forecast minus raw observed rainfall in
  mm/day; they do not depend on an anomaly climatology.
- Bar whiskers are the interquartile range across cases and are descriptive,
  not confidence intervals.
- A four-initialization paired moving-block bootstrap is saved as a robustness
  supplement. It is not used to add a significance claim to the figures.

The full date union is 6 June through 25 October, or 142 calendar days. This
is two days longer than the older 24-case pilot and is required for the last
2023 Week-4 verification period.

## Result and meeting story

FuXi-S2S has higher mean anomaly correlation and lower mean raw-rainfall MAE
than ERPAS at every lead week against both independent references. The result
is therefore not an artifact of choosing IMD gauges or IMERG satellite
rainfall.

| Reference | Metric | Week 1 | Week 2 | Week 3 | Week 4 |
|---|---|---:|---:|---:|---:|
| IMD | ACC: FuXi / ERPAS | 0.552 / 0.502 | 0.340 / 0.194 | 0.238 / 0.071 | 0.192 / 0.059 |
| IMERG | ACC: FuXi / ERPAS | 0.559 / 0.501 | 0.344 / 0.187 | 0.256 / 0.072 | 0.176 / 0.045 |
| IMD | MAE: FuXi / ERPAS (mm/day) | 3.623 / 3.796 | 4.195 / 4.748 | 4.076 / 4.912 | 4.066 / 4.552 |
| IMERG | MAE: FuXi / ERPAS (mm/day) | 3.551 / 3.721 | 4.134 / 4.622 | 4.024 / 4.842 | 4.083 / 4.497 |

The largest mean advantage occurs in Week 3: FuXi gains 0.166 ACC against IMD
and 0.183 against IMERG, while reducing MAE by 0.836 and 0.818 mm/day,
respectively. The paired moving-block 95% bootstrap intervals are wholly
positive for ACC and MAE at Weeks 2--3 against both references. The mean
advantages at Weeks 1 and 4 should be described more cautiously because some
ACC or MAE bootstrap intervals include zero.

Recommended presentation order:

1. Show the ACC bars as the headline: FuXi preserves the observed spatial
   anomaly pattern better, especially at Weeks 2--3.
2. Show the MAE bars as an independent magnitude check: the same ranking holds
   without using any climatology.
3. Show the typical spatial case (`paired_20240724`) to connect the scores to
   the rainfall-anomaly pattern. Keep the challenging and FuXi-favoring cases
   as transparent backup examples.

## Main outputs

- `figures/00_imerg_acc_mae_headline_2023_2024.*`: meeting-ready IMERG-only
  headline with ACC, MAE and RMSE in one side-by-side presentation figure.
- `figures/01_acc_grouped_bars_imd_imerg_2023_2024.*`: ACC bars against both
  references; higher is better.
- `figures/02_mae_grouped_bars_imd_imerg_2023_2024.*`: raw-rainfall MAE bars;
  lower is better.
- `figures/presentation_ic_maps/`: three objectively selected anomaly-map
  examples (nearest 25th, 50th and 75th percentile of four-week mean
  FuXi-minus-ERPAS IMERG ACC).
- `figures/all_ic_maps/` and
  `figures/04_all_31_imerg_spatial_anomaly_maps.pdf`: every initialization,
  with the exact shared reference anomaly scale at -20, -15, -10, -5, -2,
  2, 5, 10, 15 and 20 mm/day.
- `figures/native_imerg_erpas_maps/` and
  `figures/05_all_31_native_imerg_erpas_anomaly_maps.pdf`: native-resolution
  visual diagnostic with IMERG Final V07B retained at 0.1 degrees and ERPAS
  retained at 1.0 degree. No spatial interpolation is applied to either map;
  the small ACC badge remains the audited common-1.5-degree score because
  unequal native grids must not be scored directly against one another.
- `figures/smoothed_imerg_fuxi_erpas_maps/` and
  `figures/06_all_31_smoothed_imerg_fuxi_erpas_anomaly_maps.pdf`: the main
  three-row presentation atlas (IMERG, FuXi-S2S and ERPAS). Underlying
  anomalies retain their scientifically correct product/model grids and are
  bilinearly refined to 0.15 degrees only for a cleaner display. ACC badges
  remain the unchanged common-1.5-degree verification scores.
- `metrics/per_case_metrics_2023_2024.csv`: 496 rows, containing every
  reference x model x initialization x week score.
- `metrics/summary_metrics_2023_2024.csv`: plotted arithmetic means and IQRs.
- `metrics/paired_block_bootstrap_differences_2023_2024.csv`: paired
  difference robustness intervals.
- `data/processed/review_fields_2023_2024.nc`: the exact raw and anomaly fields
  used to reproduce every score and map.
- `logs/method_audit.json` and `logs/figure_audit.json`: fail-closed method and
  rendering checks.

## Run

```bash
bash deliverables/fuxi_erpas_imd_imerg_review_2023_2024/slurm/submit.sh
```

The first job is a resumable 22-year Slurm array. At most four years run at
once with two retrieval workers each to avoid overloading GES DISC. Each
failed subset is retried in progressively more conservative rounds, and each
calendar day is cached separately,
so rerunning the same submission resumes incomplete years. Aggregation starts
only if all 22 year tasks pass the exact 142-day and 48-half-hour-count checks;
the scoring and plotting job starts only after aggregation passes.

GPU hardware is not requested because NetCDF/GRIB remapping, HTTP retrieval,
and Matplotlib rendering are CPU/I/O workloads. The `gpu_prio` partition is
used only as the available priority queue.
