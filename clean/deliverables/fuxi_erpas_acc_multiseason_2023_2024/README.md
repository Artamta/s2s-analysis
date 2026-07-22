# FuXi-S2S versus ERPAS: paired multiseason rainfall ACC, 2023–2024

## Meeting result

The matched sample contains 101 ERPAS Wednesday issues and preceding-Monday
strict-00Z FuXi-S2S issues: 52 in 2023 and 49 in 2024. All scores use four
non-overlapping seven-day lead windows, the same 1.5-degree grid, and one fixed
169-cell India support that is finite for every verification date.

The clean seasonal story is not “one model always wins”:

- **MAM:** FuXi is higher at all four weeks. Combined differences are +0.11,
  +0.09, +0.20 and +0.17.
- **JJAS:** FuXi is higher at all four weeks. Using the common IMD anomaly
  baseline, the FuXi-minus-ERPAS ACC differences are +0.02, +0.13, +0.13 and
  +0.08. The direction is positive in both 2023 and 2024.
- **OND:** ERPAS is higher at all four weeks. Combined differences are -0.06,
  -0.08, -0.16 and -0.16. The direction is negative in both years.
- **JF:** the ranking is mixed. The FuXi-minus-ERPAS differences are +0.01,
  -0.12, -0.14 and +0.07 for Weeks 1--4.

This supports a scenario-dependent conclusion: FuXi is strongest for the
monsoon anomaly pattern in this sample, whereas ERPAS retains more post-monsoon
skill.

## Primary all-season method

The all-season result follows the `paper_v2` convention:

- Forecast and IMD truth anomalies both subtract the same IMD 1991–2020
  calendar-day climatology.
- ACC is computed spatially for each initialization and week with spherical
  cell-area × fixed India-fraction weights.
- The primary seasonal score is the arithmetic mean of per-initialization ACC,
  matching `paper_v2`. Fisher-z means are also included as a sensitivity.
- RMSE, MAE and bias are calculated from raw forecast minus IMD rainfall in
  mm/day.

The headline figures use four disjoint issue-month seasons: JF
(January–February), MAM (March–May), JJAS (June–September) and OND
(October–December). Their sample sizes are 17, 27, 31 and 26, summing exactly
to 101. The CSV also retains JFM (January–March) as a separate sensitivity
window for direct comparison with `paper_v2`; JFM is not used in the four-season
headline because it overlaps MAM in March.

## IMD homogeneous-region comparison

Regional scores use the four official IMD homogeneous rainfall-region masks:
Northwest India, Central India, South Peninsula and East & Northeast India.
The Survey-of-India-derived 0.25-degree masks cited to Pai et al. (2014) are
conservatively transferred to the 1.5-degree analysis grid. Boundary cells are
fractionally allocated, while forecasts, IMD truth, climatology, valid periods
and the ACC equation remain unchanged.

The regional result adds necessary nuance:

- FuXi's MAM advantage is widespread, especially over Northwest India and the
  South Peninsula at longer leads.
- During JJAS, FuXi is consistently higher over Northwest and Central India,
  but the South Peninsula and East & Northeast India have mixed lead-dependent
  rankings.
- During OND, the strongest ERPAS advantage is over East & Northeast India,
  reaching -0.36 FuXi-minus-ERPAS ACC at Week 4.

These regional differences are descriptive means across paired starts; they
should not be interpreted as significance tests.

## Why the sample remains n=101

The two-year maximum is 104 ERPAS weekly issues. Three 2024 issues cannot be
used because the matched preceding-Monday strict-00Z FuXi files are absent:
12 June, 19 June and 7 August. Substituting Thursday cycles would change the
forecast age and valid-time alignment, so it was not done. Forty-four ERPAS
issues exist in 2025, but that is an incomplete seasonal year and is therefore
not mixed into the balanced 2023--2024 headline.

## Native-resolution interpretation

The source grids were checked directly: FuXi-S2S is 1.5 degrees, ERPAS is
1.0 degree and IMD rainfall is 0.25 degree. The common 1.5-degree verification
grid is therefore the native-resolution-limited fair comparison: it preserves
FuXi's native grid and conservatively aggregates the finer ERPAS and IMD
fields. Computing each model's ACC on its own native grid would change the
spatial sample and would not be a like-for-like score. Regridding everything
to 1.0 degree would only interpolate FuXi and add no independent information.

## System-specific climatology check

The ready standardized FuXi model-climatology NetCDF is JJAS-only. Therefore a
second, clearly separated JJAS calculation uses:

- FuXi-S2S native 2002–2021 reforecast climatology;
- ERPAS provider 20-source climatology;
- IMD truth minus IMD 1991–2020 climatology.

Across 31 paired JJAS starts, FuXi remains higher at all four weeks. Mean ACC:

| Lead | FuXi-S2S | ERPAS | FuXi minus ERPAS |
|---|---:|---:|---:|
| Week 1 | 0.552 | 0.502 | +0.050 |
| Week 2 | 0.340 | 0.194 | +0.146 |
| Week 3 | 0.238 | 0.071 | +0.166 |
| Week 4 | 0.192 | 0.059 | +0.133 |

The spatial companion uses raw weekly rainfall MAE, which is independent of
the anomaly-climatology choice.  It maps `ERPAS MAE - FuXi-S2S MAE` at every
1.5-degree India grid cell.  FuXi has lower MAE over 52%, 79%, 87% and 84% of
represented India area in Weeks 1--4, respectively.  The corresponding
All-India mean MAE reductions are 0.17, 0.55, 0.84 and 0.49 mm/day.  These are
descriptive spatial means and are not marked as significance results.

JFM/OND are not labelled as system-specific-climatology scores because the
all-season FuXi climatology has not yet been standardized from the 2,080 raw
2002–2021 reforecast archives.

## ERPAS source-count qualification

ERPAS is stored as a provider-precomputed deterministic mean. Its GRIB
`numberOfForecastsInEnsemble` varies by initialization:

- 89 paired issues use 20 sources;
- 9 use 4 sources;
- one each uses 8, 12 and 19 sources.

JJAS has 30/31 issues at 20 sources and OND has 26/26 at 20. JFM has the most
mixed composition (15 at 20 sources and 11 at lower counts), so its ranking is
best described as exploratory.

## Files

- `metrics/acc_per_case_2023_2024.csv`: every method × case × model × week
  score, date window and ERPAS source count.
- `metrics/acc_summary_by_year_season.csv`: model summaries by method, year,
  named season and week.
- `metrics/acc_fuxi_minus_erpas_by_year_season.csv`: direct paired-model
  differences for plotting.
- `metrics/regional_acc_per_case_2023_2024.csv`: every region × method × case
  × model × week score.
- `metrics/regional_acc_summary_by_year_season.csv`: regional summaries by
  method, year, season, model and week.
- `metrics/regional_acc_fuxi_minus_erpas.csv`: direct regional model
  differences used by the heatmap.
- `figures/seasonal_acc_fuxi_vs_erpas_2023_2024.*`: main four-season curves.
- `figures/imd_homogeneous_region_acc_advantage_2023_2024.*`: four-season,
  four-region FuXi-minus-ERPAS ACC matrix.
- `figures/imd_briefing_all_india_acc_2023_2024.*`: restrained 16:9
  presentation version of the four-season All-India result.
- `figures/imd_briefing_homogeneous_regions_acc_2023_2024.*`: compact
  presentation scorecard for the four official IMD homogeneous regions.
- `figures/imd_briefing_jjas_all_india_and_regions_acc_2023_2024.*`: actual
  JJAS ACC curves for All India and all four official IMD regions.
- `figures/imd_briefing_all_india_and_regions_scorecard_2023_2024.*`:
  four-season lead-specific delta-ACC scorecard with All India as the first row.
- `figures/imd_briefing_all_seasons_all_india_regions_summary_2023_2024.*`:
  compact 2x2 headline summary using the Weeks 1--4 mean delta ACC for All
  India and all four official regions; lead-specific figures remain backups.
- `figures/imd_briefing_all_seasons_all_india_regions_line_2023_2024.*`:
  simple five-panel line plot of the Weeks 1--4 mean ACC across all four
  seasons for All India and each official IMD homogeneous region.
- `figures/imd_briefing_fuxi_jjas_climatology_robustness_2023_2024.*`:
  focused two-panel defense of FuXi's JJAS advantage under the common-IMD and
  system-specific forecast-climatology anomaly definitions.
- `figures/imd_briefing_fuxi_jjas_system_specific_acc_2023_2024.*`:
  single-panel headline ACC curve using the system-specific forecast
  climatologies.
- `figures/imd_briefing_jjas_spatial_mae_advantage_2023_2024.*`:
  four-week spatial map of local ERPAS-minus-FuXi MAE using the official India
  state boundary; positive/blue means lower FuXi error.
- `figures/imd_briefing_jjas_composite_anomaly_maps_2023_2024.*`:
  one-page IMD/FuXi/ERPAS comparison of the four composite weekly rainfall
  anomaly patterns across all 31 paired JJAS starts.
- `figures/imd_briefing_jjas_composite_anomaly_{imd,fuxi,erpas}_2023_2024.*`:
  separate large 2x2 source figures in the operational anomaly-map style.
- `data/processed/jjas_composite_anomaly_maps_2023_2024.nc` and
  `metrics/jjas_composite_anomaly_map_metrics_2023_2024.csv`: plotted composite
  fields plus the reproduced mean case-wise ACC values.
- `data/processed/jjas_spatial_mae_advantage_2023_2024.nc` and
  `metrics/jjas_spatial_mae_advantage_2023_2024.csv`: reproducible gridded
  fields and concise panel statistics for the spatial figure.
- `figures/year_season_acc_advantage_heatmap.*`: year-by-year robustness view.
- `figures/jjas_acc_system_specific_climatologies_2023_2024.*`: expanded
  31-start JJAS model-climatology result.
- `logs/method_audit.json`: paths, exclusions, support construction and QC.
- `scripts/build_acc_csv.py`: reproducible scoring pipeline.
- `scripts/plot_acc_csv.py`: figures generated only from the audited CSVs.
