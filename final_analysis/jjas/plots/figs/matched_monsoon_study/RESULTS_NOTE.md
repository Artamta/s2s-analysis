# JJAS 2019 Matched-Window FuXi-S2S vs ECMWF-S2S Note

## Design

FuXi-S2S and ECMWF-S2S use offset hindcast initialization calendars in the
available archives, so this analysis matches forecasts by ERA5 valid date
window. Each FuXi init and lead week is paired with the nearest ECMWF init/lead
range that verifies the same dates.

Sample:
- FuXi compact inits: 12, from 2019-06-20 to 2019-07-28
- Matched forecast windows: 72
- Variables: TP and Z500
- Regions: All India plus four IMD homogeneous regions

## Headline All-India Result

TP:
- FuXi mean PCC: 0.66; ECMWF mean PCC: 0.63
- FuXi mean RMSE: 5.90 mm/day; ECMWF mean RMSE: 6.24 mm/day
- Mean paired FuXi PCC gain: +0.03
- Mean paired FuXi RMSE reduction: +0.34 mm/day
- FuXi wins 64% of paired All-India windows by PCC

Z500:
- FuXi mean PCC: 0.72; ECMWF mean PCC: 0.75
- FuXi mean RMSE: 9.65 gpm; ECMWF mean RMSE: 9.07 gpm
- Mean paired FuXi PCC gain: -0.03
- Mean paired FuXi RMSE reduction: -0.58 gpm
- FuXi wins 49% of paired All-India windows by PCC

## Interpretation

For this compact JJAS 2019 sample, FuXi shows a modest rainfall advantage over
ECMWF when comparisons are made on the same valid dates. The advantage is
strongest at early leads and weakens by weeks 4-6. For Z500, FuXi is competitive
at early leads, but ECMWF is stronger overall, especially at later leads.

Use the paired-delta plots and regional heatmap for the main figure set. The
case-level heatmaps are better treated as audit or supplementary figures.
