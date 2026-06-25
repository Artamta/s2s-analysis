# Matched-Valid-Window JJAS Monsoon Study

Compact FuXi init dates used: 12
2019-06-20, 2019-06-23, 2019-06-27, 2019-06-30, 2019-07-04, 2019-07-07, 2019-07-11, 2019-07-14, 2019-07-18, 2019-07-21, 2019-07-25, 2019-07-28

Matched forecast windows scored: 72

## All-India Mean Skill

### TP
- FuXi: PCC 0.66, RMSE 5.90, bias -2.15
- ECMWF: PCC 0.63, RMSE 6.24, bias -1.28

### Z500
- FuXi: PCC 0.72, RMSE 9.65, bias -2.39
- ECMWF: PCC 0.75, RMSE 9.07, bias -0.78

## Paired FuXi Advantage
- TP: mean PCC gain +0.03; mean RMSE reduction +0.34; FuXi wins 64% of All-India paired windows by PCC.
- Z500: mean PCC gain -0.03; mean RMSE reduction -0.58; FuXi wins 49% of All-India paired windows by PCC.

## Lead-Week Coverage
- TP: W1 n=12, W2 n=12, W3 n=12, W4 n=12, W5 n=12, W6 n=12
- Z500: W1 n=12, W2 n=12, W3 n=12, W4 n=12, W5 n=12, W6 n=12

## Calendar Handling
FuXi and ECMWF initialization dates are offset. Metrics and figures match
forecasts by ERA5 valid date window, not by nominal lead week alone.
