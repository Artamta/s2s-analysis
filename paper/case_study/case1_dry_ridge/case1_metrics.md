# Case Study 1 — Spatial skill over India (IMD-union) land points (cosine-weighted, anomalies)

Valid week 2026-02-12 .. 2026-02-18. Units: Z500 gpm, TP mm/day, T2M deg C.


> **Caveats.** PCC is the bias-insensitive (mean-removed) pattern metric; RMSE/Bias also absorb systematic offsets.
> * **TP**: this is a near-total-dry week, so absolute precip ~0 for both model and ERA5 and the anomaly collapses to -climatology; FuXi's near-perfect TP PCC reflects its dry/smooth output coinciding with that degenerate field, not necessarily forecast skill.
> * **T2M**: anomalies are referenced to ERA5 climatology. ECMWF/NCEP supply only 6-h max/min, so T2M is a (mx2t6+mn2t6)/2 proxy; FuXi/op models carry a model-climate cold offset (~-4 to -5 deg C) absent in ERA5-calibrated Spire. Read the warm-core skill from PCC.


## Z500

| Model | Lead | PCC | RMSE | Bias |
|---|---|---|---|---|
| Spire | Week-1 lead | +0.993 | 7.730 | +7.220 |
| FuXi-S2S | Week-1 lead | +0.981 | 7.271 | -5.557 |
| ECMWF | Week-1 lead | +0.994 | 2.878 | +0.993 |
| NCEP | Week-1 lead | +0.973 | 21.969 | -21.133 |
| Spire | Week-2 lead | +0.911 | 12.404 | +7.912 |
| FuXi-S2S | Week-2 lead | +0.682 | 35.284 | -32.237 |
| ECMWF | Week-2 lead | +0.847 | 18.777 | -14.029 |
| NCEP | Week-2 lead | +0.932 | 32.649 | -31.873 |

## TP

| Model | Lead | PCC | RMSE | Bias |
|---|---|---|---|---|
| Spire | Week-1 lead | +0.986 | 0.436 | +0.227 |
| FuXi-S2S | Week-1 lead | +1.000 | 0.010 | +0.000 |
| ECMWF | Week-1 lead | +0.994 | 0.239 | +0.105 |
| NCEP | Week-1 lead | +0.982 | 0.374 | +0.153 |
| Spire | Week-2 lead | +0.994 | 0.488 | +0.289 |
| FuXi-S2S | Week-2 lead | +1.000 | 0.011 | -0.000 |
| ECMWF | Week-2 lead | +0.996 | 0.286 | +0.162 |
| NCEP | Week-2 lead | +0.953 | 0.669 | +0.425 |

## T2M

| Model | Lead | PCC | RMSE | Bias |
|---|---|---|---|---|
| Spire | Week-1 lead | +0.833 | 0.756 | +0.072 |
| FuXi-S2S | Week-1 lead | +0.122 | 5.453 | -4.839 |
| ECMWF | Week-1 lead | +0.124 | 5.808 | -5.160 |
| NCEP | Week-1 lead | +0.282 | 5.481 | -4.351 |
| Spire | Week-2 lead | +0.736 | 1.131 | +0.164 |
| FuXi-S2S | Week-2 lead | +0.142 | 6.067 | -5.481 |
| ECMWF | Week-2 lead | +0.001 | 6.422 | -5.813 |
| NCEP | Week-2 lead | +0.051 | 5.386 | -4.360 |
