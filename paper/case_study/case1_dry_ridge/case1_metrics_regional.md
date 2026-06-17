# Case Study 1 — Per-IMD-region spatial skill (cosine-weighted, anomalies)

Valid week 2026-02-12 .. 2026-02-18. Units: Z500 gpm, TP mm/day, T2M deg C.


> **Note.** This was a Northwest/Central-India event, so those two regions are the diagnostic ones. PCC over the smaller S.Peninsula (~22 cells) and East/NE (~19 cells) regions is noisy — read it with care. T2M bias caveat (op-model proxy / model-climate offset) from the all-India table still applies.


## Z500

| Region | Model | Lead | PCC | RMSE | Bias | N |
|---|---|---|---|---|---|---|
| Northwest | Spire | Week-1 lead | +0.985 | 6.784 | +6.022 | 43 |
| Northwest | FuXi-S2S | Week-1 lead | +0.957 | 9.506 | -7.909 | 43 |
| Northwest | ECMWF | Week-1 lead | +0.998 | 2.825 | +1.629 | 43 |
| Northwest | NCEP | Week-1 lead | +0.941 | 27.038 | -26.315 | 43 |
| Central | Spire | Week-1 lead | +0.991 | 7.211 | +7.137 | 42 |
| Central | FuXi-S2S | Week-1 lead | +0.935 | 6.145 | -5.641 | 42 |
| Central | ECMWF | Week-1 lead | +0.968 | 1.966 | -0.867 | 42 |
| Central | NCEP | Week-1 lead | +0.853 | 19.260 | -18.987 | 42 |
| S. Peninsula | Spire | Week-1 lead | +0.995 | 10.596 | +10.349 | 22 |
| S. Peninsula | FuXi-S2S | Week-1 lead | +0.975 | 2.084 | +0.685 | 22 |
| S. Peninsula | ECMWF | Week-1 lead | +0.996 | 0.932 | +0.276 | 22 |
| S. Peninsula | NCEP | Week-1 lead | +0.993 | 15.403 | -15.218 | 22 |
| East/NE | Spire | Week-1 lead | +0.806 | 6.485 | +6.131 | 19 |
| East/NE | FuXi-S2S | Week-1 lead | +0.914 | 8.119 | -7.996 | 19 |
| East/NE | ECMWF | Week-1 lead | +0.905 | 5.316 | +4.747 | 19 |
| East/NE | NCEP | Week-1 lead | +0.598 | 22.376 | -22.079 | 19 |
| Northwest | Spire | Week-2 lead | +0.858 | 15.409 | +12.058 | 43 |
| Northwest | FuXi-S2S | Week-2 lead | +0.759 | 40.834 | -39.141 | 43 |
| Northwest | ECMWF | Week-2 lead | +0.854 | 14.917 | -9.412 | 43 |
| Northwest | NCEP | Week-2 lead | +0.948 | 34.573 | -34.101 | 43 |
| Central | Spire | Week-2 lead | +0.450 | 9.064 | +3.415 | 42 |
| Central | FuXi-S2S | Week-2 lead | -0.286 | 34.445 | -32.353 | 42 |
| Central | ECMWF | Week-2 lead | +0.092 | 22.196 | -18.652 | 42 |
| Central | NCEP | Week-2 lead | +0.643 | 34.548 | -34.195 | 42 |
| S. Peninsula | Spire | Week-2 lead | +0.809 | 9.517 | +7.494 | 22 |
| S. Peninsula | FuXi-S2S | Week-2 lead | -0.907 | 18.252 | -15.117 | 22 |
| S. Peninsula | ECMWF | Week-2 lead | -0.852 | 11.498 | -8.393 | 22 |
| S. Peninsula | NCEP | Week-2 lead | -0.320 | 25.828 | -24.870 | 22 |
| East/NE | Spire | Week-2 lead | +0.730 | 14.625 | +9.650 | 19 |
| East/NE | FuXi-S2S | Week-2 lead | +0.637 | 39.883 | -38.198 | 19 |
| East/NE | ECMWF | Week-2 lead | +0.768 | 24.305 | -20.565 | 19 |
| East/NE | NCEP | Week-2 lead | +0.792 | 31.436 | -30.418 | 19 |

## TP

| Region | Model | Lead | PCC | RMSE | Bias | N |
|---|---|---|---|---|---|---|
| Northwest | Spire | Week-1 lead | +0.997 | 0.337 | +0.307 | 43 |
| Northwest | FuXi-S2S | Week-1 lead | +1.000 | 0.012 | -0.004 | 43 |
| Northwest | ECMWF | Week-1 lead | +0.998 | 0.137 | +0.106 | 43 |
| Northwest | NCEP | Week-1 lead | +0.957 | 0.560 | +0.314 | 43 |
| Central | Spire | Week-1 lead | +0.963 | 0.139 | +0.116 | 42 |
| Central | FuXi-S2S | Week-1 lead | +1.000 | 0.005 | +0.003 | 42 |
| Central | ECMWF | Week-1 lead | +0.987 | 0.048 | +0.021 | 42 |
| Central | NCEP | Week-1 lead | +1.000 | 0.005 | +0.000 | 42 |
| S. Peninsula | Spire | Week-1 lead | +0.986 | 0.152 | +0.099 | 22 |
| S. Peninsula | FuXi-S2S | Week-1 lead | +1.000 | 0.010 | +0.006 | 22 |
| S. Peninsula | ECMWF | Week-1 lead | +0.723 | 0.334 | +0.151 | 22 |
| S. Peninsula | NCEP | Week-1 lead | +0.590 | 0.387 | +0.177 | 22 |
| East/NE | Spire | Week-1 lead | +0.982 | 0.978 | +0.467 | 19 |
| East/NE | FuXi-S2S | Week-1 lead | +1.000 | 0.012 | -0.002 | 19 |
| East/NE | ECMWF | Week-1 lead | +0.998 | 0.445 | +0.239 | 19 |
| East/NE | NCEP | Week-1 lead | +0.999 | 0.262 | +0.121 | 19 |
| Northwest | Spire | Week-2 lead | +0.993 | 0.560 | +0.409 | 43 |
| Northwest | FuXi-S2S | Week-2 lead | +1.000 | 0.013 | -0.007 | 43 |
| Northwest | ECMWF | Week-2 lead | +0.997 | 0.336 | +0.252 | 43 |
| Northwest | NCEP | Week-2 lead | +0.947 | 0.951 | +0.768 | 43 |
| Central | Spire | Week-2 lead | +0.987 | 0.134 | +0.125 | 42 |
| Central | FuXi-S2S | Week-2 lead | +1.000 | 0.003 | -0.001 | 42 |
| Central | ECMWF | Week-2 lead | +0.993 | 0.044 | +0.028 | 42 |
| Central | NCEP | Week-2 lead | +0.859 | 0.221 | +0.148 | 42 |
| S. Peninsula | Spire | Week-2 lead | +0.856 | 0.301 | +0.180 | 22 |
| S. Peninsula | FuXi-S2S | Week-2 lead | +1.000 | 0.011 | +0.005 | 22 |
| S. Peninsula | ECMWF | Week-2 lead | +0.931 | 0.208 | +0.123 | 22 |
| S. Peninsula | NCEP | Week-2 lead | +0.044 | 0.836 | +0.559 | 22 |
| East/NE | Spire | Week-2 lead | +0.995 | 0.871 | +0.535 | 19 |
| East/NE | FuXi-S2S | Week-2 lead | +1.000 | 0.014 | +0.007 | 19 |
| East/NE | ECMWF | Week-2 lead | +0.997 | 0.492 | +0.320 | 19 |
| East/NE | NCEP | Week-2 lead | +0.999 | 0.224 | +0.144 | 19 |

## T2M

| Region | Model | Lead | PCC | RMSE | Bias | N |
|---|---|---|---|---|---|---|
| Northwest | Spire | Week-1 lead | +0.727 | 0.795 | +0.185 | 43 |
| Northwest | FuXi-S2S | Week-1 lead | -0.116 | 6.296 | -5.398 | 43 |
| Northwest | ECMWF | Week-1 lead | -0.144 | 6.597 | -5.625 | 43 |
| Northwest | NCEP | Week-1 lead | -0.152 | 5.877 | -3.379 | 43 |
| Central | Spire | Week-1 lead | +0.645 | 0.850 | +0.161 | 42 |
| Central | FuXi-S2S | Week-1 lead | +0.522 | 4.803 | -4.662 | 42 |
| Central | ECMWF | Week-1 lead | +0.502 | 4.802 | -4.644 | 42 |
| Central | NCEP | Week-1 lead | +0.602 | 4.607 | -4.312 | 42 |
| S. Peninsula | Spire | Week-1 lead | +0.702 | 0.557 | -0.364 | 22 |
| S. Peninsula | FuXi-S2S | Week-1 lead | +0.637 | 3.686 | -3.370 | 22 |
| S. Peninsula | ECMWF | Week-1 lead | +0.605 | 4.565 | -4.331 | 22 |
| S. Peninsula | NCEP | Week-1 lead | +0.504 | 4.895 | -4.684 | 22 |
| East/NE | Spire | Week-1 lead | +0.817 | 0.649 | +0.166 | 19 |
| East/NE | FuXi-S2S | Week-1 lead | +0.278 | 6.592 | -5.850 | 19 |
| East/NE | ECMWF | Week-1 lead | +0.037 | 7.265 | -6.356 | 19 |
| East/NE | NCEP | Week-1 lead | +0.191 | 6.922 | -6.147 | 19 |
| Northwest | Spire | Week-2 lead | +0.559 | 1.351 | +0.436 | 43 |
| Northwest | FuXi-S2S | Week-2 lead | -0.006 | 6.581 | -5.897 | 43 |
| Northwest | ECMWF | Week-2 lead | -0.156 | 7.418 | -6.623 | 43 |
| Northwest | NCEP | Week-2 lead | -0.227 | 6.484 | -4.682 | 43 |
| Central | Spire | Week-2 lead | +0.537 | 0.997 | -0.051 | 42 |
| Central | FuXi-S2S | Week-2 lead | +0.350 | 6.373 | -6.106 | 42 |
| Central | ECMWF | Week-2 lead | +0.255 | 5.644 | -5.421 | 42 |
| Central | NCEP | Week-2 lead | +0.319 | 4.127 | -3.765 | 42 |
| S. Peninsula | Spire | Week-2 lead | +0.628 | 0.672 | -0.480 | 22 |
| S. Peninsula | FuXi-S2S | Week-2 lead | +0.525 | 3.820 | -3.387 | 22 |
| S. Peninsula | ECMWF | Week-2 lead | +0.563 | 4.628 | -4.416 | 22 |
| S. Peninsula | NCEP | Week-2 lead | +0.403 | 4.068 | -3.852 | 22 |
| East/NE | Spire | Week-2 lead | +0.267 | 1.329 | +0.862 | 19 |
| East/NE | FuXi-S2S | Week-2 lead | +0.403 | 6.448 | -5.747 | 19 |
| East/NE | ECMWF | Week-2 lead | -0.046 | 7.564 | -6.681 | 19 |
| East/NE | NCEP | Week-2 lead | -0.108 | 6.590 | -5.647 | 19 |
