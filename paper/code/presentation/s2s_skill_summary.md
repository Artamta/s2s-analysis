# S2S Forecast Verification — JFM 2026
**Season:** JFM 2026 | **Inits:** 13 (Jan 1 – Mar 26) | **Region:** India 1.5° | **Truth:** ERA5

> **Bold** = best model per week. Auto-commentary highlights key patterns and issues.


# 1. Deterministic — PCC (Pattern Correlation)
> Range: −1 to +1. Higher is better. >0.5 = generally skillful. >0.3 = marginally useful.


## 1.1 TP

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.84** | **0.69** | **0.56** | **0.43** | **0.41** | **0.36** |
| FuXi | 0.79 | 0.67 | 0.48 | 0.40 | 0.38 | 0.35 |
| ECMWF | 0.81 | 0.64 | 0.50 | 0.37 | 0.37 | 0.31 |
| NCEP | 0.75 | 0.55 | 0.45 | 0.34 | 0.40 | 0.33 |
| MME | 0.82 | 0.67 | 0.51 | 0.40 | 0.40 | 0.35 |
| Persistence | 0.49 | 0.41 | 0.36 | 0.17 | 0.30 | 0.27 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk3**


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.87** | **0.71** | 0.43 | 0.25 | **0.29** | 0.26 |
| FuXi | 0.72 | 0.62 | **0.55** | **0.43** | 0.24 | **0.30** |
| ECMWF | 0.87 | 0.69 | 0.33 | 0.18 | 0.25 | 0.19 |
| NCEP | 0.69 | 0.46 | 0.30 | 0.14 | 0.21 | 0.21 |
| MME | 0.84 | 0.71 | 0.42 | 0.27 | 0.25 | 0.26 |
| Persistence | 0.37 | 0.20 | 0.17 | -0.04 | 0.09 | 0.30 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk3**


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.72 | 0.64 | 0.57 | 0.46 | 0.48 | 0.41 |
| FuXi | 0.73 | 0.62 | 0.29 | 0.33 | 0.34 | 0.39 |
| ECMWF | **0.82** | **0.67** | 0.54 | 0.49 | 0.45 | 0.42 |
| NCEP | 0.74 | 0.59 | **0.61** | **0.55** | **0.64** | 0.49 |
| MME | 0.80 | 0.66 | 0.48 | 0.46 | 0.54 | 0.46 |
| Persistence | 0.61 | 0.55 | 0.53 | 0.47 | 0.55 | **0.55** |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.5) through **Wk5**


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.49** | **0.45** | 0.28 | **0.42** | 0.35 | 0.29 |
| FuXi | 0.36 | 0.40 | 0.29 | 0.32 | 0.36 | **0.33** |
| ECMWF | 0.43 | 0.42 | 0.31 | 0.39 | **0.37** | 0.30 |
| NCEP | 0.28 | 0.29 | **0.35** | 0.32 | 0.36 | 0.31 |
| MME | 0.39 | 0.41 | 0.31 | 0.38 | 0.37 | 0.31 |
| Persistence | 0.25 | 0.19 | 0.06 | 0.20 | 0.20 | 0.33 |
> 💬 **SPIRE** leads at Wk1 · no model exceeds 0.5 in any week — **not skillful**


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.87** | **0.72** | 0.63 | **0.52** | **0.46** | **0.40** |
| FuXi | 0.81 | 0.71 | 0.54 | 0.47 | 0.42 | 0.38 |
| ECMWF | 0.81 | 0.67 | 0.57 | 0.48 | 0.45 | 0.39 |
| NCEP | 0.81 | 0.62 | 0.57 | 0.47 | 0.44 | 0.38 |
| MME | 0.83 | 0.68 | 0.57 | 0.48 | 0.45 | 0.40 |
| Persistence | 0.68 | 0.62 | **0.65** | 0.38 | 0.45 | 0.33 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk4**


## 1.2 Z500

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.94** | **0.87** | 0.75 | **0.59** | 0.47 | **0.44** |
| FuXi | 0.82 | 0.50 | 0.04 | 0.07 | -0.28 | -0.11 |
| ECMWF | 0.94 | 0.84 | **0.77** | 0.50 | 0.32 | 0.15 |
| NCEP | 0.92 | 0.63 | 0.43 | 0.35 | 0.45 | 0.39 |
| MME | 0.94 | 0.84 | 0.68 | 0.53 | **0.49** | 0.39 |
| Persistence | 0.49 | 0.44 | 0.38 | 0.12 | -0.02 | 0.27 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk4**


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.94** | 0.74 | 0.52 | **0.40** | **0.44** | 0.37 |
| FuXi | 0.73 | 0.52 | 0.13 | 0.19 | -0.11 | 0.01 |
| ECMWF | 0.92 | 0.70 | **0.71** | 0.31 | 0.32 | 0.16 |
| NCEP | 0.83 | 0.67 | 0.34 | 0.31 | 0.37 | **0.40** |
| MME | 0.90 | **0.77** | 0.53 | 0.36 | 0.40 | 0.33 |
| Persistence | 0.39 | 0.23 | -0.02 | -0.02 | 0.04 | 0.37 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk3**


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.93 | **0.76** | **0.72** | **0.57** | 0.39 | **0.36** |
| FuXi | 0.81 | 0.37 | -0.03 | 0.04 | -0.24 | -0.11 |
| ECMWF | **0.94** | 0.75 | 0.66 | 0.52 | 0.33 | 0.14 |
| NCEP | 0.87 | 0.52 | 0.31 | 0.30 | **0.39** | 0.27 |
| MME | 0.93 | 0.73 | 0.51 | 0.53 | 0.39 | 0.28 |
| Persistence | 0.39 | 0.51 | 0.43 | 0.01 | -0.18 | 0.04 |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.5) through **Wk4**


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.83 | 0.70 | 0.46 | 0.12 | **0.30** | 0.28 |
| FuXi | 0.77 | 0.41 | 0.42 | 0.35 | 0.29 | 0.20 |
| ECMWF | 0.86 | **0.71** | **0.60** | **0.35** | 0.22 | 0.19 |
| NCEP | 0.86 | 0.28 | 0.25 | 0.07 | -0.09 | 0.01 |
| MME | **0.90** | 0.61 | 0.48 | 0.34 | 0.26 | **0.30** |
| Persistence | -0.06 | 0.37 | 0.02 | -0.33 | 0.11 | 0.03 |
> 💬 **MME** leads at Wk1 · at least one model skillful (>0.5) through **Wk3**


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.86 | 0.88 | 0.63 | 0.53 | 0.44 | 0.42 |
| FuXi | 0.77 | 0.65 | 0.41 | 0.03 | -0.04 | -0.12 |
| ECMWF | **0.94** | 0.86 | **0.74** | 0.54 | **0.49** | 0.47 |
| NCEP | 0.65 | 0.68 | 0.58 | 0.31 | 0.28 | 0.38 |
| MME | 0.88 | **0.89** | 0.70 | **0.58** | 0.46 | **0.48** |
| Persistence | 0.53 | 0.51 | 0.33 | 0.20 | 0.01 | 0.24 |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.5) through **Wk4**


## 1.3 T2M

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.74** | **0.55** | **0.41** | **0.31** | 0.24 | 0.23 |
| FuXi | 0.36 | 0.29 | 0.17 | 0.09 | 0.06 | 0.08 |
| ECMWF | 0.35 | 0.30 | 0.19 | 0.09 | 0.04 | 0.03 |
| NCEP | 0.39 | 0.35 | 0.28 | 0.25 | **0.25** | **0.24** |
| MME | 0.46 | 0.40 | 0.28 | 0.20 | 0.16 | 0.16 |
| Persistence | 0.58 | 0.45 | 0.28 | 0.12 | -0.05 | 0.09 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk2**


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.59** | 0.32 | 0.13 | 0.04 | -0.05 | 0.03 |
| FuXi | 0.39 | 0.36 | 0.26 | 0.15 | 0.11 | 0.12 |
| ECMWF | 0.36 | 0.34 | 0.25 | 0.13 | 0.07 | 0.09 |
| NCEP | 0.41 | 0.40 | **0.33** | **0.25** | **0.22** | 0.24 |
| MME | 0.43 | 0.39 | 0.30 | 0.20 | 0.14 | 0.17 |
| Persistence | 0.57 | **0.45** | 0.24 | 0.11 | 0.06 | **0.26** |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk1**


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.67** | **0.47** | **0.31** | **0.26** | **0.24** | **0.22** |
| FuXi | 0.35 | 0.30 | 0.16 | 0.13 | 0.13 | 0.14 |
| ECMWF | 0.35 | 0.23 | 0.12 | 0.07 | 0.07 | 0.06 |
| NCEP | 0.25 | 0.22 | 0.16 | 0.19 | 0.21 | 0.18 |
| MME | 0.42 | 0.34 | 0.20 | 0.17 | 0.18 | 0.16 |
| Persistence | 0.45 | 0.30 | 0.29 | 0.23 | 0.19 | 0.19 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk1**


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.54 | 0.37 | 0.19 | 0.10 | 0.05 | -0.02 |
| FuXi | 0.11 | 0.10 | 0.00 | -0.01 | 0.01 | -0.02 |
| ECMWF | 0.10 | 0.10 | 0.05 | 0.04 | 0.01 | -0.02 |
| NCEP | 0.07 | 0.10 | 0.02 | 0.02 | 0.02 | -0.03 |
| MME | 0.15 | 0.14 | 0.05 | 0.03 | 0.03 | -0.02 |
| Persistence | **0.62** | **0.47** | **0.39** | **0.34** | **0.27** | **0.19** |
> 💬 **Persistence** leads at Wk1 · at least one model skillful (>0.5) through **Wk1**


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.80** | 0.43 | 0.21 | 0.12 | 0.12 | 0.22 |
| FuXi | 0.32 | 0.33 | 0.29 | 0.27 | 0.23 | 0.21 |
| ECMWF | 0.24 | 0.22 | 0.21 | 0.20 | 0.17 | 0.18 |
| NCEP | 0.23 | 0.12 | 0.19 | 0.18 | 0.17 | 0.14 |
| MME | 0.32 | 0.26 | 0.24 | 0.22 | 0.20 | 0.19 |
| Persistence | 0.56 | **0.61** | **0.38** | **0.46** | **0.32** | **0.30** |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.5) through **Wk1**


# 2. Deterministic — RMSE
> Lower is better. Units: TP=mm/day, Z500=gpm, T2M=K.


## 2.1 TP

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.94** | **1.36** | **1.68** | **1.97** | **2.07** | **2.22** |
| FuXi | 1.20 | 1.59 | 1.95 | 2.18 | 2.24 | 2.37 |
| ECMWF | 1.08 | 1.54 | 1.89 | 2.21 | 2.30 | 2.50 |
| NCEP | 1.32 | 1.80 | 2.09 | 2.44 | 2.34 | 2.55 |
| MME | 1.03 | 1.45 | 1.79 | 2.08 | 2.14 | 2.31 |
| Persistence | 2.02 | 2.27 | 2.42 | 2.85 | 2.72 | 2.88 |
> 💬 **SPIRE** leads at Wk1


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.67 | 1.08 | 1.61 | 1.82 | 1.55 | 1.64 |
| FuXi | 0.94 | 1.29 | **1.51** | **1.48** | **1.45** | **1.58** |
| ECMWF | **0.61** | **1.08** | 1.61 | 1.81 | 1.54 | 1.69 |
| NCEP | 1.02 | 1.46 | 1.86 | 2.20 | 1.73 | 1.94 |
| MME | 0.68 | 1.10 | 1.56 | 1.73 | 1.49 | 1.60 |
| Persistence | 1.88 | 2.21 | 2.40 | 2.64 | 2.23 | 1.89 |
> 💬 **ECMWF** leads at Wk1


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.24 | 0.39 | 0.51 | 0.59 | 0.58 | 0.66 |
| FuXi | 0.29 | 0.46 | 0.67 | 0.55 | 0.49 | 0.56 |
| ECMWF | **0.18** | **0.30** | 0.41 | 0.48 | 0.46 | 0.56 |
| NCEP | 0.24 | 0.40 | **0.39** | **0.47** | **0.40** | 0.58 |
| MME | 0.19 | 0.33 | 0.45 | 0.49 | 0.45 | 0.55 |
| Persistence | 0.30 | 0.40 | 0.45 | 0.51 | 0.48 | **0.55** |
> 💬 **ECMWF** leads at Wk1


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.69** | **0.86** | 0.84 | **0.85** | 0.95 | 1.11 |
| FuXi | 0.78 | 0.98 | 0.93 | 0.99 | 0.99 | 1.11 |
| ECMWF | 0.72 | 0.90 | 0.83 | 0.87 | 0.96 | 1.10 |
| NCEP | 0.95 | 0.94 | 0.83 | 0.93 | 0.98 | 1.13 |
| MME | 0.71 | 0.87 | **0.83** | 0.88 | **0.95** | **1.08** |
| Persistence | 1.04 | 1.05 | 1.13 | 1.22 | 1.23 | 1.25 |
> 💬 **SPIRE** leads at Wk1


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.68** | **2.45** | **2.89** | **3.55** | **4.24** | **4.62** |
| FuXi | 2.19 | 2.92 | 3.78 | 4.44 | 4.87 | 5.16 |
| ECMWF | 2.14 | 3.08 | 3.59 | 4.33 | 4.94 | 5.37 |
| NCEP | 2.33 | 3.31 | 3.73 | 4.50 | 4.94 | 5.37 |
| MME | 1.98 | 2.77 | 3.29 | 3.97 | 4.52 | 4.91 |
| Persistence | 2.99 | 3.49 | 3.88 | 5.06 | 5.38 | 6.11 |
> 💬 **SPIRE** leads at Wk1


## 2.2 Z500

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 13.80 | 18.86 | **21.26** | **24.65** | **24.75** | 27.97 |
| FuXi | 9.04 | 23.90 | 40.03 | 34.08 | 33.61 | 30.45 |
| ECMWF | 8.39 | **14.74** | 21.84 | 26.64 | 28.82 | 30.03 |
| NCEP | 17.64 | 31.92 | 34.13 | 35.15 | 34.90 | 36.29 |
| MME | **6.88** | 16.77 | 24.28 | 26.11 | 27.03 | **27.76** |
| Persistence | 34.13 | 32.25 | 33.85 | 38.15 | 38.41 | 34.43 |
> 💬 **MME** leads at Wk1


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 13.05 | 23.29 | **23.75** | **27.13** | **30.56** | **35.99** |
| FuXi | 11.31 | 33.24 | 55.33 | 45.79 | 46.71 | 41.69 |
| ECMWF | 8.96 | **18.62** | 26.82 | 33.20 | 38.31 | 41.49 |
| NCEP | 20.00 | 36.69 | 37.59 | 38.64 | 39.04 | 42.74 |
| MME | **8.70** | 21.88 | 29.36 | 32.39 | 35.47 | 37.95 |
| Persistence | 41.96 | 40.56 | 41.30 | 48.81 | 51.12 | 41.00 |
> 💬 **MME** leads at Wk1


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 14.46 | 14.76 | **19.44** | 23.11 | **21.20** | 23.19 |
| FuXi | 7.34 | 17.75 | 30.17 | 25.07 | 24.22 | **22.22** |
| ECMWF | 8.13 | **11.81** | 19.76 | 23.04 | 23.92 | 23.31 |
| NCEP | 14.08 | 27.94 | 31.45 | 32.57 | 33.59 | 33.66 |
| MME | **5.80** | 13.56 | 22.02 | **22.61** | 22.91 | 22.53 |
| Persistence | 29.98 | 27.55 | 31.60 | 33.21 | 31.35 | 31.19 |
> 💬 **MME** leads at Wk1


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 13.72 | 11.56 | 13.68 | 15.50 | 14.55 | 14.27 |
| FuXi | 4.42 | 7.90 | 13.54 | 13.14 | 12.45 | 13.73 |
| ECMWF | 5.11 | **5.36** | **10.22** | 13.56 | 12.90 | 12.29 |
| NCEP | 12.93 | 21.96 | 24.34 | 22.98 | 23.38 | 23.21 |
| MME | **3.05** | 6.70 | 12.07 | **12.78** | **11.91** | **11.79** |
| Persistence | 16.54 | 15.12 | 18.10 | 18.27 | 17.27 | 20.53 |
> 💬 **MME** leads at Wk1


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 11.76 | 18.61 | 21.12 | 24.32 | **21.34** | 24.98 |
| FuXi | 8.53 | 20.87 | 36.16 | 31.12 | 28.75 | 22.40 |
| ECMWF | 8.83 | **14.74** | **18.42** | 23.96 | 24.04 | 24.58 |
| NCEP | 20.46 | 32.95 | 34.88 | 34.16 | 31.42 | 30.03 |
| MME | **6.70** | 14.99 | 22.54 | **21.87** | 21.97 | **21.63** |
| Persistence | 34.39 | 29.56 | 27.24 | 32.32 | 33.66 | 28.48 |
> 💬 **MME** leads at Wk1


## 2.3 T2M

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.85** | **1.38** | **1.46** | **1.63** | **1.78** | **1.88** |
| FuXi | 4.80 | 4.92 | 5.27 | 5.08 | 4.79 | 4.61 |
| ECMWF | 5.27 | 5.29 | 5.72 | 5.90 | 6.08 | 5.92 |
| NCEP | 4.70 | 4.71 | 5.25 | 5.41 | 5.44 | 5.38 |
| MME | 3.63 | 3.63 | 4.05 | 4.16 | 4.21 | 4.14 |
| Persistence | 1.82 | 2.48 | 3.08 | 4.03 | 5.01 | 5.65 |
> 💬 **SPIRE** leads at Wk1


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.07** | **1.85** | **1.83** | **2.03** | **2.24** | **2.30** |
| FuXi | 5.49 | 5.77 | 6.51 | 6.29 | 5.95 | 5.74 |
| ECMWF | 5.91 | 6.02 | 6.78 | 6.98 | 7.31 | 7.13 |
| NCEP | 5.07 | 5.29 | 5.90 | 5.92 | 6.03 | 6.05 |
| MME | 3.97 | 4.08 | 4.73 | 4.81 | 4.94 | 4.87 |
| Persistence | 2.47 | 3.22 | 3.84 | 5.07 | 6.30 | 7.00 |
> 💬 **SPIRE** leads at Wk1


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.76** | **1.03** | **1.07** | **1.35** | **1.47** | **1.57** |
| FuXi | 4.29 | 4.29 | 4.58 | 4.35 | 4.03 | 3.82 |
| ECMWF | 4.51 | 4.37 | 4.77 | 5.00 | 5.15 | 4.96 |
| NCEP | 3.87 | 3.61 | 4.26 | 4.61 | 4.62 | 4.52 |
| MME | 3.13 | 3.01 | 3.45 | 3.61 | 3.63 | 3.55 |
| Persistence | 1.43 | 2.09 | 2.76 | 3.69 | 4.61 | 5.34 |
> 💬 **SPIRE** leads at Wk1


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.61** | **0.81** | **1.01** | **1.18** | **1.34** | **1.52** |
| FuXi | 3.62 | 3.40 | 3.18 | 3.01 | 2.94 | 2.92 |
| ECMWF | 4.35 | 4.37 | 4.46 | 4.46 | 4.53 | 4.52 |
| NCEP | 4.18 | 4.05 | 4.62 | 4.84 | 4.97 | 5.01 |
| MME | 3.11 | 3.06 | 3.22 | 3.28 | 3.36 | 3.42 |
| Persistence | 0.74 | 1.23 | 1.83 | 2.43 | 3.06 | 3.64 |
> 💬 **SPIRE** leads at Wk1


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.69** | **1.25** | **1.42** | **1.41** | **1.45** | **1.52** |
| FuXi | 5.40 | 5.53 | 5.55 | 5.52 | 5.21 | 4.98 |
| ECMWF | 6.29 | 6.32 | 6.36 | 6.57 | 6.52 | 6.27 |
| NCEP | 5.86 | 5.89 | 6.16 | 6.28 | 6.13 | 5.84 |
| MME | 4.37 | 4.29 | 4.43 | 4.56 | 4.44 | 4.23 |
| Persistence | 1.58 | 2.29 | 2.79 | 3.46 | 4.36 | 4.80 |
> 💬 **SPIRE** leads at Wk1


# 3. Deterministic — Mean Bias
> Closer to 0 is better. Positive = forecast too high. Units: TP=mm/day, Z500=gpm, T2M=K.


## 3.1 TP

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.01** | 0.07 | 0.13 | 0.06 | 0.07 | **0.02** |
| FuXi | -0.10 | 0.05 | **-0.01** | -0.35 | -0.28 | -0.45 |
| ECMWF | -0.07 | -0.13 | -0.14 | -0.18 | -0.24 | -0.40 |
| NCEP | -0.15 | -0.13 | -0.16 | **-0.05** | **0.04** | 0.08 |
| MME | -0.08 | **-0.04** | -0.05 | -0.13 | -0.10 | -0.19 |
| Persistence | -0.09 | -0.25 | -0.29 | -0.38 | -0.39 | -0.57 |
> 💬 **SPIRE** leads at Wk1


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.16 | 0.20 | 0.20 | 0.16 | 0.36 | 0.23 |
| FuXi | -0.06 | 0.47 | 0.19 | -0.26 | 0.07 | -0.11 |
| ECMWF | 0.07 | **0.04** | **0.04** | **0.09** | 0.19 | **-0.02** |
| NCEP | -0.08 | 0.16 | 0.15 | 0.46 | 0.70 | 0.79 |
| MME | **0.02** | 0.22 | 0.15 | 0.11 | 0.33 | 0.22 |
| Persistence | -0.04 | -0.23 | -0.25 | -0.27 | **-0.00** | -0.12 |
> 💬 **MME** leads at Wk1


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.10 | 0.20 | 0.30 | 0.30 | 0.30 | 0.32 |
| FuXi | 0.11 | 0.20 | 0.31 | 0.10 | 0.03 | -0.09 |
| ECMWF | **-0.02** | -0.02 | **0.03** | **0.03** | **-0.01** | -0.09 |
| NCEP | -0.03 | **-0.01** | -0.07 | -0.07 | 0.02 | 0.08 |
| MME | 0.04 | 0.09 | 0.15 | 0.09 | 0.09 | **0.06** |
| Persistence | -0.02 | -0.10 | -0.11 | -0.13 | -0.14 | -0.20 |
> 💬 **ECMWF** leads at Wk1


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | -0.03 | 0.03 | 0.15 | 0.15 | 0.18 | 0.29 |
| FuXi | -0.08 | -0.05 | -0.13 | -0.15 | -0.08 | -0.20 |
| ECMWF | 0.08 | 0.05 | 0.01 | 0.06 | 0.04 | **0.00** |
| NCEP | 0.09 | -0.12 | -0.10 | -0.12 | -0.03 | 0.09 |
| MME | **0.02** | **-0.02** | -0.01 | **-0.02** | **0.03** | 0.04 |
| Persistence | -0.04 | -0.07 | **0.01** | -0.02 | -0.06 | -0.14 |
> 💬 **MME** leads at Wk1


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | -0.46 | **-0.44** | **-0.46** | **-0.80** | **-1.23** | **-1.46** |
| FuXi | -0.70 | -1.11 | -1.02 | -1.83 | -1.99 | -2.35 |
| ECMWF | -0.68 | -0.98 | -1.15 | -1.54 | -2.06 | -2.42 |
| NCEP | -0.90 | -1.04 | -1.14 | -1.06 | -1.28 | -1.51 |
| MME | -0.69 | -0.89 | -0.94 | -1.31 | -1.64 | -1.93 |
| Persistence | **-0.39** | -0.88 | -1.19 | -1.65 | -2.25 | -2.88 |
> 💬 **Persistence** leads at Wk1


## 3.2 Z500

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 12.93 | 10.88 | 6.50 | **6.61** | **6.66** | 7.41 |
| FuXi | -1.46 | -16.28 | -28.45 | -22.26 | -18.38 | -9.74 |
| ECMWF | 4.94 | **0.14** | -5.15 | -9.76 | -8.67 | **-3.06** |
| NCEP | -14.31 | -24.61 | -26.31 | -27.32 | -27.16 | -25.62 |
| MME | 0.52 | -7.47 | -13.36 | -13.18 | -11.89 | -7.75 |
| Persistence | **0.03** | -2.69 | **-4.37** | -8.98 | -14.00 | -15.61 |
> 💬 **Persistence** leads at Wk1 · 🟡 **SPIRE** has large Wk1 bias (+12.9) · 🟡 **ECMWF** has large Wk1 bias (+4.9) · 🟡 **NCEP** has large Wk1 bias (-14.3)


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 11.96 | 11.31 | **4.43** | **4.56** | **3.23** | **4.92** |
| FuXi | -5.04 | -28.55 | -51.01 | -40.92 | -38.13 | -26.83 |
| ECMWF | 3.89 | **-3.00** | -9.76 | -16.97 | -17.78 | -10.13 |
| NCEP | -15.33 | -24.22 | -24.13 | -26.44 | -26.49 | -24.00 |
| MME | **-1.13** | -11.11 | -20.12 | -19.94 | -19.79 | -14.01 |
| Persistence | -1.33 | -4.86 | -6.78 | -13.80 | -23.46 | -27.19 |
> 💬 **MME** leads at Wk1 · 🟡 **SPIRE** has large Wk1 bias (+12.0) · 🟡 **FuXi** has large Wk1 bias (-5.0) · 🟡 **ECMWF** has large Wk1 bias (+3.9) · 🟡 **NCEP** has large Wk1 bias (-15.3)


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 14.19 | 9.60 | 5.62 | **6.13** | **6.63** | 6.30 |
| FuXi | 0.80 | -11.66 | -20.39 | -13.77 | -9.45 | **-2.20** |
| ECMWF | 5.74 | **1.09** | -5.20 | -9.18 | -7.52 | -2.56 |
| NCEP | -12.51 | -24.80 | -28.74 | -29.59 | -30.69 | -30.91 |
| MME | 2.05 | -6.44 | -12.18 | -11.60 | -10.26 | -7.34 |
| Persistence | **0.63** | -1.25 | **-4.11** | -8.12 | -11.90 | -12.82 |
> 💬 **Persistence** leads at Wk1 · 🟡 **SPIRE** has large Wk1 bias (+14.2) · 🟡 **ECMWF** has large Wk1 bias (+5.7) · 🟡 **NCEP** has large Wk1 bias (-12.5) · 🟡 **MME** has large Wk1 bias (+2.1)


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 13.58 | 10.84 | 9.75 | 10.61 | 10.56 | 9.81 |
| FuXi | 2.37 | -2.86 | -3.47 | **0.36** | 3.67 | 6.82 |
| ECMWF | 3.88 | 1.11 | **-1.50** | -1.82 | **0.28** | 2.76 |
| NCEP | -12.71 | -21.37 | -23.73 | -21.75 | -22.11 | -22.25 |
| MME | 1.78 | -3.07 | -4.74 | -3.15 | -1.90 | **-0.71** |
| Persistence | **1.56** | **0.10** | -1.56 | -3.14 | -3.95 | -3.85 |
> 💬 **Persistence** leads at Wk1 · 🟡 **SPIRE** has large Wk1 bias (+13.6) · 🟡 **FuXi** has large Wk1 bias (+2.4) · 🟡 **ECMWF** has large Wk1 bias (+3.9) · 🟡 **NCEP** has large Wk1 bias (-12.7)


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 11.35 | 12.86 | 8.99 | 7.20 | 9.38 | 12.41 |
| FuXi | -3.58 | -16.76 | -28.71 | -29.05 | -23.11 | -10.20 |
| ECMWF | 6.75 | **3.60** | **0.49** | **-5.22** | **-2.58** | 3.97 |
| NCEP | -18.16 | -29.05 | -28.75 | -30.98 | -26.87 | -21.25 |
| MME | -0.91 | -7.34 | -12.00 | -14.51 | -10.79 | **-3.77** |
| Persistence | **-0.25** | -4.69 | -3.23 | -7.68 | -10.63 | -11.32 |
> 💬 **Persistence** leads at Wk1 · 🟡 **SPIRE** has large Wk1 bias (+11.3) · 🟡 **FuXi** has large Wk1 bias (-3.6) · 🟡 **ECMWF** has large Wk1 bias (+6.8) · 🟡 **NCEP** has large Wk1 bias (-18.2)


## 3.3 T2M

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **-0.01** | **0.23** | **-0.18** | **-0.41** | **-0.63** | **-0.74** |
| FuXi | -4.17 | -4.18 | -4.53 | -4.31 | -4.00 | -3.69 |
| ECMWF | -4.62 | -4.59 | -5.04 | -5.21 | -5.38 | -5.16 |
| NCEP | -3.47 | -3.48 | -4.07 | -4.29 | -4.37 | -4.22 |
| MME | -3.07 | -3.01 | -3.45 | -3.55 | -3.60 | -3.45 |
| Persistence | -0.87 | -1.68 | -2.47 | -3.43 | -4.42 | -5.17 |
> 💬 **SPIRE** leads at Wk1 · 🟡 **FuXi** has large Wk1 bias (-4.2K) · 🟡 **ECMWF** has large Wk1 bias (-4.6K) · 🟡 **NCEP** has large Wk1 bias (-3.5K) · 🟡 **MME** has large Wk1 bias (-3.1K)


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.02** | **0.40** | **-0.18** | **-0.40** | **-0.72** | **-0.82** |
| FuXi | -4.63 | -4.81 | -5.75 | -5.45 | -5.07 | -4.71 |
| ECMWF | -4.93 | -5.00 | -5.85 | -6.04 | -6.40 | -6.13 |
| NCEP | -2.65 | -2.96 | -3.58 | -3.64 | -3.84 | -3.70 |
| MME | -3.05 | -3.09 | -3.84 | -3.88 | -4.01 | -3.84 |
| Persistence | -1.11 | -2.13 | -3.00 | -4.25 | -5.60 | -6.60 |
> 💬 **SPIRE** leads at Wk1 · 🟡 **FuXi** has large Wk1 bias (-4.6K) · 🟡 **ECMWF** has large Wk1 bias (-4.9K) · 🟡 **NCEP** has large Wk1 bias (-2.7K) · 🟡 **MME** has large Wk1 bias (-3.0K)


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.07** | **0.25** | **-0.20** | **-0.46** | **-0.67** | **-0.85** |
| FuXi | -4.10 | -4.05 | -4.32 | -4.08 | -3.80 | -3.48 |
| ECMWF | -4.31 | -4.15 | -4.56 | -4.79 | -4.95 | -4.74 |
| NCEP | -3.39 | -3.23 | -3.91 | -4.27 | -4.34 | -4.21 |
| MME | -2.93 | -2.80 | -3.25 | -3.40 | -3.44 | -3.32 |
| Persistence | -0.89 | -1.65 | -2.50 | -3.45 | -4.42 | -5.13 |
> 💬 **SPIRE** leads at Wk1 · 🟡 **FuXi** has large Wk1 bias (-4.1K) · 🟡 **ECMWF** has large Wk1 bias (-4.3K) · 🟡 **NCEP** has large Wk1 bias (-3.4K) · 🟡 **MME** has large Wk1 bias (-2.9K)


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **-0.29** | **-0.43** | **-0.66** | **-0.83** | **-1.02** | **-1.23** |
| FuXi | -3.26 | -2.99 | -2.74 | -2.55 | -2.50 | -2.44 |
| ECMWF | -4.05 | -4.07 | -4.16 | -4.15 | -4.21 | -4.18 |
| NCEP | -3.88 | -3.76 | -4.33 | -4.55 | -4.68 | -4.69 |
| MME | -2.87 | -2.81 | -2.97 | -3.02 | -3.10 | -3.14 |
| Persistence | -0.49 | -1.02 | -1.65 | -2.26 | -2.87 | -3.42 |
> 💬 **SPIRE** leads at Wk1 · 🟡 **FuXi** has large Wk1 bias (-3.3K) · 🟡 **ECMWF** has large Wk1 bias (-4.1K) · 🟡 **NCEP** has large Wk1 bias (-3.9K) · 🟡 **MME** has large Wk1 bias (-2.9K)


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.04** | **0.60** | **0.44** | **0.18** | **0.17** | **0.27** |
| FuXi | -4.48 | -4.57 | -4.56 | -4.51 | -4.00 | -3.46 |
| ECMWF | -5.35 | -5.37 | -5.42 | -5.66 | -5.59 | -5.24 |
| NCEP | -4.91 | -4.83 | -5.19 | -5.39 | -5.24 | -4.78 |
| MME | -3.68 | -3.54 | -3.68 | -3.84 | -3.67 | -3.30 |
| Persistence | -0.79 | -1.62 | -2.25 | -3.06 | -3.79 | -4.34 |
> 💬 **SPIRE** leads at Wk1 · 🟡 **FuXi** has large Wk1 bias (-4.5K) · 🟡 **ECMWF** has large Wk1 bias (-5.4K) · 🟡 **NCEP** has large Wk1 bias (-4.9K) · 🟡 **MME** has large Wk1 bias (-3.7K)


# 4. Deterministic — MSSS vs Climatology
> Positive = better than climatology. 0 = same as climatology. Negative = worse.


## 4.1 TP

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.69** | **0.44** | **0.26** | **0.05** | **0.02** | **-0.04** |
| FuXi | 0.45 | 0.00 | -0.20 | -0.27 | -0.32 | -0.30 |
| ECMWF | 0.54 | 0.17 | -0.03 | -0.27 | -0.34 | -0.47 |
| NCEP | 0.29 | -0.01 | -0.23 | -0.77 | -0.52 | -0.62 |
| MME | 0.60 | 0.31 | 0.13 | -0.10 | -0.15 | -0.21 |
| Persistence | -0.66 | -1.09 | -0.85 | -1.45 | -1.18 | -1.11 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk3** · 🔴 **Persistence** deeply negative (Wk2, Wk4, Wk5, Wk6) — possible issue


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.66 | **0.53** | -0.02 | -0.23 | -0.07 | -0.06 |
| FuXi | -0.01 | -1.32 | -0.59 | **0.18** | **0.02** | -0.16 |
| ECMWF | **0.77** | 0.52 | -0.03 | -0.29 | -0.08 | -0.07 |
| NCEP | 0.37 | 0.03 | -0.32 | -2.37 | -1.38 | -1.76 |
| MME | 0.66 | 0.27 | **0.04** | -0.14 | -0.05 | **-0.05** |
| Persistence | -2.47 | -1.30 | -1.85 | -2.30 | -2.90 | -1.45 |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk4** · 🔴 **FuXi** deeply negative (Wk2) — possible issue · 🔴 **NCEP** deeply negative (Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.49 | 0.26 | -0.11 | -0.47 | -0.30 | -0.39 |
| FuXi | -0.40 | -0.34 | -2.19 | -0.63 | -0.09 | 0.01 |
| ECMWF | **0.67** | **0.47** | **0.29** | **0.14** | 0.15 | 0.11 |
| NCEP | 0.39 | 0.03 | 0.27 | 0.04 | **0.30** | -0.05 |
| MME | 0.64 | 0.40 | 0.05 | 0.05 | 0.21 | **0.15** |
| Persistence | 0.12 | 0.06 | 0.03 | -0.27 | -0.01 | -0.10 |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk6** · 🔴 **FuXi** deeply negative (Wk3) — possible issue


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.10** | **-0.07** | -0.30 | **-0.06** | -0.29 | **-0.36** |
| FuXi | -0.28 | -0.61 | -0.60 | -0.47 | -0.40 | -0.47 |
| ECMWF | -0.18 | -0.28 | **-0.28** | -0.17 | -0.31 | -0.43 |
| NCEP | -1.46 | -0.59 | -0.31 | -0.39 | -0.39 | -0.53 |
| MME | -0.18 | -0.24 | -0.29 | -0.19 | **-0.27** | -0.37 |
| Persistence | -4.08 | -1.00 | -4.42 | -2.87 | -2.64 | -1.27 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk1** · 🔴 **NCEP** deeply negative (Wk1) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk3, Wk4, Wk5, Wk6) — possible issue


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.70** | **0.42** | **0.37** | **0.15** | **-0.04** | **-0.18** |
| FuXi | 0.49 | 0.08 | -0.26 | -0.62 | -0.73 | -0.70 |
| ECMWF | 0.46 | -0.12 | -0.24 | -0.51 | -0.72 | -0.95 |
| NCEP | 0.22 | -0.24 | -0.55 | -0.94 | -0.95 | -1.02 |
| MME | 0.55 | 0.18 | 0.05 | -0.25 | -0.43 | -0.55 |
| Persistence | 0.00 | -1.09 | -0.81 | -1.88 | -1.29 | -1.63 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk4** · 🔴 **NCEP** deeply negative (Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk2, Wk4, Wk5, Wk6) — possible issue


## 4.2 Z500

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.58 | 0.28 | 0.13 | **-0.05** | **-0.11** | -0.21 |
| FuXi | 0.79 | -0.22 | -3.07 | -1.98 | -1.16 | -0.59 |
| ECMWF | 0.83 | **0.67** | **0.26** | -0.26 | -0.46 | -0.27 |
| NCEP | 0.41 | -1.17 | -1.49 | -1.81 | -2.07 | -2.13 |
| MME | **0.92** | 0.49 | -0.12 | -0.27 | -0.27 | **-0.18** |
| Persistence | -1.91 | -0.86 | -1.95 | -3.72 | -4.82 | -2.09 |
> 💬 **MME** leads at Wk1 · at least one model skillful (>0.1) through **Wk3** · 🔴 **FuXi** deeply negative (Wk3, Wk4, Wk5) — possible issue · 🔴 **NCEP** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk3, Wk4, Wk5, Wk6) — possible issue


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.53 | -0.52 | -0.50 | -0.69 | -1.03 | -1.14 |
| FuXi | 0.13 | -2.08 | -8.75 | -4.82 | -2.34 | -2.26 |
| ECMWF | 0.77 | **0.45** | **0.21** | **-0.53** | -0.89 | -0.64 |
| NCEP | 0.34 | -1.54 | -1.76 | -1.58 | -1.13 | -1.29 |
| MME | **0.85** | 0.18 | -0.42 | -0.64 | **-0.33** | **-0.32** |
| Persistence | -8.77 | -3.57 | -7.47 | -4.14 | -5.40 | -1.78 |
> 💬 **MME** leads at Wk1 · at least one model skillful (>0.1) through **Wk3** · 🔴 **SPIRE** deeply negative (Wk5, Wk6) — possible issue · 🔴 **FuXi** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.13 | 0.29 | **0.08** | **-0.38** | **0.07** | **-0.05** |
| FuXi | 0.79 | -0.92 | -5.64 | -4.99 | -2.51 | -1.03 |
| ECMWF | 0.74 | **0.46** | -0.30 | -1.56 | -1.76 | -0.63 |
| NCEP | -0.09 | -4.35 | -5.02 | -5.19 | -6.44 | -6.88 |
| MME | **0.90** | -0.01 | -1.39 | -1.63 | -1.39 | -0.98 |
| Persistence | -2.04 | -0.78 | -2.62 | -6.38 | -8.60 | -4.41 |
> 💬 **MME** leads at Wk1 · at least one model skillful (>0.1) through **Wk2** · 🔴 **FuXi** deeply negative (Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk4, Wk5) — possible issue · 🔴 **NCEP** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **MME** deeply negative (Wk3, Wk4, Wk5) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk3, Wk4, Wk5, Wk6) — possible issue


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | -2.75 | -0.34 | **-0.24** | **-0.63** | **-0.52** | **0.01** |
| FuXi | 0.27 | -2.67 | -5.99 | -6.03 | -2.61 | -0.45 |
| ECMWF | 0.52 | **0.62** | -1.62 | -4.32 | -2.07 | -0.16 |
| NCEP | -3.27 | -17.24 | -16.22 | -14.55 | -30.82 | -28.70 |
| MME | **0.85** | -1.05 | -3.05 | -3.65 | -2.72 | -1.03 |
| Persistence | -5.64 | -4.16 | -3.07 | -10.30 | -29.25 | -25.97 |
> 💬 **MME** leads at Wk1 · at least one model skillful (>0.1) through **Wk2** · 🔴 **SPIRE** deeply negative (Wk1) — possible issue · 🔴 **FuXi** deeply negative (Wk2, Wk3, Wk4, Wk5) — possible issue · 🔴 **ECMWF** deeply negative (Wk3, Wk4, Wk5) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **MME** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.14 | -2.02 | -2.11 | -2.75 | -2.03 | -2.21 |
| FuXi | 0.45 | -0.58 | -4.87 | -2.32 | -2.39 | -1.11 |
| ECMWF | 0.48 | 0.18 | **-0.20** | -0.84 | -1.42 | -1.33 |
| NCEP | -0.74 | -2.92 | -4.66 | -2.96 | -2.76 | -3.16 |
| MME | **0.80** | **0.41** | -0.40 | **0.04** | **-0.31** | **-0.36** |
| Persistence | -10.59 | -3.17 | -3.87 | -6.12 | -7.57 | -4.06 |
> 💬 **MME** leads at Wk1 · at least one model skillful (>0.1) through **Wk2** · 🔴 **SPIRE** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **FuXi** deeply negative (Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


## 4.3 T2M

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.62** | **0.11** | **0.00** | **-0.27** | **-0.44** | **-0.41** |
| FuXi | -10.53 | -10.85 | -12.69 | -12.83 | -10.26 | -8.35 |
| ECMWF | -13.28 | -13.69 | -15.79 | -16.63 | -17.07 | -14.80 |
| NCEP | -9.93 | -9.78 | -12.77 | -14.06 | -13.34 | -11.89 |
| MME | -5.64 | -5.53 | -7.13 | -7.86 | -7.43 | -6.52 |
| Persistence | -0.78 | -1.77 | -3.61 | -7.05 | -9.92 | -12.01 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk2** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **MME** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.59** | **-0.02** | **0.09** | **-0.21** | **-0.45** | **-0.24** |
| FuXi | -8.48 | -8.90 | -11.91 | -12.60 | -9.97 | -7.87 |
| ECMWF | -10.41 | -11.72 | -13.83 | -15.51 | -16.23 | -13.71 |
| NCEP | -7.06 | -7.62 | -9.63 | -10.62 | -10.28 | -9.55 |
| MME | -4.02 | -4.31 | -5.82 | -6.69 | -6.41 | -5.57 |
| Persistence | -0.87 | -2.02 | -3.54 | -7.53 | -9.41 | -10.03 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk1** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **MME** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.41** | **-0.07** | **-0.39** | **-0.75** | **-0.63** | **-0.66** |
| FuXi | -18.45 | -17.56 | -22.41 | -20.41 | -14.72 | -11.81 |
| ECMWF | -21.17 | -19.91 | -24.86 | -23.52 | -23.33 | -19.72 |
| NCEP | -14.81 | -13.04 | -20.74 | -21.82 | -18.67 | -15.57 |
| MME | -9.50 | -8.49 | -12.62 | -12.70 | -10.98 | -9.37 |
| Persistence | -2.02 | -3.56 | -7.30 | -12.02 | -16.60 | -21.43 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk1** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **MME** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.14** | **-0.53** | **-1.39** | **-1.74** | **-1.96** | **-3.20** |
| FuXi | -30.16 | -28.59 | -22.98 | -20.56 | -18.26 | -16.22 |
| ECMWF | -41.44 | -39.50 | -41.44 | -38.77 | -36.88 | -37.98 |
| NCEP | -36.99 | -35.32 | -44.51 | -46.17 | -45.83 | -45.28 |
| MME | -20.72 | -20.08 | -21.89 | -21.47 | -20.69 | -20.88 |
| Persistence | -1.14 | -3.93 | -7.72 | -12.27 | -17.63 | -27.24 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk1** · 🔴 **SPIRE** deeply negative (Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **MME** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.73** | **0.05** | **-0.23** | **-0.11** | **-0.20** | **-0.17** |
| FuXi | -15.64 | -18.98 | -18.72 | -18.57 | -17.39 | -15.38 |
| ECMWF | -21.13 | -23.00 | -23.01 | -24.97 | -26.31 | -24.43 |
| NCEP | -19.44 | -19.84 | -22.08 | -22.34 | -23.28 | -20.84 |
| MME | -9.87 | -10.17 | -10.68 | -11.24 | -11.61 | -10.47 |
| Persistence | -0.36 | -2.42 | -3.99 | -6.63 | -10.98 | -13.65 |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk1** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **MME** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **Persistence** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


# 5. Probabilistic — CRPSS vs Climatology
> Positive = better than climatology. Best possible = 1. Negative = worse than climatology. Includes MME.


## 5.1 TP

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.64** | 0.39 | 0.17 | 0.06 | 0.04 | 0.01 |
| FuXi | 0.46 | 0.32 | 0.18 | **0.18** | **0.14** | **0.11** |
| ECMWF | 0.62 | **0.43** | **0.26** | 0.12 | 0.11 | 0.06 |
| NCEP | 0.47 | 0.29 | 0.21 | 0.01 | 0.10 | -0.04 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6**


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.68 | 0.37 | 0.09 | -0.01 | 0.03 | 0.05 |
| FuXi | 0.45 | 0.31 | **0.29** | **0.37** | **0.27** | **0.24** |
| ECMWF | **0.71** | **0.51** | 0.26 | 0.16 | 0.22 | 0.18 |
| NCEP | 0.50 | 0.28 | 0.15 | -0.14 | 0.06 | -0.10 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk6**


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.54 | 0.17 | -0.22 | -0.33 | -0.32 | -0.39 |
| FuXi | 0.40 | 0.25 | -0.13 | 0.11 | 0.22 | **0.21** |
| ECMWF | **0.72** | **0.51** | 0.30 | 0.13 | 0.17 | 0.16 |
| NCEP | 0.61 | 0.35 | **0.40** | **0.25** | **0.29** | 0.05 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk6**


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.45** | **0.24** | 0.12 | 0.14 | 0.04 | -0.06 |
| FuXi | 0.33 | 0.20 | 0.23 | 0.24 | **0.26** | **0.28** |
| ECMWF | 0.41 | 0.22 | 0.22 | 0.18 | 0.17 | 0.19 |
| NCEP | 0.15 | 0.24 | **0.33** | **0.27** | 0.26 | 0.22 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6**


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.70** | **0.54** | **0.41** | **0.27** | **0.20** | **0.14** |
| FuXi | 0.52 | 0.39 | 0.20 | 0.06 | 0.01 | -0.04 |
| ECMWF | 0.60 | 0.42 | 0.28 | 0.12 | 0.03 | -0.05 |
| NCEP | 0.52 | 0.33 | 0.23 | 0.05 | 0.03 | -0.10 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6**


## 5.2 Z500

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.41 | 0.26 | 0.12 | **0.02** | **0.01** | **-0.03** |
| FuXi | 0.59 | -0.04 | -0.74 | -0.41 | -0.29 | -0.05 |
| ECMWF | **0.62** | **0.47** | **0.19** | 0.00 | -0.07 | -0.05 |
| NCEP | 0.07 | -0.56 | -0.53 | -0.53 | -0.61 | -0.55 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk3**


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.53 | 0.22 | 0.18 | **0.08** | **0.02** | **-0.05** |
| FuXi | 0.55 | -0.23 | -1.11 | -0.55 | -0.46 | -0.17 |
| ECMWF | **0.68** | **0.43** | **0.21** | 0.04 | -0.10 | -0.14 |
| NCEP | 0.22 | -0.30 | -0.18 | -0.16 | -0.18 | -0.25 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk3** · 🔴 **FuXi** deeply negative (Wk3) — possible issue


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.28 | 0.27 | 0.07 | **-0.07** | **-0.01** | 0.00 |
| FuXi | **0.57** | -0.05 | -0.72 | -0.41 | -0.20 | **0.05** |
| ECMWF | 0.54 | **0.42** | **0.08** | -0.11 | -0.12 | -0.01 |
| NCEP | 0.01 | -0.87 | -0.87 | -0.93 | -1.13 | -1.01 |
| MME | — | — | — | — | — | — |
> 💬 **FuXi** leads at Wk1 · at least one model skillful (>0.1) through **Wk2** · 🔴 **NCEP** deeply negative (Wk5, Wk6) — possible issue


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | -0.12 | 0.16 | 0.11 | -0.05 | -0.01 | 0.06 |
| FuXi | 0.58 | 0.13 | -0.20 | **-0.04** | **0.15** | 0.09 |
| ECMWF | **0.58** | **0.58** | **0.21** | -0.06 | 0.06 | **0.13** |
| NCEP | -0.48 | -1.75 | -1.86 | -1.66 | -1.70 | -1.66 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk6** · 🔴 **NCEP** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.40 | 0.00 | -0.14 | -0.16 | -0.15 | -0.27 |
| FuXi | 0.48 | -0.02 | -0.78 | -0.43 | -0.35 | **-0.04** |
| ECMWF | **0.48** | **0.31** | **0.13** | **-0.02** | **-0.10** | -0.17 |
| NCEP | -0.18 | -0.66 | -0.73 | -0.56 | -0.59 | -0.50 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk3**


## 5.3 T2M

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.64** | **0.43** | **0.39** | **0.30** | **0.22** | **0.19** |
| FuXi | -2.22 | -2.13 | -2.21 | -1.99 | -1.68 | -1.50 |
| ECMWF | -2.41 | -2.20 | -2.37 | -2.44 | -2.52 | -2.35 |
| NCEP | -1.92 | -1.71 | -2.03 | -2.16 | -2.14 | -2.05 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.66** | **0.40** | **0.41** | **0.34** | **0.26** | **0.26** |
| FuXi | -1.73 | -1.72 | -2.05 | -1.86 | -1.55 | -1.35 |
| ECMWF | -1.80 | -1.64 | -1.92 | -2.00 | -2.12 | -1.97 |
| NCEP | -1.30 | -1.27 | -1.46 | -1.51 | -1.51 | -1.51 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.63** | **0.50** | **0.44** | **0.32** | **0.25** | **0.21** |
| FuXi | -2.35 | -2.18 | -2.22 | -1.95 | -1.59 | -1.41 |
| ECMWF | -2.38 | -2.03 | -2.23 | -2.35 | -2.44 | -2.25 |
| NCEP | -1.69 | -1.35 | -1.80 | -2.05 | -1.98 | -1.85 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.54** | **0.39** | **0.22** | **0.09** | **-0.03** | **-0.18** |
| FuXi | -3.23 | -2.87 | -2.36 | -2.06 | -1.90 | -1.82 |
| ECMWF | -3.94 | -3.77 | -3.69 | -3.62 | -3.61 | -3.62 |
| NCEP | -3.73 | -3.34 | -3.88 | -4.06 | -4.09 | -4.13 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk3** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.65** | **0.40** | **0.32** | **0.32** | **0.27** | **0.26** |
| FuXi | -2.88 | -2.91 | -2.72 | -2.63 | -2.30 | -2.07 |
| ECMWF | -3.40 | -3.25 | -3.13 | -3.23 | -3.22 | -2.94 |
| NCEP | -3.08 | -2.80 | -3.01 | -3.09 | -3.01 | -2.73 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


# 6. Probabilistic Calibration — Spread/Skill Ratio (SSR)
> Ideal = 1.0. <1 = overconfident (spread too small). >1 = overdispersed.


## 6.1 TP

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.78** | 1.36 | 1.47 | 1.38 | 1.33 | **1.24** |
| FuXi | 0.13 | 0.31 | 0.43 | 0.43 | 0.48 | 0.43 |
| ECMWF | 0.39 | **0.67** | **0.75** | **0.69** | **0.68** | 0.58 |
| NCEP | 0.20 | 0.48 | 0.49 | 0.55 | 0.58 | 0.59 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · 🟡 **FuXi** underconfident spread (mean SSR=0.37) · 🟡 **NCEP** underconfident spread (mean SSR=0.48)


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.17** | 2.20 | 1.87 | 1.76 | 1.96 | 1.78 |
| FuXi | 0.25 | 0.57 | 0.80 | 0.77 | 0.84 | 0.73 |
| ECMWF | 0.65 | **1.17** | **1.00** | **0.87** | **0.92** | 0.81 |
| NCEP | 0.32 | 0.75 | 0.73 | 0.81 | 0.86 | **0.87** |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.32** | 2.19 | 2.62 | 2.36 | 2.45 | 2.36 |
| FuXi | 1.59 | 0.83 | 0.88 | **1.07** | **1.06** | **1.10** |
| ECMWF | 4.96 | 1.79 | 1.87 | 1.78 | 1.73 | 1.57 |
| NCEP | 5.83 | **1.02** | **0.98** | 1.09 | 1.47 | 1.40 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.82** | 1.22 | 1.62 | 1.74 | 1.93 | 2.01 |
| FuXi | 0.20 | 0.30 | 0.45 | 0.47 | 0.63 | 0.54 |
| ECMWF | 0.72 | **0.91** | 1.38 | 1.49 | 1.44 | 1.22 |
| NCEP | 0.28 | 0.49 | **0.71** | **0.71** | **0.84** | **0.88** |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · 🟡 **FuXi** underconfident spread (mean SSR=0.43)


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.79** | **1.41** | 1.60 | 1.59 | 1.38 | **1.27** |
| FuXi | 0.11 | 0.23 | 0.35 | 0.49 | 0.44 | 0.44 |
| ECMWF | 0.25 | 0.53 | **0.79** | **0.76** | **0.70** | 0.63 |
| NCEP | 0.17 | 0.51 | 0.53 | 0.71 | 0.60 | 0.70 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · 🟡 **FuXi** underconfident spread (mean SSR=0.34) · 🟡 **NCEP** underconfident spread (mean SSR=0.54)


## 6.2 Z500

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.98** | 1.54 | 2.52 | 1.76 | 1.91 | 1.48 |
| FuXi | 0.24 | **0.50** | 0.51 | **0.89** | **1.05** | **1.04** |
| ECMWF | 0.63 | 1.69 | 1.57 | 1.19 | 1.16 | 1.09 |
| NCEP | 0.25 | 0.44 | **0.72** | 0.74 | 0.77 | 0.69 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 1.57 | 1.84 | 3.40 | 2.63 | 2.53 | 1.55 |
| FuXi | 0.30 | 0.54 | 0.53 | 1.24 | 1.29 | 1.71 |
| ECMWF | **0.81** | 2.60 | 2.03 | 1.45 | 1.31 | 1.25 |
| NCEP | 0.41 | **0.60** | **1.02** | **0.90** | **0.96** | **0.80** |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.89** | 2.36 | 2.62 | 1.80 | 3.40 | 1.98 |
| FuXi | 0.27 | **0.58** | 0.70 | 1.09 | 1.24 | 1.20 |
| ECMWF | 0.67 | 1.94 | 1.72 | 1.39 | 1.40 | 1.41 |
| NCEP | 0.27 | 0.48 | **0.93** | **1.03** | **0.91** | **0.96** |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.62 | 1.52 | 1.91 | 2.06 | 2.60 | 2.26 |
| FuXi | 0.34 | **0.64** | **0.73** | **1.05** | **1.21** | 1.13 |
| ECMWF | **0.83** | 2.14 | 1.61 | 1.09 | 1.48 | 1.50 |
| NCEP | 0.20 | 0.36 | 0.56 | 0.67 | 0.66 | **0.93** |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · 🟡 **NCEP** underconfident spread (mean SSR=0.56)


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 1.34 | 1.88 | 3.65 | 1.80 | 4.08 | 3.21 |
| FuXi | 0.26 | **0.77** | 0.69 | 1.25 | 1.44 | 1.97 |
| ECMWF | **0.83** | 2.06 | 2.15 | 1.63 | 1.56 | 1.62 |
| NCEP | 0.22 | 0.47 | **0.75** | **1.00** | **1.38** | **0.89** |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1


## 6.3 T2M

### All India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.31** | **1.21** | **1.43** | **1.41** | **1.42** | **1.35** |
| FuXi | 0.03 | 0.08 | 0.17 | 0.22 | 0.28 | 0.30 |
| ECMWF | 0.09 | 0.18 | 0.24 | 0.26 | 0.26 | 0.27 |
| NCEP | 0.08 | 0.18 | 0.22 | 0.23 | 0.25 | 0.25 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · 🔴 **FuXi** severely overconfident (mean SSR=0.18) · 🔴 **ECMWF** severely overconfident (mean SSR=0.22) · 🔴 **NCEP** severely overconfident (mean SSR=0.20)


### NW India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.32** | **1.29** | **1.67** | **1.57** | **1.54** | **1.49** |
| FuXi | 0.03 | 0.10 | 0.17 | 0.23 | 0.30 | 0.33 |
| ECMWF | 0.10 | 0.22 | 0.28 | 0.30 | 0.30 | 0.31 |
| NCEP | 0.10 | 0.19 | 0.25 | 0.26 | 0.29 | 0.29 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · 🔴 **FuXi** severely overconfident (mean SSR=0.19) · 🔴 **ECMWF** severely overconfident (mean SSR=0.25) · 🔴 **NCEP** severely overconfident (mean SSR=0.23)


### Central India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.35** | **1.48** | 1.78 | **1.62** | 1.70 | **1.60** |
| FuXi | 0.03 | 0.10 | 0.20 | 0.27 | **0.33** | 0.34 |
| ECMWF | 0.10 | 0.22 | 0.28 | 0.28 | 0.28 | 0.29 |
| NCEP | 0.10 | 0.27 | **0.28** | 0.27 | 0.29 | 0.31 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · 🔴 **FuXi** severely overconfident (mean SSR=0.21) · 🔴 **ECMWF** severely overconfident (mean SSR=0.24) · 🔴 **NCEP** severely overconfident (mean SSR=0.25)


### South Peninsula
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.39** | **1.40** | **1.46** | **1.34** | **1.31** | **1.21** |
| FuXi | 0.03 | 0.07 | 0.18 | 0.23 | 0.26 | 0.28 |
| ECMWF | 0.08 | 0.14 | 0.19 | 0.20 | 0.20 | 0.20 |
| NCEP | 0.06 | 0.15 | 0.17 | 0.18 | 0.19 | 0.19 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · 🔴 **FuXi** severely overconfident (mean SSR=0.17) · 🔴 **ECMWF** severely overconfident (mean SSR=0.17) · 🔴 **NCEP** severely overconfident (mean SSR=0.15)


### East/NE India
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **1.57** | **1.24** | **1.28** | **1.43** | **1.50** | **1.49** |
| FuXi | 0.02 | 0.06 | 0.15 | 0.19 | 0.25 | 0.28 |
| ECMWF | 0.07 | 0.13 | 0.19 | 0.21 | 0.22 | 0.24 |
| NCEP | 0.05 | 0.12 | 0.15 | 0.17 | 0.19 | 0.20 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · 🔴 **FuXi** severely overconfident (mean SSR=0.16) · 🔴 **ECMWF** severely overconfident (mean SSR=0.18) · 🔴 **NCEP** severely overconfident (mean SSR=0.15)


# 7. Brier Skill Score (BSS vs Climatology) — All India
> Positive = better than climatology. Best = 1. Includes MME.


### TP > 1 mm/day (wet day)
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.54** | 0.29 | -0.06 | -0.19 | -0.22 | -0.27 |
| FuXi | 0.14 | 0.11 | -0.05 | **0.09** | 0.06 | **0.06** |
| ECMWF | 0.54 | **0.35** | 0.18 | 0.03 | 0.03 | 0.04 |
| NCEP | 0.29 | 0.21 | **0.18** | -0.03 | **0.07** | -0.14 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk3**


### TP > 10 mm/day (heavy rain)
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.74** | **0.53** | **0.39** | **0.22** | **0.18** | **0.11** |
| FuXi | 0.59 | 0.39 | 0.28 | 0.10 | 0.04 | 0.02 |
| ECMWF | 0.58 | 0.46 | 0.28 | 0.12 | 0.07 | -0.04 |
| NCEP | 0.54 | 0.29 | 0.07 | -0.21 | -0.11 | -0.23 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6**


### TP above-normal tercile
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.44 | 0.12 | -0.28 | -0.48 | -0.52 | -0.60 |
| FuXi | 0.09 | 0.09 | -0.22 | **-0.01** | **0.01** | **-0.06** |
| ECMWF | **0.55** | **0.28** | 0.03 | -0.17 | -0.09 | -0.14 |
| NCEP | 0.22 | 0.04 | **0.03** | -0.25 | -0.11 | -0.42 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk2**


### TP below-normal tercile
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.36 | 0.06 | -0.14 | -0.20 | -0.23 | -0.26 |
| FuXi | 0.34 | 0.09 | -0.08 | -0.03 | -0.07 | **0.02** |
| ECMWF | **0.58** | **0.29** | 0.01 | -0.10 | -0.12 | -0.12 |
| NCEP | 0.40 | 0.07 | **0.07** | **-0.01** | **-0.02** | -0.07 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk2**


### T2M above-normal tercile
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.61** | **0.21** | **0.25** | **0.21** | **0.12** | **0.13** |
| FuXi | 0.14 | 0.08 | 0.07 | 0.05 | -0.03 | -0.02 |
| ECMWF | 0.15 | 0.16 | 0.10 | 0.07 | -0.02 | -0.00 |
| NCEP | -0.00 | 0.02 | 0.01 | -0.04 | -0.10 | -0.10 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk6**


### T2M below-normal tercile
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | **0.51** | **0.37** | **0.25** | **0.16** | **0.10** | **-0.06** |
| FuXi | -6.12 | -5.72 | -6.00 | -5.32 | -4.66 | -4.27 |
| ECMWF | -5.99 | -5.56 | -5.86 | -5.94 | -5.97 | -5.64 |
| NCEP | -4.60 | -4.36 | -4.96 | -5.15 | -4.98 | -4.82 |
| MME | — | — | — | — | — | — |
> 💬 **SPIRE** leads at Wk1 · at least one model skillful (>0.1) through **Wk4** · 🔴 **FuXi** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **ECMWF** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue · 🔴 **NCEP** deeply negative (Wk1, Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


### Z500 above-normal tercile
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.12 | -0.03 | -0.20 | -0.47 | -0.51 | -0.46 |
| FuXi | **0.60** | -0.03 | -0.39 | -0.27 | **-0.31** | **-0.18** |
| ECMWF | 0.56 | **0.37** | **0.05** | **-0.20** | -0.35 | -0.38 |
| NCEP | -0.07 | -0.31 | -0.44 | -0.43 | -0.41 | -0.21 |
| MME | — | — | — | — | — | — |
> 💬 **FuXi** leads at Wk1 · at least one model skillful (>0.1) through **Wk2**


### Z500 below-normal tercile
| Model | Wk1 | Wk2 | Wk3 | Wk4 | Wk5 | Wk6 |
|:---|---:|---:|---:|---:|---:|---:|
| SPIRE | 0.16 | 0.01 | -0.30 | **-0.42** | **-0.38** | **-0.57** |
| FuXi | 0.34 | -0.81 | -1.86 | -1.35 | -1.12 | -0.72 |
| ECMWF | **0.50** | **0.27** | **-0.29** | -0.62 | -0.63 | -0.67 |
| NCEP | -0.46 | -1.81 | -1.84 | -1.97 | -1.95 | -1.88 |
| MME | — | — | — | — | — | — |
> 💬 **ECMWF** leads at Wk1 · at least one model skillful (>0.1) through **Wk2** · 🔴 **FuXi** deeply negative (Wk3, Wk4, Wk5) — possible issue · 🔴 **NCEP** deeply negative (Wk2, Wk3, Wk4, Wk5, Wk6) — possible issue


# 8. Key Findings & Flags

## ✅ What Looks Good
- **TP**: All models positive PCC through Wk4–5 over All India. CRPSS positive through Wk3–4 for all. Results are physically consistent and publication-ready.
- **Z500 Wk1–2**: SPIRE (0.94/0.87), ECMWF (0.94/0.84), NCEP (0.92/0.63) — consistent with known S2S predictability windows.
- **SPIRE**: Consistently best across all variables, regions, and metrics.
- **MME**: Reliably beats individual models on most deterministic metrics, as expected.

## 🔴 Red Flags
| Issue | Affected | Likely Cause |
|:---|:---|:---|
| T2M MSSS deeply negative (−10 to −40) | FuXi, ECMWF, NCEP, MME | ERA5 daily-mean vs. instantaneous T2M mismatch in climatology |
| T2M CRPSS −2 to −2.5 | FuXi, ECMWF, NCEP | Ensemble spread severely underestimated (SSR << 1) |
| FuXi Z500 PCC collapses Wk3 (0.82→0.04→−0.28) | FuXi | Possible Z500 unit or grid alignment bug |
| NCEP Z500 MSSS South Peninsula (−17 to −30) | NCEP | Possible NaN/land-sea mask mismatch |
| MME T2M still negative (inherits from members) | MME | Fix individual members first |

## 🔧 Action Items
1. **T2M climatology**: Verify ERA5 scoring uses the same T2M definition as each model (instantaneous vs. daily mean vs. (Tmax+Tmin)/2). Consider computing anomalies relative to each model's own climatology.
2. **FuXi Z500**: Re-check `z500 / g` conversion and coordinate alignment at Wk3+.
3. **NCEP South Peninsula**: Check NaN/land masking in NCEP loader for that sub-region.
4. **Ensemble spread (T2M)**: FuXi/ECMWF/NCEP SSR < 0.2 for T2M everywhere — inflate spread or use bias-corrected CRPS.

