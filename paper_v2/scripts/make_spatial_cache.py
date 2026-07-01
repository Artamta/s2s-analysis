#!/usr/bin/env python3
r"""Reduce the large grid-cell scatter CSVs to a compact per-cell diagnostic
cache the figure script can read quickly.

The pipeline writes one row per (model, variable, week, init, grid cell) into
scatter_grid_weekly.csv (~330 MB for the full JFM run). Reading that on every
figure build is wasteful, so this script reads each source once and collapses
the initialization dimension into three standard per-cell spatial diagnostics:

  bias      = mean over inits of (forecast - truth)
  rmse      = sqrt(mean over inits of (forecast - truth)^2)
  local_acc = temporal anomaly correlation at the cell, i.e. the correlation
              across inits between the forecast anomaly and the truth anomaly
              (a per-grid-point skill map, complementary to the domain-mean,
              per-init ACC reported in the tables)

Output: paper_v2/cache/spatial_cells_<season>.csv with columns
  season, variable, model, week, lat, lon, n, bias, rmse, local_acc

All fields are on the 1.5-degree common verification grid, Indian land points
only (the same mask used for every score in the paper).

Run:
    python paper_v2/scripts/make_spatial_cache.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = "/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs"
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
os.makedirs(CACHE, exist_ok=True)

# Grid-scatter sources with full model/variable coverage per season.
SOURCES = {
    "jfm": f"{ROOT}/jfm2026/03_metrics/full_jfm2026_gridscatter/scatter_grid_weekly.csv",
    "jjas_tp": f"{ROOT}/jjas2019/03_metrics/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/scatter_grid_weekly.csv",
    "jjas_z500": f"{ROOT}/jjas2019/03_metrics/full_jjas2019_operational35_plus_fuxi_z500/scatter_grid_weekly.csv",
}

USECOLS = ["variable", "model", "week", "lat", "lon",
           "forecast_anomaly", "truth_anomaly", "error"]


def _local_acc(g: pd.DataFrame) -> float:
    fa = g["forecast_anomaly"].to_numpy()
    ta = g["truth_anomaly"].to_numpy()
    ok = np.isfinite(fa) & np.isfinite(ta)
    if ok.sum() < 3:
        return np.nan
    fa, ta = fa[ok], ta[ok]
    fsd, tsd = fa.std(), ta.std()
    if fsd == 0 or tsd == 0:
        return np.nan
    return float(np.corrcoef(fa, ta)[0, 1])


def reduce_source(season: str, path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  [skip] {season}: source missing ({path})")
        return pd.DataFrame()
    print(f"  reading {season} <- {os.path.basename(os.path.dirname(path))} ...")
    df = pd.read_csv(path, usecols=USECOLS)
    keys = ["variable", "model", "week", "lat", "lon"]

    # bias and rmse are cheap vectorized aggregates.
    agg = df.groupby(keys).agg(
        n=("error", "size"),
        bias=("error", "mean"),
        mse=("error", lambda e: float(np.mean(np.square(e)))),
    ).reset_index()
    agg["rmse"] = np.sqrt(agg["mse"])
    agg = agg.drop(columns="mse")

    # local temporal ACC needs the paired anomalies per cell.
    lacc = df.groupby(keys).apply(_local_acc, include_groups=False)
    lacc.name = "local_acc"
    out = agg.merge(lacc.reset_index(), on=keys)
    out.insert(0, "season", season)
    return out


def main() -> None:
    for season, path in SOURCES.items():
        out = reduce_source(season, path)
        if out.empty:
            continue
        dst = os.path.join(CACHE, f"spatial_cells_{season}.csv")
        out.to_csv(dst, index=False)
        print(f"  wrote {dst}  ({len(out)} cells x model x var x week)")
    print("done.")


if __name__ == "__main__":
    main()
