#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jjas/config.py  —  JJAS monsoon multi-year experiment (ECMWF + FuXi reforecasts).
================================================================================
Same injectable contract as JFM2026, retargeted to the Indian summer monsoon
(JJAS) and to the MULTI-YEAR reforecast archives:

  ECMWF  /storage/raj.ayush/archive/All_Model_Data/models/ecmwf/data
         <var>_<cf|pf>_<MMDD>.grib  with dims (number, time=20 hindcast yrs,
         step=46, lat, lon). For a hindcast YEAR we select time==year; the
         mean over all years IS the lead-dependent model-own climatology.
  Truth  WeatherBench2 ERA5 zarr (1.5°, 1959-2023) -> daily tp + z500.
  Clim   reuse the ERA5 30-yr DOY climatology (covers all days incl. JJAS).

PILOT: one year at a time (build_config(year=...)). FuXi is added after the
ECMWF+truth path is validated. Models with a multi-year reforecast carry a
model-own clima adapter -> the dual (era5 + model_own) anomaly basis.
================================================================================
"""
import os
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))                 # final_analysis/ on path

import pandas as pd
from core import ExperimentConfig, ModelSpec, GridSpec, Paths, Physics
from jfm2026.config import region_mask_path             # reuse the resolution->mask map

STORE = "/storage/raj.ayush"
ECMWF_ROOT = f"{STORE}/archive/All_Model_Data/models/ecmwf/data"
WB2_ZARR = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")

# JJAS init window: use the ECMWF reforecast MMDD files that fall in Jun-Aug
# (6-week leads then reach into mid-Sep, keeping verification inside the monsoon).
JJAS_INIT_MONTHS = ("06", "07", "08")


def _jjas_init_dates(year):
    """All ECMWF reforecast MMDDs present in Jun-Aug, as YYYY-MM-DD for `year`."""
    mmdds = set()
    for mm in JJAS_INIT_MONTHS:
        for f in glob.glob(f"{ECMWF_ROOT}/tp_pf_{mm}*.grib"):
            mmdd = os.path.basename(f).split("_")[-1].split(".")[0]
            if len(mmdd) == 4:
                mmdds.add(mmdd)
    dates = [f"{year}-{m[:2]}-{m[2:]}" for m in sorted(mmdds)]
    return tuple(d for d in dates if pd.Timestamp(d))   # keep valid calendar dates


def build_config(year: int = 2019, dgrid: float = 1.5) -> ExperimentConfig:
    inits = _jjas_init_dates(year)
    paths = Paths(
        clim_nc        = f"{STORE}/benchmark(jfm)/era5_climatology.nc",
        region_mask_nc = region_mask_path(dgrid),
        wb2_zarr       = WB2_ZARR,
        soi_shapefile  = f"{STORE}/archive/s2s-forecast-/STATE_BOUNDARY.shp",
    )
    models = [
        ModelSpec(name="ECMWF", adapter="ecmwf_reforecast",
                  kwargs={"root": ECMWF_ROOT, "year": year, "ens": "both"},
                  has_model_own_clim=True, clim_adapter="ecmwf_reforecast_clim",
                  clim_kwargs={"root": ECMWF_ROOT}),
        # FuXi: 50-member hindcast from the compact files in jjas/fuxi_combined/
        # (built by preprocess_fuxi.py). Appears only for inits whose compact file
        # exists; model-own clima is a follow-up (mean over years per MMDD).
        ModelSpec(name="FuXi", adapter="fuxi_reforecast",
                  kwargs={"root": HERE, "members": 50}),
    ]
    return ExperimentConfig(
        season_label=f"JJAS{year}",
        init_dates=inits,
        valid_end=f"{year}-10-15",
        valid_end_clim=f"{year}-10-20",
        grid=GridSpec(lat0=38.0, lat1=5.0, lon0=65.0, lon1=100.0, dgrid=dgrid),
        paths=paths,
        physics=Physics(),
        models=models,
        variables=("TP", "Z500"),
        out_dir=os.path.join(HERE, f"results_{year}_{dgrid:g}deg"),
    )


CFG = build_config()
