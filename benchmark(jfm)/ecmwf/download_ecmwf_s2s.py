#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_ecmwf_s2s.py
=====================
This script retrieves the ECMWF operational S2S (subseasonal-to-seasonal)
forecast data to serve as a physics-based benchmark against Spire forecasts.

It uses the new ECMWF Data Store (ECDS) API (cdsapi >= 0.7.7) and queries the
standard "s2s-forecasts" dataset.

Configurations:
  - Init Dates : Mondays & Thursdays in JFM 2026 (matching Spire's inits).
  - Target Grid: 1.5° x 1.5° (standard S2S resolution).
  - Bounding Box: India domain (0–50°N, 55–105°E).
  - Variables:
      - Surface: 2t (T2m-mean), mx2t6 (T2m-max), mn2t6 (T2m-min), tp (Precip).
      - Pressure Level: gh @ 500 hPa (Z500).

Outputs:
  data/sfc_<ftype>_<YYYYMMDD>.grib  (Surface variables)
  data/pl_<ftype>_<YYYYMMDD>.grib   (Pressure level variables)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import cdsapi

# ── CONFIGURATION ────────────────────────────────────────────────────────────
# Setup self-contained data output directory
OUT_DIR = Path("/storage/raj.ayush/benchmark(jfm)/ecmwf/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate forecast initialization dates (Thursdays only in JFM 2026 matching Spire/FuXi)
all_days = pd.date_range("2026-01-01", "2026-03-31", freq="D")
INIT_DATES = [d for d in all_days if d.weekday() == 3]   # Thu=3

# Grid coordinates and regional box (India domain)
AREA = [50, 55, 0, 105]       # [North, West, South, East]
GRID = [1.5, 1.5]             # 1.5 degree resolution

# Forecast lead times (1 to 46 days, step of 24h)
STEPS = [str(h) for h in range(24, 1105, 24)]

# Initialize CDS API client
client = cdsapi.Client()

def retrieve_surface_forecast(date: pd.Timestamp, ftype: str):
    """
    Downloads surface variables (2t, mx2t6, mn2t6, tp) for a specific init date.
    ftype can be:
      - 'cf' (Control Forecast)
      - 'pf' (Perturbed Forecast - all members)
    """
    target_file = OUT_DIR / f"sfc_{ftype}_{date:%Y%m%d}.grib"
    
    # Auto-resume check: skip if file already exists and is non-empty
    if target_file.exists() and target_file.stat().st_size > 0:
        print(f"  [SKIPPED] {target_file.name} already exists.")
        return

    print(f"  [DOWNLOADING] {target_file.name} ...")
    request = {
        "origin": "ecmwf",
        "forecast_type": "control_forecast" if ftype == "cf" else "perturbed_forecast",
        "level_type": "single_level",
        "variable": ["2t", "mx2t6", "mn2t6", "tp"],
        "year": str(date.year),
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
        "time": "00:00",
        "step": STEPS,
        "area": AREA,
        "grid": GRID,
        "data_format": "grib"
    }
    client.retrieve("s2s-forecasts", request, str(target_file))
    print(f"  [SUCCESS] Saved {target_file.name}")


def retrieve_pressure_level_forecast(date: pd.Timestamp, ftype: str):
    """
    Downloads pressure-level variables (gh @ 500 hPa) for a specific init date.
    """
    target_file = OUT_DIR / f"pl_{ftype}_{date:%Y%m%d}.grib"
    
    # Auto-resume check
    if target_file.exists() and target_file.stat().st_size > 0:
        print(f"  [SKIPPED] {target_file.name} already exists.")
        return

    print(f"  [DOWNLOADING] {target_file.name} ...")
    request = {
        "origin": "ecmwf",
        "forecast_type": "control_forecast" if ftype == "cf" else "perturbed_forecast",
        "level_type": "pressure",
        "variable": "gh",
        "level": "500",
        "year": str(date.year),
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
        "time": "00:00",
        "step": STEPS,
        "area": AREA,
        "grid": GRID,
        "data_format": "grib"
    }
    client.retrieve("s2s-forecasts", request, str(target_file))
    print(f"  [SUCCESS] Saved {target_file.name}")


def main():
    print("=" * 65)
    print("ECMWF S2S Data Download Tool (ECDS API Version)")
    print(f"Target Directory : {OUT_DIR}")
    print(f"Total Init Dates : {len(INIT_DATES)}")
    print(f"India Box Bounds : {AREA}")
    print("=" * 65)

    for i, date in enumerate(INIT_DATES):
        print(f"\n[{i+1}/{len(INIT_DATES)}] Processing Init Date: {date:%Y-%m-%d}")
        for ftype in ("cf", "pf"):
            # 1. Download Surface variables
            try:
                retrieve_surface_forecast(date, ftype)
            except Exception as e:
                print(f"  [ERROR] Failed surface {ftype} download for {date:%Y%m%d}: {e}")

            # 2. Download Pressure Level variables
            try:
                retrieve_pressure_level_forecast(date, ftype)
            except Exception as e:
                print(f"  [ERROR] Failed pressure level {ftype} download for {date:%Y%m%d}: {e}")

    print("\n" + "=" * 65)
    print("Download process completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
