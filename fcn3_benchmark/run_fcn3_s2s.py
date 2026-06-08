#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_fcn3_s2s.py
===============
This script runs deterministic inference for NVIDIA's FourCastNet 3 (FCN3) model
out to 46 days (184 steps of 6 hours) for the JFM 2026 period.

It uses Google's ARCO (Analysis-Ready Cloud-Optimized) ERA5 dataset as the source
for initial conditions.

Requirements:
  - Python 3.10+
  - PyTorch with CUDA (configured for your A100 GPUs)
  - earth2studio (with fcn3 dependencies)
  - torch-harmonics

Run on your HPC with GPU enabled:
  python run_fcn3_s2s.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

# We import Earth2Studio modules
try:
    from earth2studio.models.px import FCN3
    from earth2studio.data import ARCO
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic as run
except ImportError:
    print("Error: earth2studio is not installed.")
    print("Please install it on your HPC environment using:")
    print("  pip install \"earth2studio[fcn3] @ git+https://github.com/NVIDIA/earth2studio\"")
    print("  pip install torch-harmonics")
    sys.exit(1)

# Setup data output directory
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate forecast initialization dates (Mondays & Thursdays in JFM 2026)
all_days = pd.date_range("2026-01-01", "2026-03-31", freq="D")
INIT_DATES = [d for d in all_days if d.weekday() in (0, 3)]   # Mon=0, Thu=3

def main():
    print("=" * 65)
    print("NVIDIA FourCastNet 3 (FCN3) S2S Rollout Tool")
    print(f"Target Directory : {OUT_DIR}")
    print(f"Total Init Dates : {len(INIT_DATES)}")
    print("=" * 65)

    # 1. Load FCN3 model
    print("Loading FCN3 model package...")
    model = FCN3.load_model(FCN3.load_default_package())

    # 2. Set up initial conditions data source
    # ARCO accesses ERA5 archived in Google Cloud Zarr format (very fast, no API key needed)
    print("Initializing Google ARCO ERA5 data source...")
    data = ARCO(cache=True)

    # 3. Loop over init dates and run rollout
    # 46 days = 1104 hours. FCN3 step size is 6 hours, so 1104 / 6 = 184 steps.
    nsteps = 184 

    for i, date in enumerate(INIT_DATES):
        # Format date for Earth2Studio (YYYY-MM-DDTHH:MM:SS)
        date_str = date.strftime("%Y-%m-%dT00:00:00")
        target_file = OUT_DIR / f"fcn3_{date.strftime('%Y%m%d')}.zarr"
        
        print(f"\n[{i+1}/{len(INIT_DATES)}] Initializing FCN3 forecast for: {date_str}")
        
        if target_file.exists():
            print(f"  [SKIPPED] {target_file.name} already exists.")
            continue
            
        try:
            # Set up the I/O backend to save predictions as a Zarr archive
            io = ZarrBackend(str(target_file))
            
            # Execute the deterministic rollout
            run([date_str], nsteps=nsteps, model=model, data=data, io=io)
            print(f"  [SUCCESS] Saved forecast to {target_file.name}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to run FCN3 for {date_str}: {e}")

    print("\n" + "=" * 65)
    print("FCN3 Rollout completed.")
    print("=" * 65)

if __name__ == "__main__":
    main()
