#!/usr/bin/env python3
"""
test_reforecast_access.py
=========================
Minimal test: download ONE S2S reforecast file (ECMWF, single init date,
single variable, short lead, small region) to verify API access and check
what years of hindcast data are available.

Run: python test_reforecast_access.py
"""

from pathlib import Path
import cdsapi

OUT_DIR = Path("/storage/raj.ayush/All_Model_Data/ecmwf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

# ── STEP 1: Tiny test request ────────────────────────────────────────────────
# ECMWF reforecasts run on Mon/Thu; hindcast years are model-dependent.
# ECMWF typically provides ~20 years (2000-2019 for recent real-time dates).
# We request just 1 variable, 1 step, 1 date, India-only to check quickly.

print("=" * 65)
print("STEP 1: Small test request (ECMWF reforecasts, T2m, 1 step)")
print("=" * 65)

test_file = OUT_DIR / "test_ecmwf_reforecast_one_date.grib"

request = {
    "origin": "ecmwf",
    "forecast_type": "control_reforecast",   # reforecast = hindcast
    "level_type": "single_level",
    "variable": "2t",          # MARS short name for 2m temperature
    "year": "2020",           # a real-time date whose reforecasts cover ~2000-2019
    "month": "01",
    "day": "02",              # 2020-01-02 is a Thursday (valid ECMWF init)
    "time": "00:00",
    "step": "24",             # just day-1 lead
    "area": [35, 65, 5, 100],  # small India box [N, W, S, E]
    "grid": [1.5, 1.5],
    "data_format": "grib",
}

try:
    client.retrieve("s2s-reforecasts", request, str(test_file))
    size_mb = test_file.stat().st_size / 1024**2
    print(f"\n[SUCCESS] Downloaded: {test_file}")
    print(f"  File size: {size_mb:.2f} MB")
    print("\nNow inspecting the file for available years / members ...")

    # ── STEP 2: Inspect the downloaded grib to see what hindcast years came back
    try:
        import cfgrib
        import xarray as xr
        ds = cfgrib.open_datasets(str(test_file))
        for i, d in enumerate(ds):
            print(f"\n  Dataset [{i}]:")
            print(f"    Dims  : {dict(d.dims)}")
            print(f"    Coords: {list(d.coords)}")
            if "time" in d.coords:
                print(f"    Times : {d.coords['time'].values}")
            if "verifyingTime" in d.coords:
                print(f"    verifyingTime: {d.coords['verifyingTime'].values}")
    except ImportError:
        print("  cfgrib not installed; trying eccodes via subprocess ...")
        import subprocess
        result = subprocess.run(
            ["grib_ls", "-n", "time", str(test_file)],
            capture_output=True, text=True
        )
        print(result.stdout[:3000] if result.stdout else result.stderr[:1000])

except Exception as e:
    print(f"\n[FAILED] {e}")
    print("\nPossible causes:")
    print("  1. Dataset name wrong — try 's2s-reforecasts' vs 'seasonal-original-single-levels'")
    print("  2. Invalid date (must be a Mon or Thu for ECMWF)")
    print("  3. API quota/access issue")
