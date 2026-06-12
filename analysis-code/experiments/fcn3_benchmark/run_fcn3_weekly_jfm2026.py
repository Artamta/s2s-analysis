#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_fcn3_weekly_jfm2026.py
==========================
Runs NVIDIA FourCastNet-3 deterministic hindcasts for 13 weekly Thursday
initializations spanning JFM 2026 (2026-01-01 … 2026-03-26).

Initial conditions : Google ARCO ERA5 (no API key needed)
Lead time          : 46 days  (184 × 6-h steps)
Output format      : one Zarr archive per init date

Output layout
-------------
/storage/raj.ayush/fcn3-weekly/
    zarr/
        fcn3_20260101.zarr
        fcn3_20260108.zarr
        ...
    logs/
        run_fcn3_weekly_jfm2026.log

Usage
-----
  python run_fcn3_weekly_jfm2026.py

Requirements
------------
  pip install "earth2studio[fcn3] @ git+https://github.com/NVIDIA/earth2studio"
  pip install torch-harmonics
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_ROOT = Path("/storage/raj.ayush/fcn3-weekly")
ZARR_DIR = OUT_ROOT / "zarr"
LOG_DIR  = OUT_ROOT / "logs"

ZARR_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
log_file = LOG_DIR / "run_fcn3_weekly_jfm2026.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Init dates (weekly Thursdays, JFM 2026) ────────────────────────────────
INIT_DATES = pd.to_datetime([
    "2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22", "2026-01-29",
    "2026-02-05", "2026-02-12", "2026-02-19", "2026-02-26",
    "2026-03-05", "2026-03-12", "2026-03-19", "2026-03-26",
])

# 46 days at 6-h time step
N_STEPS = 184


# ── Earth2Studio imports ───────────────────────────────────────────────────
try:
    from earth2studio.models.px import FCN3
    from earth2studio.data import ARCO
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic as e2s_run
except Exception as _e:
    import traceback
    log.error(f"Failed to import earth2studio ({type(_e).__name__}): {_e}")
    log.error(traceback.format_exc())
    log.error("If earth2studio is missing, install with:")
    log.error('  pip install "earth2studio[fcn3] @ git+https://github.com/NVIDIA/earth2studio"')
    log.error("  pip install torch-harmonics")
    sys.exit(1)


def main() -> None:
    log.info("=" * 65)
    log.info("NVIDIA FourCastNet-3  —  Weekly JFM 2026 Hindcast Run")
    log.info(f"Output root : {OUT_ROOT}")
    log.info(f"Init dates  : {len(INIT_DATES)}  ({INIT_DATES[0].date()} … {INIT_DATES[-1].date()})")
    log.info(f"Lead time   : {N_STEPS} steps × 6 h = {N_STEPS * 6 // 24} days")
    log.info("=" * 65)

    # Load model once; reuse across all init dates
    log.info("Loading FCN-3 model weights …")
    model = FCN3.load_model(FCN3.load_default_package())

    log.info("Initialising ARCO ERA5 data source …")
    data = ARCO(cache=True)

    succeeded, skipped, failed = 0, 0, 0

    for idx, date in enumerate(INIT_DATES, start=1):
        date_str  = date.strftime("%Y-%m-%dT00:00:00")
        zarr_path = ZARR_DIR / f"fcn3_{date.strftime('%Y%m%d')}.zarr"

        log.info(f"[{idx:02d}/{len(INIT_DATES)}]  {date_str}")

        if zarr_path.exists():
            log.info(f"  SKIP — {zarr_path.name} already exists")
            skipped += 1
            continue

        try:
            io = ZarrBackend(str(zarr_path))
            e2s_run([date_str], nsteps=N_STEPS, model=model, data=data, io=io)
            log.info(f"  OK   — saved {zarr_path.name}")
            succeeded += 1
        except Exception as exc:
            log.error(f"  FAIL — {exc}")
            failed += 1

    log.info("=" * 65)
    log.info(f"Done.  succeeded={succeeded}  skipped={skipped}  failed={failed}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
