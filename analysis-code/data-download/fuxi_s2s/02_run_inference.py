#!/usr/bin/env python3
"""
02_run_inference.py
===================
Run FuXi-S2S inference for all JFM 2026 init dates.

For each date, runs FuXi with 11 ensemble members and 42 lead steps (42 days).
Skips dates where output is already complete (member/10/42.nc exists).

Input  : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs/{YYYYMMDD}/input.nc
Output : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/raw/{YYYYMMDD}/member/{MM}/{SS}.nc
           where MM = 00–10, SS = 01–42

Usage
-----
  # All 90 dates (skips complete ones)
  python 02_run_inference.py

  # Single date test
  python 02_run_inference.py --date 20260102

  # CPU mode (slow, for testing)
  python 02_run_inference.py --date 20260102 --device cpu

Requirements: conda activate fuxi_s2s  (needs CUDA GPU)
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/storage/raj.ayush/All_Model_Data/fuxi/jfm2026")
INPUT_DIR   = BASE_DIR / "inputs"
RAW_DIR     = BASE_DIR / "raw"
FUXI_DIR    = Path(__file__).parent / "FuXi-S2S"
MODEL_PATH  = FUXI_DIR / "model" / "fuxi_s2s.onnx"
LOG_DIR     = Path(__file__).parent / "logs"

TOTAL_STEPS   = 42   # 42-day lead time
TOTAL_MEMBERS = 11   # 1 control + 10 perturbed

DATE_START = "2026-01-01"
DATE_END   = "2026-03-31"


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def is_done(date: pd.Timestamp) -> bool:
    """Complete when last member's last step file exists."""
    final = RAW_DIR / f"{date:%Y%m%d}" / "member" / f"{TOTAL_MEMBERS-1:02d}" / f"{TOTAL_STEPS:02d}.nc"
    return final.exists()


def run_inference(date: pd.Timestamp, device: str, log: logging.Logger) -> bool:
    date_str   = f"{date:%Y%m%d}"
    input_file = INPUT_DIR / date_str / "input.nc"
    out_dir    = RAW_DIR / date_str

    if not input_file.exists():
        log.error(f"SKIP   {date_str}  (no input.nc — run 01_download_inputs.py first)")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "inference.py",
        "--model",        str(MODEL_PATH),
        "--input",        str(input_file),
        "--device",       device,
        "--total_step",   str(TOTAL_STEPS),
        "--total_member", str(TOTAL_MEMBERS),
        "--save_dir",     str(out_dir),
    ]

    env = os.environ.copy()
    conda_lib = Path(os.environ.get("CONDA_PREFIX", "")).parent / "fuxi_s2s" / "lib"
    env["LD_LIBRARY_PATH"] = str(conda_lib) + ":" + env.get("LD_LIBRARY_PATH", "")

    log.info(f"START  {date_str}  ({TOTAL_MEMBERS} members × {TOTAL_STEPS} steps)")
    result = subprocess.run(cmd, cwd=str(FUXI_DIR), env=env)

    if result.returncode != 0:
        log.error(f"FAIL   {date_str}  (exit code {result.returncode})")
        return False

    log.info(f"DONE   {date_str}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run FuXi-S2S inference for JFM 2026")
    parser.add_argument("--date",   type=str, default=None,   help="Single date YYYYMMDD")
    parser.add_argument("--start",  type=str, default=DATE_START)
    parser.add_argument("--end",    type=str, default=DATE_END)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    log_file = LOG_DIR / f"inference_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        dates = list(pd.date_range(args.start, args.end, freq="D"))

    pending = [d for d in dates if not is_done(d)]
    done_n  = len(dates) - len(pending)

    log.info("=" * 60)
    log.info("FuXi-S2S Inference")
    log.info(f"  Model      : {MODEL_PATH}")
    log.info(f"  Input dir  : {INPUT_DIR}")
    log.info(f"  Output dir : {RAW_DIR}")
    log.info(f"  Device     : {args.device}")
    log.info(f"  Members    : {TOTAL_MEMBERS}  |  Steps: {TOTAL_STEPS}")
    log.info(f"  Dates      : {len(dates)}  ({dates[0].date()} → {dates[-1].date()})")
    log.info(f"  Already done: {done_n}")
    log.info(f"  To run     : {len(pending)}")
    log.info("=" * 60)

    failed = []
    for i, date in enumerate(pending, 1):
        ok = run_inference(date, args.device, log)
        if not ok:
            failed.append(f"{date:%Y%m%d}")
        log.info(f"Progress {done_n + i}/{len(dates)}  "
                 f"done={done_n + i - len(failed)}  fail={len(failed)}")

    log.info("=" * 60)
    log.info("INFERENCE COMPLETE")
    log.info(f"  Success : {len(pending) - len(failed)}")
    log.info(f"  Failed  : {len(failed)}")
    for f in failed:
        log.info(f"    FAILED: {f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
