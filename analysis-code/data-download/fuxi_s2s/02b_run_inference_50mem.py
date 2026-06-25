#!/usr/bin/env python3
"""
02b_run_inference_50mem.py
==========================
Run FuXi-S2S inference for all JFM 2026 init dates with 50 ensemble members.
Output goes to a SEPARATE directory (jfm2026_ens50/) to keep it distinct from
the 11-member run (jfm2026/).

Input  : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs/{YYYYMMDD}/input.nc
         (same inputs as 11-member run — no re-download needed)
Output : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50/raw/{YYYYMMDD}/member/{MM}/{SS}.nc

Usage
-----
  python 02b_run_inference_50mem.py            # all 90 dates
  python 02b_run_inference_50mem.py --date 20260101
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
INPUT_DIR   = Path("/storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs")   # shared inputs
BASE_DIR    = Path("/storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50")
RAW_DIR     = BASE_DIR / "raw"
FUXI_DIR    = Path(__file__).parent / "FuXi-S2S"
MODEL_PATH  = FUXI_DIR / "model" / "fuxi_s2s.onnx"
LOG_DIR     = Path(__file__).parent / "logs"
FUXI_PYTHON = "/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"
FUXI_LIB    = "/home/raj.ayush/.conda/envs/fuxi_s2s/lib"

TOTAL_STEPS   = 42
TOTAL_MEMBERS = 50

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


def expected_files(date: pd.Timestamp):
    root = RAW_DIR / f"{date:%Y%m%d}" / "member"
    for member in range(TOTAL_MEMBERS):
        for step in range(1, TOTAL_STEPS + 1):
            yield root / f"{member:02d}" / f"{step:02d}.nc"


def raw_status(date: pd.Timestamp):
    """Return (complete, present_count, missing_examples) for one init date."""
    present = 0
    missing = []
    for path in expected_files(date):
        if path.exists() and path.stat().st_size > 0:
            present += 1
        elif len(missing) < 5:
            missing.append(path)
    return present == TOTAL_MEMBERS * TOTAL_STEPS, present, missing


def is_done(date: pd.Timestamp) -> bool:
    return raw_status(date)[0]


def run_inference(date: pd.Timestamp, device: str, log: logging.Logger) -> bool:
    date_str   = f"{date:%Y%m%d}"
    input_file = INPUT_DIR / date_str / "input.nc"
    out_dir    = RAW_DIR / date_str

    if not input_file.exists():
        log.error(f"SKIP   {date_str}  (no input.nc)")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        FUXI_PYTHON, "inference.py",
        "--model",        str(MODEL_PATH),
        "--input",        str(input_file),
        "--device",       device,
        "--total_step",   str(TOTAL_STEPS),
        "--total_member", str(TOTAL_MEMBERS),
        "--save_dir",     str(out_dir),
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = FUXI_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"]  = "1"
    env["OMP_PROC_BIND"]    = "false"

    log.info(f"START  {date_str}  ({TOTAL_MEMBERS} members × {TOTAL_STEPS} steps)")
    # Keep threaded math libraries quiet on shared GPU nodes. ONNX Runtime can
    # still emit affinity warnings on this cluster; output completeness below is
    # the authoritative success criterion.
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["KMP_AFFINITY"] = "disabled"

    result = subprocess.run(cmd, cwd=str(FUXI_DIR), env=env)
    complete, present, missing = raw_status(date)

    if result.returncode != 0:
        if complete:
            log.warning(f"DONE   {date_str}  raw complete, but child exited "
                        f"{result.returncode}; treating as usable")
            return True
        miss = ", ".join(str(p.relative_to(out_dir)) for p in missing)
        log.error(f"FAIL   {date_str}  (exit code {result.returncode}; "
                  f"{present}/{TOTAL_MEMBERS * TOTAL_STEPS} files present; "
                  f"missing examples: {miss})")
        return False

    if not complete:
        miss = ", ".join(str(p.relative_to(out_dir)) for p in missing)
        log.error(f"FAIL   {date_str}  child exited 0 but raw is incomplete "
                  f"({present}/{TOTAL_MEMBERS * TOTAL_STEPS}; missing examples: {miss})")
        return False

    log.info(f"DONE   {date_str}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",   type=str, default=None)
    parser.add_argument("--start",  type=str, default=DATE_START)
    parser.add_argument("--end",    type=str, default=DATE_END)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    log_file = LOG_DIR / f"ens50_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        dates = list(pd.date_range(args.start, args.end, freq="D"))

    pending  = [d for d in dates if not is_done(d)]
    done_n   = len(dates) - len(pending)

    log.info("=" * 60)
    log.info("FuXi-S2S Inference — 50-member ensemble")
    log.info(f"  Model      : {MODEL_PATH}")
    log.info(f"  Input dir  : {INPUT_DIR}")
    log.info(f"  Output dir : {RAW_DIR}")
    log.info(f"  Device     : {args.device}")
    log.info(f"  Members    : {TOTAL_MEMBERS}  |  Steps: {TOTAL_STEPS}")
    log.info(f"  Dates      : {len(dates)}  ({dates[0].date()} → {dates[-1].date()})")
    log.info(f"  Already done: {done_n}")
    log.info(f"  To run     : {len(pending)}")
    log.info("=" * 60)

    success = 0
    failed = []
    for i, date in enumerate(pending, 1):
        ok = run_inference(date, args.device, log)
        if not ok:
            failed.append(f"{date:%Y%m%d}")
        else:
            success += 1
        log.info(f"Progress {done_n + i}/{len(dates)}  done={done_n + success}  fail={len(failed)}")

    log.info("=" * 60)
    log.info("INFERENCE COMPLETE")
    log.info(f"  Success : {len(pending) - len(failed)}")
    log.info(f"  Failed  : {len(failed)}")
    for f in failed:
        log.info(f"    FAILED: {f}")
    log.info("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
