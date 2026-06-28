#!/usr/bin/env python3
"""
Run FuXi-S2S 50-member inference for JJAS 2025.

Input:
  /storage/raj.ayush/All_Model_Data/fuxi/jjas2025/inputs/{YYYYMMDD}/input.nc

Output:
  /storage/raj.ayush/All_Model_Data/fuxi/jjas2025_ens50/raw/{YYYYMMDD}/member/{MM}/{SS}.nc

The script is resumable: dates with all expected member/step files are skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("/storage/raj.ayush/All_Model_Data/fuxi/jjas2025/inputs")
DEFAULT_RAW_DIR = Path("/storage/raj.ayush/All_Model_Data/fuxi/jjas2025_ens50/raw")
FUXI_DIR = Path(__file__).parent / "FuXi-S2S"
MODEL_PATH = FUXI_DIR / "model" / "fuxi_s2s.onnx"
LOG_DIR = Path(__file__).parent / "logs"
FUXI_PYTHON = "/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"
FUXI_LIB = "/home/raj.ayush/.conda/envs/fuxi_s2s/lib"

DATE_START = "2025-06-01"
DATE_END = "2025-09-30"
TOTAL_STEPS = 42
TOTAL_MEMBERS = 50


def setup_logging(log_file: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def ymd(date: pd.Timestamp) -> str:
    return f"{date:%Y%m%d}"


def expected_files(raw_dir: Path, date: pd.Timestamp, members: int, steps: int):
    root = raw_dir / ymd(date) / "member"
    for member in range(members):
        for step in range(1, steps + 1):
            yield root / f"{member:02d}" / f"{step:02d}.nc"


def raw_status(raw_dir: Path, date: pd.Timestamp, members: int, steps: int):
    present = 0
    missing = []
    for path in expected_files(raw_dir, date, members, steps):
        if path.exists() and path.stat().st_size > 0:
            present += 1
        elif len(missing) < 5:
            missing.append(path)
    return present == members * steps, present, missing


def is_done(raw_dir: Path, date: pd.Timestamp, members: int, steps: int) -> bool:
    return raw_status(raw_dir, date, members, steps)[0]


def input_file(input_dir: Path, date: pd.Timestamp) -> Path:
    return input_dir / ymd(date) / "input.nc"


def run_inference(
    date: pd.Timestamp,
    input_dir: Path,
    raw_dir: Path,
    device: str,
    members: int,
    steps: int,
    log: logging.Logger,
) -> bool:
    date_str = ymd(date)
    input_path = input_file(input_dir, date)
    out_dir = raw_dir / date_str

    if not input_path.exists():
        log.error("SKIP   %s  no input.nc at %s", date_str, input_path)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        FUXI_PYTHON,
        "inference.py",
        "--model",
        str(MODEL_PATH),
        "--input",
        str(input_path),
        "--device",
        device,
        "--total_step",
        str(steps),
        "--total_member",
        str(members),
        "--save_dir",
        str(out_dir),
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = FUXI_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"] = "false"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["KMP_AFFINITY"] = "disabled"

    log.info("START  %s  (%d members x %d steps)", date_str, members, steps)
    result = subprocess.run(cmd, cwd=str(FUXI_DIR), env=env)
    complete, present, missing = raw_status(raw_dir, date, members, steps)

    if result.returncode != 0:
        if complete:
            log.warning(
                "DONE   %s  raw complete, but child exited %d; treating as usable",
                date_str,
                result.returncode,
            )
            return True
        miss = ", ".join(str(path.relative_to(out_dir)) for path in missing)
        log.error(
            "FAIL   %s  exit=%d files=%d/%d missing=%s",
            date_str,
            result.returncode,
            present,
            members * steps,
            miss,
        )
        return False

    if not complete:
        miss = ", ".join(str(path.relative_to(out_dir)) for path in missing)
        log.error(
            "FAIL   %s  child exited 0 but raw incomplete files=%d/%d missing=%s",
            date_str,
            present,
            members * steps,
            miss,
        )
        return False

    log.info("DONE   %s", date_str)
    return True


def parse_dates(args) -> list[pd.Timestamp]:
    if args.date:
        return [pd.Timestamp(args.date)]
    return list(pd.date_range(args.start, args.end, freq="D"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FuXi-S2S 50-member inference for JJAS 2025")
    parser.add_argument("--date", type=str, default=None, help="Single init date YYYYMMDD")
    parser.add_argument("--start", type=str, default=DATE_START, help="Start init date YYYYMMDD")
    parser.add_argument("--end", type=str, default=DATE_END, help="End init date YYYYMMDD")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--members", type=int, default=TOTAL_MEMBERS)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument(
        "--allow-missing-inputs",
        action="store_true",
        help="Do not fail the job for dates whose input.nc is missing.",
    )
    args = parser.parse_args()

    dates = parse_dates(args)
    if not dates:
        raise SystemExit("No dates selected")

    log_file = LOG_DIR / f"ens50_jjas2025_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_file)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    pending = [date for date in dates if not is_done(args.raw_dir, date, args.members, args.steps)]
    done_n = len(dates) - len(pending)
    missing_inputs = [date for date in pending if not input_file(args.input_dir, date).exists()]
    runnable = [date for date in pending if input_file(args.input_dir, date).exists()]

    log.info("=" * 60)
    log.info("FuXi-S2S Inference - 50-member ensemble - JJAS 2025")
    log.info("  Model       : %s", MODEL_PATH)
    log.info("  Input dir   : %s", args.input_dir)
    log.info("  Raw out dir : %s", args.raw_dir)
    log.info("  Device      : %s", args.device)
    log.info("  Members     : %d  |  Steps: %d", args.members, args.steps)
    log.info("  Dates       : %d  (%s -> %s)", len(dates), dates[0].date(), dates[-1].date())
    log.info("  Already done: %d", done_n)
    log.info("  Runnable    : %d", len(runnable))
    log.info("  Missing input.nc: %d", len(missing_inputs))
    log.info("=" * 60)

    for date in missing_inputs[:20]:
        log.warning("MISSING_INPUT %s %s", ymd(date), input_file(args.input_dir, date))
    if len(missing_inputs) > 20:
        log.warning("MISSING_INPUT ... %d more", len(missing_inputs) - 20)
    if missing_inputs and not args.allow_missing_inputs:
        log.error("Inputs are missing; rerun with --allow-missing-inputs to process available dates only.")
        return 2

    failed = []
    success = 0
    for i, date in enumerate(runnable, 1):
        if run_inference(date, args.input_dir, args.raw_dir, args.device, args.members, args.steps, log):
            success += 1
        else:
            failed.append(ymd(date))
        log.info(
            "Progress %d/%d  done=%d  fail=%d",
            i,
            len(runnable),
            done_n + success,
            len(failed),
        )

    log.info("=" * 60)
    log.info("INFERENCE COMPLETE")
    log.info("  Success      : %d", success)
    log.info("  Failed       : %d", len(failed))
    log.info("  Missing input: %d", len(missing_inputs))
    for date_str in failed:
        log.info("    FAILED: %s", date_str)
    log.info("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
