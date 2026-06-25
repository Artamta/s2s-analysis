#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_fuxi_archives_persistent.py
===================================
Persistently extract FuXi-S2S .7z hindcast archives without deleting the
extracted contents.

Default archive layout:
  /storage/raj.ayush/archive/All_Model_Data/models/fuxi/data/YYYYMMDD.7z

Default output layout:
  /storage/raj.ayush/archive/All_Model_Data/models/fuxi/extracted_full/
    YYYY/
      MM/
        YYYYMMDD/
          member/
            00/01.nc ... 00/42.nc
            ...

This script is intentionally different from the compact JJAS preprocessor:
it keeps the full raw extracted data, including all channels, members, and
lead-time files. It never removes an extracted date directory.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd


DEFAULT_ARCHIVE_DIR = Path("/storage/raj.ayush/archive/All_Model_Data/models/fuxi/data")
DEFAULT_OUT_ROOT = Path("/storage/raj.ayush/archive/All_Model_Data/models/fuxi/extracted_full")


def parse_months(value):
    if not value:
        return None
    months = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit() or not 1 <= int(item) <= 12:
            raise SystemExit(f"bad month: {item}")
        months.add(f"{int(item):02d}")
    return months


def discover_archives(args):
    archive_dir = Path(args.archive_dir)
    months = parse_months(args.months)
    archives = []
    for path in sorted(archive_dir.glob("*.7z")):
        date = path.stem
        if len(date) != 8 or not date.isdigit():
            continue
        pd.Timestamp(date)  # validate date
        if args.start and date < args.start:
            continue
        if args.end and date > args.end:
            continue
        if args.year and date[:4] != str(args.year):
            continue
        if months and date[4:6] not in months:
            continue
        archives.append(path)
    if args.max_count:
        archives = archives[:args.max_count]
    if args.array_index is not None:
        if args.array_index < 0 or args.array_index >= len(archives):
            raise SystemExit(
                f"array index {args.array_index} outside archive list length {len(archives)}"
            )
        archives = [archives[args.array_index]]
    return archives


def output_parent(out_root, date):
    return Path(out_root) / date[:4] / date[4:6]


def output_date_dir(out_root, date):
    return output_parent(out_root, date) / date


def archive_layout_date_dir(out_root, date):
    """Some older FuXi archives contain an extra top-level year directory."""
    return output_parent(out_root, date) / date[:4] / date


def candidate_date_dirs(out_root, date):
    target = output_date_dir(out_root, date)
    nested = archive_layout_date_dir(out_root, date)
    return [target] if nested == target else [target, nested]


def status_path(out_root, date):
    return output_parent(out_root, date) / f"{date}.extract_status.json"


def complete_marker(date_dir):
    return Path(date_dir) / ".complete"


def write_status(out_root, date, status, extra=None):
    parent = output_parent(out_root, date)
    parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date,
        "status": status,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    status_path(out_root, date).write_text(json.dumps(payload, indent=2) + "\n")


def validate_date_dir(target, expected_members, expected_steps):
    member_root = target / "member"
    if not member_root.is_dir():
        return False, {"reason": "missing member directory", "file_count": 0}
    files = sorted(member_root.glob("*/*.nc"))
    member_dirs = sorted(p for p in member_root.iterdir() if p.is_dir())
    expected_files = expected_members * expected_steps
    ok = len(member_dirs) >= expected_members and len(files) >= expected_files
    return ok, {
        "member_count": len(member_dirs),
        "file_count": len(files),
        "expected_members": expected_members,
        "expected_steps": expected_steps,
        "expected_files": expected_files,
    }


def find_valid_date_dir(out_root, date, expected_members, expected_steps, normalize=False):
    """Return a valid extracted date dir.

    The canonical layout is YYYY/MM/YYYYMMDD. Older archives extract as
    YYYY/MM/YYYY/YYYYMMDD; when `normalize` is true and the canonical target is
    absent, move the valid nested date directory up to the canonical location.
    """
    target = output_date_dir(out_root, date)
    for cand in candidate_date_dirs(out_root, date):
        if not cand.exists():
            continue
        ok, info = validate_date_dir(cand, expected_members, expected_steps)
        if not ok:
            continue
        if normalize and cand != target and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            cand.rename(target)
            info["normalized_from"] = str(cand)
            info["normalized_to"] = str(target)
            return target, info
        return cand, info
    return None, None


def any_existing_date_dir(out_root, date):
    return any(cand.exists() for cand in candidate_date_dirs(out_root, date))


def extract_one(archive, out_root, args):
    date = archive.stem
    parent = output_parent(out_root, date)
    target = output_date_dir(out_root, date)

    valid_dir, info = find_valid_date_dir(
        out_root, date, args.expected_members, args.expected_steps, normalize=True
    )
    if valid_dir is not None:
        marker = complete_marker(valid_dir)
        if marker.exists():
            print(f"SKIP complete {date} -> {valid_dir}", flush=True)
            return True
        if args.mark_complete_existing:
            marker.write_text(
                f"complete existing {pd.Timestamp.now().isoformat()}\n"
            )
            write_status(out_root, date, "complete-existing", info)
            print(f"MARK complete existing {date} -> {valid_dir}", flush=True)
            return True

    if any_existing_date_dir(out_root, date) and not args.extract_into_existing:
        write_status(out_root, date, "incomplete-existing", {"target": str(target)})
        print(
            f"STOP incomplete existing target for {date}: {target}\n"
            "Use --extract-into-existing to retry without deleting files, "
            "or choose a new --out-root.",
            flush=True,
        )
        return False

    parent.mkdir(parents=True, exist_ok=True)
    write_status(
        out_root,
        date,
        "running",
        {"archive": str(archive), "target": str(target), "archive_bytes": archive.stat().st_size},
    )
    print(f"START {date}: {archive} -> {parent}", flush=True)
    t0 = time.time()
    try:
        import py7zr
        with py7zr.SevenZipFile(str(archive), mode="r") as z:
            z.extractall(path=str(parent))
    except Exception as exc:
        write_status(out_root, date, "failed", {"error": repr(exc)})
        print(f"FAIL {date}: {exc}", flush=True)
        return False

    valid_dir, info = find_valid_date_dir(
        out_root, date, args.expected_members, args.expected_steps, normalize=True
    )
    ok = valid_dir is not None
    if info is None:
        info = {"reason": "no valid extracted date directory"}
    info["elapsed_min"] = round((time.time() - t0) / 60.0, 2)
    if not ok and args.validate:
        write_status(out_root, date, "failed-validation", info)
        print(f"FAIL validation {date}: {info}", flush=True)
        return False

    marker = complete_marker(valid_dir)
    marker.write_text(f"complete {pd.Timestamp.now().isoformat()}\n")
    write_status(out_root, date, "complete", info)
    print(
        f"DONE {date}: {valid_dir} members={info.get('member_count')} "
        f"files={info.get('file_count')} elapsed={info['elapsed_min']} min",
        flush=True,
    )
    return True


def main():
    p = argparse.ArgumentParser(
        description="Persistently extract FuXi-S2S .7z archives into year/month/date folders."
    )
    p.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--start", default=None, help="first YYYYMMDD archive date")
    p.add_argument("--end", default=None, help="last YYYYMMDD archive date")
    p.add_argument("--year", type=int, default=None, help="single year")
    p.add_argument("--months", default=None, help="comma list, e.g. 06,07,08,09")
    p.add_argument("--array-index", type=int, default=None,
                   help="extract only the Nth archive after filtering; for SLURM arrays")
    p.add_argument("--max-count", type=int, default=None)
    p.add_argument("--expected-members", type=int, default=51)
    p.add_argument("--expected-steps", type=int, default=42)
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument("--mark-complete-existing", action="store_true",
                   help="if an existing target validates, add a .complete marker and skip")
    p.add_argument("--extract-into-existing", action="store_true",
                   help="retry extraction into an existing incomplete target without deleting files")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    archives = discover_archives(args)
    print(f"archive count: {len(archives)}", flush=True)
    print(f"archive dir  : {args.archive_dir}", flush=True)
    print(f"out root     : {args.out_root}", flush=True)
    if args.dry_run:
        for archive in archives:
            date = archive.stem
            print(f"{date} -> {output_date_dir(args.out_root, date)}")
        return
    if not archives:
        raise SystemExit("no archives selected")

    ok = fail = 0
    for archive in archives:
        if extract_one(archive, args.out_root, args):
            ok += 1
        else:
            fail += 1
    print(f"finished: ok={ok} fail={fail}", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
