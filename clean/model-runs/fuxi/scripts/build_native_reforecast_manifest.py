#!/usr/bin/env python3
"""Create the exact JJAS task manifest for the native FuXi reforecast archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ARCHIVE = Path("/storage/raj.ayush/archive/All_Model_Data/models/fuxi/data")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "config/fuxi_native_reforecast_jjas_2002_2021.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for archive in sorted(args.archive_dir.glob("????????.7z")):
        stamp = archive.stem
        init = pd.Timestamp(stamp)
        if not (2002 <= init.year <= 2021 and init.month in {6, 7, 8, 9}):
            continue
        rows.append(
            {
                "task_index": len(rows),
                "init_date": init.strftime("%Y-%m-%d"),
                "init_yyyymmdd": stamp,
                "year": init.year,
                "archive_path": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "archive_mtime_utc": pd.Timestamp(archive.stat().st_mtime, unit="s", tz="UTC").isoformat(),
            }
        )
    manifest = pd.DataFrame(rows)
    if len(manifest) != 700 or manifest.groupby("year").size().to_dict() != {year: 35 for year in range(2002, 2022)}:
        raise ValueError(
            "Expected 700 JJAS archives (35 for every year 2002--2021); got "
            f"{len(manifest)} with yearly counts {manifest.groupby('year').size().to_dict()}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"wrote {len(manifest)} rows to {args.output}")


if __name__ == "__main__":
    main()
