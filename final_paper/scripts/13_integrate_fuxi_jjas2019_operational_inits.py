#!/usr/bin/env python3
"""Integrate staged FuXi JJAS2019 operational-date forecasts.

The FuXi archive already contains the official Sunday/Thursday hindcast dates
in ``/storage/raj.ayush/s2s_final_data/jjas/fuxi_combined``. This script adds
the staged Monday operational-date forecasts from ``/storage/raj.ayush/other``
as symlinks in that same combined directory, then writes a manifest.

It is deliberately conservative: existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGED_ROOT = Path(
    os.environ.get(
        "S2S_FUXI_JJAS_STAGED_ROOT",
        "/storage/raj.ayush/other/jjas2019_missing_forecasts/fuxi",
    )
)
DEFAULT_COMBINED_ROOT = Path(
    os.environ.get(
        "S2S_FUXI_JJAS_COMBINED_ROOT",
        "/storage/raj.ayush/s2s_final_data/jjas/fuxi_combined",
    )
)
DEFAULT_OUTPUT_MANIFEST = Path(
    os.environ.get(
        "S2S_FUXI_JJAS_MANIFEST",
        str(
            PROJECT_ROOT
            / "outputs"
            / "common"
            / "inventory"
            / "fuxi_jjas2019_operational35_manifest.csv"
        ),
    )
)


@dataclass(frozen=True)
class HeaderSummary:
    ok: bool
    channels: str = ""
    member_count: str = ""
    lead_count: str = ""
    lat_count: str = ""
    lon_count: str = ""
    notes: str = ""


def operational_dates() -> list[str]:
    """JJAS2019 ECMWF/UKMO Monday/Thursday initialization dates."""

    current = date(2019, 6, 3)
    end = date(2019, 9, 30)
    out: list[str] = []
    while current <= end:
        if current.weekday() in {0, 3}:  # Monday, Thursday
            out.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return out


def header_summary(path: Path) -> HeaderSummary:
    try:
        import xarray as xr

        ds = xr.open_dataset(path, chunks={})
        try:
            if "forecast" not in ds:
                return HeaderSummary(False, notes="forecast variable missing")
            channels = ",".join(str(x) for x in ds["channel"].values) if "channel" in ds.coords else ""
            return HeaderSummary(
                True,
                channels=channels,
                member_count=str(ds.sizes.get("member", "")),
                lead_count=str(ds.sizes.get("lead_time", ds.sizes.get("lead", ""))),
                lat_count=str(ds.sizes.get("lat", ds.sizes.get("latitude", ""))),
                lon_count=str(ds.sizes.get("lon", ds.sizes.get("longitude", ""))),
            )
        finally:
            ds.close()
    except Exception as exc:
        return HeaderSummary(False, notes=f"{type(exc).__name__}: {exc}")


def link_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.exists():
        return "file"
    return "missing"


def build_rows(
    staged_root: Path,
    combined_root: Path,
    *,
    dry_run: bool,
    validate_headers: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    combined_root.mkdir(parents=True, exist_ok=True)

    for init_date in operational_dates():
        dest = combined_root / f"{init_date}.nc"
        staged = staged_root / init_date / f"{init_date}.nc"
        action = "kept_existing"
        source_kind = "existing_combined"
        source_path = str(dest)
        notes = ""

        if not dest.exists():
            source_kind = "staged_missing_forecasts"
            source_path = str(staged)
            if not staged.is_file() or staged.stat().st_size <= 0:
                action = "missing_staged_source"
                notes = "destination absent and staged source missing/empty"
            elif dry_run:
                action = "would_create_symlink"
            else:
                os.symlink(staged, dest)
                action = "created_symlink"

        resolved = ""
        link_target = ""
        if dest.exists() or dest.is_symlink():
            try:
                resolved = str(dest.resolve(strict=True))
            except FileNotFoundError:
                resolved = ""
            if dest.is_symlink():
                link_target = os.readlink(dest)

        size_bytes = ""
        if dest.exists():
            size_bytes = str(dest.stat().st_size)
        elif staged.exists():
            size_bytes = str(staged.stat().st_size)

        header = header_summary(dest) if validate_headers and dest.exists() else HeaderSummary(True)
        if header.notes:
            notes = f"{notes}; {header.notes}".strip("; ")

        rows.append(
            {
                "init_date": init_date,
                "weekday": date(int(init_date[:4]), int(init_date[4:6]), int(init_date[6:])).strftime("%a"),
                "destination_path": str(dest),
                "destination_type": link_type(dest),
                "source_kind": source_kind,
                "source_path": source_path,
                "resolved_path": resolved,
                "link_target": link_target,
                "action": action,
                "exists": str(dest.exists()),
                "size_bytes": size_bytes,
                "header_ok": str(header.ok),
                "channels": header.channels,
                "member_count": header.member_count,
                "lead_count": header.lead_count,
                "lat_count": header.lat_count,
                "lon_count": header.lon_count,
                "notes": notes,
            }
        )
    return rows


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "init_date",
        "weekday",
        "destination_path",
        "destination_type",
        "source_kind",
        "source_path",
        "resolved_path",
        "link_target",
        "action",
        "exists",
        "size_bytes",
        "header_ok",
        "channels",
        "member_count",
        "lead_count",
        "lat_count",
        "lon_count",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-root", type=Path, default=DEFAULT_STAGED_ROOT)
    parser.add_argument("--combined-root", type=Path, default=DEFAULT_COMBINED_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-headers",
        action="store_true",
        help="Open each destination NetCDF header after linking.",
    )
    args = parser.parse_args()

    rows = build_rows(
        args.staged_root,
        args.combined_root,
        dry_run=args.dry_run,
        validate_headers=args.validate_headers,
    )
    write_manifest(rows, args.manifest)
    combined_manifest = args.combined_root / args.manifest.name
    if combined_manifest != args.manifest:
        write_manifest(rows, combined_manifest)

    summary = {
        "operational_dates": len(rows),
        "created_symlink": sum(row["action"] == "created_symlink" for row in rows),
        "kept_existing": sum(row["action"] == "kept_existing" for row in rows),
        "would_create_symlink": sum(row["action"] == "would_create_symlink" for row in rows),
        "missing_staged_source": sum(row["action"] == "missing_staged_source" for row in rows),
        "header_failures": sum(row["header_ok"] != "True" for row in rows),
        "manifest": str(args.manifest),
        "combined_manifest": str(combined_manifest),
    }
    print(json.dumps(summary, indent=2))
    if summary["missing_staged_source"] or summary["header_failures"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
