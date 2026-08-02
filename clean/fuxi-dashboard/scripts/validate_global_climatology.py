#!/usr/bin/env python3
"""Audit exact-date global climatologies against selected native reforecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


YEARS = tuple(range(2002, 2022))
MEMBERS = 51
LEAD_DAYS = 42
VARIABLES = ("tp", "t2m", "z500")
SAMPLE_LEADS = (1, 42)
SAMPLE_POINTS = ((20, 0), (60, 80), (100, 160))
TOLERANCES = {"tp": 1e-5, "t2m": 5e-4, "z500": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reforecast-root", type=Path, required=True)
    parser.add_argument("--climatology-dir", type=Path, required=True)
    parser.add_argument("--initialization-slot", default="0728")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialization_path(root: Path, year: int, slot: str) -> Path:
    return root / str(year) / slot[:2] / f"{year}{slot}"


def validate_structure(root: Path, slot: str) -> int:
    checked = 0
    expected_members = {f"{member:02d}" for member in range(MEMBERS)}
    expected_files = {f"{lead:02d}.nc" for lead in range(1, LEAD_DAYS + 1)}
    for year in YEARS:
        member_root = initialization_path(root, year, slot) / "member"
        present_members = {
            path.name for path in member_root.iterdir() if path.is_dir()
        }
        if present_members != expected_members:
            raise ValueError(f"{member_root} does not contain members 00–50")
        for member in sorted(present_members):
            member_path = member_root / member
            present_files = {
                path.name
                for path in member_path.iterdir()
                if path.is_file() and path.stat().st_size > 0
            }
            if present_files != expected_files:
                raise ValueError(
                    f"{member_path} does not contain 42 complete lead files"
                )
            checked += len(present_files)
    return checked


def load_expected(
    directory: Path,
    slot: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    expected: dict[str, np.ndarray] = {}
    checksums: dict[str, str] = {}
    latitude_indices = np.asarray([point[0] for point in SAMPLE_POINTS])
    longitude_indices = np.asarray([point[1] for point in SAMPLE_POINTS])
    lead_indices = np.asarray(SAMPLE_LEADS) - 1
    for variable in VARIABLES:
        path = directory / f"{variable}_clima_{slot}.nc"
        dataset = xr.open_dataset(path, decode_times=False)
        try:
            description = str(dataset.attrs.get("description", ""))
            if "20 years x 51 members" not in description:
                raise ValueError(f"{path} has no complete-sample declaration")
            mean = dataset[f"{variable}_mean"].transpose("step", "lat", "lon")
            if mean.shape != (LEAD_DAYS, 121, 240):
                raise ValueError(f"{path} has an unexpected shape")
            values = np.asarray(mean.values, dtype=np.float64)
            expected[variable] = values[
                lead_indices[:, None],
                latitude_indices[None, :],
                longitude_indices[None, :],
            ]
            checksums[variable] = sha256(path)
        finally:
            dataset.close()
    return expected, checksums


def recalculate_samples(root: Path, slot: str) -> dict[str, np.ndarray]:
    recalculated = {
        variable: np.zeros((len(SAMPLE_LEADS), len(SAMPLE_POINTS)))
        for variable in VARIABLES
    }
    latitude_indices = np.asarray([point[0] for point in SAMPLE_POINTS])
    longitude_indices = np.asarray([point[1] for point in SAMPLE_POINTS])
    for year in YEARS:
        yearly = {
            variable: np.zeros((len(SAMPLE_LEADS), len(SAMPLE_POINTS)))
            for variable in VARIABLES
        }
        init_path = initialization_path(root, year, slot)
        for member in range(MEMBERS):
            for lead_index, lead_day in enumerate(SAMPLE_LEADS):
                path = init_path / "member" / f"{member:02d}" / f"{lead_day:02d}.nc"
                source = xr.open_dataarray(path, decode_times=False)
                try:
                    values = np.asarray(
                        source.sel(channel=list(VARIABLES))
                        .squeeze(drop=True)
                        .transpose("channel", "lat", "lon")
                        .values,
                        dtype=np.float64,
                    )
                finally:
                    source.close()
                selected = values[:, latitude_indices, longitude_indices]
                for variable_index, variable in enumerate(VARIABLES):
                    yearly[variable][lead_index] += selected[variable_index]
        for variable in VARIABLES:
            recalculated[variable] += yearly[variable] / MEMBERS / len(YEARS)
        print(f"audited {year} {slot}", flush=True)
    return recalculated


def main() -> None:
    args = parse_args()
    if args.initialization_slot != "0728":
        raise ValueError("the current publication contract requires slot 0728")
    file_count = validate_structure(
        args.reforecast_root,
        args.initialization_slot,
    )
    expected, checksums = load_expected(
        args.climatology_dir,
        args.initialization_slot,
    )
    recalculated = recalculate_samples(
        args.reforecast_root,
        args.initialization_slot,
    )
    maximum_differences: dict[str, float] = {}
    for variable in VARIABLES:
        difference = float(np.max(np.abs(recalculated[variable] - expected[variable])))
        maximum_differences[variable] = difference
        if difference > TOLERANCES[variable]:
            raise ValueError(
                f"{variable} climatology differs from native reforecasts by {difference}"
            )
    print(
        json.dumps(
            {
                "status": "validated",
                "years": list(YEARS),
                "native_members_per_year": MEMBERS,
                "lead_days": LEAD_DAYS,
                "complete_files": file_count,
                "sample_leads": list(SAMPLE_LEADS),
                "sample_points": [list(point) for point in SAMPLE_POINTS],
                "maximum_absolute_differences": maximum_differences,
                "source_sha256": checksums,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
