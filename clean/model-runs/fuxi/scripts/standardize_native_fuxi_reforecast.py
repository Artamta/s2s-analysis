#!/usr/bin/env python3
"""Standardize one native FuXi archive into a provenance-rich India NetCDF shard.

Each task processes exactly one YYYYMMDD.7z archive.  All 51 native members
and 42 leads are retained for precipitation and 2-metre temperature; only the
common 1.5-degree India domain is selected.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
import xarray as xr


MEMBERS = tuple(f"{member:02d}" for member in range(51))
LEADS = tuple(range(1, 43))
LATITUDE = np.arange(39.0, -0.1, -1.5, dtype=np.float32)
LONGITUDE = np.arange(60.0, 99.1, 1.5, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def validated_output(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(path) as dataset:
            return (
                dataset["tp"].shape == (51, 42, 27, 27)
                and dataset["t2m"].shape == (51, 42, 27, 27)
                and np.array_equal(dataset.member.values, np.arange(51, dtype=np.int16))
                and np.array_equal(dataset.lead_day.values, np.arange(1, 43, dtype=np.int16))
                and np.all(np.isfinite(dataset["tp"].isel(member=0, lead_day=0).values))
                and np.all(np.isfinite(dataset["t2m"].isel(member=0, lead_day=0).values))
            )
    except Exception:
        return False


def load_fields(path: Path, expected_init: pd.Timestamp) -> dict[str, np.ndarray]:
    with xr.open_dataset(path) as dataset:
        if len(dataset.data_vars) != 1:
            raise ValueError(f"Expected one data variable in {path}, found {list(dataset.data_vars)}")
        variable = dataset[next(iter(dataset.data_vars))]
        required_dims = {"time", "lead_time", "channel", "lat", "lon"}
        if set(variable.dims) != required_dims:
            raise ValueError(f"Unexpected dimensions in {path}: {variable.dims}")
        if variable.sizes["time"] != 1 or variable.sizes["lead_time"] != 1:
            raise ValueError(f"Expected one time and one lead in {path}")
        source_time = pd.Timestamp(dataset.time.values[0])
        if source_time != expected_init:
            raise ValueError(f"{path} reports time {source_time}, expected {expected_init}")
        required_channels = {"tp", "t2m"}
        if not required_channels.issubset(set(dataset.channel.values.tolist())):
            raise ValueError(f"{path} lacks channels: {sorted(required_channels.difference(set(dataset.channel.values.tolist())))}")
        fields: dict[str, np.ndarray] = {}
        for channel in ("tp", "t2m"):
            selected = variable.sel(channel=channel, lat=LATITUDE, lon=LONGITUDE).squeeze(("time", "lead_time"))
            if selected.shape != (27, 27):
                raise ValueError(f"Unexpected India shape in {path}: {selected.shape}")
            values = selected.values.astype(np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {channel} values in {path}")
            fields[channel] = values
    return fields


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[manifest.task_index == args.task_index]
    if len(row) != 1:
        raise ValueError(f"Task index {args.task_index} is not unique in {args.manifest}")
    task = row.iloc[0]
    init_stamp = str(task.init_yyyymmdd)
    init_time = pd.Timestamp(task.init_date)
    archive = Path(task.archive_path)
    output = args.output_dir / f"{init_stamp}.nc"
    qc_output = args.output_dir / "qc" / f"{init_stamp}.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc_output.parent.mkdir(parents=True, exist_ok=True)
    if validated_output(output):
        print(f"already validated: {output}")
        return
    if not archive.is_file() or archive.stat().st_size != int(task.archive_size_bytes):
        raise FileNotFoundError(f"Archive missing or changed since manifest creation: {archive}")

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"fuxi_native_{init_stamp}_", dir=args.scratch_root))
    try:
        # py7zr is in the shared s2s-hind environment, unlike /usr/bin/7z on
        # the GPU nodes.  Extracting into task-local scratch avoids network-FS
        # metadata storms and is removed in the finally block below.
        with py7zr.SevenZipFile(archive, mode="r") as source_archive:
            source_archive.extractall(path=scratch)
        root = scratch / init_stamp / "member"
        fields_by_variable = {"tp": [], "t2m": []}
        for member in MEMBERS:
            member_fields = {"tp": [], "t2m": []}
            for lead in LEADS:
                raw_file = root / member / f"{lead:02d}.nc"
                if not raw_file.is_file():
                    raise FileNotFoundError(f"Missing native member/lead file: {raw_file}")
                fields = load_fields(raw_file, init_time)
                for variable, values in fields.items():
                    member_fields[variable].append(values)
            for variable in fields_by_variable:
                fields_by_variable[variable].append(np.stack(member_fields[variable], axis=0))
        fields_by_variable = {variable: np.stack(values, axis=0) for variable, values in fields_by_variable.items()}
        starts = np.array([init_time + timedelta(days=lead - 1) for lead in LEADS], dtype="datetime64[ns]")
        ends = np.array([init_time + timedelta(days=lead) for lead in LEADS], dtype="datetime64[ns]")
        dataset = xr.Dataset(
            data_vars={
                "tp": (
                    ("member", "lead_day", "latitude", "longitude"),
                    fields_by_variable["tp"],
                    {
                        "long_name": "FuXi-S2S total precipitation mean rate",
                        "units": "mm h-1",
                        "cell_methods": "time: mean (24-hour forecast period)",
                        "conversion_to_mm_day-1": "multiply by 24",
                    },
                ),
                "t2m": (
                    ("member", "lead_day", "latitude", "longitude"),
                    fields_by_variable["t2m"],
                    {
                        "long_name": "FuXi-S2S 2 metre temperature",
                        "units": "K",
                        "cell_methods": "time: mean (24-hour forecast period)",
                    },
                ),
            },
            coords={
                "member": ("member", np.arange(51, dtype=np.int16), {"source_member_labels": "00 through 50"}),
                "lead_day": ("lead_day", np.arange(1, 43, dtype=np.int16)),
                "latitude": ("latitude", LATITUDE, {"units": "degrees_north"}),
                "longitude": ("longitude", LONGITUDE, {"units": "degrees_east"}),
                "forecast_reference_time": init_time.to_datetime64(),
                "forecast_period_start": ("lead_day", starts),
                "forecast_period_end": ("lead_day", ends),
            },
            attrs={
                "title": "Native FuXi-S2S reforecast standardized to the India 1.5 degree grid",
                "source_archive": str(archive),
                "source_archive_size_bytes": int(task.archive_size_bytes),
                "source_archive_mtime_utc": str(task.archive_mtime_utc),
                "source_member_count": 51,
                "source_lead_count": 42,
                "source_time_coordinate": init_time.isoformat(),
                "temporal_contract": "forecast period bounds derived as source initialization date plus lead day; verify against provider archive documentation before external publication",
                "history": "Extracted directly from native .7z archive; no compact hindcast product used.",
            },
        )
        encoding = {
            variable: {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32", "chunksizes": (1, 14, 27, 27)}
            for variable in ("tp", "t2m")
        }
        temporary = output.with_suffix(".nc.part")
        dataset.to_netcdf(temporary, encoding=encoding)
        temporary.replace(output)
        atomic_json(
            qc_output,
            {
                "task_index": int(task.task_index),
                "init_date": str(task.init_date),
                "source_archive": str(archive),
                "source_member_count": 51,
                "source_lead_count": 42,
                "output": str(output),
                "shape": list(fields_by_variable["tp"].shape),
                "finite": bool(all(np.isfinite(values).all() for values in fields_by_variable.values())),
                "tp_min_mm_h": float(fields_by_variable["tp"].min()),
                "tp_max_mm_h": float(fields_by_variable["tp"].max()),
                "tp_mean_mm_h": float(fields_by_variable["tp"].mean()),
                "t2m_min_k": float(fields_by_variable["t2m"].min()),
                "t2m_max_k": float(fields_by_variable["t2m"].max()),
                "t2m_mean_k": float(fields_by_variable["t2m"].mean()),
            },
        )
        print(f"wrote {output}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
