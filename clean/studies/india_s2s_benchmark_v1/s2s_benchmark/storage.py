from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from numcodecs import Blosc

from .core import StandardField, field_to_dataset, qc_summary, sha256_file


def _safe_experiment(experiment_id: str) -> str:
    return experiment_id.replace("/", "__")


def store_path(root: Path, field: StandardField, grid: str, year: int) -> Path:
    return root / "forecasts" / field.model / _safe_experiment(field.experiment_id) / field.variable / grid / f"{year}.zarr"


def _encoding(ds: xr.Dataset) -> dict[str, dict]:
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    encoding: dict[str, dict] = {}
    for name, variable in ds.data_vars.items():
        if variable.dtype.kind in "fiu":
            chunks = []
            for dim, size in variable.sizes.items():
                if dim == "init":
                    chunks.append(1)
                elif dim == "member":
                    chunks.append(min(10, size))
                elif dim in {"lead_day", "lead_week"}:
                    chunks.append(min(7, size))
                else:
                    chunks.append(size)
            encoding[name] = {"compressor": compressor, "chunks": tuple(chunks)}
    return encoding


def _store_metadata_checksum(path: Path) -> str:
    metadata = path / ".zmetadata"
    if not metadata.exists():
        raise FileNotFoundError(f"consolidated metadata missing: {metadata}")
    return sha256_file(metadata)


def write_store(
    fields: Iterable[StandardField],
    destination: Path,
    grid: str,
    *,
    member_axis: np.ndarray | None = None,
) -> dict:
    """Write an immutable store through a same-filesystem incomplete directory."""
    if destination.exists():
        raise FileExistsError(f"completed store already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".incomplete-{uuid.uuid4().hex[:10]}")
    temporary.mkdir()
    qcs = []
    init_values: list[str] = []
    all_source_paths: set[str] = set()
    first_field: StandardField | None = None
    try:
        for index, field in enumerate(fields):
            first_field = first_field or field
            ds = field_to_dataset(field, grid)
            if member_axis is not None:
                ds = ds.reindex(member=np.asarray(member_axis, dtype=np.int16))
                ds["member_available"] = ds["member_available"].fillna(False).astype(bool)
                ds["ensemble_member_count"] = ds["ensemble_member_count"].fillna(0).astype(np.int16)
                if "ensemble_member_count_weekly" in ds:
                    ds["ensemble_member_count_weekly"] = ds["ensemble_member_count_weekly"].fillna(0).astype(np.int16)
            if index == 0:
                ds.to_zarr(
                    temporary,
                    mode="w",
                    consolidated=False,
                    encoding=_encoding(ds),
                )
            else:
                ds.to_zarr(temporary, mode="a", append_dim="init", consolidated=False)
            qcs.append(qc_summary(field))
            all_source_paths.update(field.source_paths)
            init_values.append(field.initialization)
            ds.close()
        if first_field is None:
            raise ValueError("no fields supplied")
        import zarr

        group = zarr.open_group(str(temporary), mode="a")
        group.attrs["source_paths"] = json.dumps(sorted(all_source_paths))
        zarr.consolidate_metadata(str(temporary))
        with xr.open_zarr(temporary, consolidated=True) as reopened:
            if reopened.sizes["init"] != len(init_values):
                raise ValueError("read-back initialization count mismatch")
            reopened["forecast"].isel(init=0, lead_day=0).load()
        os.rename(temporary, destination)
    except Exception:
        # Preserve the incomplete store for diagnosis; never disguise it as complete.
        raise
    return {
        "schema_version": 1,
        "archive_id": "india_s2s_benchmark_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
        "model": first_field.model,
        "experiment_id": first_field.experiment_id,
        "variable": first_field.variable,
        "grid": grid,
        "year": int(init_values[0][:4]),
        "store": str(destination),
        "initializations": init_values,
        "initialization_count": len(init_values),
        "distribution_representation": first_field.distribution_representation,
        "units": first_field.units,
        "temporal_statistic": first_field.temporal_statistic,
        "source_paths": sorted(all_source_paths),
        "zmetadata_sha256": _store_metadata_checksum(destination),
        "qc": qcs,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise FileExistsError(f"immutable manifest differs: {path}")
        return
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recover_manifest(
    fields: Iterable[StandardField],
    destination: Path,
    grid: str,
    validation: dict,
) -> dict:
    """Recover only a missing manifest after a store passed read-back validation.

    A worker can stop after the atomic store rename but before its manifest is
    written. Recovery never changes the store: every regenerated source field
    is compared exactly with the stored forecast before metadata are emitted.
    """
    if not destination.exists():
        raise FileNotFoundError(f"cannot recover manifest without store: {destination}")
    if validation.get("status") != "passed" or validation.get("store") != str(destination):
        raise ValueError("manifest recovery requires validation for the same passed store")

    qcs = []
    init_values: list[str] = []
    all_source_paths: set[str] = set()
    first_field: StandardField | None = None
    with xr.open_zarr(destination, consolidated=True) as stored:
        stored_members = stored["member"].values
        for index, field in enumerate(fields):
            if index >= stored.sizes["init"]:
                raise ValueError(f"too many regenerated initializations for {destination}")
            first_field = first_field or field
            expected = field_to_dataset(field, grid).reindex(member=stored_members)
            expected_forecast = expected["forecast"].isel(init=0).values
            stored_forecast = stored["forecast"].isel(init=index).values
            if not np.array_equal(stored_forecast, expected_forecast, equal_nan=True):
                raise ValueError(
                    f"{destination}: regenerated forecast differs at initialization {field.initialization}"
                )
            expected_init = np.datetime64(field.initialization, "ns")
            if stored["init"].values[index] != expected_init:
                raise ValueError(
                    f"{destination}: initialization order differs at index {index}"
                )
            qcs.append(qc_summary(field))
            all_source_paths.update(field.source_paths)
            init_values.append(field.initialization)
            expected.close()
        if first_field is None:
            raise ValueError("no fields supplied for manifest recovery")
        if len(init_values) != stored.sizes["init"]:
            raise ValueError(
                f"{destination}: recovered {len(init_values)} of {stored.sizes['init']} initializations"
            )
        stored_sources = set(json.loads(stored.attrs["source_paths"]))
        if stored_sources != all_source_paths:
            raise ValueError(f"{destination}: regenerated source paths differ from stored metadata")

    return {
        "schema_version": 1,
        "archive_id": "india_s2s_benchmark_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
        "model": first_field.model,
        "experiment_id": first_field.experiment_id,
        "variable": first_field.variable,
        "grid": grid,
        "year": int(init_values[0][:4]),
        "store": str(destination),
        "initializations": init_values,
        "initialization_count": len(init_values),
        "distribution_representation": first_field.distribution_representation,
        "units": first_field.units,
        "temporal_statistic": first_field.temporal_statistic,
        "source_paths": sorted(all_source_paths),
        "zmetadata_sha256": _store_metadata_checksum(destination),
        "qc": qcs,
        "recovery": {
            "reason": "store rename completed before manifest write",
            "store_mutated": False,
            "forecast_exact_match": True,
        },
    }


def validate_store(path: Path) -> dict:
    with xr.open_zarr(path, consolidated=True) as ds:
        required = {"forecast", "ensemble_mean", "ensemble_std", "ensemble_member_count", "member_available"}
        missing = required - set(ds.data_vars)
        if missing:
            raise ValueError(f"{path}: missing variables {sorted(missing)}")
        forecast = ds["forecast"].values
        # Match ensemble_statistics exactly. NumPy otherwise accumulates
        # float32 inputs in float32, which can create false validation failures
        # for large ensembles even though the stored statistic was correctly
        # accumulated in float64 and then encoded as float32.
        expected_mean = np.nanmean(forecast, axis=1, dtype=np.float64).astype(np.float32)
        expected_std = np.nanstd(
            forecast, axis=1, ddof=0, dtype=np.float64
        ).astype(np.float32)
        if not np.allclose(ds["ensemble_mean"].values, expected_mean, equal_nan=True, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{path}: ensemble mean mismatch")
        if not np.allclose(ds["ensemble_std"].values, expected_std, equal_nan=True, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{path}: ensemble standard deviation mismatch")
        if "forecast_weekly_total" in ds:
            if not np.allclose(
                ds["forecast_weekly_total"].values,
                7.0 * ds["forecast_weekly_mean"].values,
                equal_nan=True,
                rtol=2e-6,
                atol=2e-5,
            ):
                raise ValueError(f"{path}: weekly precipitation identity failed")
        source_paths = json.loads(ds.attrs["source_paths"])
        missing_sources = [source for source in source_paths if not Path(source).exists()]
        if missing_sources:
            raise FileNotFoundError(f"{path}: missing sources {missing_sources}")
        source_ensemble_sizes = None
        if ds.attrs["distribution_representation"] == "mean_only":
            if "source_ensemble_size" not in ds:
                raise ValueError(f"{path}: mean-only field lacks source ensemble size")
            source_ensemble_sizes = sorted(
                np.unique(ds["source_ensemble_size"].values).astype(int).tolist()
            )
            if not source_ensemble_sizes or source_ensemble_sizes[0] < 1:
                raise ValueError(f"{path}: invalid source ensemble size")
        result = {
            "store": str(path),
            "status": "passed",
            "model": ds.attrs["model"],
            "experiment_id": ds.attrs["experiment_id"],
            "variable": ds.attrs["variable"],
            "grid": ds.attrs["grid_id"],
            "initialization_count": ds.sizes["init"],
            "member_count": ds.sizes["member"],
            "lead_day_count": ds.sizes["lead_day"],
            "nonfinite_count": int((~np.isfinite(forecast)).sum()),
            "negative_tp_count": int((forecast < 0).sum()) if ds.attrs["variable"] == "tp" else 0,
            "source_ensemble_sizes": source_ensemble_sizes,
            "zmetadata_sha256": _store_metadata_checksum(path),
        }
    return result
