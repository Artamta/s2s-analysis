#!/usr/bin/env python3
"""Stage one frozen NeuralGCM initial condition from hourly ARCO-ERA5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import neuralgcm
import numpy as np
import xarray as xr
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils


COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))

import neuralgcm_run_contract as run_contract  # noqa: E402
import pilot_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=run_contract.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def verify_checkpoint(config: dict[str, Any]) -> tuple[Path, str]:
    checkpoint = run_contract.storage_path(config, "checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is not provisioned: {checkpoint}")
    expected_size = int(config["model"]["checkpoint_size_bytes"])
    if checkpoint.stat().st_size != expected_size:
        raise ValueError(
            f"checkpoint size mismatch: expected {expected_size}, "
            f"found {checkpoint.stat().st_size}"
        )
    digest = pilot_contract.sha256_file(checkpoint)
    if digest != config["model"]["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    return checkpoint, digest


def existing_stage_is_valid(input_path: Path, manifest_path: Path) -> bool:
    if not input_path.is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("status") == "complete"
        and manifest.get("input_sha256") == pilot_contract.sha256_file(input_path)
    )


def stage(config: dict[str, Any], force: bool) -> tuple[Path, Path]:
    input_path = run_contract.storage_path(config, "input")
    manifest_path = run_contract.storage_path(config, "input_manifest")
    if not force and existing_stage_is_valid(input_path, manifest_path):
        print(f"validated existing staged IC: {input_path}", flush=True)
        return input_path, manifest_path
    if (input_path.exists() or manifest_path.exists()) and not force:
        raise RuntimeError(
            "partial or invalid staged IC exists; inspect before using --force"
        )

    checkpoint, checkpoint_hash = verify_checkpoint(config)
    with checkpoint.open("rb") as handle:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(handle))

    variables = list(dict.fromkeys(model.input_variables + model.forcing_variables))
    init_time = np.datetime64(config["case"]["init_time"])
    forcing_time = np.datetime64(config["initial_conditions"]["forcing_source_time"])
    source = xr.open_zarr(
        config["source"]["url"],
        chunks=None,
        storage_options=config["source"]["storage_options"],
    )
    try:
        missing = sorted(set(variables).difference(source.data_vars))
        if missing:
            raise KeyError(f"ARCO source is missing checkpoint variables: {missing}")
        shifted = xarray_utils.selective_temporal_shift(
            source[variables],
            variables=model.forcing_variables,
            time_shift="24 hours",
        )
        selected = shifted.sel(time=[init_time]).compute()
        direct_forcing = source[model.forcing_variables].sel(
            time=[forcing_time]
        ).compute()
        for variable in model.forcing_variables:
            np.testing.assert_allclose(
                selected[variable].isel(time=0).values,
                direct_forcing[variable].isel(time=0).values,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
                err_msg=f"24-hour forcing shift failed for {variable}",
            )
        era5_grid = spherical_harmonic.Grid(
            latitude_nodes=source.sizes["latitude"],
            longitude_nodes=source.sizes["longitude"],
            latitude_spacing=xarray_utils.infer_latitude_spacing(source.latitude),
            longitude_offset=xarray_utils.infer_longitude_offset(source.longitude),
        )
    finally:
        source.close()

    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True
    )
    native = xarray_utils.regrid(selected, regridder)
    native = xarray_utils.fill_nan_with_nearest(native).astype(np.float32)

    expected_lon = int(config["model"]["native_longitude_count"])
    expected_lat = int(config["model"]["native_latitude_count"])
    if native.sizes.get("time") != 1:
        raise ValueError("staged IC must contain one time")
    if native.sizes.get("longitude") != expected_lon or native.sizes.get("latitude") != expected_lat:
        raise ValueError(
            "unexpected native grid: "
            f"{native.sizes.get('longitude')}x{native.sizes.get('latitude')}"
        )
    expected_levels = np.asarray(model.data_coords.vertical.centers)
    if not np.array_equal(native.level.values, expected_levels):
        raise ValueError("staged pressure levels differ from checkpoint")
    for variable in variables:
        if not np.isfinite(native[variable].values).all():
            raise ValueError(f"staged variable {variable} contains non-finite values")

    request_contract = {
        "source": config["source"]["url"],
        "atmosphere_source_time": config["initial_conditions"]["atmosphere_source_time"],
        "forcing_source_time": config["initial_conditions"]["forcing_source_time"],
        "input_variables": model.input_variables,
        "forcing_variables": model.forcing_variables,
        "source_grid": "721x1440 regular 0.25 degree",
        "target_grid": f"{expected_lon}x{expected_lat} Gaussian",
        "regrid": config["initial_conditions"]["horizontal_regrid"],
        "fill": config["initial_conditions"]["fill_policy"],
    }
    native.attrs.update(
        {
            "run_label": config["run_label"],
            "purpose": config["purpose"],
            "forecast_reference_time": config["case"]["forecast_reference_time"],
            "information_cutoff_time": config["case"]["information_cutoff_time"],
            "atmosphere_source_time": config["initial_conditions"]["atmosphere_source_time"],
            "forcing_source_time": config["initial_conditions"]["forcing_source_time"],
            "forcing_policy": config["initial_conditions"]["forcing_policy"],
            "source_url": config["source"]["url"],
            "source_request_sha256": canonical_hash(request_contract),
            "checkpoint_sha256": checkpoint_hash,
            "paper_status": "42-day pilot input only; not for skill scores",
        }
    )

    input_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = input_path.with_name(f".{input_path.name}.{os.getpid()}.tmp")
    encoding = {
        variable: {"zlib": True, "complevel": 1, "shuffle": True}
        for variable in native.data_vars
    }
    native.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    with xr.open_dataset(temporary) as check:
        if set(check.data_vars) != set(variables):
            raise ValueError("written IC variables differ from checkpoint contract")
        check.load()
    os.replace(temporary, input_path)

    input_hash = pilot_contract.sha256_file(input_path)
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_label": config["run_label"],
        "repository_commit": repository_commit(),
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": input_hash,
        "source_request": request_contract,
        "source_request_sha256": canonical_hash(request_contract),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "forcing_shift_verified_against_direct_D_minus_1": True,
        "dimensions": {key: int(value) for key, value in native.sizes.items()},
        "variables": variables,
    }
    write_json_atomic(manifest, manifest_path)
    print(f"staged NeuralGCM IC: {input_path}", flush=True)
    print(f"input SHA256: {input_hash}", flush=True)
    return input_path, manifest_path


def main() -> None:
    args = parse_args()
    config = run_contract.load_config(args.config)
    stage(config, args.force)


if __name__ == "__main__":
    main()
