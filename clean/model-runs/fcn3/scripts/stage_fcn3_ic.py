#!/usr/bin/env python3
"""Stage one exact D 00 UTC 72-channel ARCO-ERA5 FCN3 initial state."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from fcn3_common import (
    existing_pair_is_valid,
    load_config,
    product_paths,
    select_date,
    sha256_file,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--init-date")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    calendar_index, init_date = select_date(config, args.index, args.init_date)
    paths = product_paths(config, init_date)
    if not args.force and existing_pair_is_valid(
        paths["stage"], paths["stage_manifest"], "input_sha256"
    ):
        print(f"validated existing staged IC: {paths['stage']}", flush=True)
        return
    if (paths["stage"].exists() or paths["stage_manifest"].exists()) and not args.force:
        raise RuntimeError("partial or invalid staged IC exists; inspect before --force")

    from earth2studio.data import ARCO
    from earth2studio.data.utils import fetch_data
    from earth2studio.models.px.fcn3 import VARIABLES

    started = time.perf_counter()
    init = np.array([np.datetime64(init_date, "ns")])
    variables = np.asarray(VARIABLES)
    source = ARCO(cache=True, verbose=True, async_timeout=1800)
    print(f"fetching exact ARCO state init={init_date} variables={len(variables)}", flush=True)
    state, coords = fetch_data(
        source=source,
        time=init,
        variable=variables,
        lead_time=np.array([np.timedelta64(0, "h")]),
        device=torch.device("cpu"),
    )
    expected_shape = (1, 1, 72, 721, 1440)
    if tuple(state.shape) != expected_shape:
        raise ValueError(f"unexpected FCN3 IC shape {tuple(state.shape)}")
    if list(coords) != ["time", "lead_time", "variable", "lat", "lon"]:
        raise ValueError(f"unexpected FCN3 IC dimensions {list(coords)}")
    if not np.array_equal(coords["variable"], variables):
        raise ValueError("FCN3 IC variable order changed")
    if not np.array_equal(coords["lat"], np.linspace(90.0, -90.0, 721)):
        raise ValueError("FCN3 IC latitude coordinates are invalid")
    if not np.array_equal(coords["lon"], np.linspace(0, 360, 1440, endpoint=False)):
        raise ValueError("FCN3 IC longitude coordinates are invalid")
    if not torch.isfinite(state).all():
        raise ValueError("FCN3 IC contains non-finite values")

    values = state.numpy().astype(np.float32, copy=False)
    dataset = xr.Dataset(
        {"state": (tuple(coords), values)},
        coords={name: value for name, value in coords.items()},
        attrs={
            "init_date": init_date,
            "source": config["initial_conditions"]["source"],
            "source_path": ARCO.ARCO_PATH,
            "source_policy": config["initial_conditions"]["source_policy"],
            "model_package_uri": config["model"]["package_uri"],
        },
    )
    paths["stage"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["stage"].with_name(f".{paths['stage'].name}.{os.getpid()}.tmp")
    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={"state": {"zlib": True, "complevel": 2, "shuffle": True}},
    )
    with xr.open_dataset(temporary) as check:
        if tuple(check["state"].shape) != expected_shape:
            raise ValueError("written FCN3 IC shape is invalid")
        if not np.isfinite(check["state"].values).all():
            raise ValueError("written FCN3 IC is non-finite")
    os.replace(temporary, paths["stage"])
    digest = sha256_file(paths["stage"])
    record = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calendar_index": calendar_index,
        "init_date": init_date,
        "calendar_sha256": config["calendar"]["sha256"],
        "source": config["initial_conditions"]["source"],
        "source_path": ARCO.ARCO_PATH,
        "selection": "exact D 00 UTC; no temporal interpolation or averaging",
        "input_path": str(paths["stage"]),
        "input_size_bytes": paths["stage"].stat().st_size,
        "input_sha256": digest,
        "shape": list(expected_shape),
        "variables": variables.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    write_json_atomic(record, paths["stage_manifest"])
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
