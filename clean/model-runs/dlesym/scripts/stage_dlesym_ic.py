#!/usr/bin/env python3
"""Fetch ERA5 and stage one compact native-HEALPix DLESyM initial condition."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from dlesym_common import (
    existing_pair_is_valid,
    load_config,
    product_paths,
    repository_commit,
    select_date,
    sha256_file,
    write_json_atomic,
)


def load_latlon_model(product: str, device: str):
    if product == "v1_t2m":
        from earth2studio.models.px import DLESyMLatLon

        package = DLESyMLatLon.load_default_package()
        return package, DLESyMLatLon.load_model(package, 0, 0).to(device)
    from earth2studio.models.px import DLESyMv0_ISCCP_ERA5LatLon

    package = DLESyMv0_ISCCP_ERA5LatLon.load_default_package()
    return package, DLESyMv0_ISCCP_ERA5LatLon.load_model(
        package, use_ttr=True
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--product", required=True, choices=["v1_t2m", "v0_tp_t2m"])
    parser.add_argument("--index", type=int)
    parser.add_argument("--init-date")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    calendar_index, init_date = select_date(config, args.index, args.init_date)
    paths = product_paths(config, args.product, init_date)
    if not args.force and existing_pair_is_valid(
        paths["stage"], paths["stage_manifest"], "input_sha256"
    ):
        print(f"validated existing staged IC: {paths['stage']}", flush=True)
        return
    if (paths["stage"].exists() or paths["stage_manifest"].exists()) and not args.force:
        raise RuntimeError("partial or invalid staged IC exists; inspect before --force")
    if not torch.cuda.is_available():
        raise RuntimeError("DLESyM staging requires a CUDA GPU")

    from earth2studio.data import ARCO
    from earth2studio.data.utils import fetch_data

    started = time.perf_counter()
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    package, model = load_latlon_model(args.product, device)
    input_coords = model.input_coords()
    init = np.datetime64(init_date, "ns")
    print(
        f"staging {args.product} index={calendar_index} init={init_date} "
        f"times={input_coords['lead_time']} variables={list(input_coords['variable'])}",
        flush=True,
    )
    source = ARCO(cache=True, verbose=True, async_timeout=1800)
    fetched, coords = fetch_data(
        source=source,
        time=np.array([init]),
        variable=np.asarray(input_coords["variable"]),
        lead_time=np.asarray(input_coords["lead_time"]),
        device=device,
    )
    fetched_variables = list(coords["variable"])
    variable_dimension = list(coords).index("variable")
    for variable_index, variable in enumerate(fetched_variables):
        selected = fetched.select(
            variable_dimension,
            torch.tensor([variable_index], device=fetched.device),
        )
        if variable == "sst":
            if not torch.isfinite(selected).any():
                raise ValueError("fetched ERA5 SST has no finite ocean values")
        elif not torch.isfinite(selected).all():
            raise ValueError(f"fetched ERA5 {variable} contains non-finite values")

    coords = coords.copy()
    if args.product == "v0_tp_t2m":
        variables = list(coords["variable"])
        variables[variables.index("ttr")] = "rlut"
        coords["variable"] = np.array(variables)
    state, coords = model._prepare_derived_variables(fetched, coords)
    state = model.to_hpx(state)
    coords = model.coords_to_hpx(coords)
    state = state.unsqueeze(0)
    coords = OrderedDict({"batch": np.array([0]), **coords})
    if args.product == "v0_tp_t2m":
        state = model._ttr_to_olr_hpx(state, coords)
    if not torch.isfinite(state).all():
        raise ValueError("prepared HEALPix initial condition contains non-finite values")

    values = state.detach().cpu().numpy().astype(np.float32)
    dataset = xr.Dataset(
        {"state": (tuple(coords), values)},
        coords={name: values for name, values in coords.items()},
        attrs={
            "product": args.product,
            "init_date": init_date,
            "source": "Google ARCO-ERA5 hourly analysis",
            "source_path": ARCO.ARCO_PATH,
            "package_uri": config["products"][args.product]["package_uri"],
            "preprocessing": "official Earth2Studio LatLon derived variables and HEALPix regrid",
            "radiation_transform": (
                "bundled day-of-year ERA5 TTR to ISCCP OLR moment matching"
                if args.product == "v0_tp_t2m"
                else "not applicable"
            ),
        },
    )
    paths["stage"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["stage"].with_name(f".{paths['stage'].name}.{os.getpid()}.tmp")
    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={"state": {"zlib": True, "complevel": 4, "shuffle": True}},
    )
    with xr.open_dataset(temporary) as check:
        if not np.isfinite(check["state"].values).all():
            raise ValueError("written staged IC is non-finite")
    os.replace(temporary, paths["stage"])
    digest = sha256_file(paths["stage"])
    elapsed = time.perf_counter() - started
    record = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "product": args.product,
        "calendar_index": calendar_index,
        "init_date": init_date,
        "calendar_sha256": config["calendar"]["sha256"],
        "package_uri": config["products"][args.product]["package_uri"],
        "input_path": str(paths["stage"]),
        "input_size_bytes": paths["stage"].stat().st_size,
        "input_sha256": digest,
        "shape": list(values.shape),
        "variables": list(coords["variable"]),
        "lead_hours": [
            int(value / np.timedelta64(1, "h")) for value in coords["lead_time"]
        ],
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "repository_commit": repository_commit(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    write_json_atomic(record, paths["stage_manifest"])
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
