#!/usr/bin/env python3
"""Validate AFNOv2 precipitation directly on exact ERA5 atmospheric states."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from fcn3_common import (
    conservative_weights,
    load_config,
    product_paths,
    sha256_file,
    target_coordinates,
    write_json_atomic,
)


def regrid_fields(
    fields: np.ndarray,
    lat_weights: np.ndarray,
    lon_weights: np.ndarray,
) -> np.ndarray:
    lat_nonzero = np.flatnonzero(np.any(lat_weights > 0, axis=0))
    lon_nonzero = np.flatnonzero(np.any(lon_weights > 0, axis=0))
    lat_slice = slice(lat_nonzero[0], lat_nonzero[-1] + 1)
    lon_slice = slice(lon_nonzero[0], lon_nonzero[-1] + 1)
    lat_crop = lat_weights[:, lat_slice]
    lon_crop = lon_weights[:, lon_slice]
    return np.stack(
        [lat_crop @ field[lat_slice, lon_slice] @ lon_crop.T for field in fields]
    )


def weighted_metrics(prediction: np.ndarray, truth: np.ndarray, latitude: np.ndarray) -> dict:
    weights = np.cos(np.deg2rad(latitude))[:, None]
    weights = np.broadcast_to(weights, truth.shape)
    difference = prediction - truth
    return {
        "prediction_domain_mean_mm_day": float(np.average(prediction, weights=weights)),
        "era5_domain_mean_mm_day": float(np.average(truth, weights=weights)),
        "prediction_to_era5_ratio": float(
            np.average(prediction, weights=weights) / np.average(truth, weights=weights)
        ),
        "mae_mm_day": float(np.average(np.abs(difference), weights=weights)),
        "rmse_mm_day": float(np.sqrt(np.average(difference**2, weights=weights))),
        "spatial_correlation": float(
            np.corrcoef(prediction.ravel(), truth.ravel())[0, 1]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--init-date", default="2020-06-01")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    product_paths(config, args.init_date)
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise RuntimeError("validation output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("AFNOv2 validation requires CUDA")

    from earth2studio.data import ARCO
    from earth2studio.data.utils import fetch_data
    from earth2studio.models.dx.precipitation_afno_v2 import (
        VARIABLES,
        PrecipitationAFNOv2,
    )

    init = np.datetime64(args.init_date, "ns")
    valid_times = init + np.array([6, 12, 18, 24], dtype="timedelta64[h]")
    variables = np.asarray([*VARIABLES, "tp06"])
    source = ARCO(cache=True, verbose=True, async_timeout=1800)
    print(
        f"fetching ERA5 AFNOv2 validation states times={valid_times.astype(str).tolist()}",
        flush=True,
    )
    combined, coords = fetch_data(
        source=source,
        time=valid_times,
        variable=variables,
        lead_time=np.array([np.timedelta64(0, "h")]),
        device=torch.device("cpu"),
    )
    expected = (4, 1, 21, 721, 1440)
    if tuple(combined.shape) != expected:
        raise ValueError(f"unexpected ERA5 validation input shape {tuple(combined.shape)}")
    if list(coords["variable"]) != variables.tolist():
        raise ValueError("ERA5 validation variable order changed")
    if not torch.isfinite(combined).all():
        raise ValueError("ERA5 validation data contain non-finite values")

    input_state = combined[:, :, :20, :-1, :].to("cuda")
    truth_native = combined[:, :, 20, :-1, :].squeeze(1).numpy()
    input_coords = OrderedDict(
        {
            "time": coords["time"],
            "lead_time": coords["lead_time"],
            "variable": np.asarray(VARIABLES),
            "lat": coords["lat"][:-1],
            "lon": coords["lon"],
        }
    )
    model = PrecipitationAFNOv2.load_model(
        PrecipitationAFNOv2.load_default_package()
    ).to("cuda")
    with torch.inference_mode():
        prediction, output_coords = model(input_state, input_coords)
    if list(output_coords["variable"]) != ["tp06"]:
        raise ValueError("AFNOv2 validation output variable changed")
    prediction_native = prediction.squeeze(1).squeeze(1).float().cpu().numpy()
    if prediction_native.shape != (4, 720, 1440):
        raise ValueError(f"unexpected AFNOv2 output shape {prediction_native.shape}")
    if not np.isfinite(prediction_native).all() or prediction_native.min() < 0:
        raise ValueError("AFNOv2 validation output is invalid")

    target_lat, target_lon = target_coordinates(config)
    lat_weights, lon_weights, weight_hash = conservative_weights(
        np.asarray(input_coords["lat"]),
        np.asarray(input_coords["lon"]),
        target_lat,
        target_lon,
    )
    predicted_tp06 = regrid_fields(
        prediction_native, lat_weights, lon_weights
    ).astype(np.float32) * 1000.0
    era5_tp06 = regrid_fields(
        np.maximum(truth_native, 0.0), lat_weights, lon_weights
    ).astype(np.float32) * 1000.0
    predicted_daily = predicted_tp06.sum(axis=0)
    era5_daily = era5_tp06.sum(axis=0)
    metrics = weighted_metrics(predicted_daily, era5_daily, target_lat)

    dataset = xr.Dataset(
        {
            "afnov2_tp06": (
                ("valid_time", "latitude", "longitude"),
                predicted_tp06,
            ),
            "era5_tp06": (
                ("valid_time", "latitude", "longitude"),
                era5_tp06,
            ),
            "afnov2_tp24": (("latitude", "longitude"), predicted_daily),
            "era5_tp24": (("latitude", "longitude"), era5_daily),
        },
        coords={
            "valid_time": valid_times,
            "latitude": target_lat,
            "longitude": target_lon,
        },
        attrs={
            "purpose": "isolate AFNOv2 precipitation behavior from FCN3",
            "input_source": "exact Google ARCO-ERA5 atmospheric states",
            "surface_pressure_source": "native ERA5 sp",
            "precipitation_package_uri": config["model"]["precipitation_package_uri"],
            "precipitation_regrid_weight_sha256": weight_hash,
        },
    )
    for name in ("afnov2_tp06", "era5_tp06"):
        dataset[name].attrs.update({"units": "mm", "cell_methods": "time: sum (interval: 6 hours)"})
    for name in ("afnov2_tp24", "era5_tp24"):
        dataset[name].attrs.update({"units": "mm day-1", "cell_methods": "time: sum (interval: 24 hours)"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    dataset.to_netcdf(temporary, engine="netcdf4")
    os.replace(temporary, args.output)
    report = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "init_date": args.init_date,
        "valid_times": valid_times.astype(str).tolist(),
        "input_source": "exact Google ARCO-ERA5 atmospheric states",
        "surface_pressure_source": "native ERA5 sp",
        "precipitation_package_uri": config["model"]["precipitation_package_uri"],
        "metrics": metrics,
        "native_afnov2_tp06_minimum_m": float(prediction_native.min()),
        "native_afnov2_tp06_maximum_m": float(prediction_native.max()),
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "precipitation_regrid_weight_sha256": weight_hash,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    write_json_atomic(report, args.output.with_suffix(".json"))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
