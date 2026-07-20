#!/usr/bin/env python3
"""Retrieve only ECMWF S2S total precipitation for one initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi


STEPS = [str(hour) for hour in range(24, 46 * 24 + 1, 24)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260601")
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    day = args.date
    args.out_root.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True)
    for short, forecast_type in (("cf", "control_forecast"), ("pf", "perturbed_forecast")):
        target = args.out_root / f"{day}_{short}.nc"
        if target.exists() and target.stat().st_size > 0:
            print(f"SKIP {target}", flush=True)
            continue
        request = {
            "origin": "ecmwf",
            "forecast_type": forecast_type,
            "year": day[:4],
            "month": day[4:6],
            "day": day[6:8],
            "time": "00:00:00",
            "step": STEPS,
            "area": [40, 60, 0, 100],
            "grid": [1.5, 1.5],
            "data_format": "netcdf",
            "level_type": "single_level",
            "variable": "tp",
        }
        temporary = target.with_suffix(".nc.tmp")
        print(f"START {forecast_type} -> {target}", flush=True)
        client.retrieve("s2s-forecasts", request, str(temporary))
        temporary.replace(target)
        print(f"DONE {target} {target.stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
