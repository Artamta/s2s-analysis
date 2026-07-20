#!/usr/bin/env python3
"""Build All-India daily rainfall verification from official IMERG and IMD grids."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


MASK_FILE = Path("/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/imd_region_masks.nc")
REGIONS = ("northwest_india", "central_india", "south_peninsula", "east_northeast_india")
IMERG_BASE = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDL.07"
IMD_GAUGE_URL = "https://imdpune.gov.in/cmpg/Realtimedata/Rainfall/rain.php"
IMD_MERGED_URL = "https://imdpune.gov.in/cmpg/Realtimedata/gpm/rain.php"


def dates_between(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def union_mask(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    with xr.open_dataset(MASK_FILE) as source:
        source = source.sortby("lat").sortby("lon")
        mask = xr.zeros_like(source[REGIONS[0]], dtype=bool)
        for region in REGIONS:
            mask = mask | (source[region] > 0)
        return (mask.astype(float).interp(lat=lat, lon=lon, method="nearest") >= 0.5).load()


def area_mean(field: xr.DataArray) -> float:
    field = field.sortby("lat").sortby("lon")
    mask = union_mask(field.lat, field.lon)
    weights = xr.DataArray(np.cos(np.deg2rad(field.lat)), coords={"lat": field.lat}, dims="lat")
    return float(field.where(mask).weighted(weights).mean(("lat", "lon"), skipna=True))


def imd_grid(url: str, day: date, shape: tuple[int, int], lat0: float, lon0: float) -> xr.DataArray:
    response = requests.post(url, data={"rain": day.strftime("%d%m%Y")}, timeout=90)
    response.raise_for_status()
    expected = int(np.prod(shape)) * 4
    if len(response.content) != expected:
        raise RuntimeError(f"Unexpected IMD response for {day}: {len(response.content)} bytes, expected {expected}")
    values = np.frombuffer(response.content, dtype="<f4").reshape(shape).astype("float64")
    values[values <= -900] = np.nan
    lat = lat0 + 0.25 * np.arange(shape[0])
    lon = lon0 + 0.25 * np.arange(shape[1])
    return xr.DataArray(values, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def imerg_url(day: date) -> str:
    filename = f"3B-DAY-L.MS.MRG.3IMERG.{day:%Y%m%d}-S000000-E235959.V07C.nc4"
    return f"{IMERG_BASE}/{day:%Y}/{day:%m}/{filename}"


def imerg_grid(day: date, temp_dir: Path) -> xr.DataArray:
    target = temp_dir / f"imerg_{day:%Y%m%d}.nc4"
    cookie = Path.home() / ".urs_cookies"
    command = [
        "curl", "-sS", "--fail", "-n", "-b", str(cookie), "-c", str(cookie),
        "-L", "--retry", "3", "--retry-delay", "5", "-o", str(target), imerg_url(day),
    ]
    subprocess.run(command, check=True)
    with xr.open_dataset(target) as source:
        field = (
            source["precipitation"]
            .sel(lon=slice(60, 100), lat=slice(0, 40))
            .squeeze(drop=True)
            .transpose("lat", "lon")
            .load()
        )
    target.unlink()
    field = field.where(np.isfinite(field) & (field >= 0))
    return field


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory(prefix="rain_gt_") as temp_name:
        temp_dir = Path(temp_name)
        for day in dates_between(args.start, args.end):
            row = {"date": day.isoformat()}
            products = (
                ("imerg_late_mm", lambda: imerg_grid(day, temp_dir)),
                ("imd_gauge_mm", lambda: imd_grid(IMD_GAUGE_URL, day, (129, 135), 6.5, 66.5)),
                ("imd_gauge_gpm_merged_mm", lambda: imd_grid(IMD_MERGED_URL, day, (281, 241), -30.0, 50.0)),
            )
            for name, loader in products:
                try:
                    row[name] = area_mean(loader())
                except Exception as exc:  # keep a partial, auditable series if a provider day is absent
                    print(f"WARN {day} {name}: {exc}", flush=True)
                    row[name] = np.nan
            rows.append(row)
            print(
                f"DONE {day} IMERG={row['imerg_late_mm']:.3f} "
                f"IMD={row['imd_gauge_mm']:.3f} IMD+GPM={row['imd_gauge_gpm_merged_mm']:.3f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    provenance = {
        "coverage": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "aggregation": "cosine-latitude weighted mean over union of four IMD homogeneous regions",
        "imerg": {
            "collection": "GPM_3IMERGDL.07 (IMERG Late, daily)",
            "variable": "precipitation (mm/day)",
            "base_url": IMERG_BASE,
            "doi": "10.5067/GPM/IMERGDL/DAY/07",
        },
        "imd_gauge": {
            "product": "IMD real-time daily rainfall, 0.25 degree binary",
            "endpoint": IMD_GAUGE_URL,
            "grid": "135x129; 66.5E-100E, 6.5N-38.5N",
        },
        "imd_gauge_gpm_merged": {
            "product": "IMD daily merged satellite-gauge rainfall (GPM), 0.25 degree binary",
            "endpoint": IMD_MERGED_URL,
            "grid": "241x281; 50E-110E, 30S-40N",
        },
        "mask": str(MASK_FILE),
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(frame.count().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
