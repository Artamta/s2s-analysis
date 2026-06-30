#!/usr/bin/env python
"""Check that representative model fields can be put on the common grid."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from s2s_paper.grid import GridSpec, apply_region, make_grid, to_grid
from s2s_paper.paths import get_paths


FUXI_CHANNELS = {
    "Z500": 5,
    "T2M": 65,
    "TP": 75,
}


@dataclass(frozen=True)
class SampleSpec:
    model: str
    season: str
    variable: str
    path: Path
    kind: str
    data_var: str
    channel: int | str | None = None
    note: str = ""


def _sample_specs() -> list[SampleSpec]:
    paths = get_paths()
    all_model = paths.all_model_data
    return [
        SampleSpec("ECMWF", "JJAS2019", "T2M", all_model / "ecmwf/jjas2019/2t/20190603_pf.nc", "netcdf", "t2m"),
        SampleSpec("ECMWF", "JJAS2019", "TP", all_model / "ecmwf/jjas2019/tp/20190603_pf.nc", "netcdf", "tp"),
        SampleSpec("ECMWF", "JJAS2019", "Z500", all_model / "ecmwf/jjas2019/z/500/20190603_pf.nc", "netcdf", "gh"),
        SampleSpec("UKMO", "JJAS2019", "T2M", all_model / "ukmo/jjas2019/2t/20190603_pf.nc", "netcdf", "t2m"),
        SampleSpec("UKMO", "JJAS2019", "TP", all_model / "ukmo/jjas2019/tp/20190603_pf.nc", "netcdf", "tp"),
        SampleSpec("UKMO", "JJAS2019", "Z500", all_model / "ukmo/jjas2019/z/500/20190603_pf.nc", "netcdf", "gh"),
        SampleSpec(
            "FuXi",
            "JFM2026",
            "T2M",
            all_model / "fuxi/jfm2026/combined/20260101.nc",
            "fuxi_76",
            "forecast",
            FUXI_CHANNELS["T2M"],
        ),
        SampleSpec(
            "FuXi",
            "JFM2026",
            "TP",
            all_model / "fuxi/jfm2026/combined/20260101.nc",
            "fuxi_76",
            "forecast",
            FUXI_CHANNELS["TP"],
        ),
        SampleSpec(
            "FuXi",
            "JFM2026",
            "Z500",
            all_model / "fuxi/jfm2026/combined/20260101.nc",
            "fuxi_76",
            "forecast",
            FUXI_CHANNELS["Z500"],
        ),
        SampleSpec(
            "FuXi",
            "JJAS2019",
            "TP",
            paths.jjas_fuxi_compact / "20190602.nc",
            "fuxi_named_channel",
            "forecast",
            "tp",
        ),
        SampleSpec(
            "FuXi",
            "JJAS2019",
            "Z500",
            paths.jjas_fuxi_compact / "20190602.nc",
            "fuxi_named_channel",
            "forecast",
            "z500",
        ),
        SampleSpec(
            "DELYSM",
            "JJAS2019",
            "T2M",
            all_model / "delysm/jjas2019/20190601/india/forecast.zarr",
            "delysm_zarr",
            "t2m",
            note="DELYSM has no TP in this store; use T2M/Z500 only.",
        ),
        SampleSpec(
            "DELYSM",
            "JJAS2019",
            "Z500",
            all_model / "delysm/jjas2019/20190601/india/forecast.zarr",
            "delysm_zarr",
            "z500",
            note="DELYSM has no TP in this store; use T2M/Z500 only.",
        ),
        SampleSpec(
            "ERA5 truth",
            "JFM2026",
            "T2M",
            all_model / "fuxi/jfm2026/ground_truth/20260101.nc",
            "netcdf",
            "t2m",
        ),
        SampleSpec(
            "ERA5 truth",
            "JFM2026",
            "TP",
            all_model / "fuxi/jfm2026/ground_truth/20260101.nc",
            "netcdf",
            "tp",
        ),
        SampleSpec(
            "ERA5 truth",
            "JFM2026",
            "Z500",
            all_model / "fuxi/jfm2026/ground_truth/20260101.nc",
            "netcdf",
            "z500",
        ),
        SampleSpec(
            "IMD truth",
            "JJAS2019",
            "TP",
            paths.jjas2019_imd_rainfall_nc,
            "netcdf",
            "rain",
        ),
    ]


def _first_existing_dim(da: xr.DataArray, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in da.dims:
            return name
    return None


def _first_field(da: xr.DataArray) -> xr.DataArray:
    indexers = {}
    for dim in da.dims:
        if dim not in ("lat", "lon", "latitude", "longitude"):
            indexers[dim] = 0
    return da.isel(indexers).squeeze()


def _load_netcdf(spec: SampleSpec) -> xr.DataArray:
    with xr.open_dataset(spec.path) as ds:
        if spec.data_var not in ds:
            raise KeyError(f"{spec.data_var!r} not in {list(ds.data_vars)}")
        return _first_field(ds[spec.data_var]).load()


def _load_fuxi_76(spec: SampleSpec) -> xr.DataArray:
    with xr.open_dataset(spec.path) as ds:
        da = ds[spec.data_var].isel(member=0, lead_time=0, channel=int(spec.channel))
        return da.load()


def _load_fuxi_named_channel(spec: SampleSpec) -> xr.DataArray:
    with xr.open_dataset(spec.path) as ds:
        channel_values = [str(v) for v in ds["channel"].values]
        if str(spec.channel) not in channel_values:
            raise KeyError(f"channel {spec.channel!r} not in {channel_values}")
        channel_index = channel_values.index(str(spec.channel))
        da = ds[spec.data_var].isel(member=0, lead_time=0, channel=channel_index)
        return da.load()


def _load_delysm_zarr(spec: SampleSpec) -> xr.DataArray:
    group = zarr.open_group(str(spec.path), mode="r")
    lat = np.asarray(group["lat"][:], dtype=float)
    lon = np.asarray(group["lon"][:], dtype=float)
    lead = np.asarray(group["lead_time"][:])
    lead_index = int(np.where(lead >= np.timedelta64(0, "ns"))[0][0])
    values = np.asarray(group[spec.data_var][0, 0, lead_index, :, :])
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name=spec.data_var)


def load_sample(spec: SampleSpec) -> xr.DataArray:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    if spec.kind == "netcdf":
        return _load_netcdf(spec)
    if spec.kind == "fuxi_76":
        return _load_fuxi_76(spec)
    if spec.kind == "fuxi_named_channel":
        return _load_fuxi_named_channel(spec)
    if spec.kind == "delysm_zarr":
        return _load_delysm_zarr(spec)
    raise ValueError(f"unknown sample kind {spec.kind!r}")


def check_one(spec: SampleSpec, grid: GridSpec, expected_shape: tuple[int, int]) -> dict[str, object]:
    row: dict[str, object] = {
        "model": spec.model,
        "season": spec.season,
        "variable": spec.variable,
        "kind": spec.kind,
        "path": str(spec.path),
        "status": "FAIL",
        "source_shape": "",
        "grid_shape": "",
        "finite_fraction": "",
        "region_finite_fraction": "",
        "note": spec.note,
        "error": "",
    }
    try:
        src = load_sample(spec)
        gridded = to_grid(src, grid)
        if gridded.shape[-2:] != expected_shape:
            raise ValueError(f"grid shape {gridded.shape[-2:]} != expected {expected_shape}")
        if tuple(gridded.dims[-2:]) != ("lat", "lon"):
            raise ValueError(f"last dims are {gridded.dims[-2:]}, expected ('lat', 'lon')")
        masked = apply_region(gridded, "All India", dgrid=grid.dgrid)
        finite = float(np.isfinite(gridded).sum() / gridded.size)
        region_finite = float(np.isfinite(masked).sum() / np.isfinite(masked.where(np.isfinite(masked))).size)
        if int(np.isfinite(masked).sum()) == 0:
            raise ValueError("All-India mask has zero finite cells after regridding")
        row.update(
            {
                "status": "OK",
                "source_shape": "x".join(str(x) for x in src.shape),
                "grid_shape": "x".join(str(x) for x in gridded.shape[-2:]),
                "finite_fraction": f"{finite:.3f}",
                "region_finite_fraction": f"{region_finite:.3f}",
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "season",
        "variable",
        "kind",
        "status",
        "source_shape",
        "grid_shape",
        "finite_fraction",
        "region_finite_fraction",
        "note",
        "error",
        "path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgrid", type=float, default=1.5, help="target grid spacing in degrees")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "outputs" / "common_grid_check.csv",
        help="CSV path for detailed check output",
    )
    args = parser.parse_args()

    grid = GridSpec(dgrid=args.dgrid)
    lat, lon = make_grid(grid)
    expected_shape = (len(lat), len(lon))
    rows = [check_one(spec, grid, expected_shape) for spec in _sample_specs()]
    write_csv(rows, args.csv)

    print(f"common grid: {expected_shape[0]}x{expected_shape[1]} at {args.dgrid:g} deg")
    print(f"wrote: {args.csv}")
    for row in rows:
        prefix = "OK  " if row["status"] == "OK" else "FAIL"
        print(
            f"{prefix} {row['model']:10s} {row['season']:8s} {row['variable']:5s} "
            f"source={row['source_shape'] or '-':>12s} grid={row['grid_shape'] or '-':>7s}"
        )
        if row["error"]:
            print(f"     {row['error']}")

    return 0 if all(row["status"] == "OK" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
