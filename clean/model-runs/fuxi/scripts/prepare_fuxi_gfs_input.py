#!/usr/bin/env python3
"""Build an experimental FuXi-S2S daily input from operational GFS fields.

The state variables are means of the 00/06/12/18 UTC GFS analyses. Total
precipitation and top thermal radiation are constructed from the four
successive 0-6 hour forecasts, which cover each complete UTC day. This is a
same-day operational proxy for the ERA5 hourly daily means used to train
FuXi-S2S; it is deliberately identified as GFS-initialized in all metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pygrib
import xarray as xr

import temporal_contract


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "clean/config/fuxi_gfs_case_20260728.json"
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PRESSURE_FIELDS = (
    ("z", "gh"),
    ("t", "t"),
    ("u", "u"),
    ("v", "v"),
    ("q", "q"),
)
SURFACE_CHANNELS = (
    "t2m",
    "d2m",
    "sst",
    "ttr",
    "10u",
    "10v",
    "100u",
    "100v",
    "msl",
    "tcwv",
    "tp",
)
CYCLES = (0, 6, 12, 18)
GRAVITY = np.float32(9.80665)


def expected_channels() -> list[str]:
    channels: list[str] = []
    for prefix, _ in PRESSURE_FIELDS:
        channels.extend(f"{prefix}{level}" for level in LEVELS)
    channels.extend(SURFACE_CHANNELS)
    return channels


def expected_latitudes() -> np.ndarray:
    return np.linspace(90.0, -90.0, 121, dtype=np.float32)


def expected_longitudes() -> np.ndarray:
    return np.arange(0.0, 360.0, 1.5, dtype=np.float32)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_plan(
    issue_date: pd.Timestamp, config: dict[str, Any]
) -> list[dict[str, Any]]:
    gfs = config["input"]["gfs"]
    template = str(gfs["url_template"])
    raw_root = Path(
        str(gfs["raw_root"]).format(issue_date=issue_date.strftime("%Y%m%d"))
    )
    records = []
    for day in temporal_contract.input_days(issue_date, config):
        for cycle in CYCLES:
            for forecast_hour in (0, 6):
                values = {
                    "date": day.strftime("%Y%m%d"),
                    "cycle": f"{cycle:02d}",
                    "forecast_hour": f"{forecast_hour:03d}",
                }
                filename = (
                    f"gfs.t{cycle:02d}z.pgrb2."
                    f"{gfs['grid_label']}.f{forecast_hour:03d}"
                )
                records.append(
                    {
                        "day": day,
                        "cycle": cycle,
                        "forecast_hour": forecast_hour,
                        "url": template.format(**values),
                        "path": raw_root / values["date"] / filename,
                    }
                )
    return records


def download(url: str, path: Path) -> None:
    minimum_size = 10_000_000
    if path.is_file() and path.stat().st_size >= minimum_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "FuXi-S2S-GFS-adapter/1"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    if temporary.stat().st_size < minimum_size:
        raise ValueError(f"download is unexpectedly small: {temporary}")
    temporary.replace(path)


def message_catalog(gribs: pygrib.open) -> dict[tuple[str, str, int, str], list[Any]]:
    catalog: dict[tuple[str, str, int, str], list[Any]] = {}
    for message in gribs:
        key = (
            str(message.shortName),
            str(message.typeOfLevel),
            int(message.level),
            str(message.stepType),
        )
        catalog.setdefault(key, []).append(message)
    return catalog


def select_one(
    catalog: dict[tuple[str, str, int, str], list[Any]],
    *,
    shortName: str,
    typeOfLevel: str,
    level: int = 0,
    stepType: str = "instant",
):
    criteria = {
        "shortName": shortName,
        "typeOfLevel": typeOfLevel,
        "level": level,
        "stepType": stepType,
    }
    matches = catalog.get((shortName, typeOfLevel, level, stepType), [])
    if not matches:
        raise KeyError(f"no GFS field matches {criteria}")
    if len(matches) > 1:
        first = np.ma.filled(matches[0].values, np.nan)
        if not all(
            np.array_equal(first, np.ma.filled(candidate.values, np.nan), equal_nan=True)
            for candidate in matches[1:]
        ):
            raise ValueError(f"ambiguous non-identical GFS fields for {criteria}")
    return matches[0]


def values(message: Any) -> np.ndarray:
    array = np.ma.filled(message.values, np.nan).astype(np.float32, copy=False)
    if array.shape != (181, 360):
        raise ValueError(f"expected 1-degree GFS grid, found {array.shape}")
    return array


def open_cycle(path: Path) -> pygrib.open:
    if not path.is_file() or path.stat().st_size < 10_000_000:
        raise FileNotFoundError(path)
    return pygrib.open(str(path))


def pressure_snapshot(
    catalog: dict[tuple[str, str, int, str], list[Any]]
) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for prefix, short_name in PRESSURE_FIELDS:
        for level in LEVELS:
            message = select_one(
                catalog,
                shortName=short_name,
                typeOfLevel="isobaricInhPa",
                level=level,
            )
            field = values(message)
            if prefix == "z":
                field = field * GRAVITY
            fields[f"{prefix}{level}"] = field
    return fields


def surface_snapshot(
    catalog: dict[tuple[str, str, int, str], list[Any]]
) -> dict[str, np.ndarray]:
    selectors = {
        "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
        "d2m": {"shortName": "2d", "typeOfLevel": "heightAboveGround", "level": 2},
        "sst": {"shortName": "t", "typeOfLevel": "surface", "level": 0},
        "10u": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
        "10v": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
        "100u": {"shortName": "100u", "typeOfLevel": "heightAboveGround", "level": 100},
        "100v": {"shortName": "100v", "typeOfLevel": "heightAboveGround", "level": 100},
        "msl": {"shortName": "prmsl", "typeOfLevel": "meanSea"},
        "tcwv": {"shortName": "pwat", "typeOfLevel": "atmosphereSingleLayer"},
        "land_mask": {"shortName": "lsm", "typeOfLevel": "surface"},
    }
    return {
        name: values(select_one(catalog, **criteria))
        for name, criteria in selectors.items()
    }


def flux_snapshot(
    catalog: dict[tuple[str, str, int, str], list[Any]]
) -> dict[str, np.ndarray]:
    thermal = values(
        select_one(
            catalog,
            shortName="sulwrf",
            typeOfLevel="nominalTop",
            stepType="avg",
        )
    )
    precipitation = values(
        select_one(
            catalog,
            shortName="tp",
            typeOfLevel="surface",
            stepType="accum",
        )
    )
    # ERA5/FuXi TTR is positive downward; GFS ULWRF is positive upward.
    return {"ttr": -thermal, "tp_accumulation": precipitation}


def regrid(field: np.ndarray) -> np.ndarray:
    source = xr.DataArray(
        field,
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(90.0, -90.0, 181, dtype=np.float32),
            "lon": np.arange(360, dtype=np.float32),
        },
    )
    target = source.interp(
        lat=expected_latitudes(),
        lon=expected_longitudes(),
        method="linear",
    )
    return target.values.astype(np.float32, copy=False)


def build_day(day: pd.Timestamp, records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    daily_records = [record for record in records if record["day"] == day]
    analyses = {
        record["cycle"]: record["path"]
        for record in daily_records
        if record["forecast_hour"] == 0
    }
    forecasts = {
        record["cycle"]: record["path"]
        for record in daily_records
        if record["forecast_hour"] == 6
    }
    if set(analyses) != set(CYCLES) or set(forecasts) != set(CYCLES):
        raise ValueError(f"incomplete GFS cycle plan for {day:%Y-%m-%d}")

    samples: dict[str, list[np.ndarray]] = {
        channel: [] for channel in expected_channels() if channel not in {"ttr", "tp"}
    }
    land_masks: list[np.ndarray] = []
    for cycle in CYCLES:
        gribs = open_cycle(analyses[cycle])
        try:
            catalog = message_catalog(gribs)
            state = pressure_snapshot(catalog)
            state.update(surface_snapshot(catalog))
        finally:
            gribs.close()
        land_masks.append(state.pop("land_mask"))
        for channel, field in state.items():
            samples[channel].append(field)

    day_fields = {
        channel: regrid(np.nanmean(np.stack(channel_samples), axis=0))
        for channel, channel_samples in samples.items()
    }
    ocean = np.nanmean(np.stack(land_masks), axis=0) < 0.5
    sst_source = np.nanmean(np.stack(samples["sst"]), axis=0)
    day_fields["sst"] = regrid(np.where(ocean, sst_source, np.nan))

    fluxes = []
    for cycle in CYCLES:
        gribs = open_cycle(forecasts[cycle])
        try:
            fluxes.append(flux_snapshot(message_catalog(gribs)))
        finally:
            gribs.close()
    day_fields["ttr"] = regrid(
        np.nanmean(np.stack([sample["ttr"] for sample in fluxes]), axis=0)
    )
    # Four 6-hour accumulations form the complete daily total. FuXi TP is the
    # mean hourly rate in mm h-1, numerically kg m-2 h-1.
    day_fields["tp"] = regrid(
        np.nansum(
            np.stack([sample["tp_accumulation"] for sample in fluxes]), axis=0
        )
        / np.float32(24.0)
    )
    return day_fields


def build_input(
    issue_date: pd.Timestamp, config: dict[str, Any], allow_download: bool = True
) -> xr.DataArray:
    plan = source_plan(issue_date, config)
    if allow_download:
        for index, record in enumerate(plan, start=1):
            print(
                f"GFS file {index}/{len(plan)} {record['day']:%Y-%m-%d} "
                f"{record['cycle']:02d}Z f{record['forecast_hour']:03d}",
                flush=True,
            )
            download(record["url"], record["path"])

    days = temporal_contract.input_days(issue_date, config)
    daily = [build_day(day, plan) for day in days]
    channels = expected_channels()
    tensor = np.stack(
        [
            np.stack([daily_day[channel] for channel in channels], axis=0)
            for daily_day in daily
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    timing = temporal_contract.provenance(issue_date, config)
    latest_analysis = days[-1] + pd.Timedelta(hours=18)
    return xr.DataArray(
        tensor,
        dims=("time", "channel", "lat", "lon"),
        coords={
            "time": days.values,
            "channel": channels,
            "lat": expected_latitudes(),
            "lon": expected_longitudes(),
        },
        name="data",
        attrs={
            "source": config["input"]["source"],
            "source_class": "operational GFS analysis/short-forecast proxy for ERA5",
            "issue_date": issue_date.strftime("%Y-%m-%d"),
            "forecast_reference_time": timing["forecast_reference_time"],
            "model_state_time": timing["model_state_time"],
            "information_cutoff_time": timing["information_cutoff_time"],
            "latest_assimilated_analysis_time": latest_analysis.isoformat(),
            "information_cutoff_matches_issue_time": str(
                timing["information_cutoff_matches_issue_time"]
            ).lower(),
            "benchmark_mode": timing["benchmark_mode"],
            "temporal_statistic": config["input"]["temporal_statistic"],
            "source_days": f"{days[0]:%Y-%m-%d},{days[1]:%Y-%m-%d}",
            "state_preprocessing": "mean of GFS f000 analyses at 00/06/12/18 UTC",
            "tp_preprocessing": "sum of four GFS 0-6 h APCP fields divided by 24; mm h-1",
            "ttr_preprocessing": "negative of mean of four GFS 0-6 h top ULWRF fields; W m-2",
            "geopotential_preprocessing": "GFS geopotential height multiplied by 9.80665 m s-2",
            "sst_preprocessing": "GFS surface temperature retained over GFS ocean mask",
            "gfs_grid": config["input"]["gfs"]["grid_label"],
            "gfs_files": ",".join(str(record["path"]) for record in plan),
            "gfs_file_sha256": json.dumps(
                {str(record["path"]): sha256(record["path"]) for record in plan},
                sort_keys=True,
            ),
            "scientific_status": (
                "experimental GFS-initialized FuXi-S2S; checkpoint was trained "
                "on ERA5 daily statistics and this input has not been hindcast-calibrated"
            ),
        },
    )


def validate_input(
    path: Path, issue_date: pd.Timestamp, config: dict[str, Any]
) -> dict[str, Any]:
    data = xr.open_dataarray(path)
    try:
        if data.dims != ("time", "channel", "lat", "lon"):
            raise ValueError(f"unexpected input dimensions: {data.dims}")
        if data.shape != (2, 76, 121, 240):
            raise ValueError(f"unexpected input shape: {data.shape}")
        if data.channel.values.astype(str).tolist() != expected_channels():
            raise ValueError("FuXi GFS input channel order is incorrect")
        if not np.allclose(data.lat.values, expected_latitudes()):
            raise ValueError("FuXi GFS latitude grid is incorrect")
        if not np.allclose(data.lon.values, expected_longitudes()):
            raise ValueError("FuXi GFS longitude grid is incorrect")
        expected_times = temporal_contract.input_days(issue_date, config).values
        if not np.array_equal(data.time.values.astype("datetime64[ns]"), expected_times):
            raise ValueError(f"unexpected input times: {data.time.values}")
        if data.attrs.get("temporal_statistic") != config["input"]["temporal_statistic"]:
            raise ValueError("GFS daily-statistic declaration is missing or incorrect")
        if data.attrs.get("source_class") != (
            "operational GFS analysis/short-forecast proxy for ERA5"
        ):
            raise ValueError("input does not declare its GFS proxy source class")
        for channel in expected_channels():
            field = data.sel(channel=channel).values
            if channel == "sst":
                if np.isfinite(field).mean() < 0.5:
                    raise ValueError("SST has insufficient finite ocean coverage")
            elif not np.isfinite(field).all():
                raise ValueError(f"{channel} contains missing or infinite values")
        ranges = {
            "z1000": (-15_000.0, 15_000.0),
            "t2m": (180.0, 350.0),
            "d2m": (170.0, 340.0),
            "ttr": (-500.0, 0.0),
            "msl": (85_000.0, 110_000.0),
            "tcwv": (0.0, 100.0),
            "tp": (0.0, 20.0),
        }
        statistics = {}
        for channel, (lower, upper) in ranges.items():
            field = data.sel(channel=channel).values
            minimum = float(np.nanmin(field))
            maximum = float(np.nanmax(field))
            if minimum < lower or maximum > upper:
                raise ValueError(
                    f"{channel} outside proxy QC range {lower}..{upper}: "
                    f"{minimum}..{maximum}"
                )
            statistics[channel] = {"minimum": minimum, "maximum": maximum}
        return {
            "shape": list(data.shape),
            "time_start": str(pd.Timestamp(data.time.values[0])),
            "time_end": str(pd.Timestamp(data.time.values[-1])),
            "temporal_statistic": data.attrs["temporal_statistic"],
            "source_class": data.attrs["source_class"],
            "selected_ranges": statistics,
            "size_bytes": path.stat().st_size,
        }
    finally:
        data.close()


def quarantine(path: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.invalid.{timestamp}")
    path.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="initialization date YYYYMMDD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    issue_date = pd.Timestamp(args.date)
    if issue_date.strftime("%Y%m%d") != args.date:
        raise SystemExit("date must use YYYYMMDD format")
    config = load_config(args.config)

    if args.output.exists():
        try:
            details = validate_input(args.output, issue_date, config)
            print(f"existing GFS input valid: {args.output} {details}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            moved = quarantine(args.output)
            print(f"quarantined invalid GFS input at {moved}: {exc}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    data = build_input(issue_date, config, allow_download=not args.no_download)
    encoding = {"data": {"zlib": True, "complevel": 2, "dtype": "float32"}}
    data.to_netcdf(temporary, encoding=encoding)
    details = validate_input(temporary, issue_date, config)
    temporary.replace(args.output)
    print(f"wrote valid GFS FuXi input: {args.output} {details}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
