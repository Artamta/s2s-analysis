from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "clean/config/fuxi_operational_2020_2025.json"


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module(
    "prepare_fuxi_input",
    "clean/model-runs/fuxi/scripts/prepare_fuxi_input.py",
)
runner = load_module(
    "run_fuxi_forecast",
    "clean/model-runs/fuxi/scripts/run_fuxi_forecast.py",
)
stage = load_module(
    "stage_era5_daily",
    "clean/model-runs/fuxi/scripts/stage_era5_daily.py",
)


def test_all_calendar_rows_map_to_unique_outputs():
    config = runner.load_config(CONFIG_PATH)
    dates = runner.load_dates(config)
    outputs = [runner.paths_for(config, date)["output"] for date in dates]
    assert len(dates) == 621
    assert len(set(outputs)) == 621
    assert dates[0] == pd.Timestamp("2020-01-02")
    assert dates[-1] == pd.Timestamp("2025-12-29")


def test_monthly_requests_are_complete_unique_and_within_cds_limit():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seen = set()
    covered = []
    request_count = 0
    for year in config["years"]:
        for month in range(1, 13):
            specs = stage.build_requests(config, year, month)
            month_days = stage.required_month_dates(config, year, month)
            assert 3 <= len(specs) <= 4
            for spec in specs:
                request = spec.request
                assert request["daily_statistic"] == "daily_mean"
                assert request["frequency"] == "1_hourly"
                assert request["time_zone"] == "utc+00:00"
                assert request["grid"] == "1.5/1.5"
                assert spec.cost <= config["input"]["request_cost_limit"]
                assert spec.digest not in seen
                seen.add(spec.digest)
                request_count += 1
            pressure_days = sorted(
                day
                for spec in specs
                if spec.component == "pressure"
                for day in spec.days
            )
            surface_days = sorted(
                day
                for spec in specs
                if spec.component == "surface"
                for day in spec.days
            )
            assert pressure_days == list(month_days)
            assert surface_days == list(month_days)
            covered.extend(month_days)
    assert request_count == 287
    assert len(seen) == request_count
    assert pd.DatetimeIndex(sorted(covered)).equals(stage.required_dates(config))
    assert len(covered) == 1242


def test_daily_staging_builds_checkpoint_tensor(tmp_path: Path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["storage_root"] = str(tmp_path)
    days = pd.date_range("2020-01-01", periods=2, freq="D")
    lat = prepare.expected_latitudes()
    lon = prepare.expected_longitudes()
    level = np.asarray(prepare.LEVELS)
    staging = prepare.staging_paths(config, pd.Timestamp("2020-01-02"))
    for path in staging.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    pressure_shape = (2, len(level), len(lat), len(lon))
    pressure = xr.Dataset(
        {
            "z": (
                ("time", "level", "lat", "lon"),
                np.full(pressure_shape, 5000.0, np.float32),
            ),
            "t": (
                ("time", "level", "lat", "lon"),
                np.full(pressure_shape, 270.0, np.float32),
            ),
            "u": (
                ("time", "level", "lat", "lon"),
                np.full(pressure_shape, 5.0, np.float32),
            ),
            "v": (
                ("time", "level", "lat", "lon"),
                np.full(pressure_shape, -2.0, np.float32),
            ),
            "q": (
                ("time", "level", "lat", "lon"),
                np.full(pressure_shape, 0.002, np.float32),
            ),
        },
        coords={"time": days, "level": level, "lat": lat, "lon": lon},
        attrs={"request_set_sha256": "pressure-test"},
    )
    pressure.to_netcdf(staging["pressure"])

    surface_shape = (2, len(lat), len(lon))
    instant = xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), np.full(surface_shape, 290.0, np.float32)),
            "d2m": (("time", "lat", "lon"), np.full(surface_shape, 285.0, np.float32)),
            "sst": (("time", "lat", "lon"), np.full(surface_shape, 300.0, np.float32)),
            "10u": (("time", "lat", "lon"), np.full(surface_shape, 3.0, np.float32)),
            "10v": (("time", "lat", "lon"), np.full(surface_shape, 1.0, np.float32)),
            "100u": (("time", "lat", "lon"), np.full(surface_shape, 4.0, np.float32)),
            "100v": (("time", "lat", "lon"), np.full(surface_shape, 2.0, np.float32)),
            "msl": (
                ("time", "lat", "lon"),
                np.full(surface_shape, 101000.0, np.float32),
            ),
            "tcwv": (("time", "lat", "lon"), np.full(surface_shape, 25.0, np.float32)),
            "ttr": (
                ("time", "lat", "lon"),
                np.full(surface_shape, -864000.0, np.float32),
                {"units": "J m-2"},
            ),
            "tp": (
                ("time", "lat", "lon"),
                np.full(surface_shape, 0.001, np.float32),
                {"units": "m"},
            ),
        },
        coords={"time": days, "lat": lat, "lon": lon},
        attrs={"request_set_sha256": "surface-test"},
    )
    instant.to_netcdf(staging["surface"])

    data = prepare.build_input(pd.Timestamp("2020-01-02"), config)
    assert data.shape == (2, 76, 121, 240)
    assert data.channel.values.tolist() == prepare.expected_channels()
    assert (
        data.attrs["temporal_statistic"] == "UTC daily mean derived from 1-hourly ERA5"
    )
    assert "pressure-test" in data.attrs["staging_request_sets"]
    assert "surface-test" in data.attrs["staging_request_sets"]
    assert np.allclose(data.sel(channel="tp"), 1.0)
    assert np.allclose(data.sel(channel="ttr"), -240.0)

    output = tmp_path / "input.nc"
    data.to_netcdf(output)
    details = prepare.validate_input(output, pd.Timestamp("2020-01-02"))
    assert details["shape"] == [2, 76, 121, 240]


def test_official_sample_round_trips_through_monthly_staging(tmp_path: Path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["storage_root"] = str(tmp_path)
    days = (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-02"))
    sample_root = (
        REPO_ROOT / "analysis-code/data-download/fuxi_s2s/FuXi-S2S/data/sample"
    )

    for component, fields in stage.FIELDS.items():
        dataset_name, request, cost = stage.build_request(config, component, days)
        spec = stage.RequestSpec(
            name=f"{component}_00",
            component=component,
            dataset=dataset_name,
            days=days,
            request=request,
            digest=stage.request_hash(dataset_name, request),
            cost=cost,
        )
        paths = stage.paths_for(config, 2020, 6, spec)
        paths["download"].parent.mkdir(parents=True, exist_ok=True)
        opened = [
            xr.open_dataset(sample_root / f"{long_name}.nc") for _, long_name in fields
        ]
        try:
            merged = xr.merge(opened, compat="override")
            if component == "surface":
                merged = merged.isel(level=0, drop=True)
            merged.to_netcdf(paths["download"])
        finally:
            for dataset in opened:
                dataset.close()
        stage.normalize_download(spec, paths)
        stage.combine_component(config, 2020, 6, component, [spec])

    actual = prepare.build_input(pd.Timestamp("2020-06-02"), config)
    expected = xr.open_dataarray(
        REPO_ROOT / "analysis-code/data-download/fuxi_s2s/FuXi-S2S/data/input.nc"
    )
    try:
        assert actual.channel.values.tolist() == expected.channel.values.tolist()
        assert np.allclose(actual.values, expected.values, equal_nan=True)
    finally:
        expected.close()
