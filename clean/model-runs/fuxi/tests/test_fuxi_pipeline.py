from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "clean/config/fuxi_operational_2020_2025.json"
STRICT_CONFIG_PATH = REPO_ROOT / "clean/config/fuxi_strict00z_2020_2025.json"
SCRIPTS = REPO_ROOT / "clean/model-runs/fuxi/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


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
arco_hourly = load_module(
    "arco_hourly",
    "clean/model-runs/fuxi/scripts/arco_hourly.py",
)
arco_stage = load_module(
    "stage_arco_era5_daily",
    "clean/model-runs/fuxi/scripts/stage_arco_era5_daily.py",
)
arco_remote_stage = load_module(
    "stage_arco_hourly_daily",
    "clean/model-runs/fuxi/scripts/stage_arco_hourly_daily.py",
)


def test_all_calendar_rows_map_to_unique_outputs():
    config = runner.load_config(CONFIG_PATH)
    dates = runner.load_dates(config)
    outputs = [runner.paths_for(config, date)["output"] for date in dates]
    assert len(dates) == 621
    assert len(set(outputs)) == 621
    assert dates[0] == pd.Timestamp("2020-01-02")
    assert dates[-1] == pd.Timestamp("2025-12-29")


def test_strict_alignment_uses_only_information_available_at_issue_time():
    config = runner.load_config(STRICT_CONFIG_PATH)
    issue = pd.Timestamp("2020-01-02")
    timing = runner.temporal_contract.provenance(issue, config)
    assert timing["strict_operational"] is True
    assert timing["input_days"] == ["2019-12-31", "2020-01-01"]
    assert timing["model_state_time"] == "2020-01-01T00:00:00"
    assert timing["information_cutoff_time"] == "2020-01-02T00:00:00"
    assert timing["information_cutoff_matches_issue_time"] is True

    start, end, valid = runner.temporal_contract.forecast_periods(issue, 42, config)
    assert pd.Timestamp(start[0]) == issue
    assert pd.Timestamp(end[0]) == pd.Timestamp("2020-01-03")
    assert pd.Timestamp(valid[0]) == pd.Timestamp("2020-01-03")
    assert pd.Timestamp(start[-1]) == pd.Timestamp("2020-02-12")
    assert pd.Timestamp(end[-1]) == pd.Timestamp("2020-02-13")


def test_published_alignment_is_rejected_as_strict_00utc():
    config = runner.load_config(CONFIG_PATH)
    timing = runner.temporal_contract.provenance(pd.Timestamp("2020-01-02"), config)
    assert timing["strict_operational"] is False
    assert timing["input_days"] == ["2020-01-01", "2020-01-02"]
    assert timing["information_cutoff_time"] == "2020-01-03T00:00:00"
    assert timing["information_cutoff_matches_issue_time"] is False


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


def test_direct_arco_plan_has_no_cds_requests():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    digests = set()
    for year in config["input"]["local_daily_arco"]["years"]:
        for month in range(1, 13):
            days = stage.required_month_dates(config, year, month)
            gap = arco_stage.gap_digests(
                config["input"]["remote_hourly_arco"]["zarr"], days
            )
            assert set(gap) == {"ttr", "100u", "100v"}
            assert not digests.intersection(gap.values())
            digests.update(gap.values())
    assert len(digests) == 108

    for year in config["input"]["remote_hourly_arco"]["years"]:
        for month in range(1, 13):
            days = stage.required_month_dates(config, year, month)
            for component in arco_remote_stage.FIELDS:
                fields = arco_remote_stage.field_digests(
                    config["input"]["remote_hourly_arco"]["zarr"],
                    component,
                    days,
                )
                assert set(fields) == set(arco_remote_stage.FIELDS[component])
                assert not digests.intersection(fields.values())
                digests.update(fields.values())
    assert len(digests) == 108 + 36 * 16


def test_local_arco_matches_official_fuxi_sample():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = Path(config["input"]["local_daily_arco"]["zarr"])
    if not source.is_dir():
        pytest.skip("cluster-local daily ARCO archive is unavailable")
    days = (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-02"))
    reader = arco_stage.ConsolidatedZarrReader(source)
    time_index = arco_stage.source_time_index(reader)
    lat, lon, level_indices = arco_stage.validate_source_grid(reader)
    pressure = arco_stage.build_local_pressure(
        reader,
        days,
        time_index,
        lat,
        lon,
        level_indices,
        arco_stage.source_digest(reader, "pressure", days),
    )
    surface = arco_stage.build_local_surface(
        reader,
        days,
        time_index,
        lat,
        lon,
        arco_stage.source_digest(reader, "surface", days),
    )
    expected = xr.open_dataarray(
        REPO_ROOT / "analysis-code/data-download/fuxi_s2s/FuXi-S2S/data/input.nc"
    )
    try:
        limits = {"z": 0.12, "t": 0.001, "u": 0.001, "v": 0.001, "q": 2e-7}
        for short_name in arco_stage.LOCAL_PRESSURE_FIELDS:
            for level in stage.LEVELS:
                actual = pressure[short_name].sel(level=level).values
                target = expected.sel(channel=f"{short_name}{level}").values
                assert np.nanmax(np.abs(actual - target)) < limits[short_name]
        surface_limits = {
            "t2m": 0.001,
            "d2m": 0.001,
            "sst": 0.001,
            "10u": 0.001,
            "10v": 0.001,
            "msl": 0.1,
            "tcwv": 0.001,
            "tp": 0.001,
        }
        for short_name in arco_stage.LOCAL_SURFACE_FIELDS:
            actual = surface[short_name].values
            if short_name == "tp":
                actual = actual * 1000.0
            target = expected.sel(channel=short_name).values
            assert np.nanmax(np.abs(actual - target)) < surface_limits[short_name]
    finally:
        expected.close()


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
    details = prepare.validate_input(output, pd.Timestamp("2020-01-02"), config)
    assert details["shape"] == [2, 76, 121, 240]


def test_strict_daily_staging_dates_start_before_issue_year():
    config = json.loads(STRICT_CONFIG_PATH.read_text(encoding="utf-8"))
    dates = stage.required_dates(config)
    assert dates[0] == pd.Timestamp("2019-12-31")
    assert dates[1] == pd.Timestamp("2020-01-01")
    assert pd.Timestamp("2020-01-02") not in stage.required_month_dates(
        config, 2020, 1
    )
    assert len(dates) == 1242


def test_strict_output_has_physics_matching_daily_bounds(tmp_path: Path):
    config = runner.load_config(STRICT_CONFIG_PATH)
    config["storage_root"] = str(tmp_path)
    config["members"] = 2
    config["lead_days"] = 2
    issue = pd.Timestamp("2020-01-02")
    paths = runner.paths_for(config, issue)
    lat, lon = runner.expected_grid(config)

    for member in range(2):
        for step in range(1, 3):
            values = np.empty((1, 1, 2, len(lat), len(lon)), dtype=np.float32)
            values[:, :, 0] = 1.0 + member * 0.1
            values[:, :, 1] = 290.0 + member
            raw = xr.DataArray(
                values,
                dims=("time", "lead_time", "channel", "lat", "lon"),
                coords={
                    "time": [pd.Timestamp("2020-01-01")],
                    "lead_time": [step],
                    "channel": ["tp", "t2m"],
                    "lat": lat,
                    "lon": lon,
                },
            )
            path = paths["raw"] / "member" / f"{member:02d}" / f"{step:02d}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            raw.to_netcdf(path)

    runner.combine_output(paths["raw"], paths["output"], issue, config)
    details = runner.validate_output(paths["output"], issue, config)
    assert details["member_0_1_max_difference"] > 0
    dataset = xr.open_dataset(paths["output"])
    try:
        assert pd.Timestamp(dataset.forecast_reference_time.values) == issue
        assert pd.Timestamp(dataset.model_state_time.values) == pd.Timestamp(
            "2020-01-01"
        )
        assert pd.Timestamp(dataset.information_cutoff_time.values) == issue
        assert pd.Timestamp(dataset.forecast_period_start.values[0]) == issue
        assert pd.Timestamp(dataset.forecast_period_end.values[0]) == pd.Timestamp(
            "2020-01-03"
        )
        assert pd.Timestamp(dataset.valid_time.values[0]) == pd.Timestamp(
            "2020-01-03"
        )
        assert dataset.valid_time.attrs["bounds"] == "forecast_period_bounds"
    finally:
        dataset.close()


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
