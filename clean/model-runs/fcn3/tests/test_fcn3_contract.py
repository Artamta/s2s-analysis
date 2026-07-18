from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("fcn3_common", "fcn3_common.py")
runner = load_module("run_fcn3_afnov2", "run_fcn3_afnov2.py")
t2m_runner = load_module("run_fcn3_t2m", "run_fcn3_t2m.py")


def test_frozen_calendar_and_target_grid() -> None:
    config = common.load_config(
        common.REPO_ROOT / "model-runs/configs/fcn3_2020_2024.json"
    )
    assert len(common.read_dates(config)) == 517
    latitude, longitude = common.target_coordinates(config)
    assert latitude.tolist() == np.arange(39.0, -0.1, -1.5).tolist()
    assert longitude.tolist() == np.arange(60.0, 99.1, 1.5).tolist()


def test_period_coordinates_are_complete_utc_days() -> None:
    coords = common.period_coordinates("2020-06-01", 42)
    assert coords["lead_day"].tolist() == list(range(1, 43))
    starts = coords["forecast_period_start"][1]
    ends = coords["forecast_period_end"][1]
    assert starts[0] == np.datetime64("2020-06-01")
    assert ends[-1] == np.datetime64("2020-07-13")
    assert np.all(ends - starts == np.timedelta64(1, "D"))


def test_conservative_weights_are_normalized_and_constant_preserving() -> None:
    source_lat = np.linspace(90.0, -89.75, 720)
    source_lon = np.linspace(0.0, 359.75, 1440)
    target_lat = np.arange(39.0, -0.1, -1.5)
    target_lon = np.arange(60.0, 99.1, 1.5)
    lat_weight, lon_weight, digest = common.conservative_weights(
        source_lat, source_lon, target_lat, target_lon
    )
    assert np.allclose(lat_weight.sum(axis=1), 1.0)
    assert np.allclose(lon_weight.sum(axis=1), 1.0)
    assert len(digest) == 64
    constant = lat_weight @ np.ones((720, 1440)) @ lon_weight.T
    assert np.allclose(constant, 1.0)


def test_explicit_test_output_never_becomes_production(tmp_path: Path) -> None:
    canonical = tmp_path / "forecasts" / "case.nc"
    pilot = tmp_path / "pilots" / "case.nc"
    production, output = runner.resolve_output_mode(canonical, 168, 10, 10, pilot)
    assert not production
    assert output == pilot

    production, output = runner.resolve_output_mode(canonical, 168, 10, 10, None)
    assert production
    assert output == canonical


def test_t2m_only_contract_and_daily_mean() -> None:
    config = common.load_config(
        common.REPO_ROOT / "model-runs/configs/fcn3_t2m_2020_2024.json"
    )
    assert config["model"]["fields"] == ["t2m"]
    assert config["model"]["members"] == 3
    assert config["model"]["seeds"] == [31000, 31001, 31002]
    assert "precipitation_package_uri" not in config["model"]
    boundaries = [t2m_runner.torch.full((2, 2), float(value)) for value in range(5)]
    result = t2m_runner.daily_trapezoid(boundaries)
    assert np.allclose(result.numpy(), 2.0)


def test_t2m_explicit_test_output_never_becomes_production(tmp_path: Path) -> None:
    canonical = tmp_path / "forecasts" / "case.nc"
    pilot = tmp_path / "pilots" / "case.nc"
    production, output = t2m_runner.resolve_output_mode(canonical, 168, 10, 10, pilot)
    assert not production
    assert output == pilot
