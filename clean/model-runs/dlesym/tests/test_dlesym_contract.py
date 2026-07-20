from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("dlesym_common", "dlesym_common.py")
runner = load_module("run_dlesym_42d", "run_dlesym_42d.py")


def test_frozen_calendar_and_target_grid() -> None:
    config = common.load_config(common.REPO_ROOT / "model-runs/configs/dlesym_2020_2024.json")
    assert len(common.read_dates(config)) == 517
    latitude, longitude = common.target_coordinates(config)
    assert latitude.tolist() == np.arange(39.0, -0.1, -1.5).tolist()
    assert longitude.tolist() == np.arange(60.0, 99.1, 1.5).tolist()


def test_trapezoidal_daily_mean() -> None:
    boundaries = [torch.full((2, 2), float(index)) for index in range(169)]
    daily = runner.aggregate_t2m(boundaries)
    assert daily.shape == (42, 2, 2)
    assert torch.allclose(daily[:, 0, 0], torch.arange(2.0, 168.0, 4.0))


def test_conservative_weights_are_normalized() -> None:
    source_lat = np.linspace(90.0, -90.0, 721)
    source_lon = np.linspace(0.0, 359.75, 1440)
    target_lat = np.arange(39.0, -0.1, -1.5)
    target_lon = np.arange(60.0, 99.1, 1.5)
    lat_weight, lon_weight, digest = common.conservative_weights(
        source_lat, source_lon, target_lat, target_lon
    )
    assert np.allclose(lat_weight.sum(axis=1), 1.0)
    assert np.allclose(lon_weight.sum(axis=1), 1.0)
    assert len(digest) == 64
    constant = lat_weight @ np.ones((721, 1440)) @ lon_weight.T
    assert np.allclose(constant, 1.0)
