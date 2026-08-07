"""Year-round export behavior outside the validated JJAS climatology."""

from __future__ import annotations

import numpy as np
import xarray as xr

from scripts.build_prototype_data import build_fields


def test_outside_jjas_exports_raw_fields_and_withholds_anomalies(tmp_path) -> None:
    member = np.arange(2)
    lead = np.arange(42)
    latitude = np.asarray([30.0, 28.5])
    longitude = np.asarray([75.0, 76.5])
    starts = np.arange(
        np.datetime64("2027-01-02"),
        np.datetime64("2027-02-13"),
        dtype="datetime64[D]",
    )
    forecast = xr.Dataset(
        {
            "tp": (
                ("member", "lead_day", "latitude", "longitude"),
                np.ones((2, 42, 2, 2), dtype=np.float64),
            ),
            "t2m": (
                ("member", "lead_day", "latitude", "longitude"),
                np.full((2, 42, 2, 2), 300.0, dtype=np.float64),
            ),
            "forecast_period_start": (("lead_day",), starts),
            "forecast_period_end": (("lead_day",), starts + np.timedelta64(1, "D")),
            "forecast_reference_time": np.datetime64("2027-01-02"),
            "model_state_time": np.datetime64("2027-01-01"),
        },
        coords={
            "member": member,
            "lead_day": lead,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    climatology = xr.Dataset(
        {
            "tp_ensemble_mean": (
                ("init_slot", "hindcast_year", "lead_day", "latitude", "longitude"),
                np.ones((2, 2, 42, 2, 2), dtype=np.float64),
            ),
            "t2m_ensemble_mean": (
                ("init_slot", "hindcast_year", "lead_day", "latitude", "longitude"),
                np.full((2, 2, 42, 2, 2), 295.0, dtype=np.float64),
            ),
        },
        coords={
            "init_slot": ["0602", "0929"],
            "hindcast_year": [2020, 2021],
            "lead_day": lead,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    forecast_path = tmp_path / "forecast.nc"
    climatology_path = tmp_path / "climatology.nc"
    forecast.to_netcdf(forecast_path)
    climatology.to_netcdf(climatology_path)

    _, _, _, _, fields, diagnostics = build_fields(
        forecast_path, climatology_path
    )

    assert set(fields) == {"rainfall_total", "temperature_mean"}
    assert diagnostics["alignment"]["status"] == "unavailable_outside_jjas"
    assert diagnostics["hindcast_years"] == []
    np.testing.assert_allclose(fields["rainfall_total"], 168.0)
    np.testing.assert_allclose(fields["temperature_mean"], 26.85)
