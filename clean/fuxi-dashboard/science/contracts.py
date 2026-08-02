"""Typed, immutable contracts for the one-day FuXi dashboard prototype."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridContract:
    """Expected regular latitude/longitude grid."""

    latitude_count: int
    longitude_count: int
    latitude_first: float
    latitude_last: float
    longitude_first: float
    longitude_last: float
    spacing_degrees: float


@dataclass(frozen=True)
class ForecastContract:
    """Structural and temporal contract for a FuXi forecast."""

    initialization: str
    information_cutoff: str
    model_state_time: str
    period_start: str
    period_end_exclusive: str
    members: int
    lead_days: int
    tp_units: str
    t2m_units: str
    grid: GridContract


@dataclass(frozen=True)
class ClimatologyContract:
    """Sampling and unit contract for a climatology."""

    source_id: str
    baseline: str
    sample_count: int
    calendar_scope: str
    units: str


INDIA_GRID = GridContract(
    latitude_count=27,
    longitude_count=27,
    latitude_first=39.0,
    latitude_last=0.0,
    longitude_first=60.0,
    longitude_last=99.0,
    spacing_degrees=1.5,
)

FUXI_FORECAST = ForecastContract(
    initialization="2026-07-28T00:00:00",
    information_cutoff="2026-07-28T00:00:00",
    model_state_time="2026-07-27T00:00:00",
    period_start="2026-07-28T00:00:00",
    period_end_exclusive="2026-09-08T00:00:00",
    members=100,
    lead_days=42,
    tp_units="mm h-1",
    t2m_units="K",
    grid=INDIA_GRID,
)

FUXI_CLIMATOLOGY = ClimatologyContract(
    source_id="fuxi_native_reforecast_climatology",
    baseline="2002-2021",
    sample_count=20,
    calendar_scope="JJAS native initialization slots",
    units="TP: mm day-1; T2M: K",
)

IMD_CLIMATOLOGY = ClimatologyContract(
    source_id="imd_daily_rainfall_climatology",
    baseline="1991-2020",
    sample_count=30,
    calendar_scope="365-day calendar; Feb 29 removed",
    units="mm/day",
)

IMERG_CLIMATOLOGY = ClimatologyContract(
    source_id="imerg_final_v07b_daily_climatology",
    baseline="2001-2022",
    sample_count=22,
    calendar_scope="06-06 through 10-25 inclusive",
    units="mm day-1",
)

BASELINES = {
    "fuxi": FUXI_CLIMATOLOGY.baseline,
    "imd": IMD_CLIMATOLOGY.baseline,
    "imerg": IMERG_CLIMATOLOGY.baseline,
}
