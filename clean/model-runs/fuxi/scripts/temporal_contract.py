#!/usr/bin/env python3
"""Shared issue-time and daily-period semantics for FuXi-S2S runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_KEYS = {
    "benchmark_mode",
    "first_forecast_period_start_offset_days",
    "input_day_offsets",
    "issue_hour_utc",
    "strict_operational",
    "valid_time_role",
}


def alignment(config: dict[str, Any]) -> dict[str, Any]:
    try:
        contract = config["temporal_alignment"]
    except KeyError as exc:
        raise ValueError("config is missing temporal_alignment") from exc
    missing = REQUIRED_KEYS - set(contract)
    if missing:
        raise ValueError(f"temporal_alignment missing keys: {sorted(missing)}")

    offsets = [int(value) for value in contract["input_day_offsets"]]
    if len(offsets) != 2 or offsets[1] - offsets[0] != 1:
        raise ValueError("FuXi requires two consecutive daily input offsets")
    if int(contract["issue_hour_utc"]) != 0:
        raise ValueError("this benchmark supports only 00 UTC physics issue times")

    first_target = int(contract["first_forecast_period_start_offset_days"])
    if first_target != offsets[-1] + 1:
        raise ValueError(
            "first forecast day must immediately follow the latest FuXi input day"
        )
    if contract["valid_time_role"] not in {"period_start", "period_end"}:
        raise ValueError("valid_time_role must be period_start or period_end")
    if bool(contract["strict_operational"]):
        if offsets != [-2, -1] or first_target != 0:
            raise ValueError(
                "strict 00 UTC mode requires D-2/D-1 inputs and a day-D target"
            )
        if contract["valid_time_role"] != "period_end":
            raise ValueError("strict 00 UTC mode labels valid_time by period end")
    return contract


def input_days(issue_date: pd.Timestamp, config: dict[str, Any]) -> pd.DatetimeIndex:
    contract = alignment(config)
    issue = pd.Timestamp(issue_date).normalize()
    return pd.DatetimeIndex(
        [issue + pd.Timedelta(days=int(offset)) for offset in contract["input_day_offsets"]]
    )


def information_cutoff(issue_date: pd.Timestamp, config: dict[str, Any]) -> pd.Timestamp:
    """Return when the latest complete UTC daily mean becomes available."""

    latest_day = input_days(issue_date, config)[-1]
    return latest_day + pd.Timedelta(days=1)


def model_state_time(issue_date: pd.Timestamp, config: dict[str, Any]) -> pd.Timestamp:
    return input_days(issue_date, config)[-1]


def forecast_periods(
    issue_date: pd.Timestamp,
    lead_days: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contract = alignment(config)
    issue = pd.Timestamp(issue_date).normalize()
    lead = np.arange(1, int(lead_days) + 1, dtype=np.int16)
    first_offset = int(contract["first_forecast_period_start_offset_days"])
    start = (
        issue.to_datetime64()
        + (lead.astype(np.int64) - 1 + first_offset).astype("timedelta64[D]")
    ).astype("datetime64[ns]")
    end = (start + np.timedelta64(1, "D")).astype("datetime64[ns]")
    valid = start if contract["valid_time_role"] == "period_start" else end
    return start, end, valid


def provenance(issue_date: pd.Timestamp, config: dict[str, Any]) -> dict[str, Any]:
    issue = pd.Timestamp(issue_date).normalize()
    days = input_days(issue, config)
    cutoff = information_cutoff(issue, config)
    contract = alignment(config)
    return {
        "benchmark_mode": contract["benchmark_mode"],
        "strict_operational": bool(contract["strict_operational"]),
        "forecast_reference_time": issue.isoformat(),
        "input_days": [day.strftime("%Y-%m-%d") for day in days],
        "model_state_time": model_state_time(issue, config).isoformat(),
        "information_cutoff_time": cutoff.isoformat(),
        "information_cutoff_matches_issue_time": cutoff == issue,
        "first_forecast_period_start": (
            issue
            + pd.Timedelta(
                days=int(contract["first_forecast_period_start_offset_days"])
            )
        ).isoformat(),
        "valid_time_role": contract["valid_time_role"],
    }
