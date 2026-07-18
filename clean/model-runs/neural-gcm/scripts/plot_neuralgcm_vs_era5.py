#!/usr/bin/env python3
"""Plot one NeuralGCM TP ensemble forecast against matching ERA5 truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


DEFAULT_ERA5 = Path(
    "/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", required=True, type=Path)
    parser.add_argument("--era5", default=DEFAULT_ERA5, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_truth(forecast: xr.Dataset, era5_path: Path) -> xr.DataArray:
    period_end = forecast["forecast_period_end"].values.astype("datetime64[h]")
    period_start = period_end - np.timedelta64(24, "h")
    offsets = np.asarray([6, 12, 18, 24], dtype="timedelta64[h]")
    endpoints = (period_start[:, None] + offsets[None, :]).reshape(-1)

    era5 = xr.open_zarr(era5_path, consolidated=True, chunks=None)
    try:
        six_hourly = era5["total_precipitation_6hr"].sel(time=endpoints).sel(
            latitude=forecast.latitude.values,
            longitude=forecast.longitude.values,
            method="nearest",
            tolerance=1e-6,
        )
        values = np.asarray(six_hourly.load().values, dtype=np.float64)
    finally:
        era5.close()

    expected = (forecast.sizes["lead_day"] * 4, forecast.sizes["longitude"], forecast.sizes["latitude"])
    if values.shape != expected:
        raise ValueError(f"unexpected ERA5 selection shape {values.shape}; expected {expected}")
    if not np.isfinite(values).all() or float(values.min()) < -1e-6:
        raise ValueError("ERA5 6-hour precipitation is non-finite or materially negative")
    values = np.maximum(values, 0.0)
    daily = values.reshape(forecast.sizes["lead_day"], 4, *values.shape[1:]).sum(axis=1)
    daily = daily.transpose(0, 2, 1) * 1000.0
    if not np.isfinite(daily).all():
        raise ValueError("ERA5 daily precipitation is non-finite")
    return xr.DataArray(
        daily.astype(np.float32),
        dims=("lead_day", "latitude", "longitude"),
        coords={
            "lead_day": forecast.lead_day.values,
            "latitude": forecast.latitude.values,
            "longitude": forecast.longitude.values,
            "valid_time": ("lead_day", forecast.valid_time.values),
        },
        name="era5_tp",
        attrs={"units": "mm day-1", "cell_methods": "time: sum (24-hour period)"},
    )


def area_mean(field: xr.DataArray) -> xr.DataArray:
    weights = xr.DataArray(
        np.cos(np.deg2rad(field.latitude.values)),
        dims="latitude",
        coords={"latitude": field.latitude.values},
    )
    return field.weighted(weights).mean(("latitude", "longitude"))


def metrics(forecast: xr.DataArray, truth: xr.DataArray) -> dict[str, float]:
    weights = xr.DataArray(
        np.cos(np.deg2rad(truth.latitude.values)),
        dims="latitude",
        coords={"latitude": truth.latitude.values},
    )
    difference = forecast - truth
    rmse = np.sqrt((difference**2).weighted(weights).mean()).item()
    mae = np.abs(difference).weighted(weights).mean().item()
    correlation = float(
        np.corrcoef(forecast.values.reshape(-1), truth.values.reshape(-1))[0, 1]
    )
    return {
        "rmse_mm_day": float(rmse),
        "mae_mm_day": float(mae),
        "raw_space_time_correlation": correlation,
    }


def plot_series(
    forecasts: dict[str, xr.DataArray], truth: xr.DataArray, output: Path
) -> None:
    colors = {"1 member": "#6b7280", "5-member mean": "#007f73", "10-member mean": "#c43c39"}
    dates = truth.valid_time.values
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(dates, area_mean(truth), color="#111111", linewidth=2.2, label="ERA5")
    for label, field in forecasts.items():
        axes[0].plot(dates, area_mean(field), color=colors[label], linewidth=1.6, label=label)
    axes[0].set_ylabel("Domain-mean TP (mm/day)")
    axes[0].set_title("NeuralGCM TP initialized 2020-01-02: daily evolution")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4, frameon=False)

    week = np.arange(1, 7)
    truth_weekly = area_mean(truth).values.reshape(6, 7).mean(axis=1)
    axes[1].plot(week, truth_weekly, marker="o", color="#111111", linewidth=2.2, label="ERA5")
    for label, field in forecasts.items():
        weekly = area_mean(field).values.reshape(6, 7).mean(axis=1)
        axes[1].plot(week, weekly, marker="o", color=colors[label], linewidth=1.6, label=label)
    axes[1].set(xlabel="Lead week", ylabel="Mean daily TP (mm/day)", xticks=week)
    axes[1].set_title("Seven-day mean precipitation by lead week")
    axes[1].grid(alpha=0.25)
    axes[1].text(
        0.01,
        0.03,
        "Illustrative one-case comparison; not a skill score",
        transform=axes[1].transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_maps(forecast: xr.DataArray, truth: xr.DataArray, output: Path) -> None:
    periods = [(0, 7, "Week 1"), (14, 28, "Weeks 3-4"), (28, 42, "Weeks 5-6")]
    truth_maps = [truth.isel(lead_day=slice(start, end)).mean("lead_day") for start, end, _ in periods]
    forecast_maps = [forecast.isel(lead_day=slice(start, end)).mean("lead_day") for start, end, _ in periods]
    vmax = float(
        np.nanpercentile(
            np.concatenate([field.values.reshape(-1) for field in truth_maps + forecast_maps]),
            98,
        )
    )
    vmax = max(vmax, 1.0)
    biases = [predicted - observed for predicted, observed in zip(forecast_maps, truth_maps, strict=True)]
    bias_limit = max(float(np.nanpercentile(np.abs(np.concatenate([b.values.reshape(-1) for b in biases])), 98)), 0.5)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True, sharex=True, sharey=True)
    extent = [float(truth.longitude.min()), float(truth.longitude.max()), float(truth.latitude.min()), float(truth.latitude.max())]
    precipitation_image = None
    bias_image = None
    for column, ((_, _, title), observed, predicted, bias) in enumerate(
        zip(periods, truth_maps, forecast_maps, biases, strict=True)
    ):
        precipitation_image = axes[0, column].imshow(observed.values, extent=extent, origin="upper", aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        axes[1, column].imshow(predicted.values, extent=extent, origin="upper", aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        bias_image = axes[2, column].imshow(bias.values, extent=extent, origin="upper", aspect="auto", cmap="RdBu", vmin=-bias_limit, vmax=bias_limit)
        axes[0, column].set_title(title)
        axes[2, column].set_xlabel("Longitude (degrees east)")
        for row in range(3):
            axes[row, column].grid(alpha=0.18)
    axes[0, 0].set_ylabel("ERA5\nLatitude")
    axes[1, 0].set_ylabel("NeuralGCM 10-member mean\nLatitude")
    axes[2, 0].set_ylabel("Forecast - ERA5\nLatitude")
    fig.colorbar(precipitation_image, ax=axes[:2, :], location="right", shrink=0.82, label="TP (mm/day)")
    fig.colorbar(bias_image, ax=axes[2, :], location="right", shrink=0.82, label="TP bias (mm/day)")
    fig.suptitle("NeuralGCM versus ERA5: common 1.5-degree verification domain", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.forecast) as source:
        forecast = source.load()
    if forecast.sizes.get("member", 0) < 10:
        raise ValueError("comparison requires at least 10 forecast members")
    truth = load_truth(forecast, args.era5)
    forecasts = {
        "1 member": forecast.tp.isel(member=0),
        "5-member mean": forecast.tp.isel(member=slice(0, 5)).mean("member"),
        "10-member mean": forecast.tp.isel(member=slice(0, 10)).mean("member"),
    }

    plot_series(forecasts, truth, args.output_dir / "tp_daily_weekly_20200102.png")
    plot_maps(forecasts["10-member mean"], truth, args.output_dir / "tp_maps_20200102_ens10_vs_era5.png")
    report = {
        "forecast": str(args.forecast),
        "era5": str(args.era5),
        "case": "2020-01-02",
        "note": "One-case raw comparison only; not a skill score.",
        "metrics": {label: metrics(field, truth) for label, field in forecasts.items()},
        "domain_mean_mm_day": {
            "ERA5": float(area_mean(truth).mean()),
            **{label: float(area_mean(field).mean()) for label, field in forecasts.items()},
        },
    }
    (args.output_dir / "tp_comparison_20200102.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
