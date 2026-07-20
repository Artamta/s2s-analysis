#!/usr/bin/env python3
"""Plot one DLESyM pilot forecast against matching ERA5 verification fields."""

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


def area_mean(field: xr.DataArray) -> xr.DataArray:
    weights = xr.DataArray(
        np.cos(np.deg2rad(field.latitude.values)),
        dims="latitude",
        coords={"latitude": field.latitude.values},
    )
    return field.weighted(weights).mean(("latitude", "longitude"))


def load_truth(forecast: xr.Dataset, era5_path: Path) -> xr.Dataset:
    starts = forecast.forecast_period_start.values.astype("datetime64[h]")
    t2m_times = (starts[:, None] + np.array([0, 6, 12, 18, 24], dtype="timedelta64[h]")).reshape(-1)
    tp_times = (starts[:, None] + np.array([6, 12, 18, 24], dtype="timedelta64[h]")).reshape(-1)
    era5 = xr.open_zarr(era5_path, consolidated=True, chunks=None)
    truth = {}
    try:
        if "t2m" in forecast:
            sampled = era5["2m_temperature"].sel(time=t2m_times).sel(
                latitude=forecast.latitude,
                longitude=forecast.longitude,
                method="nearest",
                tolerance=1e-6,
            ).transpose("time", "latitude", "longitude")
            values = np.asarray(sampled.load().values, dtype=np.float64).reshape(
                42, 5, 27, 27
            )
            truth["t2m"] = (
                0.5 * values[:, 0]
                + values[:, 1]
                + values[:, 2]
                + values[:, 3]
                + 0.5 * values[:, 4]
            ) / 4.0 - 273.15
        if "tp" in forecast:
            sampled = era5["total_precipitation_6hr"].sel(time=tp_times).sel(
                latitude=forecast.latitude,
                longitude=forecast.longitude,
                method="nearest",
                tolerance=1e-6,
            ).transpose("time", "latitude", "longitude")
            values = np.asarray(sampled.load().values, dtype=np.float64).reshape(
                42, 4, 27, 27
            )
            truth["tp"] = np.maximum(values, 0.0).sum(axis=1) * 1000.0
    finally:
        era5.close()
    return xr.Dataset(
        {
            name: (("lead_day", "latitude", "longitude"), values.astype(np.float32))
            for name, values in truth.items()
        },
        coords={
            "lead_day": forecast.lead_day,
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
            "valid_time": forecast.valid_time,
        },
    )


def metric(prediction: xr.DataArray, truth: xr.DataArray) -> dict[str, float]:
    weights = xr.DataArray(
        np.cos(np.deg2rad(truth.latitude.values)),
        dims="latitude",
        coords={"latitude": truth.latitude.values},
    )
    difference = prediction - truth
    return {
        "rmse": float(np.sqrt((difference**2).weighted(weights).mean())),
        "mae": float(np.abs(difference).weighted(weights).mean()),
        "raw_space_time_correlation": float(
            np.corrcoef(prediction.values.ravel(), truth.values.ravel())[0, 1]
        ),
    }


def plot_field(
    name: str,
    forecast: xr.DataArray,
    truth: xr.DataArray,
    output_dir: Path,
) -> None:
    units = "degC" if name == "t2m" else "mm/day"
    dates = truth.valid_time.values
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(dates, area_mean(truth), color="black", linewidth=2.0, label="ERA5")
    axes[0].plot(
        dates,
        area_mean(forecast),
        color="#007f73" if name == "t2m" else "#2f6da8",
        linewidth=1.7,
        label="DLESyM ensemble mean" if "member" in forecast.dims else "DLESyM",
    )
    axes[0].set_ylabel(f"Domain mean ({units})")
    axes[0].set_title(f"Daily {name.upper()} evolution")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    week = np.arange(1, 7)
    axes[1].plot(
        week, area_mean(truth).values.reshape(6, 7).mean(1), marker="o", color="black", label="ERA5"
    )
    axes[1].plot(
        week, area_mean(forecast).values.reshape(6, 7).mean(1), marker="o", color="#c43c39", label="DLESyM"
    )
    axes[1].set(xlabel="Lead week", ylabel=f"Weekly mean ({units})", xticks=week)
    axes[1].grid(alpha=0.25)
    axes[1].text(0.01, 0.04, "One-case pipeline validation, not a paper skill score", transform=axes[1].transAxes, color="#555555")
    fig.savefig(output_dir / f"{name}_daily_weekly_vs_era5.png", dpi=180)
    plt.close(fig)

    periods = [(0, 7, "Week 1"), (14, 28, "Weeks 3-4"), (28, 42, "Weeks 5-6")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True, sharex=True, sharey=True)
    extent = [float(truth.longitude.min()), float(truth.longitude.max()), float(truth.latitude.min()), float(truth.latitude.max())]
    absolute = []
    differences = []
    for start, end, _ in periods:
        observed = truth.isel(lead_day=slice(start, end)).mean("lead_day")
        predicted = forecast.isel(lead_day=slice(start, end)).mean("lead_day")
        absolute.extend([observed, predicted])
        differences.append(predicted - observed)
    if name == "tp":
        vmin, vmax, cmap = 0.0, max(float(np.percentile(np.concatenate([x.values.ravel() for x in absolute]), 98)), 1.0), "YlGnBu"
    else:
        vmin, vmax, cmap = float(np.percentile(np.concatenate([x.values.ravel() for x in absolute]), 2)), float(np.percentile(np.concatenate([x.values.ravel() for x in absolute]), 98)), "coolwarm"
    bias_limit = max(float(np.percentile(np.abs(np.concatenate([x.values.ravel() for x in differences])), 98)), 0.1)
    main_image = bias_image = None
    for column, ((_, _, title), observed, predicted, difference) in enumerate(zip(periods, absolute[::2], absolute[1::2], differences, strict=True)):
        main_image = axes[0, column].imshow(observed, extent=extent, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, column].imshow(predicted, extent=extent, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        bias_image = axes[2, column].imshow(difference, extent=extent, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-bias_limit, vmax=bias_limit)
        axes[0, column].set_title(title)
        axes[2, column].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("ERA5\nLatitude")
    axes[1, 0].set_ylabel("DLESyM\nLatitude")
    axes[2, 0].set_ylabel("Forecast - ERA5\nLatitude")
    fig.colorbar(main_image, ax=axes[:2], shrink=0.8, label=f"{name.upper()} ({units})")
    fig.colorbar(bias_image, ax=axes[2], shrink=0.8, label=f"Bias ({units})")
    fig.savefig(output_dir / f"{name}_maps_vs_era5.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--era5", default=DEFAULT_ERA5, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.forecast) as source:
        forecast = source.load()
    truth = load_truth(forecast, args.era5)
    report = {
        "forecast": str(args.forecast),
        "era5": str(args.era5),
        "note": "One-case pipeline validation only; not a paper skill score.",
        "metrics": {},
    }
    for name in truth.data_vars:
        prediction = forecast[name].mean("member")
        report["metrics"][name] = metric(prediction, truth[name])
        report["metrics"][name]["forecast_domain_mean"] = float(area_mean(prediction).mean())
        report["metrics"][name]["era5_domain_mean"] = float(area_mean(truth[name]).mean())
        plot_field(name, prediction, truth[name], args.output_dir)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
