#!/usr/bin/env python3
"""Plot one FCN3 pilot against matching 6-hourly ERA5 verification fields."""

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
    lead_days = forecast.sizes["lead_day"]
    starts = forecast.forecast_period_start.values.astype("datetime64[h]")
    t2m_times = (
        starts[:, None] + np.array([0, 6, 12, 18, 24], dtype="timedelta64[h]")
    ).reshape(-1)
    era5 = xr.open_zarr(era5_path, consolidated=True, chunks=None)
    try:
        sampled_t2m = (
            era5["2m_temperature"]
            .sel(time=t2m_times)
            .sel(
                latitude=forecast.latitude,
                longitude=forecast.longitude,
                method="nearest",
                tolerance=1e-6,
            )
            .transpose("time", "latitude", "longitude")
        )
        t2m = np.asarray(sampled_t2m.load().values, dtype=np.float64).reshape(
            lead_days, 5, 27, 27
        )
        daily_t2m = (
            0.5 * t2m[:, 0]
            + t2m[:, 1]
            + t2m[:, 2]
            + t2m[:, 3]
            + 0.5 * t2m[:, 4]
        ) / 4.0 - 273.15

        fields = {
            "t2m": (
                ("lead_day", "latitude", "longitude"),
                daily_t2m.astype(np.float32),
            )
        }
        if "tp" in forecast:
            tp_times = (
                starts[:, None] + np.array([6, 12, 18, 24], dtype="timedelta64[h]")
            ).reshape(-1)
            sampled_tp = (
                era5["total_precipitation_6hr"]
                .sel(time=tp_times)
                .sel(
                    latitude=forecast.latitude,
                    longitude=forecast.longitude,
                    method="nearest",
                    tolerance=1e-6,
                )
                .transpose("time", "latitude", "longitude")
            )
            tp = np.asarray(sampled_tp.load().values, dtype=np.float64).reshape(
                lead_days, 4, 27, 27
            )
            fields["tp"] = (
                ("lead_day", "latitude", "longitude"),
                (np.maximum(tp, 0.0).sum(axis=1) * 1000.0).astype(np.float32),
            )
    finally:
        era5.close()

    return xr.Dataset(
        fields,
        coords={
            "lead_day": forecast.lead_day,
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
            "valid_time": forecast.valid_time,
        },
    )


def metrics(prediction: xr.DataArray, truth: xr.DataArray) -> dict[str, float]:
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


def plot_evolution(
    name: str,
    forecast: xr.DataArray,
    truth: xr.DataArray,
    output_dir: Path,
) -> None:
    units = "degC" if name == "t2m" else "mm/day"
    dates = truth.valid_time.values
    member_series = area_mean(forecast)
    ensemble_mean = member_series.mean("member")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for member in member_series.member:
        axes[0].plot(
            dates,
            member_series.sel(member=member),
            color="#5b8db8",
            linewidth=0.8,
            alpha=0.45,
        )
    axes[0].plot(dates, area_mean(truth), color="black", linewidth=2.0, label="ERA5")
    axes[0].plot(
        dates,
        ensemble_mean,
        color="#b33b32",
        linewidth=1.8,
        label="FCN3 ensemble mean",
    )
    axes[0].set_ylabel(f"Domain mean ({units})")
    axes[0].set_title(f"Daily {name.upper()} evolution")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    week = np.arange(1, 7)
    weekly_truth = area_mean(truth).values.reshape(6, 7).mean(1)
    weekly_members = member_series.values.reshape(forecast.sizes["member"], 6, 7).mean(2)
    axes[1].plot(week, weekly_truth, marker="o", color="black", label="ERA5")
    for values in weekly_members:
        axes[1].plot(week, values, color="#5b8db8", linewidth=0.8, alpha=0.45)
    axes[1].plot(
        week,
        weekly_members.mean(0),
        marker="o",
        color="#b33b32",
        label="FCN3 ensemble mean",
    )
    axes[1].set(xlabel="Lead week", ylabel=f"Weekly mean ({units})", xticks=week)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].text(
        0.01,
        0.04,
        "One-case pipeline validation, not a paper skill score",
        transform=axes[1].transAxes,
        color="#555555",
    )
    fig.savefig(output_dir / f"{name}_daily_weekly_vs_era5.png", dpi=180)
    plt.close(fig)


def plot_maps(
    name: str,
    prediction: xr.DataArray,
    truth: xr.DataArray,
    output_dir: Path,
) -> None:
    units = "degC" if name == "t2m" else "mm/day"
    periods = [(0, 7, "Week 1"), (14, 28, "Weeks 3-4"), (28, 42, "Weeks 5-6")]
    observed = [truth.isel(lead_day=slice(start, end)).mean("lead_day") for start, end, _ in periods]
    predicted = [prediction.isel(lead_day=slice(start, end)).mean("lead_day") for start, end, _ in periods]
    differences = [model - obs for model, obs in zip(predicted, observed, strict=True)]
    absolute_values = np.concatenate([item.values.ravel() for item in observed + predicted])
    if name == "tp":
        vmin, vmax, cmap = 0.0, max(float(np.percentile(absolute_values, 98)), 1.0), "YlGnBu"
    else:
        vmin = float(np.percentile(absolute_values, 2))
        vmax = float(np.percentile(absolute_values, 98))
        cmap = "coolwarm"
    bias_limit = max(
        float(np.percentile(np.abs(np.concatenate([x.values.ravel() for x in differences])), 98)),
        0.1,
    )
    extent = [
        float(truth.longitude.min()),
        float(truth.longitude.max()),
        float(truth.latitude.min()),
        float(truth.latitude.max()),
    ]
    fig, axes = plt.subplots(
        3, 3, figsize=(13, 10), constrained_layout=True, sharex=True, sharey=True
    )
    main_image = bias_image = None
    for column, ((_, _, title), obs, model, difference) in enumerate(
        zip(periods, observed, predicted, differences, strict=True)
    ):
        main_image = axes[0, column].imshow(
            obs, extent=extent, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        axes[1, column].imshow(
            model, extent=extent, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        bias_image = axes[2, column].imshow(
            difference,
            extent=extent,
            origin="upper",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-bias_limit,
            vmax=bias_limit,
        )
        axes[0, column].set_title(title)
        axes[2, column].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("ERA5\nLatitude")
    axes[1, 0].set_ylabel("FCN3 ensemble mean\nLatitude")
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
    if forecast.sizes.get("lead_day") != 42:
        raise ValueError("comparison plot requires a complete 42-day forecast")
    truth = load_truth(forecast, args.era5)
    report = {
        "forecast": str(args.forecast),
        "era5": str(args.era5),
        "note": "One-case pipeline validation only; not a paper skill score.",
        "metrics": {},
    }
    for name in (name for name in ("t2m", "tp") if name in forecast):
        prediction = forecast[name].mean("member")
        report["metrics"][name] = metrics(prediction, truth[name])
        report["metrics"][name]["forecast_domain_mean"] = float(
            area_mean(prediction).mean()
        )
        report["metrics"][name]["era5_domain_mean"] = float(
            area_mean(truth[name]).mean()
        )
        plot_evolution(name, forecast[name], truth[name], args.output_dir)
        plot_maps(name, prediction, truth[name], args.output_dir)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
