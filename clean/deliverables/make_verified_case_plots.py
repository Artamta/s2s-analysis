#!/usr/bin/env python3
"""Create matching FuXi/ECMWF cumulative-rainfall plots with observed verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path("/home/raj.ayush/s2s/s2s_anlysis/clean")
STORAGE = Path("/storage/raj.ayush")
GT_FILE = ROOT / "deliverables/ground_truth_202606/all_india_daily_ground_truth_20260602_20260718.csv"
GT_EXTRA = ROOT / "deliverables/ground_truth_202606/ground_truth_20260601_one_day.csv"
MASK_FILE = STORAGE / "s2s-forecast-data-prev/era5/daily/imd_region_masks.nc"
IMD_CLIMO = STORAGE / "All_Model_Data/ground_truth/imd_rainfall/climatology/imd_rain_1991_2020_daily_climatology.nc"
ERA5_CLIMO = STORAGE / "benchmark(jfm)/era5_climatology.nc"
REGIONS = ("northwest_india", "central_india", "south_peninsula", "east_northeast_india")


def all_india_mask(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    with xr.open_dataset(MASK_FILE) as source:
        source = source.sortby("lat").sortby("lon")
        union = xr.zeros_like(source[REGIONS[0]], dtype=bool)
        for region in REGIONS:
            union = union | (source[region] > 0)
        return (union.astype(float).interp(lat=lat, lon=lon, method="nearest") >= 0.5).load()


def spatial_mean(field: xr.DataArray) -> xr.DataArray:
    field = field.sortby("lat").sortby("lon")
    mask = all_india_mask(field.lat, field.lon)
    weights = xr.DataArray(np.cos(np.deg2rad(field.lat)), coords={"lat": field.lat}, dims="lat")
    return field.where(mask).weighted(weights).mean(("lat", "lon"), skipna=True)


def fuxi_series(init: str) -> pd.DataFrame:
    path = (
        STORAGE / "s2s_final_data/final_iteration/model-runs/fuxi" /
        f"fuxi_s2s_strict00z_case_{init}_ens50/forecasts/annual2026/{init}.nc"
    )
    with xr.open_dataset(path) as source:
        expected_date = pd.Timestamp(init).strftime("%Y-%m-%d")
        checks = {
            "init_date": expected_date,
            "input_daily_statistic": "daily_mean",
            "benchmark_mode": "strict_information_matched_00utc",
        }
        for key, expected in checks.items():
            if str(source.attrs.get(key)) != expected:
                raise ValueError(f"FuXi {key}: expected {expected!r}; found {source.attrs.get(key)!r}")
        tp = source.tp.rename({"latitude": "lat", "longitude": "lon"}) * 24.0
        cumulative = spatial_mean(tp).cumsum("lead_day").load()
        valid_dates = pd.to_datetime(source.valid_time.values)
    values = np.asarray(cumulative.values)
    return pd.DataFrame({
        "lead_day": np.arange(1, values.shape[1] + 1),
        "valid_date": valid_dates,
        "fuxi_mean": values.mean(axis=0),
        "fuxi_p10": np.quantile(values, 0.10, axis=0),
        "fuxi_p90": np.quantile(values, 0.90, axis=0),
    })


def ecmwf_series(init: str, valid_dates: pd.DatetimeIndex) -> pd.DataFrame:
    path = STORAGE / f"All_Model_Data/ecmwf/jjas2026/tp/{init}_pf.nc"
    with xr.open_dataset(path) as source:
        tp = source.tp.isel(number=slice(0, 50), step=slice(0, 42)).rename(
            {"number": "member", "latitude": "lat", "longitude": "lon"}
        )
        # Archived ECMWF tp is accumulation from initialization in kg m-2 (= mm).
        cumulative = spatial_mean(tp.clip(min=0)).load()
    values = np.asarray(cumulative.values)
    return pd.DataFrame({
        "lead_day": np.arange(1, 43),
        "valid_date": valid_dates,
        "ecmwf_mean": values.mean(axis=0),
        "ecmwf_p10": np.quantile(values, 0.10, axis=0),
        "ecmwf_p90": np.quantile(values, 0.90, axis=0),
    })


def imd_climatology(valid_dates: pd.DatetimeIndex) -> np.ndarray:
    month_days = [value.strftime("%m-%d") for value in valid_dates]
    with xr.open_dataset(IMD_CLIMO) as source:
        index = {str(value): i for i, value in enumerate(source.month_day.values)}
        rain = source.rain_mean.isel(day=[index[value] for value in month_days]).load()
    return np.asarray(spatial_mean(rain).cumsum("day").values)


def era5_climatology(valid_dates: pd.DatetimeIndex) -> np.ndarray:
    # valid_dates are 24-hour period endpoints; ERA5 daily statistics are labelled
    # by the calendar day at the start of each forecast period.
    source_dates = valid_dates - pd.Timedelta(days=1)
    dayofyear = [pd.Timestamp(2000, value.month, value.day).dayofyear for value in source_dates]
    with xr.open_dataset(ERA5_CLIMO) as source:
        rain = source.tp.sel(dayofyear=dayofyear).rename(
            {"latitude": "lat", "longitude": "lon"}
        ) * 1000.0
        rain = rain.sel(lat=slice(39.5, 5.5), lon=slice(60, 100)).load()
    return np.asarray(spatial_mean(rain).cumsum("dayofyear").values)


def add_observations(data: pd.DataFrame) -> pd.DataFrame:
    observations = pd.concat(
        [pd.read_csv(path, parse_dates=["date"]) for path in (GT_EXTRA, GT_FILE)],
        ignore_index=True,
    ).drop_duplicates("date", keep="last").sort_values("date")

    # IMERG's file date is the UTC day at the start of the accumulation;
    # plot it at the matching 24-hour forecast-period endpoint one day later.
    imerg = observations[["date", "imerg_late_mm"]].copy()
    imerg["valid_date"] = imerg.pop("date") + pd.Timedelta(days=1)

    # IMD daily gauge analysis is valid near 03 UTC on its stated date, so its
    # date is already the closest match to the forecast-period endpoint.
    imd = observations[["date", "imd_gauge_mm", "imd_gauge_gpm_merged_mm"]].rename(
        columns={"date": "valid_date"}
    )
    data = data.merge(imerg, on="valid_date", how="left", validate="one_to_one")
    data = data.merge(imd, on="valid_date", how="left", validate="one_to_one")
    for source in ("imerg_late_mm", "imd_gauge_mm", "imd_gauge_gpm_merged_mm"):
        observed = data[source].notna()
        cumulative = data[source].where(observed).cumsum()
        data[source.replace("_mm", "_cumulative_mm")] = cumulative.where(observed)
    return data


def verification_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for observation, observation_label in (
        ("imerg_late_cumulative_mm", "IMERG Late"),
        ("imd_gauge_cumulative_mm", "IMD gauge"),
    ):
        valid = data[observation].notna()
        for forecast, forecast_label in (("fuxi_mean", "FuXi-S2S"), ("ecmwf_mean", "ECMWF S2S")):
            error = data.loc[valid, forecast] - data.loc[valid, observation]
            rows.append({
                "observation": observation_label,
                "forecast": forecast_label,
                "n_days": int(valid.sum()),
                "last_date": data.loc[valid, "valid_date"].iloc[-1],
                "cumulative_mae_mm": float(error.abs().mean()),
                "cumulative_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
                "endpoint_forecast_mm": float(data.loc[valid, forecast].iloc[-1]),
                "endpoint_observed_mm": float(data.loc[valid, observation].iloc[-1]),
                "endpoint_bias_mm": float(error.iloc[-1]),
                "endpoint_absolute_error_mm": float(abs(error.iloc[-1])),
            })
    return pd.DataFrame(rows)


def plot_case(init: str) -> Path:
    valid_dates = pd.date_range(pd.Timestamp(init) + pd.Timedelta(days=1), periods=42, freq="D")
    data = fuxi_series(init)
    data = data.merge(ecmwf_series(init, valid_dates), on=["lead_day", "valid_date"], validate="one_to_one")
    data["imd_climatology_mm"] = imd_climatology(valid_dates)
    data["era5_climatology_mm"] = era5_climatology(valid_dates)
    data = add_observations(data)

    output_dir = ROOT / f"deliverables/daily_mean_case_{init}"
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / f"{init}_cumulative_with_ground_truth.csv", index=False)
    metrics = verification_metrics(data)
    metrics.to_csv(output_dir / f"{init}_verification_metrics.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.edgecolor": "#263238", "axes.linewidth": 0.9,
    })
    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    x = data.valid_date

    ax.fill_between(x, data.ecmwf_p10, data.ecmwf_p90, color="#f59e0b", alpha=0.12, linewidth=0)
    ax.plot(x, data.ecmwf_mean, color="#e67e00", lw=2.4, label="ECMWF S2S (50-member mean)")
    ax.fill_between(x, data.fuxi_p10, data.fuxi_p90, color="#16a34a", alpha=0.14, linewidth=0)
    ax.plot(x, data.fuxi_mean, color="#138a45", lw=2.9, label="FuXi-S2S (daily-mean inputs)")

    ax.plot(x, data.imerg_late_cumulative_mm, color="#111827", lw=3.0, label="IMERG Late observation")
    ax.plot(x, data.imd_gauge_cumulative_mm, color="#1565c0", lw=2.8, label="IMD gauge observation")
    ax.plot(x, data.imd_climatology_mm, color="#64748b", lw=1.8, ls="-.", label="IMD 1991–2020 climatology")
    ax.plot(x, data.era5_climatology_mm, color="#7c3aed", lw=1.8, ls=":", label="ERA5 climatology")

    init_label = pd.Timestamp(init).strftime("%-d %B %Y")
    start_label = valid_dates[0].strftime("%-d %B")
    end_label = valid_dates[-1].strftime("%-d %B %Y")
    ax.set_title(f"42-Day All-India Cumulative Rainfall — {init_label} Forecast", loc="left", fontsize=16.5, weight="bold", pad=18)
    ax.text(
        0, 1.015,
        f"FuXi IC: two complete UTC daily means  |  Forecast valid: {start_label}–{end_label}",
        transform=ax.transAxes, color="#4b5563", fontsize=10.3,
    )
    ax.set_ylabel("Cumulative rainfall (mm)")
    ax.set_xlabel("Forecast valid date")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#dbe3ea", lw=0.75, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", frameon=False, ncols=2)

    last_imerg = data.loc[data.imerg_late_cumulative_mm.notna(), "valid_date"].max()
    last_imd = data.loc[data.imd_gauge_cumulative_mm.notna(), "valid_date"].max()
    last_common = min(last_imerg, last_imd)
    if pd.notna(last_common) and last_common < valid_dates[-1]:
        ax.axvline(last_common, color="#94a3b8", lw=1.0, ls="--")
        coverage = f" IMERG through {last_imerg:%d %b}; IMD through {last_imd:%d %b}"
        ax.text(last_common, ax.get_ylim()[1] * 0.97, coverage, color="#64748b", va="top", fontsize=9)
    ax.text(
        0.995, 0.012,
        "Shading: 10–90% ensemble range  •  IMERG is UTC daily; IMD gauge day follows IMD daily convention",
        transform=ax.transAxes, ha="right", va="bottom", color="#5f6b76", fontsize=8.8,
    )

    output = output_dir / f"{init}_cumulative_fuxi_ecmwf_imerg_imd.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    summary = {
        "FuXi": data.fuxi_mean.iloc[-1], "ECMWF": data.ecmwf_mean.iloc[-1],
        "IMERG": data.imerg_late_cumulative_mm.dropna().iloc[-1],
        "IMD gauge": data.imd_gauge_cumulative_mm.dropna().iloc[-1],
        "IMERG last date": last_imerg, "IMD last date": last_imd,
    }
    print(init, output)
    print(pd.Series(summary).to_string())
    print(metrics.to_string(index=False))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="append", required=True, help="Initialization YYYYMMDD; repeatable")
    args = parser.parse_args()
    for init in args.init:
        plot_case(init)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
