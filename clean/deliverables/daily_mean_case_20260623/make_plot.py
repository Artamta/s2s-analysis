#!/usr/bin/env python3
"""Plot the single 23 June 2026 strict daily-mean FuXi forecast."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


HERE = Path(__file__).resolve().parent
FUXI = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "fuxi_s2s_strict00z_case_20260623_ens50/forecasts/annual2026/20260623.nc"
)
BASELINE = Path(
    "/home/raj.ayush/s2s/s2s_anlysis/final_paper/case-study/data/"
    "20260623_all_india_cumulative_timeseries_with_era5.csv"
)
MASK_FILE = Path(
    "/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/imd_region_masks.nc"
)
REGIONS = [
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
]


def all_india_mask(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    with xr.open_dataset(MASK_FILE) as source:
        masks = source.rename(
            {name: short for name, short in (("latitude", "lat"), ("longitude", "lon")) if name in source.dims}
        ).sortby("lat").sortby("lon")
        union = xr.zeros_like(masks[REGIONS[0]], dtype=bool)
        for name in REGIONS:
            union = union | (masks[name] > 0)
        return (union.astype(float).interp(lat=lat, lon=lon, method="nearest") >= 0.5).load()


def fuxi_series() -> tuple[pd.DataFrame, dict]:
    with xr.open_dataset(FUXI) as ds:
        required = {
            "init_date": "2026-06-23",
            "input_daily_statistic": "daily_mean",
            "benchmark_mode": "strict_information_matched_00utc",
        }
        for key, expected in required.items():
            if str(ds.attrs.get(key)) != expected:
                raise ValueError(f"{key}: expected {expected!r}, got {ds.attrs.get(key)!r}")
        tp = ds["tp"].rename({"latitude": "lat", "longitude": "lon"}) * 24.0
        mask = all_india_mask(tp.lat, tp.lon)
        weights = xr.DataArray(np.cos(np.deg2rad(tp.lat)), coords={"lat": tp.lat}, dims="lat")
        daily = tp.where(mask).weighted(weights).mean(("lat", "lon"), skipna=True)
        cumulative = daily.cumsum("lead_day").load()
        dates = pd.to_datetime(ds.valid_time.values)
        attrs = dict(ds.attrs)

    values = cumulative.values
    return pd.DataFrame(
        {
            "lead_day": np.arange(1, values.shape[1] + 1),
            "valid_date": dates,
            "fuxi_dailymean_mean": values.mean(axis=0),
            "fuxi_dailymean_p10": np.quantile(values, 0.10, axis=0),
            "fuxi_dailymean_p90": np.quantile(values, 0.90, axis=0),
        }
    ), attrs


def main() -> None:
    fuxi, attrs = fuxi_series()
    base = pd.read_csv(BASELINE, parse_dates=["valid_date"])
    keep = [
        "lead_day", "valid_date", "imd_mean", "era5_mean",
        "ecmwf_mean", "ecmwf_p10", "ecmwf_p90",
    ]
    data = base[keep].merge(fuxi, on=["lead_day", "valid_date"], validate="one_to_one")
    data.to_csv(HERE / "20260623_cumulative_dailymean_fuxi_vs_baselines.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.edgecolor": "#263238", "axes.linewidth": 0.9,
    })
    fig, ax = plt.subplots(figsize=(11.6, 6.6), constrained_layout=True)
    x = data.valid_date
    ax.fill_between(x, data.ecmwf_p10, data.ecmwf_p90, color="#f59e0b", alpha=0.13, linewidth=0)
    ax.plot(x, data.ecmwf_mean, color="#e67e00", lw=2.5, label="ECMWF S2S ensemble mean")
    ax.fill_between(x, data.fuxi_dailymean_p10, data.fuxi_dailymean_p90, color="#16a34a", alpha=0.16, linewidth=0)
    ax.plot(x, data.fuxi_dailymean_mean, color="#138a45", lw=3.0, label="FuXi-S2S — daily-mean inputs")
    ax.plot(x, data.imd_mean, color="#1565c0", lw=2.4, label="IMD 1991–2020 climatology")
    ax.plot(x, data.era5_mean, color="#7c3aed", lw=2.3, ls="--", label="ERA5 climatology")

    endpoint_specs = [
        ("FuXi", "fuxi_dailymean_mean", "#138a45", -2),
        ("ECMWF", "ecmwf_mean", "#e67e00", 7),
        ("IMD", "imd_mean", "#1565c0", 1),
        ("ERA5", "era5_mean", "#7c3aed", 0),
    ]
    for label, column, color, yoff in endpoint_specs:
        value = float(data[column].iloc[-1])
        ax.annotate(
            f"{label} {value:.0f} mm", (x.iloc[-1], value), xytext=(9, yoff),
            textcoords="offset points", color=color, weight="bold", fontsize=10,
        )

    ax.set_title("42-Day All-India Cumulative Rainfall — 23 June 2026 Forecast", loc="left", fontsize=17, weight="bold", pad=18)
    ax.text(
        0, 1.015,
        "FuXi inputs: complete 21–22 June UTC daily means  |  Valid: 24 June–4 August 2026",
        transform=ax.transAxes, color="#4b5563", fontsize=10.5,
    )
    ax.set_ylabel("Cumulative rainfall (mm)")
    ax.set_xlabel("Forecast valid date")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#dbe3ea", lw=0.75, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", frameon=False, ncols=2)
    ax.text(
        0.995, 0.015, "Shading: 10–90% ensemble range  •  IMD/ERA5 lines are climatologies, not observations",
        transform=ax.transAxes, ha="right", va="bottom", color="#5f6b76", fontsize=9,
    )
    fig.savefig(HERE / "20260623_cumulative_dailymean_fuxi_vs_imd_era5_ecmwf.png", dpi=240)
    plt.close(fig)

    (HERE / "provenance.txt").write_text(
        "Issue date: 2026-06-23 00 UTC\n"
        "FuXi input days: 2026-06-21 and 2026-06-22\n"
        "FuXi input statistic: complete UTC daily mean from hourly ERA5\n"
        f"FuXi forecast: {FUXI}\n"
        f"FuXi run label: {attrs.get('run_label')}\n"
        f"Baselines: {BASELINE}\n"
        "Important: IMD and ERA5 curves are climatologies, not verifying observations.\n",
        encoding="utf-8",
    )
    print(data.iloc[-1].to_string())


if __name__ == "__main__":
    main()
