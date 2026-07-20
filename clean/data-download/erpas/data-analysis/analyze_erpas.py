#!/usr/bin/env python3
"""Generate a compact ERPAS inventory, metadata report, and diagnostic plots."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "erpas-analysis-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")

import cfgrib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_DATA_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/raw/erpas"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

READ_KEYS = [
    "generatingProcessIdentifier",
    "productDefinitionTemplateNumber",
    "derivedForecast",
    "numberOfForecastsInEnsemble",
]

PRODUCTS = {
    "global_precipitation": "tp",
    "surface_temperature": "surface_temperature",
    "geopotential_height": "geopotential_height",
    "india_precipitation": "tp_india_0p5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-year", default="2023")
    return parser.parse_args()


def parse_date(filename: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(20\d{6})", filename)
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def build_inventory(data_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(data_root.rglob("*.grb")):
        rel = path.relative_to(data_root)
        if rel.parts[0] == "forecast":
            group = "forecast"
            year = rel.parts[1].replace("annual", "")
            product = rel.parts[2]
            init_date = parse_date(path.name)
        else:
            group = "climatology"
            year = "all"
            product = path.stem
            init_date = pd.NaT

        records.append(
            {
                "group": group,
                "year": year,
                "product": product,
                "file": path.name,
                "path": str(path),
                "init_date": init_date,
                "size_mb": path.stat().st_size / 1024**2,
            }
        )

    return pd.DataFrame(records)


def summarize_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    return (
        inventory.groupby(["group", "year", "product"], dropna=False)
        .agg(
            files=("file", "count"),
            size_gib=("size_mb", lambda values: values.sum() / 1024),
            first_init=("init_date", "min"),
            last_init=("init_date", "max"),
        )
        .reset_index()
        .sort_values(["group", "year", "product"])
    )


def first_forecast(forecast_root: Path, year: str, product: str) -> Path:
    candidates = sorted((forecast_root / f"annual{year}" / product).glob("*.grb"))
    if not candidates:
        raise FileNotFoundError(f"No GRIB files in annual{year}/{product}")
    return candidates[0]


def open_grib(path: Path) -> xr.Dataset:
    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": "", "read_keys": READ_KEYS},
    )
    if len(datasets) == 1:
        return datasets[0]
    return xr.merge(datasets, compat="override")


def standardize_latlon(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "latitude" in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords:
        rename["longitude"] = "lon"
    da = da.rename(rename)

    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    return da.sortby("lat")


def metadata_row(label: str, path: Path) -> dict[str, object]:
    ds = open_grib(path)
    variable = next(iter(ds.data_vars))
    da = ds[variable]
    lat = ds.latitude.values
    lon = ds.longitude.values
    member_dims = [
        dim
        for dim in ds.dims
        if dim.lower() in {"number", "member", "ensemble", "realization"}
    ]
    levels = (
        ", ".join(f"{value:g}" for value in ds.isobaricInhPa.values)
        if "isobaricInhPa" in ds.coords
        else ""
    )
    steps = pd.to_timedelta(ds.step.values) if "step" in ds.coords else []

    row = {
        "product": label,
        "sample_file": path.name,
        "variable": variable,
        "long_name": da.attrs.get("long_name", ""),
        "units": da.attrs.get("units", ""),
        "dimensions": str(dict(ds.sizes)),
        "nlat": lat.size,
        "nlon": lon.size,
        "latitude_min": float(lat.min()),
        "latitude_max": float(lat.max()),
        "longitude_min": float(lon.min()),
        "longitude_max": float(lon.max()),
        "latitude_resolution_deg": float(np.abs(np.diff(lat)).mean()),
        "longitude_resolution_deg": float(np.abs(np.diff(lon)).mean()),
        "lead_count": len(steps),
        "first_lead": str(steps.min()) if len(steps) else "",
        "last_lead": str(steps.max()) if len(steps) else "",
        "pressure_levels_hpa": levels,
        "raw_member_dimensions": ", ".join(member_dims) or "none",
        "source_forecasts_in_mean": da.attrs.get(
            "GRIB_numberOfForecastsInEnsemble"
        ),
        "derived_forecast_code": da.attrs.get("GRIB_derivedForecast"),
        "grib_centre": ds.attrs.get("GRIB_centre"),
        "grib_centre_description": ds.attrs.get("GRIB_centreDescription"),
        "generating_process_id": da.attrs.get("GRIB_generatingProcessIdentifier"),
    }
    ds.close()
    return row


def build_product_metadata(
    forecast_root: Path, climatology_root: Path, sample_year: str
) -> pd.DataFrame:
    samples = {
        label: first_forecast(forecast_root, sample_year, product)
        for label, product in PRODUCTS.items()
    }
    rows = [metadata_row(label, path) for label, path in samples.items()]

    climatology_samples = sorted(climatology_root.glob("*/APCP.grb"))
    if climatology_samples:
        rows.append(metadata_row("climatology_precipitation", climatology_samples[0]))
    return pd.DataFrame(rows)


def save_map(
    da: xr.DataArray,
    output_path: Path,
    title: str,
    cmap: str,
    region: tuple[float, float, float, float] | None = None,
) -> None:
    da = standardize_latlon(da).load()
    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        da = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    da.plot(ax=ax, cmap=cmap, cbar_kwargs={"label": da.attrs.get("units", "")})
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    forecast_root: Path, sample_year: str, output_dir: Path
) -> None:
    tp_path = first_forecast(forecast_root, sample_year, "tp")
    tp_ds = open_grib(tp_path)
    save_map(
        tp_ds["tp"].isel(step=0),
        output_dir / "tp_global_day1.png",
        f"ERPAS global precipitation, lead day 1 ({tp_path.name})",
        "YlGnBu",
    )

    tp = standardize_latlon(tp_ds["tp"])
    india = tp.sel(lon=slice(66, 100), lat=slice(5, 38))
    weights = np.cos(np.deg2rad(india.lat))
    lead_series = india.weighted(weights).mean(("lat", "lon")).load()
    lead_days = lead_series.step / np.timedelta64(1, "D")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(lead_days, lead_series, marker="o", linewidth=1.5)
    ax.set_title("ERPAS India-area mean precipitation by lead day")
    ax.set_xlabel("Lead day")
    ax.set_ylabel("Precipitation (mm/day)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "india_tp_lead_series.png", bbox_inches="tight")
    plt.close(fig)
    tp_ds.close()

    india_path = first_forecast(forecast_root, sample_year, "tp_india_0p5")
    india_ds = open_grib(india_path)
    save_map(
        india_ds["tp"].isel(step=0),
        output_dir / "tp_india_day1.png",
        f"ERPAS India precipitation (0.5 degree), lead day 1 ({india_path.name})",
        "YlGnBu",
        region=(66, 100, 5, 38),
    )
    india_ds.close()

    gh_path = first_forecast(forecast_root, sample_year, "geopotential_height")
    gh_ds = open_grib(gh_path)
    save_map(
        gh_ds["gh"].sel(isobaricInhPa=500).isel(step=0),
        output_dir / "gh500_day1.png",
        f"ERPAS 500 hPa geopotential height, lead day 1 ({gh_path.name})",
        "viridis",
    )
    gh_ds.close()


def save_file_count_plot(summary: pd.DataFrame, output_path: Path) -> None:
    forecast = summary[summary.group == "forecast"].copy()
    forecast["label"] = forecast["year"] + " / " + forecast["product"]
    forecast = forecast.sort_values(["year", "product"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(forecast.label, forecast.files, color="#287271")
    ax.set_title("ERPAS forecast file counts by year and product")
    ax.set_xlabel("GRIB files")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_dataset_overview(
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a communication-ready overview of archive coverage and structure."""
    colors = {
        "geopotential_height": "#4C78A8",
        "surface_temperature": "#E45756",
        "tp": "#54A24B",
        "tp_india_0p5": "#F2B134",
    }
    labels = {
        "geopotential_height": "Geopotential height",
        "surface_temperature": "Surface air temperature",
        "tp": "Global precipitation",
        "tp_india_0p5": "0.5 degree precipitation",
    }
    product_order = list(colors)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    ):
        fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1), height_ratios=(1, 1))
        ax_counts = fig.add_subplot(grid[0, 0])
        ax_timeline = fig.add_subplot(grid[1, 0])
        ax_grids = fig.add_subplot(grid[0, 1])
        ax_facts = fig.add_subplot(grid[1, 1])

        forecast_summary = summary[summary["group"] == "forecast"]
        years = sorted(forecast_summary["year"].unique())
        x = np.arange(len(years))
        width = 0.19
        for index, product in enumerate(product_order):
            product_counts = (
                forecast_summary[forecast_summary["product"] == product]
                .set_index("year")
                ["files"].reindex(years)
            )
            positions = x + (index - 1.5) * width
            bars = ax_counts.bar(
                positions,
                product_counts,
                width,
                color=colors[product],
                label=labels[product],
            )
            ax_counts.bar_label(bars, fontsize=7, padding=2)
        ax_counts.set_xticks(x, years)
        ax_counts.set_ylim(0, max(forecast_summary["files"]) + 10)
        ax_counts.set_ylabel("GRIB files")
        ax_counts.set_title("a  Forecast archive coverage", loc="left")
        ax_counts.grid(axis="y", color="#D9DEE3", linewidth=0.6)
        ax_counts.legend(frameon=False, fontsize=7, ncols=2, loc="upper right")

        forecast = inventory[inventory["group"] == "forecast"]
        for index, product in enumerate(product_order):
            dates = (
                forecast[forecast["product"] == product]
                ["init_date"].dropna()
                .drop_duplicates()
                .sort_values()
            )
            ax_timeline.scatter(
                dates,
                np.full(len(dates), index),
                s=12,
                color=colors[product],
                edgecolor="none",
            )
        ax_timeline.set_yticks(
            range(len(product_order)), [labels[product] for product in product_order]
        )
        ax_timeline.set_ylim(-0.65, len(product_order) - 0.35)
        ax_timeline.invert_yaxis()
        ax_timeline.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax_timeline.grid(axis="x", color="#D9DEE3", linewidth=0.6)
        ax_timeline.set_title("b  Weekly Wednesday initializations at 00 UTC", loc="left")

        dated_metadata = metadata[
            metadata["product"] != "climatology_precipitation"
        ].set_index("product")
        grid_products = [
            "india_precipitation",
            "global_precipitation",
            "surface_temperature",
            "geopotential_height",
        ]
        grid_labels = [
            "0.5 degree precipitation",
            "Global precipitation",
            "Surface air temperature",
            "Geopotential height",
        ]
        grid_colors = [
            colors["tp_india_0p5"],
            colors["tp"],
            colors["surface_temperature"],
            colors["geopotential_height"],
        ]
        cells = [
            dated_metadata.loc[product, "nlat"]
            * dated_metadata.loc[product, "nlon"]
            / 1000
            for product in grid_products
        ]
        bars = ax_grids.barh(grid_labels, cells, color=grid_colors, height=0.62)
        ax_grids.invert_yaxis()
        ax_grids.set_xlabel("Horizontal grid cells (thousands)")
        ax_grids.set_title("c  Native grids in this archive", loc="left")
        ax_grids.grid(axis="x", color="#D9DEE3", linewidth=0.6)
        for bar, product in zip(bars, grid_products):
            row = dated_metadata.loc[product]
            annotation = (
                f"{int(row.nlat)} x {int(row.nlon)}  |  "
                f"{row.latitude_resolution_deg:g} degree"
            )
            ax_grids.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                annotation,
                va="center",
                fontsize=8,
            )
        ax_grids.set_xlim(0, max(cells) * 1.55)

        ax_facts.axis("off")
        ax_facts.set_title("d  How to interpret the fields", loc="left", pad=8)
        facts = [
            ("Forecast horizon", "33 daily leads, day 1 through day 33"),
            ("Forecast representation", "Precomputed unweighted mean of 4 forecasts"),
            ("Raw ensemble members", "Not present; spread and probabilities unavailable"),
            ("Provider climatology", "144 MMDD files per variable; mean of 20 forecasts"),
            ("Precipitation", "24-hour accumulation; kg m-2 equals mm water"),
            (
                "Surface air temperature",
                "T2m convention; instantaneous K; GRIB surface level 0",
            ),
            ("Geopotential height", "gpm at 1000, 925, 850, 700, 500, 300, 200 hPa"),
            ("GRIB provenance", "NCEP centre kwbc; generating process identifier 96"),
        ]
        y = 0.96
        for heading, detail in facts:
            ax_facts.text(
                0.0,
                y,
                heading,
                transform=ax_facts.transAxes,
                fontsize=9,
                fontweight="bold",
                va="top",
                color="#20252B",
            )
            ax_facts.text(
                0.0,
                y - 0.052,
                detail,
                transform=ax_facts.transAxes,
                fontsize=8.5,
                va="top",
                color="#505860",
                wrap=True,
            )
            y -= 0.118

        fig.suptitle(
            "ERPAS Forecast Archive | 2023-2025 Snapshot",
            fontsize=17,
            fontweight="bold",
            x=0.01,
            ha="left",
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def format_markdown_table(frame: pd.DataFrame) -> str:
    printable = frame.copy().fillna("")
    for column in printable.columns:
        printable[column] = printable[column].map(str)
    header = "| " + " | ".join(printable.columns) + " |"
    divider = "| " + " | ".join("---" for _ in printable.columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "/") for value in row) + " |"
        for row in printable.to_numpy().tolist()
    ]
    return "\n".join([header, divider, *rows])


def write_report(
    output_path: Path,
    data_root: Path,
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    forecast = inventory[inventory.group == "forecast"]
    climatology = inventory[inventory.group == "climatology"]
    forecast_dates = (
        forecast.dropna(subset=["init_date"])
        .groupby("year")
        .init_date.nunique()
        .to_dict()
    )
    dated = metadata[metadata["product"] != "climatology_precipitation"]
    source_counts = sorted(
        int(value) for value in dated.source_forecasts_in_mean.dropna().unique()
    )
    source_counts_text = ", ".join(map(str, source_counts))
    forecast_summary = summary[summary.group == "forecast"]
    expected_by_year = forecast_summary.groupby("year").files.max().to_dict()
    shortfalls = []
    for row in forecast_summary.itertuples(index=False):
        expected = int(expected_by_year[row.year])
        if int(row.files) < expected:
            year_dates = set(
                forecast.loc[forecast["year"] == row.year, "init_date"].dropna()
            )
            product_dates = set(
                forecast.loc[
                    (forecast["year"] == row.year)
                    & (forecast["product"] == row.product),
                    "init_date",
                ].dropna()
            )
            missing_dates = ", ".join(
                date.strftime("%Y-%m-%d") for date in sorted(year_dates - product_dates)
            )
            missing_text = (
                f"; missing initialization {missing_dates}" if missing_dates else ""
            )
            shortfalls.append(
                f"{row.year}/{row.product}: {int(row.files)} files "
                f"(expected {expected}{missing_text})"
            )
    shortfall_text = "; ".join(shortfalls) or "none"
    climatology_metadata = metadata[
        metadata["product"] == "climatology_precipitation"
    ]
    climatology_source_count = (
        int(climatology_metadata.iloc[0].source_forecasts_in_mean)
        if not climatology_metadata.empty
        else None
    )

    compact_inventory = summary[
        ["group", "year", "product", "files", "size_gib"]
    ].copy()
    compact_inventory["size_gib"] = compact_inventory["size_gib"].map(
        lambda value: f"{value:.3f}"
    )
    compact_metadata = metadata[
        [
            "product",
            "variable",
            "units",
            "nlat",
            "nlon",
            "latitude_resolution_deg",
            "longitude_resolution_deg",
            "lead_count",
            "pressure_levels_hpa",
            "raw_member_dimensions",
            "source_forecasts_in_mean",
        ]
    ].copy()

    report = f"""# ERPAS Data Analysis Report

## Dataset Summary

- Data root: `{data_root}`
- GRIB files: **{len(inventory):,}**
- Total size: **{inventory.size_mb.sum() / 1024:.2f} GiB**
- Forecast files: **{len(forecast):,}**
- Climatology files: **{len(climatology):,}**
- Unique forecast initializations by year: **{forecast_dates}**

## Main Findings

- Dated forecasts have **33 daily lead steps**, from day 1 to day 33.
- Global precipitation, surface air temperature, and geopotential height use a
  **1 x 1 degree, 181 x 360** grid.
- The broad regional precipitation product uses a
  **0.5 x 0.5 degree, 161 x 241** grid spanning 30-150 E and 30 S-50 N.
- Variables are `tp` (precipitation), `t` (surface air temperature), and `gh`
  (geopotential height).
- Geopotential-height levels are **1000, 925, 850, 700, 500, 300, and 200 hPa**.
- No raw member dimension is present. The dated files are GRIB-derived
  unweighted means with source-forecast count **{source_counts_text}**.
- The sampled climatology precipitation is also a derived mean and reports
  **{climatology_source_count}** source forecasts.
- Forecast-product coverage shortfalls: **{shortfall_text}**.
- The files are stored in the ERPAS provider tree but carry NCEP/GFS GRIB
  identifiers (`centre=kwbc`, process ID 96), consistent with ERPAS using NCEP
  model components.

## Interpretation Cautions

- Precipitation units `kg m**-2` are numerically equivalent to millimetres of
  water. The product is a daily accumulation.
- The `t` field is treated as surface air temperature/T2m under the project
  convention and domain-expert guidance. Its GRIB level is encoded as
  `surface`, level 0, rather than an explicit `heightAboveGround=2` coordinate.
- ERPAS temperature is instantaneous at each daily endpoint. Compare it with
  reference T2m sampled at the same valid hour, not a daily-mean statistic.
- Individual ensemble members cannot be analysed because only their derived
  mean is present in this download.
- IMD/ERA5 comparison requires matching the same valid date and daily period,
  converting units, and regridding one field before bias/RMSE/correlation.

## Inventory

{format_markdown_table(compact_inventory)}

## Product Metadata

{format_markdown_table(compact_metadata)}

## Figures

- `dataset_overview.png`
- `file_counts.png`
- `tp_global_day1.png`
- `tp_india_day1.png`
- `gh500_day1.png`
- `india_tp_lead_series.png`

## Daily And Weekly Forecast Maps

See [`forecast-periods/README.md`](forecast-periods/README.md) for labeled
Global, Asia, and India day-1/week-1 comparisons and week-by-week evolution
plots. Generate another initialization with `plot_forecast_periods.py`.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    forecast_root = data_root / "forecast"
    climatology_root = data_root / "reforecast" / "climatology"

    if not data_root.exists():
        raise FileNotFoundError(f"ERPAS data root does not exist: {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(data_root)
    if inventory.empty:
        raise RuntimeError(f"No *.grb files found under {data_root}")

    summary = summarize_inventory(inventory)
    metadata = build_product_metadata(
        forecast_root, climatology_root, args.sample_year
    )

    summary.to_csv(output_dir / "inventory_summary.csv", index=False)
    metadata.to_csv(output_dir / "product_metadata.csv", index=False)
    save_file_count_plot(summary, output_dir / "file_counts.png")
    save_dataset_overview(
        inventory, summary, metadata, output_dir / "dataset_overview.png"
    )
    generate_plots(forecast_root, args.sample_year, output_dir)
    write_report(
        output_dir / "analysis_report.md",
        data_root,
        inventory,
        summary,
        metadata,
    )

    print(f"ERPAS files: {len(inventory):,}")
    print(f"ERPAS size:  {inventory.size_mb.sum() / 1024:.2f} GiB")
    print(f"Report:      {output_dir / 'analysis_report.md'}")
    print(f"Outputs:     {output_dir}")


if __name__ == "__main__":
    main()
