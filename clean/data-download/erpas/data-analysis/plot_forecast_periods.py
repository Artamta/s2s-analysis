#!/usr/bin/env python3
"""Plot ERPAS daily and weekly-mean forecasts over Global, Asia, and India domains."""

from __future__ import annotations

import argparse
import sys
from typing import NamedTuple
from pathlib import Path

sys.dont_write_bytecode = True

from analyze_erpas import DEFAULT_DATA_ROOT, open_grib, standardize_latlon

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from pyproj import CRS, Transformer
    from shapely import contains_xy
    from shapely.ops import transform as transform_geometry
    from shapely.ops import unary_union

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "forecast-periods"
DEFAULT_INDIA_SHAPEFILE = Path(
    "/storage/raj.ayush/archive/s2s-forecast-/91/STATE_BOUNDARY.shp"
)

DOMAINS = {
    "Global": (-180, 180, -85, 85),
    "Asia": (20, 150, -10, 60),
    "India": (65, 100, 5, 39),
}

WEEK_WINDOWS = [
    ("Week 1", "days 1-7", slice(0, 7)),
    ("Week 2", "days 8-14", slice(7, 14)),
    ("Week 3", "days 15-21", slice(14, 21)),
    ("Week 4", "days 22-28", slice(21, 28)),
    ("Week 5", "days 29-33", slice(28, 33)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--init-date",
        default="20230705",
        help="Forecast initialization in YYYYMMDD format (default: 20230705)",
    )
    parser.add_argument("--lead-day", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--india-shapefile",
        type=Path,
        default=DEFAULT_INDIA_SHAPEFILE,
        help="Survey of India STATE_BOUNDARY.shp used for India outlines",
    )
    return parser.parse_args()


class IndiaBoundaries(NamedTuple):
    states: list
    outline: object
    source: Path


def load_india_boundaries(path: Path) -> IndiaBoundaries:
    if not HAS_CARTOPY:
        raise ImportError("Cartopy, pyproj, and shapely are required for India boundaries")
    if not path.exists():
        raise FileNotFoundError(f"India boundary shapefile not found: {path}")

    projection_path = path.with_suffix(".prj")
    if not projection_path.exists():
        raise FileNotFoundError(f"Shapefile projection is missing: {projection_path}")

    source_crs = CRS.from_wkt(projection_path.read_text(encoding="utf-8"))
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    states = [
        transform_geometry(transformer.transform, record.geometry).simplify(
            0.01, preserve_topology=True
        )
        for record in shpreader.Reader(str(path)).records()
    ]
    return IndiaBoundaries(states=states, outline=unary_union(states), source=path)


def forecast_path(data_root: Path, init_date: str, product: str) -> Path:
    year = init_date[:4]
    names = {
        "tp": f"APCP_{init_date}.grb",
        "surface_temperature": f"tsfc_{init_date}.grb",
        "geopotential_height": f"gpot_{init_date}.grb",
        "tp_india_0p5": f"Ind_0.5_APCP_{init_date}.grb",
    }
    path = data_root / "forecast" / f"annual{year}" / product / names[product]
    if not path.exists():
        raise FileNotFoundError(f"Missing ERPAS forecast file: {path}")
    return path


def load_fields(data_root: Path, init_date: str) -> dict[str, xr.DataArray]:
    tp_ds = open_grib(forecast_path(data_root, init_date, "tp"))
    temperature_ds = open_grib(
        forecast_path(data_root, init_date, "surface_temperature")
    )
    gh_ds = open_grib(forecast_path(data_root, init_date, "geopotential_height"))
    india_tp_ds = open_grib(forecast_path(data_root, init_date, "tp_india_0p5"))

    tp = standardize_latlon(tp_ds["tp"]).load()
    surface_temperature = standardize_latlon(temperature_ds["t"] - 273.15).load()
    surface_temperature.attrs.update(
        long_name="Surface air temperature", units="degC"
    )
    gh500 = standardize_latlon(gh_ds["gh"].sel(isobaricInhPa=500)).load()
    india_tp = standardize_latlon(india_tp_ds["tp"]).load()

    tp_ds.close()
    temperature_ds.close()
    gh_ds.close()
    india_tp_ds.close()

    return {
        "tp": tp,
        "surface_temperature": surface_temperature,
        "gh500": gh500,
        "india_tp": india_tp,
    }


def subset_domain(
    da: xr.DataArray, extent: tuple[float, float, float, float]
) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = extent
    return da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))


def color_limits(
    fields: list[xr.DataArray], *, start_at_zero: bool = False
) -> tuple[float, float]:
    values = np.concatenate(
        [field.values[np.isfinite(field.values)].ravel() for field in fields]
    )
    if start_at_zero:
        low = 0.0
        high = float(np.nanquantile(values, 0.98))
    else:
        low, high = np.nanquantile(values, [0.02, 0.98]).astype(float)
    if np.isclose(low, high):
        high = low + 1.0
    return low, high


def create_axes(nrows: int, ncols: int, figsize: tuple[float, float]):
    subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    return plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw=subplot_kw,
        constrained_layout=True,
        squeeze=False,
    )


def draw_field(
    ax,
    field: xr.DataArray,
    domain_name: str,
    extent: tuple[float, float, float, float],
    cmap: str,
    vmin: float,
    vmax: float,
    boundaries: IndiaBoundaries,
):
    field = subset_domain(field, extent)
    kwargs = {
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "shading": "auto",
    }
    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()
    mesh = ax.pcolormesh(field.lon, field.lat, field.values, **kwargs)

    if HAS_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6, color="#3F454B")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#6B7280")
        ax.add_geometries(
            [boundaries.outline],
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="white",
            linewidth=2.2 if domain_name == "India" else 1.4,
            zorder=8,
        )
        ax.add_geometries(
            [boundaries.outline],
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#171A1F",
            linewidth=1.15 if domain_name == "India" else 0.75,
            zorder=9,
        )
        if domain_name == "India":
            ax.add_geometries(
                boundaries.states,
                ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="white",
                linewidth=0.85,
                zorder=7,
            )
            ax.add_geometries(
                boundaries.states,
                ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="#343A40",
                linewidth=0.32,
                zorder=10,
            )
        gridlines = ax.gridlines(
            draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--"
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {"size": 7}
        gridlines.ylabel_style = {"size": 7}
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
    return mesh


def plot_day_and_week_domains(
    da: xr.DataArray,
    boundaries: IndiaBoundaries,
    init_date: str,
    lead_day: int,
    variable_title: str,
    units: str,
    cmap: str,
    output_path: Path,
    *,
    start_at_zero: bool = False,
) -> None:
    daily = da.isel(step=lead_day - 1)
    week1 = da.isel(step=slice(0, 7)).mean("step")
    fig, axes = create_axes(2, 3, (17, 9))

    for column, (domain_name, extent) in enumerate(DOMAINS.items()):
        domain_fields = [
            subset_domain(daily, extent),
            subset_domain(week1, extent),
        ]
        vmin, vmax = color_limits(domain_fields, start_at_zero=start_at_zero)
        mesh = draw_field(
            axes[0, column], daily, domain_name, extent, cmap, vmin, vmax, boundaries
        )
        draw_field(
            axes[1, column], week1, domain_name, extent, cmap, vmin, vmax, boundaries
        )
        axes[0, column].set_title(f"{domain_name}: lead day {lead_day}")
        axes[1, column].set_title(f"{domain_name}: week 1 mean (days 1-7)")
        fig.colorbar(
            mesh,
            ax=axes[:, column],
            orientation="horizontal",
            shrink=0.84,
            pad=0.06,
            label=units,
        )

    fig.suptitle(
        f"ERPAS {variable_title} | initialized {init_date}", fontsize=16
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weekly_evolution(
    da: xr.DataArray,
    boundaries: IndiaBoundaries,
    init_date: str,
    variable_title: str,
    domain_name: str,
    extent: tuple[float, float, float, float],
    units: str,
    cmap: str,
    output_path: Path,
    *,
    start_at_zero: bool = False,
) -> None:
    weekly_fields = [da.isel(step=window).mean("step") for _, _, window in WEEK_WINDOWS]
    domain_fields = [subset_domain(field, extent) for field in weekly_fields]
    vmin, vmax = color_limits(domain_fields, start_at_zero=start_at_zero)
    fig, axes = create_axes(1, len(WEEK_WINDOWS), (21, 4.8))

    mesh = None
    for column, ((week_name, day_label, _), field) in enumerate(
        zip(WEEK_WINDOWS, weekly_fields)
    ):
        mesh = draw_field(
            axes[0, column], field, domain_name, extent, cmap, vmin, vmax, boundaries
        )
        axes[0, column].set_title(f"{week_name}\n{day_label}")

    fig.colorbar(
        mesh,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.55,
        pad=0.08,
        label=units,
    )
    fig.suptitle(
        f"ERPAS weekly-mean {variable_title} over {domain_name} | initialized {init_date}",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_india_variable_summary(
    fields: dict[str, xr.DataArray],
    boundaries: IndiaBoundaries,
    init_date: str,
    lead_day: int,
    output_path: Path,
) -> None:
    specifications = [
        ("india_tp", "0.5-degree precipitation", "mm/day", "YlGnBu", True),
        (
            "surface_temperature",
            "Surface air temperature (T2m)",
            "degC",
            "coolwarm",
            False,
        ),
        ("gh500", "500 hPa geopotential height", "gpm", "viridis", False),
    ]
    extent = DOMAINS["India"]
    fig, axes = create_axes(2, 3, (16, 10))

    for column, (key, title, units, cmap, start_at_zero) in enumerate(
        specifications
    ):
        daily = fields[key].isel(step=lead_day - 1)
        week1 = fields[key].isel(step=slice(0, 7)).mean("step")
        domain_fields = [
            subset_domain(daily, extent),
            subset_domain(week1, extent),
        ]
        vmin, vmax = color_limits(domain_fields, start_at_zero=start_at_zero)
        mesh = draw_field(
            axes[0, column], daily, "India", extent, cmap, vmin, vmax, boundaries
        )
        draw_field(
            axes[1, column], week1, "India", extent, cmap, vmin, vmax, boundaries
        )
        axes[0, column].set_title(f"{title}\nLead day {lead_day}")
        axes[1, column].set_title(f"{title}\nWeek 1 mean (days 1-7)")
        fig.colorbar(
            mesh,
            ax=axes[:, column],
            orientation="horizontal",
            shrink=0.8,
            pad=0.06,
            label=units,
        )

    fig.suptitle(
        f"ERPAS India forecast summary | initialized {init_date}", fontsize=16
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def india_country_mean(
    da: xr.DataArray, boundaries: IndiaBoundaries
) -> xr.DataArray:
    """Return a cosine-weighted mean over grid points inside India's outline."""
    india = subset_domain(da, DOMAINS["India"])
    lon_grid, lat_grid = np.meshgrid(india.lon.values, india.lat.values)
    mask = xr.DataArray(
        contains_xy(boundaries.outline, lon_grid, lat_grid),
        coords={"lat": india.lat, "lon": india.lon},
        dims=("lat", "lon"),
    )
    weights = xr.DataArray(
        np.cos(np.deg2rad(india.lat.values)),
        coords={"lat": india.lat},
        dims=("lat",),
    )
    return india.where(mask).weighted(weights).mean(("lat", "lon")).load()


def plot_india_lead_evolution(
    fields: dict[str, xr.DataArray],
    boundaries: IndiaBoundaries,
    init_date: str,
    output_path: Path,
    csv_path: Path,
) -> None:
    """Plot land-only India means across all 33 forecast lead days."""
    specifications = [
        ("india_tp", "Precipitation", "mm/day", "#168AAD"),
        ("surface_temperature", "Surface air temperature (T2m)", "degC", "#D1495B"),
        ("gh500", "500 hPa geopotential height", "gpm", "#4C6E3F"),
    ]
    days = np.arange(1, 34)
    series = {
        key: india_country_mean(fields[key], boundaries)
        for key, _, _, _ in specifications
    }

    table = pd.DataFrame(
        {
            "lead_day": days,
            "precipitation_mm_day": series["india_tp"].values,
            "surface_air_temperature_degc": series[
                "surface_temperature"
            ].values,
            "geopotential_height_500hpa_gpm": series["gh500"].values,
        }
    )
    table.to_csv(csv_path, index=False)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.3), sharex=True)
        for panel, (ax, (key, title, units, color)) in enumerate(
            zip(axes, specifications)
        ):
            values = series[key].values
            for week_index, (_, _, window) in enumerate(WEEK_WINDOWS):
                start_day = window.start + 1
                end_day = window.stop
                if week_index % 2 == 0:
                    ax.axvspan(
                        start_day - 0.5,
                        end_day + 0.5,
                        color="#EEF1F3",
                        zorder=0,
                    )
                weekly_mean = float(np.nanmean(values[window]))
                ax.hlines(
                    weekly_mean,
                    start_day,
                    end_day,
                    color="#22272B",
                    linewidth=3.2,
                    zorder=4,
                )
            ax.plot(
                days,
                values,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
                label="Daily lead",
            )
            ax.set_title(f"{chr(97 + panel)}  {title}", loc="left", fontweight="bold")
            ax.set_ylabel(units)
            ax.grid(axis="y", color="#D7DDE2", linewidth=0.6)
            ax.margins(y=0.16)

        axes[0].plot([], [], color="#22272B", linewidth=3.2, label="Weekly mean")
        axes[0].legend(frameon=False, ncols=2, loc="upper right")
        axes[-1].set_xlim(0.5, 33.5)
        axes[-1].set_xticks([1, 7, 14, 21, 28, 33])
        axes[-1].set_xlabel("Forecast lead day")
        for midpoint, label in zip([4, 11, 18, 25, 31], ["W1", "W2", "W3", "W4", "W5*"]):
            axes[-1].text(
                midpoint,
                -0.24,
                label,
                transform=axes[-1].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#525A61",
            )
        fig.suptitle(
            f"ERPAS India land-area mean through lead day 33 | initialized {init_date}",
            fontsize=15,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.006,
            "Cosine-weighted grid cells inside India boundary | thick segments are weekly means | *W5 = days 29-33",
            ha="center",
            fontsize=8,
            color="#525A61",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.965))
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_notes(
    output_dir: Path,
    init_date: str,
    lead_day: int,
    boundaries: IndiaBoundaries,
) -> None:
    notes = f"""# ERPAS Daily And Weekly Forecast Plots

- Initialization: **{init_date} 00 UTC**
- Daily panels: **lead day {lead_day}**
- Week-1 panels: arithmetic mean of lead days **1-7**
- Weekly-evolution panels: means for days **1-7, 8-14, 15-21, 22-28,
  and 29-33**

## Domains

- Global: 180 W to 180 E, 85 S to 85 N
- Asia: 20 E to 150 E, 10 S to 60 N
- India: 65 E to 100 E, 5 N to 39 N

India state and national outlines use the project shapefile:
`{boundaries.source}`. Its Lambert Conformal Conic geometry is reprojected to
WGS84 before plotting.

## Variables

- Precipitation: mean daily accumulation, in mm/day (`kg m**-2` is
  numerically equivalent to mm of water).
- Surface air temperature (T2m convention): converted from K to degrees C. The
  GRIB is encoded at `surface`, level 0, and each lead is an instantaneous
  daily endpoint rather than a daily mean.
- Z500: geopotential height at 500 hPa, in gpm.

Each domain-comparison image uses one color scale per domain, shared between its
daily and week-1 panels. Each weekly-evolution image uses one common color scale
across all five periods.

`india_country_mean_lead_evolution_{init_date}.png` shows cosine-weighted means
over grid cells inside the India national outline for all 33 daily leads. Thick
line segments show the five forecast-period means; Week 5 contains days 29-33.

`india_day1_week1_all_variables_{init_date}.png` is the India-first summary:
0.5-degree precipitation, surface air temperature, and Z500 for lead day
{lead_day} and week 1.
"""
    (output_dir / "README.md").write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 1 <= args.lead_day <= 33:
        raise ValueError("--lead-day must be between 1 and 33")

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries = load_india_boundaries(args.india_shapefile.resolve())
    fields = load_fields(data_root, args.init_date)

    plot_day_and_week_domains(
        fields["tp"],
        boundaries,
        args.init_date,
        args.lead_day,
        "mean daily precipitation",
        "mm/day",
        "YlGnBu",
        output_dir / f"tp_day{args.lead_day}_week1_domains_{args.init_date}.png",
        start_at_zero=True,
    )
    plot_day_and_week_domains(
        fields["surface_temperature"],
        boundaries,
        args.init_date,
        args.lead_day,
        "surface air temperature (T2m)",
        "degC",
        "coolwarm",
        output_dir
        / f"surface_temperature_day{args.lead_day}_week1_domains_{args.init_date}.png",
    )
    plot_day_and_week_domains(
        fields["gh500"],
        boundaries,
        args.init_date,
        args.lead_day,
        "500 hPa geopotential height",
        "gpm",
        "viridis",
        output_dir / f"gh500_day{args.lead_day}_week1_domains_{args.init_date}.png",
    )

    plot_india_variable_summary(
        fields,
        boundaries,
        args.init_date,
        args.lead_day,
        output_dir
        / f"india_day{args.lead_day}_week1_all_variables_{args.init_date}.png",
    )
    plot_india_lead_evolution(
        fields,
        boundaries,
        args.init_date,
        output_dir / f"india_country_mean_lead_evolution_{args.init_date}.png",
        output_dir / f"india_country_mean_lead_evolution_{args.init_date}.csv",
    )

    plot_weekly_evolution(
        fields["india_tp"],
        boundaries,
        args.init_date,
        "0.5-degree precipitation",
        "India",
        DOMAINS["India"],
        "mm/day",
        "YlGnBu",
        output_dir / f"tp_india_weekly_evolution_{args.init_date}.png",
        start_at_zero=True,
    )
    plot_weekly_evolution(
        fields["surface_temperature"],
        boundaries,
        args.init_date,
        "surface air temperature (T2m)",
        "India",
        DOMAINS["India"],
        "degC",
        "coolwarm",
        output_dir / f"surface_temperature_india_weekly_evolution_{args.init_date}.png",
    )
    plot_weekly_evolution(
        fields["gh500"],
        boundaries,
        args.init_date,
        "500 hPa geopotential height",
        "Asia",
        DOMAINS["Asia"],
        "gpm",
        "viridis",
        output_dir / f"gh500_asia_weekly_evolution_{args.init_date}.png",
    )
    write_notes(output_dir, args.init_date, args.lead_day, boundaries)

    print(f"Initialization: {args.init_date}")
    print(f"Cartopy maps:  {HAS_CARTOPY}")
    print(f"India shape:   {boundaries.source}")
    print(f"Plots:         {output_dir}")


if __name__ == "__main__":
    main()
