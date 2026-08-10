from __future__ import annotations

import datetime as dt
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "india_s2s_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


DISPLAY_NAMES = {
    "ecmwf": "ECMWF",
    "ukmo": "UKMO",
    "cma": "CMA",
    "ncep": "NCEP",
    "cnrm": "CNRM",
    "fuxi_s2s": "FuXi-S2S",
    "dlesym_v0": "DLESyM-v0",
    "dlesym_v1": "DLESyM-v1",
    "neuralgcm": "NeuralGCM",
    "fcn3": "FCN3",
    "erpas": "ERPAS",
}


def record_label(record: dict[str, Any], *, include_grid: bool = False) -> str:
    label = DISPLAY_NAMES.get(record["model"], record["model"])
    if record["model"] == "erpas":
        label += " 0.5° sensitivity" if "india_0p5_sensitivity" in record["experiment_id"] else " global"
    if include_grid and record["grid"] == "source_native_india":
        label += " [native]"
    return label


def valid_times_intersection(arrays: Iterable[np.ndarray]) -> np.ndarray:
    values = [np.asarray(item).astype("datetime64[ns]") for item in arrays]
    if not values:
        return np.array([], dtype="datetime64[ns]")
    result = values[0]
    for item in values[1:]:
        result = np.intersect1d(result, item)
    return np.sort(result)


def weighted_spatial_mean(field: xr.DataArray, weight: xr.DataArray) -> xr.DataArray:
    usable = weight.where(np.isfinite(field))
    return (field * usable).sum(("latitude", "longitude")) / usable.sum(
        ("latitude", "longitude")
    )


def _records(catalog: dict[str, Any], *, variable: str, grid: str = "common_1p5") -> list[dict[str, Any]]:
    records = [
        item for item in catalog["records"]
        if item["variable"] == variable and item["grid"] == grid
    ]
    return sorted(records, key=record_label)


def _standard_tp(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in records if "india_0p5_sensitivity" not in item["experiment_id"]]


def _load(record: dict[str, Any], name: str, **indexers: int) -> xr.DataArray:
    with xr.open_zarr(record["store"], consolidated=True) as ds:
        return ds[name].isel(**indexers).load()


def _week_period_label(record: dict[str, Any]) -> str:
    with xr.open_zarr(record["store"], consolidated=True) as ds:
        dates = ds.valid_time.isel(lead_day=slice(0, 7)).values.astype("datetime64[D]")
    start = np.datetime_as_string(dates[0], unit="D")
    end = np.datetime_as_string(dates[-1], unit="D")
    return f"{start}–{end[5:]}"


def _support_contour(ax: plt.Axes, support: xr.Dataset, level: float = 0.01) -> None:
    ax.contour(
        support.longitude,
        support.latitude,
        support.india_fraction,
        levels=[level],
        colors="black",
        linewidths=0.65,
    )


def _map(
    ax: plt.Axes,
    field: xr.DataArray,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any = None,
    support: xr.Dataset | None = None,
) -> Any:
    mesh = ax.pcolormesh(
        field.longitude,
        field.latitude,
        field,
        shading="auto",
        cmap=cmap,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        norm=norm,
        rasterized=True,
    )
    if support is not None and field.sizes.get("latitude") == support.sizes["latitude"]:
        if np.array_equal(field.longitude.values, support.longitude.values):
            _support_contour(ax, support)
    ax.set_xlim(float(field.longitude.min()), float(field.longitude.max()))
    ax.set_ylim(float(field.latitude.min()), float(field.latitude.max()))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.tick_params(labelsize=7)
    return mesh


def _shared_limits(fields: list[xr.DataArray], low: float, high: float) -> tuple[float, float]:
    values = np.concatenate([item.values[np.isfinite(item.values)] for item in fields])
    return float(np.quantile(values, low)), float(np.quantile(values, high))


def _save(fig: plt.Figure, output: Path, files: list[str], *, pdf: bool = True) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    files.append(str(output))
    if pdf:
        pdf_path = output.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        files.append(str(pdf_path))
    plt.close(fig)


def _panel_shape(count: int, columns: int = 4) -> tuple[int, int]:
    return math.ceil(count / columns), columns


def _plot_spatial_support(support: xr.Dataset, output: Path, files: list[str]) -> None:
    names = [
        "india_fraction",
        "india_area_weight_km2",
        "northwest_india_fraction",
        "central_india_fraction",
        "south_peninsula_fraction",
        "east_northeast_india_fraction",
    ]
    titles = [
        "All-India fractional support",
        "Canonical India area weight",
        "Northwest India fraction",
        "Central India fraction",
        "South Peninsula fraction",
        "East & Northeast India fraction",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, name, title in zip(axes.flat, names, titles):
        field = support[name]
        mesh = _map(
            ax,
            field,
            cmap="viridis" if name == "india_area_weight_km2" else "YlGn",
            vmin=0,
            vmax=float(field.max()),
            support=support if name == "india_area_weight_km2" else None,
        )
        ax.set_title(title, fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=0.78, label=field.attrs.get("units", "1"))
    fig.suptitle("Canonical IMD-region support on india_1p5_27x27_v1", fontsize=14)
    _save(fig, output / "00_spatial_support.png", files)


def _plot_shared_maps(
    records: list[dict[str, Any]],
    variable_name: str,
    output_file: Path,
    files: list[str],
    support: xr.Dataset,
    *,
    title: str,
    cmap: str,
    unit: str,
    quantiles: tuple[float, float] = (0.01, 0.99),
    floor_zero: bool = False,
) -> tuple[float, float]:
    fields = [_load(item, variable_name, init=0, lead_week=0) for item in records]
    vmin, vmax = _shared_limits(fields, *quantiles)
    if floor_zero:
        vmin = 0.0
    rows, columns = _panel_shape(len(records))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    mesh = None
    for ax, record, field in zip(axes_array, records, fields):
        mesh = _map(ax, field, cmap=cmap, vmin=vmin, vmax=vmax, support=support)
        ax.set_title(f"{record_label(record)}\n{_week_period_label(record)}", fontsize=9)
    for ax in axes_array[len(records):]:
        ax.set_visible(False)
    if mesh is not None:
        fig.colorbar(mesh, ax=list(axes_array[:len(records)]), shrink=0.82, label=unit)
    fig.suptitle(title, fontsize=14)
    _save(fig, output_file, files)
    return vmin, vmax


def _plot_matched_tp_total(
    records: list[dict[str, Any]], output: Path, files: list[str], support: xr.Dataset
) -> tuple[list[str], tuple[float, float]]:
    datasets = []
    try:
        for record in records:
            datasets.append(xr.open_zarr(record["store"], consolidated=True))
        common = valid_times_intersection([ds.valid_time.values for ds in datasets])
        fields = []
        for ds in datasets:
            indices = [int(np.where(ds.valid_time.values == date)[0][0]) for date in common]
            fields.append(ds.ensemble_mean.isel(init=0, lead_day=indices).sum("lead_day").load())
        _, vmax = _shared_limits(fields, 0.0, 0.99)
        rows, columns = _panel_shape(len(records), columns=3)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), constrained_layout=True)
        axes_array = np.atleast_1d(axes).ravel()
        mesh = None
        for ax, record, field in zip(axes_array, records, fields):
            mesh = _map(ax, field, cmap="YlGnBu", vmin=0, vmax=vmax, support=support)
            ax.set_title(record_label(record), fontsize=9)
        for ax in axes_array[len(records):]:
            ax.set_visible(False)
        if mesh is not None:
            fig.colorbar(mesh, ax=list(axes_array[:len(records)]), shrink=0.82, label="mm per matched period")
        date_strings = [np.datetime_as_string(date, unit="D") for date in common]
        fig.suptitle(
            f"Ensemble-mean TP total · exactly matched {len(common)}-day valid period "
            f"{date_strings[0]}–{date_strings[-1]}",
            fontsize=14,
        )
        _save(fig, output / "01b_tp_matched_6day_total_common.png", files)
        return date_strings, (0.0, vmax)
    finally:
        for ds in datasets:
            ds.close()


def _plot_noncomparable_temperature(
    catalog: dict[str, Any], output: Path, files: list[str], support: xr.Dataset
) -> None:
    records = _records(catalog, variable="t2m_proxy") + _records(catalog, variable="tsfc")
    records = [item for item in records if item["grid"] == "common_1p5"]
    fields = [_load(item, "ensemble_mean_weekly", init=0, lead_week=0) for item in records]
    vmin, vmax = _shared_limits(fields, 0.01, 0.99)
    fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for ax, record, field in zip(axes_array, records, fields):
        mesh = _map(ax, field, cmap="coolwarm", vmin=vmin, vmax=vmax, support=support)
        ax.set_title(
            f"{record_label(record)}\n{record['temporal_statistic']}\n{_week_period_label(record)}",
            fontsize=9,
        )
        fig.colorbar(mesh, ax=ax, shrink=0.8, label="°C")
    fig.suptitle(
        "Non-comparable temperature diagnostics — do not pool with daily-mean T2M",
        fontsize=13,
    )
    _save(fig, output / "03_temperature_noncomparable.png", files)


def _plot_erpas_remap(
    catalog: dict[str, Any], output: Path, files: list[str], support: xr.Dataset
) -> None:
    records = [item for item in catalog["records"] if item["model"] == "erpas" and item["variable"] == "tp"]
    records = sorted(records, key=lambda item: ("sensitivity" in item["experiment_id"], item["grid"]))
    fields = [_load(item, "ensemble_mean_weekly_total", init=0, lead_week=0) for item in records]
    _, vmax = _shared_limits(fields, 0.0, 0.99)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, record, field in zip(axes.flat, records, fields):
        mesh = _map(ax, field, cmap="YlGnBu", vmin=0, vmax=vmax, support=support)
        ax.set_title(
            f"{record_label(record, include_grid=True)}\n{_week_period_label(record)}",
            fontsize=9,
        )
        fig.colorbar(mesh, ax=ax, shrink=0.8, label="mm week-1")
    fig.suptitle("ERPAS native-to-common precipitation remapping check", fontsize=14)
    _save(fig, output / "04_erpas_tp_native_vs_common.png", files)


def _plot_spread(
    records: list[dict[str, Any]], output: Path, files: list[str], support: xr.Dataset
) -> None:
    records = [item for item in records if item["distribution_representation"] == "members"]
    fields = [_load(item, "ensemble_std_weekly", init=0, lead_week=0) for item in records]
    _, vmax = _shared_limits(fields, 0.0, 0.99)
    rows, columns = _panel_shape(len(records))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    mesh = None
    for ax, record, field in zip(axes_array, records, fields):
        with xr.open_zarr(record["store"], consolidated=True) as ds:
            members = int(ds.member_available.isel(init=0).sum())
        mesh = _map(ax, field, cmap="magma", vmin=0, vmax=vmax, support=support)
        ax.set_title(
            f"{record_label(record)} (n={members})\n{_week_period_label(record)}",
            fontsize=9,
        )
    for ax in axes_array[len(records):]:
        ax.set_visible(False)
    if mesh is not None:
        fig.colorbar(mesh, ax=list(axes_array[:len(records)]), shrink=0.82, label="mm day-1")
    fig.suptitle("Week-1 precipitation ensemble spread (member weekly means)", fontsize=14)
    _save(fig, output / "05_tp_ensemble_spread.png", files)


def _plot_member_inventory(catalog: dict[str, Any], output: Path, files: list[str]) -> list[dict[str, Any]]:
    records = [item for item in catalog["records"] if item["grid"] == "common_1p5"]
    rows = []
    for record in records:
        with xr.open_zarr(record["store"], consolidated=True) as ds:
            stored = int(ds.member_available.isel(init=0).sum())
            represented = int(ds.source_ensemble_size.isel(init=0)) if "source_ensemble_size" in ds else stored
        rows.append(
            {
                "label": f"{record_label(record)} · {record['variable']}",
                "variable": record["variable"],
                "stored_members": stored,
                "represented_members": represented,
                "representation": record["distribution_representation"],
            }
        )
    rows.sort(key=lambda item: (item["variable"], item["label"]))
    colors = {"tp": "#2b8cbe", "t2m": "#d95f0e", "t2m_proxy": "#756bb1", "tsfc": "#31a354", "gh": "#636363"}
    fig, ax = plt.subplots(figsize=(11, max(6, 0.34 * len(rows))), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(y, [item["stored_members"] for item in rows], color=[colors[item["variable"]] for item in rows], alpha=0.82, label="stored fields")
    mean_only = [index for index, item in enumerate(rows) if item["representation"] == "mean_only"]
    ax.scatter(
        [rows[index]["represented_members"] for index in mean_only],
        mean_only,
        marker="D",
        color="black",
        s=24,
        label="source forecasts represented by supplied mean",
        zorder=3,
    )
    ax.set_yticks(y, [item["label"] for item in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Pilot member and distribution-representation audit")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output / "06_member_inventory.png", files)
    return rows


def _plot_negative_tp(
    records: list[dict[str, Any]], output: Path, files: list[str], support: xr.Dataset
) -> list[dict[str, Any]]:
    diagnostics = []
    maps = []
    kept = []
    for record in records:
        with xr.open_zarr(record["store"], consolidated=True) as ds:
            forecast = ds.forecast.load()
        count = (forecast < 0).sum(("init", "member", "lead_day"))
        total = int(count.sum())
        minimum = float(forecast.min())
        diagnostics.append({"label": record_label(record), "count": total, "minimum_mm_day": minimum})
        if total:
            kept.append(record)
            maps.append(count)
    columns = min(3, max(1, len(kept)))
    rows = math.ceil(max(1, len(kept)) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.8 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    if maps:
        vmax = max(float(item.max()) for item in maps)
        for ax, record, field in zip(axes_array, kept, maps):
            mesh = _map(
                ax,
                field,
                cmap="Reds",
                norm=LogNorm(vmin=1, vmax=max(1, vmax)),
                support=support,
            )
            present = field.values > 0
            yy, xx = np.where(present)
            ax.scatter(
                field.longitude.values[xx],
                field.latitude.values[yy],
                s=5,
                facecolors="none",
                edgecolors="black",
                linewidths=0.35,
            )
            item = next(value for value in diagnostics if value["label"] == record_label(record))
            ax.set_title(
                f"{record_label(record)}\ncount={item['count']:,}; min={item['minimum_mm_day']:.6g}",
                fontsize=9,
            )
            fig.colorbar(mesh, ax=ax, shrink=0.8, label="negative member-days")
    else:
        axes_array[0].text(0.5, 0.5, "No negative TP values", ha="center", va="center")
    for ax in axes_array[len(maps):]:
        ax.set_visible(False)
    fig.suptitle("Preserved negative daily-precipitation increments", fontsize=14)
    _save(fig, output / "07_negative_tp_diagnostics.png", files)
    return diagnostics


def _plot_gh(catalog: dict[str, Any], output: Path, files: list[str], support: xr.Dataset) -> None:
    record = next(
        item for item in catalog["records"]
        if item["model"] == "erpas" and item["variable"] == "gh" and item["grid"] == "common_1p5"
    )
    field = _load(record, "ensemble_mean_weekly", init=0, lead_week=0)
    levels = field.pressure_hpa.values
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for ax, level in zip(axes.flat, levels):
        layer = field.sel(pressure_hpa=level)
        vmin, vmax = _shared_limits([layer], 0.01, 0.99)
        mesh = _map(ax, layer, cmap="viridis", vmin=vmin, vmax=vmax, support=support)
        ax.set_title(f"{int(level)} hPa", fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=0.72, label="gpm")
    for ax in axes.flat[len(levels):]:
        ax.set_visible(False)
    fig.suptitle("ERPAS week-1 mean geopotential-height samples", fontsize=14)
    _save(fig, output / "08_erpas_gh_pressure_levels.png", files)


def _representative_records(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        item for item in catalog["records"]
        if item["grid"] == "common_1p5" and "india_0p5_sensitivity" not in item["experiment_id"]
    ]
    result = []
    for model in DISPLAY_NAMES:
        model_records = [item for item in candidates if item["model"] == model]
        if not model_records:
            continue
        result.append(next((item for item in model_records if item["variable"] == "tp"), model_records[0]))
    return result


def _plot_valid_time_alignment(catalog: dict[str, Any], output: Path, files: list[str]) -> dict[str, list[str]]:
    records = _representative_records(catalog)
    date_arrays = []
    by_label: dict[str, np.ndarray] = {}
    for record in records:
        with xr.open_zarr(record["store"], consolidated=True) as ds:
            dates = ds.valid_time.values.astype("datetime64[D]")
        by_label[record_label(record)] = dates
        date_arrays.append(dates)
    union = np.unique(np.concatenate(date_arrays))
    matrix = np.array([[date in set(values) for date in union] for values in by_label.values()], dtype=int)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(by_label)), list(by_label), fontsize=8)
    ax.set_xticks(np.arange(len(union)), [np.datetime_as_string(date, unit="D") for date in union], rotation=45, ha="right")
    ax.set_xlabel("Valid time (UTC period end)")
    ax.set_title("Pilot valid-time availability — comparisons must join by valid time")
    fig.colorbar(image, ax=ax, ticks=[0, 1], label="available")
    _save(fig, output / "09_valid_time_alignment.png", files)
    return {key: [np.datetime_as_string(item, unit="D") for item in value] for key, value in by_label.items()}


def _plot_tp_daily_matched(
    records: list[dict[str, Any]], output: Path, files: list[str], support: xr.Dataset
) -> list[str]:
    arrays = []
    datasets = []
    try:
        for record in records:
            ds = xr.open_zarr(record["store"], consolidated=True)
            datasets.append(ds)
            arrays.append(ds.valid_time.values)
        common = valid_times_intersection(arrays)
        all_fields = []
        for ds in datasets:
            for date in common:
                index = int(np.where(ds.valid_time.values == date)[0][0])
                all_fields.append(ds.ensemble_mean.isel(init=0, lead_day=index).load())
        _, vmax = _shared_limits(all_fields, 0.0, 0.99)
        daily_dir = output / "daily_tp_matched_valid_time"
        result_dates = []
        for date in common:
            rows, columns = _panel_shape(len(records), columns=3)
            fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), constrained_layout=True)
            axes_array = np.atleast_1d(axes).ravel()
            mesh = None
            for ax, record, ds in zip(axes_array, records, datasets):
                index = int(np.where(ds.valid_time.values == date)[0][0])
                field = ds.ensemble_mean.isel(init=0, lead_day=index).load()
                mesh = _map(ax, field, cmap="YlGnBu", vmin=0, vmax=vmax, support=support)
                lead = int(ds.lead_day.values[index])
                ax.set_title(f"{record_label(record)} · lead {lead}", fontsize=9)
            for ax in axes_array[len(records):]:
                ax.set_visible(False)
            if mesh is not None:
                fig.colorbar(mesh, ax=list(axes_array[:len(records)]), shrink=0.82, label="mm day-1")
            date_text = np.datetime_as_string(date, unit="D")
            fig.suptitle(f"Daily ensemble-mean TP · matched valid time {date_text}", fontsize=14)
            _save(fig, daily_dir / f"tp_valid_{date_text}.png", files, pdf=False)
            result_dates.append(date_text)
        return result_dates
    finally:
        for ds in datasets:
            ds.close()


def _plot_tp_area_series(
    records: list[dict[str, Any]], output: Path, files: list[str], support: xr.Dataset
) -> None:
    weight = support.india_area_weight_km2
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for record in records:
        with xr.open_zarr(record["store"], consolidated=True) as ds:
            series = weighted_spatial_mean(ds.ensemble_mean.isel(init=0), weight).load()
            dates = ds.valid_time.values
        ax.plot(dates, series, marker="o", markersize=3, linewidth=1.3, label=record_label(record))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("India-area-weighted TP (mm day-1)")
    ax.set_xlabel("Valid time (UTC period end)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("Daily TP magnitude and temporal-alignment diagnostic")
    _save(fig, output / "10_tp_india_area_mean_timeseries.png", files)


def _write_index(output: Path, summary: dict[str, Any]) -> Path:
    index = output / "QC_PLOT_INDEX.md"
    pngs = [Path(item) for item in summary["files"] if item.endswith(".png")]
    lines = [
        "# Pilot visual-QC plot index",
        "",
        f"Catalog ID: `{summary['catalog_id']}`",
        "",
        "These figures verify preprocessing, metadata, valid-time alignment, grids, and plausible values. They are not model-skill figures.",
        "",
        "## Required cautions",
        "",
        "- NCEP `t2m_proxy` and ERPAS instantaneous `tsfc` are not daily-mean T2M.",
        "- ERPAS is a provider-precomputed mean and is not a probabilistic one-member ensemble.",
        "- Small negative accumulated-field increments are preserved and shown explicitly.",
        "- Cross-model daily panels use the intersection of `valid_time`, not equal lead indices.",
        "",
        "## Figures",
        "",
    ]
    for path in pngs:
        lines.append(f"- [{path.name}]({path.relative_to(output)})")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def plot_pilot_qc(config: dict[str, Any], archive_root: Path, output: Path | None = None) -> dict[str, Any]:
    pilot_root = archive_root / "pilots" / config["pilot"]["id"]
    finalization = json.loads((pilot_root / "finalization_report.json").read_text(encoding="utf-8"))
    if finalization.get("status") != "passed":
        raise ValueError("pilot finalization must pass before plotting")
    catalog = json.loads(Path(finalization["catalog"]).read_text(encoding="utf-8"))
    output = output or pilot_root / "qc_plots"
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    with xr.open_zarr(finalization["spatial_support"], consolidated=True) as source_support:
        support = source_support.load()
    _plot_spatial_support(support, output, files)
    tp_records = _records(catalog, variable="tp")
    standard_tp = _standard_tp(tp_records)
    tp_limits = _plot_shared_maps(
        tp_records,
        "ensemble_mean_weekly_total",
        output / "01_tp_week1_total_common.png",
        files,
        support,
        title="Week-1 ensemble-mean precipitation total · common grid",
        cmap="YlGnBu",
        unit="mm week-1",
        quantiles=(0.0, 0.99),
        floor_zero=True,
    )
    matched_period, matched_limits = _plot_matched_tp_total(
        standard_tp, output, files, support
    )
    t2m_records = _records(catalog, variable="t2m")
    t2m_limits = _plot_shared_maps(
        t2m_records,
        "ensemble_mean_weekly",
        output / "02_t2m_week1_mean_common.png",
        files,
        support,
        title="Week-1 ensemble-mean daily-mean T2M · common grid",
        cmap="coolwarm",
        unit="°C",
        quantiles=(0.01, 0.99),
    )
    _plot_noncomparable_temperature(catalog, output, files, support)
    _plot_erpas_remap(catalog, output, files, support)
    _plot_spread(standard_tp, output, files, support)
    member_rows = _plot_member_inventory(catalog, output, files)
    negative = _plot_negative_tp(standard_tp, output, files, support)
    _plot_gh(catalog, output, files, support)
    valid_times = _plot_valid_time_alignment(catalog, output, files)
    matched_dates = _plot_tp_daily_matched(standard_tp, output, files, support)
    _plot_tp_area_series(standard_tp, output, files, support)
    summary = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "visual preprocessing QC only; not model ranking",
        "status": "generated_requires_human_review",
        "pilot_id": config["pilot"]["id"],
        "catalog": finalization["catalog"],
        "catalog_id": catalog["catalog_id"],
        "spatial_support": finalization["spatial_support"],
        "plot_count_png": sum(item.endswith(".png") for item in files),
        "plot_count_pdf": sum(item.endswith(".pdf") for item in files),
        "files": files,
        "matched_tp_valid_times": matched_dates,
        "matched_tp_aggregate_valid_times": matched_period,
        "valid_times_by_model": valid_times,
        "shared_plot_limits": {
            "tp_weekly_total_mm": list(tp_limits),
            "tp_matched_period_total_mm": list(matched_limits),
            "t2m_weekly_mean_degc": list(t2m_limits),
        },
        "member_inventory": member_rows,
        "negative_tp": negative,
        "automated_checks": {
            "catalog_finalized": True,
            "all_plot_inputs_opened": True,
            "daily_tp_uses_common_valid_time_intersection": True,
            "noncomparable_temperature_separated": True,
        },
    }
    summary_path = output / "qc_plot_summary.json"
    index_path = output / "QC_PLOT_INDEX.md"
    summary["summary"] = str(summary_path)
    summary["index"] = str(index_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_index(output, summary)
    return summary
