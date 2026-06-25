#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
monsoon_truth_compare.py
========================
Publication-quality JJAS model-vs-truth maps for ECMWF-S2S and FuXi-S2S.

The comparison is deliberately strict: every model row is verified against ERA5
over the exact valid dates implied by that model's initialization date and lead
window. If ECMWF and FuXi initialization dates differ, the figure says so rather
than pretending it is a paired head-to-head.

FuXi and ECMWF hindcast calendars are offset. The default mode therefore makes a
same-valid-window comparison: pick a FuXi init + lead window, then choose the
nearest ECMWF init and adjust its lead window so both forecasts verify against
the exact same ERA5 dates.

Examples
--------
conda run -n s2s-hind python monsoon_truth_compare.py --var TP --lead 8 14
conda run -n s2s-hind python monsoon_truth_compare.py --var Z500 --lead 15 21
conda run -n s2s-hind python monsoon_truth_compare.py --mode common --var TP
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyproj import Transformer
import cartopy.io.shapereader as shpreader
from shapely.ops import transform as shp_transform

HERE = Path(__file__).resolve().parent
JJAS_DIR = HERE.parent
FA_ROOT = JJAS_DIR.parent
sys.path[:0] = [str(FA_ROOT), str(JJAS_DIR)]

from core import Physics
from core import grid as G
from core import truth as T
from core.adapters import get_adapter
from core.aggregate import valid_dates_for
from core.regions import _LCC, _WGS84
import adapters_jjas  # noqa: F401
import adapters_fuxi  # noqa: F401
from config import DATA_ROOT, build_config

INDIA_EXTENT = (65.0, 100.0, 5.0, 38.0)
VAR_META = {
    "TP": dict(title="rainfall", unit="mm day$^{-1}$", cmap="YlGnBu",
               diff_cmap="RdBu_r", abs_floor=0.0, diff_min=2.0),
    "Z500": dict(title="500 hPa geopotential height", unit="gpm", cmap="viridis",
                 diff_cmap="RdBu_r", abs_floor=None, diff_min=6.0),
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 120,
    "savefig.dpi": 260,
    "font.family": "DejaVu Sans",
})


def _state_geoms(shp_path):
    tr = Transformer.from_crs(_LCC, _WGS84, always_xy=True)
    return [shp_transform(tr.transform, rec.geometry)
            for rec in shpreader.Reader(shp_path).records()]


def _base_ax(ax, geoms):
    ax.set_extent(INDIA_EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("0.93")
    ax.add_feature(cfeature.OCEAN, facecolor="0.93", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, color="0.25", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, color="0.35", zorder=3)
    ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor="none",
                      edgecolor="0.25", linewidth=0.35, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.65",
                      linestyle=":", alpha=0.65)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlocator = ticker.FixedLocator([70, 80, 90])
    gl.ylocator = ticker.FixedLocator([10, 20, 30])


def _contour(ax, GC, da, levels, cmap, extend, geoms):
    da = da.transpose("lat", "lon")
    _base_ax(ax, geoms)
    return ax.contourf(GC["lon"], GC["lat"], da.values, levels=levels,
                       cmap=cmap, extend=extend, transform=ccrs.PlateCarree())


def _available_fuxi_inits(year):
    root = Path(DATA_ROOT) / "fuxi_combined"
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob(f"{year}*.nc")):
        s = path.stem
        out.append(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    return out


def _nearest(date, choices):
    target = pd.to_datetime(date)
    return min(choices, key=lambda d: abs(pd.to_datetime(d) - target))


def _lead_for_valid_window(init, valid_start, valid_end):
    start = (pd.to_datetime(valid_start) - pd.to_datetime(init)).days
    end = (pd.to_datetime(valid_end) - pd.to_datetime(init)).days
    return start, end


def choose_cases(cfg, mode, ds, de, ecmwf_init=None, fuxi_init=None):
    ecmwf_inits = list(cfg.init_dates)
    fuxi_inits = _available_fuxi_inits(int(cfg.season_label[-4:]))
    if not fuxi_inits:
        raise SystemExit("No FuXi compact JJAS files found in "
                         f"{Path(DATA_ROOT) / 'fuxi_combined'}")

    if mode == "common":
        common = sorted(set(ecmwf_inits) & set(fuxi_inits))
        if not common:
            raise SystemExit(
                "No common ECMWF/FuXi init dates yet. Extract FuXi for one of: "
                + ", ".join(ecmwf_inits[:8]) + " ..."
            )
        init = ecmwf_init or fuxi_init or common[0]
        if init not in common:
            raise SystemExit(f"{init} is not common to ECMWF and FuXi. Common: {common}")
        return [
            dict(model="ECMWF", init=init, ds=ds, de=de),
            dict(model="FuXi", init=init, ds=ds, de=de),
        ], "common"

    f_init = fuxi_init or fuxi_inits[0]
    if f_init not in fuxi_inits:
        raise SystemExit(f"FuXi init {f_init} not found. Available: {fuxi_inits}")
    if mode == "aligned-valid":
        valid_start = (pd.to_datetime(f_init) + pd.Timedelta(days=ds)).strftime("%Y-%m-%d")
        valid_end = (pd.to_datetime(f_init) + pd.Timedelta(days=de)).strftime("%Y-%m-%d")
        candidates = []
        for init in ecmwf_inits:
            eds, ede = _lead_for_valid_window(init, valid_start, valid_end)
            if 1 <= eds <= ede <= 42:
                candidates.append((init, eds, ede))
        if not candidates:
            raise SystemExit(f"No ECMWF init can cover FuXi valid window {valid_start}..{valid_end}")
        if ecmwf_init:
            hit = [c for c in candidates if c[0] == ecmwf_init]
            if not hit:
                raise SystemExit(f"ECMWF init {ecmwf_init} cannot cover {valid_start}..{valid_end}")
            e_init, eds, ede = hit[0]
        else:
            e_init, eds, ede = min(candidates, key=lambda c: abs(pd.to_datetime(c[0]) - pd.to_datetime(f_init)))
        return [
            dict(model="ECMWF", init=e_init, ds=eds, de=ede),
            dict(model="FuXi", init=f_init, ds=ds, de=de),
        ], "aligned-valid"

    e_init = ecmwf_init or (f_init if f_init in ecmwf_inits else _nearest(f_init, ecmwf_inits))
    if e_init not in ecmwf_inits:
        raise SystemExit(f"ECMWF init {e_init} not found in config")
    return [
        dict(model="ECMWF", init=e_init, ds=ds, de=de),
        dict(model="FuXi", init=f_init, ds=ds, de=de),
    ], "available"


def load_case(cfg, GC, spec_case, var):
    model = spec_case["model"]
    init = spec_case["init"]
    ds = spec_case["ds"]
    de = spec_case["de"]
    phys = cfg.physics
    valid = valid_dates_for(init, ds, de, cfg.valid_end)
    if not valid:
        raise RuntimeError(f"No valid ERA5 dates for {model} init {init}, lead {ds}-{de}")
    truth = T.open_truth_wb2(cfg.paths.wb2_zarr, phys, valid[0], valid[-1])
    obs = T.truth_period_mean(var, truth, valid, GC)
    spec = cfg.model(model)
    cube = get_adapter(spec.adapter)(init, var, spec, phys)
    if cube is None:
        raise RuntimeError(f"No {model} {var} cube for init {init}")
    fcst, _ = cube.weekly(ds, de, GC)
    return dict(model=model, init=init, valid=valid,
                ds=ds, de=de,
                obs=obs.transpose("lat", "lon"),
                fcst=fcst.transpose("lat", "lon"),
                bias=(fcst - obs).transpose("lat", "lon"))


def _levels(var, cases):
    meta = VAR_META[var]
    abs_vals = np.concatenate([
        c["obs"].values[np.isfinite(c["obs"].values)].ravel() for c in cases
    ] + [
        c["fcst"].values[np.isfinite(c["fcst"].values)].ravel() for c in cases
    ])
    diff_vals = np.concatenate([
        c["bias"].values[np.isfinite(c["bias"].values)].ravel() for c in cases
    ])
    if var == "TP":
        hi = max(8.0, float(np.ceil(np.nanpercentile(abs_vals, 98))))
        abs_levels = np.linspace(0, hi, 15)
        diff_hi = max(meta["diff_min"], float(np.ceil(np.nanpercentile(np.abs(diff_vals), 98))))
        diff_levels = np.linspace(-diff_hi, diff_hi, 15)
    else:
        lo = float(np.floor(np.nanpercentile(abs_vals, 2) / 10) * 10)
        hi = float(np.ceil(np.nanpercentile(abs_vals, 98) / 10) * 10)
        if hi <= lo:
            hi = lo + 80
        abs_levels = np.linspace(lo, hi, 17)
        diff_hi = max(meta["diff_min"], float(np.ceil(np.nanpercentile(np.abs(diff_vals), 98) / 5) * 5))
        diff_levels = np.linspace(-diff_hi, diff_hi, 17)
    return abs_levels, diff_levels


def _same_valid(cases):
    first = cases[0]["valid"]
    return all(c["valid"] == first for c in cases)


def _make_same_valid_figure(cfg, GC, cases, var, out, mode):
    geoms = _state_geoms(cfg.paths.soi_shapefile)
    meta = VAR_META[var]
    abs_levels, diff_levels = _levels(var, cases)
    cases = sorted(cases, key=lambda c: 0 if c["model"] == "ECMWF" else 1)

    fig, axes = plt.subplots(
        1, 5, figsize=(18.2, 4.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    axes = axes[0]
    obs_map = _contour(axes[0], GC, cases[0]["obs"], abs_levels,
                       meta["cmap"], "max" if var == "TP" else "both", geoms)
    axes[0].set_title(f"ERA5 truth\n{cases[0]['valid'][0]} to {cases[0]['valid'][-1]}")

    for ax, case in zip(axes[1:3], cases):
        _contour(ax, GC, case["fcst"], abs_levels,
                 meta["cmap"], "max" if var == "TP" else "both", geoms)
        ax.set_title(f"{case['model']} forecast\ninit {case['init']}, lead {case['ds']}-{case['de']}")

    err_map = None
    for ax, case in zip(axes[3:5], cases):
        err_map = _contour(ax, GC, case["bias"], diff_levels,
                           meta["diff_cmap"], "both", geoms)
        mb = float(np.nanmean(case["bias"].values))
        ax.set_title(f"{case['model']} error\nmean {mb:+.2f} {meta['unit']}")

    cb_abs = fig.colorbar(obs_map, ax=axes[:3].tolist(),
                          orientation="horizontal", fraction=0.05, pad=0.08,
                          aspect=45, shrink=0.82)
    cb_abs.set_label(f"{meta['title']} ({meta['unit']})")
    cb_diff = fig.colorbar(err_map, ax=axes[3:].tolist(),
                           orientation="horizontal", fraction=0.05, pad=0.08,
                           aspect=26, shrink=0.85)
    cb_diff.set_label(f"forecast - ERA5 ({meta['unit']})")

    note = "same valid dates; lead ranges differ because model initialization calendars differ"
    if mode == "common":
        note = "same initialization and same valid dates"
    fig.suptitle(
        f"JJAS monsoon {meta['title']}: ECMWF-S2S and FuXi-S2S vs ERA5\n{note}",
        fontsize=15, fontweight="bold", y=1.03,
    )
    out.mkdir(parents=True, exist_ok=True)
    fname = f"monsoon_truth_compare_{var}_{cases[0]['valid'][0]}_{cases[0]['valid'][-1]}_{mode}.png"
    path = out / fname
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_figure(cfg, GC, cases, var, out, mode):
    if len(cases) == 2 and _same_valid(cases):
        return _make_same_valid_figure(cfg, GC, cases, var, out, mode)

    geoms = _state_geoms(cfg.paths.soi_shapefile)
    meta = VAR_META[var]
    abs_levels, diff_levels = _levels(var, cases)
    nrows = len(cases)
    fig, axes = plt.subplots(
        nrows, 3, figsize=(13.2, 4.25 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    abs_maps = []
    diff_maps = []
    for i, case in enumerate(cases):
        abs_maps.append(_contour(axes[i, 0], GC, case["obs"], abs_levels,
                                 meta["cmap"], "max" if var == "TP" else "both", geoms))
        _contour(axes[i, 1], GC, case["fcst"], abs_levels,
                 meta["cmap"], "max" if var == "TP" else "both", geoms)
        diff_maps.append(_contour(axes[i, 2], GC, case["bias"], diff_levels,
                                  meta["diff_cmap"], "both", geoms))
        valid_label = f"{case['valid'][0]} to {case['valid'][-1]}"
        axes[i, 0].set_title(f"ERA5 truth\n{case['model']} valid window")
        axes[i, 1].set_title(f"{case['model']} forecast\ninit {case['init']}, lead {case['ds']}-{case['de']}")
        axes[i, 2].set_title(f"{case['model']} error\nforecast - ERA5")
        axes[i, 0].text(
            0.02, 0.02, valid_label, transform=axes[i, 0].transAxes,
            fontsize=8, ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, pad=2),
        )

    cb_abs = fig.colorbar(abs_maps[0], ax=axes[:, :2].ravel().tolist(),
                          orientation="horizontal", fraction=0.045, pad=0.06,
                          aspect=45, shrink=0.82)
    cb_abs.set_label(f"{meta['title']} ({meta['unit']})")
    cb_diff = fig.colorbar(diff_maps[0], ax=axes[:, 2].ravel().tolist(),
                           orientation="horizontal", fraction=0.045, pad=0.06,
                           aspect=28, shrink=0.82)
    cb_diff.set_label(f"error ({meta['unit']})")

    init_note = "paired common init" if mode == "common" else "row-specific valid windows; not a paired model-vs-model claim"
    fig.suptitle(
        f"JJAS monsoon {meta['title']}: S2S forecasts vs matched ERA5 truth\n"
        f"{init_note}",
        fontsize=15, fontweight="bold", y=0.995,
    )
    out.mkdir(parents=True, exist_ok=True)
    fname = f"monsoon_truth_compare_{var}_{mode}.png"
    path = out / fname
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_status(out, cfg, cases, mode):
    fuxi_inits = _available_fuxi_inits(int(cfg.season_label[-4:]))
    common = sorted(set(cfg.init_dates) & set(fuxi_inits))
    lines = [
        "# JJAS Monsoon Comparison Status",
        "",
        f"Mode: `{mode}`",
        f"ECMWF init count in config: {len(cfg.init_dates)}",
        f"FuXi compact init count: {len(fuxi_inits)}",
        f"Common exact init count: {len(common)}",
        "",
        "Cases plotted:",
    ]
    for c in cases:
        lines.append(f"- {c['model']}: init `{c['init']}`, lead `{c['ds']}-{c['de']}`")
    if not common:
        lines.extend([
            "",
            "No exact ECMWF/FuXi paired init exists because the hindcast calendars",
            "are offset in the available archives. Use `--mode aligned-valid` for",
            "the scientifically clean comparison: same ERA5 valid dates, annotated",
            "model-specific initialization dates and lead ranges.",
        ])
        if len(fuxi_inits) <= 1:
            lines.extend([
                "",
                "To strengthen the figure set, compact more FuXi archive dates from its",
                "own calendar, e.g. `20190623`, `20190627`, `20190630`, `20190704`,",
                "then rerun this script for those FuXi inits in aligned-valid mode.",
            ])
        else:
            lines.extend([
                "",
                "The compact FuXi calendar now has enough dates for representative",
                "case panels. Choose map examples by valid date and lead window, then",
                "use the aggregate matched-window study for the main skill claims.",
            ])
    else:
        lines.extend(["", "Common inits:", ", ".join(common)])
    (out / "monsoon_truth_compare_status.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--var", choices=["TP", "Z500"], default="TP")
    ap.add_argument("--lead", nargs=2, type=int, default=[8, 14],
                    metavar=("START_DAY", "END_DAY"))
    ap.add_argument("--mode", choices=["aligned-valid", "available", "common"],
                    default="aligned-valid")
    ap.add_argument("--ecmwf-init", default=None)
    ap.add_argument("--fuxi-init", default=None)
    ap.add_argument("--out", default=str(HERE / "figs" / "monsoon_compare"))
    args = ap.parse_args()

    cfg = build_config(args.year)
    GC = G.build_grid_context(cfg.grid, cfg.paths.region_mask_nc)
    ds, de = args.lead
    case_specs, mode = choose_cases(cfg, args.mode, ds, de, args.ecmwf_init, args.fuxi_init)
    cases = [load_case(cfg, GC, spec_case, args.var) for spec_case in case_specs]
    out = Path(args.out)
    path = make_figure(cfg, GC, cases, args.var, out, mode)
    write_status(out, cfg, case_specs, mode)
    print(f"wrote {path}")
    print(f"wrote {out / 'monsoon_truth_compare_status.md'}")


if __name__ == "__main__":
    main()
