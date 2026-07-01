#!/usr/bin/env python3
"""Generate figures for the two-season India S2S benchmark paper.

Reads the verification-pipeline summary CSVs and writes publication PDFs into
paper_v2/figs/. Single-column arXiv-style figures, colour-blind-safe palette,
serif typography matching the paper body, panel labels, and a restrained
visual hierarchy (Spire as the narrative anchor, MME muted/dashed).

    python paper_v2/scripts/make_figures.py
"""
from __future__ import annotations

import os
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:  # pragma: no cover - spatial figures degrade gracefully
    _HAS_CARTOPY = False

ROOT = "/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
os.makedirs(OUT, exist_ok=True)

# Paired-bootstrap 95% CIs (produced by make_bootstrap.py). Loaded lazily so
# the figures degrade gracefully to no-band mode if the file is absent.
_BOOT_CI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tables", "bootstrap_ci.csv"))


def _load_boot_ci():
    if not os.path.exists(_BOOT_CI_PATH):
        return None
    return pd.read_csv(_BOOT_CI_PATH)


# Maps the (det/prob summary) source key to the season key used in bootstrap_ci.
_BOOT_SEASON = {
    "jfm_tp": "jfm", "jfm_z500": "jfm",
    "jjas_tp_tp": "jjas_tp", "jjas_z500_z500": "jjas_z500",
    "jjas17_t2m": "jjas17",
}

DET = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/deterministic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/deterministic_summary.csv",
    "jjas_z500": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/deterministic_summary.csv",
    "jjas17": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/deterministic_summary.csv",
}
PROB = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/probabilistic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/probabilistic_summary.csv",
    "jjas_z500": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/probabilistic_summary.csv",
    "jjas17": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/probabilistic_summary.csv",
}

# Okabe-Ito colour-blind-safe palette, fixed per model. Spire gets the
# heaviest visual weight since it anchors the paper's main narrative.
COLOR = {
    "spire": "#0072B2",   # blue
    "fuxi": "#D55E00",    # vermillion
    "delysm": "#009E73",  # green
    "ecmwf": "#4D4D4D",   # dark grey (was black; softer against text)
    "ukmo": "#CC79A7",    # purple-pink
    "ncep": "#E69F00",    # orange
    "mme": "#B0B0B0",     # light grey (dashed, background reference)
}
LABEL = {
    "spire": "Spire AI-S2S", "fuxi": "FuXi-S2S", "delysm": "DLESyM",
    "ecmwf": "ECMWF", "ukmo": "UKMO", "ncep": "NCEP", "mme": "MME",
}
ORDER = ["spire", "fuxi", "delysm", "ecmwf", "ukmo", "ncep", "mme"]
WEEKS = [1, 2, 3, 4, 5, 6]
REGIONS = {
    "northwest_india": "Northwest\nIndia",
    "central_india": "Central\nIndia",
    "south_peninsula": "South\nPeninsula",
    "east_northeast_india": "East/NE\nIndia",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
})


def _allindia(path):
    df = pd.read_csv(path)
    return df[df["region"] == "All India"].copy()


def _panel_label(ax, letter):
    ax.text(-0.02, 1.10, f"({letter})", transform=ax.transAxes,
             fontsize=10, fontweight="bold", va="top", ha="left")


def _curve(ax, df, variable, value, boot_ci=None, boot_season=None, boot_metric=None):
    """Plot metric-vs-lead curves. When ``boot_ci`` and ``boot_season`` are
    given, overlay the paired-bootstrap 95% CI as a shaded band per model
    (drawn only for the narrative-anchor systems to avoid clutter)."""
    sub = df[df["variable"] == variable]
    piv = sub.pivot_table(index="model", columns="week", values=value, aggfunc="mean")

    # Systems whose CI band we actually shade: keep the plot readable by banding
    # only the anchor (Spire) plus the leading dynamical reference (ECMWF).
    band_models = {"spire", "ecmwf"}

    for m in ORDER:
        if m not in piv.index:
            continue
        y = [piv.loc[m, w] if w in piv.columns else None for w in WEEKS]
        if m == "mme":
            ls, lw, alpha, zorder = "--", 1.3, 0.85, 1
        elif m == "spire":
            ls, lw, alpha, zorder = "-", 2.4, 1.0, 5
        else:
            ls, lw, alpha, zorder = "-", 1.5, 0.95, 2

        if boot_ci is not None and boot_season is not None and m in band_models:
            bm = boot_metric if boot_metric is not None else value
            bsub = boot_ci[(boot_ci["season"] == boot_season)
                           & (boot_ci["variable"] == variable)
                           & (boot_ci["metric"] == bm)
                           & (boot_ci["model"] == m)].set_index("week")
            if not bsub.empty:
                lo = [bsub.loc[w, "ci_lo"] if w in bsub.index else None for w in WEEKS]
                hi = [bsub.loc[w, "ci_hi"] if w in bsub.index else None for w in WEEKS]
                ax.fill_between(WEEKS, lo, hi, color=COLOR[m], alpha=0.14,
                                lw=0, zorder=zorder - 1)

        ax.plot(WEEKS, y, ls, color=COLOR[m], lw=lw, alpha=alpha,
                 marker="o", ms=3.2 if m != "spire" else 4.0,
                 markeredgewidth=0, label=LABEL[m], zorder=zorder)
    ax.set_xticks(WEEKS)
    ax.set_xlabel("Lead week")
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def _style_axis(ax):
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _legend_union(axes):
    """Handles/labels from every axis, deduped and in canonical model ORDER,
    so a model plotted in only one panel (e.g. DLESyM, Z500-only) still gets
    a legend entry."""
    by_label = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            by_label.setdefault(l, h)
    order = [LABEL[m] for m in ORDER]
    labels = sorted(by_label, key=lambda l: order.index(l) if l in order else 99)
    return [by_label[l] for l in labels], labels


# ----------------------------------------------------------------------
def fig_acc_lead():
    """2x2 grid: ACC vs lead for TP / Z500, JFM (top) & JJAS (bottom).
    T2M is excluded here (3 systems, no JJAS cross-model comparison) and
    shown separately in the appendix (fig_acc_lead_t2m)."""
    djfm = _allindia(DET["jfm"])
    djjas = pd.concat([_allindia(DET["jjas_tp"]), _allindia(DET["jjas_z500"])],
                      ignore_index=True)
    boot = _load_boot_ci()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), sharex=True)
    panels = [
        (axes[0, 0], djfm, "tp", "JFM 2026 — Precipitation", "jfm"),
        (axes[0, 1], djfm, "z500", "JFM 2026 — Z500", "jfm"),
        (axes[1, 0], djjas, "tp", "JJAS 2019 — Precipitation", "jjas_tp"),
        (axes[1, 1], djjas, "z500", "JJAS 2019 — Z500", "jjas_z500"),
    ]
    letters = string.ascii_lowercase
    for k, (ax, df, var, title, bseason) in enumerate(panels):
        _curve(ax, df, var, "acc", boot_ci=boot, boot_season=bseason, boot_metric="acc")
        ax.set_title(title, pad=14)
        _panel_label(ax, letters[k])
        ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_ylim(-0.7, 1.0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("ACC")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.8, w_pad=2.4)
    fig.savefig(f"{OUT}/fig_acc_lead.pdf")
    plt.close(fig)
    print("wrote fig_acc_lead.pdf")


def fig_acc_lead_t2m():
    """Appendix figure: JJAS 2019 T2M ACC vs lead (DLESyM only).
    JFM 2026 T2M is intentionally omitted: the full_jfm2026_daily_spire run
    no longer scores t2m while its verification truth is being rebuilt."""
    djjas = _allindia(DET["jjas17"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    _curve(ax, djjas, "t2m", "acc")
    ax.set_title("JJAS 2019 — T2M", pad=14)
    ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
    ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
    ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlim(0.6, 6.4)
    _style_axis(ax)
    ax.set_ylabel("ACC")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.18), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(f"{OUT}/fig_acc_lead_t2m.pdf")
    plt.close(fig)
    print("wrote fig_acc_lead_t2m.pdf")


def fig_crpss():
    """CRPSS vs lead, TP & Z500, JFM season (probabilistic story)."""
    d = _allindia(PROB["jfm"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharex=True)
    letters = string.ascii_lowercase
    for k, (ax, var, title) in enumerate([(axes[0], "tp", "Precipitation"),
                                          (axes[1], "z500", "Z500")]):
        _curve(ax, d, var, "crpss_clim")
        ax.set_title(f"JFM 2026 CRPSS — {title}", pad=14)
        _panel_label(ax, letters[k])
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_ylim(-0.6, 1.0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
    axes[0].set_ylabel("CRPSS vs. climatology")
    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.1), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.2)
    fig.savefig(f"{OUT}/fig_crpss.pdf")
    plt.close(fig)
    print("wrote fig_crpss.pdf")


def fig_regional_scorecard():
    """Heatmap-style scorecard: model x region, week-1 precip ACC, JFM."""
    df = pd.read_csv(DET["jfm"])
    df = df[(df["variable"] == "tp") & (df["week"] == 1)]
    df = df[df["region"].isin(REGIONS)]
    piv = df.pivot_table(index="model", columns="region", values="acc", aggfunc="mean")
    models = [m for m in ORDER if m in piv.index]
    cols = list(REGIONS.keys())
    piv = piv.reindex(models)[cols]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(piv.values, cmap="RdYlBu", vmin=0.15, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([REGIONS[c] for c in cols], fontsize=8.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABEL[m] for m in models])
    for i, m in enumerate(models):
        for j in range(len(cols)):
            v = piv.values[i, j]
            if pd.notna(v):
                txt_color = "white" if (v < 0.35 or v > 0.72) else "#222"
                weight = "bold" if m == "spire" else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color=txt_color, fontweight=weight)
    ax.set_title("JFM 2026 week-1 precipitation ACC by region", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ACC", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.grid(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_regional_scorecard.pdf")
    plt.close(fig)
    print("wrote fig_regional_scorecard.pdf")


REGION_PANEL = {
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East/NE India",
}


def _regional_curve(ax, df, variable, region, value="acc"):
    """One region panel: metric vs lead for every model, filtered to a region."""
    sub = df[(df["variable"] == variable) & (df["region"] == region)]
    piv = sub.pivot_table(index="model", columns="week", values=value, aggfunc="mean")
    for m in ORDER:
        if m not in piv.index:
            continue
        y = [piv.loc[m, w] if w in piv.columns else None for w in WEEKS]
        if m == "mme":
            ls, lw, alpha, zorder = "--", 1.3, 0.85, 1
        elif m == "spire":
            ls, lw, alpha, zorder = "-", 2.4, 1.0, 5
        else:
            ls, lw, alpha, zorder = "-", 1.5, 0.95, 2
        ax.plot(WEEKS, y, ls, color=COLOR[m], lw=lw, alpha=alpha,
                marker="o", ms=3.2 if m != "spire" else 4.0,
                markeredgewidth=0, label=LABEL[m], zorder=zorder)
    ax.set_xticks(WEEKS)
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def fig_regional_acc_lead(variable="tp", season="jfm", tag="tp"):
    """2x2 grid of the four IMD homogeneous regions: ACC vs lead for every
    system, one season/variable. Shows how the domain-mean skill of
    fig_acc_lead is distributed across India's rainfall regions."""
    df = pd.read_csv(DET[season])
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), sharex=True, sharey=True)
    letters = string.ascii_lowercase
    regions = list(REGION_PANEL.keys())
    for k, region in enumerate(regions):
        ax = axes[k // 2, k % 2]
        _regional_curve(ax, df, variable, region, "acc")
        ax.set_title(REGION_PANEL[region], pad=14)
        _panel_label(ax, letters[k])
        ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_ylim(-0.7, 1.0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("ACC")
    for ax in axes[1, :]:
        ax.set_xlabel("Lead week")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.8, w_pad=2.0)
    out = f"{OUT}/fig_regional_acc_{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote fig_regional_acc_{tag}.pdf")


def fig_spread_skill():
    """Spread-skill ratio vs lead (calibration), JFM precip & Z500."""
    d = _allindia(PROB["jfm"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharex=True)
    letters = string.ascii_lowercase
    for k, (ax, var, title) in enumerate([(axes[0], "tp", "Precipitation"),
                                          (axes[1], "z500", "Z500")]):
        _curve(ax, d, var, "spread_skill_ratio")
        ax.set_title(f"JFM 2026 spread–skill — {title}", pad=14)
        _panel_label(ax, letters[k])
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", zorder=0)
        ax.text(6.3, 1.0, "ideal", fontsize=7.5, color="grey", va="bottom", ha="right")
        ax.set_ylim(0, 2.0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
    axes[0].set_ylabel("Spread–skill ratio")
    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.1), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.2)
    fig.savefig(f"{OUT}/fig_spread_skill.pdf")
    plt.close(fig)
    print("wrote fig_spread_skill.pdf")


# ----------------------------------------------------------------------
# Spatial maps (India, 1.5-degree common grid, land points only).
# Reads the compact per-cell cache written by make_spatial_cache.py.
# ----------------------------------------------------------------------
def _load_spatial(season):
    path = os.path.join(CACHE, f"spatial_cells_{season}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# India land outline + IMD homogeneous-region boundaries, drawn from the same
# Survey-of-India-derived region masks used for every score in the paper (more
# correct for an India-focused study than a generic global coastline product).
_MASK_PATH = "/home/raj.ayush/s2s/s2s_anlysis/final_paper/masks/imd_region_masks_0.25deg.nc"
_MASK_CACHE = {}


def _india_overlay(ax, regions=True):
    """Overlay the India land boundary (and optionally IMD region borders) on a
    cartopy axis, using the SOI-derived region masks. Degrades silently if the
    mask or xarray is unavailable."""
    try:
        import xarray as xr
    except Exception:
        return
    if "ds" not in _MASK_CACHE:
        if not os.path.exists(_MASK_PATH):
            _MASK_CACHE["ds"] = None
        else:
            _MASK_CACHE["ds"] = xr.open_dataset(_MASK_PATH)
    ds = _MASK_CACHE["ds"]
    if ds is None:
        return
    region_vars = list(ds.data_vars)
    lat = ds["lat"].values
    lon = ds["lon"].values
    # integer region id (0 = ocean/non-India, 1..N = regions), and a land mask
    region_id = np.zeros((len(lat), len(lon)))
    for k, rv in enumerate(region_vars, start=1):
        m = np.nan_to_num(ds[rv].values) > 0
        region_id[m] = k
    land = (region_id > 0).astype(float)
    # India land outline
    ax.contour(lon, lat, land, levels=[0.5], colors="k", linewidths=0.8,
               transform=ccrs.PlateCarree(), zorder=6)
    if regions:
        # boundaries between adjacent regions
        ax.contour(lon, lat, region_id, levels=np.arange(1.5, len(region_vars)),
                   colors="k", linewidths=0.35, alpha=0.5,
                   transform=ccrs.PlateCarree(), zorder=6)


def _grid_from_cells(sub, value):
    """Turn a (lat, lon, value) long frame into a full 2-D array on the
    regular 1.5-degree grid, NaN where there is no land cell, plus the cell-edge
    coordinates for pcolormesh."""
    piv = sub.pivot_table(index="lat", columns="lon", values=value, aggfunc="mean")
    lats = np.sort(sub["lat"].unique())
    lons = np.sort(sub["lon"].unique())
    piv = piv.reindex(index=lats, columns=lons)
    d = 1.5
    lat_edges = np.concatenate([lats - d / 2, [lats[-1] + d / 2]])
    lon_edges = np.concatenate([lons - d / 2, [lons[-1] + d / 2]])
    return piv.values, lon_edges, lat_edges


def _spatial_panel(ax, sub, value, cmap, vmin, vmax, center=None):
    grid, lon_e, lat_e = _grid_from_cells(sub, value)
    if center is not None:
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(lon_e, lat_e, grid, cmap=cmap, norm=norm,
                           transform=ccrs.PlateCarree(), shading="flat")
    else:
        im = ax.pcolormesh(lon_e, lat_e, grid, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="flat")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#888")
    _india_overlay(ax, regions=True)
    ax.set_extent([66, 99, 6, 38], crs=ccrs.PlateCarree())
    ax.set_facecolor("#f2f2f2")
    return im


def _spatial_grid_figure(season, variable, week, value, models, cmap, vmin, vmax,
                         center, cbar_label, title, outname, agg_weeks=None):
    """Grid of per-model spatial maps for one metric. If agg_weeks is given,
    the metric is averaged over those weeks (used for the weeks-1--6 mean bias);
    otherwise a single lead week is shown."""
    if not _HAS_CARTOPY:
        print(f"[skip] {outname}: cartopy unavailable")
        return
    df = _load_spatial(season)
    if df is None:
        print(f"[skip] {outname}: spatial cache for {season} missing")
        return
    df = df[df["variable"] == variable]
    models = [m for m in models if m in df["model"].unique()]
    if not models:
        print(f"[skip] {outname}: no models for {season}/{variable}")
        return

    ncol = 3 if len(models) > 4 else len(models)
    nrow = int(np.ceil(len(models) / ncol))
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.9 * nrow),
                             subplot_kw={"projection": proj})
    axes = np.atleast_1d(axes).ravel()
    im = None
    for k, m in enumerate(models):
        ax = axes[k]
        if agg_weeks is not None:
            sub = df[(df["model"] == m) & (df["week"].isin(agg_weeks))]
            sub = sub.groupby(["lat", "lon"], as_index=False)[value].mean()
        else:
            sub = df[(df["model"] == m) & (df["week"] == week)]
        im = _spatial_panel(ax, sub, value, cmap, vmin, vmax, center)
        ax.set_title(LABEL[m], fontsize=9, fontweight="bold", pad=3)
    for j in range(len(models), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02,
                        aspect=30)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(f"{OUT}/{outname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname}.pdf")


def fig_spatial_localacc_z500():
    """Per-cell temporal anomaly correlation (local ACC), JFM Z500, week 3 -
    where each system keeps circulation skill into the subseasonal range."""
    _spatial_grid_figure(
        "jfm", "z500", 3, "local_acc",
        ["spire", "fuxi", "delysm", "ecmwf", "ukmo", "ncep"],
        cmap="RdBu_r", vmin=-0.8, vmax=0.8, center=0.0,
        cbar_label="Local anomaly correlation",
        title="JFM 2026 Z500 week-3 local anomaly correlation",
        outname="fig_spatial_localacc_z500")


def fig_spatial_bias_tp():
    """Per-cell mean precipitation bias (weeks 1-6 mean), JFM - spatial wet/dry
    structure behind the domain-mean bias table."""
    _spatial_grid_figure(
        "jfm", "tp", None, "bias",
        ["spire", "fuxi", "ecmwf", "ukmo", "ncep"],
        cmap="BrBG", vmin=-3.0, vmax=3.0, center=0.0,
        cbar_label="Bias (mm day$^{-1}$)",
        title="JFM 2026 precipitation bias (weeks 1–6 mean, forecast $-$ ERA5)",
        outname="fig_spatial_bias_tp", agg_weeks=[1, 2, 3, 4, 5, 6])


def main():
    fig_acc_lead()
    fig_crpss()
    fig_regional_scorecard()
    fig_regional_acc_lead(variable="tp", season="jfm", tag="tp")
    fig_regional_acc_lead(variable="z500", season="jfm", tag="z500")
    fig_spread_skill()
    fig_spatial_localacc_z500()
    fig_spatial_bias_tp()
    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
