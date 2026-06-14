"""
make_presentation_figs.py — Projector-ready figures for the Spire briefing.

Reuses the data loaders in case1_analysis.py (identical numbers) and renders a
clean, large-font deck optimised for a talk rather than a paper:

  pres_01_truth.png        What happened — ERA5 Z500 / TP / T2M (the event)
  pres_02_t2m_hero.png     Headline — only SPIRE captures the warm core (2 leads x 5 systems)
  pres_03_z500_ridge.png   Circulation — the ridge, who holds it at week-2
  pres_04_precip_dry.png   Desiccation — ABSOLUTE precip (mm/day), no misleading PCC
  pres_05_skill_bars.png   Scorecard — PCC by system (+ T2M bias), SPIRE highlighted

Run:  conda run -n s2s-hind python make_presentation_figs.py
"""
import os, sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import case1_analysis as c1  # noqa: E402

OUT = c1.CONFIG["outdir"]
PROJ = c1.PROJ
EXTENT = c1.EXTENT
LEADS = list(c1.LEADS.keys())                      # ['Week-1 lead', 'Week-2 lead']
COLS = ["ERA5", "SPIRE", "FuXi", "ECMWF", "NCEP"]
CLAB = {"ERA5": "ERA5 (Obs)", "SPIRE": "SPIRE", "FuXi": "FuXi-S2S",
        "ECMWF": "ECMWF", "NCEP": "NCEP"}
SPIRE_C = "#E8720C"   # highlight colour for SPIRE
MODEL_BARC = {"SPIRE": SPIRE_C, "FuXi": "#5B8FB9", "ECMWF": "#4FA06A", "NCEP": "#9B7FC2"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------------
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("  wrote", name, flush=True)


def panel(ax, grid, data, cmap, norm, contour=False, highlight=False,
          ll=False, lb=False, pcc=None):
    ax.set_extent(EXTENT, crs=PROJ)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#DCE7F5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F5F3EE", zorder=0)
    im = None
    if data is not None and not np.all(np.isnan(data)):
        im = ax.pcolormesh(grid.lon, grid.lat, data, transform=PROJ, cmap=cmap,
                           norm=norm, shading="auto", rasterized=True, zorder=2)
        if contour:
            lv = np.linspace(norm.vmin, norm.vmax, 9)
            lv = lv[np.abs(lv) > 1e-6]
            cs = ax.contour(grid.lon, grid.lat, data, levels=lv, colors="k",
                            linewidths=0.5, alpha=0.45, transform=PROJ, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7, zorder=5, edgecolor="#111")
    gl = ax.gridlines(draw_labels=False, lw=0.3, color="#AAA", alpha=0.5, linestyle=":")
    gl.xlocator = mticker.FixedLocator(range(70, 100, 10))
    gl.ylocator = mticker.FixedLocator(range(10, 40, 10))
    if ll:
        gl.left_labels = True; gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {"size": 9, "color": "#555"}
    if lb:
        gl.bottom_labels = True; gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {"size": 9, "color": "#555"}
    if pcc is not None and np.isfinite(pcc):
        ax.text(0.04, 0.93, f"PCC {pcc:+.2f}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", alpha=0.9))
    if highlight:
        for sp in ax.spines.values():
            sp.set_edgecolor(SPIRE_C); sp.set_linewidth(3.2)
    return im


def cbar(fig, im, rect, label, extend="both"):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    return cb


# ----------------------------------------------------------------------------
# Figure 1 — the event (ERA5 truth, 3 panels)
# ----------------------------------------------------------------------------
def fig_truth(grid, F, scales):
    lead = LEADS[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"projection": PROJ})
    spec = [("Z500", "RdBu_r", "Z500 height anomaly (gpm)", True),
            ("TP", "BrBG", "Precipitation anomaly (mm day$^{-1}$)", False),
            ("T2M", "RdBu_r", "2 m temperature anomaly ($^{\\circ}$C)", False)]
    titles = ["(a)  Z500 anomaly", "(b)  Precipitation anomaly", "(c)  T2M anomaly"]
    for j, (v, cmap, lab, ct) in enumerate(spec):
        im = panel(axes[j], grid, F[lead][v]["ERA5"], cmap, scales[v]["norm"],
                   contour=ct, ll=(j == 0), lb=True)
        axes[j].set_title(titles[j], pad=8)
        cbar(fig, im, [0.13 + j*0.283, 0.04, 0.20, 0.022], lab)
    save(fig, "pres_01_truth")


# ----------------------------------------------------------------------------
# Hero matrix: 2 leads (rows) x 5 systems (cols), one variable
# ----------------------------------------------------------------------------
def fig_matrix(grid, F, scales, var, cmap, cbar_lab, name, contour=False,
               metrics=None):
    nrow, ncol = len(LEADS), len(COLS)
    fig = plt.figure(figsize=(18, 8.0))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrow, ncol, figure=fig, hspace=0.06, wspace=0.04,
                  top=0.94, bottom=0.13, left=0.05, right=0.985)
    rowlab = {"Week-1 lead": "Week 1", "Week-2 lead": "Week 2"}
    im_ref = None
    for ri, lead in enumerate(LEADS):
        for ci, m in enumerate(COLS):
            ax = fig.add_subplot(gs[ri, ci], projection=PROJ)
            pcc = None
            if metrics is not None and m != "ERA5":
                pcc = metrics[lead][m]
            im = panel(ax, grid, F[lead][var].get(m), cmap, scales[var]["norm"],
                       contour=contour, highlight=(m == "SPIRE"),
                       ll=(ci == 0), lb=(ri == nrow-1), pcc=pcc)
            if im is not None:
                im_ref = im
            if ri == 0:
                col = SPIRE_C if m == "SPIRE" else "#222"
                ax.set_title(CLAB[m], color=col, pad=7,
                             fontsize=15 if m == "SPIRE" else 13.5)
            if ci == 0:
                ax.text(-0.10, 0.5, rowlab[lead], transform=ax.transAxes,
                        rotation=90, fontsize=13.5, fontweight="bold",
                        ha="center", va="center", color="#333")
    cbar(fig, im_ref, [0.30, 0.06, 0.42, 0.02], cbar_lab)
    save(fig, name)


# ----------------------------------------------------------------------------
# Absolute precipitation (desiccation) — Week-1 lead, ERA5 + 4 systems
# ----------------------------------------------------------------------------
def fig_precip_abs(grid, F_abs, vmax):
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(18, 4.6))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, len(COLS), figure=fig, wspace=0.04, top=0.90, bottom=0.16,
                  left=0.04, right=0.985)
    norm = Normalize(0, vmax)
    im_ref = None
    lead = LEADS[0]
    for ci, m in enumerate(COLS):
        ax = fig.add_subplot(gs[0, ci], projection=PROJ)
        im = panel(ax, grid, F_abs[lead][m], "YlGnBu", norm,
                   highlight=(m == "SPIRE"), ll=(ci == 0), lb=True)
        if im is not None:
            im_ref = im
        col = SPIRE_C if m == "SPIRE" else "#222"
        ax.set_title(CLAB[m], color=col, pad=7,
                     fontsize=15 if m == "SPIRE" else 13.5)
    cbar(fig, im_ref, [0.30, 0.04, 0.42, 0.028],
         "Total precipitation (mm day$^{-1}$)", extend="max")
    save(fig, "pres_04_precip_dry")


# ----------------------------------------------------------------------------
# Skill scorecard — PCC bars (+ T2M bias)
# ----------------------------------------------------------------------------
def fig_bars(metrics, bias):
    mods = ["SPIRE", "FuXi", "ECMWF", "NCEP"]
    x = np.arange(len(mods))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    def grouped(ax, getter, title, ylabel, ref=None, ylim=None, fmt="{:+.2f}"):
        for k, lead in enumerate(LEADS):
            vals = [getter(lead, m) for m in mods]
            bars = ax.bar(x + (k - 0.5)*w, vals, w,
                          color=[MODEL_BARC[m] for m in mods],
                          alpha=1.0 if k == 0 else 0.55,
                          edgecolor="#222", linewidth=0.6,
                          label="Week-1 lead" if k == 0 else "Week-2 lead")
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v + (0.02 if v >= 0 else -0.06),
                        fmt.format(v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8.5)
        if ref is not None:
            ax.axhline(ref, color="#555", lw=1.0, ls="--")
        ax.axhline(0, color="#222", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=12)
        ax.get_xticklabels()[0].set_color(SPIRE_C)
        ax.get_xticklabels()[0].set_fontweight("bold")
        ax.set_title(title, pad=8); ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(*ylim)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.grid(axis="y", ls=":", alpha=0.4)

    grouped(axes[0], lambda l, m: metrics["T2M"][l][m],
            "(a)  T2M  —  PCC", "Pattern correlation",
            ref=0.5, ylim=(-0.25, 1.05))
    grouped(axes[1], lambda l, m: metrics["Z500"][l][m],
            "(b)  Z500  —  PCC", "Pattern correlation",
            ref=0.5, ylim=(0, 1.08))
    grouped(axes[2], lambda l, m: bias[l][m],
            "(c)  T2M  —  bias ($^{\\circ}$C)", "Mean bias ($^{\\circ}$C)",
            ylim=(-7, 1.5), fmt="{:+.1f}")
    axes[0].legend(loc="upper right", fontsize=10, frameon=True, edgecolor="#CCC")
    fig.tight_layout()
    save(fig, "pres_05_skill_bars")


# ----------------------------------------------------------------------------
def main():
    print("Loading data via case1_analysis ...", flush=True)
    import xarray as xr
    base = xr.open_dataset(c1.CONFIG["weekly_fields"])
    grid = c1.Grid(base.lat.values, base.lon.values)
    clim = c1.load_climatology(grid)

    F = c1.load_z_tp_anomalies(grid)
    t2m = c1.load_t2m_anomalies(grid, clim["t2m"])
    for lead in c1.LEADS:
        F[lead]["T2M"] = t2m[lead]["T2M"]

    # Shared symmetric scales (robust)
    scales = {v: c1.var_scale(F, v) for v in ("Z500", "TP", "T2M")}
    # Tighten T2M to +/-6 C so the +4-6 C warm core is vivid; cold-biased models
    # saturate deep blue (honest — colorbar carries the 'extend' arrow).
    scales["T2M"] = dict(vmin=-6.0, vmax=6.0, norm=TwoSlopeNorm(0, -6.0, 6.0))

    # Per-panel PCC metrics
    metrics = {v: {} for v in ("T2M", "Z500")}
    for v in ("T2M", "Z500"):
        for lead in c1.LEADS:
            obs = F[lead][v]["ERA5"]
            metrics[v][lead] = {m: c1.spatial_metrics(F[lead][v].get(m), obs,
                                                      grid.weights)["pcc"]
                                for m in c1.MODELS}
    bias = {}
    for lead in c1.LEADS:
        obs = F[lead]["T2M"]["ERA5"]
        bias[lead] = {m: c1.spatial_metrics(F[lead]["T2M"].get(m), obs,
                                            grid.weights)["bias"]
                      for m in c1.MODELS}

    # Absolute precip = anomaly + climatology
    F_abs = {}
    for lead in c1.LEADS:
        F_abs[lead] = {}
        for m in COLS:
            a = F[lead]["TP"].get(m)
            F_abs[lead][m] = None if a is None else a + clim["tp"]
    pvals = np.concatenate([F_abs[LEADS[0]][m][np.isfinite(F_abs[LEADS[0]][m])]
                            for m in COLS if F_abs[LEADS[0]][m] is not None])
    pmax = float(np.nanpercentile(pvals, 96)); pmax = max(pmax, 2.0)

    print("Rendering presentation figures ...", flush=True)
    fig_truth(grid, F, scales)
    fig_matrix(grid, F, scales, "T2M", "RdBu_r",
               "2 m temperature anomaly ($^{\\circ}$C)", "pres_02_t2m_hero",
               metrics=metrics["T2M"])
    fig_matrix(grid, F, scales, "Z500", "RdBu_r",
               "Z500 height anomaly (gpm)", "pres_03_z500_ridge",
               contour=True, metrics=metrics["Z500"])
    fig_precip_abs(grid, F_abs, pmax)
    fig_bars(metrics, bias)
    print("\nPRESENTATION_FIGS_DONE", flush=True)


if __name__ == "__main__":
    main()
