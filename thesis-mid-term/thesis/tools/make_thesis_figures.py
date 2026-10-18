"""Build the three spatial figures of the midterm report (serif, report width).

Usage: python make_thesis_figures.py [out_dir]   (default: ../figures)

  map_selection.pdf   best system per IMD region, JJAS and OND, by lead (#315 results cube)
  map_adapter.pdf     cellwise CRPS improvement of the regional adapter (#297 reviewed fields)
  map_forecast.pdf    one real-time s2s.scdlds.com issue: rainfall and T2M anomaly, Weeks 1-4

Every value is read from the audited paper artefacts or the published website
JSON; the maps only interpolate the native 1.5 degree cells for display.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/raj.ayush/s2s/s2s_anlysis")
sys.path.insert(0, str(REPO / "FINAL-PAPER-NEURIPS/CAMERA-READY/tools"))

import build_camera_ready_figures as cr  # noqa: E402  (map helpers, audited #297 paths)
import make_315_fig2_options as sel  # noqa: E402  (#315 results cube, region masks, colours)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "figures"
ISSUE = REPO / "clean/fuxi-dashboard/public/data/forecasts/ifs/20260927.json"
WIDTH = 6.7  # report text width in inches
INK = "#1B2A41"

plt.rcParams.update({
    "font.family": "P052", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7, "pdf.fonttype": 42,
    "text.color": INK, "axes.labelcolor": INK, "hatch.color": "#1F1F1F", "hatch.linewidth": 0.45,
})


def save(figure: plt.Figure, stem: str) -> None:
    """Write the PDF used by LaTeX and a PNG preview."""
    figure.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def bare(axis: plt.Axes) -> None:
    """Remove ticks and the frame from a map panel."""
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)


def colorbar(figure: plt.Figure, image, axes, label: str, **kwargs) -> None:
    """Slim horizontal colourbar under a row of maps."""
    bar = figure.colorbar(image, ax=axes, orientation="horizontal", fraction=0.05, pad=0.03,
                          aspect=40, shrink=0.7, **kwargs)
    bar.ax.tick_params(labelsize=6.3, length=1.6, width=0.4, pad=1.2)
    bar.outline.set_linewidth(0.4)
    bar.set_label(label, fontsize=7, labelpad=2)


def map_selection() -> None:
    """System with the highest regional ACC in JJAS and OND, by lead week (#315)."""
    seasons, leads = ["JJAS", "OND"], [1, 3, 6]
    tracks = [("tp_imerg", "Rainfall (vs IMERG)"), ("t2m_era5", "2 m temperature (vs ERA5)")]
    cube = sel.cube[sel.cube.region != "all_india"].dropna(subset=["acc"])
    best = cube.loc[cube.groupby(["track", "region", "season", "lead_week"]).acc.idxmax()]
    figure = plt.figure(figsize=(WIDTH, 2.25), layout="constrained")
    figure.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    used: set[str] = set()
    for (track, label), sub in zip(tracks, figure.subfigures(1, 2, wspace=0.04)):
        sub.suptitle(label, fontsize=8.5)
        axes = sub.subplots(len(seasons), len(leads), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
        for r, season in enumerate(seasons):
            for c, lead in enumerate(leads):
                cells = best[(best.track == track) & (best.season == season) & (best.lead_week == lead)]
                assert len(cells) == 4, (track, season, lead)
                used.update(cells.model)
                sel.draw_region_map(axes[r, c], {row.region: sel.COLORS[row.model] for row in cells.itertuples()},
                                    {row.region for row in cells.itertuples() if row.acc < 0.10})
                if r == 0:
                    axes[r, c].set_title(f"Week {lead}", fontsize=7.5, pad=2)
                if c == 0:
                    axes[r, c].set_ylabel(season, fontsize=7.5, rotation=0, ha="right", va="center")
    order = ["mme", "fuxi_s2s", "ecmwf", "ukmo", "neuralgcm", "dlesym_v0", "dlesym_v1", "fcn3", "cma", "ncep"]
    handles = [Patch(color=sel.COLORS[m], label=sel.NAMES[m]) for m in order if m in used]
    handles.append(Patch(facecolor="white", hatch="//////", edgecolor="#1F1F1F", lw=0.4, label="best ACC < 0.10"))
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
                  handlelength=1.2, columnspacing=0.9)
    save(figure, "map_selection")


def map_adapter() -> None:
    """Raw-minus-adapter CRPS per cell for Weeks 1, 2, 4 and 6 (JJAS 2020-2024, #297)."""
    if cr.sha256(cr.POOLED) != cr.POOLED_SHA256:
        raise ValueError("pooled spatial fields differ from the reviewed run")
    pooled = np.load(cr.POOLED)
    rings, outline = cr.load_outline()
    grid = cr.DisplayGrid(pooled["latitude"], pooled["longitude"], outline)
    support = (pooled["all_india_area_weight_km2"] > 0) & (pooled["temporal_support"].sum(axis=0) > 0)
    weeks = (1, 2, 4, 6)
    norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
    figure, axes = plt.subplots(1, len(weeks), figsize=(WIDTH, 2.25), layout="constrained")
    for axis, week in zip(axes, weeks):
        field = pooled["leadwise_crps_reduction"][week - 1]
        image = cr.draw_map(axis, grid, grid.smooth(field, support), rings, cmap="RdBu", norm=norm)
        share = float((field[support] > 0).mean())
        axis.set_title(f"Week {week}", fontsize=8, pad=2)
        axis.text(0.03, 0.03, f"{100 * share:.0f}% of cells\nimproved", transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=6.3, color="#3A4651")
        bare(axis)
    colorbar(figure, image, axes, r"CRPS improvement, raw $-$ adapter (mm day$^{-1}$; positive is better)",
             extend="both")
    save(figure, "map_adapter")


def map_forecast() -> None:
    """Rainfall and T2M anomaly for Weeks 1-4 of the 27 Sep 2026 website issue."""
    issue = json.loads(ISSUE.read_text(encoding="utf-8"))
    lat = np.asarray(issue["grid"]["latitude"])
    lon = np.asarray(issue["grid"]["longitude"])
    shape = tuple(issue["grid"]["shape"])
    rings, outline = cr.load_outline()
    grid = cr.DisplayGrid(lat, lon, outline)
    support = np.ones(shape, dtype=bool)
    rows = [("rainfall_anomaly", r"Rainfall anomaly (mm day$^{-1}$)"),
            ("temperature_anomaly", r"2 m temperature anomaly ($^\circ$C)")]
    figure = plt.figure(figsize=(WIDTH, 3.95), layout="constrained")
    figure.get_layout_engine().set(h_pad=0.02)
    for (key, label), sub in zip(rows, figure.subfigures(2, 1, hspace=0.03)):
        legend = issue["products"][key]["legend"]
        cmap = ListedColormap(legend["colors"]).with_extremes(under=legend["under"], over=legend["over"])
        norm = BoundaryNorm(legend["boundaries"], cmap.N)
        axes = sub.subplots(1, 4)
        for axis, week in zip(axes, issue["weeks"][:4]):
            field = np.asarray(week["fields"][key], dtype=np.float64).reshape(shape)
            image = cr.draw_map(axis, grid, grid.smooth(field, support), rings, cmap=cmap, norm=norm)
            start = np.datetime64(week["valid_start"]).astype(object).strftime("%-d %b")
            end = np.datetime64(week["valid_end"]).astype(object).strftime("%-d %b")
            axis.set_title(f"Week {week['week']}  ({start}–{end})", fontsize=7.5, pad=2)
            bare(axis)
        colorbar(sub, image, axes, label, extend="both", ticks=legend["boundaries"])
    save(figure, "map_forecast")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    map_selection()
    map_adapter()
    map_forecast()
    print("wrote", *sorted(p.name for p in OUT.glob("map_*")))
