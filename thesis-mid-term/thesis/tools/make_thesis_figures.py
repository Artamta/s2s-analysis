"""Build the three spatial figures of the midterm report (serif, report width).

Usage: python make_thesis_figures.py [out_dir]   (default: ../figures)

  map_selection.pdf   best system per IMD region, JJAS and OND, by lead (#315 results cube)
  map_adapter.pdf     cellwise CRPS improvement of the regional adapter (#297 reviewed fields)
  coverage.pdf        90% coverage: schematic and lead-wise bars (#297 camera-ready numbers)
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
    figure = plt.figure(figsize=(WIDTH, 2.05), layout="constrained")
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
    figure, axes = plt.subplots(1, len(weeks), figsize=(WIDTH, 2.05), layout="constrained")
    for axis, week in zip(axes, weeks):
        field = pooled["leadwise_crps_reduction"][week - 1]
        image = cr.draw_map(axis, grid, grid.smooth(field, support), rings, cmap="RdBu", norm=norm)
        share = float((field[support] > 0).mean())
        axis.set_title(f"Week {week}", fontsize=8, pad=2)
        axis.text(0.03, 0.03, f"{100 * share:.0f}% of cells\nimproved", transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=6.3, color="#3A4651")
        bare(axis)
    colorbar(figure, image, axes, r"CRPS improvement, raw $-$ calibrated (mm day$^{-1}$; positive is better)",
             extend="both")
    save(figure, "map_adapter")


def coverage() -> None:
    """Explain 90% coverage with a schematic, then show it by lead week (#297, 2020-2024 JJAS)."""
    raw = [0.156, 0.331, 0.543, 0.657, 0.726, 0.741]
    moment = [0.566, 0.690, 0.738, 0.785, 0.817, 0.821]
    network = [0.686, 0.761, 0.797, 0.816, 0.838, 0.841]
    colors = {"Raw FuXi-S2S": "#C8553D", "Moment correction": "#9DB4C9", "Calibration network": "#1F4E79"}
    figure, (sketch, bars) = plt.subplots(1, 2, figsize=(WIDTH, 1.95), layout="constrained",
                                          gridspec_kw={"width_ratios": [1, 1.55]})
    # (a) schematic: invented values, only to show what the score means
    observed = np.array([4.0, 7.5, 2.0, 9.0, 5.5, 3.0])
    centre = np.array([5.0, 5.2, 4.6, 5.8, 5.0, 4.4])
    for offset, half, color, label in ((0, 0.9, colors["Raw FuXi-S2S"], "Too narrow"),
                                       (7.5, 3.0, colors["Calibration network"], "Calibrated")):
        x = np.arange(6) + offset
        sketch.vlines(x, centre - half, centre + half, color=color, lw=5, alpha=0.55)
        inside = np.abs(observed - centre) <= half
        sketch.scatter(x[inside], observed[inside], s=16, color="black", zorder=3)
        sketch.scatter(x[~inside], observed[~inside], s=18, facecolor="white", edgecolor="black", lw=0.8, zorder=3)
        sketch.text(offset + 2.5, 11.2, f"{label}\n{inside.sum()} of 6 inside", ha="center", va="bottom", fontsize=7)
    sketch.scatter([], [], s=16, color="black", label="observed, inside range")
    sketch.scatter([], [], s=18, facecolor="white", edgecolor="black", lw=0.8, label="observed, outside range")
    sketch.legend(frameon=False, fontsize=6.4, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
                  handletextpad=0.2, columnspacing=0.8)
    sketch.set_ylim(-2.2, 13.5)
    sketch.set_xlim(-1, 14)
    sketch.set_xticks(())
    sketch.set_yticks(())
    for side in ("top", "right", "left"):
        sketch.spines[side].set_visible(False)
    sketch.set_xlabel("forecast cases (illustration)", fontsize=7)
    sketch.set_title("(a) What coverage measures", loc="left", fontsize=8)
    # (b) real numbers by lead week
    weeks = np.arange(1, 7)
    width = 0.26
    for k, (label, values) in enumerate(zip(colors, (raw, moment, network))):
        bars.bar(weeks + (k - 1) * width, values, width, color=colors[label], label=label, edgecolor="white", lw=0.3)
    bars.axhline(0.9, color="black", ls="--", lw=0.8)
    bars.text(6.45, 0.915, "ideal 0.90", ha="right", va="bottom", fontsize=6.8)
    bars.set_ylim(0, 1.0)
    bars.set_xticks(weeks, [f"Week {w}" for w in weeks])
    bars.set_ylabel("fraction inside 90% range")
    bars.spines[["top", "right"]].set_visible(False)
    bars.tick_params(length=2)
    bars.legend(frameon=False, fontsize=6.6, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.13), handlelength=1.0,
                columnspacing=0.8)
    bars.set_title("(b) FuXi-S2S JJAS rainfall, 2020–2024", loc="left", fontsize=8, pad=14)
    save(figure, "coverage")


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


def map_regions() -> None:
    """The four IMD homogeneous regions (0.25-degree masks) with state outlines."""
    from matplotlib.collections import LineCollection

    shades = {"northwest_india": "#E3C9A0", "central_india": "#B9D3A8",
              "south_peninsula": "#A9C6DE", "east_northeast_india": "#D7B8CF"}
    labels = {"northwest_india": ("Northwest", 73.5, 28.5), "central_india": ("Central", 77.5, 21.0),
              "south_peninsula": ("South\nPeninsula", 77.8, 13.8), "east_northeast_india": ("East &\nNortheast", 91.5, 23.2)}
    rings, _ = cr.load_outline()
    figure, axis = plt.subplots(figsize=(2.35, 2.35), layout="constrained")
    sel.draw_region_map(axis, shades, set())
    axis.add_collection(LineCollection(rings, colors="#33424F", linewidths=0.2, alpha=0.55))
    for text, x, y in labels.values():
        axis.text(x, y, text, ha="center", va="center", fontsize=7, linespacing=1.0)
    save(figure, "map_regions")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    map_selection()
    map_adapter()
    coverage()
    map_forecast()
    map_regions()
    print("wrote", *sorted(p.name for p in OUT.glob("map_*")))
