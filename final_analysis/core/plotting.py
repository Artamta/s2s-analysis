#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/plotting.py  —  Shared figure style + small helpers (season-agnostic).
================================================================================
A consistent look for every season's figures: fixed model colours/markers, a
clean rcParams theme, week-band shading, a reference line helper, and a
"skill-horizon" finder. Figure CONTENT lives in each season's plots/ script;
only the STYLE lives here so JFM2026 and JJAS look identical.
================================================================================
"""
import numpy as np

# fixed identity per model -> same colour everywhere ---------------------------
MODEL_STYLE = {
    "SPIRE":       dict(color="#0072B2", marker="o", lw=2.2, label="SPIRE"),
    "FuXi":        dict(color="#D55E00", marker="s", lw=2.2, label="FuXi"),
    "ECMWF":       dict(color="#009E73", marker="^", lw=2.2, label="ECMWF"),
    "MME":         dict(color="#000000", marker="D", lw=2.0, label="MME"),
    "Persistence": dict(color="#999999", marker="x", lw=1.6, label="Persistence"),
}
VAR_UNITS = {"TP": "mm/day", "Z500": "gpm", "T2M": "K"}
WEEK_BANDS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42)]


def apply_theme():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.30, "grid.linestyle": ":",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.constrained_layout.use": True,
    })


def style_for(model):
    return MODEL_STYLE.get(model, dict(color="#444444", marker=".", lw=1.8, label=model))


def shade_weeks(ax, ymax_frac=0.04):
    """Light alternating shading + W1..W6 ticks on a daily-lead axis."""
    for i, (a, b) in enumerate(WEEK_BANDS):
        if i % 2 == 0:
            ax.axvspan(a - 0.5, b + 0.5, color="0.92", zorder=0)
    ax.set_xticks([4, 11, 18, 25, 32, 39])
    ax.set_xticklabels([f"W{i}" for i in range(1, 7)])


def ref_line(ax, y, text=None):
    ax.axhline(y, color="0.4", lw=1.0, ls="--", zorder=1)
    if text:
        ax.text(0.99, y, text, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="0.4")


def skill_horizon(leads, values, threshold, above=True):
    """First lead where `values` cross `threshold` (PCC<0.5, CRPSS<0, …).
       Returns the last lead still on the skilful side, or np.nan."""
    leads, values = np.asarray(leads, float), np.asarray(values, float)
    ok = np.isfinite(values)
    leads, values = leads[ok], values[ok]
    if not len(leads):
        return np.nan
    skilful = values >= threshold if above else values <= threshold
    horizon = np.nan
    for L, s in sorted(zip(leads, skilful)):
        if s:
            horizon = L
        else:
            break
    return horizon
