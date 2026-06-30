#!/usr/bin/env python3
"""Generate LaTeX result tables for the two-season India S2S benchmark paper.

All numbers are read directly from the verification-pipeline CSVs so the paper
never hand-transcribes a score. Run from anywhere:

    python paper_v2/scripts/make_tables.py

Outputs .tex fragments into paper_v2/tables/ that the main .tex \input's.
"""
from __future__ import annotations

import os
import textwrap

import pandas as pd

ROOT = "/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs"
OUT = os.path.join(os.path.dirname(__file__), "..", "tables")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

# Canonical runs (see make_tables provenance note in the paper methods).
DET = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/deterministic_summary.csv",
    "jjas": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/deterministic_summary.csv",
}
PROB = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/probabilistic_summary.csv",
    "jjas": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/probabilistic_summary.csv",
}

# Display names + a fixed, sensible model order.
MODEL_LABEL = {
    "spire": "Spire AI-S2S",
    "fuxi": "FuXi-S2S",
    "delysm": "DLESyM",
    "ecmwf": "ECMWF",
    "ukmo": "UKMO",
    "ncep": "NCEP",
    "mme": "MME",
}
MODEL_ORDER = ["spire", "fuxi", "delysm", "ecmwf", "ukmo", "ncep", "mme"]
WEEKS = [1, 2, 3, 4, 5, 6]


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["region"] == "All India"].copy()


def _pivot(df: pd.DataFrame, variable: str, value: str) -> pd.DataFrame:
    sub = df[df["variable"] == variable]
    piv = sub.pivot_table(index="model", columns="week", values=value, aggfunc="mean")
    # keep only models present, in our order
    models = [m for m in MODEL_ORDER if m in piv.index]
    return piv.reindex(models)[[w for w in WEEKS if w in piv.columns]]


def _fmt(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{nd}f}"


def _bold_best(col: pd.Series, higher_is_better: bool = True, nd: int = 2):
    """Return formatted strings with the best value bolded."""
    vals = col.dropna()
    if vals.empty:
        return {i: "--" for i in col.index}
    best = vals.max() if higher_is_better else vals.min()
    out = {}
    for i, v in col.items():
        if pd.isna(v):
            out[i] = "--"
        elif v == best:
            out[i] = f"\\textbf{{{_fmt(v, nd)}}}"
        else:
            out[i] = _fmt(v, nd)
    return out


def make_skill_table(season: str, variable: str, value: str, caption: str,
                     label: str, higher_is_better: bool = True, nd: int = 2,
                     exclude_mme_from_bold: bool = True) -> str:
    df = _load(DET[season] if value in ("acc", "rmse", "bias", "mae") else PROB[season])
    piv = _pivot(df, variable, value)
    if piv.empty:
        return f"% no data for {season} {variable} {value}\n"

    # bold best per week (optionally excluding MME so it doesn't always 'win')
    bold_index = [m for m in piv.index if not (exclude_mme_from_bold and m == "mme")]
    formatted = {}
    for w in piv.columns:
        sub = piv.loc[bold_index, w]
        fb = _bold_best(sub, higher_is_better, nd)
        for m in piv.index:
            formatted[(m, w)] = fb.get(m, _fmt(piv.loc[m, w], nd))

    header = " & ".join([f"W{w}" for w in piv.columns])
    rows = []
    for m in piv.index:
        cells = " & ".join(formatted[(m, w)] for w in piv.columns)
        rows.append(f"    {MODEL_LABEL[m]:<14} & {cells} \\\\")
    body = "\n".join(rows)
    ncol = len(piv.columns)
    colspec = "l" + "c" * ncol

    return textwrap.dedent(f"""\
    \\begin{{table}}[t]
      \\centering
      \\caption{{{caption}}}
      \\label{{{label}}}
      \\small
      \\begin{{tabular}}{{{colspec}}}
        \\toprule
        Model & {header} \\\\
        \\midrule
    {body}
        \\bottomrule
      \\end{{tabular}}
    \\end{{table}}
    """)


def main():
    artifacts = []

    # --- JFM 2026 deterministic ACC (the three variables) ---
    artifacts.append(("tab_jfm_acc.tex", make_skill_table(
        "jfm", "tp", "acc",
        "JFM~2026 all-India precipitation anomaly correlation (ACC) by lead week, "
        "verified against ERA5 daily totals. Best individual-model score in each "
        "column is bold; MME is the multi-model mean.",
        "tab:jfm_acc_tp")))

    artifacts.append(("tab_jfm_acc_z500.tex", make_skill_table(
        "jfm", "z500", "acc",
        "JFM~2026 all-India Z500 ACC by lead week (ERA5 truth).",
        "tab:jfm_acc_z500")))

    artifacts.append(("tab_jfm_acc_t2m.tex", make_skill_table(
        "jfm", "t2m", "acc",
        "JFM~2026 all-India 2-m temperature ACC by lead week (ERA5 truth).",
        "tab:jfm_acc_t2m")))

    # --- JJAS 2019 deterministic ACC ---
    artifacts.append(("tab_jjas_acc_tp.tex", make_skill_table(
        "jjas", "tp", "acc",
        "JJAS~2019 all-India precipitation ACC by lead week, verified against "
        "IMD gridded rainfall (17 common initializations).",
        "tab:jjas_acc_tp")))

    artifacts.append(("tab_jjas_acc_z500.tex", make_skill_table(
        "jjas", "z500", "acc",
        "JJAS~2019 all-India Z500 ACC by lead week (ERA5 truth, 17 common "
        "initializations).",
        "tab:jjas_acc_z500")))

    # --- RMSE companions (precip, both seasons) ---
    artifacts.append(("tab_jfm_rmse_tp.tex", make_skill_table(
        "jfm", "tp", "rmse",
        "JFM~2026 all-India precipitation RMSE (mm\\,day$^{-1}$) by lead week.",
        "tab:jfm_rmse_tp", higher_is_better=False)))

    # --- Probabilistic CRPSS (precip + Z500, JFM) ---
    artifacts.append(("tab_jfm_crpss_tp.tex", make_skill_table(
        "jfm", "tp", "crpss_clim",
        "JFM~2026 all-India precipitation CRPSS versus climatology by lead week. "
        "Positive values indicate probabilistic skill above climatology.",
        "tab:jfm_crpss_tp")))

    artifacts.append(("tab_jfm_crpss_z500.tex", make_skill_table(
        "jfm", "z500", "crpss_clim",
        "JFM~2026 all-India Z500 CRPSS versus climatology by lead week.",
        "tab:jfm_crpss_z500")))

    for fname, tex in artifacts:
        with open(os.path.join(OUT, fname), "w") as fh:
            fh.write(tex)
        print(f"wrote {fname}")

    print(f"\nAll tables written to {OUT}")


if __name__ == "__main__":
    main()
