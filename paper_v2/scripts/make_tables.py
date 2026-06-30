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


def make_bias_table(season: str, variable: str, caption: str, label: str,
                    nd: int = 2) -> str:
    """Model x week bias table, bold = closest to zero per column."""
    df = _load(DET[season])
    piv = _pivot(df, variable, "bias")
    if piv.empty:
        return f"% no data for {season} {variable} bias\n"

    bold_index = [m for m in piv.index if m != "mme"]
    formatted = {}
    for w in piv.columns:
        sub = piv.loc[bold_index, w]
        best = sub.abs().idxmin() if not sub.empty else None
        for m in piv.index:
            v = piv.loc[m, w]
            s = f"{v:+.{nd}f}" if pd.notna(v) else "--"
            formatted[(m, w)] = f"\\textbf{{{s}}}" if m == best else s

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


REGION_LABEL = {
    "northwest_india": "Northwest",
    "central_india": "Central",
    "south_peninsula": "South Pen.",
    "east_northeast_india": "East/NE",
}
REGION_ORDER = ["northwest_india", "central_india", "south_peninsula",
                "east_northeast_india"]


def make_regional_table(season: str, variable: str, week: int, value: str,
                        caption: str, label: str, higher_is_better: bool = True,
                        nd: int = 2, exclude_mme_from_bold: bool = True) -> str:
    """Model x IMD-region table at a single fixed lead week."""
    path = DET[season] if value in ("acc", "rmse", "bias", "mae") else PROB[season]
    df = pd.read_csv(path)
    df = df[(df["variable"] == variable) & (df["week"] == week)
            & (df["region"].isin(REGION_ORDER))]
    piv = df.pivot_table(index="model", columns="region", values=value, aggfunc="mean")
    models = [m for m in MODEL_ORDER if m in piv.index]
    cols = [c for c in REGION_ORDER if c in piv.columns]
    piv = piv.reindex(models)[cols]
    if piv.empty:
        return f"% no data for {season} {variable} week{week} {value} (regional)\n"

    bold_index = [m for m in piv.index if not (exclude_mme_from_bold and m == "mme")]
    formatted = {}
    for c in piv.columns:
        sub = piv.loc[bold_index, c]
        fb = _bold_best(sub, higher_is_better, nd)
        for m in piv.index:
            formatted[(m, c)] = fb.get(m, _fmt(piv.loc[m, c], nd))

    header = " & ".join([REGION_LABEL[c] for c in piv.columns])
    rows = []
    for m in piv.index:
        cells = " & ".join(formatted[(m, c)] for c in piv.columns)
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


ALLREGIONS = ["All India", "northwest_india", "central_india",
              "south_peninsula", "east_northeast_india"]
ALLREGION_LABEL = {
    "All India": "All India", "northwest_india": "Northwest",
    "central_india": "Central", "south_peninsula": "S. Peninsula",
    "east_northeast_india": "East/NE",
}


def make_stacked_regional_table(season: str, value: str, variables: list[str],
                                var_label: dict, caption: str, label: str,
                                models_subset: list[str] | None = None,
                                higher_is_better: bool = True, nd: int = 2,
                                bold_closest_zero: bool = False) -> str:
    """Old-paper style table*: \\multirow variable blocks x (All India + 4
    regions) rows, one column per model, weeks-1-6 mean. Matches the format
    in paper/jfm2026_india_s2s_benchmark.tex tab:reg_pcc / tab:reg_bias."""
    path = DET[season] if value in ("acc", "rmse", "bias", "mae") else PROB[season]
    df = pd.read_csv(path)
    df = df[df["region"].isin(ALLREGIONS)]
    models = models_subset or [m for m in MODEL_ORDER if m in df["model"].unique()
                                and m != "mme"] + ["mme"]
    models = [m for m in models if m in df["model"].unique()]

    blocks = []
    for var in variables:
        sub = df[df["variable"] == var]
        if sub.empty:
            continue
        g = sub.groupby(["model", "region"])[value].mean().unstack("region")
        present = [r for r in ALLREGIONS if r in g.columns]
        rows = []
        for i, region in enumerate(present):
            col = g[region].reindex(models)
            if bold_closest_zero:
                best_idx = col.abs().idxmin()
            else:
                best_idx = col.idxmax() if higher_is_better else col.idxmin()
            cells = []
            for m in models:
                v = col.get(m, float("nan"))
                if pd.isna(v):
                    cells.append("--")
                else:
                    s = f"{v:+.{nd}f}" if bold_closest_zero else f"{v:.{nd}f}"
                    cells.append(f"\\textbf{{{s}}}" if m == best_idx else s)
            prefix = f"\\multirow{{{len(present)}}}{{*}}{{\\emph{{{var_label[var]}}}}}" if i == 0 else ""
            rows.append(f"{prefix} & {ALLREGION_LABEL[region]:<12} & " + " & ".join(cells) + " \\\\")
        blocks.append("\n".join(rows))

    header_models = " & ".join(MODEL_LABEL[m] for m in models)
    body = "\n\\midrule\n".join(blocks)
    ncol = len(models)
    colspec = "l" + "c" * ncol

    return textwrap.dedent(f"""\
    \\begin{{table*}}[t]\\centering
    \\small
    \\setlength{{\\tabcolsep}}{{5pt}}
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\begin{{tabular}}{{l{colspec}}}\\toprule
    Region & & {header_models} \\\\
    \\midrule
    {body}
    \\bottomrule
    \\end{{tabular}}\\end{{table*}}
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

    # NOTE: JFM 2026 T2M is intentionally omitted. The T2M verification truth
    # is being rebuilt (daily-mean reconstruction) as of the latest pipeline
    # run, and the JFM run used for this paper (full_jfm2026_daily_spire) no
    # longer scores t2m at all. Re-add once a fresh JFM T2M run lands.

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

    # --- IMD regional breakdown (model x region, fixed week) ---
    artifacts.append(("tab_jfm_regional_tp_w1.tex", make_regional_table(
        "jfm", "tp", 1, "acc",
        "JFM~2026 week-1 precipitation ACC by IMD homogeneous region. "
        "Best individual-model score in each column is bold.",
        "tab:reg_jfm_tp_w1")))

    artifacts.append(("tab_jfm_regional_z500_w1.tex", make_regional_table(
        "jfm", "z500", 1, "acc",
        "JFM~2026 week-1 Z500 ACC by IMD homogeneous region.",
        "tab:reg_jfm_z500_w1")))

    artifacts.append(("tab_jjas_regional_tp_w1.tex", make_regional_table(
        "jjas", "tp", 1, "acc",
        "JJAS~2019 week-1 precipitation ACC by IMD homogeneous region "
        "(IMD truth, 17 common initializations).",
        "tab:reg_jjas_tp_w1")))

    artifacts.append(("tab_jjas_regional_tp_w3.tex", make_regional_table(
        "jjas", "tp", 3, "acc",
        "JJAS~2019 week-3 precipitation ACC by IMD homogeneous region, "
        "showing the monsoon skill collapse is uniform across regions.",
        "tab:reg_jjas_tp_w3")))

    # --- Old-paper-style stacked regional scorecards (weeks 1-6 mean) ---
    var_label = {"tp": "Precipitation", "z500": "Z500"}
    jfm_models = ["spire", "fuxi", "delysm", "ecmwf", "ukmo", "ncep", "mme"]

    artifacts.append(("tab_jfm_reg_acc_full.tex", make_stacked_regional_table(
        "jfm", "acc", ["tp", "z500"], var_label,
        "JFM~2026 region-wise ACC by IMD homogeneous region, weeks~1--6 mean "
        "(90-initialization average). Best system per region in bold.",
        "tab:reg_pcc", models_subset=jfm_models)))

    artifacts.append(("tab_jfm_reg_rmse_full.tex", make_stacked_regional_table(
        "jfm", "rmse", ["tp", "z500"], var_label,
        "JFM~2026 region-wise RMSE by IMD homogeneous region, weeks~1--6 mean. "
        "Precipitation in mm\\,day$^{-1}$; Z500 in m. Best (lowest) system per "
        "region in bold.",
        "tab:reg_rmse", models_subset=jfm_models, higher_is_better=False)))

    artifacts.append(("tab_jfm_reg_bias_full.tex", make_stacked_regional_table(
        "jfm", "bias", ["tp", "z500"], var_label,
        "JFM~2026 region-wise mean bias (forecast minus ERA5) by IMD "
        "homogeneous region, weeks~1--6 mean. Precipitation bias in "
        "mm\\,day$^{-1}$; Z500 bias in m. Value closest to zero per region "
        "in bold.",
        "tab:reg_bias", models_subset=jfm_models, bold_closest_zero=True)))

    # --- Bias-by-week tables (All India) ---
    artifacts.append(("tab_jfm_bias_tp.tex", make_bias_table(
        "jfm", "tp",
        "JFM~2026 all-India precipitation bias (mm\\,day$^{-1}$, forecast "
        "minus ERA5) by lead week. Bold = closest to zero.",
        "tab:bias_tp")))

    artifacts.append(("tab_jfm_bias_z500.tex", make_bias_table(
        "jfm", "z500",
        "JFM~2026 all-India Z500 bias (m, forecast minus ERA5) by lead week. "
        "Bold = closest to zero.",
        "tab:bias_z500")))

    for fname, tex in artifacts:
        with open(os.path.join(OUT, fname), "w") as fh:
            fh.write(tex)
        print(f"wrote {fname}")

    print(f"\nAll tables written to {OUT}")


if __name__ == "__main__":
    main()
