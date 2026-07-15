#!/usr/bin/env python3
r"""Generate LaTeX result tables for the two-season India S2S benchmark paper.

All numbers are read directly from the verification-pipeline CSVs so the paper
never hand-transcribes a score. Run from anywhere:

    python paper_v2/scripts/make_tables.py

Outputs .tex fragments into paper_v2/tables/ that the main .tex \input's.
"""
from __future__ import annotations

import os
import textwrap

import pandas as pd

from paper_paths import PAPER_OUTPUT_ROOT

ROOT = str(PAPER_OUTPUT_ROOT)
OUT = os.path.join(os.path.dirname(__file__), "..", "tables")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

# Canonical runs (see make_tables provenance note in the paper methods).
DET = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/deterministic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/deterministic_summary.csv",
    "jjas_tp_era5": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_era5truth/deterministic_summary.csv",
    "jjas17": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/deterministic_summary.csv",
}
PROB = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/probabilistic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/probabilistic_summary.csv",
    "jjas_tp_era5": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_era5truth/probabilistic_summary.csv",
    "jjas17": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/probabilistic_summary.csv",
}

# Display names + a fixed, sensible model order.
MODEL_LABEL = {
    "spire": "Spire AI-S2S",
    "fuxi": "FuXi-S2S",
    "ecmwf": "ECMWF",
    "ukmo": "UKMO",
    "ncep": "NCEP",
    "mme": "MME",
}
MODEL_LABEL_SHORT = {
    "spire": "Spire",
    "fuxi": "FuXi",
    "ecmwf": "ECMWF",
    "ukmo": "UKMO",
    "ncep": "NCEP",
    "mme": "MME",
}
MODEL_ORDER = ["spire", "fuxi", "ecmwf", "ukmo", "ncep", "mme"]
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


def _num(x: float, nd: int = 2, signed: bool = False, bold: bool = False) -> str:
    """Math-mode numeric cell: proper minus sign, no negative zero (-0.00)."""
    if pd.isna(x):
        return "--"
    v = round(float(x), nd)
    if v == 0:
        v = 0.0
    s = f"{v:.{nd}f}" if v == 0 else (f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}")
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def _fmt(x: float, nd: int = 2) -> str:
    return _num(x, nd)


def _plain(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "--"
    v = round(float(x), nd)
    if v == 0:
        v = 0.0
    return f"{v:.{nd}f}"


def _range_label(weeks: list[int]) -> str:
    if len(weeks) == 1:
        return f"W{weeks[0]}"
    return f"W{weeks[0]}--W{weeks[-1]}"


def _compress_week_leaders(leaders: list[tuple[int, str]]) -> str:
    """Compress [(week, model), ...] into 'ECMWF (W1--W3); ...'."""
    if not leaders:
        return "--"
    chunks = []
    cur_model = leaders[0][1]
    cur_weeks = [leaders[0][0]]
    for week, model in leaders[1:]:
        if model == cur_model and week == cur_weeks[-1] + 1:
            cur_weeks.append(week)
        else:
            chunks.append(f"{MODEL_LABEL[cur_model]} ({_range_label(cur_weeks)})")
            cur_model = model
            cur_weeks = [week]
    chunks.append(f"{MODEL_LABEL[cur_model]} ({_range_label(cur_weeks)})")
    return "; ".join(chunks)


def _leader_by_week(piv: pd.DataFrame, higher_is_better: bool = True,
                    exclude_mme: bool = True) -> list[tuple[int, str]]:
    rows = [m for m in piv.index if not (exclude_mme and m == "mme")]
    out = []
    for week in WEEKS:
        vals = piv.loc[rows, week].dropna()
        if vals.empty:
            continue
        model = vals.idxmax() if higher_is_better else vals.idxmin()
        out.append((week, model))
    return out


def make_allindia_clean_table() -> str:
    """Two-case all-India synthesis table for the main Results section."""
    jjas_acc = _pivot(_load(DET["jjas_tp_era5"]), "tp", "acc")
    jjas_rmse = _pivot(_load(DET["jjas_tp_era5"]), "tp", "rmse")
    jjas_crps = _pivot(_load(PROB["jjas_tp_era5"]), "tp", "crps")
    jjas_imd_acc = _pivot(_load(DET["jjas_tp"]), "tp", "acc")
    jfm_acc = _pivot(_load(DET["jfm"]), "tp", "acc")
    jfm_rmse = _pivot(_load(DET["jfm"]), "tp", "rmse")
    jfm_crpss = _pivot(_load(PROB["jfm"]), "tp", "crpss_clim")

    jjas_acc_leaders = _leader_by_week(jjas_acc)
    jjas_rmse_leaders = _leader_by_week(jjas_rmse, higher_is_better=False)
    jjas_crps_leaders = _leader_by_week(jjas_crps, higher_is_better=False)
    jjas_imd_leaders = _leader_by_week(jjas_imd_acc)
    jfm_acc_leaders = _leader_by_week(jfm_acc)
    jfm_rmse_leaders = _leader_by_week(jfm_rmse, higher_is_better=False)
    jfm_crpss_leaders = _leader_by_week(jfm_crpss)

    jjas_acc_w1 = _plain(jjas_acc.loc["ecmwf", 1])
    jjas_acc_w6 = _plain(jjas_acc.loc["ecmwf", 6])
    jjas_imd_w1 = _plain(jjas_imd_acc.loc[[m for m in jjas_imd_acc.index if m != "mme"], 1].max())
    jjas_imd_w3 = _plain(jjas_imd_acc.loc[[m for m in jjas_imd_acc.index if m != "mme"], 3].max())
    jfm_acc_w1 = _plain(jfm_acc.loc["spire", 1])
    jfm_acc_w6 = _plain(jfm_acc.loc["spire", 6])

    rows = [
        ("JJAS~2019 / ERA5", "ACC",
         f"{_compress_week_leaders(jjas_acc_leaders)}; ECMWF is {jjas_acc_w1} at W1 and {jjas_acc_w6} at W6.",
         "Main ERA5-referenced monsoon ranking."),
        ("JJAS~2019 / ERA5", "RMSE / CRPS",
         f"RMSE leader: {_compress_week_leaders(jjas_rmse_leaders)}. CRPS leader: {_compress_week_leaders(jjas_crps_leaders)}.",
         "Error scores support the all-India ACC result."),
        ("JJAS~2019 / IMD", "ACC sensitivity",
         f"Best individual ACC is {jjas_imd_w1} at W1 and {jjas_imd_w3} at W3; leaders are {_compress_week_leaders(jjas_imd_leaders)}.",
         "Gauge-verified monsoon skill is much weaker."),
        ("JFM~2026 / ERA5", "ACC / RMSE",
         f"ACC leader: {_compress_week_leaders(jfm_acc_leaders)}; Spire is {jfm_acc_w1} at W1 and {jfm_acc_w6} at W6. RMSE leader: {_compress_week_leaders(jfm_rmse_leaders)}.",
         "Main deterministic winter ranking."),
        ("JFM~2026 / ERA5", "CRPSS",
         f"CRPSS leader: {_compress_week_leaders(jfm_crpss_leaders)}.",
         "Probabilistic ranking differs from ACC/RMSE."),
    ]
    body = "\n".join(
        f"    {case} & {score} & {result} & {read} \\\\" for case, score, result, read in rows
    )
    return textwrap.dedent(f"""\
    \\begin{{table}}[!tbp]
      \\centering
      \\footnotesize
      \\setlength{{\\tabcolsep}}{{4pt}}
      \\renewcommand{{\\arraystretch}}{{1.16}}
      \\caption{{Clean all-India summary for the V1 precipitation benchmark. Values are weekly precipitation scores over the common Indian land mask. The MME is not treated as a separate forecast model.}}
      \\label{{tab:allindia_clean}}
      \\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{0.17\\linewidth}}>{{\\raggedright\\arraybackslash}}p{{0.12\\linewidth}}>{{\\raggedright\\arraybackslash}}p{{0.43\\linewidth}}>{{\\raggedright\\arraybackslash}}p{{0.19\\linewidth}}@{{}}}}
        \\toprule
        Case / reference & Score & Leading individual system(s) & Interpretation \\\\
        \\midrule
    {body}
        \\bottomrule
      \\end{{tabular}}
    \\end{{table}}
    """)


def make_regions_clean_table() -> str:
    """Homogeneous-region synthesis table for JJAS 2019 and JFM 2026."""
    jjas = pd.read_csv(DET["jjas_tp_era5"])
    jjas = jjas[(jjas["variable"] == "tp") & (jjas["region"].isin(REGION_ORDER))]
    jjas_models = ["fuxi", "ecmwf", "ukmo", "ncep"]
    jfm = pd.read_csv(DET["jfm"])
    jfm = jfm[(jfm["variable"] == "tp") & (jfm["region"].isin(REGION_ORDER))]
    jfm_models = ["spire", "fuxi", "ecmwf", "ukmo", "ncep"]

    rows = []
    for region in REGION_ORDER:
        jpiv = jjas[jjas["region"] == region].pivot_table(
            index="model", columns="week", values="acc", aggfunc="mean"
        ).reindex(jjas_models)
        jleaders = _leader_by_week(jpiv, exclude_mme=False)
        jw1 = jpiv.loc[[m for m in jpiv.index if pd.notna(jpiv.loc[m, 1])], 1].max()
        jw6 = jpiv.loc[[m for m in jpiv.index if pd.notna(jpiv.loc[m, 6])], 6].max()

        jfm_mean = jfm[jfm["region"] == region].groupby("model")["acc"].mean().reindex(jfm_models)
        jfm_best = jfm_mean.dropna().idxmax()
        jfm_val = jfm_mean.loc[jfm_best]

        rows.append(
            f"    {REGION_LABEL[region]} & "
            f"{_compress_week_leaders(jleaders)}; best ACC {_plain(jw1)} at W1 and {_plain(jw6)} at W6. & "
            f"{MODEL_LABEL[jfm_best]} ({_plain(jfm_val)} weeks~1--6 mean) \\\\"
        )
    body = "\n".join(rows)
    return textwrap.dedent(f"""\
    \\begin{{table}}[!tbp]
      \\centering
      \\footnotesize
      \\setlength{{\\tabcolsep}}{{4pt}}
      \\renewcommand{{\\arraystretch}}{{1.16}}
      \\caption{{Clean regional summary for the V1 precipitation benchmark. JJAS~2019 entries show the best individual ERA5-referenced ACC by lead week in each IMD homogeneous rainfall region. JFM~2026 entries show the best individual ERA5-referenced ACC averaged over weeks~1--6.}}
      \\label{{tab:regions_clean}}
      \\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{0.14\\linewidth}}>{{\\raggedright\\arraybackslash}}p{{0.53\\linewidth}}>{{\\raggedright\\arraybackslash}}p{{0.24\\linewidth}}@{{}}}}
        \\toprule
        Region & JJAS~2019 / ERA5 & JFM~2026 / ERA5 \\\\
        \\midrule
    {body}
        \\bottomrule
      \\end{{tabular}}
    \\end{{table}}
    """)


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
        else:
            out[i] = _num(v, nd, bold=(v == best))
    return out


def make_bias_table(season: str, variable: str, caption: str, label: str,
                    nd: int = 2, placement: str = "t") -> str:
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
            formatted[(m, w)] = _num(v, nd, signed=True, bold=(m == best))

    header = " & ".join([f"W{w}" for w in piv.columns])
    rows = []
    for m in piv.index:
        cells = " & ".join(formatted[(m, w)] for w in piv.columns)
        rows.append(f"    {MODEL_LABEL[m]:<14} & {cells} \\\\")
    body = "\n".join(rows)
    ncol = len(piv.columns)
    colspec = "l" + "c" * ncol

    return textwrap.dedent(f"""\
    \\begin{{table}}[{placement}]
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
                     exclude_mme_from_bold: bool = True,
                     placement: str = "t") -> str:
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
    \\begin{{table}}[{placement}]
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
                        nd: int = 2, exclude_mme_from_bold: bool = True,
                        placement: str = "t") -> str:
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
    \\begin{{table}}[{placement}]
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


def make_regional_multiweek_table(season: str, variable: str, weeks: list[int],
                                  value: str, caption: str, label: str,
                                  higher_is_better: bool = True, nd: int = 2,
                                  exclude_mme_from_bold: bool = True,
                                  placement: str = "t") -> str:
    """Model x (region, week) table: one \\multirow region block per region,
    one column per requested week. Consolidates what would otherwise be N
    separate single-week regional tables into one."""
    path = DET[season] if value in ("acc", "rmse", "bias", "mae") else PROB[season]
    df = pd.read_csv(path)
    df = df[(df["variable"] == variable) & (df["week"].isin(weeks))
            & (df["region"].isin(REGION_ORDER))]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    rows = []
    for i, region in enumerate(REGION_ORDER):
        sub = df[df["region"] == region]
        piv = sub.pivot_table(index="model", columns="week", values=value, aggfunc="mean")
        piv = piv.reindex(models)
        cells_per_model = {m: [] for m in models}
        for w in weeks:
            if w not in piv.columns:
                for m in models:
                    cells_per_model[m].append("--")
                continue
            col = piv[w]
            bold_index = [m for m in models if not (exclude_mme_from_bold and m == "mme")]
            fb = _bold_best(col.reindex(bold_index), higher_is_better, nd)
            for m in models:
                v = col.get(m, float("nan"))
                cells_per_model[m].append(fb.get(m, _fmt(v, nd)))
        for j, m in enumerate(models):
            rowprefix = f"\\multirow{{{len(models)}}}{{*}}{{{REGION_LABEL[region]}}}" if j == 0 else ""
            cells = " & ".join(cells_per_model[m])
            rows.append(f"{rowprefix} & {MODEL_LABEL[m]:<14} & {cells} \\\\")
        if i < len(REGION_ORDER) - 1:
            rows.append("\\midrule")
    body = "\n".join(rows)
    header = " & ".join([f"W{w}" for w in weeks])
    ncol = len(weeks)
    colspec = "ll" + "c" * ncol

    return textwrap.dedent(f"""\
    \\begin{{table}}[{placement}]\\centering
    \\small
    \\setlength{{\\tabcolsep}}{{4pt}}
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\begin{{tabular}}{{{colspec}}}\\toprule
    Region & Model & {header} \\\\
    \\midrule
    {body}
    \\bottomrule
    \\end{{tabular}}\\end{{table}}
    """)


def make_jjas_regional_reference_table(caption: str, label: str,
                                       weeks: list[int] | None = None,
                                       nd: int = 2) -> str:
    """Compact main-text table for JJAS regional precipitation ACC under two
    precipitation references. Each cell is the best individual-model ACC for
    that region/reference/week, with a model-code subscript. This is a readable
    all-week comparison; full model-by-model IMD values remain in the appendix.
    """
    weeks = weeks or WEEKS
    model_codes = {"fuxi": "F", "ecmwf": "E", "ukmo": "U", "ncep": "N"}

    best = {}
    for ref, season in [("ERA5", "jjas_tp_era5"), ("IMD", "jjas_tp")]:
        df = pd.read_csv(DET[season])
        df = df[(df["variable"] == "tp") & (df["week"].isin(weeks))
                & (df["region"].isin(REGION_ORDER))]
        df = df[df["model"].isin(model_codes)]
        g = df.pivot_table(index=["region", "week"], columns="model",
                           values="acc", aggfunc="mean")
        for region in REGION_ORDER:
            for w in weeks:
                if (region, w) not in g.index:
                    best[(ref, region, w)] = "--"
                    continue
                vals = g.loc[(region, w)].dropna()
                if vals.empty:
                    best[(ref, region, w)] = "--"
                    continue
                m = vals.idxmax()
                v = round(float(vals[m]), nd)
                if v == 0:
                    v = 0.0
                s = f"{v:.{nd}f}"
                best[(ref, region, w)] = f"${s}_{{\\mathrm{{{model_codes[m]}}}}}$"

    rows = []
    for i, region in enumerate(REGION_ORDER):
        for j, ref in enumerate(["ERA5", "IMD"]):
            rowprefix = f"\\multirow{{2}}{{*}}{{{REGION_LABEL[region]}}}" if j == 0 else ""
            cells = " & ".join(best.get((ref, region, w), "--") for w in weeks)
            rows.append(f"{rowprefix} & {ref:<4} & {cells} \\\\")
        if i < len(REGION_ORDER) - 1:
            rows.append("\\midrule")
    body = "\n".join(rows)
    header = " & ".join([f"W{w}" for w in weeks])

    return textwrap.dedent(f"""\
    \\begin{{table}}[H]\\centering
    \\footnotesize
    \\setlength{{\\tabcolsep}}{{4pt}}
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\begin{{tabular}}{{llcccccc}}\\toprule
    Region & Truth & {header} \\\\
    \\midrule
    {body}
    \\bottomrule
    \\end{{tabular}}\\end{{table}}
    """)


def make_jjas_regional_best_table(season: str, truth_label: str, caption: str,
                                  label: str, weeks: list[int] | None = None,
                                  nd: int = 2) -> str:
    """Compact model-code table: best individual model by region/week."""
    weeks = weeks or WEEKS
    model_codes = {"fuxi": "F", "ecmwf": "E", "ukmo": "U", "ncep": "N"}
    df = pd.read_csv(DET[season])
    df = df[(df["variable"] == "tp") & (df["week"].isin(weeks))
            & (df["region"].isin(REGION_ORDER))]
    df = df[df["model"].isin(model_codes)]
    g = df.pivot_table(index=["region", "week"], columns="model",
                       values="acc", aggfunc="mean")

    rows = []
    for region in REGION_ORDER:
        cells = []
        for w in weeks:
            if (region, w) not in g.index:
                cells.append("--")
                continue
            vals = g.loc[(region, w)].dropna()
            if vals.empty:
                cells.append("--")
                continue
            m = vals.idxmax()
            v = round(float(vals[m]), nd)
            if v == 0:
                v = 0.0
            cells.append(f"${v:.{nd}f}_{{\\mathrm{{{model_codes[m]}}}}}$")
        rows.append(f"{REGION_LABEL[region]:<10} & {truth_label:<4} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    header = " & ".join([f"W{w}" for w in weeks])

    return textwrap.dedent(f"""\
    \\begin{{table}}[H]\\centering
    \\footnotesize
    \\setlength{{\\tabcolsep}}{{4pt}}
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\begin{{tabular}}{{llcccccc}}\\toprule
    Region & Truth & {header} \\\\
    \\midrule
    {body}
    \\bottomrule
    \\end{{tabular}}\\end{{table}}
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
    """Stacked regional table: \\multirow variable blocks x (All India + 4
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
                cells.append(_num(v, nd, signed=bold_closest_zero,
                                  bold=(m == best_idx)))
            prefix = f"\\multirow{{{len(present)}}}{{*}}{{\\emph{{{var_label[var]}}}}}" if i == 0 else ""
            rows.append(f"{prefix} & {ALLREGION_LABEL[region]:<12} & " + " & ".join(cells) + " \\\\")
        blocks.append("\n".join(rows))

    header_models = " & ".join(MODEL_LABEL[m] for m in models)
    body = "\n\\midrule\n".join(blocks)
    ncol = len(models)
    colspec = "l" + "c" * ncol

    return textwrap.dedent(f"""\
    \\begin{{table}}[H]\\centering
    \\footnotesize
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\begin{{tabular}}{{l{colspec}}}\\toprule
    Variable & Region & {header_models} \\\\
    \\midrule
    {body}
    \\bottomrule
    \\end{{tabular}}\\end{{table}}
    """)


def make_jfm_tp_regional_acc_mean_table() -> str:
    """Precipitation-only JFM regional ACC table for the main Results."""
    df = pd.read_csv(DET["jfm"])
    df = df[(df["variable"] == "tp") & (df["region"].isin(ALLREGIONS))]
    models = ["spire", "fuxi", "ecmwf", "ukmo", "ncep", "mme"]
    g = df.groupby(["model", "region"])["acc"].mean().unstack("region")
    rows = []
    for region in ALLREGIONS:
        col = g[region].reindex(models)
        individual = [m for m in models if m != "mme"]
        best_idx = col.loc[individual].idxmax()
        cells = []
        for m in models:
            v = col.get(m, float("nan"))
            cells.append(_num(v, 2, bold=(m == best_idx)))
        rows.append(f"    {ALLREGION_LABEL[region]:<12} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    return textwrap.dedent(f"""\
    \\begin{{table}}[H]
      \\centering
      \\footnotesize
      \\setlength{{\\tabcolsep}}{{4pt}}
      \\caption{{JFM~2026 precipitation ACC by IMD homogeneous region, weeks~1--6 mean (90-initialization average). Bold marks the best individual model in each region using unrounded scores; MME is shown but not bolded.}}
      \\label{{tab:reg_jfm_tp_mean}}
      \\begin{{tabular}}{{lcccccc}}
        \\toprule
        Region & Spire AI-S2S & FuXi-S2S & ECMWF & UKMO & NCEP & MME \\\\
        \\midrule
    {body}
        \\bottomrule
      \\end{{tabular}}
    \\end{{table}}
    """)


def make_main_acc_table(season: str, caption: str, label: str,
                        placement: str = "H") -> str:
    """Main-text ACC table: All India + four IMD homogeneous regions, all
    models, weeks 1--6. Bold marks the best individual model in each
    region--week; MME is shown but not bolded."""
    df = pd.read_csv(DET[season])
    regions = ["All India"] + REGION_ORDER
    df = df[(df["variable"] == "tp") & (df["week"].isin(WEEKS))
            & (df["region"].isin(regions))]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    rows = []
    for i, region in enumerate(regions):
        sub = df[df["region"] == region]
        piv = sub.pivot_table(index="model", columns="week", values="acc",
                              aggfunc="mean")
        piv = piv.reindex(models)
        cells_per_model = {m: [] for m in models}
        for w in WEEKS:
            if w not in piv.columns:
                for m in models:
                    cells_per_model[m].append("--")
                continue
            col = piv[w]
            bold_index = [m for m in models if m != "mme"]
            fb = _bold_best(col.reindex(bold_index), higher_is_better=True, nd=2)
            for m in models:
                v = col.get(m, float("nan"))
                cells_per_model[m].append(fb.get(m, _fmt(v, 2)))
        region_label = ALLREGION_LABEL.get(region, REGION_LABEL.get(region, region))
        for j, m in enumerate(models):
            rowprefix = f"\\multirow{{{len(models)}}}{{*}}{{{region_label}}}" if j == 0 else ""
            cells = " & ".join(cells_per_model[m])
            rows.append(f"{rowprefix} & {MODEL_LABEL[m]:<14} & {cells} \\\\")
        if i < len(regions) - 1:
            rows.append("\\midrule")
    body = "\n".join(rows)
    header = " & ".join([f"W{w}" for w in WEEKS])

    return textwrap.dedent(f"""\
    \\begin{{table}}[{placement}]
      \\centering
      \\scriptsize
      \\setlength{{\\tabcolsep}}{{3pt}}
      \\renewcommand{{\\arraystretch}}{{0.96}}
      \\caption{{{caption}}}
      \\label{{{label}}}
      \\begin{{tabular}}{{llcccccc}}
        \\toprule
        Region & Model & {header} \\\\
        \\midrule
    {body}
        \\bottomrule
      \\end{{tabular}}
    \\end{{table}}
    """)


def _main_acc_tabular(season: str, compact_labels: bool = False) -> str:
    """Tabular-only version of the main ACC table for side-by-side display."""
    df = pd.read_csv(DET[season])
    regions = ["All India"] + REGION_ORDER
    df = df[(df["variable"] == "tp") & (df["week"].isin(WEEKS))
            & (df["region"].isin(regions))]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    labels = MODEL_LABEL_SHORT if compact_labels else MODEL_LABEL

    rows = []
    for i, region in enumerate(regions):
        sub = df[df["region"] == region]
        piv = sub.pivot_table(index="model", columns="week", values="acc",
                              aggfunc="mean")
        piv = piv.reindex(models)
        cells_per_model = {m: [] for m in models}
        for w in WEEKS:
            if w not in piv.columns:
                for m in models:
                    cells_per_model[m].append("--")
                continue
            col = piv[w]
            bold_index = [m for m in models if m != "mme"]
            fb = _bold_best(col.reindex(bold_index), higher_is_better=True, nd=2)
            for m in models:
                v = col.get(m, float("nan"))
                cells_per_model[m].append(fb.get(m, _fmt(v, 2)))
        region_label = ALLREGION_LABEL.get(region, REGION_LABEL.get(region, region))
        for j, m in enumerate(models):
            rowprefix = f"\\multirow{{{len(models)}}}{{*}}{{{region_label}}}" if j == 0 else ""
            cells = " & ".join(cells_per_model[m])
            rows.append(f"{rowprefix} & {labels[m]:<6} & {cells} \\\\")
        if i < len(regions) - 1:
            rows.append("\\midrule")
    body = "\n".join(rows)
    header = " & ".join([f"W{w}" for w in WEEKS])

    return textwrap.dedent(f"""\
      \\begin{{tabular}}{{llcccccc}}
        \\toprule
        Region & Model & {header} \\\\
        \\midrule
    {body}
        \\bottomrule
      \\end{{tabular}}
    """)


def make_main_acc_side_by_side_table() -> str:
    """Two main Results ACC tables on one landscape page."""
    jjas_tabular = _main_acc_tabular("jjas_tp_era5", compact_labels=True)
    jfm_tabular = _main_acc_tabular("jfm", compact_labels=True)
    return textwrap.dedent(f"""\
    \\begin{{landscape}}
    \\begingroup
    \\footnotesize
    \\setlength{{\\tabcolsep}}{{2.1pt}}
    \\renewcommand{{\\arraystretch}}{{0.80}}
    \\captionsetup{{font=footnotesize}}
    \\begin{{center}}
    \\begin{{minipage}}[t]{{0.492\\linewidth}}
      \\centering
      \\captionof{{table}}{{JJAS~2019 precipitation ACC by region and lead week
      (ERA5 precipitation; 35 common Monday/Thursday initializations).}}
      \\label{{tab:jjas_acc_main}}
    {jjas_tabular}
    \\end{{minipage}}\\hfill
    \\begin{{minipage}}[t]{{0.492\\linewidth}}
      \\centering
      \\captionof{{table}}{{JFM~2026 precipitation ACC by region and lead week
      (ERA5 precipitation; 90 daily initializations).}}
      \\label{{tab:jfm_acc_main}}
    {jfm_tabular}
    \\end{{minipage}}

    \\vspace{{4pt}}
    \\footnotesize
    \\emph{{Note.}} Bold marks the best individual model in each region--week;
    MME is shown for comparison but is not bolded. Spire is unavailable for
    JJAS~2019.
    \\end{{center}}
    \\endgroup
    \\end{{landscape}}
    """)


def _main_acc_lookup(season: str) -> dict[str, dict[str, list[str]]]:
    """Formatted ACC cells by region/model/week for the combined main table."""
    df = pd.read_csv(DET[season])
    regions = ["All India"] + REGION_ORDER
    df = df[(df["variable"] == "tp") & (df["week"].isin(WEEKS))
            & (df["region"].isin(regions))]

    out: dict[str, dict[str, list[str]]] = {}
    for region in regions:
        sub = df[df["region"] == region]
        piv = sub.pivot_table(index="model", columns="week", values="acc",
                              aggfunc="mean").reindex(MODEL_ORDER)
        out[region] = {m: [] for m in MODEL_ORDER}
        for w in WEEKS:
            if w not in piv.columns:
                for m in MODEL_ORDER:
                    out[region][m].append("--")
                continue
            col = piv[w]
            bold_index = [m for m in MODEL_ORDER if m != "mme"]
            fb = _bold_best(col.reindex(bold_index),
                            higher_is_better=True, nd=2)
            for m in MODEL_ORDER:
                v = col.get(m, float("nan"))
                out[region][m].append(fb.get(m, _fmt(v, 2)))
    return out


def make_main_acc_combined_table() -> str:
    """Single portrait Results table with JJAS and JFM week blocks side by side."""
    regions = ["All India"] + REGION_ORDER
    jjas = _main_acc_lookup("jjas_tp_era5")
    jfm = _main_acc_lookup("jfm")

    rows = []
    for r_i, region in enumerate(regions):
        region_label = ALLREGION_LABEL.get(region, REGION_LABEL.get(region, region))
        for m_i, model in enumerate(MODEL_ORDER):
            region_cell = (
                f"\\multirow{{{len(MODEL_ORDER)}}}{{*}}{{\\textbf{{{region_label}}}}}"
                if m_i == 0 else ""
            )
            jjas_cells = " & ".join(jjas[region][model])
            jfm_cells = " & ".join(jfm[region][model])
            rows.append(f"{region_cell} & {MODEL_LABEL_SHORT[model]} & "
                        f"{jjas_cells} & {jfm_cells} \\\\")
        if r_i < len(regions) - 1:
            rows.append("\\midrule")
    body = "\n".join(rows)
    header = " & ".join([f"W{w}" for w in WEEKS])

    return textwrap.dedent(f"""\
    \\begin{{table}}[H]
    \\centering
    \\tiny
    \\setlength{{\\tabcolsep}}{{2.1pt}}
    \\renewcommand{{\\arraystretch}}{{1.58}}
    \\caption{{Main precipitation ACC table by region, model, and lead week.
    JJAS~2019 columns are shown first, followed by JFM~2026; both seasons are
    verified against ERA5 precipitation. JJAS uses 35 common Monday/Thursday
    initializations and JFM uses 90 daily initializations.}}
    \\label{{tab:main_acc}}
    \\resizebox{{0.98\\linewidth}}{{!}}{{%
    \\begin{{tabular}}{{@{{}}llcccccc@{{\\hspace{{5pt}}}}cccccc@{{}}}}
    \\toprule
    Region & Model & \\multicolumn{{6}}{{c}}{{JJAS~2019}} &
    \\multicolumn{{6}}{{c}}{{JFM~2026}} \\\\
    \\cmidrule(lr){{3-8}}\\cmidrule(l){{9-14}}
    & & {header} & {header} \\\\
    \\midrule
    {body}
    \\bottomrule
    \\end{{tabular}}
    }}

    \\vspace{{3pt}}
    \\begin{{minipage}}{{0.98\\linewidth}}
    \\tiny\\emph{{Note.}} FuXi denotes FuXi-S2S and Spire denotes Spire
    AI-S2S. Bold marks the best individual model in each
    region--season--week using unrounded scores; MME is shown for comparison
    but is not bolded. Spire AI-S2S is unavailable for JJAS~2019.
    \\end{{minipage}}
    \\end{{table}}
    """)


BOOT_PAIR = os.path.join(OUT, "bootstrap_pairwise.csv")
BOOT_PAIR_BLOCK = os.path.join(OUT, "bootstrap_block_pairwise.csv")
# Primary block length per season (in initializations): 7 daily inits for JFM,
# 4 Mon/Thu inits (~2 calendar weeks) for JJAS.
BLOCK_LEN_PRIMARY = {"jfm": 7, "jjas_tp": 4}


def make_significance_table(season: str, variable: str, reference: str,
                            caption: str, label: str, metric: str = "acc",
                            nd: int = 2, placement: str = "t") -> str:
    """Paired-bootstrap significance of one system's lead over every other
    system, by lead week. Each cell is the mean paired ACC difference
    (reference minus competitor) with a marker: * = 95% CI excludes zero under
    BOTH the moving-block bootstrap (primary; accounts for serial correlation
    between nearby initializations) and the i.i.d. date bootstrap; (*) =
    significant under the i.i.d. bootstrap only; dagger = neither. Reads
    bootstrap_pairwise.csv (make_bootstrap.py) and
    bootstrap_block_pairwise.csv (make_block_bootstrap.py)."""
    if not os.path.exists(BOOT_PAIR):
        return f"% bootstrap_pairwise.csv missing; run make_bootstrap.py first\n"
    pw = pd.read_csv(BOOT_PAIR)
    pw = pw[(pw["season"] == season) & (pw["variable"] == variable)
            & (pw["metric"] == metric)]
    if pw.empty:
        return f"% no bootstrap rows for {season} {variable} {metric}\n"
    blk = None
    if os.path.exists(BOOT_PAIR_BLOCK):
        blk = pd.read_csv(BOOT_PAIR_BLOCK)
        blk = blk[(blk["season"] == season) & (blk["variable"] == variable)
                  & (blk["metric"] == metric)
                  & (blk["block_len"] == BLOCK_LEN_PRIMARY.get(season, 7))]
        if blk.empty:
            blk = None

    others = [m for m in MODEL_ORDER if m not in (reference, "mme")
              and m in set(pw["model_a"]).union(pw["model_b"])]
    rows = []
    for other in others:
        cells = []
        for w in WEEKS:
            q = pw[(pw["week"] == w)
                   & (((pw["model_a"] == reference) & (pw["model_b"] == other))
                      | ((pw["model_a"] == other) & (pw["model_b"] == reference)))]
            if q.empty:
                cells.append("--")
                continue
            r = q.iloc[0]
            # sign so that positive = reference is better (higher ACC)
            d = r["diff"] if r["model_a"] == reference else -r["diff"]
            d = round(float(d), nd)
            if d == 0:
                d = 0.0
            s = f"{d:.{nd}f}" if d == 0 else f"{d:+.{nd}f}"
            sig_iid = bool(r["significant_95"])
            sig_blk = sig_iid  # fall back to iid if block file missing
            if blk is not None:
                qb = blk[(blk["week"] == w)
                         & (((blk["model_a"] == reference) & (blk["model_b"] == other))
                            | ((blk["model_a"] == other) & (blk["model_b"] == reference)))]
                if not qb.empty:
                    sig_blk = bool(qb.iloc[0]["significant_95"])
            if sig_blk and sig_iid:
                mark = "^{*}"
            elif sig_iid:
                mark = "^{(*)}"
            else:
                mark = "^{\\dagger}"
            cells.append(f"${s}{mark}$")
        rows.append(f"    {MODEL_LABEL[other]:<14} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    header = " & ".join([f"W{w}" for w in WEEKS])
    colspec = "l" + "c" * len(WEEKS)
    return textwrap.dedent(f"""\
    \\begin{{table}}[{placement}]
      \\centering
      \\caption{{{caption}}}
      \\label{{{label}}}
      \\small
      \\begin{{tabular}}{{{colspec}}}
        \\toprule
        vs.\\ {MODEL_LABEL[reference]} & {header} \\\\
        \\midrule
    {body}
        \\bottomrule
      \\end{{tabular}}

      \\vspace{{2pt}}
      \\footnotesize\\raggedright
      Mean paired difference in ACC ({MODEL_LABEL[reference]} minus competitor)
      over the season's initializations; positive favours
      {MODEL_LABEL[reference]}. Both bootstraps use 10{{,}}000 paired resamples
      of initialization dates. $^{{*}}$: 95\\% confidence interval excludes
      zero under both the moving-block bootstrap (primary; blocks of
      {BLOCK_LEN_PRIMARY.get(season, 7)} consecutive initializations, which
      accounts for serial correlation between overlapping forecasts) and the
      i.i.d. bootstrap; $^{{(*)}}$: excludes zero under the i.i.d. bootstrap
      only, so it is not robust to serial correlation; $^{{\\dagger}}$:
      includes zero under both (not distinguishable from sampling noise).
    \\end{{table}}
    """)


def main():
    artifacts = []

    # --- Main Results ACC table ---
    artifacts.append(("tab_main_acc_combined.tex",
                      make_main_acc_combined_table()))

    # --- JFM 2026 deterministic ACC ---
    artifacts.append(("tab_jfm_acc.tex", make_skill_table(
        "jfm", "tp", "acc",
        "JFM~2026 all-India precipitation anomaly correlation (ACC) by lead week, "
        "verified against ERA5 daily totals. Best individual-model score in each "
        "column is bold; MME is the multi-model mean.",
        "tab:jfm_acc_tp", placement="H")))

    # --- Paired-bootstrap differences relative to Spire (JFM) ---
    artifacts.append(("tab_jfm_sig_tp.tex", make_significance_table(
        "jfm", "tp", "spire",
        "JFM~2026 precipitation: paired-bootstrap all-India ACC difference "
        "between Spire AI-S2S and each other system, by lead week.",
        "tab:jfm_sig_tp", placement="H")))

    # NOTE: JFM 2026 T2M is intentionally omitted. The T2M verification truth
    # is being rebuilt (daily-mean reconstruction) as of the latest pipeline
    # run, and the JFM run used for this paper (full_jfm2026_daily_spire) no
    # longer scores t2m at all. Re-add once a fresh JFM T2M run lands.

    # --- JJAS 2019 deterministic ACC ---
    artifacts.append(("tab_jjas_acc_tp_era5.tex", make_skill_table(
        "jjas_tp_era5", "tp", "acc",
        "JJAS~2019 all-India precipitation ACC by lead week, verified against "
        "ERA5 precipitation (35 common Monday/Thursday initializations).",
        "tab:jjas_acc_tp", placement="H")))

    artifacts.append(("tab_jjas_acc_tp.tex", make_skill_table(
        "jjas_tp", "tp", "acc",
        "JJAS~2019 all-India precipitation ACC by lead week, verified against "
        "IMD gridded rainfall (35 common Monday/Thursday initializations).",
        "tab:jjas_acc_tp_imd", placement="H")))

    # --- RMSE companions (precip, both seasons) ---
    artifacts.append(("tab_jfm_rmse_tp.tex", make_skill_table(
        "jfm", "tp", "rmse",
        "JFM~2026 all-India precipitation RMSE (mm\\,day$^{-1}$) by lead week.",
        "tab:jfm_rmse_tp", higher_is_better=False, placement="H")))

    # --- Probabilistic CRPSS (precip, JFM) ---
    artifacts.append(("tab_jfm_crpss_tp.tex", make_skill_table(
        "jfm", "tp", "crpss_clim",
        "JFM~2026 all-India precipitation CRPSS versus climatology by lead week. "
        "Positive values indicate probabilistic skill above climatology.",
        "tab:jfm_crpss_tp", placement="H")))

    # --- IMD regional breakdown (model x region, fixed week) ---
    artifacts.append(("tab_jfm_regional_tp_w1.tex", make_regional_table(
        "jfm", "tp", 1, "acc",
        "JFM~2026 week-1 precipitation ACC by IMD homogeneous region. "
        "Best individual-model score in each column is bold.",
        "tab:reg_jfm_tp_w1", placement="H")))

    artifacts.append(("tab_jjas_regional_tp_ref_allweeks.tex", make_jjas_regional_reference_table(
        "JJAS~2019 best individual-model precipitation ACC by IMD homogeneous "
        "region under ERA5 and IMD verification, weeks~1--6. Subscripts denote "
        "the model attaining the value: E=ECMWF, F=FuXi-S2S, U=UKMO, N=NCEP. "
        "The same 35 common Monday/Thursday initializations are used for both "
        "references; ERA5 is the main reanalysis benchmark and IMD is the "
        "gauge-based sensitivity check.",
        "tab:reg_jjas_tp_ref_allweeks")))

    artifacts.append(("tab_jjas_regional_tp_era5_best_allweeks.tex",
                      make_jjas_regional_best_table(
        "jjas_tp_era5", "ERA5",
        "JJAS~2019 best individual-model precipitation ACC by IMD homogeneous "
        "region under ERA5 verification, weeks~1--6. Subscripts denote the "
        "model attaining the value: E=ECMWF, F=FuXi-S2S, U=UKMO, N=NCEP.",
        "tab:reg_jjas_tp_era5_best_allweeks")))

    artifacts.append(("tab_jjas_regional_tp_w1w3.tex", make_regional_multiweek_table(
        "jjas_tp", "tp", [1, 2, 3, 4, 5, 6], "acc",
        "JJAS~2019 precipitation ACC by IMD homogeneous region for every lead "
        "week~1--6 (IMD reference, 35 common Monday/Thursday initializations). "
        "Bold marks the best individual model in each region--week; MME is shown "
        "but not bolded.",
        "tab:reg_jjas_tp_w1w3", placement="H")))

    artifacts.append(("tab_jjas_regional_tp_era5_allweeks.tex", make_regional_multiweek_table(
        "jjas_tp_era5", "tp", [1, 2, 3, 4, 5, 6], "acc",
        "JJAS~2019 precipitation ACC by IMD homogeneous region for every lead "
        "week~1--6 (ERA5 reference, 35 common Monday/Thursday initializations). "
        "Bold marks the best individual model in each region--week; "
        "MME is shown but not bolded.",
        "tab:reg_jjas_tp_era5_allweeks", placement="H")))

    # --- Stacked regional scorecards (weeks 1-6 mean) ---
    var_label = {"tp": "Precipitation"}
    jfm_models = ["spire", "fuxi", "ecmwf", "ukmo", "ncep", "mme"]

    artifacts.append(("tab_jfm_reg_acc_tp.tex",
                      make_jfm_tp_regional_acc_mean_table()))

    artifacts.append(("tab_jfm_reg_rmse_tp.tex", make_stacked_regional_table(
        "jfm", "rmse", ["tp"], var_label,
        "JFM~2026 precipitation RMSE by IMD homogeneous region, weeks~1--6 mean "
        "(mm\\,day$^{-1}$). Best (lowest) system per region is bold.",
        "tab:reg_rmse_tp", models_subset=jfm_models, higher_is_better=False)))

    artifacts.append(("tab_jfm_reg_bias_tp.tex", make_stacked_regional_table(
        "jfm", "bias", ["tp"], var_label,
        "JFM~2026 precipitation mean bias (forecast minus ERA5) by IMD "
        "homogeneous region, weeks~1--6 mean (mm\\,day$^{-1}$). Value closest "
        "to zero per region is bold.",
        "tab:reg_bias_tp", models_subset=jfm_models, bold_closest_zero=True)))

    # --- Bias-by-week tables (All India) ---
    artifacts.append(("tab_jfm_bias_tp.tex", make_bias_table(
        "jfm", "tp",
        "JFM~2026 all-India precipitation bias (mm\\,day$^{-1}$, forecast "
        "minus ERA5) by lead week. Bold = closest to zero.",
        "tab:bias_tp", placement="H")))

    for fname, tex in artifacts:
        with open(os.path.join(OUT, fname), "w") as fh:
            fh.write(tex)
        print(f"wrote {fname}")

    print(f"\nAll tables written to {OUT}")


if __name__ == "__main__":
    main()
