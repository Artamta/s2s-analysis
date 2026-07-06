#!/usr/bin/env python3
r"""Paired bootstrap confidence intervals and significance tests for the
two-season India S2S benchmark.

Why this exists
---------------
The headline claims in the paper (e.g. "Spire leads precipitation ACC at every
lead") are point estimates from a single season's sample of initializations.
A reviewer will (correctly) ask whether a 0.78-vs-0.73 ACC gap is distinguishable
from sampling noise. This script answers that with a *paired* bootstrap over
initialization dates, which is the natural resampling unit here:

  * The paper's season ACC is defined as the mean over initializations of the
    per-initialization spatial ACC (Eq. 2). The stored per-init `acc` column in
    deterministic_summary.csv is exactly that per-init ACC, and its mean over
    inits reproduces the reported table value (verified). So resampling inits
    and re-averaging is an exact bootstrap of the reported statistic.
  * CRPSS is aggregated in the tables as the mean of the per-init `crpss_clim`
    skill score, so the same resample-and-average logic applies.
  * All models share an identical init set within each run (90 / 35 / 17), so
    we can resample init *dates* once per bootstrap draw and apply the same draw
    to every model. This preserves the between-model correlation that makes a
    paired difference test far more powerful than comparing two independent CIs.

Outputs (written next to the other paper artifacts, under paper_v2/tables/):
  bootstrap_ci.csv        per model x week x variable x metric:
                          point estimate, lo/hi percentile CI, n_init
  bootstrap_pairwise.csv  per (model vs reference) x week x variable x metric:
                          mean paired difference, CI, two-sided bootstrap p-value

Run:
    python paper_v2/scripts/make_bootstrap.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from paper_paths import PAPER_OUTPUT_ROOT

ROOT = str(PAPER_OUTPUT_ROOT)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tables"))
os.makedirs(OUT, exist_ok=True)

# Same canonical runs as make_tables.py / make_figures.py.
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

WEEKS = [1, 2, 3, 4, 5, 6]
# Metrics that aggregate as a plain mean over initializations (matches how the
# tables/figures aggregate the per-init summary rows).
METRICS = {
    "acc": DET,
    "rmse": DET,
    "bias": DET,
    "crpss_clim": PROB,
    "crps": PROB,
    "spread_skill_ratio": PROB,
}

N_BOOT = 10000
CI = (2.5, 97.5)  # 95% percentile interval
SEED = 20260702


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["region"] == "All India"].copy()


def _matrix(df: pd.DataFrame, variable: str, week: int, metric: str):
    """Return (init_dates, {model: values aligned to init_dates}).

    Only models with a value for *every* shared init are kept, so the paired
    resample is well defined. Returns (None, None) if fewer than 2 usable inits.
    """
    sub = df[(df["variable"] == variable) & (df["week"] == week)]
    if sub.empty or metric not in sub.columns:
        return None, None
    wide = sub.pivot_table(index="init_date", columns="model", values=metric, aggfunc="mean")
    # keep only inits with no missing model (paired), and drop all-NaN models
    wide = wide.dropna(axis=1, how="all")
    wide = wide.dropna(axis=0, how="any")
    if wide.shape[0] < 2 or wide.shape[1] < 1:
        return None, None
    return wide.index.to_numpy(), wide


def _bootstrap_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def run_season(season_key: str, variable: str, rng: np.random.Generator):
    ci_rows = []
    pair_rows = []
    for metric, src in METRICS.items():
        df = _load(src[season_key])
        for week in WEEKS:
            inits, wide = _matrix(df, variable, week, metric)
            if wide is None:
                continue
            n = len(inits)
            models = list(wide.columns)
            vals = wide.to_numpy()  # (n_init, n_model)
            idx = _bootstrap_indices(n, N_BOOT, rng)  # (n_boot, n_init)
            # boot means: (n_boot, n_model)
            boot_means = vals[idx, :].mean(axis=1)
            point = vals.mean(axis=0)  # (n_model,)

            for j, m in enumerate(models):
                lo, hi = np.percentile(boot_means[:, j], CI)
                ci_rows.append({
                    "season": season_key, "variable": variable, "week": week,
                    "metric": metric, "model": m, "n_init": n,
                    "point": point[j], "ci_lo": lo, "ci_hi": hi,
                })

            # pairwise vs each candidate reference (only where sensible: the
            # "leader" comparisons the paper actually makes). We compute all
            # non-mme pairs so the paper can quote any of them.
            non_mme = [m for m in models if m != "mme"]
            for a in non_mme:
                ja = models.index(a)
                for b in non_mme:
                    if a >= b:  # one direction only; a<b lexicographic
                        continue
                    jb = models.index(b)
                    diff_boot = boot_means[:, ja] - boot_means[:, jb]
                    diff_point = point[ja] - point[jb]
                    lo, hi = np.percentile(diff_boot, CI)
                    # two-sided bootstrap p-value: proportion of draws on the
                    # opposite side of zero from the point estimate, doubled.
                    if diff_point >= 0:
                        p = 2.0 * np.mean(diff_boot <= 0.0)
                    else:
                        p = 2.0 * np.mean(diff_boot >= 0.0)
                    p = min(1.0, p)
                    pair_rows.append({
                        "season": season_key, "variable": variable, "week": week,
                        "metric": metric, "model_a": a, "model_b": b, "n_init": n,
                        "diff": diff_point, "ci_lo": lo, "ci_hi": hi,
                        "p_value": p, "significant_95": bool(lo > 0 or hi < 0),
                    })
    return ci_rows, pair_rows


def main():
    rng = np.random.default_rng(SEED)
    all_ci, all_pair = [], []
    jobs = [
        ("jfm", "tp"), ("jfm", "z500"),
        ("jjas_tp", "tp"), ("jjas_z500", "z500"),
        ("jjas17", "tp"), ("jjas17", "z500"), ("jjas17", "t2m"),
    ]
    for season_key, variable in jobs:
        ci_rows, pair_rows = run_season(season_key, variable, rng)
        all_ci.extend(ci_rows)
        all_pair.extend(pair_rows)
        print(f"{season_key}/{variable}: {len(ci_rows)} CI rows, {len(pair_rows)} pairwise rows")

    ci_df = pd.DataFrame(all_ci)
    pair_df = pd.DataFrame(all_pair)
    ci_path = os.path.join(OUT, "bootstrap_ci.csv")
    pair_path = os.path.join(OUT, "bootstrap_pairwise.csv")
    ci_df.to_csv(ci_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    print(f"\nwrote {ci_path}  ({len(ci_df)} rows)")
    print(f"wrote {pair_path}  ({len(pair_df)} rows)")
    print(f"\nB = {N_BOOT} resamples, {CI[1]-CI[0]:.0f}% percentile CI, seed = {SEED}")


if __name__ == "__main__":
    main()
