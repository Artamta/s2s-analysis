#!/usr/bin/env python3
r"""Moving-block bootstrap sensitivity for the paired significance tests.

Why this exists
---------------
make_bootstrap.py resamples initialization dates i.i.d. For JFM 2026 the 90
initializations are *consecutive daily* starts, so adjacent forecasts verify on
heavily overlapping target windows and share the same synoptic events; the
i.i.d. bootstrap therefore understates sampling uncertainty at long leads
(the effective sample size is smaller than 90). The standard remedy is a
circular moving-block bootstrap (Wilks 2019, Sec. 5.3.5): resample contiguous
blocks of initialization dates so that serial correlation within a block is
preserved.

This script reruns the paired pairwise comparisons with block lengths chosen
to at least cover the weekly verification window:
  * JFM 2026 (daily inits):    L = 7 and 14 days
  * JJAS 2019 (Mon/Thu inits): L = 4 and 8 inits (~2 and ~4 calendar weeks)

The same block draw is applied to every model (paired), exactly as in the
i.i.d. version. Outputs:

  tables/bootstrap_block_pairwise.csv   same schema as bootstrap_pairwise.csv
                                        plus a block_len column
  tables/bootstrap_block_ci.csv         per-model CIs (same schema as
                                        bootstrap_ci.csv plus block_len)

Run:
    python paper_v2/scripts/make_block_bootstrap.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from paper_paths import PAPER_OUTPUT_ROOT

ROOT = str(PAPER_OUTPUT_ROOT)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tables"))
os.makedirs(OUT, exist_ok=True)

DET = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/deterministic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/deterministic_summary.csv",
    "jjas_z500": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/deterministic_summary.csv",
}
PROB = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/probabilistic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/probabilistic_summary.csv",
    "jjas_z500": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/probabilistic_summary.csv",
}

WEEKS = [1, 2, 3, 4, 5, 6]
METRICS = {"acc": DET, "crpss_clim": PROB}

# Block lengths per season key (in units of initializations).
BLOCK_LENS = {"jfm": [7, 14], "jjas_tp": [4, 8], "jjas_z500": [4, 8]}

N_BOOT = 10000
CI = (2.5, 97.5)
SEED = 20260703


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["region"] == "All India"].copy()


def _matrix(df: pd.DataFrame, variable: str, week: int, metric: str):
    sub = df[(df["variable"] == variable) & (df["week"] == week)]
    if sub.empty or metric not in sub.columns:
        return None
    wide = sub.pivot_table(index="init_date", columns="model", values=metric, aggfunc="mean")
    wide = wide.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if wide.shape[0] < 2 or wide.shape[1] < 1:
        return None
    return wide.sort_index()  # chronological order matters for blocks


def _block_indices(n: int, block_len: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap index matrix (n_boot, n)."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))  # (n_boot, n_blocks)
    offsets = np.arange(block_len)  # (block_len,)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n  # (n_boot, n_blocks, L)
    return idx.reshape(n_boot, n_blocks * block_len)[:, :n]


def run(season_key: str, variable: str, rng: np.random.Generator):
    rows = []
    ci_rows = []
    for metric, src in METRICS.items():
        df = _load(src[season_key])
        for week in WEEKS:
            wide = _matrix(df, variable, week, metric)
            if wide is None:
                continue
            n = wide.shape[0]
            models = list(wide.columns)
            vals = wide.to_numpy()
            point = vals.mean(axis=0)
            for block_len in BLOCK_LENS[season_key]:
                idx = _block_indices(n, block_len, N_BOOT, rng)
                boot_means = vals[idx, :].mean(axis=1)  # (n_boot, n_model)
                for j, m in enumerate(models):
                    lo, hi = np.percentile(boot_means[:, j], CI)
                    ci_rows.append({
                        "season": season_key, "variable": variable, "week": week,
                        "metric": metric, "block_len": block_len, "model": m,
                        "n_init": n, "point": point[j], "ci_lo": lo, "ci_hi": hi,
                    })
                non_mme = [m for m in models if m != "mme"]
                for a in non_mme:
                    ja = models.index(a)
                    for b in non_mme:
                        if a >= b:
                            continue
                        jb = models.index(b)
                        diff_boot = boot_means[:, ja] - boot_means[:, jb]
                        diff_point = point[ja] - point[jb]
                        lo, hi = np.percentile(diff_boot, CI)
                        if diff_point >= 0:
                            p = 2.0 * np.mean(diff_boot <= 0.0)
                        else:
                            p = 2.0 * np.mean(diff_boot >= 0.0)
                        rows.append({
                            "season": season_key, "variable": variable, "week": week,
                            "metric": metric, "block_len": block_len,
                            "model_a": a, "model_b": b, "n_init": n,
                            "diff": diff_point, "ci_lo": lo, "ci_hi": hi,
                            "p_value": min(1.0, p),
                            "significant_95": bool(lo > 0 or hi < 0),
                        })
    return rows, ci_rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows, all_ci = [], []
    for season_key, variable in [("jfm", "tp"), ("jfm", "z500"),
                                 ("jjas_tp", "tp"), ("jjas_z500", "z500")]:
        rows, ci_rows = run(season_key, variable, rng)
        all_rows.extend(rows)
        all_ci.extend(ci_rows)
        print(f"{season_key}/{variable}: {len(rows)} pairwise rows, {len(ci_rows)} CI rows")
    out = pd.DataFrame(all_rows)
    path = os.path.join(OUT, "bootstrap_block_pairwise.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")
    ci_out = pd.DataFrame(all_ci)
    ci_path = os.path.join(OUT, "bootstrap_block_ci.csv")
    ci_out.to_csv(ci_path, index=False)
    print(f"wrote {ci_path} ({len(ci_out)} rows)")

    # Console summary: which i.i.d.-significant headline results survive blocking?
    iid = pd.read_csv(os.path.join(OUT, "bootstrap_pairwise.csv"))
    iid = iid[iid["metric"].isin(["acc"])]
    for season_key, variable in [("jfm", "tp"), ("jfm", "z500")]:
        for block_len in BLOCK_LENS[season_key]:
            blk = out[(out.season == season_key) & (out.variable == variable)
                      & (out.metric == "acc") & (out.block_len == block_len)]
            ii = iid[(iid.season == season_key) & (iid.variable == variable)]
            merged = blk.merge(
                ii[["week", "model_a", "model_b", "significant_95"]],
                on=["week", "model_a", "model_b"], suffixes=("_blk", "_iid"))
            spire = merged[(merged.model_a == "spire") | (merged.model_b == "spire")]
            changed = spire[spire.significant_95_blk != spire.significant_95_iid]
            print(f"\n{season_key}/{variable} ACC, L={block_len}: "
                  f"{len(spire)} Spire pairs, {len(changed)} change significance:")
            for _, r in changed.iterrows():
                other = r.model_b if r.model_a == "spire" else r.model_a
                print(f"  wk{int(r.week)} vs {other}: diff={r['diff']:+.3f} "
                      f"iid_sig={r.significant_95_iid} -> block_sig={r.significant_95_blk} "
                      f"(p={r.p_value:.3f})")


if __name__ == "__main__":
    main()
