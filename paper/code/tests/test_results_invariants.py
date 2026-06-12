"""
Scientific regression tests. These encode the manuscript's headline findings so
that any change to the pipeline that would silently overturn them fails loudly.
All use the All-India domain and the 13-init mean.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import DATA, MODELS, require


def _wk(df, var, model, week=1, region="All India", col="pcc"):
    s = df[(df.variable == var) & (df.region == region) & (df.model == model)]
    s = s[s.week.str.contains(str(week))]
    return float(s[col].mean())


@pytest.fixture(scope="module")
def tp():
    return pd.read_csv(require(DATA / "skill_tp_corrected.csv"))


@pytest.fixture(scope="module")
def z():
    df = pd.read_csv(require(DATA / "skill_per_init_full.csv"))
    return df[df.variable == "Z500"]


@pytest.fixture(scope="module")
def t2():
    return pd.read_csv(require(DATA / "skill_t2m.csv"))


# ---------------- deterministic skill ranking ----------------
def test_spire_best_precip_week1(tp):
    pccs = {m: _wk(tp, "TP", m) for m in MODELS}
    assert pccs["SPIRE"] == max(pccs.values()), pccs
    assert pccs["SPIRE"] >= 0.80


def test_spire_top_circulation_week1(z):
    pccs = {m: _wk(z, "Z500", m) for m in MODELS}
    assert pccs["SPIRE"] >= 0.85
    ranked = sorted(pccs, key=pccs.get, reverse=True)
    assert "SPIRE" in ranked[:2], f"SPIRE not top-2 for Z500: {pccs}"


def test_spire_dominates_temperature_week1(t2):
    pccs = {m: _wk(t2, "T2M", m) for m in MODELS}
    assert pccs["SPIRE"] == max(pccs.values()), pccs
    assert pccs["SPIRE"] >= 0.70
    others = max(pccs[m] for m in MODELS if m != "SPIRE")
    assert pccs["SPIRE"] >= 1.5 * others, "SPIRE T2M lead unexpectedly small"


# ---------------- FuXi precipitation unit harmonization ----------------
def test_fuxi_precip_units_harmonized(tp):
    """Guards the mm/h -> mm/day (x24) fix: FuXi domain-mean must be physical,
    not the ~0.03 mm/day collapse seen with the wrong units."""
    fc = _wk(tp, "TP", "FuXi", col="fcst_mean")
    obs = _wk(tp, "TP", "FuXi", col="obs_mean")
    assert fc > 0.3, f"FuXi precip looks collapsed ({fc:.3f} mm/day) -- unit bug?"
    assert abs(fc - obs) < 0.4, f"FuXi precip mean {fc:.2f} far from ERA5 {obs:.2f}"


def test_spire_precip_unbiased(tp):
    fc = _wk(tp, "TP", "SPIRE", col="fcst_mean")
    obs = _wk(tp, "TP", "SPIRE", col="obs_mean")
    assert abs(fc - obs) < 0.15, f"SPIRE precip mean {fc:.2f} vs ERA5 {obs:.2f}"


# ---------------- probabilistic calibration ----------------
@pytest.fixture(scope="module")
def prob():
    return pd.read_csv(require(DATA / "prob_skill.csv"))


def _ssr(prob, var, model):
    s = prob[(prob.variable == var) & (prob.region == "All India")
             & (prob.model == model) & prob.week.str.extract(r"(\d)")[0].astype(int).le(3)]
    return float(s.ssr.mean())


@pytest.mark.parametrize("var", ["TP", "Z500", "T2M"])
def test_spire_better_calibrated_than_fuxi(prob, var):
    """SPIRE's spread-skill ratio is closer to 1 than FuXi's (FuXi overconfident)."""
    spire = _ssr(prob, var, "SPIRE")
    fuxi = _ssr(prob, var, "FuXi")
    assert abs(spire - 1) < abs(fuxi - 1), f"{var}: SPIRE SSR {spire:.2f}, FuXi {fuxi:.2f}"
    assert spire > fuxi


def test_fuxi_is_overconfident(prob):
    """FuXi week-1 spread-skill ratio is far below 1 for at least one field."""
    wk1 = prob[(prob.region == "All India") & (prob.model == "FuXi")
               & prob.week.str.contains("1")]
    assert float(wk1.ssr.min()) < 0.3, "FuXi no longer flagged overconfident"
