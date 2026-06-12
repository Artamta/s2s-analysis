"""
Integrity of the precomputed skill tables that the figure scripts consume:
schema, coverage (13 inits, 6 weeks, 4 systems, 5 regions), and value ranges.
Tests skip cleanly if a table is absent.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import DATA, MODELS, N_INITS, WEEKS, REGIONS, require

SKILL_CSVS = {
    "skill_per_init_full.csv": ["Z500", "TP"],
    "skill_tp_corrected.csv": ["TP"],
    "skill_t2m.csv": ["T2M"],
}


@pytest.mark.parametrize("fname,expected_vars", SKILL_CSVS.items())
def test_skill_csv_schema_and_coverage(fname, expected_vars):
    df = pd.read_csv(require(DATA / fname))
    for col in ["variable", "region", "week", "init_date", "model", "pcc", "rmse", "bias"]:
        assert col in df.columns, f"{fname} missing column {col}"
    assert set(expected_vars).issubset(set(df.variable.unique()))
    assert set(MODELS).issubset(set(df.model.unique()))
    assert df.init_date.nunique() == N_INITS, f"{fname}: expected {N_INITS} inits"
    assert set(WEEKS).issubset(set(df.week.unique()))
    assert REGIONS.issubset(set(df.region.unique()))


@pytest.mark.parametrize("fname", list(SKILL_CSVS))
def test_skill_values_in_range(fname):
    df = pd.read_csv(require(DATA / fname))
    pcc = df.pcc.dropna()
    assert pcc.between(-1.001, 1.001).all(), f"{fname}: PCC out of [-1,1]"
    assert (df.rmse.dropna() >= 0).all(), f"{fname}: negative RMSE"
    # not silently all-NaN
    assert df.pcc.notna().mean() > 0.5, f"{fname}: >half PCC are NaN"


def test_prob_skill_schema_and_ranges():
    df = pd.read_csv(require(DATA / "prob_skill.csv"))
    for col in ["variable", "region", "week", "init_date", "model",
                "crps", "crps_clim", "crpss", "rmse", "spread", "ssr"]:
        assert col in df.columns, f"prob_skill.csv missing {col}"
    assert (df.crps.dropna() >= 0).all()
    assert (df.spread.dropna() >= 0).all()
    assert (df.ssr.dropna() >= 0).all()
    assert (df.crpss.dropna() <= 1.0001).all()        # skill score upper-bounded by 1


def test_era5_daily_references_exist_and_sane():
    import xarray as xr
    tp = xr.open_dataset(require(DATA / "era5_daily_tp.nc"))["tp"]
    t2 = xr.open_dataset(require(DATA / "era5_daily_t2m.nc"))["t2m"]
    assert tp.sizes.get("time", 0) >= 120                      # ~135 days
    assert 0.5 < float(tp.mean()) < 5.0                        # mm/day domain mean
    assert 270 < float(t2.mean()) < 305                        # Kelvin, India winter/spring


def test_scatter_points_are_anomalies():
    npz = np.load(require(DATA / "scatter_points.npz"))
    for var in ["TP", "Z500", "T2M"]:
        for m in MODELS:
            k = f"{var}_{m}_obs"
            if k not in npz.files:
                continue
            arr = npz[k]
            assert np.isfinite(arr).all()
            # observed anomalies are centred near zero (climatology removed)
            assert abs(float(arr.mean())) < 0.5 * float(arr.std()) + 1e-6, \
                f"{k}: looks un-centred (not an anomaly?)"
