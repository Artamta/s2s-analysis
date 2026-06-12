"""
Known-answer unit tests for the verification primitives. These guard the core
maths of every score in the paper.
"""
import numpy as np
import xarray as xr
import pytest

from utils.verification_wmo import (
    get_cosine_latitude_weights, calc_wmo_acc, calc_wmo_rmse, calc_wmo_bias)
from utils.verification_extra import bootstrap_ci, paired_bootstrap_diff

LAT = np.array([10.0, 20.0, 30.0])
LON = np.array([70.0, 80.0, 90.0])
rng = np.random.default_rng(0)


def field(values):
    return xr.DataArray(values, coords={"lat": LAT, "lon": LON}, dims=["lat", "lon"])


@pytest.fixture
def w():
    return get_cosine_latitude_weights(LAT)


# ---------------- cosine weights ----------------
def test_cosine_weights_normalised_and_monotonic():
    w = get_cosine_latitude_weights(LAT)
    assert np.isclose(float(w.mean()), 1.0)            # normalised to mean 1
    assert float(w.sel(lat=10)) > float(w.sel(lat=30))  # more weight near equator


# ---------------- ACC ----------------
def test_acc_perfect_forecast_is_one(w):
    o = field(rng.normal(size=(3, 3)))
    clim = field(np.zeros((3, 3)))
    assert calc_wmo_acc(o, o, clim, w) == pytest.approx(1.0, abs=1e-10)


def test_acc_anticorrelated_is_minus_one(w):
    o = field(rng.normal(size=(3, 3)))
    clim = field(np.zeros((3, 3)))
    assert calc_wmo_acc(-o, o, clim, w) == pytest.approx(-1.0, abs=1e-10)


def test_acc_invariant_to_uniform_offset(w):
    """ACC is on centered anomalies -> adding a constant to the forecast is a no-op."""
    o = field(rng.normal(size=(3, 3)))
    clim = field(np.zeros((3, 3)))
    base = calc_wmo_acc(o * 1.0 + 0.3, o, clim, w)
    off = calc_wmo_acc(o * 1.0 + 0.3 + 5.0, o, clim, w)
    assert base == pytest.approx(off, abs=1e-9)


# ---------------- RMSE / bias ----------------
def test_rmse_zero_for_identical(w):
    o = field(rng.normal(size=(3, 3)))
    assert calc_wmo_rmse(o, o, w) == pytest.approx(0.0, abs=1e-12)


def test_rmse_equals_constant_offset(w):
    o = field(rng.normal(size=(3, 3)))
    assert calc_wmo_rmse(o + 2.5, o, w) == pytest.approx(2.5, abs=1e-9)


def test_bias_sign_and_magnitude(w):
    o = field(rng.normal(size=(3, 3)))
    assert calc_wmo_bias(o + 2.5, o, w) == pytest.approx(2.5, abs=1e-9)
    assert calc_wmo_bias(o - 1.0, o, w) == pytest.approx(-1.0, abs=1e-9)


# ---------------- bootstrap ----------------
def test_bootstrap_ci_brackets_mean():
    vals = rng.normal(size=13)
    m, lo, hi = bootstrap_ci(vals, n_boot=500, seed=1)
    assert m == pytest.approx(float(np.mean(vals)))
    assert lo <= m <= hi


def test_bootstrap_ci_handles_nan_and_empty():
    m, lo, hi = bootstrap_ci([1.0, np.nan, 3.0])
    assert m == pytest.approx(2.0)
    assert all(np.isnan(x) for x in bootstrap_ci([np.nan, np.nan]))


def test_paired_bootstrap_identical_is_zero_diff():
    a = {i: v for i, v in enumerate(rng.normal(size=13))}
    md, lo, hi, p = paired_bootstrap_diff(a, dict(a))
    assert md == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)


def test_paired_bootstrap_detects_offset():
    a = {i: 1.0 for i in range(13)}
    b = {i: 0.0 for i in range(13)}
    md, lo, hi, p = paired_bootstrap_diff(a, b)
    assert md == pytest.approx(1.0)
    assert lo > 0 and p < 0.05          # significantly positive
