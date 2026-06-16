#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics.py  —  Verification formulas for the S2S benchmark (pure, no I/O).
================================================================================
Every function here takes already-gridded, land-masked xarray DataArrays
(f = forecast, o = observation, clim = climatology) plus a cosine-latitude
weight DataArray `w`, and returns plain floats. NaN / ocean points are skipped
automatically by xarray's `.weighted().mean()`.

These are the SINGLE source of truth for the maths, so they can be unit-tested
and read in isolation. Definitions follow WMO spatial-verification practice and
Gneiting & Raftery (2007) for the Gaussian CRPS.

Conventions
-----------
  weight   w(lat) = cos(lat) / mean(cos(lat))        (mean weight = 1.0)
  anomaly  a      = field - climatology
  <x>             = area-weighted spatial mean of x over (lat, lon)

Deterministic
-------------
  ACC/PCC  = < (af-<af>)(ao-<ao>) > / sqrt(<(af-<af>)^2><(ao-<ao>)^2>)   (centred APCC)
  RMSE     = sqrt( <(f-o)^2> )
  bias     = < f - o >
  MSSS     = 1 - MSE(f,o) / MSE(ref,o)              (ref = climatology OR persistence)
  std_ratio= std_w(af) / std_w(ao)                 (amplitude fidelity)

Probabilistic (Gaussian forecast N(mu, sigma))
----------------------------------------------
  CRPS(N(mu,s), y) = s * [ z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],  z=(y-mu)/s
  CRPSS    = 1 - CRPS_model / CRPS_ref             (ref = climatology OR persistence)
  SSR      = mean(sigma) / RMSE(mu, o)             (=1 well-calibrated)

Event / Brier (binary event E)
------------------------------
  Brier    = < (P_fcst(E) - 1{o in E})^2 >
  BrierSS  = 1 - Brier_model / Brier_ref           (ref = climatological frequency)
================================================================================
"""
import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------------ weighting --
def cos_latitude_weights(lat_array, xr):
    """Cosine-latitude weights normalised to mean 1.0 (WMO). Returns DataArray."""
    w = np.cos(np.deg2rad(np.asarray(lat_array)))
    w = w / w.mean()
    return xr.DataArray(w, coords={'lat': np.asarray(lat_array)}, dims=['lat'])


def wmean(x, w):
    """Cosine-weighted spatial mean over (lat, lon). Returns DataArray scalar."""
    return x.weighted(w).mean(['lat', 'lon'])


def _f(x):
    """DataArray scalar -> python float."""
    return float(np.asarray(x).item())


# --------------------------------------------------------------- deterministic --
def wstd_anom(a, w):
    """Cosine-weighted spatial std of anomaly field `a` about its own mean."""
    m = wmean(a, w)
    return _f(np.sqrt(wmean((a - m) ** 2, w)))


def acc(f, o, clim, w):
    """WMO anomaly pattern correlation (spatially centred, area-weighted)."""
    af, ao = f - clim, o - clim
    afc = af - wmean(af, w)
    aoc = ao - wmean(ao, w)
    cov = wmean(afc * aoc, w)
    vf = wmean(afc ** 2, w)
    vo = wmean(aoc ** 2, w)
    return _f(cov / np.sqrt(vf * vo))


def rmse(f, o, w):
    """Cosine-weighted area RMSE."""
    return _f(np.sqrt(wmean((f - o) ** 2, w)))


def bias(f, o, w):
    """Cosine-weighted mean bias < f - o >."""
    return _f(wmean(f - o, w))


def msss(f, o, ref, w):
    """MSE skill score vs a reference field `ref`: 1 - MSE(f,o)/MSE(ref,o)."""
    mse_f = _f(wmean((f - o) ** 2, w))
    mse_r = _f(wmean((ref - o) ** 2, w))
    return (1.0 - mse_f / mse_r) if mse_r > 0 else np.nan


def std_ratio(f, o, clim, w):
    """Forecast-vs-obs anomaly spatial-std ratio (amplitude fidelity)."""
    sf = wstd_anom(f - clim, w)
    so = wstd_anom(o - clim, w)
    return (sf / so) if so > 0 else np.nan


# --------------------------------------------------------------- probabilistic --
def crps_gauss(mu, sig, y):
    """Closed-form CRPS of a Gaussian(mu, sigma) forecast vs obs y (arrays).
       Gneiting & Raftery (2007). Element-wise over the grid."""
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-6)
    z = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sig
    return sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def crpss(crps_model, crps_ref):
    """CRPS skill score vs a reference: 1 - CRPS_model/CRPS_ref."""
    return (1.0 - crps_model / crps_ref) if (crps_ref and crps_ref > 0) else np.nan


def ssr(mean_spread, ens_mean_rmse):
    """Spread-skill ratio = mean ensemble spread / ensemble-mean RMSE."""
    return (mean_spread / ens_mean_rmse) if ens_mean_rmse > 0 else np.nan


# ----------------------------------------------------- Gaussian event probabilities --
def prob_exceed(thr, mu, sig):
    """Gaussian P(Y > thr)."""
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-6)
    return 1.0 - norm.cdf((thr - np.asarray(mu, dtype=float)) / sig)


def prob_below(thr, mu, sig):
    """Gaussian P(Y < thr)."""
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-6)
    return norm.cdf((thr - np.asarray(mu, dtype=float)) / sig)


# ----------------------------------------------------------------------- Brier --
def brier_score(p_fcst, outcome, w, xr_like):
    """Area-weighted Brier score < (P_fcst - outcome)^2 >.
       p_fcst, outcome are numpy arrays on the grid; xr_like is a DataArray
       template (same lat/lon) so we can weight them; w is the weight DataArray."""
    pf = xr_like.copy(data=p_fcst)
    yy = xr_like.copy(data=outcome)
    return _f(wmean((pf - yy) ** 2, w))


def brier_clim(base_rate, outcome, w, xr_like):
    """Climatological-reference Brier: < (base_rate - outcome)^2 >."""
    br = xr_like.copy(data=base_rate)
    yy = xr_like.copy(data=outcome)
    return _f(wmean((br - yy) ** 2, w))


def briss(bs_model, bs_ref):
    """Brier skill score vs climatology: 1 - BS_model/BS_ref."""
    return (1.0 - bs_model / bs_ref) if (bs_ref and bs_ref > 0) else np.nan


# ------------------------------------------------------------------ reliability --
def accumulate_reliability(arr, prob, outcome, nbins):
    """Add one field's (prob, binary-outcome) pairs into a (3, nbins) accumulator:
       row 0 = sum of outcomes, row 1 = count, row 2 = sum of forecast prob.
       Used to build reliability diagrams (obs_freq vs fcst_prob per bin)."""
    p = np.asarray(prob).ravel()
    y = np.asarray(outcome).ravel()
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    idx = np.clip((p * nbins).astype(int), 0, nbins - 1)
    for b in range(nbins):
        sel = idx == b
        if sel.any():
            arr[0, b] += y[sel].sum()
            arr[1, b] += sel.sum()
            arr[2, b] += p[sel].sum()
