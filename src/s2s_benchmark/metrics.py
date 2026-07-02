"""Metric library shared by JFM2026 and JJAS2019.

All functions expect fields that already share a lat/lon grid. They skip NaNs,
so callers can mask ocean or regions before scoring.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .grid import cosine_weights, normalize_lat_lon

try:
    from scipy.special import erf as _erf
except ImportError:  # pragma: no cover - scipy is expected, fallback is safe.
    from math import erf as _math_erf

    def _erf(values):
        return np.vectorize(_math_erf)(values)


def _broadcast_weights(da: xr.DataArray, weights: xr.DataArray | None = None) -> xr.DataArray:
    da = normalize_lat_lon(da)
    if weights is None:
        weights = cosine_weights(da["lat"])
    return weights.broadcast_like(da)


def weighted_mean(da: xr.DataArray, weights: xr.DataArray | None = None) -> float:
    """Finite-cell weighted mean over lat/lon."""

    da = normalize_lat_lon(da)
    w = _broadcast_weights(da, weights)
    ok = np.isfinite(da)
    den = w.where(ok).sum(("lat", "lon"), skipna=True)
    if float(den) == 0.0:
        return float("nan")
    return float((da.where(ok) * w.where(ok)).sum(("lat", "lon"), skipna=True) / den)


def _align_fields(*fields: xr.DataArray) -> tuple[xr.DataArray, ...]:
    """Normalize and align fields on shared lat/lon coordinates."""

    return xr.align(*(normalize_lat_lon(field) for field in fields), join="inner")


def bias(forecast: xr.DataArray, truth: xr.DataArray, weights: xr.DataArray | None = None) -> float:
    """Weighted mean error, forecast minus truth."""

    f, t = _align_fields(forecast, truth)
    return weighted_mean(f - t, weights)


def mae(forecast: xr.DataArray, truth: xr.DataArray, weights: xr.DataArray | None = None) -> float:
    """Weighted mean absolute error."""

    f, t = _align_fields(forecast, truth)
    diff = abs(f - t)
    return weighted_mean(diff, weights)


def rmse(forecast: xr.DataArray, truth: xr.DataArray, weights: xr.DataArray | None = None) -> float:
    """Weighted root-mean-square error."""

    f, t = _align_fields(forecast, truth)
    diff = f - t
    mse = weighted_mean(diff * diff, weights)
    return float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")


def _weighted_corr(a: xr.DataArray, b: xr.DataArray, weights: xr.DataArray | None = None) -> float:
    """Weighted spatial correlation over finite shared grid cells."""

    aa, bb = _align_fields(a, b)
    w = _broadcast_weights(aa, weights)
    ok = np.isfinite(aa) & np.isfinite(bb) & np.isfinite(w)
    if int(ok.sum()) < 3:
        return float("nan")

    ww = w.where(ok)
    den = ww.sum(("lat", "lon"), skipna=True)
    if float(den) == 0.0:
        return float("nan")
    ww = ww / den

    am = (aa.where(ok) * ww).sum(("lat", "lon"), skipna=True)
    bm = (bb.where(ok) * ww).sum(("lat", "lon"), skipna=True)
    anom = aa - am
    bnom = bb - bm
    cov = (anom.where(ok) * bnom.where(ok) * ww).sum(("lat", "lon"), skipna=True)
    avar = ((anom.where(ok) ** 2) * ww).sum(("lat", "lon"), skipna=True)
    bvar = ((bnom.where(ok) ** 2) * ww).sum(("lat", "lon"), skipna=True)
    denom = np.sqrt(float(avar) * float(bvar))
    return float(cov) / denom if denom > 0 else float("nan")


def acc(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    climatology: xr.DataArray,
    weights: xr.DataArray | None = None,
) -> float:
    """Weighted anomaly correlation coefficient relative to a climatology field.

    ``climatology`` must already be selected for the same valid period as the
    forecast/truth field, then regridded or interpolated to the scoring grid.
    """

    f, t, clim = _align_fields(forecast, truth, climatology)
    return _weighted_corr(f - clim, t - clim, weights)


def score_deterministic(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    *,
    climatology: xr.DataArray,
    weights: xr.DataArray | None = None,
) -> dict[str, float]:
    """Return the core deterministic scores used throughout the benchmark."""

    return {
        "acc": acc(forecast, truth, climatology, weights),
        "rmse": rmse(forecast, truth, weights),
        "bias": bias(forecast, truth, weights),
        "mae": mae(forecast, truth, weights),
    }


def ensemble_mean_spread(
    ensemble: xr.DataArray,
    member_dim: str = "member",
    ddof: int = 1,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ensemble mean and spread over the member dimension."""

    ens = normalize_lat_lon(ensemble)
    if member_dim not in ens.dims:
        raise ValueError(f"member dimension {member_dim!r} not found in {ens.dims}")
    return ens.mean(member_dim, skipna=True), ens.std(member_dim, ddof=ddof, skipna=True)


def crps_ensemble(
    ensemble: xr.DataArray,
    truth: xr.DataArray,
    weights: xr.DataArray | None = None,
    member_dim: str = "member",
) -> float:
    """Weighted ensemble CRPS using the finite-member sample formula.

    CRPS = mean_i |x_i - y| - 0.5 mean_ij |x_i - x_j|
    """

    ens, obs = _align_fields(ensemble, truth)
    if member_dim not in ens.dims:
        raise ValueError(f"member dimension {member_dim!r} not found in {ens.dims}")

    term1 = abs(ens - obs).mean(member_dim, skipna=True)
    left = ens.rename({member_dim: "_member_i"})
    right = ens.rename({member_dim: "_member_j"})
    term2 = abs(left - right).mean(("_member_i", "_member_j"), skipna=True)
    return weighted_mean(term1 - 0.5 * term2, weights)


def _normal_cdf(x: xr.DataArray) -> xr.DataArray:
    """Standard-normal CDF with scipy when available and a local fallback."""

    return xr.apply_ufunc(
        lambda values: 0.5 * (1.0 + _erf(values / np.sqrt(2.0))),
        x,
        dask="allowed",
    )


def _normal_pdf(x: xr.DataArray) -> xr.DataArray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def crps_gaussian(
    mean: xr.DataArray,
    spread: xr.DataArray,
    truth: xr.DataArray,
    weights: xr.DataArray | None = None,
    sigma_floor: float = 1e-6,
) -> float:
    """Weighted Gaussian CRPS for mean/spread forecast systems."""

    mu, sig, obs = _align_fields(mean, spread, truth)
    sig = abs(sig).clip(min=sigma_floor)
    z = (obs - mu) / sig
    crps = sig * (z * (2.0 * _normal_cdf(z) - 1.0) + 2.0 * _normal_pdf(z) - 1.0 / np.sqrt(np.pi))
    return weighted_mean(crps, weights)


def crps_skill_score(crps_model: float, crps_reference: float) -> float:
    """CRPS skill score: positive is better than reference."""

    return 1.0 - crps_model / crps_reference if crps_reference > 0 else float("nan")


def spread_skill_ratio(
    spread: xr.DataArray,
    forecast_mean: xr.DataArray,
    truth: xr.DataArray,
    weights: xr.DataArray | None = None,
) -> float:
    """Mean spread divided by ensemble-mean RMSE; ideal is near 1."""

    skill = rmse(forecast_mean, truth, weights)
    mean_spread = weighted_mean(spread, weights)
    return mean_spread / skill if skill > 0 else float("nan")


def spread_skill_ratio_ensemble(
    ensemble: xr.DataArray,
    truth: xr.DataArray,
    weights: xr.DataArray | None = None,
    member_dim: str = "member",
    spread_ddof: int = 1,
) -> float:
    """Spread-skill ratio from raw ensemble members."""

    mean, spread = ensemble_mean_spread(ensemble, member_dim=member_dim, ddof=spread_ddof)
    return spread_skill_ratio(spread, mean, truth, weights)


def probability_exceedance(
    forecast: xr.DataArray,
    threshold: float,
    member_dim: str = "member",
) -> xr.DataArray:
    """Ensemble probability of exceeding a threshold."""

    fcst = normalize_lat_lon(forecast)
    if member_dim not in fcst.dims:
        raise ValueError(f"member dimension {member_dim!r} not found in {fcst.dims}")
    valid_members = fcst.notnull()
    valid_cell = valid_members.any(member_dim)
    return (fcst > threshold).where(valid_members).mean(member_dim, skipna=True).where(valid_cell)


def probability_below(
    forecast: xr.DataArray,
    threshold: float,
    member_dim: str = "member",
) -> xr.DataArray:
    """Ensemble probability of falling below a threshold."""

    fcst = normalize_lat_lon(forecast)
    if member_dim not in fcst.dims:
        raise ValueError(f"member dimension {member_dim!r} not found in {fcst.dims}")
    valid_members = fcst.notnull()
    valid_cell = valid_members.any(member_dim)
    return (fcst < threshold).where(valid_members).mean(member_dim, skipna=True).where(valid_cell)


def gaussian_probability_exceedance(
    mean: xr.DataArray,
    spread: xr.DataArray,
    threshold: float,
    sigma_floor: float = 1e-6,
) -> xr.DataArray:
    """Gaussian probability of exceeding a threshold."""

    mu, sig = _align_fields(mean, spread)
    sig = abs(sig).clip(min=sigma_floor)
    return 1.0 - _normal_cdf((threshold - mu) / sig)


def gaussian_probability_below(
    mean: xr.DataArray,
    spread: xr.DataArray,
    threshold: float,
    sigma_floor: float = 1e-6,
) -> xr.DataArray:
    """Gaussian probability of falling below a threshold."""

    mu, sig = _align_fields(mean, spread)
    sig = abs(sig).clip(min=sigma_floor)
    return _normal_cdf((threshold - mu) / sig)


def brier_score(
    probability: xr.DataArray,
    outcome: xr.DataArray,
    weights: xr.DataArray | None = None,
) -> float:
    """Weighted Brier score for a binary event."""

    prob, event = _align_fields(probability, outcome.astype(float))
    return weighted_mean((prob - event) ** 2, weights)


def brier_skill_score(bs_model: float, bs_reference: float) -> float:
    """Brier skill score: positive is better than reference."""

    return 1.0 - bs_model / bs_reference if bs_reference > 0 else float("nan")


def reliability_bins(
    probability: xr.DataArray,
    outcome: xr.DataArray,
    nbins: int = 10,
) -> list[dict[str, float]]:
    """Return reliability-bin counts, mean forecast probability and obs frequency."""

    prob, event = _align_fields(probability, outcome.astype(float))
    p = prob.values.ravel()
    y = event.values.ravel()
    ok = np.isfinite(p) & np.isfinite(y)
    p = p[ok]
    y = y[ok]
    edges = np.linspace(0.0, 1.0, nbins + 1)
    indices = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, nbins - 1)
    rows = []
    for i in range(nbins):
        sel = indices == i
        count = int(sel.sum())
        rows.append(
            {
                "bin": i,
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": count,
                "forecast_probability": float(np.mean(p[sel])) if count else float("nan"),
                "observed_frequency": float(np.mean(y[sel])) if count else float("nan"),
            }
        )
    return rows


def score_probabilistic_ensemble(
    ensemble: xr.DataArray,
    truth: xr.DataArray,
    weights: xr.DataArray | None = None,
    member_dim: str = "member",
    spread_ddof: int = 1,
) -> dict[str, float]:
    """Return core probabilistic scores for a raw ensemble."""

    mean, spread = ensemble_mean_spread(ensemble, member_dim=member_dim, ddof=spread_ddof)
    return {
        "crps": crps_ensemble(ensemble, truth, weights, member_dim=member_dim),
        "spread_skill_ratio": spread_skill_ratio(spread, mean, truth, weights),
    }
