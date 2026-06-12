"""
Extra verification utilities added for the JFM-2026 SPIRE benchmark revision:
  * land masking via the offline global_land_mask coastline (ERA5-free, reproducible)
  * bootstrap confidence intervals over the 13 initialization dates
  * paired bootstrap for between-system score differences (SPIRE - FuXi)

These replace the previous normal-approximation CI (1.96*std/sqrt(n)) and the
bounding-box-only domain, which included the Arabian Sea and Bay of Bengal.
"""
import numpy as np
import xarray as xr
from global_land_mask import globe


def get_land_mask(target_lat, target_lon):
    """Boolean (lat, lon) DataArray, True over land, on the verification grid."""
    lon180 = np.where(np.asarray(target_lon) > 180, np.asarray(target_lon) - 360, np.asarray(target_lon))
    LON, LAT = np.meshgrid(lon180, np.asarray(target_lat))
    mask = globe.is_land(LAT, LON)
    return xr.DataArray(mask, coords={'lat': np.asarray(target_lat), 'lon': np.asarray(target_lon)},
                        dims=['lat', 'lon'])


def mask_land(da, land_mask):
    """Keep land points only; ocean -> NaN (xarray weighted metrics skip NaN)."""
    return da.where(land_mask)


def bootstrap_ci(values, n_boot=1000, seed=0, ci=95):
    """Percentile bootstrap over initialization dates. Returns (mean, lo, hi)."""
    vals = np.asarray([v for v in np.asarray(values, dtype=float) if np.isfinite(v)])
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(vals.mean()), float(lo), float(hi)


def paired_bootstrap_diff(values_a, values_b, n_boot=1000, seed=0, ci=95):
    """
    Paired bootstrap of (A - B) over common initializations.
    values_a, values_b are dicts {init_date: score}. Returns (mean_diff, lo, hi, p_two_sided).
    """
    keys = [k for k in values_a if k in values_b
            and np.isfinite(values_a[k]) and np.isfinite(values_b[k])]
    if len(keys) == 0:
        return np.nan, np.nan, np.nan, np.nan
    diff = np.array([values_a[k] - values_b[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    # two-sided bootstrap p for H0: mean diff = 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(min(p, 1.0))
