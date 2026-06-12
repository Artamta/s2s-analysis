"""
03_skill_maps.py

Compute ACC, RMSE, and bias skill maps from weekly_anomalies.nc.

Input:  weekly_anomalies.nc   (output of Script 2)
Output: skill_metrics.nc

Dimensions of output:
    acc_t2m     (week=4, latitude, longitude)   Pearson CC  [-1, 1]
    rmse_t2m    (week=4, latitude, longitude)   °C
    bias_t2m    (week=4, latitude, longitude)   °C  (Spire − ERA5 mean)
    acc_precip  (week=4, latitude, longitude)
    rmse_precip (week=4, latitude, longitude)   mm/day
    bias_precip (week=4, latitude, longitude)   mm/day
    acc_z500    (week=4, latitude, longitude)
    rmse_z500   (week=4, latitude, longitude)   gpm
    bias_z500   (week=4, latitude, longitude)   gpm

ACC is Pearson correlation across the 90 init dates at every grid point.
"""

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

INPUT_FILE  = "weekly_anomalies.nc"
OUTPUT_FILE = "skill_metrics.nc"

# ── Load anomalies ─────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} …")
ds = xr.open_dataset(INPUT_FILE)
print(ds)

lat  = ds["latitude"].values
lon  = ds["longitude"].values
weeks = ds["week"].values     # [1, 2, 3, 4]
n_inits = ds.sizes["init_time"]
n_weeks = len(weeks)
n_lat   = len(lat)
n_lon   = len(lon)

print(f"\n{n_inits} init dates  |  {n_weeks} weeks  |  grid {n_lat}×{n_lon}")

# ── Helper: vectorised ACC using np.corrcoef ───────────────────────────────────
def compute_acc(fcst: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Pearson CC across init_time dimension at every grid point.

    Args:
        fcst, obs : shape (n_inits, lat, lon)
    Returns:
        acc       : shape (lat, lon)
    """
    # Demean along init_time axis (axis=0)
    f = fcst - fcst.mean(axis=0)
    o = obs  - obs.mean(axis=0)

    num = (f * o).sum(axis=0)
    denom = np.sqrt((f**2).sum(axis=0) * (o**2).sum(axis=0))

    # Avoid division by zero where a grid point is constant
    with np.errstate(invalid="ignore"):
        acc = np.where(denom > 0, num / denom, np.nan)
    return acc.astype(np.float32)


def compute_rmse(fcst: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """RMSE across init_time at every grid point. Shape (lat, lon)."""
    return np.sqrt(((fcst - obs) ** 2).mean(axis=0)).astype(np.float32)


def compute_bias(fcst: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Mean bias (Spire − ERA5) across init_time. Shape (lat, lon)."""
    return (fcst - obs).mean(axis=0).astype(np.float32)


# ── Allocate output arrays ─────────────────────────────────────────────────────
shape = (n_weeks, n_lat, n_lon)
results = {
    "acc_t2m":     np.full(shape, np.nan, np.float32),
    "rmse_t2m":    np.full(shape, np.nan, np.float32),
    "bias_t2m":    np.full(shape, np.nan, np.float32),
    "acc_precip":  np.full(shape, np.nan, np.float32),
    "rmse_precip": np.full(shape, np.nan, np.float32),
    "bias_precip": np.full(shape, np.nan, np.float32),
    "acc_z500":    np.full(shape, np.nan, np.float32),
    "rmse_z500":   np.full(shape, np.nan, np.float32),
    "bias_z500":   np.full(shape, np.nan, np.float32),
}

# ── Compute metrics for each week ─────────────────────────────────────────────
print("\nComputing skill metrics …")

for wi, wk in enumerate(weeks):
    sp_t2m    = ds["spire_t2m_anom"].sel(week=wk).values     # (90, lat, lon)
    e5_t2m    = ds["era5_t2m_anom"].sel(week=wk).values
    sp_precip = ds["spire_precip_anom"].sel(week=wk).values
    e5_precip = ds["era5_precip_anom"].sel(week=wk).values
    sp_z500   = ds["spire_z500_anom"].sel(week=wk).values
    e5_z500   = ds["era5_z500_anom"].sel(week=wk).values

    results["acc_t2m"][wi]     = compute_acc(sp_t2m,    e5_t2m)
    results["rmse_t2m"][wi]    = compute_rmse(sp_t2m,   e5_t2m)
    results["bias_t2m"][wi]    = compute_bias(sp_t2m,   e5_t2m)

    results["acc_precip"][wi]  = compute_acc(sp_precip,  e5_precip)
    results["rmse_precip"][wi] = compute_rmse(sp_precip, e5_precip)
    results["bias_precip"][wi] = compute_bias(sp_precip, e5_precip)

    results["acc_z500"][wi]    = compute_acc(sp_z500,    e5_z500)
    results["rmse_z500"][wi]   = compute_rmse(sp_z500,   e5_z500)
    results["bias_z500"][wi]   = compute_bias(sp_z500,   e5_z500)

    # Quick sanity print — area-mean ACC over India core box (8–35°N, 68–98°E)
    lat_mask = (lat >= 8)  & (lat <= 35)
    lon_mask = (lon >= 68) & (lon <= 98)
    india_acc_t2m    = np.nanmean(results["acc_t2m"][wi][np.ix_(lat_mask, lon_mask)])
    india_acc_precip = np.nanmean(results["acc_precip"][wi][np.ix_(lat_mask, lon_mask)])
    india_acc_z500   = np.nanmean(results["acc_z500"][wi][np.ix_(lat_mask, lon_mask)])
    print(f"  W{wk}  ACC  T2M={india_acc_t2m:.3f}  Precip={india_acc_precip:.3f}  Z500={india_acc_z500:.3f}")

# ── Pack into xarray Dataset and save ─────────────────────────────────────────
print("\nPacking dataset …")

dims   = ["week", "latitude", "longitude"]
coords = {"week": weeks, "latitude": lat, "longitude": lon}

def make_da(arr, long_name, units):
    return xr.DataArray(arr, dims=dims, coords=coords,
                        attrs={"long_name": long_name, "units": units})

ds_out = xr.Dataset({
    "acc_t2m":     make_da(results["acc_t2m"],     "T2M ACC",          "1"),
    "rmse_t2m":    make_da(results["rmse_t2m"],    "T2M RMSE",         "°C"),
    "bias_t2m":    make_da(results["bias_t2m"],    "T2M bias",         "°C"),
    "acc_precip":  make_da(results["acc_precip"],  "Precip ACC",       "1"),
    "rmse_precip": make_da(results["rmse_precip"], "Precip RMSE",      "mm/day"),
    "bias_precip": make_da(results["bias_precip"], "Precip bias",      "mm/day"),
    "acc_z500":    make_da(results["acc_z500"],     "Z500 ACC",         "1"),
    "rmse_z500":   make_da(results["rmse_z500"],    "Z500 RMSE",        "gpm"),
    "bias_z500":   make_da(results["bias_z500"],    "Z500 bias",        "gpm"),
}, attrs={
    "description": "Skill metrics — Spire JFM 2026 vs ERA5, India domain",
    "n_inits":     n_inits,
    "method_acc":  "Pearson CC across 90 init dates at each grid point",
})

ds_out.to_netcdf(OUTPUT_FILE)
print(f"Saved → {OUTPUT_FILE}")
print(ds_out)
