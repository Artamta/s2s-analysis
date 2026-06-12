import numpy as np
import xarray as xr

def get_cosine_latitude_weights(lat_array):
    """
    Computes a 1D xarray DataArray of cosine-latitude weights.
    WMO standard for spatial verification.
    """
    weights = np.cos(np.deg2rad(lat_array))
    # Normalize to 1.0 (mean weight = 1.0)
    weights = weights / weights.mean()
    return xr.DataArray(weights, coords={'lat': lat_array}, dims=['lat'])

def calc_wmo_acc(f, o, clim, w_da):
    """
    WMO-compliant Anomaly Pattern Correlation Coefficient (APCC).
    Requires cosine-latitude weighting.
    """
    f_anom = f - clim
    o_anom = o - clim
    
    # Cosine-weighted spatial centering
    f_anom_mean = f_anom.weighted(w_da).mean(dim=['lat', 'lon'])
    o_anom_mean = o_anom.weighted(w_da).mean(dim=['lat', 'lon'])
    
    f_centered = f_anom - f_anom_mean
    o_centered = o_anom - o_anom_mean
    
    cov = (f_centered * o_centered).weighted(w_da).mean(dim=['lat', 'lon'])
    var_f = (f_centered**2).weighted(w_da).mean(dim=['lat', 'lon'])
    var_o = (o_centered**2).weighted(w_da).mean(dim=['lat', 'lon'])
    
    return float(np.asarray(cov / np.sqrt(var_f * var_o)).item())

def calc_wmo_rmse(f, o, w_da):
    """WMO-compliant Area-Weighted RMSE"""
    mse = ((f - o)**2).weighted(w_da).mean(dim=['lat', 'lon'])
    return float(np.asarray(np.sqrt(mse)).item())

def calc_wmo_bias(f, o, w_da):
    """WMO-compliant Area-Weighted Mean Bias"""
    return float(np.asarray((f - o).weighted(w_da).mean(dim=['lat', 'lon'])).item())
