import numpy as np
import xarray as xr

# ==========================================
# GEOGRAPHIC MASKS
# ==========================================

def get_full_domain(ds):
    """
    Returns the full downloaded domain (0–50°N, 55–105°E).
    This includes the surrounding oceans, Himalayas, and neighboring countries.
    """
    return ds

def get_india_mainland(ds):
    """
    Strictly crops the dataset to the exact bounding box of Mainland India 
    (roughly 8°N - 38°N, 68°E - 98°E).
    This removes the deep ocean and far-off landmasses from the calculation.
    """
    # Xarray handles slicing beautifully. 
    # Note: Depending on if latitude is ascending or descending, slice order matters.
    # We use sortby to make it foolproof.
    ds_sorted = ds.sortby(['latitude', 'longitude'])
    return ds_sorted.sel(latitude=slice(8.0, 38.0), longitude=slice(68.0, 98.0))

def get_northwest_india(ds):
    """IMD Homogeneous Region: Northwest India (NWI). Covers J&K, Punjab, Rajasthan."""
    ds_sorted = ds.sortby(['latitude', 'longitude'])
    return ds_sorted.sel(latitude=slice(21.0, 38.0), longitude=slice(68.0, 80.0))

def get_central_india(ds):
    """IMD Homogeneous Region: Central India (CI). Covers Gujarat, MP, Maharashtra."""
    ds_sorted = ds.sortby(['latitude', 'longitude'])
    return ds_sorted.sel(latitude=slice(15.0, 28.0), longitude=slice(73.0, 86.0))

def get_south_peninsular_india(ds):
    """IMD Homogeneous Region: South Peninsular India (SPI). Covers Kerala, TN, Karnataka."""
    ds_sorted = ds.sortby(['latitude', 'longitude'])
    return ds_sorted.sel(latitude=slice(8.0, 20.0), longitude=slice(72.0, 85.0))

def get_northeast_india(ds):
    """IMD Homogeneous Region: East & Northeast India (ENEI). Covers Bengal, Assam, NE States."""
    ds_sorted = ds.sortby(['latitude', 'longitude'])
    return ds_sorted.sel(latitude=slice(21.0, 29.0), longitude=slice(84.0, 98.0))

# ==========================================
# CORE MATHEMATICAL METRICS
# ==========================================

def calculate_rmse(forecast, truth, dims=('latitude', 'longitude')):
    """
    Root Mean Square Error (RMSE).
    Measures the absolute magnitude of the error between the forecast and the true ERA5 data.
    
    Returns a spatial mean (one number per lead_time) by default.
    """
    squared_error = (forecast - truth) ** 2
    rmse = np.sqrt(squared_error.mean(dim=dims))
    return rmse

def calculate_bias(forecast, truth, dims=('latitude', 'longitude')):
    """
    Mean Bias.
    Measures if the model systematically overpredicts (positive) or underpredicts (negative).
    """
    bias = (forecast - truth).mean(dim=dims)
    return bias

def calculate_acc(forecast_anom, truth_anom, dims=('latitude', 'longitude')):
    """
    Anomaly Correlation Coefficient (ACC).
    This is the gold standard for S2S models. It requires you to calculate 
    the anomalies FIRST (Forecast - Climatology_Mean) before passing them here.
    
    It calculates the Pearson correlation coefficient over the spatial grid.
    Score ranges from -1 to 1. A score above 0.6 is considered "useful skill".
    """
    # Numerator: Covariance of the anomalies
    covar = (forecast_anom * truth_anom).mean(dim=dims)
    
    # Denominator: Standard Deviation of the anomalies
    f_std = np.sqrt((forecast_anom ** 2).mean(dim=dims))
    t_std = np.sqrt((truth_anom ** 2).mean(dim=dims))
    
    acc = covar / (f_std * t_std)
    return acc

def calculate_ensemble_spread(ensemble_forecast, dim='member'):
    """
    Ensemble Spread.
    Calculates the standard deviation of the forecast across all ensemble members.
    Helps us understand how 'uncertain' the AI or physics model is.
    """
    return ensemble_forecast.std(dim=dim)
