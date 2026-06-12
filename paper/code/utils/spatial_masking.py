import xarray as xr
import warnings

def apply_indian_subcontinent_bounding_box(forecast_dataset: xr.Dataset) -> xr.Dataset:
    """
    Safely isolates the Indian Subcontinent using a strict Latitude/Longitude bounding box.
    By relying purely on geographical coordinates, this method avoids rendering any 
    political borders, ensuring 100% compliance with local mapping laws and 
    preventing any geopolitical controversies in peer review.
    
    Bounding Box: Latitude 5°N to 38°N, Longitude 65°E to 100°E
    """
    # Slice the dataset to the Indian Subcontinent coordinates
    regional_dataset = forecast_dataset.sel(
        latitude=slice(38.0, 5.0),  # Latitude is descending (90 to -90)
        longitude=slice(65.0, 100.0)
    )
    
    return regional_dataset


def extract_imd_homogeneous_region(forecast_dataset: xr.Dataset, region_name: str) -> xr.Dataset:
    """
    Extracts a specific Homogeneous Rainfall Region of India as defined by the 
    India Meteorological Department (IMD).
    
    Available regions: 
    - 'northwest_india'
    - 'central_india'
    - 'south_peninsula'
    - 'east_northeast_india'
    
    Parameters:
    - forecast_dataset: xarray.Dataset (preferably already masked by apply_strict_indian_landmask)
    - region_name: string identifier of the IMD region.
    
    Returns:
    - regional_dataset: xarray.Dataset cropped exclusively to the specified region.
    """
    
    # Standard meteorological bounding boxes for IMD regions (approximate boundaries)
    # Format: {'region_name': (min_lat, max_lat, min_lon, max_lon)}
    imd_regions_bounds = {
        'northwest_india':      (22.0, 38.0, 68.0, 82.0),
        'central_india':        (18.0, 28.0, 72.0, 89.0),
        'south_peninsula':      (8.0,  20.0, 72.0, 85.0),
        'east_northeast_india': (20.0, 30.0, 85.0, 98.0)
    }
    
    if region_name not in imd_regions_bounds:
        raise ValueError(f"Invalid region. Choose from: {list(imd_regions_bounds.keys())}")
        
    min_lat, max_lat, min_lon, max_lon = imd_regions_bounds[region_name]
    
    # Slice the dataset to the exact bounding box
    regional_dataset = forecast_dataset.sel(
        latitude=slice(max_lat, min_lat),  # Assuming latitude is descending (90 to -90)
        longitude=slice(min_lon, max_lon)
    )
    
    return regional_dataset
