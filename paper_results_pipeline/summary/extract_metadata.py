import os
import xarray as xr
import cfgrib
import numpy as np

OUT_MD = "/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/summary/model_data_dictionary.md"

def inspect_xarray_dataset(ds, model_name):
    """Extracts metadata from a single xarray Dataset into a markdown chunk."""
    md = f"### {model_name}\n\n"
    
    # Dimensions
    md += "**Dimensions & Grid:**\n"
    for dim, size in ds.dims.items():
        md += f"- `{dim}`: {size}\n"
        
    # Variables
    md += "\n**Variables & Units:**\n"
    for var_name, variable in ds.data_vars.items():
        long_name = variable.attrs.get('long_name', variable.attrs.get('GRIB_name', 'N/A'))
        units = variable.attrs.get('units', 'N/A')
        md += f"- `{var_name}`: {long_name} ({units})\n"
        
    # Coordinates / Lead Times / Ensembles
    md += "\n**Key Coordinates Details:**\n"
    if 'step' in ds.coords:
        steps = ds['step'].values
        try:
            md += f"- **Lead Times (step)**: {len(steps)} steps, Max: {steps[-1]}\n"
        except:
            md += f"- **Lead Times (step)**: scalar value: {steps}\n"
    elif 'lead_time' in ds.coords:
        steps = ds['lead_time'].values
        try:
            md += f"- **Lead Times**: {len(steps)} steps, Max: {steps[-1]}\n"
        except:
             md += f"- **Lead Times**: scalar value: {steps}\n"
             
    if 'number' in ds.coords:
        nums = ds['number'].values
        md += f"- **Ensemble Members (number)**: {len(np.atleast_1d(nums))} members.\n"
    
    md += "\n---\n"
    return md

def main():
    with open(OUT_MD, 'w') as f:
        f.write("# S2S Master Model Data Dictionary\n\n")
        f.write("This document contains the exact variables, units, lead times, and grid dimensions for all 5 datasets used in the analysis.\n\n")

    # 1. ERA5
    try:
        ds_era5 = xr.open_dataset("/storage/raj.ayush/benchmark(jfm)/era5/data/era5_surface_z500_20260101.nc")
        md = inspect_xarray_dataset(ds_era5, "ERA5 (Ground Truth)")
        with open(OUT_MD, 'a') as f: f.write(md)
    except Exception as e:
        print(f"ERA5 Error: {e}")

    # 2. FuXi
    try:
        ds_fuxi = xr.open_dataset("/storage/raj.ayush/fuxi-init-jfm-weekely/data/20260101/input.nc")
        md = inspect_xarray_dataset(ds_fuxi, "FuXi S2S")
        with open(OUT_MD, 'a') as f: f.write(md)
    except Exception as e:
        print(f"FuXi Error: {e}")

    # 3. Spire
    try:
        import zarr
        ds_spire = xr.open_zarr("/storage/raj.ayush/spire-hindecast-weekely-initialized/spire_hindcast_jfm.zarr", group="mean_stddev", consolidated=False)
        md = inspect_xarray_dataset(ds_spire, "Spire (mean_stddev group)")
        with open(OUT_MD, 'a') as f: f.write(md)
    except Exception as e:
        print(f"Spire Error: {e}")

    # 4. ECMWF
    try:
        md = "### ECMWF (S2S Archive)\n\n"
        dsets = cfgrib.open_datasets("/storage/raj.ayush/benchmark(jfm)/ecmwf/data/sfc_pf_20260101.grib")
        for i, ds in enumerate(dsets):
            md += f"#### GRIB Message Group {i+1}\n"
            md += inspect_xarray_dataset(ds, f"ECMWF Group {i+1}").replace('###', '')
        with open(OUT_MD, 'a') as f: f.write(md)
    except Exception as e:
        print(f"ECMWF Error: {e}")

    # 5. NCEP
    try:
        md = "### NCEP (S2S Archive)\n\n"
        dsets = cfgrib.open_datasets("/storage/raj.ayush/benchmark(jfm)/ncep/data/sfc_pf_20260101.grib")
        for i, ds in enumerate(dsets):
            md += f"#### GRIB Message Group {i+1}\n"
            md += inspect_xarray_dataset(ds, f"NCEP Group {i+1}").replace('###', '')
        with open(OUT_MD, 'a') as f: f.write(md)
    except Exception as e:
        print(f"NCEP Error: {e}")

    print(f"Successfully generated dictionary at {OUT_MD}")

if __name__ == '__main__':
    main()
