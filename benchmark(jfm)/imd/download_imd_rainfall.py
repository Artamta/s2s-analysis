import imdlib as imd
import os
import glob

def download_imd():
    # We download to the central storage drive just like the others
    out_dir = "/storage/raj.ayush/benchmark(jfm)/imd/data"
    os.makedirs(out_dir, exist_ok=True)
    
    # We are evaluating JFM 2026, but some models have a 45-day lead time.
    # So we need ground truth up to mid-May 2026. 
    # imdlib downloads year by year, so we download the entire 2026 dataset.
    start_yr = 2026
    end_yr = 2026
    variable = 'rain' # High-resolution gridded rainfall (0.25 deg)
    
    print(f"Downloading IMD {variable} for {start_yr}-{end_yr}...")
    try:
        # Download the .grd file from IMD Pune Server
        data = imd.get_data(variable, start_yr, end_yr, fn_format='yearwise', file_dir=out_dir)
        print("Download successful! Converting to NetCDF...")
        
        # Convert IMD's proprietary format to standard NetCDF (.nc)
        nc_filename = "imd_rain_2026.nc"
        data.to_netcdf(nc_filename, out_dir)
        
        print(f"Successfully saved NetCDF to {os.path.join(out_dir, nc_filename)}")
        
    except Exception as e:
        print(f"Error downloading IMD data: {e}")

if __name__ == '__main__':
    download_imd()
