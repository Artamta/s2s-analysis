import os
import cdsapi

client = cdsapi.Client()

def main():
    print("Testing CDSAPI connection...")
    try:
        # Just a small test request
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "2m_temperature",
                "year": "2026",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf"
            },
            "test_cds.nc"
        )
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
