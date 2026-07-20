# Ground truth used in the June 2026 case plots

`all_india_daily_ground_truth_20260602_20260718.csv` contains cosine-latitude-weighted means over the union of the four IMD homogeneous regions.

- `imerg_late_mm`: NASA GPM IMERG Late daily V07, `GPM_3IMERGDL.07`, variable `precipitation` in mm/day. DOI: `10.5067/GPM/IMERGDL/DAY/07`. The daily interval is UTC.
- `imd_gauge_mm`: IMD Pune real-time daily gridded gauge rainfall, 0.25 degree, 135 x 129 grid. IMD daily values follow the IMD daily observing convention rather than an exact UTC boundary.
- `imd_gauge_gpm_merged_mm`: IMD Pune real-time daily merged satellite-gauge GPM rainfall, 0.25 degree, 241 x 281 grid. This series is retained in the CSV as an additional check but is not drawn on the main figures to avoid confusing it with the independent IMERG line.

Official landing pages:

- NASA IMERG: https://gpm.nasa.gov/data/imerg
- IMD Pune gridded rainfall: https://imdpune.gov.in/lrfindex.php/cmpg/Griddata/cmpg/Product/Rainfall_Data.html

Coverage in this extraction is 2 June through 18 July 2026. IMERG Late is valid through 17 July; the 18 July granule was not yet a readable NetCDF file at extraction time. Both IMD products are present through 18 July.
