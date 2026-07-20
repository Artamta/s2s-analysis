# IMD precipitation story plots

These plots intentionally use **IMD gauge rainfall as the only verification
reference**. ERA5 and IMERG are excluded. IMD 1991–2020 climatology appears in
the first figure only as context, never as an observation or skill target.

## Recommended order and talk track

1. **Full 42-day verification:** In the 1 June 2026 case, FuXi follows the IMD
   cumulative trajectory more closely than ECMWF. Its trajectory RMSE is
   40.7 mm versus 62.9 mm (35% lower).
2. **Where the difference comes from:** FuXi has the smaller weekly absolute
   error in 5 of 6 weeks. Both forecasts are too wet overall; this plot makes that limitation visible.
3. **Does the signal repeat?:** FuXi is closer to IMD in the second June issue
   as well, but that forecast is only partially verified and the two cases
   overlap. Present this as case evidence—not as a general skill claim.

## Files

- `01_imd_42day_cumulative_20260601.png`
- `02_imd_weekly_rainfall_and_bias_20260601.png`
- `03_imd_two_case_scorecard.png`

## Method note

All-India means use the cosine-latitude-weighted union of the four IMD
homogeneous-region masks. Forecast trajectories are ensemble means; shading in
Figure 1 is the 10–90% ensemble range. IMD real-time 0.25° gauge rainfall uses
the product's stated daily convention (valid near 03 UTC), while forecast
period endpoints are at 00 UTC, leaving an approximately three-hour timing
offset.
