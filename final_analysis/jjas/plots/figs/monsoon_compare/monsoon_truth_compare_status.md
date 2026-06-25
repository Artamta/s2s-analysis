# JJAS Monsoon Comparison Status

Mode: `aligned-valid`
ECMWF init count in config: 27
FuXi compact init count: 12
Common exact init count: 0

Cases plotted:
- ECMWF: init `2019-07-09`, lead `10-16`
- FuXi: init `2019-07-11`, lead `8-14`

No exact ECMWF/FuXi paired init exists because the hindcast calendars
are offset in the available archives. Use `--mode aligned-valid` for
the scientifically clean comparison: same ERA5 valid dates, annotated
model-specific initialization dates and lead ranges.

The compact FuXi calendar now has enough dates for representative
case panels. Choose map examples by valid date and lead window, then
use the aggregate matched-window study for the main skill claims.
