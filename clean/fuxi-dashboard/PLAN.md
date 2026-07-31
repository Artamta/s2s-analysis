# One-Day FuXi-S2S Validated Website Prototype

## Goal

Create a small working website for the validated 28 July 2026 FuXi-S2S
forecast, with a scientific-validation layer that can later support daily
automation. Scientific logic remains in tested Python modules and is not
duplicated in the frontend.

## Scope

The prototype publishes one 100-member, 42-day forecast on the native 27 × 27,
1.5° India-domain grid. It provides Weeks 1–6 and four products:

- seven-day rainfall total;
- weekly-mean rainfall model anomaly;
- weekly-mean 2-metre temperature;
- weekly-mean temperature model anomaly.

It also provides source validation and methods pages. Wind, ERPAS controls,
raw-member controls, login, a database, a backend server, and a scheduler are
deliberately excluded.

## Scientific contracts

### FuXi forecast

- 100 stochastic members and 42 lead days.
- 27 × 27 grid from 39°N to 0°N and 60°E to 99°E at 1.5°.
- TP in `mm h-1`; T2M in Kelvin.
- Explicit initialization, information cutoff, model-state time, and
  forecast-period bounds.
- Finite fields, physically plausible ranges, ensemble spread, provenance,
  and SHA-256 validation.

### FuXi model climatology

- Native 2002–2021 reforecast climatology: exactly 20 years.
- Native members are averaged within each year first; annual means then
  receive equal weight.
- The 27 July model-state calendar position is linearly interpolated between
  the native 25 July and 28 July slots, with right-slot weight 2/3.
- TP and T2M share the forecast grid and units.
- The standardized product is currently JJAS-only. This issue is valid, but
  full-year automation requires a full-year climatology.

### Observation climatologies

- IMD: native daily rainfall climatology, baseline exactly `1991-2020`,
  365 calendar keys with an explicit Feb-29 rule, `mm/day`, and stable support.
- IMERG: audited Final V07B 2001–2022 baseline, exactly 22 samples for every
  available calendar key, `mm/day`, conservative remapping, and frozen
  verification support.
- The audited IMERG file is seasonal (6 June–25 October, 142 keys). It fully
  covers this forecast but is explicitly warned as insufficient for year-round
  automation.
- IMERG Late observations and IMERG Final climatology are distinct products.
  IMD observation timing is also labelled separately from UTC model periods.

## Locked formulas

All formulas are implemented once in `science/formulas.py`.

- Daily FuXi rainfall: `tp_mm_hour × 24`.
- Weekly rainfall total: sum of seven daily rainfall fields.
- Weekly rainfall rate: weekly total divided by seven.
- Temperature: `t2m_kelvin − 273.15`.
- FuXi anomaly: forecast weekly mean minus matched FuXi model-climatology
  weekly mean.
- Calendar interpolation: `(1 − w) × left + w × right`.
- IMD anomaly: observation minus IMD 1991–2020 calendar-day climatology.
- IMERG anomaly: observation minus IMERG Final 2001–2022 calendar-day
  climatology.
- Forecast spread: population standard deviation over members (`ddof=0`).
- Climatology spread: sample standard deviation over 20 annual means
  (`ddof=1`).
- Terciles: 1/3 and 2/3 quantiles over annual means.
- Above/below-normal probability: percentage of forecast members beyond the
  corresponding climatological tercile.
- Verification weights: `cos(latitude) × supported land fraction`.
- Bias, MAE, RMSE, and weighted spatial Pearson ACC use identical periods,
  grid, mask, and weights. Scores are withheld until all required observation
  days are available.

## Validation gates

1. File integrity: existence, size, SHA-256, readability, and provenance.
2. Scientific structure: dimensions, coordinates, bounds, units, samples,
   and grid orientation.
3. Physical checks: finite values, nonnegative rainfall, plausible
   temperature, and ensemble spread.
4. Independent formula checks: hand-calculated tests and cross-check against
   the existing 28 July product builder.
5. Publication checks: JSON Schema, finite JSON numbers, no internal paths,
   no credentials, no restricted/raw data, and a deployed-file checksum
   manifest.

`public/data/validation.json` records green, warning, or failure status. The
forecast page does not render a failed product.

## Public data

- `index.json`: available issues and latest successful issue.
- `forecasts/20260728.json`: Weeks 1–6 and four derived fields.
- `sources.json`: public-safe names, periods, grids, units, and checksums.
- `formulas.json`: formula definitions and version.
- `validation.json`: validation results and warnings.
- `india-outline.json`: simplified derived boundary for display.
- `manifest.json`: checksum inventory for deployed data files.

Only compact derived fields and a simplified boundary are public. Full
forecasts, GFS inputs, IMD/IMERG grids, FuXi reforecasts, model weights, and
cluster paths remain outside the production bundle.

## Delivery

The source is a dependency-light TypeScript/Vite static site. A GitHub Actions
workflow builds it, stamps the deployed commit into the generated public
manifest, and deploys through GitHub Pages.

## Next milestones

1. Build a full-year 2002–2021 FuXi climatology.
2. Add date-driven GFS/FuXi automation.
3. Add probability and spread layers.
4. Add automatic IMD and IMERG verification.
5. Add historical FuXi-versus-ERPAS research.
6. Add a rolling one-year archive without changing the frontend data contract.
