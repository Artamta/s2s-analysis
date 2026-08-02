# S2S Research

A static, source-aware India subseasonal research forecast interface with a
separate global viewer in beta. India is the default page. It separates
near-real-time GFS-proxy initializations from delayed ERA5 reference
initializations, exposes the available issue dates and ensemble sizes, and
provides compact issue-level downloads.

The global beta viewer animates 42 daily ensemble-mean fields for
precipitation, 2 m temperature, 500 hPa geopotential height, 850 hPa wind,
mean sea-level pressure, sea-surface temperature, outgoing longwave radiation,
and total-column water vapour.
The site publishes compact ensemble-derived fields only. It does not contain
raw model inputs, member fields, observations, reforecasts, model weights,
credentials, or cluster paths.

## Scientific status

Each forecast package records its initialization-source family explicitly:

- **GFS operational proxy** uses near-real-time daily proxy fields for
  experimental guidance. The model was trained with ERA5-style inputs, and no
  matched GFS-initialized hindcast calibration exists yet.
- **ERA5 reanalysis reference** uses scientifically matched but delayed inputs
  for reference experiments and initialization-sensitivity comparisons. It is
  not a near-real-time forecast.

The 22 July 2026 issue is available from both sources with 100 members, giving
one controlled comparison in which the source is changed without changing the
initialization date. All displayed products remain experimental research
guidance, not an operational weather forecast or warning.

The global animation is an independently sampled 100-member companion ensemble
generated from the same frozen input as the earlier India case. The raw members
behind the India export had already been removed, so its exact stochastic draws
could not be reused.

The public interface uses the neutral identity **S2S Research · Experimental
Subseasonal Forecasting**. Model identity and scientific provenance remain in
this repository.

The downloadable PDF masthead uses unmodified Ashoka University and Safexpress
Centre marks sourced from the official SCDLDS website. These institutional
brand assets are stored under `scripts/assets/`, remain the property of their
respective owners, and are not covered by any source-code licence for this
prototype. The web interface itself retains the neutral S2S Research identity.

The India anomalies use a calendar-interpolated, lead-matched FuXi 2002–2021
native reforecast climatology. They are not IMD or IMERG anomalies. The standardized
FuXi climatology and audited IMERG Final climatology are currently
season-limited; both cover this issue, while full-year automation remains a
pipeline milestone. The India climatology does not extend beyond its 27 × 27
grid. The global viewer uses a separate exact-28-July, lead-matched 2002–2021
model climatology for
precipitation, 2 m temperature, and Z500 anomalies on the full global grid.

## Rebuild and validate

From this directory:

```bash
python scripts/validate_sources.py
python scripts/build_prototype_data.py
python scripts/build_forecast_pdfs.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python scripts/validate_web_data.py
npm ci
npm run build
```

`validate_sources.py` reads the private scientific inputs at their configured
locations but writes only public-safe metadata. Paths can be overridden with
command-line arguments. `build_prototype_data.py` refuses source failures.
`validate_web_data.py` performs schema, finite-number, leakage, checksum, and
independent legacy-builder comparisons.

Run locally with:

```bash
npm run dev
```

## Data contract

The global contract stores little-endian `uint16` binary arrays in lead-day,
latitude, longitude order: eight ensemble means, eight population standard
deviations, three signed anomaly fields, and separate U/V component means for
850 hPa wind. A one-byte
ocean mask limits SST display support. `metadata.json` locks dimensions,
quantization, units, legends, valid periods, file sizes, and SHA-256 hashes.
The default precipitation layer loads first; other fields load and verify
lazily when selected. The complete optional global payload is about 51 MB
before HTTP compression.

Country boundaries and labels use Natural Earth. `india-admin.json` is a
simplified display derivative of the Survey of India Administrative Boundary
Database state/UT layer and contains no raw shapefile.

The India startup path uses `india-map-geography.json`, a compact,
pre-projected display derivative of those same checked geography files. It
preserves the displayed boundaries while avoiding the full global geography
download and thousands of per-cell SVG nodes. Global metadata and forecast
fields are loaded only when the Global Beta route is opened.

The India forecast JSON stores latitude and longitude vectors, a support mask,
six week records, and flattened row-major fields. Every displayed field
contains exactly `latitude.length × longitude.length` finite values.

Source-aware packages live under
`public/data/forecasts/{gfs,era5}/YYYYMMDD.json`. Each issue links to its compact
web JSON and a four-page PDF briefing under
`public/downloads/<source>/YYYYMMDD/`. The public download tree contains PDFs
only; NetCDF, CSV, raw initial conditions, and ensemble-member fields are never
published.

Each PDF contains four product pages with a 2 × 2 Weeks 1–4 map layout. The
interactive website retains all six forecast weeks for exploratory research.

The India presentation follows the supplied four-panel PDF structure: one
shared fixed legend, exact reference colour levels, surrounding-country and
India administrative outlines, and selectable Weeks 1–4 or Weeks 3–6 for all
four products. Interpolation is visual only; hover values remain native-grid
values.

The site has no backend. Hash navigation keeps the GitHub Pages deployment
portable, and a failed validation status blocks forecast rendering while
the scientific checks remain internal to the publication workflow.

## Deployment

The repository-root `.github/workflows/deploy-fuxi-dashboard.yml` workflow
builds and deploys the static site on dashboard changes pushed to `main`.
During the build it stamps the exact `GITHUB_SHA` into
`public/data/manifest.json`; this avoids the impossible self-referential task
of committing a file containing its own commit hash.

See [PLAN.md](PLAN.md) for the complete prototype contract and next steps.

Global geography is derived from the public-domain Natural Earth 1:110m
administrative boundary dataset.
