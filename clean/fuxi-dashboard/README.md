# Atmosphere 42

A static interactive global outlook and validated India case for the 28 July
2026 experimental forecast. The global viewer animates 42 daily ensemble-mean
fields for precipitation, 2 m temperature, and 500 hPa geopotential height.
The site publishes compact ensemble-derived fields only. It does not contain
raw model inputs, member fields, observations, reforecasts, model weights,
credentials, or cluster paths.

## Scientific status

The forecast is a 100-member FuXi-S2S run initialized with operational GFS
daily proxy fields. FuXi-S2S was trained with ERA5-style daily inputs, and no
matched GFS-initialized hindcast calibration exists. It is research guidance,
not an official forecast or warning.

The global animation is an independently sampled 100-member companion ensemble
generated from the same frozen input as the earlier India case. The raw members
behind the India export had already been removed, so its exact stochastic draws
could not be reused.

The forecast interface uses the neutral product name **Atmosphere 42**. Model
identity remains in this repository and the Methods page for scientific
provenance.

Anomalies use a calendar-interpolated, lead-matched FuXi 2002–2021 native
reforecast climatology. They are not IMD or IMERG anomalies. The standardized
FuXi climatology and audited IMERG Final climatology are currently
season-limited; both cover this issue, while full-year automation remains a
pipeline milestone.

## Rebuild and validate

From this directory:

```bash
python scripts/validate_sources.py
python scripts/build_prototype_data.py
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

The global contract stores six little-endian `uint16` binary arrays in
lead-day, latitude, longitude order: three ensemble means and three population
standard deviations across the 100 members. `metadata.json` locks dimensions,
quantization, units, legends, valid periods, file sizes, and SHA-256 hashes.
The browser verifies each binary before presentation. The complete global
payload is about 14.6 MB before HTTP compression.

The India forecast JSON stores latitude and longitude vectors, a support mask,
six week records, and flattened row-major fields. Every displayed field
contains exactly `latitude.length × longitude.length` finite values.

The site has no backend. Hash navigation keeps the GitHub Pages deployment
portable, and a failed validation status blocks forecast rendering while
leaving the validation and methods pages available.

## Deployment

`.github/workflows/deploy-pages.yml` builds and deploys the static site on
push to `main`. During the build it stamps the exact `GITHUB_SHA` into
`public/data/manifest.json`; this avoids the impossible self-referential task
of committing a file containing its own commit hash.

See [PLAN.md](PLAN.md) for the complete prototype contract and next steps.

Global geography is derived from the public-domain Natural Earth 1:110m
administrative boundary dataset.
