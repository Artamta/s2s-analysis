# FuXi-S2S India Forecast Lab

A static, validated prototype for the 28 July 2026 experimental FuXi-S2S
forecast. The site publishes compact ensemble-derived fields only. It does not
contain raw model inputs, member fields, observations, reforecasts, model
weights, credentials, or cluster paths.

## Scientific status

The forecast is a 100-member FuXi-S2S run initialized with operational GFS
daily proxy fields. FuXi-S2S was trained with ERA5-style daily inputs, and no
matched GFS-initialized hindcast calibration exists. It is research guidance,
not an official forecast or warning.

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

The main forecast JSON stores latitude and longitude vectors, an India support
mask, six week records, and flattened row-major fields. Every field contains
exactly `latitude.length × longitude.length` finite values. The support mask
controls display; no `NaN` or infinity is serialized.

The site has no backend. Hash navigation keeps the GitHub Pages deployment
portable, and a failed validation status blocks forecast rendering while
leaving the validation and methods pages available.

## Deployment

`.github/workflows/deploy-pages.yml` builds and deploys the static site on
push to `main`. During the build it stamps the exact `GITHUB_SHA` into
`public/data/manifest.json`; this avoids the impossible self-referential task
of committing a file containing its own commit hash.

See [PLAN.md](PLAN.md) for the complete prototype contract and next steps.
