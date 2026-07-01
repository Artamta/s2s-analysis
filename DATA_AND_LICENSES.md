# Data And License Boundaries

This repository uses multiple data sources with different access terms. The
project license does not override provider terms.

## Repository Code

Code authored in this repository is licensed under the MIT License. See
[`LICENSE`](LICENSE).

## Paper, Figures, Tables, And Documentation

Project-authored manuscript text, generated figures, generated tables, and
documentation are licensed under Creative Commons Attribution 4.0 International
unless a file states otherwise. See [`LICENSE-DOCS.md`](LICENSE-DOCS.md).

## Third-Party Or Restricted Data

Raw or provider-delivered data are not relicensed by this repository. Users must
obtain and use these data under the applicable provider terms:

- Spire AI-S2S forecast and hindcast products.
- FuXi-S2S model weights, code, inputs, and hindcast or forecast products.
- DLESyM forecasts supplied by model providers.
- ECMWF S2S and ERA5 products from ECMWF/Copernicus services.
- UKMO S2S products.
- NCEP CFSv2/S2S products.
- India Meteorological Department gridded rainfall products.

The repository may contain scripts, derived aggregate metrics, generated tables,
and generated figures. Those derived artifacts should not be read as granting
permission to redistribute the underlying raw datasets.

The conda environment file describes software dependencies only. It does not
grant access to third-party data or override any provider license.

## Secrets And Credentials

Do not commit API credentials or private configuration files. For CDS downloads,
use `~/.cdsapirc` or environment variables such as:

```bash
export CDSAPI_URL="https://cds.climate.copernicus.eu/api"
export CDSAPI_KEY="<your-key>"
```

If a credential has ever been committed, rotate it with the provider and treat
the old value as public.
