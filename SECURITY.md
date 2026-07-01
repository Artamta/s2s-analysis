# Security Policy

Do not open public issues containing credentials, provider tokens, private data
paths, or restricted data.

If you discover a secret in the repository, notify the maintainer privately and
rotate the exposed credential with the provider. Removing a secret from the
latest commit is not enough if it has appeared in git history.

For CDS access, use `~/.cdsapirc` or environment variables such as
`CDSAPI_URL` and `CDSAPI_KEY`; do not hardcode keys in scripts.
