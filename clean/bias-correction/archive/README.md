# Local archive

This directory holds non-canonical evidence removed from the active working
tree. Compressed payloads are intentionally Git-ignored; their contents and
SHA-256 checksums are recorded in `docs/CLEANUP_20260819.md`.

`source_snapshots/` contains small, tracked historical source files required
to verify immutable manifests after live code moves. They are provenance
records, not importable application code.

The active result tree must contain only complete scientific runs, current
screens, and explicitly documented exceptions. Never restore an archived
smoke run as publication evidence.
