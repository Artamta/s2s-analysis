"""Region-mask helpers for IMD homogeneous rainfall regions."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from .paths import get_paths, mask_path


REGION_KEYS = (
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)

REGION_LABELS = {
    "All India": "All India",
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast India",
}


def open_region_masks(dgrid: float = 1.5, path: str | Path | None = None) -> xr.Dataset:
    """Open copied IMD masks and add an All-India union mask."""

    p = Path(path) if path is not None else mask_path(dgrid, get_paths())
    ds = xr.open_dataset(p)
    missing = [key for key in REGION_KEYS if key not in ds]
    if missing:
        raise ValueError(f"mask file {p} is missing variables: {missing}")

    all_india = None
    for key in REGION_KEYS:
        mask = ds[key].astype(bool)
        all_india = mask if all_india is None else (all_india | mask)
    ds = ds.assign({"All India": all_india.astype("int8")})
    return ds


def mask_summary(dgrid: float = 1.5) -> list[dict[str, object]]:
    """Return grid shape and cell counts for all masks."""

    ds = open_region_masks(dgrid)
    rows = []
    for key in ("All India",) + REGION_KEYS:
        rows.append(
            {
                "region": key,
                "label": REGION_LABELS[key],
                "lat": int(ds.sizes["lat"]),
                "lon": int(ds.sizes["lon"]),
                "cells": int(ds[key].sum().item()),
            }
        )
    ds.close()
    return rows

