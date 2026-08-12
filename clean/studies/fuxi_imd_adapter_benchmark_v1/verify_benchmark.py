#!/usr/bin/env python3
"""Verify the common-date FuXi adapter benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


EXPECTED_DATES = 517
EXPECTED_LEADS = 6
EXPECTED_MODELS = 9
EXPECTED_REGIONS = 5


def assert_close_csv(actual: pd.DataFrame, saved: pd.DataFrame, keys: list[str]) -> None:
    metrics = ["acc", "mae_mm_day", "rmse_mm_day", "bias_mm_day"]
    actual = actual.sort_values(keys).reset_index(drop=True)
    saved = saved.sort_values(keys).reset_index(drop=True)
    if not actual[keys].equals(saved[keys]):
        raise AssertionError(f"summary keys differ: {keys}")
    if not np.allclose(actual[metrics], saved[metrics], rtol=1e-11, atol=1e-12):
        raise AssertionError(f"summary metrics differ: {keys}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = args.result.resolve()

    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise AssertionError("manifest is not complete")
    if manifest["paired_initialization_total"] != EXPECTED_DATES:
        raise AssertionError("paired-date count changed")
    if manifest["adapter_support_cells"] != 171:
        raise AssertionError("adapter support changed")
    if len(manifest["models"]) != EXPECTED_MODELS:
        raise AssertionError("model count changed")

    cases = pd.read_csv(result / "case_metrics.csv", parse_dates=["init"])
    expected_rows = EXPECTED_DATES * EXPECTED_LEADS * EXPECTED_MODELS * EXPECTED_REGIONS
    if len(cases) != expected_rows:
        raise AssertionError(f"case table has {len(cases)} rows, expected {expected_rows}")
    key = ["model", "region", "lead_week", "init"]
    if cases.duplicated(key).any():
        raise AssertionError("duplicate model-region-lead-date rows")
    if cases[key].isna().any().any():
        raise AssertionError("missing case keys")
    if not np.isfinite(cases[["acc", "mae_mm_day", "rmse_mm_day", "bias_mm_day"]]).all().all():
        raise AssertionError("non-finite case metric")

    metrics = ["acc", "mae_mm_day", "rmse_mm_day", "bias_mm_day"]
    weekly = cases.groupby(
        ["region", "region_label", "model", "model_label", "lead_week"],
        as_index=False,
    )[metrics].mean()
    saved_weekly = pd.read_csv(result / "weekly_summary.csv")
    assert_close_csv(
        weekly,
        saved_weekly,
        ["region", "region_label", "model", "model_label", "lead_week"],
    )

    seasonal = cases.groupby(
        ["season", "region", "region_label", "model", "model_label", "lead_week"],
        as_index=False,
    )[metrics].mean()
    saved_seasonal = pd.read_csv(result / "seasonal_regional_weekly_summary.csv")
    assert_close_csv(
        seasonal,
        saved_seasonal,
        ["season", "region", "region_label", "model", "model_label", "lead_week"],
    )

    with xr.open_zarr(result / "predictions.zarr", consolidated=True) as dataset:
        expected_sizes = {
            "method": 3,
            "init": EXPECTED_DATES,
            "lead_week": EXPECTED_LEADS,
            "latitude": 27,
            "longitude": 27,
        }
        if dict(dataset.sizes) != expected_sizes:
            raise AssertionError(f"prediction dimensions changed: {dict(dataset.sizes)}")
        if int(dataset.adapter_support.sum()) != 171:
            raise AssertionError("prediction support changed")
        support = dataset.adapter_support.values.astype(bool)
        values = dataset.prediction.values[..., support]
        if not np.isfinite(values).all() or np.nanmin(values) < 0.0:
            raise AssertionError("invalid prediction values on adapter support")

    run = Path(manifest["adapter_run"])
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    checkpoint_root = run / "models" / selection["selected_model"]
    checkpoints = sorted(checkpoint_root.glob("seed_*/checkpoints/best.pt"))
    if len(checkpoints) != 3 or any(path.stat().st_size == 0 for path in checkpoints):
        raise AssertionError("selected three-seed checkpoint ensemble is incomplete")

    print(
        "PASS: 517 dates, 9 models, 6 leads, 5 regions, "
        f"3 frozen checkpoints: {result}"
    )


if __name__ == "__main__":
    main()
