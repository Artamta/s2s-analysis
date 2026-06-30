#!/usr/bin/env python
"""Exact hand-value checks for deterministic and probabilistic metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from s2s_paper.metrics import (
    brier_score,
    brier_skill_score,
    crps_ensemble,
    crps_gaussian,
    crps_skill_score,
    probability_exceedance,
    reliability_bins,
    score_deterministic,
    spread_skill_ratio_ensemble,
)


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    print(f"{name}: {actual:.12f} expected {expected:.12f}")
    if not np.isfinite(actual) or abs(actual - expected) > tol:
        raise SystemExit(f"{name} check failed")


def main() -> int:
    lat = xr.DataArray([10.0, 11.0], dims="lat", coords={"lat": [10.0, 11.0]})
    lon = xr.DataArray([70.0, 71.0], dims="lon", coords={"lon": [70.0, 71.0]})
    truth = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    forecast = truth + 1.0
    climatology = xr.zeros_like(truth)
    det = score_deterministic(forecast, truth, climatology=climatology)
    _assert_close("ACC", det["acc"], 1.0)
    _assert_close("RMSE", det["rmse"], 1.0)
    _assert_close("Bias", det["bias"], 1.0)
    _assert_close("MAE", det["mae"], 1.0)

    one_cell_truth = xr.DataArray([[1.0]], dims=("lat", "lon"), coords={"lat": [10.0], "lon": [70.0]})
    ens_crps = xr.DataArray(
        [[[0.0]], [[2.0]]],
        dims=("member", "lat", "lon"),
        coords={"member": [0, 1], "lat": [10.0], "lon": [70.0]},
    )
    _assert_close("ensemble CRPS", crps_ensemble(ens_crps, one_cell_truth), 0.5)

    gaussian = crps_gaussian(one_cell_truth, xr.ones_like(one_cell_truth), one_cell_truth)
    expected_gaussian = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    _assert_close("Gaussian CRPS", gaussian, expected_gaussian)
    _assert_close("CRPSS", crps_skill_score(0.25, 1.0), 0.75)

    ens_ssr = xr.DataArray(
        [[[0.0]], [[1.0]], [[2.0]]],
        dims=("member", "lat", "lon"),
        coords={"member": [0, 1, 2], "lat": [10.0], "lon": [70.0]},
    )
    zero_truth = xr.zeros_like(one_cell_truth)
    _assert_close("SSR sample-spread", spread_skill_ratio_ensemble(ens_ssr, zero_truth), 1.0)

    probability = xr.DataArray([[0.75]], dims=("lat", "lon"), coords={"lat": [10.0], "lon": [70.0]})
    event = xr.DataArray([[1.0]], dims=("lat", "lon"), coords={"lat": [10.0], "lon": [70.0]})
    _assert_close("Brier", brier_score(probability, event), 0.0625)
    _assert_close("Brier skill score", brier_skill_score(0.25, 1.0), 0.75)

    masked = xr.DataArray(
        [
            [[2.0, np.nan]],
            [[np.nan, np.nan]],
            [[4.0, np.nan]],
        ],
        dims=("member", "lat", "lon"),
        coords={"member": [0, 1, 2], "lat": [10.0], "lon": [70.0, 71.0]},
    )
    prob = probability_exceedance(masked, threshold=3.0)
    _assert_close("valid-member probability", float(prob.isel(lat=0, lon=0)), 0.5)
    if not np.isnan(float(prob.isel(lat=0, lon=1))):
        raise SystemExit("masked probability check failed")

    rel_event = xr.where(np.isfinite(prob), 1.0, np.nan)
    rel_count = sum(row["count"] for row in reliability_bins(prob, rel_event, nbins=5))
    print(f"reliability finite count: {rel_count} expected 1")
    if rel_count != 1:
        raise SystemExit("reliability finite-count check failed")

    print("all exact metric formula checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
