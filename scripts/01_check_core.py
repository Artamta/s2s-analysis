#!/usr/bin/env python
"""Lightweight checks for shared grid/mask/metric code."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2s_benchmark.constants import WEEKLY_WINDOWS
from s2s_benchmark.grid import GridSpec, apply_region, cosine_weights, make_grid
from s2s_benchmark.metrics import (
    brier_score,
    probability_exceedance,
    reliability_bins,
    score_deterministic,
    score_probabilistic_ensemble,
)


def main() -> int:
    spec = GridSpec(dgrid=1.5)
    lat, lon = make_grid(spec)
    base = xr.DataArray(
        np.add.outer(np.arange(len(lat), dtype=float), np.arange(len(lon), dtype=float)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="toy",
    )
    truth = apply_region(base, "All India", dgrid=1.5)
    forecast = truth + 1.0
    climatology = xr.zeros_like(truth)
    ensemble = xr.concat([truth, truth + 1.0, truth + 2.0], dim="member").assign_coords(member=[0, 1, 2])
    weights = cosine_weights(truth["lat"])
    scores = score_deterministic(forecast, truth, climatology=climatology, weights=weights)
    prob_scores = score_probabilistic_ensemble(ensemble, truth, weights)
    prob = probability_exceedance(ensemble, threshold=float(truth.min()) - 1.0)
    event = (truth > float(truth.min()) - 1.0).where(np.isfinite(truth))
    bs = brier_score(prob, event, weights)
    rel = reliability_bins(prob, event, nbins=5)
    reliability_count = sum(row["count"] for row in rel)
    finite_region_cells = int(np.isfinite(truth).sum())

    print("weekly windows:", ", ".join(f"{w.name}={w.start}-{w.end}" for w in WEEKLY_WINDOWS))
    print(f"grid: {len(lat)}x{len(lon)}")
    print("toy scores:")
    for key, value in scores.items():
        print(f"  {key}: {value:.6f}")
    print("toy probabilistic scores:")
    for key, value in prob_scores.items():
        print(f"  {key}: {value:.6f}")
    print(f"  brier: {bs:.6f}")
    print(f"  reliability_count: {reliability_count}")

    if abs(scores["bias"] - 1.0) > 1e-10:
        raise SystemExit("bias check failed")
    if abs(scores["rmse"] - 1.0) > 1e-10:
        raise SystemExit("rmse check failed")
    if abs(scores["acc"] - 1.0) > 1e-10:
        raise SystemExit("acc check failed")
    if abs(prob_scores["crps"] - 5.0 / 9.0) > 1e-10:
        raise SystemExit("crps check failed")
    if abs(prob_scores["spread_skill_ratio"] - 1.0) > 1e-10:
        raise SystemExit("ssr check failed")
    if abs(bs) > 1e-10:
        raise SystemExit("brier check failed")
    if reliability_count != finite_region_cells:
        raise SystemExit("reliability mask check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
