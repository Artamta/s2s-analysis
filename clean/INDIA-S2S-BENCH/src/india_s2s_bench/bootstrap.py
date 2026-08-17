"""Paired moving-block uncertainty over initialization dates."""

from __future__ import annotations

import numpy as np


def circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n < 1 or block_length < 1:
        raise ValueError("n and block_length must be positive")
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]


def paired_interval(
    values_a: np.ndarray,
    values_b: np.ndarray,
    initialization: np.ndarray,
    *,
    draws: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    initialization = np.asarray(initialization)
    if not (values_a.shape == values_b.shape == initialization.shape):
        raise ValueError("paired arrays must have identical shapes")
    dates = np.unique(initialization)
    # Keep every lead/region row belonging to a sampled initialization together.
    date_effect = np.array([
        np.nanmean((values_a - values_b)[initialization == date]) for date in dates
    ])
    if not np.all(np.isfinite(date_effect)):
        raise ValueError("non-finite paired date effect")
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled[draw] = np.mean(date_effect[circular_block_indices(len(dates), block_length, rng)])
    return {
        "effect": float(np.mean(date_effect)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "draws": int(draws),
        "block_length": int(block_length),
        "initializations": int(len(dates)),
    }
