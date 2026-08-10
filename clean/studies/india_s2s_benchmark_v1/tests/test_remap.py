from __future__ import annotations

import numpy as np

from s2s_benchmark.remap import conservative_remap, intensive_remap


def test_conservative_remap_preserves_constant_field() -> None:
    source_lat = np.arange(-2.0, 4.1, 1.0)
    source_lon = np.arange(58.0, 65.1, 1.0)
    target_lat = np.array([3.0, 1.5, 0.0])
    target_lon = np.array([60.0, 61.5, 63.0])
    values = np.full((2, 7, len(source_lat), len(source_lon)), 4.25, dtype=np.float32)
    remapped, audit = conservative_remap(values, source_lat, source_lon, target_lat, target_lon)
    np.testing.assert_allclose(remapped, 4.25, rtol=0, atol=1e-6)
    assert audit["target_full_coverage"] is True


def test_intensive_remap_preserves_linear_field() -> None:
    source_lat = np.arange(-2.0, 4.1, 1.0)
    source_lon = np.arange(58.0, 65.1, 1.0)
    target_lat = np.array([3.0, 1.5, 0.0])
    target_lon = np.array([60.0, 61.5, 63.0])
    yy, xx = np.meshgrid(source_lat, source_lon, indexing="ij")
    values = (2 * yy + 3 * xx)[None, None]
    remapped, _ = intensive_remap(values, source_lat, source_lon, target_lat, target_lon)
    ty, tx = np.meshgrid(target_lat, target_lon, indexing="ij")
    np.testing.assert_allclose(remapped[0, 0], 2 * ty + 3 * tx, atol=1e-5)
