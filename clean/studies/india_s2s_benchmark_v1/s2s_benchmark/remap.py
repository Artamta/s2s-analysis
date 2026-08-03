from __future__ import annotations

import numpy as np


def centers_to_edges(centers: np.ndarray, *, clip_latitude: bool = False) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("cell centers must be a one-dimensional coordinate")
    if not (np.all(np.diff(centers) > 0) or np.all(np.diff(centers) < 0)):
        raise ValueError("cell centers must be strictly monotonic")
    if centers[0] > centers[-1]:
        centers = centers[::-1]
    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.clip(edges, -90.0, 90.0) if clip_latitude else edges


def _operators(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sy = centers_to_edges(source_lat, clip_latitude=True)
    sx = centers_to_edges(source_lon)
    ty = centers_to_edges(target_lat, clip_latitude=True)
    tx = centers_to_edges(target_lon)
    lat_op = np.zeros((len(target_lat), len(source_lat)), dtype=np.float64)
    lon_op = np.zeros((len(target_lon), len(source_lon)), dtype=np.float64)
    for j in range(len(target_lat)):
        lo = np.maximum(sy[:-1], ty[j])
        hi = np.minimum(sy[1:], ty[j + 1])
        keep = hi > lo
        lat_op[j, keep] = np.sin(np.deg2rad(hi[keep])) - np.sin(np.deg2rad(lo[keep]))
    for i in range(len(target_lon)):
        lo = np.maximum(sx[:-1], tx[i])
        hi = np.minimum(sx[1:], tx[i + 1])
        keep = hi > lo
        lon_op[i, keep] = np.deg2rad(hi[keep] - lo[keep])
    area = (
        (np.sin(np.deg2rad(ty[1:])) - np.sin(np.deg2rad(ty[:-1])))[:, None]
        * np.deg2rad(tx[1:] - tx[:-1])[None, :]
    )
    coverage = np.outer(lat_op.sum(axis=1), lon_op.sum(axis=1))
    if not np.allclose(coverage, area, rtol=2e-6, atol=1e-12):
        raise ValueError("source grid does not fully cover the target grid")
    return lat_op, lon_op, area


def conservative_remap(
    values: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat_desc: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Spherical cell-overlap area average with finite-value normalization."""
    lat_order = np.argsort(source_lat)
    lon_order = np.argsort(source_lon)
    target_order = np.argsort(target_lat_desc)
    source_lat_asc = np.asarray(source_lat)[lat_order]
    source_lon_asc = np.asarray(source_lon)[lon_order]
    target_lat_asc = np.asarray(target_lat_desc)[target_order]
    data = np.asarray(values)[..., lat_order, :][..., :, lon_order]
    lat_op, lon_op, area = _operators(
        source_lat_asc, source_lon_asc, target_lat_asc, np.asarray(target_lon)
    )
    flat = data.reshape((-1,) + data.shape[-2:])
    output = np.empty((len(flat), len(target_lat_asc), len(target_lon)), dtype=np.float64)
    for index, field in enumerate(flat):
        finite = np.isfinite(field)
        numerator = lat_op @ np.nan_to_num(field, nan=0.0) @ lon_op.T
        denominator = lat_op @ finite.astype(np.float64) @ lon_op.T
        output[index] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
    reverse = np.argsort(target_order)
    output = output[:, reverse].reshape(data.shape[:-2] + (len(target_lat_desc), len(target_lon)))
    source_area_mean = float(np.nanmean(data))
    target_area_desc = area[reverse]
    target_area_mean = float(
        np.nansum(output * target_area_desc) / np.nansum(np.isfinite(output) * target_area_desc)
    )
    return output.astype(np.float32), {
        "source_unweighted_mean": source_area_mean,
        "target_area_weighted_mean": target_area_mean,
        "target_full_coverage": True,
    }

def intensive_remap(
    values: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat_desc: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Bilinear interpolation for intensive instantaneous/mean fields."""
    from scipy.interpolate import RegularGridInterpolator

    lat_order = np.argsort(source_lat)
    lon_order = np.argsort(source_lon)
    source_lat_asc = np.asarray(source_lat)[lat_order]
    source_lon_asc = np.asarray(source_lon)[lon_order]
    data = np.asarray(values)[..., lat_order, :][..., :, lon_order]
    yy, xx = np.meshgrid(target_lat_desc, target_lon, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    flat = data.reshape((-1,) + data.shape[-2:])
    out = np.empty((len(flat), len(target_lat_desc), len(target_lon)), dtype=np.float32)
    for index, field in enumerate(flat):
        interpolator = RegularGridInterpolator(
            (source_lat_asc, source_lon_asc), field, method="linear", bounds_error=True
        )
        out[index] = interpolator(points).reshape(len(target_lat_desc), len(target_lon))
    return out.reshape(data.shape[:-2] + out.shape[-2:]), {
        "source_unweighted_mean": float(np.nanmean(data)),
        "target_unweighted_mean": float(np.nanmean(out)),
        "target_full_coverage": True,
    }
