#!/usr/bin/env python3
"""Build the compact, pre-projected geography used by the India forecast maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public" / "data"
OUTPUT = DATA / "india-map-geography.json"

MAP_SIZE = 620.0
LON_MIN = 59.25
LON_MAX = 99.75
LAT_MIN = -0.75
LAT_MAX = 39.75


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_x(longitude: float) -> float:
    return ((longitude - LON_MIN) / (LON_MAX - LON_MIN)) * MAP_SIZE


def map_y(latitude: float) -> float:
    return ((LAT_MAX - latitude) / (LAT_MAX - LAT_MIN)) * MAP_SIZE


def number(value: float) -> str:
    """Format display coordinates to sub-pixel precision without padding."""

    return f"{value:.1f}".rstrip("0").rstrip(".")


def ring_path(ring: list[list[float]]) -> str:
    """Project a GeoJSON ring and remove points identical at display precision."""

    points: list[tuple[str, str]] = []
    for longitude, latitude, *_ in ring:
        point = (number(map_x(longitude)), number(map_y(latitude)))
        if not points or point != points[-1]:
            points.append(point)
    if len(points) < 3:
        return ""
    return "M" + "L".join(f"{x},{y}" for x, y in points) + "Z"


def bounds_intersect(rings: Iterable[list[list[float]]]) -> bool:
    """Return whether a polygon intersects the India-map display bounds."""

    points = [point for ring in rings for point in ring]
    if not points:
        return False
    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]
    return not (
        max(longitudes) < LON_MIN
        or min(longitudes) > LON_MAX
        or max(latitudes) < LAT_MIN
        or min(latitudes) > LAT_MAX
    )


def polygon_path(rings: list[list[list[float]]], regional_only: bool) -> str:
    if regional_only and not bounds_intersect(rings):
        return ""
    return "".join(filter(None, (ring_path(ring) for ring in rings)))


def geometry_path(geometry: dict[str, Any], regional_only: bool = False) -> str:
    """Convert Polygon or MultiPolygon GeoJSON geometry to one SVG path."""

    coordinates = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        return polygon_path(coordinates, regional_only)
    if geometry["type"] == "MultiPolygon":
        return "".join(
            polygon_path(polygon, regional_only) for polygon in coordinates
        )
    raise ValueError(f"unsupported geometry type: {geometry['type']}")


def feature_collection_path(payload: dict[str, Any], regional_only: bool) -> str:
    """Combine all supported feature geometry into one display path."""

    return "".join(
        geometry_path(feature["geometry"], regional_only)
        for feature in payload["features"]
        if feature.get("geometry")
    )


def main() -> int:
    world_path = DATA / "world-countries.geojson"
    outline_path = DATA / "india-outline.json"
    admin_path = DATA / "india-admin.json"
    world = json.loads(world_path.read_text(encoding="utf-8"))
    outline = json.loads(outline_path.read_text(encoding="utf-8"))
    admin = json.loads(admin_path.read_text(encoding="utf-8"))

    payload = {
        "schema_version": 1,
        "description": (
            "Pre-projected display paths for the India forecast panels; "
            "scientific field values remain on the native forecast grid."
        ),
        "view_box": [0, 0, int(MAP_SIZE), int(MAP_SIZE)],
        "world_path": feature_collection_path(world, regional_only=True),
        "india_outline_path": geometry_path(outline["geometry"]),
        "india_admin_path": feature_collection_path(admin, regional_only=False),
        "sources": {
            "world_countries_sha256": sha256(world_path),
            "india_outline_sha256": sha256(outline_path),
            "india_admin_sha256": sha256(admin_path),
        },
    }
    if not all(payload[key] for key in ("world_path", "india_outline_path", "india_admin_path")):
        raise ValueError("one or more India map paths are empty")
    OUTPUT.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
