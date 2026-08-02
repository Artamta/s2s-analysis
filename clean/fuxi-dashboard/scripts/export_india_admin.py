#!/usr/bin/env python3
"""Export a compact display derivative of the Survey of India ABDB state layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "public/data/india-admin.json",
    )
    parser.add_argument("--simplify-degrees", type=float, default=0.025)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, tuple):
        return [rounded(item) for item in value]
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    required = [
        args.source,
        args.source.with_suffix(".shx"),
        args.source.with_suffix(".dbf"),
        args.source.with_suffix(".prj"),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete India ABDB boundary: {missing}")

    reader = shapefile.Reader(str(args.source), encoding="utf-8")
    field_names = [field[0] for field in reader.fields[1:]]
    if "STATE" not in field_names:
        raise ValueError("India ABDB state layer has no STATE field")
    source_crs = CRS.from_wkt(
        args.source.with_suffix(".prj").read_text(encoding="utf-8")
    )
    target_crs = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    features = []
    for item in reader.iterShapeRecords():
        properties = dict(zip(field_names, item.record))
        name = str(properties["STATE"]).strip()
        geometry = transform(transformer.transform, shape(item.shape.__geo_interface__))
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        geometry = geometry.simplify(
            args.simplify_degrees,
            preserve_topology=True,
        )
        if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"{name} produced invalid display geometry")
        label_point = geometry.representative_point()
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "label": not name.startswith("DISPUTED"),
                    "label_longitude": round(label_point.x, 5),
                    "label_latitude": round(label_point.y, 5),
                },
                "geometry": rounded(mapping(geometry)),
            }
        )

    if len(features) != 40:
        raise ValueError(f"expected 40 ABDB state-layer features; found {len(features)}")
    names = {feature["properties"]["name"] for feature in features}
    required_names = {
        "JAMMU AND KASHMIR",
        "LADAKH",
        "ARUNACHAL PRADESH",
        "ANDAMAN & NICOBAR",
    }
    if not required_names.issubset(names):
        raise ValueError(f"ABDB state coverage is incomplete: {required_names - names}")

    payload = {
        "type": "FeatureCollection",
        "name": "India state and union-territory display boundaries",
        "description": (
            "Simplified display derivative of the Survey of India Administrative "
            "Boundary Database state/UT layer, retaining the complete supplied depiction."
        ),
        "source": {
            "name": "Survey of India Administrative Boundary Database (ABDB)",
            "product": "Entire-country state/UT boundary layer",
            "source_sha256": sha256(args.source),
            "display_note": (
                "Presentation reference only; not a substitute for an official "
                "legal or cadastral boundary determination."
            ),
        },
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} with {len(features)} features")


if __name__ == "__main__":
    main()
