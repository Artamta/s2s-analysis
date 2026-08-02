#!/usr/bin/env python3
"""Add validated TP, T2M, and Z500 anomalies to a global web package."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_global_forecast import (  # noqa: E402
    ANOMALY_STYLES,
    VARIABLES,
    load_climatologies,
    quantize_with,
    range_record,
    write_binary,
)
from science.formulas import anomaly as calculate_anomaly  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--climatology-dir", type=Path, required=True)
    parser.add_argument("--initialization", default="2026-07-28")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["schema_version"] not in {2, 3}:
        raise ValueError("unsupported source global schema")
    initialization = dt.date.fromisoformat(args.initialization)
    climatologies, provenance = load_climatologies(
        args.climatology_dir,
        initialization,
    )
    variable_by_key = {variable.key: variable for variable in VARIABLES}

    for key, climatology in climatologies.items():
        definition = metadata["variables"][key]
        absolute_path = args.data_dir / definition["path"]
        encoded = np.fromfile(absolute_path, dtype="<u2")
        if encoded.size != 42 * 121 * 240:
            raise ValueError(f"{absolute_path} has an invalid element count")
        absolute = (
            encoded.reshape(42, 121, 240).astype(np.float64)
            * float(definition["scale"])
            + float(definition["offset"])
        )
        anomaly = calculate_anomaly(absolute, climatology)
        style = ANOMALY_STYLES[key]
        frames = [
            quantize_with(
                frame,
                offset=style.offset,
                scale=style.scale,
                label=f"{key} anomaly",
            )
            for frame in anomaly
        ]
        frame_ranges = [range_record(frame) for frame in anomaly]
        record = write_binary(
            args.data_dir / f"{key}-anomaly.bin",
            frames,
            offset=style.offset,
            scale=style.scale,
            frame_ranges=frame_ranges,
        )
        variable = variable_by_key[key]
        definition["anomaly"] = {
            "label": style.label,
            "short_label": style.short_label,
            "units": variable.units,
            "description": style.description,
            "baseline": {
                "name": (
                    "Native model reforecast climatology · 2002–2021 · "
                    f"initialization {initialization.strftime('%d %b')}"
                ),
                **provenance[key],
            },
            "legend": {
                "boundaries": style.legend_boundaries,
                "colors": style.legend_colors,
                "under": style.under,
                "over": style.over,
            },
            **record,
        }
        print(
            f"{key} anomaly: {float(anomaly.min()):.3f} to "
            f"{float(anomaly.max()):.3f} {variable.units}",
            flush=True,
        )

    metadata["schema_version"] = 3
    metadata["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    checks = metadata["validation"]["checks"]
    for check in (
        "global TP, T2M, and Z500 anomalies use an exact 28 July slot",
        "20 complete hindcast years with 51 native members per year",
        "lead-matched anomaly subtraction on the native global grid",
    ):
        if check not in checks:
            checks.append(check)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
