#!/usr/bin/env python3
"""Download packages required by an FCN3 config and write a hashed inventory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fcn3_common import load_config, product_paths, sha256_file, write_json_atomic


FCN3_FILES = [
    "README.md",
    "config.json",
    "global_means.npy",
    "global_stds.npy",
    "land_mask.nc",
    "maxs.npy",
    "metadata.json",
    "mins.npy",
    "orography.nc",
    "training_checkpoints/best_ckpt_mp0.tar",
]
AFNOV2_FILES = [
    "afno_precip.mdlus",
    "global_means.npy",
    "global_stds.npy",
    "land_sea_mask.nc",
    "orography.nc",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)

    from earth2studio.models.px import FCN3

    packages = [("fcn3", FCN3.load_default_package(), FCN3_FILES)]
    if "precipitation_package_uri" in config["model"]:
        from earth2studio.models.dx import PrecipitationAFNOv2

        packages.append(
            ("afnov2", PrecipitationAFNOv2.load_default_package(), AFNOV2_FILES)
        )
    inventory = []
    total = 0
    for package_name, package, files in packages:
        for relative in files:
            local = Path(package.resolve(relative))
            size = local.stat().st_size
            digest = sha256_file(local)
            total += size
            inventory.append(
                {
                    "package": package_name,
                    "relative_path": relative,
                    "local_path": str(local),
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
            print(f"{package_name}/{relative}: {size:,} bytes sha256={digest}", flush=True)

    record = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_package_uri": config["model"]["package_uri"],
        "precipitation_package_uri": config["model"].get("precipitation_package_uri"),
        "file_count": len(inventory),
        "total_size_bytes": total,
        "files": inventory,
    }
    path = product_paths(config, "2020-01-02")["inventory"]
    write_json_atomic(record, path)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
