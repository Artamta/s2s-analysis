#!/usr/bin/env python3
"""Download each pinned DLESyM package once and write a hashed inventory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dlesym_common import load_config, package_for, sha256_file, write_json_atomic


FILES = {
    "v1_t2m": [
        "README.md",
        "config.yaml",
        "hpx_lat.npy",
        "hpx_lon.npy",
        "land_sea_mask.npy",
        "topography.npy",
        *[f"atmos_model_{index}.mdlus" for index in range(4)],
        *[f"ocean_model_{index}.mdlus" for index in range(4)],
    ],
    "v0_tp_t2m": [
        "README.md",
        "THIRD_PARTY.txt",
        "config.yaml",
        "era5_ttr_doy_stats_hpx64.nc",
        "hpx_lat.npy",
        "hpx_lon.npy",
        "isccp_olr_doy_stats_hpx64.nc",
        "land_sea_mask.npy",
        "topography.npy",
        "atmos_model_0.mdlus",
        "ocean_model_0.mdlus",
        "precip_model_0.mdlus",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--product", choices=["v1_t2m", "v0_tp_t2m", "all"], default="all"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    products = list(FILES) if args.product == "all" else [args.product]
    for product in products:
        package = package_for(product)
        inventory = []
        total = 0
        print(f"prefetching {product} from {config['products'][product]['package_uri']}")
        for relative in FILES[product]:
            local = Path(package.resolve(relative))
            size = local.stat().st_size
            digest = sha256_file(local)
            total += size
            inventory.append(
                {
                    "relative_path": relative,
                    "local_path": str(local),
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
            print(f"  {relative}: {size:,} bytes sha256={digest}", flush=True)
        record = {
            "status": "passed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "product": product,
            "package_uri": config["products"][product]["package_uri"],
            "package_revision": config["products"][product]["package_revision"],
            "file_count": len(inventory),
            "total_size_bytes": total,
            "files": inventory,
        }
        root = (
            Path(config["storage"]["root"])
            / "dlesym"
            / config["products"][product]["run_label"]
            / "provenance"
        )
        write_json_atomic(record, root / "checkpoint_inventory.json")
        print(
            f"{product}: {len(inventory)} files, {total / 2**20:.1f} MiB downloaded/cached",
            flush=True,
        )


if __name__ == "__main__":
    main()
