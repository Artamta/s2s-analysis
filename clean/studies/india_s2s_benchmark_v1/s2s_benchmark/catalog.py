from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_catalog(manifest_paths: list[Path], output_dir: Path, scope: str) -> tuple[Path, Path]:
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(manifest_paths)]
    bad = [item["store"] for item in manifests if item.get("status") != "complete"]
    if bad:
        raise ValueError(f"non-complete manifests cannot enter catalog: {bad}")
    records = []
    init_rows = []
    for item in manifests:
        record = {
            key: item[key]
            for key in (
                "model", "experiment_id", "variable", "grid", "year", "store",
                "initialization_count", "distribution_representation", "units",
                "temporal_statistic", "zmetadata_sha256",
            )
        }
        record["manifest"] = item["manifest_path"]
        record["qc_status"] = "passed" if all(qc["status"] == "passed" for qc in item["qc"]) else "failed"
        records.append(record)
        for initialization in item["initializations"]:
            init_rows.append({
                "model": item["model"],
                "experiment_id": item["experiment_id"],
                "variable": item["variable"],
                "grid": item["grid"],
                "initialization": initialization,
                "store": item["store"],
                "qc_status": record["qc_status"],
            })
    content: dict[str, Any] = {
        "schema_version": 1,
        "archive_id": "india_s2s_benchmark_v1",
        "scope": scope,
        "common_grid_id": "india_1p5_27x27_v1",
        "records": records,
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    catalog_id = hashlib.sha256(canonical).hexdigest()[:16]
    body = {
        **content,
        "catalog_id": catalog_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"catalog_{scope}_{catalog_id}.json"
    parquet_path = output_dir / f"init_index_{scope}_{catalog_id}.parquet"
    if json_path.exists() or parquet_path.exists():
        if not (json_path.exists() and parquet_path.exists()):
            raise FileExistsError(f"incomplete catalog pair for {catalog_id}")
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        existing_content = {
            key: existing[key]
            for key in ("schema_version", "archive_id", "scope", "common_grid_id", "records")
        }
        if existing_content != content or existing.get("catalog_id") != catalog_id:
            raise FileExistsError(f"catalog ID collision or immutable catalog mismatch: {json_path}")
        return json_path, parquet_path
    json_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(init_rows).to_parquet(parquet_path, index=False)
    return json_path, parquet_path
