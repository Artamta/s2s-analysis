#!/usr/bin/env bash
set -euo pipefail
SOURCE_REPO="${S2S_SOURCE_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
ARCHIVE="/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1"
STUDY_SOURCE="${SOURCE_REPO}/studies/india_s2s_benchmark_v1"
INVENTORY_SOURCE="${SOURCE_REPO}/deliverables/s2s_data_inventory_20260803/inventory.json"
HASH="$({ find "${STUDY_SOURCE}" -type f ! -path '*/__pycache__/*' -print0; printf '%s\0' "${INVENTORY_SOURCE}"; } \
  | sort -z | xargs -0 sha256sum | sha256sum | awk '{print substr($1,1,16)}')"
RUNTIME="${ARCHIVE}/runtime/${HASH}"
if [[ ! -f "${RUNTIME}/runtime_manifest.json" ]]; then
  mkdir -p "${RUNTIME}/studies/india_s2s_benchmark_v1" \
    "${RUNTIME}/deliverables/s2s_data_inventory_20260803"
  cp -a "${STUDY_SOURCE}/." "${RUNTIME}/studies/india_s2s_benchmark_v1/"
  cp -a "${INVENTORY_SOURCE}" \
    "${RUNTIME}/deliverables/s2s_data_inventory_20260803/inventory.json"
  python - "${RUNTIME}" "${HASH}" "${SOURCE_REPO}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path
runtime, digest, source = sys.argv[1:]
payload = {
    "schema_version": 1,
    "runtime_sha256_prefix": digest,
    "source_repository": source,
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "contents": ["study code/config/registries", "frozen source inventory"],
}
Path(runtime, "runtime_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
fi
echo "${RUNTIME}"
