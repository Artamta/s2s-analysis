#!/usr/bin/env python3
"""Build a read-only, manifest-first inventory of local S2S datasets.

The storage datasets are never modified.  The only writes are the JSON, CSV,
and Markdown reports beneath ``--output-dir``.  Existing download and model-run
manifests are authoritative; bounded filesystem inspection is used only to
confirm that declared files still exist and to catalog datasets without case
manifests (ERPAS and the native FuXi reforecast archives).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE = Path("/storage/raj.ayush/s2s_final_data/final_iteration")
DEFAULT_FUXI_ARCHIVE = Path("/storage/raj.ayush/archive/All_Model_Data/models/fuxi/data")
DEFAULT_CALENDAR = WORKSPACE / "config/all_season_dates_2020_2025.csv"
DEFAULT_OUTPUT = WORKSPACE / "deliverables" / f"s2s_data_inventory_{dt.date.today():%Y%m%d}"

DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
GOOD_MANIFEST_STATUSES = {
    "downloaded_valid",
    "existing_valid",
    "generated_valid",
    "passed",
    "complete",
    "valid",
}
FORMAT_BY_SUFFIX = {
    ".grib": "grib",
    ".grb": "grib",
    ".grib2": "grib",
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".7z": "7z",
    ".json": "json",
}
SIGNATURES = {
    "grib": (b"GRIB",),
    "netcdf": (b"CDF\x01", b"CDF\x02", b"\x89HDF\r\n\x1a\n"),
    "7z": (b"7z\xbc\xaf\x27\x1c",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--fuxi-archive", type=Path, default=DEFAULT_FUXI_ARCHIVE)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--open-qc",
        choices=("none", "sample", "all"),
        default="sample",
        help="Open no files, first/middle/last files per experiment, or every file.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Recompute declared SHA-256 hashes (potentially expensive).",
    )
    parser.add_argument(
        "--open-timeout-seconds",
        type=int,
        default=20,
        help="Maximum time for one sampled metadata-open operation.",
    )
    parser.add_argument("--skip-erpas", action="store_true")
    parser.add_argument("--skip-fuxi-archive", action="store_true")
    return parser.parse_args()


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    raw = match.group(1)
    try:
        return dt.datetime.strptime(raw, "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def infer_format(path: Path) -> str:
    return FORMAT_BY_SUFFIX.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "unknown")


def file_identity(
    path: Path,
    *,
    manifest_status: str | None = None,
    declared_size: int | None = None,
    declared_sha256: str | None = None,
    variable: str | None = None,
    forecast_type: str | None = None,
    members: int | None = None,
    lead_days: int | None = None,
    manifest_path: Path | None = None,
    verify_checksum: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "format": infer_format(path),
        "variable": variable,
        "forecast_type": forecast_type,
        "members": members,
        "member_ids": None,
        "member_id_source": None,
        "lead_days": lead_days,
        "manifest_status": manifest_status,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "declared_size_bytes": declared_size,
        "declared_sha256": declared_sha256,
        "exists": path.is_file(),
        "size_bytes": None,
        "signature_ok": None,
        "checksum_verified": None,
        "open_qc": {"status": "not_run"},
        "status": "missing",
        "issues": [],
    }
    if members is not None:
        if forecast_type in {"cf", "control_forecast"}:
            record["member_ids"] = ["control"]
        elif forecast_type in {"pf", "perturbed_forecast"}:
            record["member_ids"] = [str(index) for index in range(1, int(members) + 1)]
        elif forecast_type == "precomputed_ensemble_mean":
            record["member_ids"] = ["ensemble_mean"]
        elif forecast_type == "deterministic":
            record["member_ids"] = ["0"]
        else:
            record["member_ids"] = [str(index) for index in range(int(members))]
        record["member_id_source"] = "inferred_from_manifest_count_and_forecast_type"
    if not path.is_file():
        record["issues"].append("declared_path_missing")
        return record

    size = path.stat().st_size
    record["size_bytes"] = size
    if size == 0:
        record["status"] = "empty"
        record["issues"].append("zero_byte_file")
        return record
    if declared_size is not None and int(declared_size) != size:
        record["issues"].append("size_differs_from_manifest")

    expected = SIGNATURES.get(record["format"])
    if expected:
        with path.open("rb") as handle:
            header = handle.read(max(map(len, expected)))
        record["signature_ok"] = any(header.startswith(sig) for sig in expected)
        if not record["signature_ok"]:
            record["issues"].append("file_signature_mismatch")

    if verify_checksum and declared_sha256:
        actual = sha256(path)
        record["actual_sha256"] = actual
        record["checksum_verified"] = actual == declared_sha256
        if not record["checksum_verified"]:
            record["issues"].append("sha256_mismatch")

    if record["issues"]:
        record["status"] = "metadata_mismatch"
    elif manifest_status in GOOD_MANIFEST_STATUSES:
        record["status"] = "manifest_valid"
    else:
        record["status"] = "present_unvalidated"
    return record


def open_metadata_qc(record: dict[str, Any], timeout_seconds: int = 20) -> None:
    """Open one file without loading its full payload and record its schema."""
    path = Path(record["path"])
    fmt = record["format"]
    if not record["exists"] or fmt not in {"netcdf", "grib"}:
        record["open_qc"] = {"status": "not_supported"}
        return
    try:
        if fmt == "grib":
            child = r'''
import json
import sys
import cfgrib

datasets = cfgrib.open_datasets(sys.argv[1], backend_kwargs={"indexpath": ""})
try:
    variables = {}
    dimensions = {}
    for dataset in datasets:
        dimensions.update({str(k): int(v) for k, v in dataset.sizes.items()})
        for name, array in dataset.data_vars.items():
            variables[str(name)] = {
                "dims": list(array.dims),
                "dtype": str(array.dtype),
                "units": array.attrs.get("units"),
            }
    if not variables:
        raise ValueError("no GRIB data variables found")
    print(json.dumps({"variables": variables, "dimensions": dimensions}))
finally:
    for dataset in datasets:
        dataset.close()
'''
            try:
                result = subprocess.run(
                    [sys.executable, "-c", child, str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                record["open_qc"] = {
                    "status": "timed_out",
                    "timeout_seconds": timeout_seconds,
                    "note": "decoder slowness is not evidence of corruption",
                }
                return
            if result.returncode != 0:
                error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown cfgrib error"
                raise RuntimeError(error)
            metadata = json.loads(result.stdout.strip().splitlines()[-1])
            variables = metadata["variables"]
            dimensions = metadata["dimensions"]
        else:
            import xarray as xr

            datasets = [xr.open_dataset(path, decode_times=False)]
            try:
                variables: dict[str, Any] = {}
                dimensions: dict[str, int] = {}
                for dataset in datasets:
                    dimensions.update({str(k): int(v) for k, v in dataset.sizes.items()})
                    for name, array in dataset.data_vars.items():
                        variables[str(name)] = {
                            "dims": list(array.dims),
                            "dtype": str(array.dtype),
                            "units": array.attrs.get("units"),
                        }
            finally:
                for dataset in datasets:
                    dataset.close()
        record["open_qc"] = {
            "status": "opened",
            "variables": variables,
            "dimensions": dimensions,
        }
        if record["status"] == "present_unvalidated":
            record["status"] = "opened_valid"
    except Exception as exc:  # surfaced in machine-readable audit output
        record["open_qc"] = {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
        record["status"] = "unreadable"
        record["issues"].append("open_failed")


def empty_experiment(
    experiment_id: str,
    model: str,
    product: str,
    root: Path,
    source_kind: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "model": model,
        "product": product,
        "source_kind": source_kind,
        "root": str(root),
        "cases": {},
        "notes": [],
    }


def add_file(experiment: dict[str, Any], init_date: str, record: dict[str, Any]) -> None:
    case = experiment["cases"].setdefault(
        init_date,
        {
            "initialization_date": init_date,
            "initialization_time": f"{init_date}T00:00:00Z",
            "year": int(init_date[:4]),
            "files": [],
            "acc_ready": False,
            "exclusion_reasons": [],
        },
    )
    case["files"].append(record)


def read_calendar(path: Path) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    core: dict[int, set[str]] = defaultdict(set)
    cnrm: dict[int, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            date = row["init_date"]
            year = int(date[:4])
            core[year].add(date)
            if row.get("cnrm_available") == "1":
                cnrm[year].add(date)
    return dict(core), dict(cnrm)


def scan_physics(
    storage: Path,
    verify_checksums: bool,
) -> list[dict[str, Any]]:
    manifest_root = storage / "manifests"
    experiments = []
    for provider in ("ecmwf", "ukmo", "ncep", "cma", "cnrm"):
        exp = empty_experiment(
            f"physics/{provider}_operational_2020_2025",
            provider,
            "operational_forecast",
            storage / "raw" / provider / "forecast",
            "append_only_jsonl_manifests",
        )
        latest: dict[str, tuple[Path, dict[str, Any]]] = {}
        for manifest in sorted((manifest_root / provider / "forecast").glob("annual*.jsonl")):
            with manifest.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = item.get("request_hash") or item.get("target")
                    previous = latest.get(key)
                    if previous is None or item.get("timestamp", "") >= previous[1].get("timestamp", ""):
                        latest[key] = (manifest, item)
        for manifest, item in latest.values():
            init_date = item.get("init_date") or infer_date(item.get("target", ""))
            if not init_date:
                continue
            fields = item.get("fields", {})
            details = next(iter(fields.values()), {}) if len(fields) == 1 else {}
            record = file_identity(
                Path(item["target"]),
                manifest_status=item.get("status"),
                declared_size=item.get("size_bytes"),
                variable=item.get("field"),
                forecast_type=item.get("forecast_type"),
                members=item.get("member_count") or details.get("members"),
                lead_days=item.get("lead_days"),
                manifest_path=manifest,
                verify_checksum=verify_checksums,
            )
            record["manifest_fields"] = fields
            record["request_hash"] = item.get("request_hash")
            add_file(exp, init_date, record)
        if exp["cases"]:
            experiments.append(exp)
    return experiments


def manifest_output(item: dict[str, Any]) -> str | None:
    for key in ("output", "output_path", "forecast_path"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def manifest_variables(item: dict[str, Any]) -> list[str]:
    for key in ("variables", "fields", "retained_fields"):
        value = item.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
    if isinstance(item.get("statistics"), dict):
        return list(item["statistics"])
    if isinstance(item.get("field_stats"), dict):
        return list(item["field_stats"])
    return []


def scan_model_runs(storage: Path, verify_checksums: bool) -> list[dict[str, Any]]:
    experiments = []
    root = storage / "model-runs"
    for model in ("fuxi", "dlesym", "fcn3", "neural-gcm"):
        model_root = root / model
        if not model_root.is_dir():
            continue
        for run in sorted(p for p in model_root.iterdir() if p.is_dir()):
            manifests = run / "manifests"
            if not manifests.is_dir():
                continue
            exp = empty_experiment(
                f"model-run/{model}/{run.name}",
                model,
                "operational_forecast" if "2020_202" in run.name else "case_or_pilot_forecast",
                run,
                "per_initialization_json_manifests",
            )
            for manifest in sorted(manifests.glob("*/*.json")):
                try:
                    item = read_json(manifest)
                except (OSError, json.JSONDecodeError) as exc:
                    exp["notes"].append(f"unreadable manifest {manifest}: {exc}")
                    continue
                output = manifest_output(item)
                init_date = item.get("init_date") or item.get("initialization_date") or infer_date(manifest.name)
                if not output or not init_date:
                    continue
                members = item.get("members") or item.get("member_count")
                if members is None and isinstance(item.get("member_seeds"), list):
                    members = len(item["member_seeds"])
                variables = manifest_variables(item)
                record = file_identity(
                    Path(output),
                    manifest_status=item.get("status"),
                    declared_size=item.get("output_size_bytes") or item.get("qc", {}).get("size_bytes"),
                    declared_sha256=item.get("output_sha256"),
                    variable=",".join(variables) or None,
                    forecast_type="native_ensemble" if (members or 1) > 1 else "deterministic",
                    members=int(members) if members is not None else None,
                    lead_days=item.get("lead_days"),
                    manifest_path=manifest,
                    verify_checksum=verify_checksums,
                )
                record["manifest_variables"] = variables
                record["temporal_statistics"] = {
                    key: item[key]
                    for key in ("daily_t2m_method", "daily_tp_method", "product", "benchmark_mode")
                    if key in item
                }
                add_file(exp, str(init_date)[:10], record)
            if exp["cases"] or exp["notes"]:
                experiments.append(exp)
    return experiments


def scan_erpas(storage: Path, verify_checksums: bool = False) -> list[dict[str, Any]]:
    root = storage / "raw/erpas"
    manifest_root = storage / "manifests/erpas/google_drive_20260714"
    source_inventory_path = manifest_root / "inventory.json"
    sha_path = manifest_root / "sha256.txt"
    declared_sizes: dict[str, int] = {}
    declared_sha: dict[str, str] = {}
    if source_inventory_path.is_file():
        for item in read_json(source_inventory_path):
            if not item.get("IsDir") and item.get("Path"):
                declared_sizes[str(item["Path"]).lstrip("./")] = int(item["Size"])
    if sha_path.is_file():
        with sha_path.open(encoding="utf-8") as handle:
            for line in handle:
                digest, relative = line.rstrip().split(maxsplit=1)
                declared_sha[relative.lstrip("./")] = digest
    exp = empty_experiment(
        "provider/erpas_forecast_2023_2025",
        "erpas",
        "provider_ensemble_mean_forecast",
        root / "forecast",
        "bounded_provider_tree_scan",
    )
    for annual in sorted((root / "forecast").glob("annual20??")):
        for variable_dir in sorted(p for p in annual.iterdir() if p.is_dir()):
            for path in sorted(p for p in variable_dir.iterdir() if p.is_file()):
                if path.suffix.lower() not in {".grb", ".grib", ".grib2"}:
                    continue
                init_date = infer_date(path.name)
                if not init_date:
                    continue
                relative = str(path.relative_to(root))
                record = file_identity(
                    path,
                    manifest_status="valid" if relative in declared_sizes and relative in declared_sha else None,
                    declared_size=declared_sizes.get(relative),
                    declared_sha256=declared_sha.get(relative),
                    variable=variable_dir.name,
                    forecast_type="precomputed_ensemble_mean",
                    members=1,
                    lead_days=33,
                    manifest_path=sha_path if sha_path.is_file() else source_inventory_path,
                    verify_checksum=verify_checksums,
                )
                record["source_member_note"] = "unweighted mean of four source forecasts; raw members unavailable"
                add_file(exp, init_date, record)
    if exp["cases"]:
        exp["notes"].append("ERPAS uses a Wednesday calendar and 33 leads; it is not directly calendar-equivalent to the main physics archive.")
    return [exp] if exp["cases"] else []


def scan_fuxi_archive(root: Path) -> list[dict[str, Any]]:
    exp = empty_experiment(
        "reforecast/fuxi_native_2002_2021",
        "fuxi",
        "native_reforecast_archive",
        root,
        "bounded_archive_directory_scan",
    )
    for path in sorted(root.glob("20??????.7z")):
        init_date = infer_date(path.name)
        if not init_date:
            continue
        record = file_identity(
            path,
            variable="76_native_channels",
            forecast_type="native_stochastic_ensemble",
            members=51,
            lead_days=42,
        )
        record["member_ids"] = [f"{member:02d}" for member in range(51)]
        record["member_id_source"] = "confirmed_from_archive_directory_listing"
        record["archive_layout"] = "member/00-50, daily files 01.nc-42.nc"
        add_file(exp, init_date, record)
    exp["notes"].append("FuXi member numbers 00-50 do not identify a deterministic control member.")
    return [exp] if exp["cases"] else []


def scan_reforecast_manifests(storage: Path, verify_checksums: bool) -> list[dict[str, Any]]:
    experiments = []
    for provider in ("ecmwf", "ukmo", "ncep", "cma", "cnrm"):
        root = storage / "manifests" / provider / "reforecast"
        if not root.is_dir():
            continue
        exp = empty_experiment(
            f"reforecast/{provider}_native_climatology",
            provider,
            "native_reforecast_or_smoke_test",
            storage / "raw" / provider / "reforecast",
            "append_only_jsonl_manifests",
        )
        for manifest in sorted(root.glob("*.jsonl")):
            with manifest.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    target = item.get("target")
                    if not target:
                        continue
                    init_date = item.get("version_date") or infer_date(target) or "1900-01-01"
                    record = file_identity(
                        Path(target),
                        manifest_status=item.get("status"),
                        declared_size=item.get("size_bytes"),
                        declared_sha256=item.get("sha256"),
                        variable=item.get("field"),
                        forecast_type=item.get("forecast_type"),
                        members=item.get("member_count"),
                        lead_days=item.get("lead_days"),
                        manifest_path=manifest,
                        verify_checksum=verify_checksums,
                    )
                    record["smoke_test"] = bool(item.get("smoke"))
                    record["hindcast_dates"] = item.get("hindcast_dates")
                    add_file(exp, str(init_date)[:10], record)
        if exp["cases"]:
            experiments.append(exp)
    return experiments


def choose_open_qc_records(experiment: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    records = [record for case in experiment["cases"].values() for record in case["files"]]
    records = [record for record in records if record["format"] in {"netcdf", "grib"}]
    if mode == "none" or not records:
        return []
    if mode == "all" or len(records) <= 3:
        return records
    indices = sorted({0, len(records) // 2, len(records) - 1})
    return [records[index] for index in indices]


def finalize_experiment(
    experiment: dict[str, Any],
    expected_core: dict[int, set[str]],
    expected_cnrm: dict[int, set[str]],
) -> None:
    years: dict[int, list[str]] = defaultdict(list)
    file_statuses = Counter()
    variables = Counter()
    formats = Counter()
    member_counts = Counter()
    lead_counts = Counter()
    open_qc_statuses = Counter()
    total_bytes = 0
    for date, case in sorted(experiment["cases"].items()):
        years[int(date[:4])].append(date)
        reasons = []
        case_variables = set()
        case_formats = set()
        case_leads = set()
        perturbed_counts = []
        other_member_counts = []
        for record in case["files"]:
            file_statuses[record["status"]] += 1
            open_qc_statuses[record["open_qc"]["status"]] += 1
            if record.get("variable"):
                variables[record["variable"]] += 1
                case_variables.update(str(record["variable"]).split(","))
            formats[record["format"]] += 1
            case_formats.add(record["format"])
            if record.get("members") is not None:
                member_counts[str(record["members"])] += 1
                if record.get("forecast_type") == "pf":
                    perturbed_counts.append(int(record["members"]))
                elif record.get("forecast_type") != "cf":
                    other_member_counts.append(int(record["members"]))
            if record.get("lead_days") is not None:
                lead_counts[str(record["lead_days"])] += 1
                case_leads.add(int(record["lead_days"]))
            total_bytes += record.get("size_bytes") or 0
            if record["status"] in {"missing", "empty", "metadata_mismatch", "unreadable"}:
                reasons.append(f"{record['status']}:{record['path']}")
        case["exclusion_reasons"] = reasons
        case["variables"] = sorted(case_variables)
        case["formats"] = sorted(case_formats)
        case["lead_days"] = sorted(case_leads)
        if perturbed_counts:
            case["ensemble_member_count"] = 1 + max(perturbed_counts)
            case["ensemble_structure"] = "control_plus_perturbed"
        elif other_member_counts:
            case["ensemble_member_count"] = max(other_member_counts)
            case["ensemble_structure"] = "native_or_precomputed"
        else:
            case["ensemble_member_count"] = None
            case["ensemble_structure"] = "unknown"
        if case_leads:
            start = dt.date.fromisoformat(date)
            case["valid_time_start"] = f"{date}T00:00:00Z"
            case["valid_time_end"] = f"{start + dt.timedelta(days=max(case_leads)):%Y-%m-%d}T00:00:00Z"
        else:
            case["valid_time_start"] = None
            case["valid_time_end"] = None
        case["acc_ready"] = not reasons and experiment["product"] in {
            "operational_forecast",
            "provider_ensemble_mean_forecast",
        }

    coverage = {}
    for year, dates in sorted(years.items()):
        observed = set(dates)
        expected = expected_cnrm.get(year, set()) if experiment["model"] == "cnrm" else expected_core.get(year, set())
        coverage[str(year)] = {
            "initialization_count": len(observed),
            "first_initialization": min(observed),
            "last_initialization": max(observed),
            "expected_count": len(expected) if expected else None,
            "missing_expected_dates": sorted(expected - observed) if expected else [],
            "unexpected_dates": sorted(observed - expected) if expected else [],
        }
    experiment["summary"] = {
        "initialization_count": len(experiment["cases"]),
        "years": sorted(years),
        "coverage_by_year": coverage,
        "file_count": sum(file_statuses.values()),
        "total_size_bytes": total_bytes,
        "file_statuses": dict(sorted(file_statuses.items())),
        "variables": dict(sorted(variables.items())),
        "formats": dict(sorted(formats.items())),
        "member_counts": dict(sorted(member_counts.items())),
        "lead_day_counts": dict(sorted(lead_counts.items())),
        "open_qc_statuses": dict(sorted(open_qc_statuses.items())),
        "acc_ready_cases": sum(case["acc_ready"] for case in experiment["cases"].values()),
    }
    experiment["cases"] = dict(sorted(experiment["cases"].items()))


def coverage_rows(experiments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for exp in experiments:
        summary = exp["summary"]
        for year, coverage in summary["coverage_by_year"].items():
            year_cases = [case for date, case in exp["cases"].items() if date.startswith(f"{year}-")]
            year_records = [record for case in year_cases for record in case["files"]]
            year_variables = sorted({part for record in year_records for part in str(record.get("variable") or "").split(",") if part})
            year_formats = sorted({record["format"] for record in year_records})
            year_members = sorted({str(record["members"]) for record in year_records if record.get("members") is not None}, key=int)
            year_leads = sorted({str(record["lead_days"]) for record in year_records if record.get("lead_days") is not None}, key=int)
            year_statuses = Counter(record["status"] for record in year_records)
            rows.append(
                {
                    "experiment_id": exp["experiment_id"],
                    "model": exp["model"],
                    "product": exp["product"],
                    "year": year,
                    "initialization_count": coverage["initialization_count"],
                    "expected_count": coverage["expected_count"],
                    "missing_expected_count": len(coverage["missing_expected_dates"]),
                    "unexpected_count": len(coverage["unexpected_dates"]),
                    "first_initialization": coverage["first_initialization"],
                    "last_initialization": coverage["last_initialization"],
                    "variables": ";".join(year_variables),
                    "formats": ";".join(year_formats),
                    "member_counts": ";".join(year_members),
                    "lead_days": ";".join(year_leads),
                    "file_statuses": json.dumps(year_statuses, sort_keys=True),
                }
            )
    return rows


def issue_rows(experiments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for exp in experiments:
        for date, case in exp["cases"].items():
            for record in case["files"]:
                if record["issues"] or record["status"] in {"missing", "empty", "unreadable"}:
                    rows.append(
                        {
                            "experiment_id": exp["experiment_id"],
                            "model": exp["model"],
                            "initialization_date": date,
                            "variable": record.get("variable"),
                            "path": record["path"],
                            "status": record["status"],
                            "issues": ";".join(record["issues"]),
                            "open_qc": record["open_qc"]["status"],
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_report(catalog: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    lines = [
        "# S2S storage inventory",
        "",
        f"Generated: `{catalog['generated_at']}`",
        "",
        "This is a read-only, manifest-first inventory. `manifest_valid` means the file passed its producing pipeline's validation and still has the expected identity metadata. Only records with `open_qc.status=opened` were reopened during this audit.",
        "",
        "## Dataset summary",
        "",
        "| experiment | product | years | ICs | files | members | leads | variables | ACC-ready |",
        "|---|---|---:|---:|---:|---|---|---|---:|",
    ]
    for exp in catalog["experiments"].values():
        summary = exp["summary"]
        lines.append(
            "| {id} | {product} | {years} | {ics} | {files} | {members} | {leads} | {variables} | {ready} |".format(
                id=exp["experiment_id"],
                product=exp["product"],
                years=",".join(map(str, summary["years"])),
                ics=summary["initialization_count"],
                files=summary["file_count"],
                members=",".join(summary["member_counts"]) or "unknown",
                leads=",".join(summary["lead_day_counts"]) or "unknown",
                variables=",".join(summary["variables"]) or "unknown",
                ready=summary["acc_ready_cases"],
            )
        )
    lines.extend(
        [
            "",
            "## Integrity findings",
            "",
            f"Files with inventory issues: **{len(issues)}**.",
            "",
        ]
    )
    if issues:
        counts = Counter(row["status"] for row in issues)
        for status, count in sorted(counts.items()):
            lines.append(f"- `{status}`: {count}")
    else:
        lines.append("No missing, empty, signature-mismatched, manifest-size-mismatched, checksum-mismatched, or sampled-open failures were found.")
    open_counts = Counter()
    for exp in catalog["experiments"].values():
        open_counts.update(exp["summary"]["open_qc_statuses"])
    lines.extend(
        [
            "",
            "## Sampled metadata-open checks",
            "",
            *(f"- `{status}`: {count}" for status, count in sorted(open_counts.items())),
            "",
            "A timeout means the decoder exceeded the bounded audit window; it is not classified as corruption.",
            "",
            "## Calendar gaps",
            "",
        ]
    )
    gaps = []
    for exp in catalog["experiments"].values():
        if exp["product"] != "operational_forecast":
            continue
        for year, coverage in exp["summary"]["coverage_by_year"].items():
            if coverage["missing_expected_dates"]:
                gaps.append((exp["experiment_id"], year, coverage["missing_expected_dates"]))
    if gaps:
        for experiment_id, year, missing in gaps:
            shown = missing[:12]
            remainder = f" (+{len(missing) - len(shown)} more in inventory.json)" if len(missing) > len(shown) else ""
            lines.append(f"- `{experiment_id}` {year}: {len(missing)} missing expected IC(s): {', '.join(shown)}{remainder}")
    else:
        lines.append("No expected initialization dates are missing from operational experiments.")
    lines.extend(
        [
            "",
            "See `inventory.json` for initialization- and file-level records, `coverage_matrix.csv` for year coverage, and `file_issues.csv` for exact exclusions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    expected_core, expected_cnrm = read_calendar(args.calendar)
    experiments: list[dict[str, Any]] = []
    experiments.extend(scan_physics(args.storage_root, args.verify_checksums))
    experiments.extend(scan_model_runs(args.storage_root, args.verify_checksums))
    experiments.extend(scan_reforecast_manifests(args.storage_root, args.verify_checksums))
    if not args.skip_erpas:
        experiments.extend(scan_erpas(args.storage_root, args.verify_checksums))
    if not args.skip_fuxi_archive:
        experiments.extend(scan_fuxi_archive(args.fuxi_archive))

    for experiment in experiments:
        for record in choose_open_qc_records(experiment, args.open_qc):
            open_metadata_qc(record, args.open_timeout_seconds)
        finalize_experiment(experiment, expected_core, expected_cnrm)

    experiments.sort(key=lambda item: item["experiment_id"])
    catalog = {
        "schema_version": 1,
        "generated_at": iso_now(),
        "audit_mode": {
            "storage_is_read_only": True,
            "manifest_first": True,
            "open_qc": args.open_qc,
            "checksums_recomputed": args.verify_checksums,
            "corruption_limit": "A successful header/open check cannot prove every data block is intact; declared SHA-256 verification is definitive for model outputs when enabled.",
        },
        "calendar": str(args.calendar),
        "storage_roots": [str(args.storage_root), str(args.fuxi_archive)],
        "status_definitions": {
            "manifest_valid": "producer validation passed; file exists and current size/signature agree",
            "opened_valid": "file opened successfully during this audit without producer validation",
            "present_unvalidated": "file exists with a valid signature but was not opened or producer-validated",
            "metadata_mismatch": "size, signature, or optional checksum disagrees",
            "unreadable": "the selected backend could not open the file",
            "missing": "declared physical file is absent",
            "empty": "physical file has zero bytes",
        },
        "experiments": {exp["experiment_id"]: exp for exp in experiments},
    }
    issues = issue_rows(experiments)
    rows = coverage_rows(experiments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inventory.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "coverage_matrix.csv", rows)
    write_csv(args.output_dir / "file_issues.csv", issues)
    (args.output_dir / "AUDIT_REPORT.md").write_text(render_report(catalog, issues), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "experiments": len(experiments),
        "cases": sum(exp["summary"]["initialization_count"] for exp in experiments),
        "files": sum(exp["summary"]["file_count"] for exp in experiments),
        "issues": len(issues),
    }, indent=2))
    return 1 if any(row["status"] in {"missing", "empty", "unreadable"} for row in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
