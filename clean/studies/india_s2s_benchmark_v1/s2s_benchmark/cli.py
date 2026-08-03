from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .catalog import build_catalog
from .readers import EXPERIMENT_IDS, MODEL_VARIABLES, load_field
from .spatial import build_spatial_support, validate_spatial_support
from .storage import store_path, validate_store, write_manifest, write_store
from .core import COMMON_LAT, COMMON_LON, sha256_file


STUDY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = STUDY_ROOT.parents[1]
DEFAULT_CONFIG = STUDY_ROOT / "config/benchmark.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="India S2S Benchmark v1 preprocessing")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive-root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    sub.add_parser("build-spatial")
    plots = sub.add_parser("plot-pilot-qc")
    plots.add_argument("--output", type=Path)
    sub.add_parser("pilot")
    pilot_task = sub.add_parser("pilot-task")
    pilot_task.add_argument("--task-index", required=True, type=int)
    matrix = sub.add_parser("make-matrix")
    matrix.add_argument("--pilot", action="store_true")
    matrix.add_argument("--output", type=Path, required=True)
    matrix_task = sub.add_parser("matrix-task")
    matrix_task.add_argument("--matrix", type=Path, required=True)
    matrix_task.add_argument("--task-index", required=True, type=int)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--pilot", action="store_true")
    finalize.add_argument("--expected-tasks", type=int)
    preprocess = sub.add_parser("preprocess")
    preprocess.add_argument("--model", required=True, choices=sorted(MODEL_VARIABLES))
    preprocess.add_argument("--variable", required=True)
    preprocess.add_argument("--year", required=True, type=int)
    preprocess.add_argument("--grid", default="common_1p5", choices=["common_1p5", "source_native_india"])
    preprocess.add_argument("--lead-days", type=int)
    validate = sub.add_parser("validate")
    validate.add_argument("--pilot", action="store_true")
    index = sub.add_parser("build-index")
    index.add_argument("--pilot", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--pilot", action="store_true")
    return parser.parse_args(argv)


def load_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inventory_path = Path(config["inventory"])
    if not inventory_path.is_absolute():
        inventory_path = WORKSPACE / inventory_path
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    root = args.archive_root or Path(config["archive_root"])
    return config, inventory, root


def grids_for(model: str) -> tuple[str, ...]:
    return ("source_native_india", "common_1p5") if model == "erpas" else ("common_1p5",)


def manifest_path(root: Path, field, grid: str, year: int) -> Path:
    safe = field.experiment_id.replace("/", "__")
    return root / "manifests" / field.model / safe / field.variable / grid / f"{year}.json"


def _write_one(
    root: Path,
    field,
    grid: str,
    year: int,
    *,
    member_axis: np.ndarray | None = None,
) -> dict:
    destination = store_path(root, field, grid, year)
    path = manifest_path(root, field, grid, year)
    if destination.exists() and path.exists():
        validation = validate_store(destination)
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("zmetadata_sha256") != validation["zmetadata_sha256"]:
            raise ValueError(f"completed store metadata differs from manifest: {destination}")
        return existing
    manifest = write_store([field], destination, grid, member_axis=member_axis)
    validation = validate_store(destination)
    manifest["validation"] = validation
    manifest["manifest_path"] = str(path)
    write_manifest(manifest, path)
    return manifest


def inventory_command(config: dict[str, Any], inventory: dict[str, Any], root: Path) -> dict:
    rows = []
    for model, experiment_id in EXPERIMENT_IDS.items():
        source = inventory["experiments"][experiment_id]
        summary = source["summary"]
        variables = MODEL_VARIABLES[model]
        rows.append({
            "model": model,
            "experiment_id": experiment_id,
            "years": ",".join(map(str, summary["years"])),
            "initialization_count": summary["initialization_count"],
            "usable_initialization_count": summary["acc_ready_cases"],
            "variables": ",".join(variables),
            "lead_day_counts": json.dumps(summary["lead_day_counts"], sort_keys=True),
            "source_root": source["root"],
        })
    output = root / "inventory"
    output.mkdir(parents=True, exist_ok=True)
    table = output / "source_inventory.parquet"
    pd.DataFrame(rows).to_parquet(table, index=False)
    gaps = {
        "schema_version": 1,
        "fcn3_2021-03-08": "missing after non-finite ERA5 initial condition; 516/517 available",
        "erpas_tp_india_0p5_2024-12-18": "not supplied; global ERPAS TP is available",
        "future_2025_ai": "DLESyM, NeuralGCM, and FCN3 2025 are not present yet and can be appended",
    }
    gap_path = output / "known_gaps.json"
    gap_path.write_text(json.dumps(gaps, indent=2, sort_keys=True) + "\n")
    return {"inventory": str(table), "known_gaps": str(gap_path), "models": len(rows)}


def spatial_command(config: dict[str, Any], root: Path) -> dict:
    support = config["spatial_support"]
    destination = root / support["output"]
    result = build_spatial_support(
        Path(support["source"]),
        destination,
        support["regions"],
        COMMON_LAT,
        COMMON_LON,
        config["common_grid"]["id"],
    )
    manifest_path = destination.with_suffix(".manifest.json")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in result.items() if key != "created_at"}
        existing_comparable = {key: value for key, value in existing.items() if key != "created_at"}
        if comparable != existing_comparable:
            raise FileExistsError(f"immutable spatial manifest differs: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["manifest"] = str(manifest_path)
    return result


def pilot_command(config: dict[str, Any], inventory: dict[str, Any], root: Path) -> dict:
    pilot = config["pilot"]
    pilot_root = root / "pilots" / pilot["id"]
    results = []
    failures = []
    for model in pilot["models"]:
        initialization = pilot["erpas_initialization"] if model == "erpas" else pilot["dense_initialization"]
        for variable in MODEL_VARIABLES[model]:
            for grid in grids_for(model):
                try:
                    field = load_field(
                        inventory,
                        model,
                        variable,
                        initialization,
                        pilot["lead_days"],
                        grid,
                        config["native_india_box"],
                    )
                    manifest = _write_one(
                        pilot_root,
                        field,
                        grid,
                        int(initialization[:4]),
                        member_axis=np.asarray(field.member),
                    )
                    results.append(manifest)
                except Exception as error:
                    failures.append({
                        "model": model,
                        "variable": variable,
                        "grid": grid,
                        "initialization": initialization,
                        "error": f"{type(error).__name__}: {error}",
                    })
    report = {
        "schema_version": 1,
        "pilot_id": pilot["id"],
        "purpose": "adapter and archive validation only; not a model ranking",
        "completed_store_count": len(results),
        "failure_count": len(failures),
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    report_path = pilot_root / "pilot_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"pilot has {len(failures)} failures; see {report_path}")
    return report


def pilot_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    pilot = config["pilot"]
    specs = []
    for model in pilot["models"]:
        initialization = pilot["erpas_initialization"] if model == "erpas" else pilot["dense_initialization"]
        for variable in MODEL_VARIABLES[model]:
            for grid in grids_for(model):
                specs.append({
                    "model": model,
                    "variable": variable,
                    "grid": grid,
                    "initialization": initialization,
                    "lead_days": pilot["lead_days"],
                    "year": int(initialization[:4]),
                })
    return specs


def pilot_task_command(
    args: argparse.Namespace, config: dict[str, Any], inventory: dict[str, Any], root: Path
) -> dict:
    specs = pilot_specs(config)
    if not 0 <= args.task_index < len(specs):
        raise IndexError(f"pilot task index {args.task_index} outside 0..{len(specs)-1}")
    spec = specs[args.task_index]
    field = load_field(
        inventory,
        spec["model"],
        spec["variable"],
        spec["initialization"],
        spec["lead_days"],
        spec["grid"],
        config["native_india_box"],
    )
    base = root / "pilots" / config["pilot"]["id"]
    manifest = _write_one(
        base, field, spec["grid"], spec["year"], member_axis=np.asarray(field.member)
    )
    return {"task_index": args.task_index, "spec": spec, "store": manifest["store"], "status": "passed"}


def full_specs(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for model, experiment_id in EXPERIMENT_IDS.items():
        source = inventory["experiments"][experiment_id]
        for year in source["summary"]["years"]:
            cases = {
                date: case for date, case in source["cases"].items() if date.startswith(str(year))
            }
            for variable in MODEL_VARIABLES[model]:
                if not any(_variable_present(case, model, variable) for case in cases.values()):
                    continue
                for grid in grids_for(model):
                    specs.append({"model": model, "variable": variable, "year": year, "grid": grid})
    return specs


def make_matrix_command(
    args: argparse.Namespace, config: dict[str, Any], inventory: dict[str, Any]
) -> dict:
    specs = pilot_specs(config) if args.pilot else full_specs(inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scope": config["pilot"]["id"] if args.pilot else "full",
        "task_count": len(specs),
        "tasks": specs,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"matrix": str(args.output), "scope": payload["scope"], "task_count": len(specs)}


def matrix_task_command(
    args: argparse.Namespace, config: dict[str, Any], inventory: dict[str, Any], root: Path
) -> dict:
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if not 0 <= args.task_index < len(matrix["tasks"]):
        raise IndexError(f"matrix task index {args.task_index} is invalid")
    task = matrix["tasks"][args.task_index]
    if matrix["scope"] == config["pilot"]["id"]:
        pilot_args = argparse.Namespace(task_index=args.task_index)
        return pilot_task_command(pilot_args, config, inventory, root)
    preprocess_args = argparse.Namespace(
        model=task["model"], variable=task["variable"], year=task["year"],
        grid=task["grid"], lead_days=None,
    )
    manifest = preprocess_command(preprocess_args, config, inventory, root)
    return {"task_index": args.task_index, "spec": task, "store": manifest["store"], "status": "passed"}


def _variable_present(case: dict[str, Any], model: str, variable: str) -> bool:
    if variable == "t2m_proxy":
        return any(item["variable"] == "surface" for item in case["files"])
    source = {"tsfc": "surface_temperature", "gh": "geopotential_height"}.get(variable, variable)
    return any(
        item["variable"] == source
        or source in item["variable"].split(",")
        or source in (item.get("manifest_variables") or [])
        or source in (item.get("manifest_fields") or {})
        or (source == "tp" and item["variable"] == "surface")
        for item in case["files"]
    )


def preprocess_command(
    args: argparse.Namespace, config: dict[str, Any], inventory: dict[str, Any], root: Path
) -> dict:
    if args.variable not in MODEL_VARIABLES[args.model]:
        raise ValueError(f"{args.variable} is not registered for {args.model}")
    if args.grid not in grids_for(args.model):
        raise ValueError(f"{args.model} does not expose {args.grid}")
    source = inventory["experiments"][EXPERIMENT_IDS[args.model]]
    dates = [
        date for date, case in sorted(source["cases"].items())
        if date.startswith(str(args.year)) and _variable_present(case, args.model, args.variable)
    ]
    if not dates:
        raise FileNotFoundError(f"no {args.model}/{args.variable} cases for {args.year}")
    lead_limit = args.lead_days or max(max(source["cases"][date]["lead_days"]) for date in dates)
    max_members = max(source["cases"][date]["ensemble_member_count"] for date in dates)
    member_axis = np.arange(max_members, dtype=np.int16)

    def fields() -> Iterable:
        for date in dates:
            yield load_field(
                inventory, args.model, args.variable, date, lead_limit, args.grid,
                config["native_india_box"],
            )

    iterator = fields()
    first = next(iterator)
    destination = store_path(root, first, args.grid, args.year)
    existing_manifest_path = manifest_path(root, first, args.grid, args.year)
    if destination.exists() and existing_manifest_path.exists():
        validation = validate_store(destination)
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing.get("zmetadata_sha256") != validation["zmetadata_sha256"]:
            raise ValueError(f"completed store metadata differs from manifest: {destination}")
        return existing

    def all_fields():
        yield first
        yield from iterator

    manifest = write_store(all_fields(), destination, args.grid, member_axis=member_axis)
    manifest["validation"] = validate_store(destination)
    path = manifest_path(root, first, args.grid, args.year)
    manifest["manifest_path"] = str(path)
    write_manifest(manifest, path)
    return manifest


def scope_root(config: dict[str, Any], root: Path, pilot: bool) -> tuple[Path, str]:
    if pilot:
        return root / "pilots" / config["pilot"]["id"], config["pilot"]["id"]
    return root, "full"


def manifest_files(base: Path) -> list[Path]:
    return sorted((base / "manifests").rglob("*.json")) if (base / "manifests").exists() else []


def validate_command(config: dict[str, Any], root: Path, pilot: bool) -> dict:
    base, scope = scope_root(config, root, pilot)
    paths = manifest_files(base)
    results = []
    for path in paths:
        manifest = json.loads(path.read_text())
        results.append(validate_store(Path(manifest["store"])))
    report = {"scope": scope, "store_count": len(results), "status": "passed", "stores": results}
    output = base / "validation_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def index_command(config: dict[str, Any], root: Path, pilot: bool) -> dict:
    base, scope = scope_root(config, root, pilot)
    paths = manifest_files(base)
    if not paths:
        raise FileNotFoundError(f"no manifests below {base}")
    json_path, parquet_path = build_catalog(paths, base / "indexes", scope)
    return {"catalog": str(json_path), "init_index": str(parquet_path), "manifest_count": len(paths)}


def finalize_command(
    args: argparse.Namespace, config: dict[str, Any], root: Path
) -> dict:
    validation = validate_command(config, root, args.pilot)
    base, scope = scope_root(config, root, args.pilot)
    expected = args.expected_tasks if args.expected_tasks is not None else (
        len(pilot_specs(config)) if args.pilot else None
    )
    if expected is not None and validation["store_count"] != expected:
        raise ValueError(
            f"{scope}: expected {expected} completed stores, found {validation['store_count']}"
        )
    incomplete = list(base.rglob("*.incomplete-*"))
    if incomplete:
        raise ValueError(f"{scope}: incomplete stores remain: {[str(path) for path in incomplete]}")
    spatial_path = root / config["spatial_support"]["output"]
    spatial = validate_spatial_support(spatial_path)
    index = index_command(config, root, args.pilot)
    report = {
        "schema_version": 1,
        "scope": scope,
        "status": "passed",
        "expected_store_count": expected,
        "completed_store_count": validation["store_count"],
        "catalog": index["catalog"],
        "catalog_sha256": sha256_file(Path(index["catalog"])),
        "init_index": index["init_index"],
        "init_index_sha256": sha256_file(Path(index["init_index"])),
        "spatial_support": spatial["store"],
        "spatial_support_zmetadata_sha256": spatial["zmetadata_sha256"],
    }
    report_path = base / "finalization_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def status_command(config: dict[str, Any], root: Path, pilot: bool) -> dict:
    base, scope = scope_root(config, root, pilot)
    paths = manifest_files(base)
    manifests = [json.loads(path.read_text()) for path in paths]
    incomplete = list(base.rglob("*.incomplete-*")) if base.exists() else []
    return {
        "scope": scope,
        "complete_store_count": len(manifests),
        "incomplete_store_count": len(incomplete),
        "complete_stores": [item["store"] for item in manifests],
        "incomplete_stores": [str(path) for path in incomplete],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config, inventory, root = load_context(args)
    if args.command == "inventory":
        result = inventory_command(config, inventory, root)
    elif args.command == "build-spatial":
        result = spatial_command(config, root)
    elif args.command == "plot-pilot-qc":
        from .plotting import plot_pilot_qc

        result = plot_pilot_qc(config, root, args.output)
    elif args.command == "pilot":
        result = pilot_command(config, inventory, root)
    elif args.command == "pilot-task":
        result = pilot_task_command(args, config, inventory, root)
    elif args.command == "make-matrix":
        result = make_matrix_command(args, config, inventory)
    elif args.command == "matrix-task":
        result = matrix_task_command(args, config, inventory, root)
    elif args.command == "finalize":
        result = finalize_command(args, config, root)
    elif args.command == "preprocess":
        result = preprocess_command(args, config, inventory, root)
    elif args.command == "validate":
        result = validate_command(config, root, args.pilot)
    elif args.command == "build-index":
        result = index_command(config, root, args.pilot)
    elif args.command == "status":
        result = status_command(config, root, args.pilot)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
