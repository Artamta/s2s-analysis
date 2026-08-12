"""Run directories, event logs, manifests, and the one-shot test lock."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import canonical_config, config_sha256, write_json


FREEZE_SCHEMA_VERSION = 2
FROZEN_RUN_ARTIFACTS = {
    "checkpoint": Path("checkpoints/best.pt"),
    "success": Path("SUCCESS.json"),
    "normalization": Path("normalization.json"),
    "resolved_config": Path("resolved_config.json"),
    "source_manifest": Path("source_manifest.json"),
    "source_snapshot": Path("source_snapshot.zip"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_run_directory(output_root: Path, experiment: str, model: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(output_root) / "runs" / f"{experiment}__{model}__seed{seed}__{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(str(base) + f"__{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    for name in ("checkpoints", "logs", "metrics", "predictions", "figures"):
        (candidate / name).mkdir()
    return candidate


class EventLogger:
    """Append-only, line-buffered JSON event logger."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def log(self, event: str, **payload: Any) -> None:
        record = {"time": utc_now(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _command_output(command: Iterable[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            list(command), check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def environment_manifest() -> Dict[str, Any]:
    packages: Dict[str, Optional[str]] = {}
    for module_name in ("numpy", "pandas", "xarray", "zarr", "torch", "matplotlib"):
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # recorded rather than hiding a broken environment
            packages[module_name] = f"unavailable: {type(exc).__name__}: {exc}"
    torch_details: Dict[str, Any] = {}
    try:
        import torch

        torch_details = {
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except Exception:
        pass
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": packages,
        "torch": torch_details,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--short", "--", "."]),
    }


def initialize_run(
    run_directory: Path,
    config: Dict[str, Any],
    model: str,
    seed: int,
    split_manifest: Dict[str, Any],
) -> None:
    package_directory = Path(__file__).resolve().parent
    shutil.make_archive(
        str(run_directory / "source_snapshot"),
        "zip",
        root_dir=package_directory.parent,
        base_dir=package_directory.name,
    )
    write_json(run_directory / "resolved_config.json", canonical_config(config))
    write_json(run_directory / "environment.json", environment_manifest())
    write_json(run_directory / "split_manifest.json", split_manifest)
    write_json(
        run_directory / "RUNNING.json",
        {
            "status": "running",
            "created_at": utc_now(),
            "model": model,
            "seed": seed,
            "config_sha256": config_sha256(config),
        },
    )


def mark_success(run_directory: Path, payload: Dict[str, Any]) -> None:
    running = run_directory / "RUNNING.json"
    if running.exists():
        running.unlink()
    write_json(run_directory / "SUCCESS.json", {"status": "success", "time": utc_now(), **payload})


def mark_failure(run_directory: Path, message: str) -> None:
    write_json(
        run_directory / "FAILED.json",
        {"status": "failed", "time": utc_now(), "message": message},
    )


def freeze_development_run(
    output_path: Path,
    config: Dict[str, Any],
    run_directories: Iterable[Path],
) -> Dict[str, Any]:
    """Freeze chosen development checkpoints before the test set is opened."""

    if Path(output_path).exists():
        raise FileExistsError(f"freeze manifest already exists: {output_path}")
    frozen_config = canonical_config(config)
    runs = []
    for directory in run_directories:
        directory = Path(directory).resolve()
        artifact_paths = {
            name: directory / relative
            for name, relative in FROZEN_RUN_ARTIFACTS.items()
        }
        missing = [path for path in artifact_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "run is incomplete; missing frozen artifacts: "
                + ", ".join(str(path) for path in missing)
            )

        with artifact_paths["success"].open("r", encoding="utf-8") as stream:
            success_payload = json.load(stream)
        if success_payload.get("status") != "success":
            raise ValueError(f"run SUCCESS.json does not report success: {directory}")

        with artifact_paths["resolved_config"].open("r", encoding="utf-8") as stream:
            run_config = json.load(stream)
        if canonical_config(run_config) != frozen_config:
            raise ValueError(
                f"run resolved_config.json does not match freeze configuration: {directory}"
            )

        frozen_run: Dict[str, Any] = {"run_directory": str(directory)}
        for name, path in artifact_paths.items():
            frozen_run[name] = str(path.resolve())
            frozen_run[f"{name}_sha256"] = file_sha256(path)
        runs.append(frozen_run)
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen_for_single_test_evaluation",
        "created_at": utc_now(),
        "config": frozen_config,
        "config_sha256": config_sha256(config),
        "runs": runs,
        "test_evaluated": False,
    }
    write_json(output_path, manifest)
    return manifest


def load_unused_freeze(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "frozen_for_single_test_evaluation":
        raise ValueError("invalid freeze manifest status")
    if manifest.get("test_evaluated"):
        raise RuntimeError("this freeze manifest has already consumed the test set")
    if manifest.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise RuntimeError(
            "freeze manifest does not bind all preprocessing artifacts; create a new freeze"
        )
    if manifest.get("config_sha256") != config_sha256(dict(manifest.get("config", {}))):
        raise RuntimeError("frozen configuration hash is invalid")
    for run in manifest["runs"]:
        directory = Path(run["run_directory"]).resolve()
        for name, relative in FROZEN_RUN_ARTIFACTS.items():
            path_key = name
            hash_key = f"{name}_sha256"
            if path_key not in run or hash_key not in run:
                raise RuntimeError(f"freeze manifest does not bind required {name} artifact")
            artifact = Path(run[path_key]).resolve()
            expected_path = (directory / relative).resolve()
            if artifact != expected_path:
                raise RuntimeError(
                    f"frozen {name} path is inconsistent with its run directory: {artifact}"
                )
            if not artifact.is_file():
                raise RuntimeError(f"frozen {name} artifact is missing: {artifact}")
            if file_sha256(artifact) != run[hash_key]:
                raise RuntimeError(f"{name} changed after freeze: {artifact}")
    return manifest


def mark_test_consumed(path: Path, result_directory: Path) -> None:
    """Atomically mark a freeze manifest used after test outputs are complete."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("test_evaluated"):
        raise RuntimeError("test set was already evaluated")
    manifest["test_evaluated"] = True
    manifest["test_evaluated_at"] = utc_now()
    manifest["test_result_directory"] = str(Path(result_directory).resolve())
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, manifest)
    temporary.replace(path)
