#!/usr/bin/env python3
"""Run, compact, validate, and manifest one official FuXi-S2S forecast."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "clean/config/fuxi_operational_2020_2025.json"
EXPECTED_CHANNELS = ("tp", "t2m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-index", type=int)
    selection.add_argument("--date", help="initialization date YYYYMMDD")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--verify-large-checkpoint", action="store_true")
    return parser.parse_args()


def absolute_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "calendar",
        "date_count",
        "domain",
        "input",
        "lead_days",
        "member_generation",
        "members",
        "model",
        "run_label",
        "runtime",
        "storage_root",
        "years",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    return config


def load_dates(config: dict[str, Any]) -> list[pd.Timestamp]:
    calendar = absolute_repo_path(config["calendar"])
    allowed_years = {int(year) for year in config["years"]}
    dates = []
    with calendar.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["year"]) in allowed_years:
                dates.append(pd.Timestamp(row["init_date"]))
    if len(dates) != int(config["date_count"]):
        raise ValueError(
            f"expected {config['date_count']} calendar dates, found {len(dates)}"
        )
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError("calendar dates must be sorted and unique")
    return dates


def select_date(args: argparse.Namespace, dates: list[pd.Timestamp]) -> pd.Timestamp:
    if args.task_index is not None:
        if not 0 <= args.task_index < len(dates):
            raise ValueError(f"task index must be in 0..{len(dates) - 1}")
        return dates[args.task_index]
    selected = pd.Timestamp(args.date)
    if selected not in dates:
        raise ValueError(f"{args.date} is not in the frozen physics calendar")
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_runtime_assets(
    config: dict[str, Any], verify_large_checkpoint: bool = False
) -> dict[str, Any]:
    model = config["model"]
    inference = absolute_repo_path(model["inference_script"])
    if inference.name != "inference.py":
        raise ValueError(
            f"only the official inference.py is allowed, found {inference.name}"
        )

    checks = (
        ("inference", inference),
        ("onnx", absolute_repo_path(model["onnx"])),
        ("mask", absolute_repo_path(model["mask"])),
    )
    verified: dict[str, Any] = {}
    for label, path in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        expected_size = int(model[f"{label}_size_bytes"])
        if actual_size != expected_size:
            raise ValueError(
                f"{label} size mismatch: expected {expected_size}, found {actual_size}"
            )
        actual_hash = sha256(path)
        expected_hash = model[f"{label}_sha256"]
        if actual_hash != expected_hash:
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected_hash}, found {actual_hash}"
            )
        verified[label] = {
            "path": str(path),
            "size_bytes": actual_size,
            "sha256": actual_hash,
        }

    external = absolute_repo_path(model["external_data"])
    if not external.is_file():
        raise FileNotFoundError(external)
    external_size = external.stat().st_size
    expected_external_size = int(model["external_data_size_bytes"])
    if external_size != expected_external_size:
        raise ValueError(
            f"external checkpoint size mismatch: expected {expected_external_size}, found {external_size}"
        )
    external_record = {
        "path": str(external),
        "size_bytes": external_size,
    }
    if verify_large_checkpoint:
        external_hash = sha256(external)
        expected_external_hash = model["external_data_sha256"]
        if external_hash != expected_external_hash:
            raise ValueError(
                "external checkpoint SHA256 mismatch: "
                f"expected {expected_external_hash}, found {external_hash}"
            )
        external_record.update(sha256=external_hash, hash_check="verified")
    else:
        external_record.update(
            sha256=model["external_data_sha256"],
            hash_check="declared_hash_size_verified",
        )
    verified["external_data"] = external_record
    return verified


def repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def paths_for(config: dict[str, Any], date: pd.Timestamp) -> dict[str, Path]:
    root = Path(config["storage_root"])
    year = f"annual{date.year}"
    ymd = date.strftime("%Y%m%d")
    return {
        "root": root,
        "input": root / "inputs" / year / ymd / "input.nc",
        "raw": root / "work" / year / ymd,
        "output": root / "forecasts" / year / f"{ymd}.nc",
        "manifest": root / "manifests" / year / f"{ymd}.json",
        "inference_log": root / "logs" / "inference" / year / f"{ymd}.log",
    }


def expected_raw_files(raw_dir: Path, members: int, steps: int):
    for member in range(members):
        for step in range(1, steps + 1):
            yield raw_dir / "member" / f"{member:02d}" / f"{step:02d}.nc"


def raw_status(raw_dir: Path, members: int, steps: int) -> tuple[bool, int]:
    present = sum(
        path.exists() and path.stat().st_size > 0
        for path in expected_raw_files(raw_dir, members, steps)
    )
    return present == members * steps, present


def expected_grid(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    domain = config["domain"]
    lat = np.linspace(
        domain["latitude_first"],
        domain["latitude_last"],
        domain["latitude_count"],
        dtype=np.float32,
    )
    lon = np.linspace(
        domain["longitude_first"],
        domain["longitude_last"],
        domain["longitude_count"],
        dtype=np.float32,
    )
    return lat, lon


def read_raw_fields(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    source = xr.open_dataarray(path)
    try:
        available = {str(value) for value in source.channel.values.tolist()}
        missing = set(EXPECTED_CHANNELS) - available
        if missing:
            raise ValueError(f"{path} missing channels {sorted(missing)}")
        selected = source.sel(
            channel=list(EXPECTED_CHANNELS),
            lat=lat,
            lon=lon,
        ).squeeze(drop=True)
        selected = selected.transpose("channel", "lat", "lon")
        values = selected.values.astype(np.float32, copy=False)
        if values.shape != (2, len(lat), len(lon)):
            raise ValueError(f"{path} produced shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains missing TP/T2M values over India")
        return values
    finally:
        source.close()


def combine_output(
    raw_dir: Path,
    output: Path,
    date: pd.Timestamp,
    config: dict[str, Any],
) -> None:
    members = int(config["members"])
    steps = int(config["lead_days"])
    lat, lon = expected_grid(config)
    values = np.empty((members, steps, 2, len(lat), len(lon)), dtype=np.float32)
    for member in range(members):
        for step in range(1, steps + 1):
            path = raw_dir / "member" / f"{member:02d}" / f"{step:02d}.nc"
            values[member, step - 1] = read_raw_fields(path, lat, lon)

    lead_day = np.arange(1, steps + 1, dtype=np.int16)
    valid_time = (date.to_datetime64() + lead_day.astype("timedelta64[D]")).astype(
        "datetime64[ns]"
    )
    dataset = xr.Dataset(
        data_vars={
            "tp": (("member", "lead_day", "latitude", "longitude"), values[:, :, 0]),
            "t2m": (("member", "lead_day", "latitude", "longitude"), values[:, :, 1]),
        },
        coords={
            "member": np.arange(members, dtype=np.int16),
            "lead_day": lead_day,
            "latitude": lat,
            "longitude": lon,
            "valid_time": ("lead_day", valid_time),
            "init_time": date.to_datetime64(),
        },
        attrs={
            "model": "FuXi-S2S",
            "run_label": config["run_label"],
            "init_date": date.strftime("%Y-%m-%d"),
            "input_source": config["input"]["source"],
            "ensemble": "50 repeated calls to the official stochastic ONNX model; no separate control member",
            "member_generation": config["member_generation"],
            "input_daily_statistic": config["input"]["daily_statistic"],
            "input_hourly_sampling": config["input"]["hourly_sampling"],
            "input_time_zone": config["input"]["time_zone"],
            "domain": "exact physics grid: 39-0 N, 60-99 E, 1.5 degrees",
            "forecast_time_statistic": "global daily mean at daily resolution",
            "excluded_entry_point": "inference_ensemble.py is not used",
            "model_onnx_sha256": config["model"]["onnx_sha256"],
            "model_external_data_sha256": config["model"]["external_data_sha256"],
        },
    )
    dataset["tp"].attrs.update(
        {
            "long_name": "FuXi-S2S total precipitation mean rate",
            "units": "mm h-1",
            "cell_methods": "time: mean (24-hour period)",
            "comparison_conversion": "multiply by 24 for mm day-1",
        }
    )
    dataset["t2m"].attrs.update(
        {
            "long_name": "2 metre temperature",
            "units": "K",
            "cell_methods": "time: mean (24-hour period)",
        }
    )
    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (1, 7, len(lat), len(lon)),
        }
        for name in EXPECTED_CHANNELS
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary, encoding=encoding)
    validate_output(temporary, date, config)
    temporary.replace(output)


def validate_output(
    path: Path, date: pd.Timestamp, config: dict[str, Any]
) -> dict[str, Any]:
    dataset = xr.open_dataset(path)
    try:
        members = int(config["members"])
        steps = int(config["lead_days"])
        expected_sizes = {
            "member": members,
            "lead_day": steps,
            "latitude": int(config["domain"]["latitude_count"]),
            "longitude": int(config["domain"]["longitude_count"]),
        }
        for dimension, size in expected_sizes.items():
            if dataset.sizes.get(dimension) != size:
                raise ValueError(
                    f"{dimension}: expected {size}, found {dataset.sizes.get(dimension)}"
                )
        if set(dataset.data_vars) != set(EXPECTED_CHANNELS):
            raise ValueError(f"expected TP/T2M only, found {sorted(dataset.data_vars)}")
        lat, lon = expected_grid(config)
        if not np.allclose(dataset.latitude.values, lat) or not np.allclose(
            dataset.longitude.values, lon
        ):
            raise ValueError("final grid does not match the physics-model grid")
        if pd.Timestamp(dataset.init_time.values) != date:
            raise ValueError(f"wrong init time: {dataset.init_time.values}")
        expected_leads = np.arange(1, steps + 1, dtype=np.int16)
        if not np.array_equal(dataset.lead_day.values, expected_leads):
            raise ValueError(f"wrong lead-day coordinate: {dataset.lead_day.values}")
        expected_valid_time = (
            date.to_datetime64() + expected_leads.astype("timedelta64[D]")
        ).astype("datetime64[ns]")
        if not np.array_equal(
            dataset.valid_time.values.astype("datetime64[ns]"), expected_valid_time
        ):
            raise ValueError("valid-time coordinate does not match init + lead day")
        if (
            dataset.attrs.get("forecast_time_statistic")
            != "global daily mean at daily resolution"
        ):
            raise ValueError(
                "output does not declare the FuXi daily-mean forecast statistic"
            )
        stats: dict[str, Any] = {}
        for name in EXPECTED_CHANNELS:
            values = dataset[name].values
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains missing or infinite values")
            stats[name] = {
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        if stats["tp"]["minimum"] < -1e-6 or stats["tp"]["maximum"] > 100.0:
            raise ValueError(f"implausible FuXi TP range: {stats['tp']}")
        if stats["t2m"]["minimum"] < 180.0 or stats["t2m"]["maximum"] > 350.0:
            raise ValueError(f"implausible FuXi T2M range: {stats['t2m']}")
        spread = max(
            float(
                abs(dataset[name].isel(member=0) - dataset[name].isel(member=1)).max()
            )
            for name in EXPECTED_CHANNELS
        )
        if spread <= 1e-6:
            raise ValueError(
                "FuXi members 0 and 1 are identical; stochastic ensemble QC failed"
            )
        return {
            "size_bytes": path.stat().st_size,
            "member_0_1_max_difference": spread,
            "fields": stats,
        }
    finally:
        dataset.close()


def quarantine(path: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.invalid.{timestamp}")
    path.replace(destination)
    return destination


def write_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def clean_raw_output(path: Path, keep_raw: bool) -> str:
    if not path.exists():
        return "not_present"
    if keep_raw:
        return "retained_by_request"
    try:
        shutil.rmtree(path)
        return "deleted_after_qc"
    except OSError as exc:
        return f"cleanup_failed: {type(exc).__name__}: {exc}"


def base_record(
    config: dict[str, Any],
    config_path: Path,
    date: pd.Timestamp,
    paths: dict[str, Path],
    runtime_assets: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": "FuXi-S2S",
        "run_label": config["run_label"],
        "init_date": date.strftime("%Y-%m-%d"),
        "members": config["members"],
        "member_generation": config["member_generation"],
        "lead_days": config["lead_days"],
        "variables": list(EXPECTED_CHANNELS),
        "calendar": config["calendar"],
        "input_contract": config["input"],
        "input": str(paths["input"]),
        "output": str(paths["output"]),
        "inference_log": str(paths["inference_log"]),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "model_artifacts": config["model"],
        "runtime_asset_checks": runtime_assets,
        "pipeline_git_commit": repository_commit(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get(
            "SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID")
        ),
        "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    runtime_assets = verify_runtime_assets(config, args.verify_large_checkpoint)
    dates = load_dates(config)
    date = select_date(args, dates)
    paths = paths_for(config, date)
    record = base_record(config, args.config, date, paths, runtime_assets)
    summary = {
        "task_index": dates.index(date),
        "date": date.strftime("%Y%m%d"),
        "members": config["members"],
        "lead_days": config["lead_days"],
        "input": str(paths["input"]),
        "output": str(paths["output"]),
        "manifest": str(paths["manifest"]),
        "inference_log": str(paths["inference_log"]),
        "member_generation": config["member_generation"],
        "large_checkpoint_hash_verified": args.verify_large_checkpoint,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return 0

    try:
        if paths["output"].exists():
            try:
                qc = validate_output(paths["output"], date, config)
            except Exception as exc:  # noqa: BLE001
                moved = quarantine(paths["output"])
                print(f"quarantined invalid forecast at {moved}: {exc}", flush=True)
            else:
                record.update(
                    status="existing_valid",
                    qc=qc,
                    output_sha256=sha256(paths["output"]),
                    raw_cleanup=clean_raw_output(paths["raw"], args.keep_raw),
                    timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                )
                if paths["input"].exists():
                    record["input_sha256"] = sha256(paths["input"])
                write_manifest(paths["manifest"], record)
                print(f"existing forecast valid: {paths['output']}", flush=True)
                return 0

        input_builder = Path(__file__).with_name("prepare_fuxi_input.py")
        input_command = [
            config["runtime"]["input_python"],
            str(input_builder),
            "--date",
            date.strftime("%Y%m%d"),
            "--output",
            str(paths["input"]),
            "--config",
            str(args.config),
        ]
        subprocess.run(input_command, check=True)

        members = int(config["members"])
        steps = int(config["lead_days"])
        complete, present = raw_status(paths["raw"], members, steps)
        inference_returncode: int | None = None
        if not complete:
            if paths["raw"].exists():
                shutil.rmtree(paths["raw"])
            paths["raw"].mkdir(parents=True, exist_ok=True)
            model_root = absolute_repo_path(config["model"]["repository"])
            inference_command = [
                config["runtime"]["inference_python"],
                "-u",
                str(absolute_repo_path(config["model"]["inference_script"])),
                "--model",
                str(absolute_repo_path(config["model"]["onnx"])),
                "--input",
                str(paths["input"]),
                "--device",
                "cuda",
                "--total_step",
                str(steps),
                "--total_member",
                str(members),
                "--save_dir",
                str(paths["raw"]),
            ]
            paths["inference_log"].parent.mkdir(parents=True, exist_ok=True)
            print(f"official inference log: {paths['inference_log']}", flush=True)
            with paths["inference_log"].open("w", encoding="utf-8") as log_handle:
                log_handle.write(json.dumps({"command": inference_command}) + "\n")
                log_handle.flush()
                inference = subprocess.run(
                    inference_command,
                    cwd=model_root,
                    check=False,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            inference_returncode = inference.returncode
            complete, present = raw_status(paths["raw"], members, steps)
        if not complete:
            raise RuntimeError(
                f"inference raw output incomplete: {present}/{members * steps}"
            )

        combine_output(paths["raw"], paths["output"], date, config)
        qc = validate_output(paths["output"], date, config)
        raw_cleanup = clean_raw_output(paths["raw"], args.keep_raw)
        record.update(
            status="generated_valid",
            input_sha256=sha256(paths["input"]),
            output_sha256=sha256(paths["output"]),
            inference_returncode=inference_returncode,
            raw_files=members * steps,
            raw_cleanup=raw_cleanup,
            qc=qc,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        write_manifest(paths["manifest"], record)
        print(f"generated valid forecast: {paths['output']}", flush=True)
        return 0
    except Exception as exc:
        record.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        write_manifest(paths["manifest"], record)
        raise


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
