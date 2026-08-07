#!/usr/bin/env python3
"""Create, probe, stage, submit, publish, and audit operational FuXi cases.

The command keeps GFS-proxy and ERA5-reference runs scientifically separate,
while giving both sources the same 100-member, 42-day inference contract.
Generated case files are lightweight and date-specific; heavy inputs and model
outputs remain below the configured storage root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(
    os.environ.get("S2S_REPO_ROOT", Path(__file__).resolve().parents[4])
).resolve()
CLEAN_ROOT = REPO_ROOT / "clean"
SCRIPT_ROOT = CLEAN_ROOT / "model-runs/fuxi/scripts"
SLURM_ROOT = CLEAN_ROOT / "model-runs/fuxi/slurm"
CONFIG_ROOT = CLEAN_ROOT / "config/operational"
DASHBOARD_ROOT = Path(
    os.environ.get("S2S_DASHBOARD_ROOT", CLEAN_ROOT / "fuxi-dashboard")
).resolve()
STORAGE_ROOT = Path(
    os.environ.get(
        "S2S_FUXI_STORAGE_ROOT",
        "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/operational",
    )
)
DRIVER_PYTHON = os.environ.get(
    "S2S_DRIVER_PYTHON", "/home/raj.ayush/.conda/envs/s2s-hind/bin/python"
)
GFS_TEMPLATE = Path(
    os.environ.get(
        "S2S_FUXI_GFS_TEMPLATE",
        CLEAN_ROOT / "config/fuxi_gfs_case_20260730_31_ens100.json",
    )
)
ERA5_TEMPLATE = Path(
    os.environ.get(
        "S2S_FUXI_ERA5_TEMPLATE",
        CLEAN_ROOT / "config/fuxi_strict00z_case_20260722_ens100.json",
    )
)
FU_XI_CLIMATOLOGY = Path(
    os.environ.get(
        "S2S_FUXI_CLIMATOLOGY",
        "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
        "native_reforecast_jjas_2002_2021/"
        "fuxi_s2s_jjas_model_climatology_2002_2021_loyo.nc",
    )
)
PRODUCTION_MEMBERS = 100
PRODUCTION_LEADS = 42
GFS_ARCHIVE_URL = (
    "https://storage.googleapis.com/global-forecast-system/"
    "gfs.{date}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.1p00.f{forecast_hour}"
)


def parse_date(value: str) -> pd.Timestamp:
    normalized = value.replace("-", "")
    timestamp = pd.Timestamp(value)
    if timestamp.strftime("%Y%m%d") != normalized:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD or YYYY-MM-DD")
    return timestamp.normalize()


def ymd(date: pd.Timestamp) -> str:
    return date.strftime("%Y%m%d")


def input_days(date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    return date - pd.Timedelta(days=2), date - pd.Timedelta(days=1)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generated_paths(
    source: str, date: pd.Timestamp, members: int = PRODUCTION_MEMBERS
) -> dict[str, Path]:
    stem = f"fuxi_{source}_{ymd(date)}_ens{members}"
    return {
        "config": CONFIG_ROOT / f"{stem}.json",
        "calendar": CONFIG_ROOT / f"{stem}.csv",
    }


def run_root(
    source: str, date: pd.Timestamp, members: int = PRODUCTION_MEMBERS
) -> Path:
    return STORAGE_ROOT / source / ymd(date) / f"ens{members}"


def forecast_paths(config: dict[str, Any], date: pd.Timestamp) -> dict[str, Path]:
    root = Path(config["storage_root"])
    annual = f"annual{date.year}"
    issue = ymd(date)
    return {
        "input": root / "inputs" / annual / issue / "input.nc",
        "raw": root / "work" / annual / issue,
        "forecast": root / "forecasts" / annual / f"{issue}.nc",
        "manifest": root / "manifests" / annual / f"{issue}.json",
        "log": root / "logs" / "inference" / annual / f"{issue}.log",
    }


def create_calendar(path: Path, source: str, date: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if source == "gfs":
            fields = (
                "year",
                "init_date",
                "provider",
                "init_offset_days",
                "common_end_day",
                "selection",
                "source",
            )
            row = {
                "year": date.year,
                "init_date": date.strftime("%Y-%m-%d"),
                "provider": "gfs",
                "init_offset_days": 0,
                "common_end_day": PRODUCTION_LEADS,
                "selection": "operational_gfs_proxy",
                "source": "gfs_analysis_short_forecast_daily_mean",
            }
        else:
            fields = (
                "year",
                "init_date",
                "core_providers",
                "cnrm_available",
                "lead_days",
                "date_policy",
                "source",
            )
            row = {
                "year": date.year,
                "init_date": date.strftime("%Y-%m-%d"),
                "core_providers": "era5",
                "cnrm_available": 0,
                "lead_days": PRODUCTION_LEADS,
                "date_policy": "single_case_strict_daily_mean",
                "source": "arco_hourly_daily_mean",
            }
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def create_case(
    source: str,
    date: pd.Timestamp,
    *,
    members: int = PRODUCTION_MEMBERS,
    era5_probe: dict[str, Any] | None = None,
) -> Path:
    if members not in {5, PRODUCTION_MEMBERS}:
        raise ValueError("operational cases support only 5 or 100 members")
    paths = generated_paths(source, date, members)
    template = GFS_TEMPLATE if source == "gfs" else ERA5_TEMPLATE
    config = json.loads(template.read_text(encoding="utf-8"))
    first_day, second_day = input_days(date)
    storage = run_root(source, date, members)

    config.update(
        schema_version=4,
        run_label=f"fuxi_s2s_{source}_case_{ymd(date)}_ens{members}",
        calendar=str(paths["calendar"].relative_to(REPO_ROOT)),
        years=[date.year],
        date_count=1,
        cadence=f"single_case_{source}_operational_framework",
        members=members,
        lead_days=PRODUCTION_LEADS,
        storage_root=str(storage),
    )
    config["temporal_alignment"].update(
        issue_hour_utc=0,
        input_day_offsets=[-2, -1],
        first_forecast_period_start_offset_days=0,
        valid_time_role="period_end",
    )
    if source == "gfs":
        config["temporal_alignment"].update(
            benchmark_mode="experimental_gfs_initialized_00utc",
            strict_operational=True,
            contract=(
                f"For the {date:%Y-%m-%d} 00 UTC issue, use complete GFS "
                f"daily proxy means for {first_day:%Y-%m-%d} and "
                f"{second_day:%Y-%m-%d}. State fields use four analyses per "
                "day; TP/TTR use four successive 0-6 h forecasts."
            ),
        )
        config["input"].update(
            builder="gfs_daily_proxy",
            source=(
                "NCEP operational GFS 1.0-degree analyses and 0-6 hour "
                f"forecasts for {first_day:%Y-%m-%d} and {second_day:%Y-%m-%d}; "
                "experimental proxy for ERA5-initialized FuXi-S2S"
            ),
        )
        config["input"]["gfs"]["raw_root"] = str(storage / "gfs-raw")
        config["input"]["gfs"]["url_template"] = GFS_ARCHIVE_URL
    else:
        if not era5_probe or not era5_probe.get("available"):
            raise ValueError("ERA5 case creation requires a successful chunk probe")
        config["temporal_alignment"].update(
            benchmark_mode="strict_information_matched_00utc",
            strict_operational=True,
            contract=(
                f"For the {date:%Y-%m-%d} 00 UTC issue, use complete hourly "
                f"ERA5 daily means for {first_day:%Y-%m-%d} and "
                f"{second_day:%Y-%m-%d}; lead 1 covers the following 24 hours."
            ),
        )
        config["input"].pop("builder", None)
        config["input"]["source"] = (
            "Public hourly ARCO ERA5 v3 reduced to complete UTC daily means "
            f"for {first_day:%Y-%m-%d} and {second_day:%Y-%m-%d}"
        )
        remote = config["input"]["remote_hourly_arco"]
        remote["years"] = [date.year]
        remote["coverage_override_stop"] = date.strftime("%Y-%m-%dT00:00:00")
        remote["coverage_override_evidence"] = (
            "operational framework verified every required ARCO object before "
            f"submission: {era5_probe['present_objects']}/"
            f"{era5_probe['required_objects']} chunks; probe time "
            f"{era5_probe['checked_at']}"
        )
    create_calendar(paths["calendar"], source, date)
    json_write(paths["config"], config)
    return paths["config"]


def gfs_records(date: pd.Timestamp, config: dict[str, Any] | None = None) -> list[str]:
    if config is None:
        config = json.loads(GFS_TEMPLATE.read_text(encoding="utf-8"))
        config["input"]["gfs"]["url_template"] = GFS_ARCHIVE_URL
    template = str(config["input"]["gfs"]["url_template"])
    records = []
    for day in input_days(date):
        for cycle in (0, 6, 12, 18):
            for forecast_hour in (0, 6):
                records.append(
                    template.format(
                        date=day.strftime("%Y%m%d"),
                        cycle=f"{cycle:02d}",
                        forecast_hour=f"{forecast_hour:03d}",
                    )
                )
    return records


def head_url(url: str) -> tuple[str, bool, int | None, str | None]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "FuXi-S2S-operational-preflight/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            size = response.headers.get("Content-Length")
            return url, response.status == 200, int(size) if size else None, None
    except (OSError, urllib.error.URLError) as exc:
        return url, False, None, f"{type(exc).__name__}: {exc}"


def probe_gfs(date: pd.Timestamp, workers: int = 8) -> dict[str, Any]:
    urls = gfs_records(date)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(head_url, urls))
    missing = [url for url, ok, _, _ in results if not ok]
    return {
        "source": "gfs",
        "issue_date": date.strftime("%Y-%m-%d"),
        "available": not missing,
        "required_files": len(urls),
        "present_files": len(urls) - len(missing),
        "missing": missing,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _arco_required_objects(date: pd.Timestamp) -> tuple[Any, list[str]]:
    sys.path.insert(0, str(SCRIPT_ROOT))
    import arco_hourly  # noqa: PLC0415

    dataset = arco_hourly.open_arco()
    timestamps: list[tuple[str, bool, pd.Timestamp]] = []
    days = input_days(date)
    for source_name in arco_hourly.PRESSURE_FIELDS.values():
        for day in days:
            for hour in range(24):
                timestamps.append((source_name, True, day + pd.Timedelta(hours=hour)))
    for short_name, source_name in arco_hourly.SURFACE_FIELDS.items():
        offset = 1 if short_name in {"ttr", "tp"} else 0
        for day in days:
            for hour in range(24):
                timestamps.append(
                    (source_name, False, day + pd.Timedelta(hours=hour + offset))
                )
    objects = []
    for source_name, pressure, timestamp in timestamps:
        index = dataset._time_index(timestamp)  # exact object mapping used by reader
        key = f"{index}.0.0.0" if pressure else f"{index}.0.0"
        objects.append(f"{dataset.root}/{source_name}/{key}")
    return dataset, objects


def probe_era5(date: pd.Timestamp, workers: int = 32) -> dict[str, Any]:
    dataset, objects = _arco_required_objects(date)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            present = list(executor.map(dataset.fs.exists, objects))
    finally:
        dataset.close()
    missing = [path for path, exists in zip(objects, present, strict=True) if not exists]
    return {
        "source": "era5",
        "issue_date": date.strftime("%Y-%m-%d"),
        "available": not missing,
        "required_objects": len(objects),
        "present_objects": len(objects) - len(missing),
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "stable_metadata_stop": str(dataset.valid_time_stop),
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_source(config: dict[str, Any]) -> str:
    return "gfs" if config["input"].get("builder") == "gfs_daily_proxy" else "era5"


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    print(json.dumps({"command": command, "cwd": str(cwd) if cwd else None}), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def stage_case(config_path: Path, date: pd.Timestamp) -> None:
    config = load_config(config_path)
    paths = forecast_paths(config, date)
    source = config_source(config)
    if source == "gfs":
        run_command(
            [
                config["runtime"]["input_python"],
                str(SCRIPT_ROOT / "prepare_fuxi_gfs_input.py"),
                "--date",
                ymd(date),
                "--output",
                str(paths["input"]),
                "--config",
                str(config_path),
            ]
        )
    else:
        for year, month in sorted({(day.year, day.month) for day in input_days(date)}):
            run_command(
                [
                    config["runtime"]["input_python"],
                    str(SCRIPT_ROOT / "stage_arco_hourly_daily.py"),
                    "--year",
                    str(year),
                    "--month",
                    str(month),
                    "--config",
                    str(config_path),
                ]
            )
        run_command(
            [
                config["runtime"]["input_python"],
                str(SCRIPT_ROOT / "prepare_fuxi_input.py"),
                "--date",
                ymd(date),
                "--output",
                str(paths["input"]),
                "--config",
                str(config_path),
            ]
        )


def issue_record(
    source: str,
    date: pd.Timestamp,
    members: int,
    regional_available: bool | None = None,
) -> dict[str, Any]:
    record = {
        "id": ymd(date),
        "initialization": date.strftime("%Y-%m-%dT00:00:00Z"),
        "members": members,
        "status": "warning",
        "role": (
            "rapid_prototype"
            if members < PRODUCTION_MEMBERS
            else ("operational_experimental" if source == "gfs" else "reference")
        ),
        "forecast": f"forecasts/{source}/{ymd(date)}.json",
    }
    if regional_available is None:
        # Compatibility for callers that build a record without an exported
        # forecast. Operational publication always passes the capability
        # derived from that forecast explicitly.
        regional_available = members == PRODUCTION_MEMBERS
    if regional_available:
        record["regional_outlook"] = f"regional/{source}/{ymd(date)}.json"
    return record


def update_dashboard_index(
    entries: Iterable[tuple[str, pd.Timestamp, int, bool]],
) -> None:
    path = DASHBOARD_ROOT / "public/data/index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    by_source = {item["id"]: item for item in index["initial_condition_sources"]}
    for source, date, members, regional_available in entries:
        record = issue_record(source, date, members, regional_available)
        catalog = by_source[source]
        catalog["issues"] = [
            item for item in catalog["issues"] if item["id"] != record["id"]
        ] + [record]
        catalog["issues"].sort(key=lambda item: item["id"], reverse=True)
    for catalog in by_source.values():
        production_issues = [
            item
            for item in catalog["issues"]
            if int(item.get("members", 0)) == PRODUCTION_MEMBERS
        ]
        if production_issues:
            catalog["default_issue"] = max(
                production_issues, key=lambda item: item["id"]
            )["id"]
    gfs_issues = by_source["gfs"]["issues"]
    production_gfs_issues = [
        item
        for item in gfs_issues
        if int(item.get("members", 0)) == PRODUCTION_MEMBERS
    ]
    if production_gfs_issues:
        index["latest_successful_issue"] = max(
            item["id"] for item in production_gfs_issues
        )
        index["available_issues"] = [
            {
                "id": item["id"],
                "initialization": item["initialization"],
                "status": item["status"],
                "forecast": item["forecast"],
            }
            for item in gfs_issues
        ]
    index["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    json_write(path, index)


def publish_cases(config_paths: list[Path], date: pd.Timestamp) -> None:
    entries: list[tuple[str, pd.Timestamp, int, bool]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        source = config_source(config)
        paths = forecast_paths(config, date)
        if not paths["forecast"].is_file() or not paths["manifest"].is_file():
            raise FileNotFoundError(f"completed {source} forecast is missing")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("status") not in {"generated_valid", "existing_valid"}:
            raise ValueError(f"{source} manifest is not publishable: {manifest.get('status')}")
        if manifest.get("output_sha256") != sha256(paths["forecast"]):
            raise ValueError(f"{source} manifest checksum does not match forecast")
        run_command(
            [
                DRIVER_PYTHON,
                str(DASHBOARD_ROOT / "scripts/build_prototype_data.py"),
                "--forecast",
                str(paths["forecast"]),
                "--fuxi-climatology",
                str(FU_XI_CLIMATOLOGY),
                "--source-id",
                source,
                "--run-manifest",
                str(paths["manifest"]),
                "--forecast-only",
            ],
            cwd=DASHBOARD_ROOT,
        )
        members = int(config["members"])
        public_forecast = (
            DASHBOARD_ROOT
            / "public/data/forecasts"
            / source
            / f"{ymd(date)}.json"
        )
        public_payload = json.loads(public_forecast.read_text(encoding="utf-8"))
        regional_available = bool(
            members == PRODUCTION_MEMBERS
            and public_payload.get("issue", {})
            .get("capabilities", {})
            .get("regional_probabilities_eligible", False)
        )
        if regional_available:
            run_command(
                [
                    DRIVER_PYTHON,
                    str(DASHBOARD_ROOT / "scripts/build_regional_outlook.py"),
                    "--forecast",
                    str(paths["forecast"]),
                    "--public-forecast",
                    str(public_forecast),
                    "--climatology",
                    str(FU_XI_CLIMATOLOGY),
                ],
                cwd=DASHBOARD_ROOT,
            )
        entries.append((source, date, members, regional_available))
    update_dashboard_index(entries)
    for source, issue_date, _, _ in entries:
        run_command(
            [
                DRIVER_PYTHON,
                str(DASHBOARD_ROOT / "scripts/build_forecast_pdfs.py"),
                "--source",
                source,
                "--issue",
                ymd(issue_date),
            ],
            cwd=DASHBOARD_ROOT,
        )
    matched_forecasts = [
        DASHBOARD_ROOT
        / "public/data/forecasts"
        / source
        / f"{ymd(date)}.json"
        for source in ("gfs", "era5")
    ]
    if all(path.is_file() for path in matched_forecasts):
        run_command(
            [
                DRIVER_PYTHON,
                str(DASHBOARD_ROOT / "scripts/build_initialization_comparison.py"),
                "--date",
                ymd(date),
            ],
            cwd=DASHBOARD_ROOT,
        )
    run_command(
        [DRIVER_PYTHON, str(DASHBOARD_ROOT / "scripts/stamp_deploy.py")],
        cwd=DASHBOARD_ROOT,
    )


def submit_job(
    script: Path,
    exports: dict[str, str],
    dependency: str | None = None,
    *,
    dependency_mode: str = "afterok",
    array: str | None = None,
) -> str:
    export_value = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    command = ["sbatch", "--parsable", f"--export={export_value}"]
    if dependency:
        command.append(f"--dependency={dependency_mode}:{dependency}")
    if array:
        command.append(f"--array={array}")
    command.append(str(script))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().split(";", maxsplit=1)[0]


def submit_cases(config_paths: list[Path], date: pd.Timestamp) -> dict[str, Any]:
    jobs: dict[str, Any] = {"issue_date": date.strftime("%Y-%m-%d"), "sources": {}}
    inference_jobs: list[str] = []
    for path in config_paths:
        config = load_config(path)
        source = config_source(config)
        exports = {
            "FUXI_CONFIG": str(path.resolve()),
            "FUXI_DATE": ymd(date),
            "FUXI_SOURCE": source,
        }
        stage_job = submit_job(SLURM_ROOT / "stage_fuxi_operational.sbatch", exports)
        inference_job = submit_job(
            SLURM_ROOT / "run_fuxi_operational.sbatch",
            exports,
            dependency=stage_job,
        )
        jobs["sources"][source] = {
            "config": str(path),
            "stage_job": stage_job,
            "inference_job": inference_job,
        }
        inference_jobs.append(inference_job)
    publish_exports = {
        "FUXI_CONFIGS": ":".join(str(path.resolve()) for path in config_paths),
        "FUXI_DATE": ymd(date),
    }
    publish_job = submit_job(
        SLURM_ROOT / "publish_fuxi_operational.sbatch",
        publish_exports,
        dependency=":".join(inference_jobs),
    )
    jobs["publish_job"] = publish_job
    state_path = STORAGE_ROOT / "submissions" / f"{ymd(date)}.json"
    json_write(state_path, jobs)
    jobs["submission_record"] = str(state_path)
    return jobs


def config_date(config: dict[str, Any]) -> pd.Timestamp:
    calendar = Path(config["calendar"])
    if not calendar.is_absolute():
        calendar = REPO_ROOT / calendar
    with calendar.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"operational case calendar must contain one row: {calendar}")
    return pd.Timestamp(rows[0]["init_date"])


def infer_case(config_path: Path, date: pd.Timestamp) -> None:
    config = load_config(config_path)
    run_command(
        [
            config["runtime"]["driver_python"],
            str(SCRIPT_ROOT / "run_fuxi_forecast.py"),
            "--date",
            ymd(date),
            "--config",
            str(config_path),
            "--verify-large-checkpoint",
        ]
    )


def load_batch(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        raise ValueError(f"batch has no items: {path}")
    for item in items:
        config_path = Path(item["config"])
        config = load_config(config_path)
        date = config_date(config)
        if item["date"] != ymd(date):
            raise ValueError(f"batch date does not match config: {config_path}")
        if item["source"] != config_source(config):
            raise ValueError(f"batch source does not match config: {config_path}")
        if int(item["members"]) != int(config["members"]):
            raise ValueError(f"batch members do not match config: {config_path}")
    return payload


def array_item(batch_path: Path, index: int, phase: str) -> None:
    batch = load_batch(batch_path)
    if not 0 <= index < len(batch["items"]):
        raise IndexError(f"array index {index} outside batch size {len(batch['items'])}")
    item = batch["items"][index]
    config_path = Path(item["config"])
    date = pd.Timestamp(item["date"])
    print(json.dumps({"array_index": index, "phase": phase, **item}, indent=2), flush=True)
    if phase == "stage":
        stage_case(config_path, date)
    elif phase == "infer":
        infer_case(config_path, date)
    else:
        raise ValueError(f"unsupported array phase {phase!r}")


def publish_batch(batch_path: Path) -> None:
    batch = load_batch(batch_path)
    grouped: dict[str, list[Path]] = {}
    for item in batch["items"]:
        grouped.setdefault(item["date"], []).append(Path(item["config"]))
    for date_text, configs in sorted(grouped.items()):
        publish_cases(configs, pd.Timestamp(date_text))


def submit_array(config_paths: list[Path], concurrency: int = 2) -> dict[str, Any]:
    items = []
    for path in config_paths:
        resolved = path.resolve()
        config = load_config(resolved)
        date = config_date(config)
        items.append(
            {
                "config": str(resolved),
                "date": ymd(date),
                "source": config_source(config),
                "members": int(config["members"]),
            }
        )
    items.sort(key=lambda item: (item["date"], item["source"]))
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    batch_path = STORAGE_ROOT / "submissions" / f"batch_{digest}.json"
    json_write(
        batch_path,
        {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "items": items,
        },
    )
    exports = {"FUXI_BATCH": str(batch_path)}
    array_range = f"0-{len(items) - 1}%{concurrency}"
    stage_job = submit_job(
        SLURM_ROOT / "stage_fuxi_operational_array.sbatch",
        exports,
        array=array_range,
    )
    inference_job = submit_job(
        SLURM_ROOT / "run_fuxi_operational_array.sbatch",
        exports,
        dependency=stage_job,
        dependency_mode="aftercorr",
        array=array_range,
    )
    publish_job = submit_job(
        SLURM_ROOT / "publish_fuxi_operational_batch.sbatch",
        exports,
        dependency=inference_job,
    )
    result = {
        "batch": str(batch_path),
        "tasks": len(items),
        "stage_array_job": stage_job,
        "inference_array_job": inference_job,
        "publish_job": publish_job,
        "items": items,
    }
    json_write(batch_path.with_suffix(".submission.json"), result)
    return result


def audit_case(config_path: Path, date: pd.Timestamp) -> dict[str, Any]:
    config = load_config(config_path)
    paths = forecast_paths(config, date)
    result: dict[str, Any] = {
        "source": config_source(config),
        "date": ymd(date),
        "config": str(config_path),
        "input_exists": paths["input"].is_file(),
        "forecast_exists": paths["forecast"].is_file(),
        "manifest_exists": paths["manifest"].is_file(),
    }
    if paths["manifest"].is_file():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        result.update(
            status=manifest.get("status"),
            members=manifest.get("members"),
            lead_days=manifest.get("lead_days"),
            qc=manifest.get("qc"),
            error=manifest.get("error"),
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="Check source availability")
    probe.add_argument("--source", choices=("gfs", "era5", "both"), required=True)
    probe.add_argument("--date", type=parse_date, required=True)

    create = subparsers.add_parser("create", help="Write immutable case config")
    create.add_argument("--source", choices=("gfs", "era5"), required=True)
    create.add_argument("--date", type=parse_date, required=True)
    create.add_argument("--members", type=int, choices=(5, 100), default=100)
    create.add_argument("--era5-probe", type=Path)

    stage = subparsers.add_parser("stage", help="Build and validate one input")
    stage.add_argument("--config", type=Path, required=True)
    stage.add_argument("--date", type=parse_date, required=True)

    submit = subparsers.add_parser("submit", help="Submit source jobs and publication")
    submit.add_argument("--configs", type=Path, nargs="+", required=True)
    submit.add_argument("--date", type=parse_date, required=True)

    submit_array_parser = subparsers.add_parser(
        "submit-array", help="Submit mixed 5/100-member source/date job arrays"
    )
    submit_array_parser.add_argument("--configs", type=Path, nargs="+", required=True)
    submit_array_parser.add_argument("--concurrency", type=int, default=2)

    item = subparsers.add_parser("array-item", help="Run one Slurm array item")
    item.add_argument("--batch", type=Path, required=True)
    item.add_argument("--index", type=int, required=True)
    item.add_argument("--phase", choices=("stage", "infer"), required=True)

    publish_batch_parser = subparsers.add_parser(
        "publish-batch", help="Publish every completed case in a batch"
    )
    publish_batch_parser.add_argument("--batch", type=Path, required=True)

    publish = subparsers.add_parser("publish", help="Publish completed cases")
    publish.add_argument("--configs", type=Path, nargs="+", required=True)
    publish.add_argument("--date", type=parse_date, required=True)

    audit = subparsers.add_parser("audit", help="Report one case status")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--date", type=parse_date, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "probe":
        sources = ("gfs", "era5") if args.source == "both" else (args.source,)
        records = [
            probe_gfs(args.date) if source == "gfs" else probe_era5(args.date)
            for source in sources
        ]
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0 if all(record["available"] for record in records) else 2
    if args.command == "create":
        probe = None
        if args.source == "era5":
            if args.era5_probe:
                probe = json.loads(args.era5_probe.read_text(encoding="utf-8"))
            else:
                probe = probe_era5(args.date)
        path = create_case(
            args.source, args.date, members=args.members, era5_probe=probe
        )
        print(path)
        return 0
    if args.command == "stage":
        stage_case(args.config, args.date)
        return 0
    if args.command == "submit":
        print(json.dumps(submit_cases(args.configs, args.date), indent=2))
        return 0
    if args.command == "submit-array":
        if args.concurrency < 1:
            raise SystemExit("concurrency must be positive")
        print(json.dumps(submit_array(args.configs, args.concurrency), indent=2))
        return 0
    if args.command == "array-item":
        array_item(args.batch, args.index, args.phase)
        return 0
    if args.command == "publish-batch":
        publish_batch(args.batch)
        return 0
    if args.command == "publish":
        publish_cases(args.configs, args.date)
        return 0
    if args.command == "audit":
        print(json.dumps(audit_case(args.config, args.date), indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
