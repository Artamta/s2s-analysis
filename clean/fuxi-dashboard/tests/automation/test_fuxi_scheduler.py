from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from automation.fuxi_scheduler import (
    UTC,
    CommandFailure,
    CompletedCommand,
    Controller,
    Settings,
    scheduled_issue_dates,
)


def utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "config_path": tmp_path / "scheduler.json",
        "operational_python": Path("/test/python"),
        "operational_script": Path("/test/fuxi_operational.py"),
        "sbatch_path": Path("/test/sbatch"),
        "sacct_path": Path("/test/sacct"),
        "slurm_root": tmp_path / "slurm",
        "submission_mode": "legacy_driver",
        "dashboard_root": tmp_path / "dashboard",
        "generated_config_root": tmp_path / "generated",
        "state_root": tmp_path / "state",
        "start_date": dt.date(2026, 8, 1),
        "gfs_window_start": dt.time(0, 15, tzinfo=UTC),
        "gfs_window_end": dt.time(12, 0, tzinfo=UTC),
        "gfs_poll_minutes": 15,
        "era5_lookback_days": 35,
        "era5_max_probes_per_run": 4,
        "era5_poll_hours": 24,
        "probe_timeout_seconds": 30,
        "create_timeout_seconds": 30,
        "submit_timeout_seconds": 30,
        "slurm_status_timeout_seconds": 30,
        "audit_timeout_seconds": 30,
        "max_safe_attempts": 3,
        "retry_base_seconds": 60,
        "retry_max_seconds": 300,
        "submission_stale_hours": 12,
        "sendmail_path": tmp_path / "sendmail",
    }
    values.update(overrides)
    return Settings(**values)


class FakeRunner:
    def __init__(self, config_root: Path) -> None:
        self.config_root = config_root
        self.commands: list[tuple[str, ...]] = []
        self.probes: dict[tuple[str, str], list[bool | Exception]] = {}
        self.submit_error: Exception | None = None
        self.audit_payload: dict[str, Any] | None = None
        self.sacct_states: dict[str, str] = {}
        self.smoke_error: Exception | None = None

    def availability(self, source: str, date: str, *values: bool | Exception) -> None:
        self.probes[(source, date)] = list(values)

    def run(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int,
        allowed_returncodes: Sequence[int] = (0,),
    ) -> CompletedCommand:
        del timeout_seconds, allowed_returncodes
        command = tuple(command)
        self.commands.append(command)
        if command[0] == "/test/publisher":
            source = command[command.index("--source") + 1]
            issue = command[command.index("--issue") + 1]
            receipt = Path(command[command.index("--receipt") + 1])
            receipt.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "pushed",
                "pushed": True,
                "source": source,
                "issue": issue,
                "commit_sha": "a" * 40,
                "forecast_sha256": "b" * 64,
            }
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            return CompletedCommand(command, 0, json.dumps(payload), "")
        if command[0] == "/test/smoke":
            if self.smoke_error:
                raise self.smoke_error
            return CompletedCommand(command, 0, '{"status":"healthy"}', "")
        if command[0] == "/test/sbatch":
            job_id = "200" if command[-1].endswith("stage_fuxi_operational.sbatch") else "201"
            return CompletedCommand(command, 0, job_id + "\n", "")
        if command[0] == "/test/sacct":
            output = "".join(
                f"{job}|{state}|\n" for job, state in self.sacct_states.items()
            )
            return CompletedCommand(command, 0, output, "")
        operation = command[2]
        if operation == "probe":
            source = command[command.index("--source") + 1]
            date = command[command.index("--date") + 1]
            queue = self.probes[(source, date)]
            value = queue.pop(0)
            if isinstance(value, Exception):
                raise value
            required_key = "required_files" if source == "gfs" else "required_objects"
            present_key = "present_files" if source == "gfs" else "present_objects"
            count = 16 if source == "gfs" else 768
            payload = [
                {
                    "source": source,
                    "issue_date": dt.datetime.strptime(date, "%Y%m%d").date().isoformat(),
                    "available": value,
                    required_key: count,
                    present_key: count if value else count - 1,
                }
            ]
            return CompletedCommand(command, 0 if value else 2, json.dumps(payload), "")
        if operation == "create":
            source = command[command.index("--source") + 1]
            date = command[command.index("--date") + 1]
            path = self.config_root / f"fuxi_{source}_{date}_ens100.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            input_contract = {"builder": "gfs_daily_proxy"} if source == "gfs" else {}
            path.write_text(
                json.dumps(
                    {
                        "members": 100,
                        "run_label": f"fuxi_s2s_{source}_case_{date}_ens100",
                        "input": input_contract,
                        "storage_root": str(self.config_root.parent / "private"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return CompletedCommand(command, 0, str(path) + "\n", "")
        if operation == "submit":
            if self.submit_error:
                raise self.submit_error
            source = Path(command[-1]).name.split("_")[1]
            return CompletedCommand(
                command,
                0,
                json.dumps(
                    {
                        "sources": {source: {"stage_job": "100", "inference_job": "101"}},
                        "publish_job": "102",
                    }
                ),
                "",
            )
        if operation == "audit":
            return CompletedCommand(
                command, 0, json.dumps(self.audit_payload or {}), ""
            )
        raise AssertionError(operation)


def timeout(command: str = "probe") -> CommandFailure:
    return CommandFailure(
        ["python", "driver", command], None, "", "deadline", timed_out=True
    )


def test_schedule_is_only_wednesday_and_saturday() -> None:
    assert scheduled_issue_dates(
        dt.date(2026, 8, 12),
        start_date=dt.date(2026, 8, 1),
        lookback_days=20,
    ) == [
        dt.date(2026, 8, 1),
        dt.date(2026, 8, 5),
        dt.date(2026, 8, 8),
        dt.date(2026, 8, 12),
    ]


def test_gfs_poll_waits_then_submits_exactly_once(tmp_path: Path) -> None:
    config = settings(tmp_path)
    runner = FakeRunner(config.generated_config_root)
    runner.availability("gfs", "20260805", False, True)
    controller = Controller(config, runner=runner)

    first = controller.gfs_poll(utc("2026-08-05T00:15:00Z"))
    second = controller.gfs_poll(utc("2026-08-05T00:30:00Z"))
    duplicate = controller.gfs_poll(utc("2026-08-05T00:45:00Z"))

    assert first[0]["outcome"] == "waiting"
    assert second[0]["stage"] == "submitted"
    assert duplicate[0]["outcome"] == "skipped"
    assert [command[2] for command in runner.commands].count("submit") == 1
    record = controller.store.load("gfs", dt.date(2026, 8, 5))
    assert record is not None
    assert record["submission_guard"]["completed_at"].endswith("Z")


def test_gfs_ignores_wrong_day_and_outside_window(tmp_path: Path) -> None:
    config = settings(tmp_path)
    runner = FakeRunner(config.generated_config_root)
    controller = Controller(config, runner=runner)

    monday = controller.gfs_poll(utc("2026-08-03T00:15:00Z"))
    too_early = controller.gfs_poll(utc("2026-08-05T00:00:00Z"))

    assert "Wednesday/Saturday" in monday[0]["reason"]
    assert "before the configured" in too_early[0]["reason"]
    assert runner.commands == []


def test_era5_watcher_backfills_ready_dates_independently(tmp_path: Path) -> None:
    config = settings(tmp_path, era5_max_probes_per_run=3)
    runner = FakeRunner(config.generated_config_root)
    runner.availability("era5", "20260801", False)
    runner.availability("era5", "20260805", True)
    runner.availability("era5", "20260808", False)
    controller = Controller(config, runner=runner)

    results = controller.era5_watch(utc("2026-08-10T13:00:00Z"))

    by_date = {result["issue_date"]: result for result in results}
    assert by_date["20260801"]["outcome"] == "waiting"
    assert by_date["20260805"]["stage"] == "submitted"
    assert by_date["20260808"]["outcome"] == "waiting"
    submit_commands = [command for command in runner.commands if command[2] == "submit"]
    assert len(submit_commands) == 1
    assert "fuxi_era5_20260805_ens100.json" in submit_commands[0][-1]


def test_safe_timeout_retries_then_alerts(tmp_path: Path) -> None:
    config = settings(tmp_path, max_safe_attempts=2)
    runner = FakeRunner(config.generated_config_root)
    runner.availability("gfs", "20260805", timeout(), timeout())
    alerts: list[tuple[str, str]] = []

    def alert(subject: str, body: str) -> dict[str, Any]:
        alerts.append((subject, body))
        return {"sent": True, "reason": None}

    controller = Controller(config, runner=runner, alert=alert)
    first = controller.gfs_poll(utc("2026-08-05T00:15:00Z"))
    second = controller.gfs_poll(utc("2026-08-05T00:16:00Z"))

    assert first[0]["outcome"] == "retry"
    assert second[0]["stage"] == "failed"
    assert len(alerts) == 1
    assert "gfs 20260805" in alerts[0][0]


def test_submit_timeout_is_never_retried_automatically(tmp_path: Path) -> None:
    config = settings(tmp_path)
    runner = FakeRunner(config.generated_config_root)
    runner.availability("gfs", "20260805", True)
    runner.submit_error = timeout("submit")
    controller = Controller(
        config,
        runner=runner,
        alert=lambda *_: {"sent": False, "reason": "test"},
    )

    failed = controller.gfs_poll(utc("2026-08-05T00:15:00Z"))
    again = controller.gfs_poll(utc("2026-08-05T00:30:00Z"))

    assert failed[0]["stage"] == "failed"
    assert again[0]["outcome"] == "skipped"
    assert [command[2] for command in runner.commands].count("submit") == 1
    record = controller.store.load("gfs", dt.date(2026, 8, 5))
    assert record is not None
    assert record["submission_guard"].get("completed_at") is None


def test_safe_mode_submits_compute_without_legacy_publisher(tmp_path: Path) -> None:
    config = settings(tmp_path, submission_mode="inference_only")
    runner = FakeRunner(config.generated_config_root)
    runner.availability("gfs", "20260805", True)
    controller = Controller(config, runner=runner)

    result = controller.gfs_poll(utc("2026-08-05T00:15:00Z"))

    assert result[0]["stage"] == "submitted"
    sbatch_commands = [command for command in runner.commands if command[0] == "/test/sbatch"]
    assert len(sbatch_commands) == 2
    assert sbatch_commands[1][-2] == "--dependency=afterok:200"
    assert not any(len(command) > 2 and command[2] == "submit" for command in runner.commands)
    record = controller.store.load("gfs", dt.date(2026, 8, 5))
    assert record is not None
    assert record["submission"]["publication"] == "external_clean_clone_required"


def test_reconcile_alerts_immediately_on_slurm_failure(tmp_path: Path) -> None:
    config = settings(tmp_path, submission_mode="inference_only")
    runner = FakeRunner(config.generated_config_root)
    runner.availability("gfs", "20260805", True)
    runner.sacct_states = {"200": "COMPLETED", "201": "OUT_OF_MEMORY"}
    alerts: list[str] = []
    controller = Controller(
        config,
        runner=runner,
        alert=lambda subject, _body: alerts.append(subject)
        or {"sent": True, "reason": None},
    )
    controller.gfs_poll(utc("2026-08-05T00:15:00Z"))

    result = controller.reconcile(utc("2026-08-05T00:30:00Z"))

    assert result[0]["stage"] == "failed"
    assert "OUT_OF_MEMORY" in result[0]["reason"]
    assert alerts == ["FuXi automation failed: gfs 20260805"]


def test_reconcile_advances_validated_then_exported(tmp_path: Path) -> None:
    config = settings(tmp_path)
    runner = FakeRunner(config.generated_config_root)
    runner.availability("gfs", "20260805", True)
    runner.audit_payload = {
        "source": "gfs",
        "date": "20260805",
        "manifest_exists": True,
        "status": "generated_valid",
        "members": 100,
        "lead_days": 42,
    }
    controller = Controller(config, runner=runner)
    controller.gfs_poll(utc("2026-08-05T00:15:00Z"))

    forecast = config.dashboard_root / "public/data/forecasts/gfs/20260805.json"
    forecast.parent.mkdir(parents=True, exist_ok=True)
    forecast.write_text(
        json.dumps(
            {
                "issue": {
                    "initialization": "2026-08-05T00:00:00Z",
                    "members": 100,
                    "lead_days": 42,
                    "initial_condition_source": {"id": "gfs"},
                }
            }
        ),
        encoding="utf-8",
    )
    checksum = hashlib.sha256(forecast.read_bytes()).hexdigest()
    index = config.dashboard_root / "public/data/index.json"
    index.write_text(
        json.dumps(
            {
                "initial_condition_sources": [
                    {
                        "id": "gfs",
                        "issues": [
                            {
                                "id": "20260805",
                                "members": 100,
                                "forecast": "forecasts/gfs/20260805.json",
                                "checksums": {"forecast_sha256": checksum},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results = controller.reconcile(utc("2026-08-05T02:00:00Z"))

    assert [result.get("stage") for result in results] == ["validated", "exported"]
    state = controller.store.load("gfs", dt.date(2026, 8, 5))
    assert state["stage"] == "exported"
    assert state["artifacts"]["private_forecast"].endswith(
        "/private/forecasts/annual2026/20260805.nc"
    )
    assert "next_action" not in state


def test_reconcile_automatically_publishes_and_confirms_deployment(
    tmp_path: Path,
) -> None:
    config = settings(
        tmp_path,
        publisher_command=(
            "/test/publisher",
            "--source",
            "{source}",
            "--issue",
            "{issue}",
            "--config",
            "{config}",
            "--receipt",
            "{receipt}",
        ),
        deployment_check_command=(
            "/test/smoke",
            "--expected-commit",
            "{commit_sha}",
            "--expected-source",
            "{source}",
            "--expected-issue",
            "{issue}",
        ),
    )
    runner = FakeRunner(config.generated_config_root)
    runner.availability("gfs", "20260805", True)
    runner.audit_payload = {
        "source": "gfs",
        "date": "20260805",
        "manifest_exists": True,
        "status": "generated_valid",
        "members": 100,
        "lead_days": 42,
    }
    controller = Controller(config, runner=runner)
    controller.gfs_poll(utc("2026-08-05T00:15:00Z"))

    results = controller.reconcile(utc("2026-08-05T02:00:00Z"))

    assert [item.get("stage") for item in results] == [
        "validated",
        "pushed",
        "deployed",
    ]
    state = controller.store.load("gfs", dt.date(2026, 8, 5))
    assert state is not None
    assert state["stage"] == "deployed"
    assert state["deployment"]["pushed_commit"] == "a" * 40
    assert state["deployment"]["deployed_commit"] == "a" * 40
    assert "next_action" not in state


def test_postdeploy_latency_keeps_pushed_state_for_retry(tmp_path: Path) -> None:
    config = settings(
        tmp_path,
        publisher_command=(
            "/test/publisher",
            "--source",
            "{source}",
            "--issue",
            "{issue}",
            "--receipt",
            "{receipt}",
        ),
        deployment_check_command=("/test/smoke", "{commit_sha}"),
    )
    runner = FakeRunner(config.generated_config_root)
    runner.availability("gfs", "20260805", True)
    runner.audit_payload = {
        "source": "gfs",
        "date": "20260805",
        "manifest_exists": True,
        "status": "generated_valid",
        "members": 100,
        "lead_days": 42,
    }
    runner.smoke_error = timeout("postdeploy")
    controller = Controller(config, runner=runner)
    controller.gfs_poll(utc("2026-08-05T00:15:00Z"))

    results = controller.reconcile(utc("2026-08-05T02:00:00Z"))

    assert results[-1]["outcome"] == "waiting"
    state = controller.store.load("gfs", dt.date(2026, 8, 5))
    assert state is not None and state["stage"] == "pushed"


def test_dry_run_makes_no_commands_or_state(tmp_path: Path) -> None:
    config = settings(tmp_path)
    runner = FakeRunner(config.generated_config_root)
    controller = Controller(config, runner=runner, dry_run=True)

    result = controller.gfs_poll(utc("2026-08-05T00:15:00Z"))

    assert result[0]["outcome"] == "dry-run"
    assert runner.commands == []
    assert not config.state_root.exists()


@pytest.mark.parametrize("bad", [-1, -10])
def test_schedule_rejects_negative_lookback(bad: int) -> None:
    with pytest.raises(ValueError):
        scheduled_issue_dates(
            dt.date(2026, 8, 12),
            start_date=dt.date(2026, 8, 1),
            lookback_days=bad,
        )
