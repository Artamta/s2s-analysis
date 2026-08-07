#!/usr/bin/env python3
"""Idempotent Wednesday/Saturday scheduler for operational FuXi forecasts.

This controller probes availability, asks the existing ``fuxi_operational.py``
driver to create each source/date config, and submits the existing Slurm stage
and inference scripts. GFS and ERA5 have independent entry points, so delayed
ERA5 can never hold up a near-real-time GFS issue.

The private state directory is the source of truth for scheduler decisions.
Publication and deployment may be performed by a separate clean-clone worker;
that worker can advance records with the ``mark`` command.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


UTC = dt.timezone.utc
ISSUE_STAGES = (
    "pending",
    "submitted",
    "validated",
    "exported",
    "pushed",
    "deployed",
    "failed",
)
PROGRESS_STAGES = ISSUE_STAGES[:-1]
SCHEDULED_WEEKDAYS = (2, 5)  # Wednesday and Saturday, datetime convention.
DATE_PATTERN = re.compile(r"^\d{8}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


class SchedulerError(RuntimeError):
    """Base error raised for a safe, actionable scheduler failure."""


class CommandFailure(SchedulerError):
    def __init__(
        self,
        command: Sequence[str],
        returncode: int | None,
        stdout: str,
        stderr: str,
        *,
        timed_out: bool = False,
    ) -> None:
        label = "timed out" if timed_out else f"exited {returncode}"
        super().__init__(f"command {label}: {' '.join(command)}")
        self.command = list(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


@dataclasses.dataclass(frozen=True)
class CompletedCommand:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    """Run commands without a shell and with a hard wall-clock timeout."""

    def run(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int,
        allowed_returncodes: Sequence[int] = (0,),
    ) -> CompletedCommand:
        try:
            result = subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandFailure(
                command,
                None,
                _text(exc.stdout),
                _text(exc.stderr),
                timed_out=True,
            ) from exc
        except OSError as exc:
            raise CommandFailure(command, None, "", f"{type(exc).__name__}: {exc}") from exc
        completed = CompletedCommand(
            tuple(command), result.returncode, result.stdout, result.stderr
        )
        if result.returncode not in allowed_returncodes:
            raise CommandFailure(
                command, result.returncode, result.stdout, result.stderr
            )
        return completed


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_now(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--now must include a UTC offset")
    return parsed.astimezone(UTC)


def parse_issue_date(value: str) -> dt.date:
    normalized = value.replace("-", "")
    if not DATE_PATTERN.fullmatch(normalized):
        raise argparse.ArgumentTypeError("date must use YYYYMMDD or YYYY-MM-DD")
    try:
        return dt.datetime.strptime(normalized, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def issue_id(value: dt.date) -> str:
    return value.strftime("%Y%m%d")


def parse_clock(value: str) -> dt.time:
    try:
        parsed = dt.datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise SchedulerError(f"invalid UTC clock {value!r}; expected HH:MM") from exc
    return parsed.replace(tzinfo=UTC)


def scheduled_issue_dates(
    today: dt.date,
    *,
    start_date: dt.date,
    lookback_days: int,
    weekdays: Sequence[int] = SCHEDULED_WEEKDAYS,
) -> list[dt.date]:
    """Return due issue dates oldest first, bounded by an explicit start date."""

    if lookback_days < 0:
        raise ValueError("lookback_days cannot be negative")
    lower = max(start_date, today - dt.timedelta(days=lookback_days))
    dates: list[dt.date] = []
    cursor = lower
    while cursor <= today:
        if cursor.weekday() in weekdays:
            dates.append(cursor)
        cursor += dt.timedelta(days=1)
    return dates


@dataclasses.dataclass(frozen=True)
class Settings:
    config_path: Path
    operational_python: Path
    operational_script: Path
    sbatch_path: Path
    sacct_path: Path
    slurm_root: Path
    submission_mode: str
    dashboard_root: Path
    generated_config_root: Path
    state_root: Path
    start_date: dt.date
    gfs_window_start: dt.time
    gfs_window_end: dt.time
    gfs_poll_minutes: int
    era5_lookback_days: int
    era5_max_probes_per_run: int
    era5_poll_hours: int
    probe_timeout_seconds: int
    create_timeout_seconds: int
    submit_timeout_seconds: int
    slurm_status_timeout_seconds: int
    audit_timeout_seconds: int
    max_safe_attempts: int
    retry_base_seconds: int
    retry_max_seconds: int
    submission_stale_hours: int
    sendmail_path: Path
    publisher_command: tuple[str, ...] = ()
    publisher_timeout_seconds: int = 7200
    deployment_check_command: tuple[str, ...] = ()
    deployment_check_timeout_seconds: int = 120
    deployment_stale_hours: int = 6

    @classmethod
    def load(cls, path: Path) -> "Settings":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise SchedulerError("scheduler config schema_version must be 1")
        base = path.resolve().parent

        def resolved(name: str) -> Path:
            candidate = Path(payload[name]).expanduser()
            return candidate if candidate.is_absolute() else (base / candidate).resolve()

        schedule = payload.get("schedule", {})
        timeouts = payload.get("timeouts_seconds", {})
        retry = payload.get("retry", {})
        publication = payload.get("publication", {})

        def command(name: str) -> tuple[str, ...]:
            value = publication.get(name, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise SchedulerError(f"publication.{name} must be a string list")
            return tuple(value)

        settings = cls(
            config_path=path.resolve(),
            operational_python=resolved("operational_python"),
            operational_script=resolved("operational_script"),
            sbatch_path=resolved("sbatch_path"),
            sacct_path=resolved("sacct_path"),
            slurm_root=resolved("slurm_root"),
            submission_mode=str(payload.get("submission_mode", "inference_only")),
            dashboard_root=resolved("dashboard_root"),
            generated_config_root=resolved("generated_config_root"),
            state_root=resolved("state_root"),
            start_date=parse_issue_date(schedule["start_date"]),
            gfs_window_start=parse_clock(schedule.get("gfs_window_start_utc", "00:15")),
            gfs_window_end=parse_clock(schedule.get("gfs_window_end_utc", "12:00")),
            gfs_poll_minutes=int(schedule.get("gfs_poll_minutes", 15)),
            era5_lookback_days=int(schedule.get("era5_lookback_days", 35)),
            era5_max_probes_per_run=int(schedule.get("era5_max_probes_per_run", 4)),
            era5_poll_hours=int(schedule.get("era5_poll_hours", 24)),
            probe_timeout_seconds=int(timeouts.get("probe", 300)),
            create_timeout_seconds=int(timeouts.get("create", 1200)),
            submit_timeout_seconds=int(timeouts.get("submit", 120)),
            slurm_status_timeout_seconds=int(timeouts.get("slurm_status", 30)),
            audit_timeout_seconds=int(timeouts.get("audit", 120)),
            max_safe_attempts=int(retry.get("max_safe_attempts", 4)),
            retry_base_seconds=int(retry.get("base_seconds", 60)),
            retry_max_seconds=int(retry.get("max_seconds", 1800)),
            submission_stale_hours=int(payload.get("submission_stale_hours", 12)),
            sendmail_path=resolved("sendmail_path"),
            publisher_command=command("command"),
            publisher_timeout_seconds=int(publication.get("timeout_seconds", 7200)),
            deployment_check_command=command("deployment_check_command"),
            deployment_check_timeout_seconds=int(
                publication.get("deployment_check_timeout_seconds", 120)
            ),
            deployment_stale_hours=int(
                publication.get("deployment_stale_hours", 6)
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        positive = {
            "gfs_poll_minutes": self.gfs_poll_minutes,
            "era5_lookback_days": self.era5_lookback_days,
            "era5_max_probes_per_run": self.era5_max_probes_per_run,
            "era5_poll_hours": self.era5_poll_hours,
            "probe_timeout_seconds": self.probe_timeout_seconds,
            "create_timeout_seconds": self.create_timeout_seconds,
            "submit_timeout_seconds": self.submit_timeout_seconds,
            "slurm_status_timeout_seconds": self.slurm_status_timeout_seconds,
            "audit_timeout_seconds": self.audit_timeout_seconds,
            "max_safe_attempts": self.max_safe_attempts,
            "retry_base_seconds": self.retry_base_seconds,
            "retry_max_seconds": self.retry_max_seconds,
            "submission_stale_hours": self.submission_stale_hours,
            "publisher_timeout_seconds": self.publisher_timeout_seconds,
            "deployment_check_timeout_seconds": self.deployment_check_timeout_seconds,
            "deployment_stale_hours": self.deployment_stale_hours,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise SchedulerError(f"config values must be positive: {', '.join(invalid)}")
        if self.gfs_window_start >= self.gfs_window_end:
            raise SchedulerError("GFS start must be before the end of the UTC window")
        if self.submission_mode not in {"inference_only", "legacy_driver"}:
            raise SchedulerError(
                "submission_mode must be inference_only or legacy_driver"
            )
        required_files = [self.operational_python, self.operational_script]
        if self.submission_mode == "inference_only":
            required_files.extend(
                [
                    self.sbatch_path,
                    self.sacct_path,
                    self.slurm_root / "stage_fuxi_operational.sbatch",
                    self.slurm_root / "run_fuxi_operational.sbatch",
                ]
            )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise SchedulerError(
                "required scheduler executables/files are missing: " + ", ".join(missing)
            )
        if not self.dashboard_root.is_dir():
            raise SchedulerError(f"dashboard_root does not exist: {self.dashboard_root}")
        for name, template in (
            ("publication.command", self.publisher_command),
            ("publication.deployment_check_command", self.deployment_check_command),
        ):
            if not template:
                continue
            if not Path(template[0]).is_absolute():
                raise SchedulerError(f"{name} executable must be an absolute path")
            if any("\n" in item or "\0" in item for item in template):
                raise SchedulerError(f"{name} contains an unsafe argument")

    def operational(self, *arguments: str) -> list[str]:
        return [str(self.operational_python), str(self.operational_script), *arguments]

    def generated_config(self, source: str, date: dt.date) -> Path:
        return self.generated_config_root / f"fuxi_{source}_{issue_id(date)}_ens100.json"


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, source: str, date: dt.date) -> Path:
        return self.root / "issues" / source / f"{issue_id(date)}.json"

    def load(self, source: str, date: dt.date) -> dict[str, Any] | None:
        path = self.path(source, date)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        _validate_record(value, source, date)
        return value

    def save(self, record: dict[str, Any]) -> None:
        source = str(record["source"])
        date = parse_issue_date(str(record["issue_date"]))
        _validate_record(record, source, date)
        _atomic_json_write(self.path(source, date), record)

    def records(self) -> Iterator[dict[str, Any]]:
        issue_root = self.root / "issues"
        if not issue_root.is_dir():
            return
        for path in sorted(issue_root.glob("*/*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            _validate_record(value, path.parent.name, parse_issue_date(path.stem))
            yield value

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / "controller.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SchedulerError("another scheduler process holds the state lock") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_record(record: dict[str, Any], source: str, date: dt.date) -> None:
    if source not in {"gfs", "era5"}:
        raise SchedulerError(f"invalid state source {source!r}")
    if record.get("source") != source or record.get("issue_date") != issue_id(date):
        raise SchedulerError("state record identity does not match its path")
    if record.get("stage") not in ISSUE_STAGES:
        raise SchedulerError(f"invalid state stage {record.get('stage')!r}")


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            with contextlib.suppress(OSError):
                temporary.unlink()


def new_record(source: str, date: dt.date, now: dt.datetime) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": source,
        "issue_date": issue_id(date),
        "members": 100,
        "stage": "pending",
        "created_at": _iso(now),
        "updated_at": _iso(now),
        "attempts": {},
        "history": [
            {"at": _iso(now), "stage": "pending", "reason": "scheduled issue discovered"}
        ],
        "artifacts": {},
    }


def advance(
    record: dict[str, Any],
    stage: str,
    now: dt.datetime,
    reason: str,
) -> None:
    current = str(record["stage"])
    if stage not in ISSUE_STAGES:
        raise SchedulerError(f"invalid target stage {stage!r}")
    if current == stage:
        return
    if current == "failed":
        if stage != "pending":
            raise SchedulerError("failed state must be explicitly resumed before advancing")
        record["stage"] = stage
        record["updated_at"] = _iso(now)
        record.setdefault("history", []).append(
            {"at": _iso(now), "stage": stage, "reason": reason}
        )
        return
    if stage != "failed":
        if current not in PROGRESS_STAGES:
            raise SchedulerError(f"cannot advance {current} to {stage}")
        if PROGRESS_STAGES.index(stage) < PROGRESS_STAGES.index(current):
            raise SchedulerError(f"cannot move state backwards from {current} to {stage}")
    record["stage"] = stage
    record["updated_at"] = _iso(now)
    record.setdefault("history", []).append(
        {"at": _iso(now), "stage": stage, "reason": reason}
    )


class AlertManager:
    """Send private alerts through local sendmail when S2S_ALERT_EMAIL is set."""

    def __init__(self, sendmail_path: Path) -> None:
        self.sendmail_path = sendmail_path

    def send(self, subject: str, body: str) -> dict[str, Any]:
        recipient = os.environ.get("S2S_ALERT_EMAIL", "").strip()
        if not recipient:
            return {"sent": False, "reason": "S2S_ALERT_EMAIL is unset"}
        if not re.fullmatch(r"[^@\s,]+@[^@\s,]+", recipient):
            return {"sent": False, "reason": "S2S_ALERT_EMAIL is invalid"}
        if not self.sendmail_path.is_file():
            return {"sent": False, "reason": f"sendmail missing: {self.sendmail_path}"}
        message = (
            f"To: {recipient}\n"
            f"Subject: {subject}\n"
            "Content-Type: text/plain; charset=utf-8\n\n"
            f"{body}\n"
        )
        try:
            result = subprocess.run(
                [str(self.sendmail_path), "-t"],
                input=message,
                text=True,
                check=False,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"sent": False, "reason": f"sendmail failed: {exc}"}
        return {
            "sent": result.returncode == 0,
            "reason": result.stderr.strip() or None,
        }


class Controller:
    def __init__(
        self,
        settings: Settings,
        *,
        runner: CommandRunner | None = None,
        alert: Callable[[str, str], dict[str, Any]] | None = None,
        dry_run: bool = False,
    ) -> None:
        self.settings = settings
        self.store = StateStore(settings.state_root)
        self.runner = runner or CommandRunner()
        manager = AlertManager(settings.sendmail_path)
        self.alert = alert or manager.send
        self.dry_run = dry_run

    def gfs_poll(
        self, now: dt.datetime, requested_date: dt.date | None = None
    ) -> list[dict[str, Any]]:
        date = requested_date or now.date()
        if date.weekday() not in SCHEDULED_WEEKDAYS:
            return [self._skip("gfs", date, "not a Wednesday/Saturday issue")]
        if date > now.date():
            return [self._skip("gfs", date, "future issue dates are not eligible")]
        if date < self.settings.start_date:
            return [self._skip("gfs", date, "before configured automation start")]
        current_time = now.timetz()
        if requested_date is None and current_time < self.settings.gfs_window_start:
            return [self._skip("gfs", date, "before the configured GFS UTC window")]
        if requested_date is None and current_time > self.settings.gfs_window_end:
            record = self.store.load("gfs", date)
            if record:
                return [self._skip("gfs", date, f"window closed; already {record['stage']}")]
            if self._public_export_exists("gfs", date):
                return [self._process_pending("gfs", date, now)]
            if self.dry_run:
                return [
                    {
                        "source": "gfs",
                        "issue_date": issue_id(date),
                        "outcome": "dry-run",
                        "stage": "pending",
                        "actions": ["record missed GFS availability window", "alert"],
                    }
                ]
            record = new_record("gfs", date, now)
            self.store.save(record)
            return [
                self._terminal_failure(
                    record, now, "GFS polling window closed before submission"
                )
            ]
        return [self._process_pending("gfs", date, now)]

    def era5_watch(
        self, now: dt.datetime, requested_date: dt.date | None = None
    ) -> list[dict[str, Any]]:
        if requested_date is not None:
            candidates = [requested_date]
        else:
            candidates = scheduled_issue_dates(
                now.date(),
                start_date=self.settings.start_date,
                lookback_days=self.settings.era5_lookback_days,
            )
            # Probe never-seen dates first, then the least recently checked
            # pending dates. This prevents a permanently missing old issue from
            # starving newer delayed references when the per-run cap is active.
            def priority(date: dt.date) -> tuple[int, str, dt.date]:
                record = self.store.load("era5", date)
                if record is None:
                    return (0, "", date)
                checked = str(
                    record.get("last_probe", {}).get("checked_at")
                    or record.get("updated_at", "")
                )
                return (1, checked, date)

            candidates.sort(key=priority)
        results: list[dict[str, Any]] = []
        probed = 0
        for date in candidates:
            if date.weekday() not in SCHEDULED_WEEKDAYS:
                results.append(self._skip("era5", date, "not a Wednesday/Saturday issue"))
                continue
            if date > now.date():
                results.append(self._skip("era5", date, "future issue dates are not eligible"))
                continue
            if date < self.settings.start_date:
                results.append(self._skip("era5", date, "before configured automation start"))
                continue
            record = self.store.load("era5", date)
            if record and record["stage"] not in {"pending"}:
                results.append(self._skip("era5", date, f"already {record['stage']}"))
                continue
            if record and not self._retry_due(record, now):
                results.append(self._skip("era5", date, "next ERA5 probe is not due"))
                continue
            if probed >= self.settings.era5_max_probes_per_run:
                results.append(self._skip("era5", date, "per-run ERA5 probe limit reached"))
                continue
            results.append(self._process_pending("era5", date, now))
            probed += 1
        return results

    def reconcile(self, now: dt.datetime) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for record in list(self.store.records()):
            stage = record["stage"]
            if stage not in {"submitted", "validated", "pushed"}:
                continue
            source = str(record["source"])
            date = parse_issue_date(str(record["issue_date"]))
            if stage == "submitted":
                slurm_failure = self._slurm_failure(record, now)
                if slurm_failure:
                    results.append(self._terminal_failure(record, now, slurm_failure))
                    continue
                result = self._reconcile_validation(record, source, date, now)
                results.append(result)
                if result["outcome"] == "failed":
                    continue
            if record["stage"] == "validated":
                if self.settings.publisher_command:
                    result = self._publish_validated(record, source, date, now)
                    results.append(result)
                    if result["outcome"] == "failed":
                        continue
                elif self._public_export_exists(source, date):
                    advance(record, "exported", now, "validated public forecast export found")
                    record.pop("next_action", None)
                    record["artifacts"]["public_forecast"] = str(
                        self._public_forecast_path(source, date)
                    )
                    if not self.dry_run:
                        self.store.save(record)
                    results.append(self._result(record, "advanced", "exported"))
            if (
                record["stage"] == "pushed"
                and self.settings.deployment_check_command
            ):
                results.append(
                    self._confirm_deployment(record, source, date, now)
                )
        return results

    def mark(
        self,
        source: str,
        date: dt.date,
        stage: str,
        now: dt.datetime,
        *,
        commit_sha: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        record = self.store.load(source, date)
        if record is None:
            raise SchedulerError("cannot mark an issue that has no scheduler state")
        prerequisites = {
            "validated": "submitted",
            "exported": "validated",
            "pushed": "exported",
            "deployed": "pushed",
        }
        if stage not in prerequisites:
            raise SchedulerError("mark supports validated/exported/pushed/deployed only")
        current = str(record["stage"])
        if current == stage or (
            current in PROGRESS_STAGES
            and PROGRESS_STAGES.index(current) > PROGRESS_STAGES.index(stage)
        ):
            return self._result(record, "skipped", f"already {current}")
        if current != prerequisites[stage]:
            raise SchedulerError(
                f"{stage} requires {prerequisites[stage]}, current state is {current}"
            )
        if stage in {"pushed", "deployed"}:
            if not commit_sha or not SHA_PATTERN.fullmatch(commit_sha.lower()):
                raise SchedulerError(f"{stage} requires a 40-64 character commit SHA")
            record.setdefault("deployment", {})[f"{stage}_commit"] = commit_sha.lower()
        advance(record, stage, now, reason or f"externally confirmed {stage}")
        if stage == "exported":
            record.pop("next_action", None)
        if not self.dry_run:
            self.store.save(record)
        return self._result(record, "advanced", stage)

    def resume(
        self,
        source: str,
        date: dt.date,
        now: dt.datetime,
        *,
        acknowledge_safe_to_resubmit: bool,
    ) -> dict[str, Any]:
        record = self.store.load(source, date)
        if not record or record["stage"] != "failed":
            raise SchedulerError("resume requires an existing failed record")
        previous_stage = next(
            (
                str(item.get("stage"))
                for item in reversed(record.get("history", [])[:-1])
                if item.get("stage") in PROGRESS_STAGES
            ),
            "pending",
        )
        guard = record.get("submission_guard", {})
        if guard.get("started_at") and not guard.get("completed_at"):
            if not acknowledge_safe_to_resubmit:
                raise SchedulerError(
                    "submission outcome is ambiguous; first verify that no jobs are active, "
                    "then pass --acknowledge-safe-to-resubmit"
                )
            record.pop("submission_guard", None)
        publication_guard = record.get("publication_guard", {})
        if publication_guard.get("started_at") and not publication_guard.get(
            "completed_at"
        ):
            if not acknowledge_safe_to_resubmit:
                raise SchedulerError(
                    "publication outcome is ambiguous; inspect the remote branch and "
                    "receipt, then pass --acknowledge-safe-to-resubmit"
                )
            record.pop("publication_guard", None)
        record["attempts"] = {}
        record.pop("next_attempt_at", None)
        record.pop("last_error", None)
        advance(record, "pending", now, "operator resumed failed issue")
        if previous_stage in {"validated", "exported", "pushed"}:
            advance(
                record,
                previous_stage,
                now,
                f"operator restored completed {previous_stage} prerequisite",
            )
        if not self.dry_run:
            self.store.save(record)
        return self._result(record, "advanced", str(record["stage"]))

    def _process_pending(
        self, source: str, date: dt.date, now: dt.datetime
    ) -> dict[str, Any]:
        existing = self.store.load(source, date)
        record = existing or new_record(source, date, now)
        if record["stage"] != "pending":
            return self._skip(source, date, f"already {record['stage']}")
        if not self._retry_due(record, now):
            return self._skip(source, date, "retry backoff has not elapsed")
        if self._public_export_exists(source, date):
            advance(record, "exported", now, "adopted existing validated 100-member web export")
            record["artifacts"]["public_forecast"] = str(
                self._public_forecast_path(source, date)
            )
            if not self.dry_run:
                self.store.save(record)
            return self._result(record, "adopted", "existing export")
        if self.dry_run:
            return {
                "source": source,
                "issue_date": issue_id(date),
                "outcome": "dry-run",
                "stage": record["stage"],
                "actions": ["probe", "create if available", "submit once"],
            }
        self.store.save(record)
        try:
            probe = self._probe(source, date)
        except CommandFailure as exc:
            return self._safe_failure(record, now, "probe", exc)
        record["last_probe"] = probe
        record["attempts"]["probe"] = 0
        if not probe.get("available"):
            delay = (
                dt.timedelta(minutes=self.settings.gfs_poll_minutes)
                if source == "gfs"
                else dt.timedelta(hours=self.settings.era5_poll_hours)
            )
            record["next_attempt_at"] = _iso(now + delay)
            record["updated_at"] = _iso(now)
            if source == "gfs" and self._gfs_last_poll(now):
                message = "GFS inputs were incomplete at the end of the polling window"
                return self._terminal_failure(record, now, message)
            self.store.save(record)
            return self._result(record, "waiting", "source inputs incomplete")

        record.pop("next_attempt_at", None)
        config_path = self._ensure_config(record, source, date, now, probe)
        if config_path is None:
            outcome = "failed" if record["stage"] == "failed" else "retry"
            return self._result(record, outcome, str(record.get("last_error")))
        return self._submit_once(record, source, date, now, config_path)

    def _probe(self, source: str, date: dt.date) -> dict[str, Any]:
        completed = self.runner.run(
            self.settings.operational(
                "probe", "--source", source, "--date", issue_id(date)
            ),
            timeout_seconds=self.settings.probe_timeout_seconds,
            allowed_returncodes=(0, 2),
        )
        try:
            payload = json.loads(completed.stdout)
            if not isinstance(payload, list) or len(payload) != 1:
                raise ValueError("expected one probe record")
            probe = payload[0]
        except (json.JSONDecodeError, ValueError) as exc:
            raise CommandFailure(
                completed.command,
                completed.returncode,
                completed.stdout,
                f"invalid probe JSON: {exc}",
            ) from exc
        if probe.get("source") != source or probe.get("issue_date") != date.isoformat():
            raise CommandFailure(
                completed.command,
                completed.returncode,
                completed.stdout,
                "probe identity does not match requested source/date",
            )
        required_key = "required_files" if source == "gfs" else "required_objects"
        present_key = "present_files" if source == "gfs" else "present_objects"
        expected = 16 if source == "gfs" else 768
        if probe.get("available") and (
            int(probe.get(required_key, -1)) != expected
            or int(probe.get(present_key, -1)) != expected
        ):
            raise CommandFailure(
                completed.command,
                completed.returncode,
                completed.stdout,
                f"available probe did not prove all {expected} required inputs",
            )
        return probe

    def _ensure_config(
        self,
        record: dict[str, Any],
        source: str,
        date: dt.date,
        now: dt.datetime,
        probe: dict[str, Any],
    ) -> Path | None:
        stored = record["artifacts"].get("config")
        if stored and Path(stored).is_file():
            try:
                self._validate_generated_config(Path(stored), source, date)
            except SchedulerError as exc:
                self._terminal_failure(record, now, str(exc))
                return None
            return Path(stored)
        expected = self.settings.generated_config(source, date)
        if expected.is_file():
            try:
                self._validate_generated_config(expected, source, date)
            except SchedulerError as exc:
                self._terminal_failure(record, now, str(exc))
                return None
            record["artifacts"]["config"] = str(expected)
            self.store.save(record)
            return expected
        command = self.settings.operational(
            "create", "--source", source, "--date", issue_id(date), "--members", "100"
        )
        if source == "era5":
            receipt = self.settings.state_root / "probes" / source / f"{issue_id(date)}.json"
            _atomic_json_write(receipt, probe)
            command.extend(["--era5-probe", str(receipt)])
            record["artifacts"]["availability_probe"] = str(receipt)
        try:
            completed = self.runner.run(
                command, timeout_seconds=self.settings.create_timeout_seconds
            )
        except CommandFailure as exc:
            if expected.is_file():
                try:
                    self._validate_generated_config(expected, source, date)
                except SchedulerError as validation_error:
                    self._terminal_failure(record, now, str(validation_error))
                    return None
                record["artifacts"]["config"] = str(expected)
                self.store.save(record)
                return expected
            self._safe_failure(record, now, "create", exc)
            return None
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        candidate = Path(lines[-1]) if lines else expected
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.is_file():
            self._terminal_failure(record, now, f"create did not produce config: {candidate}")
            return None
        try:
            self._validate_generated_config(candidate, source, date)
        except SchedulerError as exc:
            self._terminal_failure(record, now, str(exc))
            return None
        record["artifacts"]["config"] = str(candidate)
        record["updated_at"] = _iso(now)
        self.store.save(record)
        return candidate

    def _validate_generated_config(
        self, path: Path, source: str, date: dt.date
    ) -> None:
        if path.resolve().parent != self.settings.generated_config_root.resolve():
            raise SchedulerError(f"generated config is outside the configured root: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            configured_source = (
                "gfs"
                if payload.get("input", {}).get("builder") == "gfs_daily_proxy"
                else "era5"
            )
            valid = (
                int(payload.get("members", 0)) == 100
                and configured_source == source
                and issue_id(date) in str(payload.get("run_label", ""))
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SchedulerError(f"cannot validate generated config {path}: {exc}") from exc
        if not valid:
            raise SchedulerError(f"generated config identity is invalid: {path}")

    def _submit_once(
        self,
        record: dict[str, Any],
        source: str,
        date: dt.date,
        now: dt.datetime,
        config_path: Path,
    ) -> dict[str, Any]:
        guard = record.get("submission_guard")
        if guard and not guard.get("completed_at"):
            return self._terminal_failure(
                record,
                now,
                "submission outcome is ambiguous; automatic resubmission is disabled",
            )
        token = uuid.uuid4().hex
        record["submission_guard"] = {"token": token, "started_at": _iso(now)}
        self.store.save(record)  # durable before the side effect
        try:
            if self.settings.submission_mode == "inference_only":
                submission = self._submit_inference_only(source, date, config_path)
            else:
                completed = self.runner.run(
                    self.settings.operational(
                        "submit",
                        "--date",
                        issue_id(date),
                        "--configs",
                        str(config_path),
                    ),
                    timeout_seconds=self.settings.submit_timeout_seconds,
                )
                submission = json.loads(completed.stdout)
        except CommandFailure as exc:
            # sbatch may have accepted work before a timeout or partial failure.
            # Never retry automatically because duplicate inference is expensive.
            record["last_error"] = _command_error("submit", exc)
            return self._terminal_failure(
                record,
                now,
                "submission command failed with an ambiguous outcome; inspect Slurm jobs",
            )
        except (json.JSONDecodeError, ValueError) as exc:
            record["last_error"] = {"operation": "submit", "message": str(exc)}
            return self._terminal_failure(
                record,
                now,
                "submission returned invalid JSON; inspect Slurm jobs before resuming",
            )
        record["submission_guard"]["completed_at"] = _iso(now)
        record["submission"] = submission
        reason = (
            "staging and inference jobs accepted; clean-clone publication is pending"
            if self.settings.submission_mode == "inference_only"
            else "staging, inference, and legacy publication jobs accepted"
        )
        advance(record, "submitted", now, reason)
        self.store.save(record)
        return self._result(record, "submitted", "job graph accepted")

    def _submit_inference_only(
        self, source: str, date: dt.date, config_path: Path
    ) -> dict[str, Any]:
        """Submit compute only; a clean publishing clone owns all web writes."""

        export_values = {
            "FUXI_CONFIG": str(config_path.resolve()),
            "FUXI_DATE": issue_id(date),
            "FUXI_SOURCE": source,
        }
        if any("," in value or "\n" in value for value in export_values.values()):
            raise ValueError("Slurm export values cannot contain commas or newlines")
        export = "ALL," + ",".join(
            f"{key}={value}" for key, value in export_values.items()
        )
        stage_script = self.settings.slurm_root / "stage_fuxi_operational.sbatch"
        inference_script = self.settings.slurm_root / "run_fuxi_operational.sbatch"
        stage = self.runner.run(
            [
                str(self.settings.sbatch_path),
                "--parsable",
                f"--export={export}",
                str(stage_script),
            ],
            timeout_seconds=self.settings.submit_timeout_seconds,
        )
        stage_job = _parse_slurm_job_id(stage.stdout)
        inference = self.runner.run(
            [
                str(self.settings.sbatch_path),
                "--parsable",
                f"--export={export}",
                f"--dependency=afterok:{stage_job}",
                str(inference_script),
            ],
            timeout_seconds=self.settings.submit_timeout_seconds,
        )
        inference_job = _parse_slurm_job_id(inference.stdout)
        return {
            "mode": "inference_only",
            "source": source,
            "issue_date": issue_id(date),
            "stage_job": stage_job,
            "inference_job": inference_job,
            "publication": "external_clean_clone_required",
        }

    def _reconcile_validation(
        self,
        record: dict[str, Any],
        source: str,
        date: dt.date,
        now: dt.datetime,
    ) -> dict[str, Any]:
        config = record.get("artifacts", {}).get("config")
        if not config:
            return self._terminal_failure(record, now, "submitted state has no config path")
        try:
            completed = self.runner.run(
                self.settings.operational(
                    "audit", "--config", str(config), "--date", issue_id(date)
                ),
                timeout_seconds=self.settings.audit_timeout_seconds,
            )
            audit = json.loads(completed.stdout)
        except (CommandFailure, json.JSONDecodeError) as exc:
            if self._submission_is_stale(record, now):
                return self._terminal_failure(
                    record, now, f"submitted job did not validate: {exc}"
                )
            return self._result(record, "waiting", "audit is not ready")
        record["last_audit"] = audit
        manifest_status = audit.get("status")
        if audit.get("manifest_exists") and manifest_status in {
            "generated_valid",
            "existing_valid",
        }:
            if (
                audit.get("source") != source
                or audit.get("date") != issue_id(date)
                or int(audit.get("members", 0)) != 100
                or int(audit.get("lead_days", 0)) != 42
            ):
                return self._terminal_failure(
                    record, now, "valid manifest identity does not match the scheduled issue"
                )
            try:
                private_artifacts = self._private_forecast_artifacts(Path(config), date)
            except SchedulerError as exc:
                return self._terminal_failure(record, now, str(exc))
            advance(record, "validated", now, "private run manifest passed validation")
            record["artifacts"].update(private_artifacts)
            record["next_action"] = {
                "kind": "export_and_publish_from_clean_clone",
                "source": source,
                "issue_date": issue_id(date),
                "config": str(config),
                **private_artifacts,
                "ready_at": _iso(now),
            }
            self.store.save(record)
            return self._result(record, "advanced", "validated")
        if manifest_status and manifest_status not in {"generated_valid", "existing_valid"}:
            return self._terminal_failure(
                record, now, f"run manifest is not valid: {manifest_status}"
            )
        if self._submission_is_stale(record, now):
            return self._terminal_failure(
                record, now, "submitted job exceeded the validation deadline"
            )
        self.store.save(record)
        return self._result(record, "waiting", "forecast manifest not ready")

    @staticmethod
    def _render_command(
        template: Sequence[str], values: dict[str, str]
    ) -> list[str]:
        try:
            rendered = [item.format_map(values) for item in template]
        except KeyError as exc:
            raise SchedulerError(
                f"publication command uses unknown placeholder {exc.args[0]!r}"
            ) from exc
        if any(not item or "\n" in item or "\0" in item for item in rendered):
            raise SchedulerError("publication command rendered an unsafe argument")
        return rendered

    def _publication_values(
        self,
        record: dict[str, Any],
        source: str,
        date: dt.date,
    ) -> dict[str, str]:
        receipt = (
            self.settings.state_root
            / "publication-receipts"
            / source
            / f"{issue_id(date)}.json"
        )
        config = str(record.get("artifacts", {}).get("config", ""))
        return {
            "source": source,
            "issue": issue_id(date),
            "date": issue_id(date),
            "config": config,
            "receipt": str(receipt),
            "state_root": str(self.settings.state_root),
            "dashboard_root": str(self.settings.dashboard_root),
            "commit_sha": str(
                record.get("deployment", {}).get("pushed_commit", "")
            ),
        }

    def _publish_validated(
        self,
        record: dict[str, Any],
        source: str,
        date: dt.date,
        now: dt.datetime,
    ) -> dict[str, Any]:
        """Run the clean export/publisher once and bind state to its receipt."""

        values = self._publication_values(record, source, date)
        if not values["config"]:
            return self._terminal_failure(
                record, now, "validated state has no immutable config path"
            )
        if self.dry_run:
            return self._result(record, "dry-run", "would export and publish")
        guard = record.get("publication_guard")
        if guard and not guard.get("completed_at"):
            return self._terminal_failure(
                record,
                now,
                "publication outcome is ambiguous; inspect Git and the receipt",
            )
        token = uuid.uuid4().hex
        record["publication_guard"] = {
            "token": token,
            "started_at": _iso(now),
        }
        self.store.save(record)
        command = self._render_command(self.settings.publisher_command, values)
        try:
            completed = self.runner.run(
                command,
                timeout_seconds=self.settings.publisher_timeout_seconds,
            )
        except CommandFailure as exc:
            record["last_error"] = _command_error("publish", exc)
            return self._terminal_failure(
                record,
                now,
                "clean publication failed or timed out; inspect remote state before resuming",
            )
        receipt_path = Path(values["receipt"])
        try:
            receipt = (
                json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt_path.is_file()
                else json.loads(completed.stdout)
            )
            commit_sha = str(receipt["commit_sha"]).lower()
            valid = (
                receipt.get("status") == "pushed"
                and receipt.get("pushed") is True
                and receipt.get("source") == source
                and receipt.get("issue") == issue_id(date)
                and SHA_PATTERN.fullmatch(commit_sha)
            )
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            record["last_error"] = {
                "operation": "publish_receipt",
                "message": str(exc),
            }
            return self._terminal_failure(
                record,
                now,
                "publication returned an invalid receipt; inspect remote state before resuming",
            )
        if not valid:
            return self._terminal_failure(
                record,
                now,
                "publication receipt did not prove a normal push",
            )
        record["publication_guard"]["completed_at"] = _iso(now)
        record["artifacts"]["publication_receipt"] = str(receipt_path)
        record["artifacts"]["public_forecast_sha256"] = str(
            receipt.get("forecast_sha256", "")
        )
        record.setdefault("deployment", {})["pushed_commit"] = commit_sha
        advance(record, "exported", now, "clean worktree export validated")
        advance(record, "pushed", now, "allow-listed publication pushed without force")
        record.pop("next_action", None)
        self.store.save(record)
        return self._result(record, "advanced", "pushed")

    def _confirm_deployment(
        self,
        record: dict[str, Any],
        source: str,
        date: dt.date,
        now: dt.datetime,
    ) -> dict[str, Any]:
        """Confirm Pages asynchronously; wait during normal CDN/deploy latency."""

        values = self._publication_values(record, source, date)
        command = self._render_command(
            self.settings.deployment_check_command, values
        )
        try:
            self.runner.run(
                command,
                timeout_seconds=self.settings.deployment_check_timeout_seconds,
            )
        except CommandFailure as exc:
            record["last_deployment_check"] = {
                "at": _iso(now),
                "ready": False,
                "error": _command_error("postdeploy_smoke", exc),
            }
            pushed_at = next(
                (
                    parse_now(item["at"])
                    for item in reversed(record.get("history", []))
                    if item.get("stage") == "pushed"
                ),
                now,
            )
            if now - pushed_at > dt.timedelta(
                hours=self.settings.deployment_stale_hours
            ):
                return self._terminal_failure(
                    record,
                    now,
                    "pushed forecast was not confirmed on the deployed site in time",
                )
            self.store.save(record)
            return self._result(record, "waiting", "deployment not visible yet")
        commit_sha = values["commit_sha"]
        record["last_deployment_check"] = {"at": _iso(now), "ready": True}
        record.setdefault("deployment", {})["deployed_commit"] = commit_sha
        advance(record, "deployed", now, "deployed checksum and issue smoke test passed")
        self.store.save(record)
        return self._result(record, "advanced", "deployed")

    def _slurm_failure(
        self, record: dict[str, Any], now: dt.datetime
    ) -> str | None:
        submission = record.get("submission", {})
        if submission.get("mode") != "inference_only":
            return None
        jobs = {
            "stage": str(submission.get("stage_job", "")),
            "inference": str(submission.get("inference_job", "")),
        }
        if not all(value.isdigit() for value in jobs.values()):
            return "inference-only submission is missing valid Slurm job ids"
        try:
            completed = self.runner.run(
                [
                    str(self.settings.sacct_path),
                    "--noheader",
                    "--allocations",
                    "--parsable2",
                    "--jobs",
                    ",".join(jobs.values()),
                    "--format=JobIDRaw,State",
                ],
                timeout_seconds=self.settings.slurm_status_timeout_seconds,
            )
        except CommandFailure as exc:
            record["last_slurm_check"] = {
                "at": _iso(now),
                "available": False,
                "error": _command_error("sacct", exc),
            }
            self.store.save(record)
            return None
        observed: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 2 and fields[0] in jobs.values():
                observed[fields[0]] = fields[1].split(maxsplit=1)[0].rstrip("+").upper()
        record["last_slurm_check"] = {
            "at": _iso(now),
            "available": True,
            "states": {name: observed.get(job) for name, job in jobs.items()},
        }
        failed_states = {
            "BOOT_FAIL",
            "CANCELLED",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "TIMEOUT",
        }
        for name, job in jobs.items():
            state = observed.get(job)
            if state in failed_states:
                return f"Slurm {name} job {job} ended in {state}"
        return None

    @staticmethod
    def _private_forecast_artifacts(
        config_path: Path, date: dt.date
    ) -> dict[str, str]:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            root = Path(config["storage_root"])
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise SchedulerError(
                f"cannot resolve private forecast artifacts from {config_path}: {exc}"
            ) from exc
        annual = f"annual{date.year}"
        issue = issue_id(date)
        return {
            "private_forecast": str(root / "forecasts" / annual / f"{issue}.nc"),
            "private_manifest": str(root / "manifests" / annual / f"{issue}.json"),
        }

    def _safe_failure(
        self,
        record: dict[str, Any],
        now: dt.datetime,
        operation: str,
        error: CommandFailure,
    ) -> dict[str, Any]:
        attempts = int(record["attempts"].get(operation, 0)) + 1
        record["attempts"][operation] = attempts
        record["last_error"] = _command_error(operation, error)
        if attempts >= self.settings.max_safe_attempts:
            return self._terminal_failure(
                record, now, f"{operation} failed {attempts} consecutive times"
            )
        delay = min(
            self.settings.retry_base_seconds * (2 ** (attempts - 1)),
            self.settings.retry_max_seconds,
        )
        record["next_attempt_at"] = _iso(now + dt.timedelta(seconds=delay))
        record["updated_at"] = _iso(now)
        self.store.save(record)
        return self._result(record, "retry", f"{operation} failed; retry in {delay}s")

    def _terminal_failure(
        self, record: dict[str, Any], now: dt.datetime, message: str
    ) -> dict[str, Any]:
        advance(record, "failed", now, message)
        record["last_error"] = record.get("last_error") or {"message": message}
        alert_result = self.alert(
            f"FuXi automation failed: {record['source']} {record['issue_date']}",
            f"{message}\n\nState: {self.store.path(record['source'], parse_issue_date(record['issue_date']))}",
        )
        record["last_alert"] = {"at": _iso(now), **alert_result}
        self.store.save(record)
        return self._result(record, "failed", message)

    def _retry_due(self, record: dict[str, Any], now: dt.datetime) -> bool:
        value = record.get("next_attempt_at")
        return not value or parse_now(str(value)) <= now

    def _gfs_last_poll(self, now: dt.datetime) -> bool:
        next_time = now + dt.timedelta(minutes=self.settings.gfs_poll_minutes)
        end = dt.datetime.combine(now.date(), self.settings.gfs_window_end)
        return next_time > end

    def _submission_is_stale(self, record: dict[str, Any], now: dt.datetime) -> bool:
        submitted = next(
            (
                parse_now(item["at"])
                for item in reversed(record.get("history", []))
                if item.get("stage") == "submitted"
            ),
            now,
        )
        return now - submitted > dt.timedelta(hours=self.settings.submission_stale_hours)

    def _public_forecast_path(self, source: str, date: dt.date) -> Path:
        return (
            self.settings.dashboard_root
            / "public/data/forecasts"
            / source
            / f"{issue_id(date)}.json"
        )

    def _public_export_exists(self, source: str, date: dt.date) -> bool:
        path = self._public_forecast_path(source, date)
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            issue = payload["issue"]
            forecast_valid = (
                issue.get("initial_condition_source", {}).get("id") == source
                and issue.get("initialization", "")[:10] == date.isoformat()
                and int(issue.get("members", 0)) == 100
                and int(issue.get("lead_days", 0)) == 42
            )
            index = json.loads(
                (self.settings.dashboard_root / "public/data/index.json").read_text(
                    encoding="utf-8"
                )
            )
            catalog = next(
                item
                for item in index["initial_condition_sources"]
                if item.get("id") == source
            )
            catalog_issue = next(
                item for item in catalog["issues"] if item.get("id") == issue_id(date)
            )
            checksum = catalog_issue.get("checksums", {}).get("forecast_sha256")
            catalog_valid = (
                int(catalog_issue.get("members", 0)) == 100
                and catalog_issue.get("forecast")
                == f"forecasts/{source}/{issue_id(date)}.json"
                and isinstance(checksum, str)
                and len(checksum) == 64
                and hashlib.sha256(path.read_bytes()).hexdigest() == checksum
            )
            return forecast_valid and catalog_valid
        except (
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ):
            return False

    @staticmethod
    def _skip(source: str, date: dt.date, reason: str) -> dict[str, Any]:
        return {
            "source": source,
            "issue_date": issue_id(date),
            "outcome": "skipped",
            "reason": reason,
        }

    @staticmethod
    def _result(
        record: dict[str, Any], outcome: str, reason: str
    ) -> dict[str, Any]:
        return {
            "source": record["source"],
            "issue_date": record["issue_date"],
            "outcome": outcome,
            "stage": record["stage"],
            "reason": reason,
        }


def _command_error(operation: str, error: CommandFailure) -> dict[str, Any]:
    return {
        "operation": operation,
        "timed_out": error.timed_out,
        "returncode": error.returncode,
        "message": str(error),
        "stderr_tail": error.stderr[-2000:],
    }


def _parse_slurm_job_id(stdout: str) -> str:
    value = stdout.strip().split(";", maxsplit=1)[0]
    if not value.isdigit():
        raise ValueError(f"sbatch returned an invalid job id: {stdout!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now", type=parse_now, help="UTC test/recovery clock")
    commands = parser.add_subparsers(dest="command", required=True)

    gfs = commands.add_parser("gfs-poll", help="Poll and submit today's GFS issue")
    gfs.add_argument("--date", type=parse_issue_date)

    era5 = commands.add_parser("era5-watch", help="Backfill ready delayed ERA5 issues")
    era5.add_argument("--date", type=parse_issue_date)

    commands.add_parser("reconcile", help="Advance submitted/validated issue states")
    commands.add_parser("status", help="Print private scheduler state summary")

    mark = commands.add_parser("mark", help="Record an externally completed stage")
    mark.add_argument("--source", choices=("gfs", "era5"), required=True)
    mark.add_argument("--date", type=parse_issue_date, required=True)
    mark.add_argument(
        "--stage", choices=("validated", "exported", "pushed", "deployed"), required=True
    )
    mark.add_argument("--commit-sha")
    mark.add_argument("--reason")

    resume = commands.add_parser("resume", help="Explicitly resume one failed issue")
    resume.add_argument("--source", choices=("gfs", "era5"), required=True)
    resume.add_argument("--date", type=parse_issue_date, required=True)
    resume.add_argument("--acknowledge-safe-to-resubmit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = args.now or _utc_now()
    try:
        settings = Settings.load(args.config)
        controller = Controller(settings, dry_run=args.dry_run)
        lock = contextlib.nullcontext() if args.dry_run else controller.store.lock()
        with lock:
            if args.command == "gfs-poll":
                result: Any = controller.gfs_poll(now, args.date)
            elif args.command == "era5-watch":
                result = controller.era5_watch(now, args.date)
            elif args.command == "reconcile":
                result = controller.reconcile(now)
            elif args.command == "status":
                result = list(controller.store.records())
            elif args.command == "mark":
                result = controller.mark(
                    args.source,
                    args.date,
                    args.stage,
                    now,
                    commit_sha=args.commit_sha,
                    reason=args.reason,
                )
            elif args.command == "resume":
                result = controller.resume(
                    args.source,
                    args.date,
                    now,
                    acknowledge_safe_to_resubmit=args.acknowledge_safe_to_resubmit,
                )
            else:  # pragma: no cover - argparse guarantees the command
                raise AssertionError(args.command)
    except (
        SchedulerError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        argparse.ArgumentError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
