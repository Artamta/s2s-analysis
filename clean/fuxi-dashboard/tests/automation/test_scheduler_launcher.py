from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from automation import run_scheduler


def private_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def valid_environment(extra: str = "") -> str:
    return (
        "S2S_ALERT_EMAIL=forecast-alerts@institution.example\n"
        "S2S_GIT_AUTHOR_NAME=S2S Forecast Automation\n"
        "GIT_SSH_COMMAND='ssh -i /srv/s2s/key -o BatchMode=yes'\n"
        f"{extra}"
    )


def test_private_environment_is_parsed_without_shell_evaluation(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    path = private_file(
        tmp_path / "automation.env",
        valid_environment(f"S2S_PUBLISH_BRANCH=$(touch {marker})\n"),
    )

    values = run_scheduler.parse_private_environment(path)

    assert values["S2S_GIT_AUTHOR_NAME"] == "S2S Forecast Automation"
    assert values["GIT_SSH_COMMAND"] == "ssh -i /srv/s2s/key -o BatchMode=yes"
    assert values["S2S_PUBLISH_BRANCH"] == f"$(touch {marker})"
    assert not marker.exists()


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o666])
def test_private_environment_rejects_group_or_other_access(
    tmp_path: Path, mode: int
) -> None:
    path = private_file(tmp_path / "automation.env", valid_environment())
    path.chmod(mode)

    with pytest.raises(run_scheduler.LaunchError, match="group/other"):
        run_scheduler.parse_private_environment(path)


def test_private_environment_rejects_symlink(tmp_path: Path) -> None:
    target = private_file(tmp_path / "target.env", valid_environment())
    link = tmp_path / "automation.env"
    link.symlink_to(target)

    with pytest.raises(run_scheduler.LaunchError, match="non-symlink"):
        run_scheduler.parse_private_environment(link)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (valid_environment("UNEXPECTED=value\n"), "not allowed"),
        (
            valid_environment("S2S_ALERT_EMAIL=second@institution.example\n"),
            "duplicate",
        ),
        ("S2S_ALERT_EMAIL=\n", "valid private address"),
        ("S2S_ALERT_EMAIL=one@example.org,two@example.org\n", "valid private address"),
    ],
)
def test_private_environment_fails_closed(
    tmp_path: Path, content: str, message: str
) -> None:
    path = private_file(tmp_path / "automation.env", content)

    with pytest.raises(run_scheduler.LaunchError, match=message):
        run_scheduler.parse_private_environment(path)


def test_scheduler_config_requires_private_schema_v1_json(tmp_path: Path) -> None:
    valid = private_file(
        tmp_path / "scheduler.json", json.dumps({"schema_version": 1})
    )
    run_scheduler.verify_scheduler_config(valid)

    invalid = private_file(
        tmp_path / "invalid.json", json.dumps({"schema_version": 2})
    )
    with pytest.raises(run_scheduler.LaunchError, match="schema_version"):
        run_scheduler.verify_scheduler_config(invalid)


def test_launch_environment_is_deterministic_and_removes_python_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    monkeypatch.setenv("PYTHONHOME", "/untrusted/home")
    monkeypatch.setenv("PATH", "/untrusted/bin")

    environment = run_scheduler.launch_environment(
        {
            "S2S_ALERT_EMAIL": "alerts@example.org",
            "PATH": "/opt/node/bin:/usr/bin:/bin",
        }
    )

    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["PATH"] == "/opt/node/bin:/usr/bin:/bin"
    assert environment["S2S_ALERT_EMAIL"] == "alerts@example.org"


def test_launcher_builds_fixed_scheduler_command() -> None:
    command = run_scheduler.scheduler_command(["gfs-poll"])

    assert command == [
        "/usr/bin/python3",
        str(run_scheduler.SCHEDULER),
        "--config",
        "/home/raj.ayush/.config/s2s-fuxi/scheduler.json",
        "gfs-poll",
    ]


def test_cron_uses_host_time_and_staggers_the_lock_users() -> None:
    cron = Path("automation/cron/fuxi-automation.crontab.example").read_text(
        encoding="utf-8"
    )

    assert not any(
        line.startswith("CRON_TZ=") for line in cron.splitlines()
    )
    assert "*/15 * * * * /usr/bin/python3 $S2S_LAUNCHER gfs-poll" in cron
    assert "35 18 * * * /usr/bin/python3 $S2S_LAUNCHER era5-watch" in cron
    assert "2-52/10 * * * * /usr/bin/python3 $S2S_LAUNCHER reconcile" in cron
    assert cron.count("/usr/bin/logger -t s2s-fuxi-") == 3


def test_default_path_does_not_depend_on_an_interactive_shell() -> None:
    environment = run_scheduler.launch_environment(
        {"S2S_ALERT_EMAIL": "alerts@example.org"}
    )
    assert environment["PATH"] == run_scheduler.SAFE_PATH
    assert os.path.isabs(environment["PATH"].split(":")[0])
