#!/usr/bin/env python3
"""Launch the FuXi scheduler from cron with a checked private environment."""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import sys
from pathlib import Path
from typing import Sequence


PRIVATE_ROOT = Path("/home/raj.ayush/.config/s2s-fuxi")
ENV_FILE = PRIVATE_ROOT / "automation.env"
CONFIG_FILE = PRIVATE_ROOT / "scheduler.json"
PYTHON = Path("/usr/bin/python3")
SCHEDULER = Path(__file__).resolve().with_name("fuxi_scheduler.py")
ALLOWED_ENVIRONMENT = {
    "GIT_SSH_COMMAND",
    "PATH",
    "S2S_ALERT_EMAIL",
    "S2S_DASHBOARD_SUBDIR",
    "S2S_DRIVER_PYTHON",
    "S2S_GIT_AUTHOR_EMAIL",
    "S2S_GIT_AUTHOR_NAME",
    "S2S_PUBLISH_BRANCH",
    "S2S_PUBLISH_CLONE",
    "S2S_REPOSITORY_URL",
}
EMAIL_PATTERN = re.compile(r"^[^@\s,]+@[^@\s,]+$")
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class LaunchError(RuntimeError):
    """A private-file or launcher contract failed before scheduler execution."""


def verify_private_file(path: Path) -> None:
    """Require a mode-0600-style regular file owned by the service user."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LaunchError(f"private file is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise LaunchError(f"private path must be a regular non-symlink file: {path}")
    if metadata.st_uid != os.getuid():
        raise LaunchError(f"private file is not owned by uid {os.getuid()}: {path}")
    if metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise LaunchError(f"private file must not grant group/other access: {path}")
    if not metadata.st_mode & stat.S_IRUSR:
        raise LaunchError(f"private file is not owner-readable: {path}")


def parse_private_environment(path: Path) -> dict[str, str]:
    """Parse allow-listed KEY=VALUE records without evaluating shell syntax."""

    verify_private_file(path)
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LaunchError(f"cannot read private environment {path}: {exc}") from exc
    for number, original in enumerate(lines, start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise LaunchError(f"invalid environment record at {path}:{number}")
        key, raw_value = line.split("=", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in ALLOWED_ENVIRONMENT:
            raise LaunchError(f"environment key is not allowed at {path}:{number}: {key}")
        if key in values:
            raise LaunchError(f"duplicate environment key at {path}:{number}: {key}")
        if raw_value.startswith(("'", '"')):
            try:
                parsed = shlex.split(raw_value, comments=False, posix=True)
            except ValueError as exc:
                raise LaunchError(
                    f"invalid quoted value at {path}:{number}: {exc}"
                ) from exc
            if len(parsed) != 1:
                raise LaunchError(f"quoted value must be one field at {path}:{number}")
            value = parsed[0]
        else:
            # Unquoted text is literal. In particular, $, backticks, and shell
            # operators are never expanded or executed by this launcher.
            value = raw_value
        if "\x00" in value or "\n" in value or "\r" in value:
            raise LaunchError(f"environment value contains a control byte: {key}")
        values[key] = value
    email = values.get("S2S_ALERT_EMAIL", "")
    if not EMAIL_PATTERN.fullmatch(email):
        raise LaunchError("S2S_ALERT_EMAIL must contain one valid private address")
    return values


def verify_scheduler_config(path: Path) -> None:
    verify_private_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"scheduler config is not valid JSON: {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise LaunchError("scheduler config schema_version must be 1")


def scheduler_command(arguments: Sequence[str]) -> list[str]:
    if not arguments:
        raise LaunchError("scheduler command is required")
    return [
        str(PYTHON),
        str(SCHEDULER),
        "--config",
        str(CONFIG_FILE),
        *arguments,
    ]


def launch_environment(private_values: dict[str, str]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = private_values.get("PATH", SAFE_PATH)
    environment.update(private_values)
    return environment


def main(arguments: Sequence[str] | None = None) -> int:
    scheduler_arguments = list(sys.argv[1:] if arguments is None else arguments)
    try:
        private_values = parse_private_environment(ENV_FILE)
        verify_scheduler_config(CONFIG_FILE)
        command = scheduler_command(scheduler_arguments)
    except LaunchError as exc:
        print(json.dumps({"ok": False, "launcher_error": str(exc)}), file=sys.stderr)
        return 2
    os.execve(str(PYTHON), command, launch_environment(private_values))
    raise AssertionError("os.execve returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
