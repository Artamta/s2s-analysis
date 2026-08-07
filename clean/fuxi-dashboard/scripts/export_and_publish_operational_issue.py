#!/usr/bin/env python3
"""Export one validated private FuXi run and publish it from clean worktrees.

The scientific checkout is never used as a web-export destination.  A
disposable worktree based on the latest publication branch receives the
derived JSON/PDF package, then ``publish_operational_site.py`` copies only its
allow-listed public assets into a second disposable worktree for validation,
commit, and a normal non-force push.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publish_operational_site import (  # noqa: E402
    create_disposable_worktree,
    prepare_clone,
    remove_disposable_worktree,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("gfs", "era5"), required=True)
    parser.add_argument("--issue", required=True, help="Initialization YYYYMMDD.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--operational-script",
        type=Path,
        default=Path(
            os.environ.get(
                "S2S_OPERATIONAL_SCRIPT",
                ROOT.parent / "model-runs/fuxi/scripts/fuxi_operational.py",
            )
        ),
    )
    parser.add_argument(
        "--operational-python",
        default=os.environ.get("S2S_DRIVER_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--publisher-script",
        type=Path,
        default=ROOT / "scripts/publish_operational_site.py",
    )
    parser.add_argument(
        "--publish-clone",
        type=Path,
        default=os.environ.get("S2S_PUBLISH_CLONE"),
    )
    parser.add_argument("--clone-url", default=os.environ.get("S2S_REPOSITORY_URL"))
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--branch", default=os.environ.get("S2S_PUBLISH_BRANCH", "main")
    )
    parser.add_argument(
        "--dashboard-subdir",
        default=os.environ.get("S2S_DASHBOARD_SUBDIR", "clean/fuxi-dashboard"),
    )
    parser.add_argument("--expected-base-sha")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args()


def load_config(path: Path, source: str, issue: str) -> dict[str, Any]:
    """Validate the immutable 100-member config before any Git operation."""

    if len(issue) != 8 or not issue.isdigit():
        raise ValueError("issue must use YYYYMMDD")
    payload = json.loads(path.read_text(encoding="utf-8"))
    configured_source = (
        "gfs"
        if payload.get("input", {}).get("builder") == "gfs_daily_proxy"
        else "era5"
    )
    if configured_source != source:
        raise ValueError(f"config source is {configured_source}, expected {source}")
    if int(payload.get("members", 0)) != 100:
        raise ValueError("only validated 100-member runs may be published")
    if issue not in str(payload.get("run_label", "")):
        raise ValueError("config run label does not match the requested issue")
    return payload


def publisher_command(
    args: argparse.Namespace,
    source_dashboard: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.publisher_script),
        "--source",
        args.source,
        "--issue",
        args.issue,
        "--source-worktree",
        str(source_dashboard),
        "--publish-clone",
        str(args.publish_clone),
        "--remote",
        args.remote,
        "--branch",
        args.branch,
        "--dashboard-subdir",
        args.dashboard_subdir,
        "--receipt",
        str(args.receipt),
    ]
    if args.clone_url:
        command.extend(["--clone-url", args.clone_url])
    if args.skip_install:
        command.append("--skip-install")
    if args.skip_build:
        command.append("--skip-build")
    if args.no_push:
        command.append("--no-push")
    return command


def main() -> int:
    args = parse_args()
    load_config(args.config, args.source, args.issue)
    if args.timeout < 1:
        raise ValueError("timeout must be positive")
    if args.publish_clone is None:
        raise ValueError("--publish-clone or S2S_PUBLISH_CLONE is required")

    clone_args = SimpleNamespace(
        publish_clone=args.publish_clone,
        clone_url=args.clone_url,
        remote=args.remote,
        branch=args.branch,
        expected_base_sha=args.expected_base_sha,
    )
    publish_root, base_sha = prepare_clone(clone_args)
    export_worktree = create_disposable_worktree(publish_root, base_sha)
    try:
        dashboard_subdir = Path(args.dashboard_subdir)
        if dashboard_subdir.is_absolute() or ".." in dashboard_subdir.parts:
            raise ValueError("dashboard subdirectory must stay inside the worktree")
        export_dashboard = (export_worktree / dashboard_subdir).resolve()
        if not (export_dashboard / "package.json").is_file():
            raise ValueError(f"dashboard is absent from export worktree: {export_dashboard}")
        environment = os.environ.copy()
        environment["S2S_DASHBOARD_ROOT"] = str(export_dashboard)
        subprocess.run(
            [
                args.operational_python,
                str(args.operational_script),
                "publish",
                "--configs",
                str(args.config.resolve()),
                "--date",
                args.issue,
            ],
            cwd=export_dashboard,
            env=environment,
            check=True,
            timeout=args.timeout,
        )
        subprocess.run(
            publisher_command(args, export_dashboard),
            cwd=ROOT,
            check=True,
            timeout=args.timeout,
        )
    finally:
        remove_disposable_worktree(publish_root, export_worktree)

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("source") != args.source or receipt.get("issue") != args.issue:
        raise ValueError("publisher receipt identity mismatch")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
        raise SystemExit(f"operational export/publication failed: {error}") from error
