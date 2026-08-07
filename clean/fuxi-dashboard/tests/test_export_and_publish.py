"""Small contract tests for the clean operational export bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.export_and_publish_operational_issue import (
    load_config,
    publisher_command,
)


def write_config(path: Path, *, source: str = "gfs", members: int = 100) -> None:
    path.write_text(
        json.dumps(
            {
                "members": members,
                "run_label": "fuxi_s2s_case_20260808_ens100",
                "input": {"builder": "gfs_daily_proxy"} if source == "gfs" else {},
            }
        ),
        encoding="utf-8",
    )


def test_private_config_identity_is_bound_before_export(tmp_path: Path) -> None:
    config = tmp_path / "case.json"
    write_config(config)
    assert load_config(config, "gfs", "20260808")["members"] == 100
    with pytest.raises(ValueError, match="source"):
        load_config(config, "era5", "20260808")
    write_config(config, members=5)
    with pytest.raises(ValueError, match="100-member"):
        load_config(config, "gfs", "20260808")


def test_publisher_command_carries_only_explicit_paths(tmp_path: Path) -> None:
    args = argparse.Namespace(
        publisher_script=Path("/opt/s2s/publish.py"),
        source="gfs",
        issue="20260808",
        publish_clone=Path("/srv/s2s/publisher"),
        remote="origin",
        branch="main",
        dashboard_subdir="clean/fuxi-dashboard",
        receipt=tmp_path / "receipt.json",
        clone_url=None,
        skip_install=False,
        skip_build=False,
        no_push=False,
    )
    command = publisher_command(args, tmp_path / "export")
    assert "--source-worktree" in command
    assert str(tmp_path / "export") in command
    assert "--receipt" in command
    assert "--no-push" not in command
