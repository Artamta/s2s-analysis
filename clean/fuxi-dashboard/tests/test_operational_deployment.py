"""Focused tests for safe catalog validation and static-site publication."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.postdeploy_smoke as postdeploy
from scripts.publish_operational_site import allowed_change
from scripts.publish_operational_site import (
    commit_environment,
    create_disposable_worktree,
    remove_disposable_worktree,
)
from scripts.validate_operational_publication import (
    choose_scope,
    issue_records,
    public_path,
    source_map,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_checked_in_catalog_passes_issue_scoped_live_inventory() -> None:
    args = SimpleNamespace(
        public_root=ROOT / "public",
        scope="issue",
        source="gfs",
        issue="20260805",
        expected_gfs_issue="20260805",
        expected_era5_issue="20260722",
        expected_forecast_sha256=None,
        live_inventory=True,
    )
    receipt = validate(args)
    assert receipt["status"] == "validated"
    assert receipt["issues_validated"] == ["gfs/20260805"]
    assert receipt["catalog_schema"] >= 3


def test_current_pointer_cannot_select_limited_ensemble() -> None:
    index = {
        "current": {"source_id": "gfs", "issue_id": "20260801"},
        "latest_reference": {"source_id": "era5", "issue_id": "20260722"},
        "initial_condition_sources": [
            {
                "id": "gfs",
                "default_issue": "20260801",
                "issues": [
                    {
                        "id": "20260801",
                        "initialization": "2026-08-01T00:00:00Z",
                        "members": 5,
                        "forecast": "forecasts/gfs/20260801.json",
                    }
                ],
            },
            {
                "id": "era5",
                "default_issue": "20260722",
                "issues": [
                    {
                        "id": "20260722",
                        "initialization": "2026-07-22T00:00:00Z",
                        "members": 100,
                        "forecast": "forecasts/era5/20260722.json",
                    }
                ],
            },
        ],
    }
    sources = source_map(index)
    records = issue_records(sources)
    args = SimpleNamespace(
        scope="current",
        source=None,
        issue=None,
        expected_gfs_issue=None,
        expected_era5_issue=None,
    )
    with pytest.raises(ValueError, match="current GFS issue must be a 100-member"):
        choose_scope(args, index, sources, records)


def test_public_path_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe public path"):
        public_path(tmp_path, "../private/config.json")


@pytest.mark.parametrize(
    ("relative", "accepted"),
    [
        ("clean/fuxi-dashboard/public/data/index.json", True),
        ("clean/fuxi-dashboard/public/data/forecasts/gfs/20260805.json", True),
        ("clean/fuxi-dashboard/public/data/forecasts/era5/20260805.json", True),
        ("clean/fuxi-dashboard/public/data/regional/gfs/20260805.json", True),
        ("clean/fuxi-dashboard/public/downloads/gfs/20260805/brief.pdf", True),
        ("clean/fuxi-dashboard/src/main.ts", False),
        ("clean/config/operational/private.json", False),
        ("README.md", False),
    ],
)
def test_publisher_git_allowlist(relative: str, accepted: bool) -> None:
    assert (
        allowed_change(relative, "gfs", "20260805", "clean/fuxi-dashboard")
        is accepted
    )


def test_failed_publication_worktree_cannot_dirty_publishing_clone(
    tmp_path: Path,
) -> None:
    clone = tmp_path / "publisher"
    clone.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=clone, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=clone,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Publication Test"],
        cwd=clone,
        check=True,
    )
    (clone / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=clone, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=clone, check=True)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    worktree = create_disposable_worktree(clone, base_sha)
    (worktree / "unexpected-private-file").write_text("blocked\n", encoding="utf-8")
    remove_disposable_worktree(clone, worktree)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=clone,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert not worktree.exists()
    assert status == ""


def test_publisher_requires_explicit_institutional_git_identity() -> None:
    with pytest.raises(ValueError, match="required"):
        commit_environment(None, None)
    with pytest.raises(ValueError, match="real institution-owned"):
        commit_environment("S2S Bot", "forecast-ops@institution.example")
    environment = commit_environment("S2S Bot", "forecast-ops@university.edu")
    assert environment["GIT_AUTHOR_NAME"] == "S2S Bot"
    assert environment["GIT_COMMITTER_EMAIL"] == "forecast-ops@university.edu"


def build_smoke_site(root: Path) -> None:
    (root / "data/forecasts/gfs").mkdir(parents=True)
    (root / "data/forecasts/era5").mkdir(parents=True)
    (root / "index.html").write_text('<main id="app"></main>\n', encoding="utf-8")
    index = {
        "current": {"source_id": "gfs", "issue_id": "20260805"},
        "latest_reference": {"source_id": "era5", "issue_id": "20260722"},
        "initial_condition_sources": [],
    }
    for source, issue in (("gfs", "20260805"), ("era5", "20260722")):
        forecast_path = f"forecasts/{source}/{issue}.json"
        index["initial_condition_sources"].append(
            {
                "id": source,
                "issues": [
                    {
                        "id": issue,
                        "members": 100,
                        "forecast": forecast_path,
                    }
                ],
            }
        )
        forecast = {
            "issue": {
                "initialization": f"{issue[:4]}-{issue[4:6]}-{issue[6:]}T00:00:00Z",
                "members": 100,
                "initial_condition_source": {"id": source},
            }
        }
        (root / "data" / forecast_path).write_text(
            json.dumps(forecast), encoding="utf-8"
        )
    (root / "data/index.json").write_text(json.dumps(index), encoding="utf-8")
    entries = []
    for path in sorted((root / "data").rglob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(payload),
                "sha256": digest(payload),
            }
        )
    manifest = {
        "schema_version": 1,
        "deployed_commit": "abc123",
        "files": entries,
    }
    (root / "data/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_postdeploy_smoke_verifies_commit_pointers_and_forecasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_smoke_site(tmp_path)
    monkeypatch.setattr(
        postdeploy,
        "fetch",
        lambda _base, relative, _attempts, _timeout: (tmp_path / relative).read_bytes(),
    )
    args = SimpleNamespace(
        base_url="https://example.invalid/s2s/",
        expected_commit="abc123",
        expected_index=tmp_path / "data/index.json",
        expected_gfs_issue=None,
        expected_era5_issue=None,
        attempts=1,
        timeout=2.0,
    )
    receipt = postdeploy.smoke(args)
    assert receipt["status"] == "healthy"
    assert receipt["current_gfs_issue"] == "20260805"
    assert receipt["latest_era5_issue"] == "20260722"
    assert len(receipt["verified_assets"]) == 2


def test_postdeploy_fetch_rejects_non_http_origin() -> None:
    with pytest.raises(ValueError, match="must be HTTP"):
        postdeploy.fetch("file:///tmp/site", "data/index.json", 1, 1.0)


def test_postdeploy_retries_whole_snapshot_during_cdn_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def eventually_healthy(_args: SimpleNamespace) -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("old catalog still cached")
        return {"status": "healthy"}

    monkeypatch.setattr(postdeploy, "smoke_once", eventually_healthy)
    monkeypatch.setattr(postdeploy.time, "sleep", lambda _seconds: None)
    receipt = postdeploy.smoke(SimpleNamespace(attempts=3))
    assert receipt == {"status": "healthy"}
    assert calls == 3


def test_root_workflow_template_runs_all_release_gates() -> None:
    workflow = (ROOT / "deployment/github/deploy-pages.yml").read_text(
        encoding="utf-8"
    )
    assert "clean/fuxi-dashboard" in workflow
    assert "validate_operational_publication.py --scope all" in workflow
    assert "validate_global_web_data.py" in workflow
    assert "npm run build" in workflow
    assert "postdeploy_smoke.py" in workflow
    assert "--expected-commit" in workflow
