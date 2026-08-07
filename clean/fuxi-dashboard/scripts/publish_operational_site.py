#!/usr/bin/env python3
"""Publish one generated issue through a dedicated, clean Git clone.

This helper never stages the scientific working tree.  It fast-forwards a
dedicated publishing clone from ``origin/main``, copies only issue-scoped public
artifacts, rebuilds the checksum inventory, validates the complete archive,
builds the site, creates one scoped commit, and performs a normal (non-force)
push.  ``--dry-run`` is side-effect free with respect to Git and is suitable
for scheduler rehearsals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("gfs", "era5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--issue", required=True, help="Initialization as YYYYMMDD.")
    parser.add_argument(
        "--source-worktree",
        type=Path,
        default=ROOT,
        help="Dashboard tree containing newly generated public artifacts.",
    )
    parser.add_argument(
        "--publish-clone",
        type=Path,
        default=os.environ.get("S2S_PUBLISH_CLONE"),
        help="Dedicated clean clone (or S2S_PUBLISH_CLONE).",
    )
    parser.add_argument(
        "--clone-url", default=os.environ.get("S2S_REPOSITORY_URL")
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--branch", default=os.environ.get("S2S_PUBLISH_BRANCH", "main")
    )
    parser.add_argument(
        "--dashboard-subdir",
        default=os.environ.get("S2S_DASHBOARD_SUBDIR", "clean/fuxi-dashboard"),
        help="Dashboard path inside the repository clone; use '.' for a standalone repo.",
    )
    parser.add_argument("--expected-base-sha")
    parser.add_argument("--expected-forecast-sha256")
    parser.add_argument("--commit-message")
    parser.add_argument(
        "--git-author-name", default=os.environ.get("S2S_GIT_AUTHOR_NAME")
    )
    parser.add_argument(
        "--git-author-email", default=os.environ.get("S2S_GIT_AUTHOR_EMAIL")
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip npm ci when the publishing clone already has exact dependencies.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Create the scoped commit but leave it in the publishing clone.",
    )
    return parser.parse_args()


def run(
    command: Sequence[str],
    *,
    cwd: Path,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        # Keep stdout reserved for the final machine-readable receipt. Build
        # and Git progress remain visible on stderr for service logs.
        stdout=subprocess.PIPE if capture else sys.stderr,
        stderr=subprocess.PIPE if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value!r} in {path}")
        ),
    )


def validate_issue_identity(source_root: Path, source: str, issue: str) -> tuple[Path, str]:
    if len(issue) != 8 or not issue.isdigit():
        raise ValueError("issue must be YYYYMMDD")
    forecast = source_root / "public/data/forecasts" / source / f"{issue}.json"
    payload = load_json(forecast)
    issue_metadata = payload.get("issue", {})
    source_id = issue_metadata.get("initial_condition_source", {}).get("id")
    initialization = str(issue_metadata.get("initialization", ""))
    if source_id != source or initialization[:10].replace("-", "") != issue:
        raise ValueError(f"forecast identity mismatch in {forecast}")
    if issue_metadata.get("members") != 100:
        raise ValueError("operational publisher accepts only 100-member issues")
    return forecast, sha256(forecast)


def candidate_paths(source_root: Path, source: str, issue: str) -> list[Path]:
    """Return existing generated paths that may cross the publication boundary."""

    relative_candidates = [
        Path("public/data/index.json"),
        Path("public/data/validation.json"),
        Path("public/data/ops/status.json"),
        Path("public/data/operations.json"),
        Path(f"public/data/forecasts/{source}/{issue}.json"),
        Path(f"public/data/regional/{source}/{issue}.json"),
        Path(f"public/data/comparisons/{issue}.json"),
        Path(f"public/downloads/{source}/{issue}"),
    ]
    # A delayed ERA5 publication may add initialization-sensitivity metadata to
    # the already-published same-date GFS package (and conversely in rehearsal).
    for counterpart in SOURCES:
        if counterpart != source:
            relative_candidates.append(
                Path(f"public/data/forecasts/{counterpart}/{issue}.json")
            )
    return [path for path in relative_candidates if (source_root / path).exists()]


def allowed_change(
    relative: str, source: str, issue: str, dashboard_subdir: str
) -> bool:
    prefix = PurePosixPath(dashboard_subdir)
    path = PurePosixPath(relative)
    if str(prefix) not in ("", "."):
        if prefix not in path.parents:
            return False
        path = path.relative_to(prefix)
    normalized = path.as_posix()
    exact = {
        "public/data/index.json",
        "public/data/manifest.json",
        "public/data/validation.json",
        "public/data/ops/status.json",
        "public/data/operations.json",
        f"public/data/forecasts/gfs/{issue}.json",
        f"public/data/forecasts/era5/{issue}.json",
        f"public/data/regional/{source}/{issue}.json",
        f"public/data/comparisons/{issue}.json",
    }
    if normalized in exact:
        return True
    download_prefix = PurePosixPath(f"public/downloads/{source}/{issue}")
    return path != download_prefix and download_prefix in path.parents


def allowed_retention_change(
    relative: str,
    dashboard_subdir: str,
    now: datetime,
) -> bool:
    """Allow only policy-expired forecast/PDF paths changed by retention."""

    prefix = PurePosixPath(dashboard_subdir)
    path = PurePosixPath(relative)
    if str(prefix) not in ("", "."):
        if prefix not in path.parents:
            return False
        path = path.relative_to(prefix)
    normalized = path.as_posix()
    patterns = (
        (r"^public/data/forecasts/(?:gfs|era5)/(\d{8})\.json$", 56),
        (r"^public/data/regional/(?:gfs|era5)/(\d{8})\.json$", 365),
        (r"^public/downloads/(?:gfs|era5)/(\d{8})/.+\.pdf$", 56),
    )
    for pattern, days in patterns:
        match = re.fullmatch(pattern, normalized)
        if not match:
            continue
        issue_day = datetime.strptime(match.group(1), "%Y%m%d").replace(
            tzinfo=timezone.utc
        )
        return now - issue_day > timedelta(days=days)
    return False


def copy_candidates(source_root: Path, publish_root: Path, paths: list[Path]) -> None:
    for relative in paths:
        source = source_root / relative
        destination = publish_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            for child in sorted(source.rglob("*")):
                if child.is_file():
                    target = destination / child.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, target)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def porcelain_paths(publish_root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=publish_root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    paths: list[str] = []
    fields = output.split(b"\0")
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        text = field.decode("utf-8", errors="strict")
        status = text[:2]
        relative = text[3:]
        if "R" in status or "C" in status:
            if index >= len(fields):
                raise ValueError("truncated Git rename/copy status")
            relative = fields[index].decode("utf-8", errors="strict")
            index += 1
        paths.append(relative)
    return paths


def write_receipt(path: Path | None, receipt: dict[str, Any]) -> None:
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(serialized, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def commit_environment(name: str | None, email: str | None) -> dict[str, str]:
    """Require an institution-owned public Git identity for automated commits."""

    if not name or not email:
        raise ValueError(
            "S2S_GIT_AUTHOR_NAME and S2S_GIT_AUTHOR_EMAIL are required for publication"
        )
    if "@" not in email or email.lower().endswith(".example"):
        raise ValueError("Git author email must be a real institution-owned address")
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
    )
    return environment


def prepare_clone(args: argparse.Namespace) -> tuple[Path, str]:
    if args.publish_clone is None:
        raise ValueError("--publish-clone or S2S_PUBLISH_CLONE is required")
    publish_root = args.publish_clone.resolve()
    if not publish_root.exists():
        if not args.clone_url:
            raise ValueError("new publishing clone requires --clone-url")
        publish_root.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--branch",
                args.branch,
                "--single-branch",
                args.clone_url,
                str(publish_root),
            ],
            cwd=publish_root.parent,
        )
    if not (publish_root / ".git").exists():
        raise ValueError(f"publishing path is not a dedicated Git clone: {publish_root}")
    if args.clone_url:
        configured_url = run(
            ["git", "remote", "get-url", args.remote],
            cwd=publish_root,
            capture=True,
        )
        if configured_url != args.clone_url:
            raise ValueError(
                f"publishing clone remote is {configured_url!r}, expected configured "
                f"repository {args.clone_url!r}"
            )
    dirty = run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=publish_root,
        capture=True,
    )
    if dirty:
        raise ValueError("publishing clone is not clean; refusing to overwrite it")
    run(["git", "fetch", "--prune", args.remote, args.branch], cwd=publish_root)
    run(["git", "switch", args.branch], cwd=publish_root)
    run(["git", "merge", "--ff-only", f"{args.remote}/{args.branch}"], cwd=publish_root)
    base_sha = run(["git", "rev-parse", "HEAD"], cwd=publish_root, capture=True)
    if args.expected_base_sha and base_sha != args.expected_base_sha:
        raise ValueError(
            f"publishing base is {base_sha}, expected {args.expected_base_sha}"
        )
    return publish_root, base_sha


def create_disposable_worktree(publish_root: Path, base_sha: str) -> Path:
    """Create an exact-base worktree so a failed gate cannot dirty the clone."""

    worktree = Path(
        tempfile.mkdtemp(prefix="s2s-publish-", dir=str(publish_root.parent))
    )
    worktree.rmdir()
    run(
        ["git", "worktree", "add", "--detach", str(worktree), base_sha],
        cwd=publish_root,
    )
    return worktree


def remove_disposable_worktree(publish_root: Path, worktree: Path) -> None:
    if worktree.exists():
        run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=publish_root,
        )
    run(["git", "worktree", "prune"], cwd=publish_root)


def main() -> int:
    args = parse_args()
    source_root = args.source_worktree.resolve()
    forecast_path, forecast_sha = validate_issue_identity(
        source_root, args.source, args.issue
    )
    if (
        args.expected_forecast_sha256
        and forecast_sha != args.expected_forecast_sha256.lower()
    ):
        raise SystemExit(
            f"forecast checksum mismatch: expected {args.expected_forecast_sha256}, "
            f"found {forecast_sha}"
        )
    candidates = candidate_paths(source_root, args.source, args.issue)
    if Path(f"public/data/forecasts/{args.source}/{args.issue}.json") not in candidates:
        raise SystemExit(f"forecast is missing from publication candidates: {forecast_path}")

    if args.dry_run:
        run(
            [
                sys.executable,
                "scripts/validate_operational_publication.py",
                "--scope",
                "issue",
                "--source",
                args.source,
                "--issue",
                args.issue,
                "--expected-forecast-sha256",
                forecast_sha,
                "--live-inventory",
            ],
            cwd=source_root,
        )
        if not args.skip_build:
            run(["npm", "run", "build"], cwd=source_root)
        receipt = {
            "status": "dry-run",
            "source": args.source,
            "issue": args.issue,
            "forecast_sha256": forecast_sha,
            "candidate_paths": [path.as_posix() for path in candidates],
            "git_mutated": False,
        }
        write_receipt(args.receipt, receipt)
        return 0

    git_environment = commit_environment(args.git_author_name, args.git_author_email)
    publish_root, base_sha = prepare_clone(args)
    dashboard_subdir = Path(args.dashboard_subdir)
    if dashboard_subdir.is_absolute() or ".." in dashboard_subdir.parts:
        raise ValueError("dashboard subdirectory must stay inside the publishing clone")
    worktree = create_disposable_worktree(publish_root, base_sha)
    local_branch: str | None = None
    try:
        publish_dashboard = (worktree / dashboard_subdir).resolve()
        worktree_root = worktree.resolve()
        if (
            publish_dashboard != worktree_root
            and worktree_root not in publish_dashboard.parents
        ):
            raise ValueError("dashboard subdirectory resolves outside the publishing worktree")
        if not (publish_dashboard / "package.json").is_file():
            raise ValueError(f"dashboard is absent from publishing clone: {publish_dashboard}")
        copy_candidates(source_root, publish_dashboard, candidates)
        run(
            [sys.executable, "scripts/apply_archive_retention.py"],
            cwd=publish_dashboard,
        )
        run(
            [sys.executable, "scripts/stamp_deploy.py", "--commit", "pending-ci"],
            cwd=publish_dashboard,
        )
        run(
            [
                sys.executable,
                "scripts/validate_operational_publication.py",
                "--scope",
                "all",
            ],
            cwd=publish_dashboard,
        )
        if not args.skip_build:
            if not args.skip_install:
                run(["npm", "ci"], cwd=publish_dashboard)
            run(["npm", "run", "build"], cwd=publish_dashboard)

        changed = porcelain_paths(worktree)
        rejected = [
            relative
            for relative in changed
            if not allowed_change(
                relative, args.source, args.issue, args.dashboard_subdir
            )
            and not allowed_retention_change(
                relative,
                args.dashboard_subdir,
                datetime.now(timezone.utc),
            )
        ]
        if rejected:
            raise ValueError(
                "publishing worktree contains non-allowlisted changes: "
                + ", ".join(rejected)
            )
        if not changed:
            raise ValueError("publication produced no Git changes")
        run(["git", "add", "--", *changed], cwd=worktree)
        message = (
            args.commit_message
            or f"Publish {args.source.upper()} FuXi issue {args.issue}"
        )
        run(
            ["git", "commit", "-m", message],
            cwd=worktree,
            env=git_environment,
        )
        commit_sha = run(["git", "rev-parse", "HEAD"], cwd=worktree, capture=True)
        if args.no_push:
            local_branch = (
                f"s2s-publish/{args.source}-{args.issue}-{commit_sha[:12]}"
            )
            run(["git", "branch", local_branch, commit_sha], cwd=publish_root)
        else:
            # A plain refspec is intentional: no force, no lease override.
            run(
                ["git", "push", args.remote, f"HEAD:refs/heads/{args.branch}"],
                cwd=worktree,
            )
    finally:
        remove_disposable_worktree(publish_root, worktree)
    receipt = {
        "status": "committed" if args.no_push else "pushed",
        "source": args.source,
        "issue": args.issue,
        "forecast_sha256": forecast_sha,
        "base_sha": base_sha,
        "commit_sha": commit_sha,
        "changed_paths": sorted(changed),
        "pushed": not args.no_push,
        "local_branch": local_branch,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_receipt(args.receipt, receipt)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(f"operational publication failed: {error}") from error
