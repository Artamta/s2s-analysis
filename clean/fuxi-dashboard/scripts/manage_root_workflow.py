#!/usr/bin/env python3
"""Check or install the dashboard workflow at the Git repository root.

GitHub only discovers workflows in the repository-root ``.github`` directory;
the dashboard itself lives below ``clean/fuxi-dashboard``.  Checking is the
default and never writes.  Installation is explicit and uses an atomic replace.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "deployment/github/deploy-pages.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Verify exact installation.")
    action.add_argument("--install", action="store_true", help="Install the root workflow.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a different existing workflow (only with --install).",
    )
    return parser.parse_args()


def repository_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return Path(completed.stdout.strip()).resolve()


def main() -> int:
    args = parse_args()
    if args.force and not args.install:
        raise SystemExit("--force is only valid with --install")
    destination = repository_root() / ".github/workflows/deploy-fuxi-dashboard.yml"
    expected = TEMPLATE.read_bytes()
    installed = destination.read_bytes() if destination.is_file() else None
    if not args.install:
        if installed != expected:
            raise SystemExit(
                f"root workflow is not installed from the checked template: {destination}"
            )
        print(f"root workflow: current ({destination})")
        return 0
    if installed is not None and installed != expected and not args.force:
        raise SystemExit(
            f"a different root workflow exists at {destination}; inspect it, then use "
            "--install --force only if replacement is intended"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".yml.tmp")
    temporary.write_bytes(expected)
    os.replace(temporary, destination)
    print(f"installed root workflow: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
