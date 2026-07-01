"""Shared paths for paper-build scripts.

Defaults are repository-relative so the scripts work after cloning. Local
machines can override heavy generated-output locations with environment
variables instead of editing source files.
"""

from __future__ import annotations

import os
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parent


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


PAPER_OUTPUT_ROOT = env_path(
    "S2S_PAPER_OUTPUT_ROOT",
    REPO_ROOT / "final_paper" / "outputs" / "s2s_paper_outputs",
)
IMD_MASK_025 = env_path(
    "S2S_IMD_MASK_025",
    REPO_ROOT / "final_paper" / "masks" / "imd_region_masks_0.25deg.nc",
)
