"""Small, explicit experiment configuration helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate one JSON experiment configuration."""

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "experiment_name",
        "archive_root",
        "output_root",
        "train_years",
        "validation_years",
        "test_years",
        "models",
        "seeds",
        "verification_start_offset_days",
        "verification_day_count",
        "verification_interval_convention",
        "non_overlapping_split_targets",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError("configuration is missing keys: " + ", ".join(missing))
    if set(config["train_years"]) & set(config["validation_years"]):
        raise ValueError("training and validation years overlap")
    if (set(config["train_years"]) | set(config["validation_years"])) & set(
        config["test_years"]
    ):
        raise ValueError("development and test years overlap")
    temporal_contract = {
        "verification_start_offset_days": 0,
        "verification_day_count": 42,
        "verification_interval_convention": "start_inclusive_end_exclusive",
        "non_overlapping_split_targets": True,
    }
    for key, expected in temporal_contract.items():
        if config[key] != expected:
            raise ValueError(
                f"{key} must be {expected!r} for the FuXi/IMERG interval contract"
            )
    config["config_path"] = str(path)
    return config


def canonical_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable subset used for run identity."""

    return {key: value for key, value in config.items() if key != "config_path"}


def config_sha256(config: Dict[str, Any]) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write readable JSON; callers create the parent intentionally."""

    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
