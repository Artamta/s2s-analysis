import json

import pytest

from fuxi_adapter.config import config_sha256, load_config


def test_config_hash_is_order_independent(tmp_path):
    payload = {
        "experiment_name": "x",
        "archive_root": "/a",
        "output_root": "/b",
        "train_years": [2020],
        "validation_years": [2021],
        "test_years": [2022],
        "models": ["m"],
        "seeds": [1],
        "verification_start_offset_days": 0,
        "verification_day_count": 42,
        "verification_interval_convention": "start_inclusive_end_exclusive",
        "non_overlapping_split_targets": True,
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload), encoding="utf-8")
    second.write_text(json.dumps(dict(reversed(list(payload.items())))), encoding="utf-8")
    assert config_sha256(load_config(first)) == config_sha256(load_config(second))


def test_year_overlap_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "experiment_name": "x",
                "archive_root": "/a",
                "output_root": "/b",
                "train_years": [2020],
                "validation_years": [2020],
                "test_years": [2022],
                "models": ["m"],
                "seeds": [1],
                "verification_start_offset_days": 0,
                "verification_day_count": 42,
                "verification_interval_convention": "start_inclusive_end_exclusive",
                "non_overlapping_split_targets": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_config(path)


def test_shifted_truth_window_is_rejected(tmp_path):
    path = tmp_path / "shifted.json"
    path.write_text(
        json.dumps(
            {
                "experiment_name": "x",
                "archive_root": "/a",
                "output_root": "/b",
                "train_years": [2020],
                "validation_years": [2021],
                "test_years": [2022],
                "models": ["m"],
                "seeds": [1],
                "verification_start_offset_days": 1,
                "verification_day_count": 42,
                "verification_interval_convention": "start_inclusive_end_exclusive",
                "non_overlapping_split_targets": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="verification_start_offset_days"):
        load_config(path)
