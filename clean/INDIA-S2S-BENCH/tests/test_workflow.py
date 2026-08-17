from pathlib import Path

import numpy as np

from india_s2s_bench import workflow


def test_native_compact_dates_are_calendar_dates(tmp_path, monkeypatch):
    (tmp_path / "20250602.nc").touch()
    (tmp_path / "20250929.nc").touch()
    monkeypatch.setitem(workflow.NATIVE_2025, "example", tmp_path)
    assert workflow._native_dates("example") == {
        np.datetime64("2025-06-02"),
        np.datetime64("2025-09-29"),
    }


def test_protocol_is_frozen_and_neural_mismatch_is_explicit():
    root = Path(__file__).resolve().parents[1]
    protocol = workflow.load_protocol(root)
    assert protocol["status"] == "frozen_before_2025_skill_scoring"
    assert protocol["valid_date_alignment"]["week_1"].endswith("+ 7 days")
    assert protocol["neural_correction"]["current_status"].startswith("excluded")


def test_confirmatory_runner_refuses_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    try:
        workflow.run_experiment(tmp_path, output)
    except FileExistsError as error:
        assert "refusing existing output directory" in str(error)
    else:
        raise AssertionError("runner must not overwrite an existing confirmatory result")
