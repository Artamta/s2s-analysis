import json
from pathlib import Path

import pytest
import torch

from fuxi_adapter.artifacts import (
    FROZEN_RUN_ARTIFACTS,
    freeze_development_run,
    load_unused_freeze,
    mark_test_consumed,
)


def _config():
    return {
        "experiment_name": "x",
        "archive_root": "/a",
        "output_root": "/b",
        "train_years": [2020],
        "validation_years": [2021],
        "test_years": [2022],
        "models": ["m"],
        "seeds": [1],
    }


def _complete_run(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    torch.save({"value": 1}, run / "checkpoints" / "best.pt")
    (run / "SUCCESS.json").write_text(
        json.dumps({"status": "success", "model": "m", "seed": 1}) + "\n",
        encoding="utf-8",
    )
    (run / "normalization.json").write_text('{"scale": 1.0}\n', encoding="utf-8")
    (run / "resolved_config.json").write_text(
        json.dumps(_config(), sort_keys=True) + "\n", encoding="utf-8"
    )
    (run / "source_manifest.json").write_text(
        '{"source": "synthetic"}\n', encoding="utf-8"
    )
    (run / "source_snapshot.zip").write_bytes(b"synthetic source snapshot")
    return run


def test_freeze_is_one_shot(tmp_path):
    run = _complete_run(tmp_path)
    freeze = tmp_path / "freeze.json"
    manifest = freeze_development_run(freeze, _config(), [run])
    assert manifest["schema_version"] == 2
    assert not load_unused_freeze(freeze)["test_evaluated"]
    for name in FROZEN_RUN_ARTIFACTS:
        assert name in manifest["runs"][0]
        assert f"{name}_sha256" in manifest["runs"][0]
    result = tmp_path / "test-result"
    result.mkdir()
    mark_test_consumed(freeze, result)
    with pytest.raises(RuntimeError, match="already consumed"):
        load_unused_freeze(freeze)


@pytest.mark.parametrize("artifact", sorted(FROZEN_RUN_ARTIFACTS))
def test_freeze_detects_every_changed_run_artifact(tmp_path, artifact):
    run = _complete_run(tmp_path)
    freeze = tmp_path / "freeze.json"
    manifest = freeze_development_run(freeze, _config(), [run])
    path = Path(manifest["runs"][0][artifact])
    with path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(RuntimeError, match=rf"{artifact} changed after freeze"):
        load_unused_freeze(freeze)


def test_freeze_requires_preprocessing_and_provenance_files(tmp_path):
    run = _complete_run(tmp_path)
    (run / "normalization.json").unlink()

    with pytest.raises(FileNotFoundError, match="normalization.json"):
        freeze_development_run(tmp_path / "freeze.json", _config(), [run])


def test_freeze_rejects_run_from_a_different_configuration(tmp_path):
    run = _complete_run(tmp_path)
    other = _config()
    other["train_years"] = [2019]
    (run / "resolved_config.json").write_text(
        json.dumps(other, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match freeze configuration"):
        freeze_development_run(tmp_path / "freeze.json", _config(), [run])
