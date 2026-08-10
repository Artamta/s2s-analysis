"""Focused correctness tests for the global TP training path."""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

import train
from model import TPProbUNet


def test_rps_loss_matches_hand_calculation() -> None:
    # Forecast CDF: [0.1, 0.3, 0.6, 0.8, 1.0]
    # Category-2 observed CDF: [0, 0, 1, 1, 1]
    # RPS = .1^2 + .3^2 + (.6-1)^2 + (.8-1)^2 = 0.30.
    probabilities = torch.tensor(
        [[[[[0.1]], [[0.2]], [[0.3]], [[0.2]], [[0.2]]]]],
        dtype=torch.float32,
    )
    target = torch.tensor([[[[2]]]], dtype=torch.int64)
    weights = torch.ones(1, 1)

    score = train.rps_loss(probabilities, target, weights)

    torch.testing.assert_close(score, torch.tensor(0.30), rtol=0.0, atol=1.0e-7)


def test_rps_loss_masks_invalid_targets() -> None:
    valid_distribution = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
    ignored_distribution = torch.tensor([0.9, 0.025, 0.025, 0.025, 0.025])
    probabilities = torch.stack((valid_distribution, ignored_distribution), dim=-1)
    probabilities = probabilities.reshape(1, 1, 5, 1, 2)
    target = torch.tensor([[[[2, -1]]]], dtype=torch.int64)
    weights = torch.tensor([[1.0, 100.0]])

    score = train.rps_loss(probabilities, target, weights)

    # The invalid second target contributes neither score nor denominator,
    # despite its much larger spatial weight.
    torch.testing.assert_close(score, torch.tensor(0.30), rtol=0.0, atol=1.0e-7)


def test_optimizer_excludes_biases_and_norm_parameters_from_decay() -> None:
    model = TPProbUNet()
    optimizer = train.optimizer_for(model, learning_rate=3.0e-4, weight_decay=1.0e-4)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 1.0e-4
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    decay_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert decay_ids.isdisjoint(no_decay_ids)

    for name, parameter in model.named_parameters():
        expected = decay_ids if parameter.ndim >= 2 and not name.endswith("bias") else no_decay_ids
        assert id(parameter) in expected, name


def test_spatial_weights_are_nonnegative_at_both_poles() -> None:
    latitude = np.array([90.0, 45.0, 0.0, -45.0, -90.0], dtype=np.float32)
    land_fraction = np.ones((5, 3), dtype=np.float32)

    weights = train.spatial_weights(latitude, land_fraction)

    assert weights.shape == (5, 3)
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(weights[2], torch.ones(3))


def test_one_epoch_cpu_smoke_writes_complete_run(tmp_path, monkeypatch) -> None:
    run_directory = tmp_path / "smoke_run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--smoke",
            "--device",
            "cpu",
            "--run-dir",
            str(run_directory),
            "--max-epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--seed",
            "7",
        ],
    )

    train.main()

    checkpoint_path = run_directory / "best.pt"
    assert checkpoint_path.is_file()
    assert (run_directory / "history.csv").is_file()
    assert (run_directory / "figures" / "loss_curve.png").is_file()
    config_path = run_directory / "config.json"
    assert config_path.is_file()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(checkpoint) == {"model_state", "model_config", "normalization", "metadata"}
    assert checkpoint["model_config"] == {
        "in_channels": 18,
        "base_channels": 16,
        "dropout": 0.1,
    }
    assert checkpoint["metadata"]["best_epoch"] == 1
    assert checkpoint["metadata"]["fuxi_competition_use"] == "written_permission_required"

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["command"]["smoke"] is True
    assert saved_config["command"]["device"] == "cpu"
    assert saved_config["result"]["best_epoch"] == 1
