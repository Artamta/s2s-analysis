"""Focused integration tests for the anchored v3 training loop."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fuxi_adapter.v3_training import AnchoredSequenceDataset, train_anchored_model


def _synthetic_fields(case_count):
    leads, height, width = 6, 4, 4
    case = np.arange(case_count, dtype=np.float32)[:, None, None, None]
    lead = np.arange(leads, dtype=np.float32)[None, :, None, None]
    latitude = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, None, :, None]
    longitude = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, None, None, :]
    spatial_pattern = latitude + 0.5 * longitude

    baseline = np.broadcast_to(
        2.0 + 0.05 * case + 0.1 * lead + 0.2 * spatial_pattern,
        (case_count, leads, height, width),
    ).copy()
    log_residual = np.broadcast_to(
        0.04 + 0.01 * lead + 0.02 * spatial_pattern,
        baseline.shape,
    ).copy()
    truth = np.expm1(np.log1p(baseline) + log_residual)
    climatology = np.broadcast_to(
        1.0 + 0.1 * spatial_pattern,
        baseline.shape,
    ).copy()
    lead_feature = np.broadcast_to(lead / (leads - 1), baseline.shape)
    spatial_feature = np.broadcast_to(spatial_pattern, baseline.shape)
    features = np.stack((spatial_feature, lead_feature), axis=2)
    valid_mask = np.ones_like(baseline, dtype=bool)
    return {
        "features": features.astype(np.float32),
        "target": log_residual.astype(np.float32),
        "bias_baseline": baseline.astype(np.float32),
        "truth": truth.astype(np.float32),
        "climatology": climatology.astype(np.float32),
        "valid_mask": valid_mask,
    }


def test_anchored_sequence_dataset_returns_validated_tensors():
    arrays = _synthetic_fields(case_count=3)
    dataset = AnchoredSequenceDataset(**arrays)

    sample = dataset[1]

    assert len(dataset) == 3
    assert len(sample) == 6
    assert sample[0].shape == (6, 2, 4, 4)
    assert all(value.shape == (6, 4, 4) for value in sample[1:])
    assert sample[0].dtype == torch.float32
    assert sample[-1].dtype == torch.bool


def test_anchored_sequence_dataset_rejects_mismatched_field_shape():
    arrays = _synthetic_fields(case_count=2)
    arrays["target"] = arrays["target"][:, :5]

    with pytest.raises(ValueError, match="target shape .* does not match"):
        AnchoredSequenceDataset(**arrays)


@pytest.mark.parametrize("field", ["features", "target"])
def test_anchored_sequence_dataset_rejects_nonfinite_learning_arrays(field):
    arrays = _synthetic_fields(case_count=2)
    arrays[field].flat[0] = np.nan

    with pytest.raises(ValueError, match="features and target must be finite"):
        AnchoredSequenceDataset(**arrays)


@pytest.mark.parametrize("field", ["bias_baseline", "truth", "climatology"])
@pytest.mark.parametrize("invalid_value", [-1.0, np.nan])
def test_anchored_sequence_dataset_rejects_invalid_physical_fields_on_support(
    field, invalid_value
):
    arrays = _synthetic_fields(case_count=2)
    arrays[field].flat[0] = invalid_value

    with pytest.raises(ValueError, match=f"{field} must be finite and nonnegative"):
        AnchoredSequenceDataset(**arrays)


def test_anchored_sequence_dataset_allows_invalid_physical_values_off_support():
    arrays = _synthetic_fields(case_count=2)
    arrays["valid_mask"].flat[0] = False
    arrays["bias_baseline"].flat[0] = np.nan
    arrays["truth"].flat[0] = -1.0
    arrays["climatology"].flat[0] = np.inf

    dataset = AnchoredSequenceDataset(**arrays)

    assert len(dataset) == 2
    assert not dataset.valid_mask.flatten()[0]


class _TinyLateLeadModel(nn.Module):
    """Fast integration-test model with the same inactive-lead contract."""

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(2, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features):
        batch, leads, channels, height, width = features.shape
        residual = self.head(
            features.reshape(batch * leads, channels, height, width)
        )[:, 0].reshape(batch, leads, height, width)
        return torch.cat((torch.zeros_like(residual[:, :2]), residual[:, 2:]), dim=1)


def test_tiny_cpu_anchored_training_writes_best_checkpoint_and_history(tmp_path):
    train_dataset = AnchoredSequenceDataset(**_synthetic_fields(case_count=4))
    validation_dataset = AnchoredSequenceDataset(**_synthetic_fields(case_count=2))
    model = _TinyLateLeadModel()
    run_directory = tmp_path / "run"
    (run_directory / "checkpoints").mkdir(parents=True)
    (run_directory / "logs").mkdir()
    area_weights = np.array(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )

    result = train_anchored_model(
        model,
        train_dataset,
        validation_dataset,
        area_weights=area_weights,
        target_scale=np.ones(6, dtype=np.float32),
        lead_weights=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        loss_coefficients={"smooth_l1": 0.6, "acc": 0.3, "bias": 0.1},
        run_directory=run_directory,
        seed=17,
        device="cpu",
        batch_size=2,
        max_epochs=2,
        patience=2,
        learning_rate=1.0e-2,
        num_workers=0,
        use_amp=False,
    )

    expected_columns = {
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "train_smooth_l1",
        "train_acc_loss",
        "train_mean_spatial_acc",
        "train_mean_bias_squared",
        "validation_smooth_l1",
        "validation_acc_loss",
        "validation_mean_spatial_acc",
        "validation_mean_bias_squared",
    }
    assert len(result.history) == 2
    assert expected_columns.issubset(result.history.columns)
    assert np.isfinite(result.history[list(expected_columns)].to_numpy()).all()
    assert result.best_epoch in {0, 1}
    assert np.isfinite(result.best_validation_loss)
    assert result.elapsed_seconds >= 0.0

    checkpoint_path = run_directory / "checkpoints" / "best.pt"
    history_path = run_directory / "logs" / "training_history.csv"
    event_path = run_directory / "logs" / "events.jsonl"
    assert checkpoint_path.is_file()
    assert history_path.is_file()
    assert event_path.is_file()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["best_epoch"] == result.best_epoch
    assert checkpoint["best_validation_loss"] == pytest.approx(
        result.best_validation_loss
    )
    assert np.array_equal(
        checkpoint["lead_weights"],
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    assert all(
        torch.equal(model.state_dict()[name].cpu(), value.cpu())
        for name, value in checkpoint["model_state_dict"].items()
    )

    saved_history = pd.read_csv(history_path)
    assert list(saved_history.columns) == list(result.history.columns)
    assert len(saved_history) == len(result.history)
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert events[0]["event"] == "anchored_training_started"
    assert sum(event["event"] == "anchored_epoch_completed" for event in events) == 2
    assert events[-1]["event"] == "anchored_training_completed"

