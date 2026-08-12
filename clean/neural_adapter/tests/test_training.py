import numpy as np
import pytest
import torch

from fuxi_adapter.models import ResidualUNet
from fuxi_adapter.training import SequenceDataset, masked_weighted_smooth_l1, predict


def test_masked_loss_ignores_zero_weight_cells():
    target = torch.zeros(1, 1, 2, 2)
    prediction = target.clone()
    prediction[..., 0, 0] = 1000.0
    weights = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    assert masked_weighted_smooth_l1(prediction, target, weights).item() == 0.0


def test_dataset_rejects_nonfinite_features():
    features = np.zeros((2, 6, 3, 5, 5), dtype=np.float32)
    target = np.zeros((2, 6, 5, 5), dtype=np.float32)
    features[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SequenceDataset(features, target)


def test_zero_initialized_model_predicts_zero_on_cpu():
    features = np.ones((3, 6, 2, 9, 9), dtype=np.float32)
    model = ResidualUNet(in_channels=2, base_channels=4, dropout=0.0)
    result = predict(model, features, device="cpu", batch_size=2, use_amp=False)
    assert result.shape == (3, 6, 9, 9)
    assert np.count_nonzero(result) == 0

