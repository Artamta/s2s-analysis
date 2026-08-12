"""Synthetic contracts for anchored v3 targets and physical losses."""

import numpy as np
import pytest
import torch

from fuxi_adapter.anchored import (
    anchored_composite_loss,
    fit_anchored_target_scale,
    reconstruct_anchored_precipitation,
    standardize_anchored_target,
)


def test_training_scale_is_area_weighted_per_lead():
    log_residual = np.array([[[[1.0, 3.0]], [[2.0, 2.0]]]])
    baseline = np.zeros_like(log_residual)
    truth = np.expm1(log_residual)
    weights = np.array([[1.0, 3.0]])

    scale = fit_anchored_target_scale(
        truth, baseline, weights, split_name="train"
    )

    assert scale == pytest.approx([np.sqrt(7.0), 2.0])
    target = standardize_anchored_target(truth, baseline, scale)
    assert target[0, 0, 0] == pytest.approx(
        [1.0 / np.sqrt(7.0), 3.0 / np.sqrt(7.0)]
    )
    assert target[0, 1, 0] == pytest.approx([1.0, 1.0])


def test_scale_rejects_nontraining_split():
    values = np.ones((1, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="train split"):
        fit_anchored_target_scale(
            values, values, np.ones((2, 2)), split_name="validation"
        )


def test_zero_standardized_residual_is_exact_baseline_identity():
    baseline = np.array(
        [[[[0.0, 1.0], [3.0, 10.0]], [[2.0, 4.0], [6.0, 8.0]]]],
        dtype=np.float32,
    )
    result = reconstruct_anchored_precipitation(
        baseline, np.zeros_like(baseline), np.array([0.5, 2.0])
    )
    assert np.array_equal(result, baseline)


def test_reconstruction_is_finite_and_nonnegative_for_large_negative_residual():
    baseline = np.ones((1, 2, 2, 2), dtype=np.float32)
    residual = np.full_like(baseline, -100.0)
    result = reconstruct_anchored_precipitation(
        baseline, residual, np.array([1.0, 1.0])
    )
    assert np.isfinite(result).all()
    assert np.count_nonzero(result) == 0


def test_reconstruction_sanitizes_nonfinite_values_outside_valid_support():
    baseline = np.ones((1, 2, 2, 2), dtype=np.float32)
    residual = np.zeros_like(baseline)
    mask = np.ones_like(baseline, dtype=bool)
    mask[..., 0, 0] = False
    baseline[..., 0, 0] = np.nan
    residual[..., 0, 0] = np.nan

    result = reconstruct_anchored_precipitation(
        baseline,
        residual,
        np.array([1.0, 1.0]),
        valid_mask=mask,
    )
    assert np.isfinite(result).all()
    assert np.all(result[..., 0, 0] == 0.0)
    assert np.all(result[..., 1, 1] == 1.0)


def _loss_fields():
    truth = torch.tensor(
        [
            [
                [[1.0, 2.0], [4.0, 8.0]],
                [[2.0, 3.0], [5.0, 9.0]],
            ]
        ]
    )
    climatology = torch.tensor(
        [
            [
                [[0.5, 1.0], [1.5, 2.0]],
                [[0.5, 1.0], [1.5, 2.0]],
            ]
        ]
    )
    baseline = truth.clone()
    target = torch.zeros_like(truth)
    weights = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    return truth, climatology, baseline, target, weights


def test_perfect_prediction_has_zero_composite_loss():
    truth, climatology, baseline, target, weights = _loss_fields()
    prediction = torch.zeros_like(target)

    total, components = anchored_composite_loss(
        prediction,
        target,
        baseline,
        truth,
        climatology,
        target_scale=[1.0, 1.0],
        area_weights=weights,
        lead_weights=[0.5, 0.5],
        return_components=True,
    )

    assert total.item() == pytest.approx(0.0, abs=1.0e-7)
    assert components["smooth_l1"].item() == pytest.approx(0.0)
    assert components["acc_loss"].item() == pytest.approx(0.0, abs=1.0e-7)
    assert components["mean_spatial_acc"].item() == pytest.approx(1.0)
    assert components["mean_bias_squared"].item() == pytest.approx(0.0)


def test_nonzero_perfect_anchored_residual_has_zero_loss():
    truth, climatology, _, _, weights = _loss_fields()
    baseline = torch.ones_like(truth)
    scale = torch.tensor([0.5, 1.5])
    target = (torch.log1p(truth) - torch.log1p(baseline)) / scale[
        None, :, None, None
    ]
    prediction = target.clone()

    total = anchored_composite_loss(
        prediction,
        target,
        baseline,
        truth,
        climatology,
        target_scale=scale,
        area_weights=weights,
        lead_weights=[0.5, 0.5],
    )
    assert total.item() == pytest.approx(0.0, abs=1.0e-7)


def test_inactive_lead_does_not_change_any_loss_component():
    truth, climatology, baseline, target, weights = _loss_fields()
    clean = torch.zeros_like(target)
    damaged = clean.clone()
    damaged[:, 1] = 100.0

    clean_loss, clean_components = anchored_composite_loss(
        clean,
        target,
        baseline,
        truth,
        climatology,
        target_scale=[1.0, 1.0],
        area_weights=weights,
        lead_weights=[1.0, 0.0],
        return_components=True,
    )
    damaged_loss, damaged_components = anchored_composite_loss(
        damaged,
        target,
        baseline,
        truth,
        climatology,
        target_scale=[1.0, 1.0],
        area_weights=weights,
        lead_weights=[1.0, 0.0],
        return_components=True,
    )

    assert damaged_loss.item() == pytest.approx(clean_loss.item())
    for name in clean_components:
        assert damaged_components[name].item() == pytest.approx(
            clean_components[name].item()
        )


def test_composite_loss_has_finite_nonzero_gradients():
    truth, climatology, _, _, weights = _loss_fields()
    baseline = torch.full_like(truth, 1.0)
    target = torch.log1p(truth) - torch.log1p(baseline)
    prediction = torch.zeros_like(target, requires_grad=True)

    loss = anchored_composite_loss(
        prediction,
        target,
        baseline,
        truth,
        climatology,
        target_scale=[1.0, 1.0],
        area_weights=weights,
        lead_weights=[0.0, 1.0],
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[:, 0]).item() == 0
    assert torch.count_nonzero(prediction.grad[:, 1]).item() > 0


@pytest.mark.parametrize(
    "change,match",
    [
        ("shape", "shape"),
        ("negative_weight", "nonnegative"),
        ("inactive", "active lead"),
        ("nonfinite", "finite on active"),
        ("negative_rain", "nonnegative"),
        ("coefficients", "coefficient"),
    ],
)
def test_composite_loss_rejects_invalid_inputs(change, match):
    truth, climatology, baseline, target, weights = _loss_fields()
    prediction = torch.zeros_like(target)
    lead_weights = [0.5, 0.5]
    kwargs = {}
    if change == "shape":
        truth = truth[:, :, :, :1]
    elif change == "negative_weight":
        weights = weights.clone()
        weights[0, 0] = -1.0
    elif change == "inactive":
        lead_weights = [0.0, 0.0]
    elif change == "nonfinite":
        prediction = prediction.clone()
        prediction[0, 0, 0, 0] = float("nan")
    elif change == "negative_rain":
        baseline = baseline.clone()
        baseline[0, 0, 0, 0] = -1.0
    elif change == "coefficients":
        kwargs["acc_coefficient"] = -1.0

    with pytest.raises(ValueError, match=match):
        anchored_composite_loss(
            prediction,
            target,
            baseline,
            truth,
            climatology,
            target_scale=[1.0, 1.0],
            area_weights=weights,
            lead_weights=lead_weights,
            **kwargs,
        )


def test_nonfinite_values_outside_mask_and_inactive_leads_are_ignored():
    truth, climatology, baseline, target, weights = _loss_fields()
    prediction = torch.zeros_like(target)
    prediction[0, 0, 0, 0] = float("nan")
    prediction[:, 1] = float("nan")
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 0, 0, 0] = False

    loss = anchored_composite_loss(
        prediction,
        target,
        baseline,
        truth,
        climatology,
        target_scale=[1.0, 1.0],
        area_weights=weights,
        lead_weights=[1.0, 0.0],
        valid_mask=mask,
    )
    assert torch.isfinite(loss)
