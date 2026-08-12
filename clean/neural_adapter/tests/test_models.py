"""Tests for the compact FuXi residual adapters."""

import pytest
import torch

from fuxi_adapter.cli import _build_seeded_model
from fuxi_adapter.models import (
    AttentiveClimatologyAllLeadMultiScaleUNet,
    AttentiveClimatologyAllLeadUNet,
    AttentiveClimatologyLateLeadUNet,
    AttentiveClimatologyMultiScaleUNet,
    ClimatologyAttentionConditioner,
    FixedClimatologyAllLeadMultiScaleUNet,
    FixedClimatologyAllLeadUNet,
    FixedClimatologyLateLeadUNet,
    FixedClimatologyMultiScaleUNet,
    LateLeadTemporalUNet,
    MultiScaleAllLeadTemporalUNet,
    MultiScaleLateLeadTemporalUNet,
    ResidualUNet,
    TemporalAttentionUNet,
    build_model,
    count_parameters,
)


@pytest.mark.parametrize("model_class", [ResidualUNet, TemporalAttentionUNet])
def test_adapter_shape_is_finite_and_zero_initialized(model_class):
    torch.manual_seed(7)
    model = model_class(in_channels=5)
    inputs = torch.randn(2, 6, 5, 27, 27)

    outputs = model(inputs)

    assert outputs.shape == (2, 6, 27, 27)
    assert torch.isfinite(outputs).all()
    assert torch.count_nonzero(outputs).item() == 0


@pytest.mark.parametrize("spatial_shape", [(27, 27), (25, 31), (28, 26)])
def test_residual_unet_preserves_odd_and_even_spatial_shapes(spatial_shape):
    model = ResidualUNet(in_channels=3, dropout=0.0)
    height, width = spatial_shape
    outputs = model(torch.randn(1, 6, 3, height, width))

    assert outputs.shape == (1, 6, height, width)


def test_temporal_attention_accepts_up_to_configured_number_of_leads():
    model = TemporalAttentionUNet(in_channels=2, max_leads=6, dropout=0.0)

    assert model(torch.randn(1, 3, 2, 27, 27)).shape == (1, 3, 27, 27)
    with pytest.raises(ValueError, match="max_leads"):
        model(torch.randn(1, 7, 2, 27, 27))


def test_late_lead_temporal_unet_shape_and_exact_inactive_zeros():
    model = LateLeadTemporalUNet(in_channels=3, base_channels=4, dropout=0.0)
    with torch.no_grad():
        model.residual_head.bias.fill_(2.5)

    outputs = model(torch.randn(2, 6, 3, 9, 11))

    assert outputs.shape == (2, 6, 9, 11)
    assert torch.equal(outputs[:, :2], torch.zeros_like(outputs[:, :2]))
    assert torch.equal(outputs[:, 2:], torch.full_like(outputs[:, 2:], 2.5))


@pytest.mark.parametrize("lead_count", [5, 7])
def test_late_lead_temporal_unet_requires_exactly_six_leads(lead_count):
    model = LateLeadTemporalUNet(in_channels=2, base_channels=4, dropout=0.0)

    with pytest.raises(ValueError, match="requires exactly six lead weeks"):
        model(torch.randn(1, lead_count, 2, 9, 9))


def test_late_lead_temporal_unet_active_leads_propagate_gradients():
    torch.manual_seed(11)
    model = LateLeadTemporalUNet(in_channels=2, base_channels=4, dropout=0.0)
    with torch.no_grad():
        model.residual_head.weight.fill_(0.1)
    inputs = torch.randn(1, 6, 2, 8, 8, requires_grad=True)

    active_loss = model(inputs)[:, 2:].square().mean()
    active_loss.backward()

    assert active_loss.item() > 0.0
    assert model.residual_head.weight.grad is not None
    assert model.residual_head.weight.grad.abs().sum().item() > 0.0
    attention_gradient = model.temporal_attention.self_attn.in_proj_weight.grad
    assert attention_gradient is not None
    assert attention_gradient.abs().sum().item() > 0.0
    assert inputs.grad is not None
    assert inputs.grad.abs().sum().item() > 0.0


def test_late_lead_temporal_unet_parameter_count_matches_temporal_model():
    temporal = TemporalAttentionUNet(in_channels=5)
    late_lead = LateLeadTemporalUNet(in_channels=5)

    assert count_parameters(late_lead) == count_parameters(temporal) == 143_825


def test_multiscale_temporal_unet_is_zero_initialized_and_preserves_w1_w4():
    model = MultiScaleLateLeadTemporalUNet(
        in_channels=3,
        base_channels=8,
        spatial_dropout=0.0,
        temporal_dropout=0.0,
        attention_heads=4,
        context_dropout=0.0,
    )
    inputs = torch.randn(2, 6, 3, 11, 13)
    assert torch.count_nonzero(model(inputs)).item() == 0

    with torch.no_grad():
        model.residual_head.bias.fill_(1.25)
    outputs = model(inputs)
    assert torch.equal(outputs[:, :4], torch.zeros_like(outputs[:, :4]))
    assert torch.equal(outputs[:, 4:], torch.full_like(outputs[:, 4:], 1.25))


def test_multiscale_temporal_unet_has_expected_a100_capacity():
    model = MultiScaleLateLeadTemporalUNet(in_channels=11)

    assert count_parameters(model) == 2_544_049


def test_multiscale_allweek_model_can_correct_every_week():
    model = MultiScaleAllLeadTemporalUNet(
        in_channels=3,
        base_channels=8,
        spatial_dropout=0.0,
        temporal_dropout=0.0,
        attention_heads=4,
        context_dropout=0.0,
    )
    inputs = torch.randn(1, 6, 3, 9, 11)
    with torch.no_grad():
        model.residual_head.bias.fill_(1.25)

    assert torch.equal(model(inputs), torch.full((1, 6, 9, 11), 1.25))


def test_factory_builds_multiscale_temporal_unet():
    model = build_model(
        "big-temporal-unet",
        in_channels=3,
        base_channels=8,
        spatial_dropout=0.0,
        temporal_dropout=0.0,
        attention_heads=4,
        context_dropout=0.0,
    )

    assert isinstance(model, MultiScaleLateLeadTemporalUNet)


def test_climatology_attention_is_convex_and_inactive_before_week_five():
    torch.manual_seed(3)
    conditioner = ClimatologyAttentionConditioner(initial_gate=0.05)
    inputs = torch.randn(2, 6, 29, 7, 9)
    effective, weights, gate = conditioner(inputs)

    assert effective.shape == (2, 6, 11, 7, 9)
    assert weights.shape == (2, 6, 9, 7, 9)
    assert torch.allclose(weights.sum(dim=2), torch.ones_like(weights[:, :, 0]))
    assert torch.equal(effective[:, :4, 2], inputs[:, :4, 2])
    assert torch.equal(effective[:, :4, 9], inputs[:, :4, 9])
    assert torch.equal(gate[:4], torch.zeros_like(gate[:4]))
    assert torch.all(gate[4:] > 0.0)


def test_attention_and_fixed_climatology_models_preserve_identity_contract():
    inputs = torch.randn(1, 6, 29, 9, 11)
    fixed = FixedClimatologyMultiScaleUNet()
    attentive = AttentiveClimatologyMultiScaleUNet()

    assert torch.count_nonzero(fixed(inputs)).item() == 0
    assert torch.count_nonzero(attentive(inputs)).item() == 0
    assert count_parameters(fixed) == 2_544_049
    assert count_parameters(attentive) == 2_544_475


def test_week36_climatology_models_are_small_and_preserve_w1_w2():
    inputs = torch.randn(1, 6, 29, 9, 11)
    fixed = FixedClimatologyLateLeadUNet(dropout=0.0)
    attentive = AttentiveClimatologyLateLeadUNet(dropout=0.0)
    with torch.no_grad():
        fixed.backbone.residual_head.bias.fill_(1.25)
        attentive.backbone.residual_head.bias.fill_(1.25)

    for model in (fixed, attentive):
        outputs = model(inputs)
        assert torch.equal(outputs[:, :2], torch.zeros_like(outputs[:, :2]))
        assert torch.equal(outputs[:, 2:], torch.full_like(outputs[:, 2:], 1.25))

    assert count_parameters(fixed) == 144_689
    assert count_parameters(attentive) == 145_115


def test_week36_attention_conditioner_activates_only_week_three_onward():
    conditioner = ClimatologyAttentionConditioner(first_active_lead=2)
    inputs = torch.randn(1, 6, 29, 7, 9)
    effective, _, gate = conditioner(inputs)

    assert torch.equal(effective[:, :2, 2], inputs[:, :2, 2])
    assert torch.equal(effective[:, :2, 9], inputs[:, :2, 9])
    assert torch.equal(gate[:2], torch.zeros_like(gate[:2]))
    assert torch.all(gate[2:] > 0.0)


def test_allweek_climatology_models_can_correct_every_week():
    inputs = torch.randn(1, 6, 29, 9, 11)
    fixed = FixedClimatologyAllLeadUNet(dropout=0.0)
    attentive = AttentiveClimatologyAllLeadUNet(dropout=0.0)
    with torch.no_grad():
        fixed.backbone.residual_head.bias.fill_(1.25)
        attentive.backbone.residual_head.bias.fill_(1.25)

    for model in (fixed, attentive):
        outputs = model(inputs)
        assert torch.equal(outputs, torch.full_like(outputs, 1.25))

    assert count_parameters(fixed) == 144_689
    assert count_parameters(attentive) == 145_115


def test_large_allweek_climatology_models_match_expected_capacity():
    inputs = torch.randn(1, 6, 29, 9, 11)
    fixed = FixedClimatologyAllLeadMultiScaleUNet()
    attentive = AttentiveClimatologyAllLeadMultiScaleUNet()
    with torch.no_grad():
        fixed.backbone.residual_head.bias.fill_(1.25)
        attentive.backbone.residual_head.bias.fill_(1.25)

    assert torch.equal(fixed(inputs), torch.full((1, 6, 9, 11), 1.25))
    assert torch.equal(attentive(inputs), torch.full((1, 6, 9, 11), 1.25))
    assert count_parameters(fixed) == 2_544_049
    assert count_parameters(attentive) == 2_544_475


def test_allweek_attention_conditioner_has_no_inactive_gate():
    conditioner = ClimatologyAttentionConditioner(first_active_lead=0)
    _, _, gate = conditioner(torch.randn(1, 6, 29, 7, 9))

    assert torch.all(gate > 0.0)


@pytest.mark.parametrize(
    "name",
    [
        "late_lead_temporal_unet",
        "late-lead-temporal-unet",
        "late_lead_attention_unet",
        "late_lead_unet",
    ],
)
def test_factory_builds_late_lead_temporal_unet_aliases(name):
    model = build_model(name, in_channels=3, base_channels=4, dropout=0.0)

    assert isinstance(model, LateLeadTemporalUNet)


def test_factory_and_parameter_counts_are_small():
    unet = build_model("residual_unet", in_channels=5)
    temporal = build_model("temporal-attention-unet", in_channels=5)

    unet_parameters = count_parameters(unet)
    temporal_parameters = count_parameters(temporal)

    assert isinstance(unet, ResidualUNet)
    assert isinstance(temporal, TemporalAttentionUNet)
    assert 50_000 < unet_parameters < 500_000
    assert unet_parameters < temporal_parameters < 750_000


def test_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("not-a-model", in_channels=5)


def test_input_channel_validation_is_clear():
    model = ResidualUNet(in_channels=4)
    with pytest.raises(ValueError, match="expected 4 input channels"):
        model(torch.randn(1, 6, 3, 27, 27))


def test_cli_seeds_model_initialization_before_construction():
    first = _build_seeded_model(
        "residual_unet", seed=42, in_channels=3, base_channels=4, dropout=0.0
    )
    second = _build_seeded_model(
        "residual_unet", seed=42, in_channels=3, base_channels=4, dropout=0.0
    )
    different = _build_seeded_model(
        "residual_unet", seed=43, in_channels=3, base_channels=4, dropout=0.0
    )

    first_state = first.state_dict()
    second_state = second.state_dict()
    different_state = different.state_dict()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert any(
        not torch.equal(first_state[name], different_state[name]) for name in first_state
    )
