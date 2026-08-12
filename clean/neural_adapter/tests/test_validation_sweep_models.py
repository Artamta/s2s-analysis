"""Focused tests for isolated validation-sweep model variants."""

import pickle

import pytest
import torch

from fuxi_adapter.validation_sweep_models import (
    CompactLeadReliabilityTemporalUNet,
    FixedCapacityPhysicalTemporalUNet,
    FixedClimatologyFactorized3DUNet,
    SixHeadTemporalAttentionUNet,
    SmoothNoiseTemporalAdapter,
)
from fuxi_adapter.models import TemporalAttentionUNet


def _full_context_inputs(
    batch_size: int = 2,
    leads: int = 6,
    height: int = 9,
    width: int = 11,
) -> torch.Tensor:
    inputs = torch.zeros(batch_size, leads, 29, height, width)
    inputs[:, :, 8, 2:-2, 3:-3] = 1.0
    return inputs


def test_smooth_noise_is_training_only_masked_and_physically_consistent():
    ratios = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    model = SmoothNoiseTemporalAdapter(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
        noise_std=0.2,
        noise_probability=1.0,
        mean_to_anomaly_ratio=ratios,
        t2m_noise_std=0.1,
        coarse_size=3,
    )
    model.train()
    inputs = _full_context_inputs(batch_size=1)
    captured = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )

    torch.manual_seed(17)
    model(inputs)
    handle.remove()
    effective = captured[0]
    original = inputs[:, :, :11]
    support = inputs[:, :, 8].bool()

    unchanged_channels = [2, 3, 4, 5, 6, 7, 8]
    assert torch.equal(
        effective[:, :, unchanged_channels], original[:, :, unchanged_channels]
    )
    for channel in (0, 1, 9, 10):
        difference = effective[:, :, channel] - original[:, :, channel]
        assert torch.count_nonzero(difference[support]).item() > 0
        assert torch.count_nonzero(difference[~support]).item() == 0

    ratio = torch.tensor(ratios)[None, :, None, None]
    assert torch.allclose(effective[:, :, 9], effective[:, :, 0] * ratio)
    assert torch.equal(model.mean_to_anomaly_ratio, torch.tensor(ratios))
    assert "mean_to_anomaly_ratio" in model.state_dict()


def test_smooth_noise_eval_is_exact_and_does_not_advance_rng():
    model = SmoothNoiseTemporalAdapter(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
        noise_std=1.0,
        noise_probability=1.0,
        mean_to_anomaly_ratio=(1.0,) * 6,
    ).eval()
    with torch.no_grad():
        torch.manual_seed(5)
        model.backbone.residual_head.weight.normal_()
    inputs = torch.randn(1, 6, 29, 9, 11)
    rng_before = torch.random.get_rng_state().clone()

    wrapped = model(inputs)
    rng_after = torch.random.get_rng_state().clone()
    direct = model.backbone(inputs[:, :, :11])

    assert torch.equal(rng_before, rng_after)
    assert torch.equal(wrapped, direct)


def test_smooth_noise_probability_zero_is_an_exact_training_noop():
    model = SmoothNoiseTemporalAdapter(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
        noise_std=1.0,
        noise_probability=0.0,
        mean_to_anomaly_ratio=(1.0,) * 6,
    ).train()
    inputs = torch.randn(1, 6, 29, 9, 11)
    rng_before = torch.random.get_rng_state().clone()

    wrapped = model(inputs)
    rng_after = torch.random.get_rng_state().clone()
    direct = model.backbone(inputs[:, :, :11])

    assert torch.equal(rng_before, rng_after)
    assert torch.equal(wrapped, direct)


def test_six_head_temporal_model_has_six_zero_initialized_outputs():
    model = SixHeadTemporalAttentionUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
    )

    outputs = model(torch.randn(2, 6, 29, 9, 11))

    assert outputs.shape == (2, 6, 9, 11)
    assert len(model.residual_heads) == 6
    assert not hasattr(model, "residual_head")
    assert torch.count_nonzero(outputs).item() == 0
    for head in model.residual_heads:
        assert torch.count_nonzero(head.weight).item() == 0
        assert torch.count_nonzero(head.bias).item() == 0


def test_six_head_temporal_model_routes_each_lead_to_an_independent_head():
    model = SixHeadTemporalAttentionUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        for lead, head in enumerate(model.residual_heads):
            head.bias.fill_(float(lead + 1))

    outputs = model(torch.randn(1, 6, 29, 9, 11))

    expected = torch.arange(1, 7, dtype=outputs.dtype)[None, :, None, None]
    assert torch.equal(outputs, expected.expand_as(outputs))
    assert len({id(head.weight) for head in model.residual_heads}) == 6
    assert len({head.weight.data_ptr() for head in model.residual_heads}) == 6


def test_six_head_temporal_model_routes_head_gradients_by_lead():
    model = SixHeadTemporalAttentionUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        for head in model.residual_heads:
            head.weight.fill_(0.1)

    selected_lead = 3
    outputs = model(torch.randn(2, 6, 29, 9, 11))
    outputs[:, selected_lead].sum().backward()

    for lead, head in enumerate(model.residual_heads):
        weight_gradient = head.weight.grad
        bias_gradient = head.bias.grad
        if lead == selected_lead:
            assert weight_gradient is not None
            assert torch.count_nonzero(weight_gradient).item() > 0
            assert bias_gradient is not None
            assert torch.count_nonzero(bias_gradient).item() > 0
        else:
            assert weight_gradient is None or torch.count_nonzero(weight_gradient) == 0
            assert bias_gradient is None or torch.count_nonzero(bias_gradient) == 0


def test_compact_shared_wrapper_can_consume_appended_member_channels():
    model = CompactLeadReliabilityTemporalUNet(
        input_channels=15,
        backbone_channels=15,
        base_channels=4,
        dropout=0.0,
        lead_rank=0,
    ).eval()
    direct = TemporalAttentionUNet(
        in_channels=15,
        base_channels=4,
        dropout=0.0,
        max_leads=6,
    ).eval()
    with torch.no_grad():
        model.residual_head.weight.fill_(0.1)
        model.residual_head.bias.fill_(0.2)
    direct.load_state_dict(model.state_dict())
    inputs = torch.randn(1, 6, 15, 9, 11)

    wrapped = model(inputs)
    expected = direct(inputs)
    changed_member_input = inputs.clone()
    changed_member_input[:, :, 14] += 3.0

    assert torch.equal(wrapped, expected)
    assert model.encoder_1.block[0].in_channels == 15
    assert not torch.equal(model(changed_member_input), wrapped)


def test_compact_rank_two_model_is_zero_initialized_with_one_shared_head():
    model = CompactLeadReliabilityTemporalUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
        lead_rank=2,
    )

    outputs = model(_full_context_inputs())

    assert outputs.shape == (2, 6, 9, 11)
    assert torch.count_nonzero(outputs).item() == 0
    assert model.residual_head.out_channels == 1
    assert model.lead_delta_head.out_channels == 2
    assert torch.count_nonzero(model.residual_head.weight).item() == 0
    assert torch.count_nonzero(model.lead_delta_head.weight).item() == 0
    assert model.lead_basis.shape == (6, 2)


def test_compact_rank_two_delta_varies_smoothly_by_fixed_lead_basis():
    model = CompactLeadReliabilityTemporalUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
        lead_rank=2,
    ).eval()
    with torch.no_grad():
        model.lead_delta_head.bias.copy_(torch.tensor([1.0, 0.0]))

    outputs = model(_full_context_inputs(batch_size=1))
    expected = model.lead_basis[:, 0][None, :, None, None].expand_as(outputs)

    assert torch.allclose(outputs, expected)
    first_difference = torch.diff(outputs[0, :, 0, 0])
    assert torch.all(first_difference > 0.0)
    assert torch.allclose(
        first_difference, first_difference[0].expand_as(first_difference)
    )


def test_compact_model_parameter_budget_is_smaller_than_six_independent_heads():
    shared = CompactLeadReliabilityTemporalUNet(
        input_channels=29, base_channels=24, lead_rank=0
    )
    rank_two = CompactLeadReliabilityTemporalUNet(
        input_channels=29, base_channels=24, lead_rank=2
    )
    gated_rank_two = CompactLeadReliabilityTemporalUNet(
        input_channels=29,
        base_channels=24,
        lead_rank=2,
        use_spread_gate=True,
    )

    shared_count = sum(parameter.numel() for parameter in shared.parameters())
    rank_two_count = sum(parameter.numel() for parameter in rank_two.parameters())
    gated_count = sum(parameter.numel() for parameter in gated_rank_two.parameters())

    assert shared_count == 323_017
    assert rank_two_count == shared_count + 2 * (24 + 1)
    assert gated_count == rank_two_count + 12
    assert rank_two_count < 323_142


def test_compact_spread_gate_is_monotonic_and_zero_outside_support():
    model = CompactLeadReliabilityTemporalUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
        lead_rank=0,
        use_spread_gate=True,
    ).eval()
    inputs = _full_context_inputs(batch_size=1)
    inputs[:, :, 8] = 1.0
    inputs[:, :, 1, :, :5] = -2.0
    inputs[:, :, 1, :, 5:] = 2.0
    inputs[:, :, 8, 0, 0] = 0.0
    with torch.no_grad():
        model.residual_head.bias.fill_(1.0)

    gate = model.reliability_gate(inputs)
    outputs = model(inputs)

    assert torch.equal(outputs, gate)
    low_spread_gate = gate[:, :, 1:, 1:5].mean(dim=(-2, -1))
    high_spread_gate = gate[:, :, 1:, 5:].mean(dim=(-2, -1))
    assert torch.all(low_spread_gate > high_spread_gate)
    assert torch.all((gate >= 0.0) & (gate < 1.0))
    assert torch.count_nonzero(outputs[:, :, 0, 0]).item() == 0


def test_compact_spread_gate_parameters_receive_gradients():
    model = CompactLeadReliabilityTemporalUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
        lead_rank=0,
        use_spread_gate=True,
    ).eval()
    with torch.no_grad():
        model.residual_head.bias.fill_(1.0)
    inputs = _full_context_inputs(batch_size=1)
    inputs[:, :, 1] = torch.randn_like(inputs[:, :, 1])

    model(inputs)[:, 4].sum().backward()

    assert model.gate_intercept.grad is not None
    assert torch.count_nonzero(model.gate_intercept.grad).item() == 1
    assert model.gate_raw_slope.grad is not None
    assert torch.count_nonzero(model.gate_raw_slope.grad).item() == 1


def test_fixed_capacity_physical_adapter_is_an_exact_zero_initialized_noop():
    model = FixedCapacityPhysicalTemporalUNet(
        input_channels=38,
        physical_channel_indices=tuple(range(29, 38)),
        active_physical_slots=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        base_channels=4,
        dropout=0.0,
    )
    inputs = torch.randn(2, 6, 38, 9, 11)

    effective = model.effective_backbone_inputs(inputs)
    outputs = model(inputs)

    assert torch.equal(effective, inputs[:, :, :11])
    assert outputs.shape == (2, 6, 9, 11)
    assert torch.count_nonzero(outputs).item() == 0
    assert torch.count_nonzero(model.physical_projection.weight).item() == 0
    assert torch.count_nonzero(model.backbone.residual_head.weight).item() == 0


def test_fixed_capacity_physical_adapter_masks_inactive_slots():
    model = FixedCapacityPhysicalTemporalUNet(
        input_channels=38,
        physical_channel_indices=tuple(range(29, 38)),
        active_physical_slots=(0, 3),
        base_channels=4,
        dropout=0.0,
    )
    with torch.no_grad():
        model.physical_projection.weight.fill_(1.0)
    inputs = torch.zeros(1, 6, 38, 9, 11)
    inputs[:, :, 29] = 2.0
    inputs[:, :, 32] = 3.0
    inputs[:, :, 30] = 1000.0

    effective = model.effective_backbone_inputs(inputs)
    changed_inactive = inputs.clone()
    changed_inactive[:, :, 30] += 50_000.0
    changed_active = inputs.clone()
    changed_active[:, :, 29] += 1.0

    assert torch.equal(effective, torch.full_like(effective, 5.0))
    assert torch.equal(
        model.effective_backbone_inputs(changed_inactive), effective
    )
    assert not torch.equal(
        model.effective_backbone_inputs(changed_active), effective
    )


def test_physical_ablation_capacity_and_backbone_initialization_are_fixed():
    torch.manual_seed(71)
    control = SmoothNoiseTemporalAdapter(
        input_channels=38,
        base_channels=4,
        dropout=0.0,
        noise_std=0.0,
        noise_probability=0.0,
        mean_to_anomaly_ratio=(1.0,) * 6,
    )
    control_rng_state = torch.random.get_rng_state().clone()
    torch.manual_seed(71)
    tcwv = FixedCapacityPhysicalTemporalUNet(
        input_channels=38,
        physical_channel_indices=tuple(range(29, 38)),
        active_physical_slots=(0,),
        base_channels=4,
        dropout=0.0,
    )
    tcwv_rng_state = torch.random.get_rng_state().clone()
    torch.manual_seed(71)
    full = FixedCapacityPhysicalTemporalUNet(
        input_channels=38,
        physical_channel_indices=tuple(range(29, 38)),
        active_physical_slots=tuple(range(9)),
        base_channels=4,
        dropout=0.0,
    )
    full_rng_state = torch.random.get_rng_state().clone()

    for name, tensor in control.backbone.state_dict().items():
        assert torch.equal(tensor, tcwv.backbone.state_dict()[name])
        assert torch.equal(tensor, full.backbone.state_dict()[name])
    assert sum(parameter.numel() for parameter in tcwv.parameters()) == sum(
        parameter.numel() for parameter in full.parameters()
    )
    assert tcwv.physical_projection.weight.numel() == 9 * 11
    assert torch.equal(control_rng_state, tcwv_rng_state)
    assert torch.equal(control_rng_state, full_rng_state)


@pytest.mark.parametrize("spatial_shape", [(9, 11), (12, 10), (27, 27)])
def test_factorized_3d_unet_is_zero_initialized_and_preserves_shape(spatial_shape):
    height, width = spatial_shape
    model = FixedClimatologyFactorized3DUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
    )
    inputs = torch.randn(1, 6, 29, height, width)

    outputs = model(inputs)

    assert outputs.shape == (1, 6, height, width)
    assert torch.isfinite(outputs).all()
    assert torch.count_nonzero(outputs).item() == 0


def test_factorized_3d_unet_pooling_keeps_every_lead():
    model = FixedClimatologyFactorized3DUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
    )
    observed_shapes = []
    handle = model.pool.register_forward_hook(
        lambda _module, _arguments, output: observed_shapes.append(output.shape)
    )

    model(torch.randn(2, 6, 29, 9, 11))
    handle.remove()

    assert [shape[2] for shape in observed_shapes] == [6, 6]
    assert observed_shapes[0][-2:] == (4, 5)
    assert observed_shapes[1][-2:] == (2, 2)


def test_factorized_3d_unet_residual_head_can_correct_every_lead():
    model = FixedClimatologyFactorized3DUNet(
        input_channels=29,
        base_channels=4,
        dropout=0.0,
    )
    with torch.no_grad():
        model.residual_head.bias.fill_(1.25)

    outputs = model(torch.randn(1, 6, 29, 9, 11))

    assert torch.equal(outputs, torch.full((1, 6, 9, 11), 1.25))


@pytest.mark.parametrize(
    "model",
    [
        SmoothNoiseTemporalAdapter(
            input_channels=29,
            base_channels=4,
            mean_to_anomaly_ratio=(1.0,) * 6,
        ),
        SixHeadTemporalAttentionUNet(input_channels=29, base_channels=4),
        CompactLeadReliabilityTemporalUNet(
            input_channels=29,
            base_channels=4,
            lead_rank=2,
            use_spread_gate=True,
        ),
        FixedCapacityPhysicalTemporalUNet(
            input_channels=38,
            physical_channel_indices=tuple(range(29, 38)),
            active_physical_slots=(0, 1),
            base_channels=4,
        ),
        FixedClimatologyFactorized3DUNet(input_channels=29, base_channels=4),
    ],
)
def test_validation_sweep_models_are_picklable(model):
    restored = pickle.loads(pickle.dumps(model))

    assert type(restored) is type(model)
    assert restored.input_channels == model.input_channels


@pytest.mark.parametrize(
    ("constructor", "kwargs", "message"),
    [
        (
            SmoothNoiseTemporalAdapter,
            {"input_channels": 29, "mean_to_anomaly_ratio": (1.0,) * 5},
            "exactly six",
        ),
        (
            SmoothNoiseTemporalAdapter,
            {
                "input_channels": 29,
                "noise_probability": 1.1,
                "mean_to_anomaly_ratio": (1.0,) * 6,
            },
            "noise_probability",
        ),
        (
            FixedClimatologyFactorized3DUNet,
            {"input_channels": 29, "backbone_channels": 10},
            "backbone_channels=11",
        ),
        (
            CompactLeadReliabilityTemporalUNet,
            {"input_channels": 15, "backbone_channels": 16},
            "between 11 and input_channels",
        ),
        (
            CompactLeadReliabilityTemporalUNet,
            {"input_channels": 29, "lead_rank": 1},
            "lead_rank",
        ),
        (
            CompactLeadReliabilityTemporalUNet,
            {
                "input_channels": 29,
                "use_spread_gate": True,
                "initial_reliability": 1.0,
            },
            "initial_reliability",
        ),
        (
            FixedCapacityPhysicalTemporalUNet,
            {
                "input_channels": 38,
                "physical_channel_indices": tuple(range(29, 37)),
                "active_physical_slots": (0,),
            },
            "exactly 9",
        ),
        (
            FixedCapacityPhysicalTemporalUNet,
            {
                "input_channels": 38,
                "physical_channel_indices": tuple(range(29, 38)),
                "active_physical_slots": (9,),
            },
            "between zero and eight",
        ),
    ],
)
def test_validation_sweep_models_reject_invalid_configuration(
    constructor, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        constructor(**kwargs)
