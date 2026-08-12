"""Compact model variants used by the validation-loss sweep.

This module is deliberately isolated from the production model registry.  It
contains small experimental adapters that obey the same residual-output
contract as :class:`fuxi_adapter.models.TemporalAttentionUNet`:

* :class:`SmoothNoiseTemporalAdapter` adds physically consistent, smooth
  training-only perturbations before calling the existing temporal U-Net.
* :class:`FixedClimatologyFactorized3DUNet` mixes lead weeks throughout a
  compact 3-D U-Net without ever pooling along the lead axis.
* :class:`SixHeadTemporalAttentionUNet` retains the temporal U-Net backbone
  but gives each of the six forecast weeks its own residual output head.
* :class:`CompactLeadReliabilityTemporalUNet` keeps one shared output head,
  optionally adds a smooth rank-two lead delta, and can monotonically shrink
  learned corrections when FuXi ensemble spread is high.
* :class:`FixedCapacityPhysicalTemporalUNet` adds a zero-initialized ``1 x 1``
  projection from a fixed bank of physical predictors into the unchanged
  eleven-channel temporal backbone.

All public classes are defined at module scope so complete model objects can
be pickled by training and checkpointing workers.
"""

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .models import TemporalAttentionUNet


BACKBONE_CHANNELS = 11
SUPPORT_CHANNEL = 8
NOISY_CHANNELS = (0, 1, 9, 10)
PHYSICAL_PROJECTION_SLOTS = 9


def _group_count(channels: int, maximum_groups: int = 8) -> int:
    """Return the largest valid GroupNorm group count."""

    for groups in range(min(maximum_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _validate_wrapper_channels(input_channels: int, backbone_channels: int) -> None:
    if input_channels < BACKBONE_CHANNELS:
        raise ValueError(
            f"input_channels must be at least {BACKBONE_CHANNELS}, got "
            f"{input_channels}"
        )
    if backbone_channels != BACKBONE_CHANNELS:
        raise ValueError(
            "validation-sweep models require backbone_channels=11 so the "
            "frozen-climatology feature contract is unambiguous"
        )


def _validate_flexible_wrapper_channels(
    input_channels: int, backbone_channels: int
) -> None:
    """Validate a full-input wrapper that may consume appended predictors.

    The first eleven channels retain the fixed-climatology contract.  Extra
    channels can contain ensemble-member summaries or other forecast-only
    predictors.  A caller opts into those channels explicitly by increasing
    ``backbone_channels``; trailing channels remain available to wrapper
    logic but are not silently passed to the convolutional backbone.
    """

    if input_channels < BACKBONE_CHANNELS:
        raise ValueError(
            f"input_channels must be at least {BACKBONE_CHANNELS}, got "
            f"{input_channels}"
        )
    if not BACKBONE_CHANNELS <= backbone_channels <= input_channels:
        raise ValueError(
            "backbone_channels must be between 11 and input_channels "
            f"inclusive, got {backbone_channels} for {input_channels} inputs"
        )


class SmoothNoiseTemporalAdapter(nn.Module):
    """Temporal-attention U-Net with smooth training-only input perturbations.

    The wrapper accepts the full feature tensor but deliberately passes only
    the first eleven fixed-climatology channels to its backbone.  During
    training, one Bernoulli decision is made per sample.  Selected samples get
    independent, bilinearly upsampled Gaussian fields for precipitation mean,
    precipitation spread, and temperature.  Perturbations are confined to the
    support in channel 8.

    Channels 0 (normalized log FuXi mean) and 9 (normalized explicit FuXi
    anomaly) represent the same physical mean-forecast perturbation.  They
    therefore share a noise field; channel 9 is scaled by the supplied
    lead-wise ratio ``std(mean) / std(anomaly)``.  The ratio is a persistent
    buffer so it follows the model across devices and checkpoints.

    Evaluation mode is an exact no-noise path: it slices the original tensor
    and calls the backbone without cloning the input or drawing random values.

    Parameters
    ----------
    input_channels:
        Number of channels in the full input tensor (normally 29).
    backbone_channels:
        Fixed at 11 for this experimental contract.
    base_channels, dropout:
        Passed to :class:`TemporalAttentionUNet`.
    noise_std:
        Standard deviation in normalized units for channels 0 and 1.
    noise_probability:
        Probability that an entire training sample is perturbed.
    mean_to_anomaly_ratio:
        Six lead-wise ``std(mean) / std(anomaly)`` normalization ratios.
    t2m_noise_std:
        Temperature perturbation standard deviation in normalized units.  If
        ``None``, ``noise_std`` is used.  Pass zero to disable T2M noise.
    coarse_size:
        Maximum side length of the Gaussian grid before bilinear upsampling.
    """

    def __init__(
        self,
        input_channels: int,
        backbone_channels: int = BACKBONE_CHANNELS,
        base_channels: int = 16,
        dropout: float = 0.30,
        noise_std: float = 0.05,
        noise_probability: float = 0.50,
        mean_to_anomaly_ratio: Sequence[float] = (1.0,) * 6,
        t2m_noise_std: Optional[float] = None,
        coarse_size: int = 4,
    ) -> None:
        super().__init__()
        _validate_wrapper_channels(input_channels, backbone_channels)
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if not 0.0 <= noise_probability <= 1.0:
            raise ValueError("noise_probability must be in [0, 1]")
        if t2m_noise_std is not None and t2m_noise_std < 0.0:
            raise ValueError("t2m_noise_std must be non-negative or None")
        if coarse_size < 2:
            raise ValueError("coarse_size must be at least 2")

        ratio = torch.as_tensor(mean_to_anomaly_ratio, dtype=torch.float32)
        if ratio.ndim != 1 or ratio.numel() != 6:
            raise ValueError("mean_to_anomaly_ratio must contain exactly six values")
        if not torch.isfinite(ratio).all() or torch.any(ratio <= 0.0):
            raise ValueError("mean_to_anomaly_ratio values must be finite and positive")

        self.input_channels = int(input_channels)
        self.backbone_channels = int(backbone_channels)
        self.noise_std = float(noise_std)
        self.noise_probability = float(noise_probability)
        self.t2m_noise_std = float(
            noise_std if t2m_noise_std is None else t2m_noise_std
        )
        self.coarse_size = int(coarse_size)
        self.register_buffer("mean_to_anomaly_ratio", ratio.clone())

        self.backbone = TemporalAttentionUNet(
            in_channels=backbone_channels,
            base_channels=base_channels,
            dropout=dropout,
            max_leads=6,
        )

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                "expected input shape "
                f"[batch, lead, {self.input_channels}, height, width], got "
                f"{tuple(inputs.shape)}"
            )
        if not 1 <= inputs.shape[1] <= self.mean_to_anomaly_ratio.numel():
            raise ValueError("lead count must be between one and six")

    def _smooth_unit_noise(
        self,
        batch_size: int,
        leads: int,
        fields: int,
        height: int,
        width: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Sample zero-mean, unit-RMS coarse noise and upsample it smoothly."""

        coarse_height = min(self.coarse_size, height)
        coarse_width = min(self.coarse_size, width)
        coarse = torch.randn(
            batch_size * leads,
            fields,
            coarse_height,
            coarse_width,
            dtype=reference.dtype,
            device=reference.device,
        )
        smooth = F.interpolate(
            coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        smooth = smooth - smooth.mean(dim=(-2, -1), keepdim=True)
        epsilon = torch.finfo(smooth.dtype).eps
        root_mean_square = (
            smooth.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(epsilon)
        )
        smooth = smooth / root_mean_square
        return smooth.reshape(batch_size, leads, fields, height, width)

    def _add_training_noise(self, effective: torch.Tensor) -> torch.Tensor:
        """Return a perturbed copy of the eleven-channel backbone input."""

        if (
            not self.training
            or self.noise_probability == 0.0
            or (self.noise_std == 0.0 and self.t2m_noise_std == 0.0)
        ):
            return effective

        batch_size, leads, _, height, width = effective.shape
        fields = self._smooth_unit_noise(
            batch_size,
            leads,
            fields=3,
            height=height,
            width=width,
            reference=effective,
        )
        sample_gate = (
            torch.rand(
                batch_size,
                1,
                1,
                1,
                device=effective.device,
            )
            < self.noise_probability
        ).to(dtype=effective.dtype)
        support = (effective[:, :, SUPPORT_CHANNEL] > 0.0).to(dtype=effective.dtype)
        gate = sample_gate * support

        mean_noise = self.noise_std * fields[:, :, 0] * gate
        spread_noise = self.noise_std * fields[:, :, 1] * gate
        temperature_noise = self.t2m_noise_std * fields[:, :, 2] * gate
        ratio = self.mean_to_anomaly_ratio[:leads].to(dtype=effective.dtype)[
            None, :, None, None
        ]

        perturbed = effective.clone()
        perturbed[:, :, 0] = perturbed[:, :, 0] + mean_noise
        perturbed[:, :, 1] = perturbed[:, :, 1] + spread_noise
        perturbed[:, :, 9] = perturbed[:, :, 9] + mean_noise * ratio
        perturbed[:, :, 10] = perturbed[:, :, 10] + temperature_noise
        return perturbed

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        effective = inputs[:, :, : self.backbone_channels]
        effective = self._add_training_noise(effective)
        return self.backbone(effective)


class SixHeadTemporalAttentionUNet(TemporalAttentionUNet):
    """Temporal-attention U-Net with one residual head per forecast week.

    The spatial encoder, decoder, skip connections, bottleneck Transformer,
    normalization, activations, and dropout are inherited unchanged from
    :class:`TemporalAttentionUNet`.  The only architectural difference is the
    output projection: decoded week ``i`` is passed through
    ``residual_heads[i]`` instead of a projection shared by every week.

    Each head is an independent ``1 x 1`` convolution from ``base_channels``
    to one residual channel.  All six heads are initialized to zero, retaining
    the exact no-op initialization used by the shared-head control.

    Parameters
    ----------
    input_channels:
        Number of channels in the full input tensor (normally 29).
    backbone_channels:
        Fixed at 11 for the validation-sweep fixed-climatology contract.
    base_channels, dropout:
        Passed unchanged to :class:`TemporalAttentionUNet`.

    Notes
    -----
    Inputs and outputs have shapes ``[B, L, C, H, W]`` and ``[B, L, H, W]``
    respectively.  Between one and six leading weeks are accepted; week
    indices are always routed to the corresponding first ``L`` heads.
    """

    def __init__(
        self,
        input_channels: int,
        backbone_channels: int = BACKBONE_CHANNELS,
        base_channels: int = 16,
        dropout: float = 0.30,
    ) -> None:
        _validate_wrapper_channels(input_channels, backbone_channels)
        super().__init__(
            in_channels=backbone_channels,
            base_channels=base_channels,
            dropout=dropout,
            max_leads=6,
        )
        self.input_channels = int(input_channels)
        self.backbone_channels = int(backbone_channels)

        # Remove the inherited shared projection from the module registry so
        # checkpoints and parameter counts contain only the six routed heads.
        del self.residual_head
        self.residual_heads = nn.ModuleList(
            [nn.Conv2d(base_channels, 1, kernel_size=1) for _ in range(6)]
        )
        for head in self.residual_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _validate_full_input(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                "expected input shape "
                f"[batch, lead, {self.input_channels}, height, width], got "
                f"{tuple(inputs.shape)}"
            )
        if not 1 <= inputs.shape[1] <= len(self.residual_heads):
            raise ValueError("lead count must be between one and six")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_full_input(inputs)
        effective = inputs[:, :, : self.backbone_channels]
        # Call the inherited eleven-channel validation explicitly because the
        # public model consumes the larger full-context feature tensor.
        super()._validate_input(effective)

        batch_size, leads, channels, height, width = effective.shape
        flat = effective.reshape(batch_size * leads, channels, height, width)

        skip_1 = self.encoder_1(flat)
        skip_2 = self.encoder_2(self.pool(skip_1))
        encoded = self.bottleneck(self.pool(skip_2))
        encoded = self._transform_bottleneck(encoded, batch_size, leads)

        decoded = F.interpolate(
            encoded,
            size=skip_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.up_2(decoded)
        decoded = self.decoder_2(torch.cat((decoded, skip_2), dim=1))

        decoded = F.interpolate(
            decoded,
            size=skip_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.up_1(decoded)
        decoded = self.decoder_1(torch.cat((decoded, skip_1), dim=1))
        decoded = decoded.reshape(batch_size, leads, self.base_channels, height, width)

        return torch.stack(
            [
                self.residual_heads[lead](decoded[:, lead])[:, 0]
                for lead in range(leads)
            ],
            dim=1,
        )


class CompactLeadReliabilityTemporalUNet(TemporalAttentionUNet):
    """Shared-head temporal U-Net with compact lead and reliability options.

    This adapter accepts a full predictor tensor while allowing the temporal
    U-Net to consume any leading subset from eleven channels through the full
    input width.  Setting ``backbone_channels=input_channels`` therefore lets
    the shared spatial encoder consume appended ensemble-member summaries;
    setting it to eleven reproduces the fixed-climatology predictor contract.

    The default rank-two lead delta keeps the inherited shared residual head
    and adds only two zero-initialized residual basis maps.  A fixed smooth
    linear/quadratic basis combines those maps across forecast weeks::

        residual_l = shared_l + sum_k lead_basis[l, k] * delta_l[k]

    This constrains lead specialization to evolve smoothly instead of fitting
    six independent heads.  ``lead_rank=0`` disables the delta and gives a
    member-capable shared-head wrapper.

    The optional reliability gate uses normalized FuXi log-spread and has a
    positive per-lead slope, so increasing spread can only reduce the learned
    residual.  Multiplication by the support channel makes the gated residual
    exactly zero outside IMD support.  Because every output projection is
    initialized to zero, all configurations retain the exact anchored no-op:
    a new model returns zero residual and therefore the fixed log-bias field.

    Parameters
    ----------
    input_channels:
        Number of channels in the supplied full-context tensor.
    backbone_channels:
        Number of leading channels consumed by the U-Net.  Must be between 11
        and ``input_channels`` inclusive.
    lead_rank:
        Either zero (shared head only) or two (smooth rank-two lead delta).
    use_spread_gate:
        Apply the monotonic spread-reliability gate to the final residual.
    spread_channel, support_channel:
        Channel indices in the full input tensor.  Defaults match the current
        fixed-climatology feature contract.
    initial_reliability:
        Initial gate value before its small spread adjustment.
    initial_spread_slope:
        Strictly positive initial monotonic slope.
    """

    def __init__(
        self,
        input_channels: int,
        backbone_channels: int = BACKBONE_CHANNELS,
        base_channels: int = 16,
        dropout: float = 0.30,
        lead_rank: int = 2,
        use_spread_gate: bool = False,
        spread_channel: int = 1,
        support_channel: int = SUPPORT_CHANNEL,
        initial_reliability: float = 0.95,
        initial_spread_slope: float = 0.05,
    ) -> None:
        _validate_flexible_wrapper_channels(input_channels, backbone_channels)
        if lead_rank not in (0, 2):
            raise ValueError("lead_rank must be either 0 or 2")
        if not 0.0 < initial_reliability < 1.0:
            raise ValueError("initial_reliability must be strictly between 0 and 1")
        if not math.isfinite(initial_spread_slope) or initial_spread_slope <= 0.0:
            raise ValueError("initial_spread_slope must be finite and positive")
        for name, index in (
            ("spread_channel", spread_channel),
            ("support_channel", support_channel),
        ):
            if int(index) != index or not 0 <= int(index) < input_channels:
                raise ValueError(
                    f"{name} must index the full input tensor, got {index}"
                )

        super().__init__(
            in_channels=backbone_channels,
            base_channels=base_channels,
            dropout=dropout,
            max_leads=6,
        )
        self.input_channels = int(input_channels)
        self.backbone_channels = int(backbone_channels)
        self.lead_rank = int(lead_rank)
        self.use_spread_gate = bool(use_spread_gate)
        self.spread_channel = int(spread_channel)
        self.support_channel = int(support_channel)

        if self.lead_rank:
            lead_coordinate = torch.linspace(-1.0, 1.0, steps=self.max_leads)
            quadratic = lead_coordinate.square()
            quadratic = quadratic - quadratic.mean()
            basis = torch.stack((lead_coordinate, quadratic), dim=1)
            basis = basis / basis.square().mean(dim=0, keepdim=True).sqrt()
            self.register_buffer("lead_basis", basis)
            self.lead_delta_head = nn.Conv2d(
                base_channels, self.lead_rank, kernel_size=1
            )
            nn.init.zeros_(self.lead_delta_head.weight)
            nn.init.zeros_(self.lead_delta_head.bias)

        if self.use_spread_gate:
            reliability_logit = math.log(
                initial_reliability / (1.0 - initial_reliability)
            )
            raw_slope = math.log(math.expm1(initial_spread_slope))
            self.gate_intercept = nn.Parameter(
                torch.full((self.max_leads,), reliability_logit)
            )
            self.gate_raw_slope = nn.Parameter(
                torch.full((self.max_leads,), raw_slope)
            )

    def _validate_full_input(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                "expected input shape "
                f"[batch, lead, {self.input_channels}, height, width], got "
                f"{tuple(inputs.shape)}"
            )
        if not 1 <= inputs.shape[1] <= self.max_leads:
            raise ValueError("lead count must be between one and six")

    def _decode_features(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, int, int, int, int]:
        """Return decoded shared features plus their sequence dimensions."""

        self._validate_full_input(inputs)
        effective = inputs[:, :, : self.backbone_channels]
        # Apply the inherited spatial-size and channel checks explicitly: this
        # public wrapper consumes more channels than its temporal backbone.
        super()._validate_input(effective)
        batch_size, leads, channels, height, width = effective.shape
        flat = effective.reshape(batch_size * leads, channels, height, width)

        skip_1 = self.encoder_1(flat)
        skip_2 = self.encoder_2(self.pool(skip_1))
        encoded = self.bottleneck(self.pool(skip_2))
        encoded = self._transform_bottleneck(encoded, batch_size, leads)

        decoded = F.interpolate(
            encoded,
            size=skip_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.up_2(decoded)
        decoded = self.decoder_2(torch.cat((decoded, skip_2), dim=1))

        decoded = F.interpolate(
            decoded,
            size=skip_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.up_1(decoded)
        decoded = self.decoder_1(torch.cat((decoded, skip_1), dim=1))
        return decoded, batch_size, leads, height, width

    def _compute_reliability_gate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a support-masked gate that decreases with FuXi spread."""

        leads = inputs.shape[1]
        spread = inputs[:, :, self.spread_channel]
        uncertainty = F.softplus(spread)
        intercept = self.gate_intercept[:leads].to(dtype=inputs.dtype)[
            None, :, None, None
        ]
        slope = F.softplus(self.gate_raw_slope[:leads]).to(dtype=inputs.dtype)[
            None, :, None, None
        ]
        gate = torch.sigmoid(intercept - slope * uncertainty)
        support = (inputs[:, :, self.support_channel] > 0.0).to(dtype=inputs.dtype)
        return gate * support

    def reliability_gate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Expose the deterministic gate for diagnostics and verification."""

        if not self.use_spread_gate:
            raise RuntimeError("this model was constructed without a spread gate")
        self._validate_full_input(inputs)
        return self._compute_reliability_gate(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        decoded, batch_size, leads, height, width = self._decode_features(inputs)
        residual = self.residual_head(decoded)[:, 0].reshape(
            batch_size, leads, height, width
        )

        if self.lead_rank:
            delta = self.lead_delta_head(decoded).reshape(
                batch_size, leads, self.lead_rank, height, width
            )
            basis = self.lead_basis[:leads].to(dtype=delta.dtype)
            residual = residual + torch.einsum("blrhw,lr->blhw", delta, basis)

        if self.use_spread_gate:
            residual = residual * self._compute_reliability_gate(inputs)
        return residual


class FixedCapacityPhysicalTemporalUNet(nn.Module):
    """Temporal U-Net with a fixed-size additive physical-predictor adapter.

    The first eleven input channels retain the established FuXi--IMD feature
    contract.  Nine additional, explicitly indexed forecast-only predictors
    are projected back to eleven channels with a learned ``1 x 1``
    convolution and added to that unchanged backbone input.  Every physical
    ablation therefore has the same parameter count, irrespective of how many
    predictor slots are active.

    Inactive physical slots are multiplied by a persistent binary mask before
    the projection.  The projection is initialized to zero, so construction
    is an exact input no-op.  The inherited rainfall residual head is also
    zero initialized; consequently every new model initially predicts zero
    residual and reconstructs the fixed log-bias anchor exactly.

    Parameters
    ----------
    input_channels:
        Width of the complete supplied feature tensor.
    physical_channel_indices:
        Exactly nine distinct indices locating the normalized physical bank
        in the complete tensor.  None may overlap the first eleven channels.
    active_physical_slots:
        Subset of slot numbers ``0..8`` enabled for this ablation.  Slot
        numbering follows ``physical_channel_indices``, not absolute input
        channel numbering.
    base_channels, dropout:
        Passed unchanged to :class:`TemporalAttentionUNet`.
    """

    def __init__(
        self,
        input_channels: int,
        physical_channel_indices: Sequence[int],
        active_physical_slots: Sequence[int],
        base_channels: int = 16,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if input_channels < BACKBONE_CHANNELS + PHYSICAL_PROJECTION_SLOTS:
            raise ValueError(
                "input_channels must accommodate the eleven-channel backbone "
                f"and {PHYSICAL_PROJECTION_SLOTS} physical slots"
            )
        physical_indices = tuple(int(index) for index in physical_channel_indices)
        if len(physical_indices) != PHYSICAL_PROJECTION_SLOTS:
            raise ValueError(
                "physical_channel_indices must contain exactly "
                f"{PHYSICAL_PROJECTION_SLOTS} values"
            )
        if len(set(physical_indices)) != len(physical_indices):
            raise ValueError("physical_channel_indices must be unique")
        if any(
            index < BACKBONE_CHANNELS or index >= input_channels
            for index in physical_indices
        ):
            raise ValueError(
                "physical_channel_indices must index channels after the fixed "
                "eleven-channel backbone"
            )
        active_slots = tuple(int(slot) for slot in active_physical_slots)
        if len(set(active_slots)) != len(active_slots):
            raise ValueError("active_physical_slots must be unique")
        if any(not 0 <= slot < PHYSICAL_PROJECTION_SLOTS for slot in active_slots):
            raise ValueError("active_physical_slots must be between zero and eight")

        # Construct the backbone first so a fixed random seed gives it the same
        # initialization as the established temporal control.  The additional
        # projection is deterministic because its parameters are then zeroed.
        self.backbone = TemporalAttentionUNet(
            in_channels=BACKBONE_CHANNELS,
            base_channels=base_channels,
            dropout=dropout,
            max_leads=6,
        )
        # Conv2d's constructor samples an initialization even though this
        # adapter is immediately zeroed.  Restore the CPU RNG afterward so a
        # paired seed reaches training with the same shuffle/dropout stream as
        # the unchanged control; the ablation then differs only by predictors.
        cpu_rng_state = torch.random.get_rng_state()
        self.physical_projection = nn.Conv2d(
            PHYSICAL_PROJECTION_SLOTS,
            BACKBONE_CHANNELS,
            kernel_size=1,
            bias=False,
        )
        torch.random.set_rng_state(cpu_rng_state)
        nn.init.zeros_(self.physical_projection.weight)

        active_mask = torch.zeros(PHYSICAL_PROJECTION_SLOTS, dtype=torch.float32)
        if active_slots:
            active_mask[list(active_slots)] = 1.0
        self.register_buffer("active_physical_mask", active_mask)
        self.input_channels = int(input_channels)
        self.backbone_channels = BACKBONE_CHANNELS
        self.physical_channel_indices = physical_indices
        self.active_physical_slots = active_slots

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                "expected input shape "
                f"[batch, lead, {self.input_channels}, height, width], got "
                f"{tuple(inputs.shape)}"
            )
        if not 1 <= inputs.shape[1] <= 6:
            raise ValueError("lead count must be between one and six")

    def effective_backbone_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the eleven channels supplied to the temporal backbone."""

        self._validate_input(inputs)
        batch_size, leads, _, height, width = inputs.shape
        baseline = inputs[:, :, :BACKBONE_CHANNELS]
        physical = inputs[:, :, self.physical_channel_indices]
        mask = self.active_physical_mask.to(dtype=inputs.dtype)[
            None, None, :, None, None
        ]
        physical = (physical * mask).reshape(
            batch_size * leads,
            PHYSICAL_PROJECTION_SLOTS,
            height,
            width,
        )
        projected = self.physical_projection(physical).reshape(
            batch_size,
            leads,
            BACKBONE_CHANNELS,
            height,
            width,
        )
        return baseline + projected

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.effective_backbone_inputs(inputs))


class FactorizedConvBlock3D(nn.Module):
    """One spatial then one temporal 3-D convolution."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class FixedClimatologyFactorized3DUNet(nn.Module):
    """Compact residual 3-D U-Net with spatial-only downsampling.

    Inputs use the repository-wide ``[batch, lead, channel, height, width]``
    layout.  The first eleven fixed-climatology features are transposed to
    ``[batch, channel, lead, height, width]`` internally.  Every pooling layer
    has kernel and stride ``(1, 2, 2)``, so all lead weeks remain present at
    every resolution.  Factorized ``(1, 3, 3)`` spatial and ``(3, 1, 1)``
    temporal kernels reduce capacity relative to dense 3-D convolutions.
    """

    def __init__(
        self,
        input_channels: int,
        backbone_channels: int = BACKBONE_CHANNELS,
        base_channels: int = 16,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        _validate_wrapper_channels(input_channels, backbone_channels)
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_channels = int(input_channels)
        self.backbone_channels = int(backbone_channels)
        self.base_channels = int(base_channels)

        width_1 = base_channels
        width_2 = 2 * base_channels
        width_3 = 4 * base_channels

        self.encoder_1 = FactorizedConvBlock3D(backbone_channels, width_1, dropout)
        self.encoder_2 = FactorizedConvBlock3D(width_1, width_2, dropout)
        self.bottleneck = FactorizedConvBlock3D(width_2, width_3, dropout)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.up_2 = nn.Conv3d(width_3, width_2, kernel_size=1)
        self.decoder_2 = FactorizedConvBlock3D(width_2 + width_2, width_2, dropout)
        self.up_1 = nn.Conv3d(width_2, width_1, kernel_size=1)
        self.decoder_1 = FactorizedConvBlock3D(width_1 + width_1, width_1, dropout)

        self.residual_head = nn.Conv3d(width_1, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                "expected input shape "
                f"[batch, lead, {self.input_channels}, height, width], got "
                f"{tuple(inputs.shape)}"
            )
        if inputs.shape[1] < 1:
            raise ValueError("the lead dimension must be non-empty")
        if inputs.shape[-2] < 4 or inputs.shape[-1] < 4:
            raise ValueError("height and width must both be at least 4")

    @staticmethod
    def _resize_spatially(
        inputs: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Resize latitude/longitude while preserving the lead count."""

        return F.interpolate(
            inputs,
            size=(inputs.shape[-3], reference.shape[-2], reference.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        effective = inputs[:, :, : self.backbone_channels]
        encoded_input = effective.permute(0, 2, 1, 3, 4)

        skip_1 = self.encoder_1(encoded_input)
        skip_2 = self.encoder_2(self.pool(skip_1))
        encoded = self.bottleneck(self.pool(skip_2))

        decoded = self._resize_spatially(encoded, skip_2)
        decoded = self.up_2(decoded)
        decoded = self.decoder_2(torch.cat((decoded, skip_2), dim=1))

        decoded = self._resize_spatially(decoded, skip_1)
        decoded = self.up_1(decoded)
        decoded = self.decoder_1(torch.cat((decoded, skip_1), dim=1))

        residual = self.residual_head(decoded)
        return residual[:, 0]


__all__ = [
    "CompactLeadReliabilityTemporalUNet",
    "FixedCapacityPhysicalTemporalUNet",
    "FixedClimatologyFactorized3DUNet",
    "SixHeadTemporalAttentionUNet",
    "SmoothNoiseTemporalAdapter",
]
