"""Small spatial and spatiotemporal residual adapters for weekly FuXi fields.

The models consume a sequence of gridded feature maps with shape
``[batch, lead, channel, latitude, longitude]`` and predict one additive
residual map per lead.  The final convolution is initialized to zero.  This
makes a newly constructed adapter an exact no-op when its residual is added
to the FuXi baseline, which is a useful and safe starting point for training.
"""

import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum_groups: int = 8) -> int:
    """Return the largest valid GroupNorm group count up to ``maximum_groups``."""

    for groups in range(min(maximum_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Two 3x3 convolutions followed by GroupNorm, SiLU, and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUNet(nn.Module):
    """A compact two-level U-Net shared across all forecast lead weeks.

    Lead weeks are folded into the batch dimension for the convolutional
    encoder and decoder.  Consequently, the same spatial correction operator
    is used at every lead, while lead-dependent input features can still tell
    the network which week it is processing.

    Parameters
    ----------
    in_channels:
        Number of predictor channels per lead week.
    base_channels:
        Width of the first encoder level.  The default gives widths
        16, 32, and 64.
    dropout:
        Spatial dropout probability in each convolutional block.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_channels = in_channels
        self.base_channels = base_channels

        width_1 = base_channels
        width_2 = 2 * base_channels
        width_3 = 4 * base_channels

        self.encoder_1 = ConvBlock(in_channels, width_1, dropout)
        self.encoder_2 = ConvBlock(width_1, width_2, dropout)
        self.bottleneck = ConvBlock(width_2, width_3, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_2 = nn.Conv2d(width_3, width_2, kernel_size=1)
        self.decoder_2 = ConvBlock(width_2 + width_2, width_2, dropout)
        self.up_1 = nn.Conv2d(width_2, width_1, kernel_size=1)
        self.decoder_1 = ConvBlock(width_1 + width_1, width_1, dropout)

        self.residual_head = nn.Conv2d(width_1, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                "expected input shape [batch, lead, channel, height, width], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[2]}"
            )
        if x.shape[1] < 1:
            raise ValueError("the lead dimension must be non-empty")
        if x.shape[-2] < 4 or x.shape[-1] < 4:
            raise ValueError("height and width must both be at least 4")

    def _transform_bottleneck(
        self, bottleneck: torch.Tensor, batch_size: int, leads: int
    ) -> torch.Tensor:
        """Hook for lead-aware processing in spatiotemporal subclasses."""

        del batch_size, leads
        return bottleneck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        batch_size, leads, channels, height, width = x.shape
        flat = x.reshape(batch_size * leads, channels, height, width)

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

        residual = self.residual_head(decoded)
        return residual[:, 0].reshape(batch_size, leads, height, width)


class TemporalAttentionUNet(ResidualUNet):
    """Residual U-Net with attention across lead weeks at each map location.

    After spatial encoding, each bottleneck grid location becomes an
    independent sequence of lead-week tokens.  A single Transformer encoder
    layer mixes information along that six-week sequence; it never mixes
    distinct examples or spatial locations.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        dropout: float = 0.1,
        max_leads: int = 6,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
        )
        if max_leads < 1:
            raise ValueError("max_leads must be positive")

        bottleneck_channels = 4 * base_channels
        if bottleneck_channels % 4 != 0:
            raise ValueError("four attention heads must divide bottleneck channels")

        self.max_leads = max_leads
        self.lead_position = nn.Parameter(
            torch.zeros(1, max_leads, bottleneck_channels)
        )
        nn.init.normal_(self.lead_position, mean=0.0, std=0.02)
        self.temporal_attention = nn.TransformerEncoderLayer(
            d_model=bottleneck_channels,
            nhead=4,
            dim_feedforward=2 * bottleneck_channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def _transform_bottleneck(
        self, bottleneck: torch.Tensor, batch_size: int, leads: int
    ) -> torch.Tensor:
        if leads > self.max_leads:
            raise ValueError(
                f"received {leads} leads, but max_leads is {self.max_leads}"
            )

        _, channels, height, width = bottleneck.shape
        tokens = bottleneck.reshape(batch_size, leads, channels, height, width)
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, leads, channels
        )
        tokens = tokens + self.lead_position[:, :leads]
        tokens = self.temporal_attention(tokens)
        return (
            tokens.reshape(batch_size, height, width, leads, channels)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch_size * leads, channels, height, width)
        )


class _LeadMixer(nn.Module):
    """Mix a short lead sequence independently at every grid location."""

    def __init__(
        self,
        channels: int,
        *,
        max_leads: int,
        layers: int,
        heads: int,
        feedforward_channels: int,
        dropout: float,
        layer_scale: float,
    ) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("attention heads must divide mixer channels")
        if layers < 1:
            raise ValueError("mixer layers must be positive")
        if layer_scale <= 0.0:
            raise ValueError("layer_scale must be positive")

        self.max_leads = max_leads
        self.position = nn.Parameter(torch.empty(1, max_leads, channels))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=channels,
                    nhead=heads,
                    dim_feedforward=feedforward_channels,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.scale = nn.Parameter(torch.full((channels,), float(layer_scale)))

    def forward(
        self,
        fields: torch.Tensor,
        *,
        batch_size: int,
        leads: int,
    ) -> torch.Tensor:
        if leads > self.max_leads:
            raise ValueError(
                f"received {leads} leads, but max_leads is {self.max_leads}"
            )
        _, channels, height, width = fields.shape
        tokens = fields.reshape(batch_size, leads, channels, height, width)
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, leads, channels
        )
        original = tokens
        mixed = tokens + self.position[:, :leads]
        for layer in self.layers:
            mixed = layer(mixed)
        mixed = original + self.scale.view(1, 1, -1) * (mixed - original)
        return (
            mixed.reshape(batch_size, height, width, leads, channels)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch_size * leads, channels, height, width)
        )


class MultiScaleLateLeadTemporalUNet(ResidualUNet):
    """Large regularized temporal U-Net for Week 5--6 residual correction.

    Lead attention is applied both at the second encoder scale and at the
    bottleneck. Small learnable LayerScale gates make both mixers near-identity
    at initialization. The first four output weeks are always exact zeros.
    """

    required_leads = 6
    first_active_lead = 4

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 48,
        spatial_dropout: float = 0.25,
        temporal_dropout: float = 0.20,
        skip_layers: int = 2,
        bottleneck_layers: int = 3,
        attention_heads: int = 8,
        layer_scale: float = 1.0e-3,
        context_dropout: float = 0.10,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=spatial_dropout,
        )
        if not 0.0 <= context_dropout < 1.0:
            raise ValueError("context_dropout must be in [0, 1)")
        self.context_dropout = float(context_dropout)
        self.skip_mixer = _LeadMixer(
            2 * base_channels,
            max_leads=self.required_leads,
            layers=skip_layers,
            heads=attention_heads,
            feedforward_channels=8 * base_channels,
            dropout=temporal_dropout,
            layer_scale=layer_scale,
        )
        self.bottleneck_mixer = _LeadMixer(
            4 * base_channels,
            max_leads=self.required_leads,
            layers=bottleneck_layers,
            heads=attention_heads,
            feedforward_channels=16 * base_channels,
            dropout=temporal_dropout,
            layer_scale=layer_scale,
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        super()._validate_input(x)
        if x.shape[1] != self.required_leads:
            raise ValueError(
                "multi_scale_late_lead_temporal_unet requires exactly six lead "
                f"weeks; got {x.shape[1]}"
            )

    def _drop_context(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.context_dropout == 0.0:
            return x
        keep = torch.rand(
            x.shape[0],
            self.first_active_lead,
            1,
            1,
            1,
            device=x.device,
        ) >= self.context_dropout
        result = x.clone()
        result[:, : self.first_active_lead] *= keep.to(dtype=x.dtype)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        x = self._drop_context(x)
        batch_size, leads, channels, height, width = x.shape
        flat = x.reshape(batch_size * leads, channels, height, width)

        skip_1 = self.encoder_1(flat)
        skip_2 = self.encoder_2(self.pool(skip_1))
        skip_2 = self.skip_mixer(skip_2, batch_size=batch_size, leads=leads)
        encoded = self.bottleneck(self.pool(skip_2))
        encoded = self.bottleneck_mixer(
            encoded, batch_size=batch_size, leads=leads
        )

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

        residual = self.residual_head(decoded)
        residual = residual[:, 0].reshape(batch_size, leads, height, width)
        inactive = torch.zeros_like(residual[:, : self.first_active_lead])
        return torch.cat((inactive, residual[:, self.first_active_lead :]), dim=1)


class MultiScaleAllLeadTemporalUNet(MultiScaleLateLeadTemporalUNet):
    """Large multi-scale temporal U-Net that corrects all six lead weeks."""

    first_active_lead = 0


class FixedClimatologyMultiScaleUNet(nn.Module):
    """Normal-climatology control for an appended climatology-candidate bank."""

    def __init__(self, input_channels: int = 29, backbone_channels: int = 11) -> None:
        super().__init__()
        if input_channels < backbone_channels:
            raise ValueError("input_channels must include the backbone channels")
        self.input_channels = input_channels
        self.backbone_channels = backbone_channels
        self.backbone = MultiScaleLateLeadTemporalUNet(
            in_channels=backbone_channels,
            base_channels=48,
            spatial_dropout=0.25,
            temporal_dropout=0.20,
            skip_layers=2,
            bottleneck_layers=3,
            attention_heads=8,
            layer_scale=1.0e-3,
            context_dropout=0.10,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                f"expected [batch, lead, {self.input_channels}, height, width]"
            )
        return self.backbone(inputs[:, :, : self.backbone_channels])


class ClimatologyAttentionConditioner(nn.Module):
    """Choose a convex, forecast-conditioned seasonal climatology.

    The input contains eleven standard channels, followed by nine normalized
    log-climatology maps and their nine corresponding normalized FuXi anomaly
    maps. Attention is local to each grid cell and convex over the nine maps.
    """

    query_channels = (0, 1, 3, 4, 5, 6, 7, 10)
    normal_climatology_channel = 2
    normal_anomaly_channel = 9
    candidate_count = 9
    first_bank_channel = 11

    def __init__(
        self,
        offsets_days: Sequence[int] = (-28, -21, -14, -7, 0, 7, 14, 21, 28),
        heads: int = 4,
        head_channels: int = 8,
        initial_gate: float = 0.05,
        first_active_lead: int = 4,
    ) -> None:
        super().__init__()
        offsets = tuple(int(value) for value in offsets_days)
        if len(offsets) != self.candidate_count or 0 not in offsets:
            raise ValueError("offsets_days must contain nine values including zero")
        if heads < 1 or head_channels < 1:
            raise ValueError("heads and head_channels must be positive")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be in (0, 1)")
        if not 0 <= first_active_lead < 6:
            raise ValueError("first_active_lead must be in [0, 5]")
        channels = heads * head_channels
        self.heads = heads
        self.head_channels = head_channels
        self.query_projection = nn.Conv2d(len(self.query_channels), channels, 1)
        self.key_projection = nn.Conv2d(3, channels, 1)
        self.log_temperature = nn.Parameter(torch.zeros(heads))
        gate_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.lead_gate_logit = nn.Parameter(torch.full((6,), gate_logit))
        angle = 2.0 * math.pi * torch.tensor(offsets, dtype=torch.float32) / 365.2425
        self.register_buffer("offset_sin", torch.sin(angle), persistent=True)
        self.register_buffer("offset_cos", torch.cos(angle), persistent=True)
        self.register_buffer(
            "active_leads",
            torch.tensor(
                [0.0] * first_active_lead + [1.0] * (6 - first_active_lead),
                dtype=torch.float32,
            ),
            persistent=True,
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[2] != 29:
            raise ValueError("attention conditioner expects 29 input channels")
        batch, leads, _, height, width = inputs.shape
        if leads != 6:
            raise ValueError("attention conditioner requires six lead weeks")

        query = inputs[:, :, self.query_channels].reshape(
            batch * leads, len(self.query_channels), height, width
        )
        query = self.query_projection(query).reshape(
            batch, leads, self.heads, self.head_channels, height, width
        )
        climatology_bank = inputs[
            :, :, self.first_bank_channel : self.first_bank_channel + self.candidate_count
        ]
        anomaly_bank = inputs[
            :,
            :,
            self.first_bank_channel
            + self.candidate_count : self.first_bank_channel
            + 2 * self.candidate_count,
        ]
        offset_sin = self.offset_sin.view(1, 1, -1, 1, 1).expand(
            batch, leads, -1, height, width
        )
        offset_cos = self.offset_cos.view(1, 1, -1, 1, 1).expand_as(offset_sin)
        key_input = torch.stack((climatology_bank, offset_sin, offset_cos), dim=3)
        keys = self.key_projection(
            key_input.reshape(batch * leads * self.candidate_count, 3, height, width)
        ).reshape(
            batch,
            leads,
            self.candidate_count,
            self.heads,
            self.head_channels,
            height,
            width,
        )
        score = (query.unsqueeze(2) * keys).sum(dim=4) / math.sqrt(self.head_channels)
        temperature = torch.nn.functional.softplus(self.log_temperature) + 1.0e-4
        score = score / temperature.view(1, 1, 1, self.heads, 1, 1)
        head_weights = torch.softmax(score, dim=2)
        weights = head_weights.mean(dim=3)
        attentive_climatology = (weights * climatology_bank).sum(dim=2)
        attentive_anomaly = (weights * anomaly_bank).sum(dim=2)

        gate = torch.sigmoid(self.lead_gate_logit) * self.active_leads
        gate_field = gate.view(1, leads, 1, 1)
        effective = inputs[:, :, :11].clone()
        effective[:, :, self.normal_climatology_channel] = (
            inputs[:, :, self.normal_climatology_channel]
            + gate_field
            * (
                attentive_climatology
                - inputs[:, :, self.normal_climatology_channel]
            )
        )
        effective[:, :, self.normal_anomaly_channel] = (
            inputs[:, :, self.normal_anomaly_channel]
            + gate_field
            * (attentive_anomaly - inputs[:, :, self.normal_anomaly_channel])
        )
        return effective, weights, gate


class AttentiveClimatologyMultiScaleUNet(nn.Module):
    """Large temporal adapter conditioned on a training-only climatology bank."""

    def __init__(self, input_channels: int = 29) -> None:
        super().__init__()
        if input_channels != 29:
            raise ValueError("attentive climatology model requires 29 input channels")
        self.input_channels = input_channels
        self.conditioner = ClimatologyAttentionConditioner()
        self.backbone = MultiScaleLateLeadTemporalUNet(
            in_channels=11,
            base_channels=48,
            spatial_dropout=0.25,
            temporal_dropout=0.20,
            skip_layers=2,
            bottleneck_layers=3,
            attention_heads=8,
            layer_scale=1.0e-3,
            context_dropout=0.10,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        effective, _, _ = self.conditioner(inputs)
        return self.backbone(effective)


class FixedClimatologyAllLeadMultiScaleUNet(nn.Module):
    """Large all-week temporal control using fixed climatology channels."""

    def __init__(
        self,
        input_channels: int = 29,
        backbone_channels: int = 11,
        spatial_dropout: float = 0.25,
        temporal_dropout: float = 0.20,
        context_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_channels < backbone_channels:
            raise ValueError("input_channels must include the backbone channels")
        self.input_channels = input_channels
        self.backbone_channels = backbone_channels
        self.backbone = MultiScaleAllLeadTemporalUNet(
            in_channels=backbone_channels,
            base_channels=48,
            spatial_dropout=spatial_dropout,
            temporal_dropout=temporal_dropout,
            skip_layers=2,
            bottleneck_layers=3,
            attention_heads=8,
            layer_scale=1.0e-3,
            context_dropout=context_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                f"expected [batch, lead, {self.input_channels}, height, width]"
            )
        return self.backbone(inputs[:, :, : self.backbone_channels])


class AttentiveClimatologyAllLeadMultiScaleUNet(nn.Module):
    """Large all-week adapter with forecast-conditioned climatology."""

    def __init__(
        self,
        input_channels: int = 29,
        spatial_dropout: float = 0.25,
        temporal_dropout: float = 0.20,
        context_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_channels != 29:
            raise ValueError("attentive climatology model requires 29 input channels")
        self.input_channels = input_channels
        self.conditioner = ClimatologyAttentionConditioner(first_active_lead=0)
        self.backbone = MultiScaleAllLeadTemporalUNet(
            in_channels=11,
            base_channels=48,
            spatial_dropout=spatial_dropout,
            temporal_dropout=temporal_dropout,
            skip_layers=2,
            bottleneck_layers=3,
            attention_heads=8,
            layer_scale=1.0e-3,
            context_dropout=context_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        effective, _, _ = self.conditioner(inputs)
        return self.backbone(effective)


class LateLeadTemporalUNet(TemporalAttentionUNet):
    """Temporal adapter that can correct only forecast Weeks 3--6.

    The spatial encoder, decoder, and lead-wise attention are identical to
    :class:`TemporalAttentionUNet`.  The first two output maps are replaced by
    exact zeros after decoding, so adding this adapter to a baseline cannot
    alter Weeks 1--2.  All six input weeks remain available to temporal
    attention when it constructs the active Week 3--6 corrections.
    """

    required_leads = 6
    first_active_lead = 2

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            max_leads=self.required_leads,
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        super()._validate_input(x)
        if x.shape[1] != self.required_leads:
            raise ValueError(
                "late_lead_temporal_unet requires exactly six lead weeks; "
                f"got {x.shape[1]}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = super().forward(x)
        inactive = torch.zeros_like(residual[:, : self.first_active_lead])
        return torch.cat((inactive, residual[:, self.first_active_lead :]), dim=1)


class FixedClimatologyLateLeadUNet(nn.Module):
    """Regularized Week 3--6 control using the fixed climatology channels."""

    def __init__(
        self,
        input_channels: int = 29,
        backbone_channels: int = 11,
        base_channels: int = 16,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if input_channels < backbone_channels:
            raise ValueError("input_channels must include the backbone channels")
        self.input_channels = input_channels
        self.backbone_channels = backbone_channels
        self.backbone = LateLeadTemporalUNet(
            in_channels=backbone_channels,
            base_channels=base_channels,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                f"expected [batch, lead, {self.input_channels}, height, width]"
            )
        return self.backbone(inputs[:, :, : self.backbone_channels])


class AttentiveClimatologyLateLeadUNet(nn.Module):
    """Regularized Week 3--6 adapter with forecast-conditioned climatology."""

    def __init__(
        self,
        input_channels: int = 29,
        base_channels: int = 16,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if input_channels != 29:
            raise ValueError("attentive climatology model requires 29 input channels")
        self.input_channels = input_channels
        self.conditioner = ClimatologyAttentionConditioner(first_active_lead=2)
        self.backbone = LateLeadTemporalUNet(
            in_channels=11,
            base_channels=base_channels,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        effective, _, _ = self.conditioner(inputs)
        return self.backbone(effective)


class FixedClimatologyAllLeadUNet(nn.Module):
    """Regularized temporal control that corrects all six forecast weeks."""

    def __init__(
        self,
        input_channels: int = 29,
        backbone_channels: int = 11,
        base_channels: int = 16,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if input_channels < backbone_channels:
            raise ValueError("input_channels must include the backbone channels")
        self.input_channels = input_channels
        self.backbone_channels = backbone_channels
        self.backbone = TemporalAttentionUNet(
            in_channels=backbone_channels,
            base_channels=base_channels,
            dropout=dropout,
            max_leads=6,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[2] != self.input_channels:
            raise ValueError(
                f"expected [batch, lead, {self.input_channels}, height, width]"
            )
        return self.backbone(inputs[:, :, : self.backbone_channels])


class AttentiveClimatologyAllLeadUNet(nn.Module):
    """All-week temporal adapter with forecast-conditioned climatology."""

    def __init__(
        self,
        input_channels: int = 29,
        base_channels: int = 16,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if input_channels != 29:
            raise ValueError("attentive climatology model requires 29 input channels")
        self.input_channels = input_channels
        self.conditioner = ClimatologyAttentionConditioner(first_active_lead=0)
        self.backbone = TemporalAttentionUNet(
            in_channels=11,
            base_channels=base_channels,
            dropout=dropout,
            max_leads=6,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        effective, _, _ = self.conditioner(inputs)
        return self.backbone(effective)


def build_model(name: str, in_channels: int, **kwargs: Any) -> nn.Module:
    """Construct a supported adapter by its configuration name."""

    normalized_name = name.strip().lower().replace("-", "_")
    if normalized_name in {"residual_unet", "unet"}:
        return ResidualUNet(in_channels=in_channels, **kwargs)
    if normalized_name in {
        "temporal_attention_unet",
        "temporal_unet",
        "attention_unet",
    }:
        return TemporalAttentionUNet(in_channels=in_channels, **kwargs)
    if normalized_name in {
        "late_lead_temporal_unet",
        "late_lead_attention_unet",
        "late_lead_unet",
    }:
        return LateLeadTemporalUNet(in_channels=in_channels, **kwargs)
    if normalized_name in {
        "multi_scale_late_lead_temporal_unet",
        "multiscale_late_lead_temporal_unet",
        "big_temporal_unet",
    }:
        return MultiScaleLateLeadTemporalUNet(in_channels=in_channels, **kwargs)
    if normalized_name in {
        "multi_scale_all_lead_temporal_unet",
        "multiscale_all_lead_temporal_unet",
        "big_allweek_temporal_unet",
    }:
        return MultiScaleAllLeadTemporalUNet(in_channels=in_channels, **kwargs)
    if normalized_name in {
        "fixed_climatology_multiscale_unet",
        "fixed_climatology_temporal_unet",
    }:
        return FixedClimatologyMultiScaleUNet(input_channels=in_channels, **kwargs)
    if normalized_name in {
        "attentive_climatology_multiscale_unet",
        "attention_climatology_temporal_unet",
    }:
        return AttentiveClimatologyMultiScaleUNet(input_channels=in_channels, **kwargs)
    if normalized_name in {
        "fixed_climatology_all_lead_multiscale_unet",
        "fixed_climatology_big_allweek_unet",
    }:
        return FixedClimatologyAllLeadMultiScaleUNet(
            input_channels=in_channels, **kwargs
        )
    if normalized_name in {
        "attentive_climatology_all_lead_multiscale_unet",
        "attention_climatology_big_allweek_unet",
    }:
        return AttentiveClimatologyAllLeadMultiScaleUNet(
            input_channels=in_channels, **kwargs
        )
    if normalized_name in {
        "fixed_climatology_late_lead_unet",
        "fixed_climatology_week36_unet",
    }:
        return FixedClimatologyLateLeadUNet(input_channels=in_channels, **kwargs)
    if normalized_name in {
        "attentive_climatology_late_lead_unet",
        "attention_climatology_week36_unet",
    }:
        return AttentiveClimatologyLateLeadUNet(input_channels=in_channels, **kwargs)
    if normalized_name in {
        "fixed_climatology_all_lead_unet",
        "fixed_climatology_allweeks_unet",
    }:
        return FixedClimatologyAllLeadUNet(input_channels=in_channels, **kwargs)
    if normalized_name in {
        "attentive_climatology_all_lead_unet",
        "attention_climatology_allweeks_unet",
    }:
        return AttentiveClimatologyAllLeadUNet(input_channels=in_channels, **kwargs)
    raise ValueError(
        f"unknown model {name!r}; expected 'residual_unet' or "
        "'temporal_attention_unet', 'late_lead_temporal_unet', or "
        "'multi_scale_late_lead_temporal_unet'"
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count scalar model parameters, optionally excluding frozen parameters."""

    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
