"""Compact probabilistic residual U-Net for global Quest precipitation.

The network shares one spatial correction operator across the two forecast
periods.  It predicts five residual logits per grid cell and adds them to the
log-probabilities of an ensemble anchor.  The correction head is initialized
to zero, so a newly constructed model reproduces a valid anchor distribution.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


PROBABILITY_CATEGORIES = 5
FORECAST_PERIODS = 2
ANCHOR_EPSILON = 1.0e-8


def _group_count(channels: int, maximum_groups: int = 8) -> int:
    """Return the largest GroupNorm group count that divides ``channels``."""

    for groups in range(min(maximum_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _GlobalConv3x3(nn.Module):
    """A 3x3 convolution with periodic longitude and bounded latitude."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=0,
            bias=False,
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        # Latitude must not wrap from one pole to the other. Longitude is a
        # periodic coordinate, including at the two latitude-padding rows.
        fields = F.pad(fields, (0, 0, 1, 1), mode="replicate")
        fields = F.pad(fields, (1, 1, 0, 0), mode="circular")
        return self.conv(fields)


class _ConvBlock(nn.Module):
    """Two padded convolutions with GroupNorm, SiLU, and spatial dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            _GlobalConv3x3(in_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            _GlobalConv3x3(out_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.block(fields)


class TPProbUNet(nn.Module):
    """Predict calibrated global TP quintile probabilities.

    Parameters
    ----------
    in_channels:
        Predictor channels per forecast period. The approved feature contract
        contains 18 channels.
    base_channels:
        First encoder width. The default produces widths 16, 32, and 64.
    dropout:
        Spatial dropout probability in every convolutional block.
    Notes
    -----
    Inputs have shape ``[batch, 2, 18, latitude, longitude]``. Forecast periods
    are folded into the batch dimension, so every period uses exactly the same
    U-Net weights. The five output categories are ordered from the lowest to
    the highest precipitation quintile.
    """

    def __init__(
        self,
        in_channels: int = 18,
        base_channels: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.dropout = float(dropout)

        width_1 = self.base_channels
        width_2 = 2 * self.base_channels
        width_3 = 4 * self.base_channels

        self.encoder_1 = _ConvBlock(self.in_channels, width_1, self.dropout)
        self.encoder_2 = _ConvBlock(width_1, width_2, self.dropout)
        self.bottleneck = _ConvBlock(width_2, width_3, self.dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_2 = nn.Conv2d(width_3, width_2, kernel_size=1)
        self.decoder_2 = _ConvBlock(width_2 + width_2, width_2, self.dropout)
        self.up_1 = nn.Conv2d(width_2, width_1, kernel_size=1)
        self.decoder_1 = _ConvBlock(width_1 + width_1, width_1, self.dropout)

        self.correction_head = nn.Conv2d(
            width_1, PROBABILITY_CATEGORIES, kernel_size=1
        )
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                "x must have shape [batch, period, channel, latitude, longitude]"
            )
        if x.shape[1] != FORECAST_PERIODS:
            raise ValueError(f"x must contain exactly {FORECAST_PERIODS} periods")
        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"x has {x.shape[2]} channels; expected {self.in_channels}"
            )
        if x.shape[-2] < 4 or x.shape[-1] < 4:
            raise ValueError("latitude and longitude dimensions must be at least 4")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")

    def _validate_inputs(self, x: torch.Tensor, p0: torch.Tensor) -> None:
        self._validate_x(x)
        expected_anchor_shape = (
            x.shape[0],
            FORECAST_PERIODS,
            PROBABILITY_CATEGORIES,
            x.shape[-2],
            x.shape[-1],
        )
        if tuple(p0.shape) != expected_anchor_shape:
            raise ValueError(
                f"p0 must have shape {expected_anchor_shape}; got {tuple(p0.shape)}"
            )
        if not p0.is_floating_point():
            raise TypeError("p0 must be a floating-point tensor")

    def _correction_logits(self, x: torch.Tensor) -> torch.Tensor:
        batch, periods, channels, height, width = x.shape
        fields = x.reshape(batch * periods, channels, height, width)

        skip_1 = self.encoder_1(fields)
        skip_2 = self.encoder_2(self.pool(skip_1))
        encoded = self.bottleneck(self.pool(skip_2))

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

        correction = self.correction_head(decoded)
        return correction.reshape(
            batch, periods, PROBABILITY_CATEGORIES, height, width
        )

    def forward_corrections(self, x: torch.Tensor) -> torch.Tensor:
        """Return the five additive correction logits for each period and cell."""

        self._validate_x(x)
        return self._correction_logits(x)

    def forward(
        self,
        x: torch.Tensor,
        p0: torch.Tensor,
    ) -> torch.Tensor:
        """Return calibrated quintile probabilities with the same grid as ``p0``."""

        self._validate_inputs(x, p0)
        correction = self._correction_logits(x)
        anchor_logits = torch.log(p0.clamp_min(ANCHOR_EPSILON))
        logits = anchor_logits + correction
        return torch.softmax(logits, dim=2)


__all__ = ["TPProbUNet"]
