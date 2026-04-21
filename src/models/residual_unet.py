"""Phase 0 / Phase 2 backbone: Residual U-Net with deep supervision.

Architecture: 4-level encoder-decoder with residual skip connections.
Deep supervision: auxiliary segmentation heads at dec3 and dec2.
Loss: boundary-weighted BCE + Dice (main) + 0.3*aux3 + 0.1*aux2.
Params: 8.11M (Phase 0 static), 8.90M with TAM appended (Phase 2).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """BN -> ReLU -> Conv(in->out) -> BN -> ReLU -> Conv(out->out) + skip.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # TODO: populate from notebook
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ResidualUNetDS(nn.Module):
    """Residual U-Net with deep supervision for fetal head segmentation.

    Base channels: 32 -> 64 -> 128 -> 256 -> 512 (bottleneck).
    Input: (B, 1, 256, 384) normalized grayscale.
    Output: (B, 1, 256, 384) sigmoid logits  [+ aux heads during training].

    Args:
        base_ch: Base channel multiplier (default: 32).
        in_ch: Input channels (default: 1, grayscale).
    """

    def __init__(self, base_ch: int = 32, in_ch: int = 1) -> None:
        super().__init__()
        # TODO: populate from notebook
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """Return (bottleneck_features, skip_tuple) for TAM integration."""
        raise NotImplementedError

    def decode(
        self,
        bottleneck: torch.Tensor,
        skips: tuple,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """Decode bottleneck features back to segmentation map."""
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits during inference, (logits, aux3, aux2) during training."""
        raise NotImplementedError
