"""Phase 4a / Phase 4b: Pruned model definitions.

Hybrid crossover filter synthesis: retains the original pre-pruning
intermediate channel width in PrunedResidualBlock (the 'mid' dimension),
only the output width is pruned. This preserves representational capacity
in the filter bank while reducing the output dimensionality.

Channel counts are loaded from checkpoint state_dict at inference time
via _resolve_channel_counts() to avoid hard-coding pruned widths.

Phase 4a (pruned static):   8.11M → 4.57M params  (−43.7%)
Phase 4b (pruned temporal): 8.90M → 5.20M params  (−41.6%)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class PrunedResidualBlock(nn.Module):
    """Post-pruning residual block with preserved intermediate width.

    Architecture: BN -> ReLU -> Conv(in->mid) -> BN -> ReLU -> Conv(mid->out) + skip.
    The mid_ch stays at the original pre-pruning width (CrossoverFilter property).
    The skip connection is always a 1x1 Conv (in_ch != out_ch after pruning).

    Args:
        in_ch: Input channels.
        mid_ch: Intermediate channels (pre-pruning width, preserved).
        out_ch: Output channels (post-pruning width).
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        # TODO: populate from notebook
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PrunedResidualUNetDS(nn.Module):
    """Pruned Residual U-Net loaded from a hybrid crossover pruning checkpoint.

    Unpruned layers: enc1, enc2, dec2, dec1 (base 32/64 channels, stable).
    Pruned layers:   enc3, enc4, bottleneck, dec4, dec3 (irregular widths).

    Channel counts are inferred from checkpoint keys at load time:
        channel_counts = {
            'enc3':       int,   # e.g. 71
            'enc4':       int,   # e.g. 129
            'bottleneck': int,   # e.g. 257
            'dec4':       int,   # e.g. 129
            'dec3':       int,   # e.g. 65
        }

    Args:
        channel_counts: Dict of pruned output widths per layer.
        base_ch: Unpruned base channel count (default: 32).
        aux_d3_ch: Aux head input channels (inferred from checkpoint).
    """

    def __init__(
        self,
        channel_counts: Dict[str, int],
        base_ch: int = 32,
        aux_d3_ch: Optional[int] = None,
    ) -> None:
        super().__init__()
        # TODO: populate from notebook
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "PrunedResidualUNetDS":
        """Load pruned model from checkpoint, inferring channel counts automatically."""
        raise NotImplementedError
