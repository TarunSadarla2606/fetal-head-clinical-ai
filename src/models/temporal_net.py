"""Phase 2 / Phase 4b: Temporal segmentation with bottleneck self-attention.

TemporalAttentionModule: lightweight (~200K param) self-attention over the
bottleneck feature sequence from a shared 2D encoder.

TemporalFetaSegNet: shared 2D ResUNet encoder + TAM + shared decoder.
Processes 16-frame clips; outputs per-frame logits + T×T attention weights.
Params: 8.90M (Phase 2), 5.20M (Phase 4b — pruned backbone, TAM preserved).
"""

import torch
import torch.nn as nn
from typing import Tuple


class TemporalAttentionModule(nn.Module):
    """Lightweight temporal self-attention over bottleneck feature sequences.

    Spatial avg-pool -> project to attn_dim -> positional encoding ->
    multi-head self-attention (8 heads) -> FFN -> project back -> sigmoid gate.

    Args:
        bottleneck_ch: Bottleneck channel count (512 for baseline, 257 for pruned).
        attn_dim: Internal attention dimension (default: 256).
        n_heads: Number of attention heads (default: 8).
        n_frames: Temporal sequence length (default: 16).
    """

    def __init__(
        self,
        bottleneck_ch: int = 512,
        attn_dim: int = 256,
        n_heads: int = 8,
        n_frames: int = 16,
    ) -> None:
        super().__init__()
        # TODO: populate from notebook
        raise NotImplementedError

    def forward(
        self, bn_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal attention gating.

        Args:
            bn_seq: Bottleneck features [B, T, C, h, w].

        Returns:
            modulated: Gated bottleneck [B, T, C, h, w].
            attn_weights: Attention matrix [T, T].
        """
        raise NotImplementedError


class TemporalFetaSegNet(nn.Module):
    """Shared 2D encoder + TAM + shared decoder for cine-clip segmentation.

    Input:  clip [B, T, 1, H, W]  (T=16 frames, H=256, W=384)
    Output: logits [B, T, 1, H, W], attn_weights [T, T]

    Training stages:
        1. Frozen encoder (train TAM + decoder only)
        2. Partial unfreeze (enc3, enc4, bottleneck)
        3. Full fine-tune (all layers)

    Args:
        backbone: Pre-trained ResidualUNetDS instance (Phase 0 weights).
        n_frames: Temporal window size (default: 16).
    """

    def __init__(
        self,
        backbone,
        n_frames: int = 16,
    ) -> None:
        super().__init__()
        # TODO: populate from notebook
        raise NotImplementedError

    def forward(
        self, clip: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment all frames in clip with temporal attention.

        Args:
            clip: [B, T, 1, H, W] cine clip.

        Returns:
            logits: [B, T, 1, H, W] per-frame segmentation logits.
            attn_weights: [T, T] temporal attention matrix.
        """
        raise NotImplementedError
