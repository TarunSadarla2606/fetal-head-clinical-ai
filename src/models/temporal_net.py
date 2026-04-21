"""Phase 2 / Phase 4b: Temporal segmentation with bottleneck self-attention.

TemporalAttentionModule (TAM): spatial avg-pool over bottleneck features,
projection to 256-dim attention space, multi-head self-attention (8 heads),
FFN with GELU, projection back and sigmoid gating of bottleneck features.

TemporalFetaSegNet: shared 2D encoder + TAM + shared decoder.
Input:  [B, T, 1, H, W]  (T=16 frames)
Output: [B, T, 1, H, W] per-frame logits  +  [T, T] attention weights

Three-stage training on 806 synthetic clips (564/121/121 train/val/test):
  Stage 1: TAM only (backbone frozen)     lr=3e-4  8 ep  → Val Dice 96.03%
  Stage 2: TAM + decoder (enc frozen)     lr=1e-4  15 ep → Val Dice 96.24%
  Stage 3: Full fine-tune                 lr=3e-5  30 ep → Val Dice 96.29%
"""

import torch
import torch.nn as nn
from typing import Tuple


class TemporalAttentionModule(nn.Module):
    """Lightweight self-attention over T bottleneck feature vectors.

    Spatial avg-pool → Linear(bn_ch→256) + positional encoding →
    LayerNorm → MHA(8 heads) → residual → LayerNorm → FFN(256→512→256, GELU)
    → residual → Linear(256→bn_ch) → Sigmoid → element-wise gate.

    Args:
        bottleneck_ch: Bottleneck channel count (512 for baseline, 257 for Phase 4b).
        n_frames:      Temporal window size (default: 16).
        n_heads:       Number of attention heads (default: 8).
        dropout:       Dropout rate in attention and FFN (default: 0.1).
    """

    def __init__(
        self,
        bottleneck_ch: int = 512,
        n_frames: int = 16,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bottleneck_ch = bottleneck_ch
        self.attn_dim = 256

        self.proj_in = nn.Linear(bottleneck_ch, self.attn_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_frames, self.attn_dim) * 0.02
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.attn_dim)
        self.norm2 = nn.LayerNorm(self.attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.attn_dim, self.attn_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.attn_dim * 2, self.attn_dim),
        )
        self.proj_out = nn.Sequential(
            nn.Linear(self.attn_dim, bottleneck_ch),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.proj_out[0].bias, 1.0)

    def forward(
        self, bottleneck_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal attention gating to bottleneck sequence.

        Args:
            bottleneck_seq: [B, T, C, h, w] bottleneck features.

        Returns:
            attended: Gated bottleneck [B, T, C, h, w].
            attn_weights: Attention matrix [T, T].
        """
        B, T, C, h, w = bottleneck_seq.shape

        # Spatial avg-pool + project + add positional encoding
        x = self.proj_in(bottleneck_seq.mean(dim=(-2, -1)))
        x = x + self.pos_encoding[:, :T, :]

        # Self-attention with pre-norm
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))

        # Gate bottleneck features
        gate = self.proj_out(x).unsqueeze(-1).unsqueeze(-1)  # [B, T, C, 1, 1]
        return bottleneck_seq * gate, attn_weights


class TemporalFetaSegNet(nn.Module):
    """Shared 2D ResUNet encoder + TAM + shared decoder for cine-clip segmentation.

    The backbone's encode() and decode() methods are called frame-by-frame
    (after flattening B×T), with TAM applied at the bottleneck sequence.

    Args:
        backbone:  ResidualUNetDS instance (Phase 0 pretrained).
        attn:      TemporalAttentionModule instance.
    """

    def __init__(self, backbone, attn: TemporalAttentionModule) -> None:
        super().__init__()
        self.backbone = backbone
        self.attn     = attn

    def forward(
        self, clip: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment all frames with temporal attention.

        Args:
            clip: [B, T, 1, H, W] cine clip.

        Returns:
            logits:       [B, T, 1, H, W] per-frame segmentation logits.
            attn_weights: [T, T] temporal attention matrix.
        """
        B, T, C, H, W = clip.shape

        # Encode all frames with shared backbone
        flat = clip.view(B * T, C, H, W)
        bns, skips = self.backbone.encode(flat)
        _, Cb, hb, wb = bns.shape

        # Apply temporal attention at bottleneck
        bn_attended, attn_weights = self.attn(bns.view(B, T, Cb, hb, wb))

        # Decode all frames with shared backbone
        logits = self.backbone.decode(
            bn_attended.view(B * T, Cb, hb, wb), skips
        ).view(B, T, 1, H, W)

        return logits, attn_weights
