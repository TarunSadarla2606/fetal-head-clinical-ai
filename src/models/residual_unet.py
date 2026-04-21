"""Phase 0 / Phase 2 backbone: Pre-activation Residual U-Net with deep supervision.

Architecture uses pre-activation residual blocks (BN → ReLU → Conv, following
He et al. 2016 Identity Mappings). Deep supervision adds auxiliary segmentation
heads at dec3 and dec2 to improve gradient flow to early encoder layers.

Params: 8.11M (base_ch=32). Input: [B, 1, 256, 384] per-image z-score normalised.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN→ReLU→Conv→BN→ReLU→Conv + skip.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ResidualUNetDS(nn.Module):
    """Residual U-Net with deep supervision for fetal head segmentation.

    Base channel progression: 1 → 32 → 64 → 128 → 256 → 512 (bottleneck).
    Auxiliary heads at dec3 (128-ch) and dec2 (64-ch) provide deep supervision.

    During training:  forward() returns (main_logits, aux_d2_sigmoid, aux_d3_sigmoid).
    During inference: forward() returns main_logits only.

    encode() and decode() are split for use with TemporalFetaSegNet.

    Args:
        in_ch:   Input channels (default: 1, grayscale).
        base_ch: Base channel multiplier (default: 32).
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        b = base_ch
        self.enc1      = ResidualBlock(in_ch,   b)
        self.enc2      = ResidualBlock(b,        b * 2)
        self.enc3      = ResidualBlock(b * 2,    b * 4)
        self.enc4      = ResidualBlock(b * 4,    b * 8)
        self.pool      = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(b * 8,   b * 16)

        self.up4  = nn.ConvTranspose2d(b * 16, b * 8,  2, stride=2)
        self.dec4 = ResidualBlock(b * 16, b * 8)
        self.up3  = nn.ConvTranspose2d(b * 8,  b * 4,  2, stride=2)
        self.dec3 = ResidualBlock(b * 8,  b * 4)
        self.up2  = nn.ConvTranspose2d(b * 4,  b * 2,  2, stride=2)
        self.dec2 = ResidualBlock(b * 4,  b * 2)
        self.up1  = nn.ConvTranspose2d(b * 2,  b,      2, stride=2)
        self.dec1 = ResidualBlock(b * 2,  b)

        self.final  = nn.Conv2d(b,     1, 1)
        self.aux_d3 = nn.Conv2d(b * 4, 1, 1)  # after dec3
        self.aux_d2 = nn.Conv2d(b * 2, 1, 1)  # after dec2

    # ------------------------------------------------------------------
    # encode / decode split — used by TemporalFetaSegNet
    # ------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Run encoder and return (bottleneck, (e1, e2, e3, e4)) skip tuple."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        return bn, (e1, e2, e3, e4)

    def decode(
        self,
        bn: torch.Tensor,
        skips: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Run decoder from bottleneck + skip connections. Returns main logits."""
        e1, e2, e3, e4 = skips
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

    # ------------------------------------------------------------------
    # full forward — used for standalone training / inference
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits (inference) or (logits, aux_d2, aux_d3) (training)."""
        bn, (e1, e2, e3, e4) = self.encode(x)
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        if self.training:
            aux3 = torch.sigmoid(
                F.interpolate(self.aux_d3(d3), size=x.shape[2:],
                              mode="bilinear", align_corners=False)
            )
            aux2 = torch.sigmoid(
                F.interpolate(self.aux_d2(d2), size=x.shape[2:],
                              mode="bilinear", align_corners=False)
            )
            return self.final(d1), aux2, aux3

        return self.final(d1)


class BoundaryWeightedDiceLoss(nn.Module):
    """Dice loss with higher weight on boundary pixels (Sobel-detected edges).

    Args:
        smooth:          Smoothing constant (default: 1.0).
        boundary_weight: Multiplier for edge pixels in weight map (default: 2.0).
    """

    def __init__(self, smooth: float = 1.0, boundary_weight: float = 2.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.boundary_weight = boundary_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat   = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        kernel = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=target.device,
        ).view(1, 1, 3, 3) / 8.0
        edges    = F.conv2d(target.unsqueeze(1), kernel, padding=1).squeeze()
        edge_mask = (edges.abs() > 0.1).float()
        weights   = 1.0 + self.boundary_weight * edge_mask.view(-1)

        intersection = (pred_flat * target_flat * weights).sum()
        union        = (pred_flat * weights).sum() + (target_flat * weights).sum()
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


def deep_supervision_loss(
    main_logits: torch.Tensor,
    aux2: torch.Tensor,
    aux3: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.3),
) -> torch.Tensor:
    """Combined deep supervision loss: main + 0.5*aux2 + 0.3*aux3.

    Args:
        main_logits: Main decoder output [B, 1, H, W].
        aux2:        Auxiliary head 2 output (sigmoid applied) [B, 1, H, W].
        aux3:        Auxiliary head 3 output (sigmoid applied) [B, 1, H, W].
        masks:       Binary ground-truth masks [B, 1, H, W].
        criterion:   Loss function (e.g. BoundaryWeightedDiceLoss).
        weights:     (main, aux2, aux3) loss weights.
    """
    w_main, w_aux2, w_aux3 = weights
    pred_sigmoid = torch.sigmoid(main_logits)
    return (
        w_main * criterion(pred_sigmoid, masks)
        + w_aux2 * criterion(aux2, masks)
        + w_aux3 * criterion(aux3, masks)
    )
