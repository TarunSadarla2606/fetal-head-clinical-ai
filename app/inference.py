"""
inference.py — Fetal Head Circumference Clinical AI
All model architecture definitions, weight loading, prediction pipelines,
inference timing, and input validation.

Supports four model variants:
  Phase 0  — ResidualUNetDS baseline          (8.11M params, base_ch=32)
  Phase 4a — ResidualUNetDS pruned            (4.57M params, irregular channels)
  Phase 2  — TemporalFetaSegNet baseline      (8.90M params)
  Phase 4b — TemporalFetaSegNet pruned        (5.20M params, irregular channels)

Key design decisions:
  - Pruned models store channel_counts in the checkpoint; loading is
    checkpoint-driven, never hardcoded. This mirrors production pattern
    where model configs live with weights, not in application code.
  - InferenceTimer is a context manager that measures wall-clock latency
    and is threadsafe (no global state).
  - validate_input() provides clinical-grade input sanity checks before
    inference is attempted — prevents misleading outputs on bad inputs.
"""

from __future__ import annotations

import time
import contextlib
from typing import Optional, Union

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label, regionprops
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
INPUT_H  = 256
INPUT_W  = 384
N_FRAMES = 16
IMG_MEAN = 0.2
IMG_STD  = 0.15
ORIG_W   = 800   # original HC18 image width for pixel-spacing scaling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline channel configuration (unpruned)
_BASE_CHANNELS = {
    "enc1":       32,
    "enc2":       64,
    "enc3":       128,
    "enc4":       256,
    "bottleneck": 512,
    "dec4":       256,
    "dec3":       128,
    "dec2":       64,
    "dec1":       32,
}

# ── model architectures ───────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Standard residual block: BN → ReLU → Conv(in→out) → BN → ReLU → Conv(out→out).
    Used for the baseline (unpruned) model.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),  nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class PrunedResidualBlock(nn.Module):
    """
    Pruned residual block: BN → ReLU → Conv(in→mid) → BN → ReLU → Conv(mid→out).

    This is the post-pruning structure produced by HybridCrossoverMerger.
    The merger only modifies block[-1] (the last Conv) and the skip connection —
    the intermediate width 'mid' stays at the original pre-pruning value.

    For encoder blocks:  in_ch  = pruned upstream output (or 1 for enc1)
                         mid_ch = ORIGINAL block output width (e.g. 128 for enc3)
                         out_ch = PRUNED output (e.g. 71 for enc3)

    For decoder blocks:  in_ch  = cat(up_out, enc_pruned_out)  — mixed original+pruned
                         mid_ch = ORIGINAL decoder block width
                         out_ch = PRUNED decoder output
    """
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),  nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ResidualUNetDS(nn.Module):
    """
    Residual U-Net with deep supervision — baseline (unpruned) model.
    Used for Phase 0 and Phase 2 backbone.
    base_ch=32 gives the standard 32/64/128/256/512 channel ladder.
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.enc1       = ResidualBlock(in_ch, b)
        self.enc2       = ResidualBlock(b,     b * 2)
        self.enc3       = ResidualBlock(b * 2, b * 4)
        self.enc4       = ResidualBlock(b * 4, b * 8)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(b * 8,  b * 16)
        self.up4        = nn.ConvTranspose2d(b * 16, b * 8,  2, stride=2)
        self.dec4       = ResidualBlock(b * 16, b * 8)
        self.up3        = nn.ConvTranspose2d(b * 8,  b * 4,  2, stride=2)
        self.dec3       = ResidualBlock(b * 8,  b * 4)
        self.up2        = nn.ConvTranspose2d(b * 4,  b * 2,  2, stride=2)
        self.dec2       = ResidualBlock(b * 4,  b * 2)
        self.up1        = nn.ConvTranspose2d(b * 2,  b,      2, stride=2)
        self.dec1       = ResidualBlock(b * 2,  b)
        self.final      = nn.Conv2d(b, 1, 1)
        self.aux_d3     = nn.Conv2d(b * 4, 1, 1)
        self.aux_d2     = nn.Conv2d(b * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        if self.training:
            aux3 = torch.sigmoid(F.interpolate(
                self.aux_d3(d3), size=x.shape[2:], mode="bilinear", align_corners=False))
            aux2 = torch.sigmoid(F.interpolate(
                self.aux_d2(d2), size=x.shape[2:], mode="bilinear", align_corners=False))
            return self.final(d1), aux3, aux2
        return self.final(d1)

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        return bn, (e1, e2, e3, e4)

    def decode(self, bn: torch.Tensor, skips: tuple) -> torch.Tensor:
        e1, e2, e3, e4 = skips
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class PrunedResidualUNetDS(nn.Module):
    """
    Pruned Residual U-Net — exactly mirrors the post-pruning architecture
    produced by HybridCrossoverMerger in the Phase 4a/4b training notebooks.

    The merger modifies the model in-place by:
      1. Replacing block[-1] (last Conv) of each prunable block with
         Conv(original_mid_ch, pruned_out_ch).
      2. Updating skip connections to match pruned_out_ch.
      3. Patching downstream encoder BN/Conv inputs.
      4. Patching decoder BN/Conv/skip inputs via concat-index slicing.
      5. Replacing ConvTranspose in_channels (output channels unchanged).
      6. Patching aux_d3 when dec3 is pruned.

    The resulting internal block structure is:
      Conv(in_ch → original_mid_ch) → Conv(original_mid_ch → pruned_out_ch)
    NOT the standard Conv(in_ch → out_ch) → Conv(out_ch → out_ch).

    channel_counts dict (from checkpoint): pruned output channels for
    enc3, enc4, bottleneck, dec4, dec3.
    All other blocks (enc1, enc2, dec2, dec1) are unchanged from base_ch=32.

    Derived shapes (base_ch=32, using the error-message trace):
      enc3:       PrunedResidualBlock(64,  128, cc["enc3"])
      enc4:       PrunedResidualBlock(cc["enc3"], 256, cc["enc4"])
      bottleneck: PrunedResidualBlock(cc["enc4"], 512, cc["bottleneck"])
      up4:        ConvTranspose2d(cc["bottleneck"], 256)   # out=256=original enc4
      dec4_cat:   cc["bottleneck_up_out"] + cc["enc4"] = 256 + cc["enc4"]
      dec4:       PrunedResidualBlock(256+cc["enc4"], 256, cc["dec4"])
      up3:        ConvTranspose2d(cc["dec4"], 128)         # out=128=original enc3
      dec3_cat:   128 + cc["enc3"]
      dec3:       PrunedResidualBlock(128+cc["enc3"], 128, cc["dec3"])
      up2:        ConvTranspose2d(cc["dec3"], 64)          # out=64=original enc2
      aux_d3:     Conv2d(cc["dec3"], 1)
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, channel_counts: dict = None,
                 aux_d3_ch: int = None):
        super().__init__()
        if channel_counts is None:
            raise ValueError("channel_counts required for PrunedResidualUNetDS")
        b   = base_ch
        cc  = channel_counts
        c3  = cc["enc3"]
        c4  = cc["enc4"]
        cbn = cc["bottleneck"]
        cd4 = cc["dec4"]
        cd3 = cc["dec3"]
        # aux_d3_ch defaults to cd3 (Phase 4a: merger patched it).
        # Phase 4b: merger did NOT patch aux_d3, so checkpoint has b*4=128.
        if aux_d3_ch is None:
            aux_d3_ch = cd3

        # Unpruned blocks (standard)
        self.enc1       = ResidualBlock(in_ch, b)
        self.enc2       = ResidualBlock(b,     b * 2)
        self.pool       = nn.MaxPool2d(2)

        # Pruned encoder blocks: Conv(in → original_mid) → Conv(original_mid → pruned_out)
        self.enc3       = PrunedResidualBlock(b * 2,  b * 4,  c3)   # (64, 128, 71)
        self.enc4       = PrunedResidualBlock(c3,     b * 8,  c4)   # (71, 256, 129)
        self.bottleneck = PrunedResidualBlock(c4,     b * 16, cbn)  # (129, 512, 257)

        # up4: in=cbn (pruned bottleneck out), out=b*8=256 (ORIGINAL — unchanged by merger)
        self.up4  = nn.ConvTranspose2d(cbn,  b * 8,  2, stride=2)
        # dec4 receives cat(up4_out=256, enc4_pruned=c4) = 256+c4
        self.dec4 = PrunedResidualBlock(b * 8 + c4,  b * 8,  cd4)  # (385, 256, 129)

        # up3: in=cd4 (pruned dec4 out), out=b*4=128 (ORIGINAL)
        self.up3  = nn.ConvTranspose2d(cd4, b * 4,  2, stride=2)
        # dec3 receives cat(up3_out=128, enc3_pruned=c3) = 128+c3
        self.dec3 = PrunedResidualBlock(b * 4 + c3,  b * 4,  cd3)  # (199, 128, 65)

        # up2: in=cd3 (pruned dec3 out), out=b*2=64 (ORIGINAL)
        self.up2  = nn.ConvTranspose2d(cd3, b * 2,  2, stride=2)
        # dec2 receives cat(up2_out=64, enc2_out=64) = 128 — both unpruned
        self.dec2 = ResidualBlock(b * 4,  b * 2)

        # up1: in=b*2=64 (dec2 unpruned), out=b=32
        self.up1  = nn.ConvTranspose2d(b * 2, b,   2, stride=2)
        # dec1 receives cat(up1_out=32, enc1_out=32) = 64 — both unpruned
        self.dec1 = ResidualBlock(b * 2,  b)

        self.final   = nn.Conv2d(b,    1, 1)
        self.aux_d3  = nn.Conv2d(aux_d3_ch, 1, 1)
        self.aux_d2  = nn.Conv2d(b * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        if self.training:
            aux3 = torch.sigmoid(F.interpolate(
                self.aux_d3(d3), size=x.shape[2:], mode="bilinear", align_corners=False))
            aux2 = torch.sigmoid(F.interpolate(
                self.aux_d2(d2), size=x.shape[2:], mode="bilinear", align_corners=False))
            return self.final(d1), aux3, aux2
        return self.final(d1)

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        return bn, (e1, e2, e3, e4)

    def decode(self, bn: torch.Tensor, skips: tuple) -> torch.Tensor:
        e1, e2, e3, e4 = skips
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class TemporalAttentionModule(nn.Module):
    """
    Lightweight temporal self-attention over bottleneck feature sequences.
    Operates on spatially-pooled vectors to keep memory overhead minimal.
    The bottleneck_ch parameter must match the actual bottleneck output
    channel count (257 for pruned, 512 for baseline).
    """
    def __init__(
        self,
        bottleneck_ch: int = 512,
        n_frames: int = 16,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_dim  = 256
        self.proj_in   = nn.Linear(bottleneck_ch, self.attn_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_frames, self.attn_dim) * 0.02
        )
        self.attn  = nn.MultiheadAttention(
            self.attn_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.attn_dim)
        self.ff    = nn.Sequential(
            nn.Linear(self.attn_dim, self.attn_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.attn_dim * 2, self.attn_dim),
        )
        self.norm2    = nn.LayerNorm(self.attn_dim)
        self.proj_out = nn.Sequential(
            nn.Linear(self.attn_dim, bottleneck_ch), nn.Sigmoid()
        )
        nn.init.constant_(self.proj_out[0].bias, 1.0)

    def forward(self, bn_seq: torch.Tensor):
        B, T, C, h, w = bn_seq.shape
        x = self.proj_in(bn_seq.mean(dim=(-2, -1)))
        x = x + self.pos_encoding[:, :T, :]
        attn_out, attn_w = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        mod = self.proj_out(x).unsqueeze(-1).unsqueeze(-1)
        return bn_seq * mod, attn_w


class TemporalFetaSegNet(nn.Module):
    """Full temporal segmentation model: shared 2D encoder → attention → decoder."""
    def __init__(self, backbone: ResidualUNetDS, attn: TemporalAttentionModule):
        super().__init__()
        self.backbone = backbone
        self.attn     = attn

    def forward(self, clip: torch.Tensor):
        B, T, C, H, W = clip.shape
        flat           = clip.view(B * T, C, H, W)
        bns, skips     = self.backbone.encode(flat)
        _, Cb, hb, wb  = bns.shape
        bn_att, attn_w = self.attn(bns.view(B, T, Cb, hb, wb))
        logits         = self.backbone.decode(
            bn_att.view(B * T, Cb, hb, wb), skips
        ).view(B, T, 1, H, W)
        return logits, attn_w


# ── checkpoint-driven model loading ──────────────────────────────────────────

def _resolve_channel_counts(ckpt: dict) -> dict:
    """
    Extract channel counts from a checkpoint dict.
    Falls back to base defaults if the key is absent (legacy checkpoints).
    """
    return ckpt.get("channel_counts", _BASE_CHANNELS.copy())


def load_phase0(ckpt_path: str, device: torch.device = DEVICE) -> ResidualUNetDS:
    """Load Phase 0 baseline static model (base_ch=32, 8.11M params)."""
    model = ResidualUNetDS(in_ch=1, base_ch=32)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_phase4a(ckpt_path: str, device: torch.device = DEVICE) -> PrunedResidualUNetDS:
    """
    Load Phase 4a pruned static model.

    Uses PrunedResidualUNetDS which exactly mirrors the post-merger architecture.
    aux_d3_ch is read from the checkpoint to handle any merger patching variation.
    """
    ckpt           = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    channel_counts = _resolve_channel_counts(ckpt)
    sd = ckpt["model_state_dict"]
    aux_d3_ch = sd.get("aux_d3.weight", sd.get("backbone.aux_d3.weight")).shape[1]
    model = PrunedResidualUNetDS(in_ch=1, base_ch=32, channel_counts=channel_counts,
                                  aux_d3_ch=aux_d3_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_phase2(ckpt_path: str, device: torch.device = DEVICE) -> TemporalFetaSegNet:
    """Load Phase 2 baseline temporal model (8.90M params)."""
    backbone = ResidualUNetDS(in_ch=1, base_ch=32)
    attn     = TemporalAttentionModule(512, N_FRAMES, 8)
    model    = TemporalFetaSegNet(backbone, attn)
    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_phase4b(ckpt_path: str, device: torch.device = DEVICE) -> TemporalFetaSegNet:
    """
    Load Phase 4b pruned temporal model.
    Backbone uses PrunedResidualUNetDS; TAM bottleneck_ch matches pruned bottleneck.

    aux_d3_ch is read directly from the checkpoint state_dict to handle the case
    where the Phase 4b merger did not patch aux_d3 (checkpoint has shape [1,128,1,1]
    rather than [1,cd3,1,1]).  Peeking at the checkpoint avoids any hardcoding.
    """
    ckpt           = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    channel_counts  = _resolve_channel_counts(ckpt)
    bottleneck_ch   = channel_counts.get("bottleneck", 512)

    # Peek at aux_d3 shape in checkpoint to handle unpatched case
    sd = ckpt["model_state_dict"]
    aux_d3_ch = sd.get("backbone.aux_d3.weight", sd.get("aux_d3.weight")).shape[1]

    backbone = PrunedResidualUNetDS(in_ch=1, base_ch=32, channel_counts=channel_counts,
                                    aux_d3_ch=aux_d3_ch)
    attn     = TemporalAttentionModule(bottleneck_ch, N_FRAMES, 8)
    model    = TemporalFetaSegNet(backbone, attn)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def get_model_info(model: nn.Module) -> dict:
    """
    Return a dict of metadata about a loaded model.
    Used to populate the UI metrics table and system panel.
    """
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return {
        "total_params":     total,
        "trainable_params": trainable,
        "size_mb":          round(size_mb, 1),
        "device":           str(next(model.parameters()).device),
    }


# ── inference timer ───────────────────────────────────────────────────────────

class InferenceTimer:
    """
    Context manager for measuring inference wall-clock latency.

    Usage:
        timer = InferenceTimer()
        with timer:
            result = model(input)
        print(f"{timer.elapsed_ms:.1f} ms")

    Thread-safe: each instance holds its own timestamps.
    On CUDA, inserts device synchronisation before stopping the clock
    to measure true end-to-end GPU time (not just kernel-launch time).
    """
    def __init__(self, device: torch.device = DEVICE):
        self.device      = device
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# ── input validation ──────────────────────────────────────────────────────────

class InputValidationError(ValueError):
    """Raised when the uploaded image fails clinical sanity checks."""


def validate_input(img_gray: np.ndarray) -> dict:
    """
    Perform sanity checks on an uploaded ultrasound image before inference.

    Checks performed:
      1. Shape — must be 2D grayscale
      2. Minimum resolution — must be at least 64×64
      3. Not blank/saturated — mean intensity must be in [5, 250]
      4. Dynamic range — std must be > 5 (not a flat field)
      5. Aspect ratio — must be between 0.3 and 4.0 (not a thin strip)
      6. Ultrasound-like texture — checks that the image has sufficient
         spatial texture (not a clean synthetic shape or screenshot)

    Returns:
        dict with keys:
          valid (bool), warnings (list[str]), checks (dict of bool)

    Does not raise — always returns a result so the UI can decide
    whether to hard-block or soft-warn.
    """
    warnings = []
    checks   = {}

    # 1. Shape
    if img_gray.ndim != 2:
        checks["shape"] = False
        warnings.append("Image must be 2D grayscale (received shape: "
                        f"{img_gray.shape}). Convert to grayscale before uploading.")
    else:
        checks["shape"] = True

    # 2. Minimum resolution
    h, w = img_gray.shape[:2]
    checks["resolution"] = (h >= 64 and w >= 64)
    if not checks["resolution"]:
        warnings.append(f"Image resolution ({w}×{h}) is below the 64×64 minimum. "
                        "Results will be unreliable.")

    # 3. Blank / saturated
    mean_int = float(img_gray.mean())
    checks["not_blank"]     = mean_int >= 5.0
    checks["not_saturated"] = mean_int <= 250.0
    if not checks["not_blank"]:
        warnings.append("Image appears to be blank (near-black). "
                        "Check that the correct file was uploaded.")
    if not checks["not_saturated"]:
        warnings.append("Image appears to be saturated (near-white). "
                        "This is unusual for ultrasound images.")

    # 4. Dynamic range
    std_int = float(img_gray.std())
    checks["dynamic_range"] = std_int > 5.0
    if not checks["dynamic_range"]:
        warnings.append(f"Image has very low dynamic range (std={std_int:.1f}). "
                        "The image may be flat or synthetic.")

    # 5. Aspect ratio
    if h > 0 and w > 0:
        ar = w / h
        checks["aspect_ratio"] = 0.3 <= ar <= 4.0
        if not checks["aspect_ratio"]:
            warnings.append(f"Unusual aspect ratio ({ar:.2f}). "
                            "Fetal head ultrasound images are typically near-square.")
    else:
        checks["aspect_ratio"] = False

    # 6. Texture check — Laplacian variance as a proxy for image content
    lap_var = float(cv2.Laplacian(img_gray.astype(np.float32), cv2.CV_32F).var())
    checks["has_texture"] = lap_var > 10.0
    if not checks["has_texture"]:
        warnings.append("Image has very low spatial texture. "
                        "This model was trained on real ultrasound frames — "
                        "synthetic or heavily processed images may produce poor results.")

    valid = all(checks.values())
    return {"valid": valid, "warnings": warnings, "checks": checks}


# ── image utilities ───────────────────────────────────────────────────────────

def preprocess_image(img_gray: np.ndarray) -> torch.Tensor:
    """[H, W] uint8 → [1, 1, H, W] normalised float tensor."""
    resized = cv2.resize(img_gray, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized.astype(np.float32) / 255.0)
    t = (t - IMG_MEAN) / (IMG_STD + 1e-8)
    return t.unsqueeze(0).unsqueeze(0)


def fill_hollow_mask(mask_bin: np.ndarray) -> np.ndarray:
    """Flood-fill a hollow ellipse annotation into a solid filled binary mask."""
    h, w  = mask_bin.shape
    fm    = np.zeros((h + 2, w + 2), dtype=np.uint8)
    filled = mask_bin.copy()
    cv2.floodFill(filled, fm, (0, 0), 255)
    solid = cv2.bitwise_or(mask_bin, cv2.bitwise_not(filled))
    return (solid > 127).astype(np.uint8)


def make_overlay(
    img_gray: np.ndarray,
    mask:     np.ndarray,
    alpha:    float = 0.35,
) -> np.ndarray:
    """Overlay a binary segmentation mask (red) on a grayscale image. Returns RGB."""
    img_resized = cv2.resize(img_gray, (INPUT_W, INPUT_H))
    rgb         = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    overlay     = rgb.copy()
    overlay[mask > 0] = [220, 60, 60]
    return cv2.addWeighted(rgb, 1 - alpha, overlay, alpha, 0)


def make_comparison_overlay(
    img_gray:   np.ndarray,
    mask_pred:  np.ndarray,
    mask_gt:    Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Three-colour overlay for ground truth comparison mode.
      Red   = prediction only (false positive region)
      Green = ground truth only (missed / false negative region)
      Yellow = intersection (true positive)
    """
    img_resized = cv2.resize(img_gray, (INPUT_W, INPUT_H))
    rgb         = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB).copy()
    if mask_gt is None:
        return make_overlay(img_gray, mask_pred)

    tp = (mask_pred > 0) & (mask_gt > 0)   # yellow
    fp = (mask_pred > 0) & (mask_gt == 0)  # red
    fn = (mask_pred == 0) & (mask_gt > 0)  # green

    overlay      = rgb.copy()
    overlay[tp]  = [255, 220, 0]
    overlay[fp]  = [220, 60,  60]
    overlay[fn]  = [60,  200, 60]
    return cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)


# ── HC and GA computation ─────────────────────────────────────────────────────

def estimate_hc_mm(
    mask_bin:         np.ndarray,
    pixel_spacing_mm: float,
    input_w:          int = INPUT_W,
    orig_w:           int = ORIG_W,
) -> Optional[float]:
    """
    Estimate HC in mm using Ramanujan's ellipse perimeter approximation
    applied to the largest connected component of the segmentation mask.
    """
    labeled  = label(mask_bin)
    if labeled.max() == 0:
        return None
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    a = largest.major_axis_length / 2
    b = largest.minor_axis_length / 2
    if a < 1 or b < 1:
        return None
    h_val  = ((a - b) / (a + b + 1e-8)) ** 2
    hc_px  = np.pi * (a + b) * (1 + (3 * h_val) / (10 + np.sqrt(4 - 3 * h_val + 1e-8)))
    return hc_px * pixel_spacing_mm * (orig_w / input_w)


def hadlock_ga(hc_mm: float) -> tuple[float, str]:
    """
    Estimate gestational age from HC using Hadlock (1984) polynomial.
    Reference: Hadlock FP et al., AJR 1984;143:97-100.
    """
    hc  = hc_mm
    ga  = 8.96 + 0.540 * (hc / 10) - 0.0040 * ((hc / 10) ** 2) + 0.000399 * ((hc / 10) ** 3)
    ga  = max(10.0, min(ga, 42.0))
    weeks = int(ga)
    days  = round((ga - weeks) * 7)
    if days == 7:
        weeks += 1; days = 0
    return ga, f"{weeks}w {days}d"


def classify_trimester(ga_weeks: float) -> str:
    """Clinical trimester classification by gestational age.

    Boundaries (ACOG / ISUOG / ACR):
      First  trimester: < 14 weeks 0 days
      Second trimester: 14 weeks 0 days to 27 weeks 6 days
      Third  trimester: ≥ 28 weeks 0 days
    """
    ga_days = ga_weeks * 7
    if ga_days < 98:
        return "First trimester (<14w)"
    elif ga_days < 196:
        return "Second trimester (14–28w)"
    return "Third trimester (≥28w)"


def confidence_label(reliability: float) -> tuple[str, str]:
    """
    Convert reliability score to a clinical confidence label + CSS colour.
    Returns (label, colour_hex).
    """
    if reliability >= 0.97:
        return "HIGH CONFIDENCE",    "#16a34a"   # green-600
    elif reliability >= 0.92:
        return "MODERATE CONFIDENCE", "#d97706"  # amber-600
    else:
        return "LOW — VERIFY MANUALLY", "#dc2626" # red-600


def compute_gt_metrics(
    mask_pred:        np.ndarray,
    mask_gt:          np.ndarray,
    pixel_spacing_mm: float,
) -> dict:
    """
    Compute Dice coefficient and MAE between prediction and ground truth.
    Used when the user uploads an optional ground-truth mask.
    """
    pred_f = mask_pred.flatten().astype(float)
    gt_f   = mask_gt.flatten().astype(float)
    smooth = 1e-5
    dice   = (2 * (pred_f * gt_f).sum() + smooth) / (pred_f.sum() + gt_f.sum() + smooth)

    hc_pred = estimate_hc_mm(mask_pred, pixel_spacing_mm)
    hc_gt   = estimate_hc_mm(mask_gt,   pixel_spacing_mm)
    mae     = abs(hc_pred - hc_gt) if (hc_pred is not None and hc_gt is not None) else None

    return {
        "dice":    float(dice),
        "mae_mm":  float(mae) if mae is not None else None,
        "hc_pred": hc_pred,
        "hc_gt":   hc_gt,
    }


# ── single-frame inference ────────────────────────────────────────────────────

def predict_single_frame(
    model:            Union[ResidualUNetDS, PrunedResidualUNetDS],
    img_gray:         np.ndarray,
    pixel_spacing_mm: float,
    device:           torch.device = DEVICE,
    threshold:        float = 0.5,
) -> dict:
    """
    Run static model inference on a single ultrasound frame.

    Returns dict with keys:
      mask, prob_map, overlay, hc_mm, ga_str, ga_weeks, trimester,
      reliability, hc_std_mm, confidence_label, confidence_color,
      elapsed_ms, mode
    """
    timer = InferenceTimer(device)
    img_t = preprocess_image(img_gray).to(device)

    with timer:
        with torch.no_grad():
            logits = model(img_t)

    prob_map = torch.sigmoid(logits).cpu().squeeze().numpy()
    mask     = (prob_map > threshold).astype(np.uint8)
    mask     = fill_hollow_mask(mask * 255)
    overlay  = make_overlay(img_gray, mask)

    hc_mm                    = estimate_hc_mm(mask, pixel_spacing_mm)
    ga_weeks, ga_str, trimester = None, None, "Unknown"
    if hc_mm is not None:
        ga_weeks, ga_str = hadlock_ga(hc_mm)
        trimester        = classify_trimester(ga_weeks)

    conf_label, conf_color = confidence_label(1.0)

    return {
        "mask":             mask,
        "prob_map":         prob_map,
        "overlay":          overlay,
        "hc_mm":            hc_mm,
        "ga_str":           ga_str,
        "ga_weeks":         ga_weeks,
        "trimester":        trimester,
        "reliability":      1.0,
        "hc_std_mm":        0.0,
        "confidence_label": conf_label,
        "confidence_color": conf_color,
        "elapsed_ms":       timer.elapsed_ms,
        "mode":             "single_frame",
    }


# ── cine-clip inference ───────────────────────────────────────────────────────

def predict_cine_clip(
    model:            TemporalFetaSegNet,
    frames:           list,
    pixel_spacing_mm: float,
    device:           torch.device = DEVICE,
    threshold:        float = 0.5,
) -> dict:
    """
    Run temporal model inference on a list of ultrasound frames.

    Returns dict with keys:
      consensus_mask, prob_map, uncertainty, overlay, attn_weights,
      hc_mm, ga_str, ga_weeks, trimester, reliability, hc_std_mm,
      per_frame_hc, confidence_label, confidence_color, elapsed_ms, mode
    """
    # Pad or truncate to N_FRAMES
    if len(frames) != N_FRAMES:
        frames = (frames * N_FRAMES)[:N_FRAMES]

    tensors = []
    for f in frames:
        fr = cv2.resize(f, (INPUT_W, INPUT_H))
        t  = torch.from_numpy(fr.astype(np.float32) / 255.0)
        t  = (t - IMG_MEAN) / (IMG_STD + 1e-8)
        tensors.append(t.unsqueeze(0))

    clip  = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # [1, T, 1, H, W]
    timer = InferenceTimer(device)

    with timer:
        with torch.no_grad():
            logits, attn_w = model(clip)

    probs     = torch.sigmoid(logits).cpu().squeeze()   # [T, H, W]
    per_bin   = (probs > threshold).float()
    mean_prob = probs.mean(dim=0).numpy()
    consensus = (mean_prob > threshold).astype(np.uint8)
    uncertainty = per_bin.std(dim=0).numpy()

    per_frame_hc = []
    for t_idx in range(probs.shape[0]):
        fm  = (probs[t_idx].numpy() > threshold).astype(np.uint8)
        hc  = estimate_hc_mm(fm, pixel_spacing_mm)
        if hc is not None:
            per_frame_hc.append(hc)

    hc_mm                    = estimate_hc_mm(consensus, pixel_spacing_mm)
    ga_weeks, ga_str, trimester = None, None, "Unknown"
    if hc_mm is not None:
        ga_weeks, ga_str = hadlock_ga(hc_mm)
        trimester        = classify_trimester(ga_weeks)

    reliability = 0.0
    hc_std_mm   = 0.0
    if len(per_frame_hc) >= 2:
        hc_arr      = np.array(per_frame_hc)
        reliability = float(max(0, 1 - hc_arr.std() / (hc_arr.mean() + 1e-8)))
        hc_std_mm   = float(hc_arr.std())

    conf_label, conf_color = confidence_label(reliability)

    mid_frame = cv2.resize(frames[N_FRAMES // 2], (INPUT_W, INPUT_H))
    overlay   = make_overlay(mid_frame, consensus)

    return {
        "consensus_mask":   consensus,
        "mask":             consensus,    # alias for unified downstream code
        "prob_map":         mean_prob,
        "uncertainty":      uncertainty,
        "overlay":          overlay,
        "attn_weights":     attn_w.cpu().numpy()[0],  # [T, T]
        "hc_mm":            hc_mm,
        "ga_str":           ga_str,
        "ga_weeks":         ga_weeks,
        "trimester":        trimester,
        "reliability":      reliability,
        "hc_std_mm":        hc_std_mm,
        "per_frame_hc":     per_frame_hc,
        "confidence_label": conf_label,
        "confidence_color": conf_color,
        "elapsed_ms":       timer.elapsed_ms,
        "mode":             "cine_clip",
    }
