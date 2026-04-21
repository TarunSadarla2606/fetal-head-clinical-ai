"""Phase 4a / Phase 4b: Hybrid Crossover structural pruning engine.

Pruning is performed on the live ResidualUNetDS / TemporalFetaSegNet model
in-place using HybridCrossoverMerger. There is no separate "PrunedUNet" class
for training — the merger modifies the existing model's weight tensors.

For loading pruned checkpoints at inference time, see app/inference.py
(PrunedResidualUNetDS / PrunedResidualBlock), which handles irregular channel
counts inferred from the saved state_dict.

---

Hybrid Crossover merging (per dropped channel):
  1. Collect activations of keep and drop channels on calibration set.
  2. Compute hybrid target (max-pool or adaptive blend of both maps).
  3. Regress new filter weights for keep channel via 50-step Adam to match target.
  4. Remove drop channel; replace keep filter with regressed weights.
  5. Propagate new channel counts to downstream blocks via surgical index slicing.

ILR Importance Scoring:
  ILR(ch) = [0.6 × RMS_activation + 0.4 × filter_L1_norm + 0.2 × Frobenius_norm] / 1.2

Pruning schedule:
  - 3 prune-FT cycles
  - Burst-sequential pruning per cycle (adaptive burst size)
  - Inter-cycle KD fine-tune: 15 epochs, AdamW, CosineAnnealingLR
  - KD: α=0.5, T=4.0, teacher = frozen baseline

Guard rails:
  - Dice drop ≤ 4pp from baseline
  - MAE increase ≤ 1.5 mm from baseline
  - Hard channel floors: enc3≥64, enc4≥128, bottleneck≥256, dec4≥128, dec3≥64

Phase 4b additional step:
  When bottleneck is pruned, TAM proj_in and proj_out linear layers are resized
  via weight slicing (not reinitialization) to match the new bottleneck channel count.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional

ILR_WEIGHTS = {"rms": 0.6, "bn_gamma": 0.4, "hrank": 0.2}

HARD_FLOORS = {
    "enc3":       64,
    "enc4":       128,
    "bottleneck": 256,
    "dec4":       128,
    "dec3":       64,
}

PRUNABLE_BLOCKS = ["enc3", "enc4", "bottleneck", "dec4", "dec3"]


def compute_ilr_scores(
    model: nn.Module,
    calib_pairs: list,
    n_batches: int = 8,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Compute ILR importance scores for all prunable blocks.

    ILR = [0.6×RMS_activation + 0.4×filter_L1_norm + 0.2×Frobenius_norm] / 1.2
    Lower score = less important channel = pruning candidate.

    Args:
        model:       ResidualUNetDS or TemporalFetaSegNet backbone.
        calib_pairs: List of (img_path, mask_path) tuples for calibration.
        n_batches:   Number of images to use for activation statistics.
        device:      Torch device string.

    Returns:
        Dict mapping block name → per-channel importance tensor.
    """
    model.eval()
    blocks = {
        "enc3":       {"conv": model.enc3.block[-1]},
        "enc4":       {"conv": model.enc4.block[-1]},
        "bottleneck": {"conv": model.bottleneck.block[-1]},
        "dec4":       {"conv": model.dec4.block[-1]},
        "dec3":       {"conv": model.dec3.block[-1]},
    }
    act_accum  = {k: None for k in blocks}
    frob_accum = {k: None for k in blocks}
    act_count  = {k: 0    for k in blocks}
    hooks: list = []

    def make_hook(name: str):
        def h(m, inp, out):
            out_d = out.detach().float()
            rms   = out_d.pow(2).mean(dim=[0, 2, 3])
            frob  = out_d.pow(2).sum(dim=[2, 3]).sqrt().mean(dim=0)
            if act_accum[name] is None:
                act_accum[name]  = torch.zeros(rms.shape)
                frob_accum[name] = torch.zeros(frob.shape)
            act_accum[name]  += rms.cpu()
            frob_accum[name] += frob.cpu()
            act_count[name]  += 1
        return h

    for name, spec in blocks.items():
        hooks.append(spec["conv"].register_forward_hook(make_hook(name)))

    # TODO: populate calibration forward passes from notebook

    for h in hooks:
        h.remove()

    def normalize(x: torch.Tensor) -> torch.Tensor:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-10) if (hi - lo) > 1e-10 else torch.ones_like(x) * 0.5

    ilr: Dict[str, torch.Tensor] = {}
    for name, spec in blocks.items():
        n   = max(1, act_count[name])
        rms = normalize((act_accum[name] / n).sqrt())
        frb = normalize(frob_accum[name] / n)
        l1  = normalize(spec["conv"].weight.data.abs().sum(dim=[1, 2, 3]).cpu())
        ilr[name] = (
            ILR_WEIGHTS["rms"] * rms
            + ILR_WEIGHTS["bn_gamma"] * l1
            + ILR_WEIGHTS["hrank"] * frb
        ) / sum(ILR_WEIGHTS.values())
    return ilr


class HybridCrossoverMerger:
    """In-place Hybrid Crossover pruning engine for ResidualUNetDS.

    Merges dropped channel features into kept channel via regression
    rather than discarding, preserving representational capacity.

    Phase 4b usage: pass model=temporal_net.backbone; TAM projection
    layers are resized automatically when bottleneck is pruned.

    Args:
        model:       ResidualUNetDS model to prune in-place.
        calib_pairs: Calibration data (img/mask path pairs).
        device:      Torch device.
    """

    def __init__(self, model: nn.Module, calib_pairs: list, device: str = "cuda") -> None:
        self.model       = model
        self.calib_pairs = calib_pairs
        self.device      = device

    def merge(
        self,
        block_name: str,
        keep_idx: int,
        drop_idx: int,
        temporal_model: Optional[nn.Module] = None,
    ) -> str:
        """Prune one output channel from block_name.

        Args:
            block_name:     One of 'enc3', 'enc4', 'bottleneck', 'dec4', 'dec3'.
            keep_idx:       Index of channel to keep (receives merged features).
            drop_idx:       Index of channel to remove.
            temporal_model: If pruning Phase 4b backbone, pass TemporalFetaSegNet
                            so TAM projections are updated on bottleneck prune.

        Returns:
            'hybrid' if regression succeeded, 'standard_drop' otherwise.
        """
        # TODO: populate full merge implementation from notebook
        raise NotImplementedError(
            "Full implementation in notebooks/fetal_head_phase4a.ipynb "
            "(HybridCrossoverMerger.merge)"
        )

    def _collect_activations(
        self, block_name: str, keep_idx: int, drop_idx: int
    ) -> Tuple:
        """Collect per-channel feature maps from calibration set."""
        raise NotImplementedError

    def _hybrid_target(self, fm_keep: torch.Tensor, fm_drop: torch.Tensor) -> torch.Tensor:
        """Compute hybrid target (max-pool or adaptive blend) for regression."""
        raise NotImplementedError

    def _regress_filter(
        self, block_name: str, keep_idx: int, target: torch.Tensor, inputs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """50-step Adam regression to match hybrid target. Returns new filter or None."""
        raise NotImplementedError

    def _resize_tam_projections(
        self, keep_indices: List[int], attn_module: nn.Module
    ) -> None:
        """Resize TAM proj_in and proj_out when bottleneck is pruned (Phase 4b).

        proj_in:  Linear(old_bn_ch → 256) → slice weight[:, keep_indices]
        proj_out: Linear(256 → old_bn_ch) → slice weight[keep_indices, :]
        """
        old_pi = attn_module.proj_in
        new_bn_ch = len(keep_indices)
        new_pi = nn.Linear(new_bn_ch, old_pi.out_features).to(self.device)
        new_pi.weight = nn.Parameter(old_pi.weight.data[:, keep_indices])
        new_pi.bias   = nn.Parameter(old_pi.bias.data.clone())
        attn_module.proj_in = new_pi

        old_po = attn_module.proj_out[0]
        new_po = nn.Linear(old_po.in_features, new_bn_ch).to(self.device)
        new_po.weight = nn.Parameter(old_po.weight.data[keep_indices])
        new_po.bias   = nn.Parameter(old_po.bias.data[keep_indices])
        attn_module.proj_out[0] = new_po
        attn_module.bottleneck_ch = new_bn_ch
