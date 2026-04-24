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

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional

from src.data.dataset import INPUT_H, INPUT_W

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

    with torch.no_grad():
        for i, (img_path, _) in enumerate(calib_pairs[:n_batches]):
            img = cv2.resize(cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE), (INPUT_W, INPUT_H))
            t = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
            model(t.to(device))

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

    # Maps: block -> next sequential block (encoder path or decoder path)
    _NEXT_BLOCK = {
        "enc3": "enc4",
        "enc4": "bottleneck",
    }
    # Maps: encoder block -> corresponding decoder block (skip connection)
    _ENC_DEC = {"enc3": "dec3", "enc4": "dec4"}

    def __init__(self, model: nn.Module, calib_pairs: list, device: str = "cuda") -> None:
        self.model       = model
        self.calib_pairs = calib_pairs
        self.device      = device
        self._acts: dict = {}

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
        block    = getattr(self.model, block_name)
        old_conv = block.block[-1]
        assert isinstance(old_conv, nn.Conv2d)
        n_ch = old_conv.out_channels
        keep_indices = [keep_idx] + [i for i in range(n_ch) if i != drop_idx and i != keep_idx]

        # Steps 1-3: activations, hybrid target, regress
        FM_keep, FM_drop, X_inputs = self._collect_activations(block_name, keep_idx, drop_idx)
        target = self._hybrid_target(FM_keep, FM_drop)
        F_new = self._regress_filter(block_name, keep_idx, target, X_inputs)
        mode  = 'hybrid' if F_new is not None else 'standard_drop'

        # Step 4: rebuild block[-1] (output channels reduced)
        W = old_conv.weight.data.clone()
        if F_new is not None: W[keep_idx] = F_new.to(W.device)
        new_W = W[keep_indices]
        new_last = nn.Conv2d(old_conv.in_channels, len(keep_indices),
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None).to(self.device)
        new_last.weight = nn.Parameter(new_W)
        if old_conv.bias is not None:
            new_last.bias = nn.Parameter(old_conv.bias.data[keep_indices])
        block.block[-1] = new_last

        # Step 5: rebuild skip
        if isinstance(block.skip, nn.Conv2d):
            old_skip = block.skip
            new_skip = nn.Conv2d(old_skip.in_channels, len(keep_indices), 1, bias=False).to(self.device)
            new_skip.weight = nn.Parameter(old_skip.weight.data[keep_indices])
            block.skip = new_skip

        # Step 6: patch next encoder block (enc3->enc4, enc4->bottleneck)
        self._patch_next_encoder_block(block_name, keep_indices)

        # Step 7: patch decoder skip-connection (enc3->dec3, enc4->dec4)
        if block_name in self._ENC_DEC:
            self._patch_block_input(self._ENC_DEC[block_name], keep_indices)

        # Step 8: handle bottleneck and decoder pruning cascades
        if block_name == 'bottleneck':
            # up4.in_channels must shrink (up4 takes bottleneck output)
            self.model.up4 = self._rebuild_convtranspose_in(self.model.up4, keep_indices, self.device)
            # dec4 input = cat(up4_output, enc4_output)
            # up4 output channels are UNCHANGED (they match enc4), so dec4 cat only changes
            # if up4 output changes. But up4 output = old up4.out_channels (unchanged).
            # So dec4 is NOT affected by bottleneck pruning. Only up4 input side changes.
            if temporal_model is not None:
                self._resize_tam_projections(keep_indices, temporal_model.attn)

        if block_name == 'dec4':
            # dec4 output feeds into up3 input
            self.model.up3 = self._rebuild_convtranspose_in(self.model.up3, keep_indices, self.device)
            # dec3 input = cat(up3_output, enc3_output)
            # up3 output channels are UNCHANGED, so dec3 cat is not affected.

        if block_name == 'dec3':
            # dec3 output feeds into up2 input
            self.model.up2 = self._rebuild_convtranspose_in(self.model.up2, keep_indices, self.device)
            # Also update aux_d3 head (takes dec3 output)
            old_aux = self.model.aux_d3
            self.model.aux_d3 = nn.Conv2d(len(keep_indices), old_aux.out_channels, 1).to(self.device)
            self.model.aux_d3.weight = nn.Parameter(old_aux.weight.data[:, keep_indices, :, :])
            if old_aux.bias is not None:
                self.model.aux_d3.bias = nn.Parameter(old_aux.bias.data.clone())

        print(f'  [{mode}] {block_name}: ch{drop_idx} dropped | {n_ch} -> {len(keep_indices)}')
        return mode

    def _collect_activations(
        self, block_name: str, keep_idx: int, drop_idx: int
    ) -> Tuple:
        """Collect per-channel feature maps from calibration set."""
        conv = self._get_last_conv(block_name)
        maps_keep, maps_drop, inputs = [], [], []
        def hook(m, inp, out):
            self._acts['out'] = out.detach()
            self._acts['inp'] = inp[0].detach()
        h = conv.register_forward_hook(hook)
        self.model.eval()
        with torch.no_grad():
            for img_path, _ in self.calib_pairs:
                img = cv2.resize(cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE), (INPUT_W, INPUT_H))
                t = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
                self.model(t.to(self.device))
                act = self._acts.get('out')
                inp = self._acts.get('inp')
                if act is not None:
                    maps_keep.append(act[0, keep_idx].cpu())
                    maps_drop.append(act[0, drop_idx].cpu())
                    inputs.append(inp[0].cpu())
        h.remove()
        return torch.stack(maps_keep), torch.stack(maps_drop), torch.stack(inputs)

    def _hybrid_target(self, fm_keep: torch.Tensor, fm_drop: torch.Tensor) -> torch.Tensor:
        """Compute hybrid target (max-pool or adaptive blend) for regression."""
        if True:  # mode == 'max' (hardcoded default matching notebook)
            return torch.maximum(fm_keep, fm_drop)
        sk = fm_keep.abs().mean(dim=[1, 2], keepdim=True)
        sd = fm_drop.abs().mean(dim=[1, 2], keepdim=True)
        a  = sk / (sk + sd + 1e-8)
        return a * fm_keep + (1 - a) * fm_drop

    def _regress_filter(
        self, block_name: str, keep_idx: int, target: torch.Tensor, inputs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """50-step Adam regression to match hybrid target. Returns new filter or None."""
        conv  = self._get_last_conv(block_name)
        F_new = nn.Parameter(conv.weight[keep_idx].clone().to(self.device))
        opt   = optim.Adam([F_new], lr=1e-3)
        loss_fn = nn.MSELoss()
        prev_loss = float('inf')
        for _ in range(50):
            opt.zero_grad()
            total = torch.tensor(0.0, device=self.device, requires_grad=True)
            for i, x in enumerate(inputs):
                pred  = F.conv2d(x.unsqueeze(0).to(self.device), F_new.unsqueeze(0),
                                 stride=conv.stride, padding=conv.padding)
                tgt   = target[i].unsqueeze(0).unsqueeze(0).to(self.device)
                total = total + loss_fn(pred, tgt)
            avg = total / len(inputs)
            avg.backward(); opt.step()
            if torch.isnan(avg) or avg.item() > prev_loss * 10: return None
            prev_loss = avg.item()
        return F_new.detach().cpu()

    def _get_last_conv(self, block_name: str) -> nn.Conv2d:
        return getattr(self.model, block_name).block[-1]

    @staticmethod
    def _rebuild_bn(old_bn: nn.BatchNorm2d, channel_indices: List[int], device: str) -> nn.BatchNorm2d:
        new_bn = nn.BatchNorm2d(len(channel_indices)).to(device)
        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
            d = getattr(old_bn, attr).data[channel_indices]
            if attr in ['weight', 'bias']: setattr(new_bn, attr, nn.Parameter(d))
            else: setattr(new_bn, attr, d)
        new_bn.num_batches_tracked = old_bn.num_batches_tracked.clone()
        return new_bn

    @staticmethod
    def _rebuild_conv_in(old_conv: nn.Conv2d, in_indices: List[int], device: str) -> nn.Conv2d:
        new_conv = nn.Conv2d(len(in_indices), old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None).to(device)
        new_conv.weight = nn.Parameter(old_conv.weight.data[:, in_indices, :, :])
        if old_conv.bias is not None:
            new_conv.bias = nn.Parameter(old_conv.bias.data.clone())
        return new_conv

    @staticmethod
    def _rebuild_conv_out(old_conv: nn.Conv2d, out_indices: List[int], device: str) -> nn.Conv2d:
        new_conv = nn.Conv2d(old_conv.in_channels, len(out_indices),
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None).to(device)
        new_conv.weight = nn.Parameter(old_conv.weight.data[out_indices])
        if old_conv.bias is not None:
            new_conv.bias = nn.Parameter(old_conv.bias.data[out_indices])
        return new_conv

    @staticmethod
    def _rebuild_convtranspose_in(
        old_ct: nn.ConvTranspose2d, in_indices: List[int], device: str
    ) -> nn.ConvTranspose2d:
        """Rebuild ConvTranspose2d when input channels shrink."""
        new_ct = nn.ConvTranspose2d(len(in_indices), old_ct.out_channels,
            kernel_size=old_ct.kernel_size, stride=old_ct.stride,
            padding=old_ct.padding, bias=old_ct.bias is not None).to(device)
        new_ct.weight = nn.Parameter(old_ct.weight.data[in_indices])
        if old_ct.bias is not None:
            new_ct.bias = nn.Parameter(old_ct.bias.data.clone())
        return new_ct

    @staticmethod
    def _rebuild_convtranspose_out(
        old_ct: nn.ConvTranspose2d, out_indices: List[int], device: str
    ) -> nn.ConvTranspose2d:
        """Rebuild ConvTranspose2d when output channels shrink."""
        new_ct = nn.ConvTranspose2d(old_ct.in_channels, len(out_indices),
            kernel_size=old_ct.kernel_size, stride=old_ct.stride,
            padding=old_ct.padding, bias=old_ct.bias is not None).to(device)
        new_ct.weight = nn.Parameter(old_ct.weight.data[:, out_indices])
        if old_ct.bias is not None:
            new_ct.bias = nn.Parameter(old_ct.bias.data[out_indices])
        return new_ct

    def _patch_block_input(self, target_block_name: str, keep_indices: List[int]) -> None:
        """Rebuild target_block's BN[0], Conv[2], skip to accept fewer input channels."""
        blk = getattr(self.model, target_block_name)
        old_bn0 = blk.block[0]
        old_cat_ch = old_bn0.num_features
        enc_old_ch = len(keep_indices) + 1
        up_ch = old_cat_ch - enc_old_ch
        cat_keep = list(range(up_ch)) + [up_ch + k for k in keep_indices]

        blk.block[0] = self._rebuild_bn(old_bn0, cat_keep, self.device)
        blk.block[2] = self._rebuild_conv_in(blk.block[2], cat_keep, self.device)
        if isinstance(blk.skip, nn.Conv2d):
            blk.skip = self._rebuild_conv_in(blk.skip, cat_keep, self.device)

    def _patch_next_encoder_block(self, block_name: str, keep_indices: List[int]) -> None:
        """Rebuild next encoder block's input layers."""
        if block_name not in self._NEXT_BLOCK: return
        nxt = getattr(self.model, self._NEXT_BLOCK[block_name])
        nxt.block[0] = self._rebuild_bn(nxt.block[0], keep_indices, self.device)
        nxt.block[2] = self._rebuild_conv_in(nxt.block[2], keep_indices, self.device)
        if isinstance(nxt.skip, nn.Conv2d):
            nxt.skip = self._rebuild_conv_in(nxt.skip, keep_indices, self.device)

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
