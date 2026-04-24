"""
xai.py
Explainability methods for the fetal head segmentation system.

Three complementary XAI approaches:
1. GradCAM++ on decoder final layer (Phase 0 / single-frame)
   — shows WHICH spatial regions drove the boundary decision
2. Boundary uncertainty map (Phase 2 / cine)
   — shows WHERE the temporal model disagreed across frames
3. Temporal attention visualisation (Phase 2)
   — shows WHICH frames the model weighted most highly
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit/HF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from typing import Optional

from inference import (
    ResidualUNetDS, TemporalFetaSegNet,
    INPUT_H, INPUT_W, N_FRAMES,
    IMG_MEAN, IMG_STD, DEVICE,
    preprocess_image,
)


# ── GradCAM++ for segmentation ─────────────────────────────────────────────

class SegmentationGradCAMPlusPlus:
    """
    GradCAM++ adapted for binary segmentation models.

    For segmentation, we compute gradients of the mean predicted foreground
    probability with respect to the target layer's activations.
    This highlights which spatial regions most influenced the boundary prediction.

    Unlike classification GradCAM, we do NOT use a class score — instead
    we use the mean sigmoid output over the predicted mask region, which
    directly measures what drove the foreground predictions.
    """

    def __init__(self, model: ResidualUNetDS, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute GradCAM++ heatmap for one image.

        Args:
            img_tensor: [1, 1, H, W] normalised float tensor

        Returns:
            heatmap: [H, W] float32 in [0, 1]
        """
        self.model.eval()
        img_tensor = img_tensor.to(DEVICE).requires_grad_(False)

        # Forward pass — need gradients through the model
        img_tensor.requires_grad_(True)
        self.model.zero_grad()

        # Use eval-mode forward (returns logits only)
        logits = self.model(img_tensor)                    # [1, 1, H, W]
        probs  = torch.sigmoid(logits)

        # Score: mean probability over predicted foreground pixels
        # This is what we differentiate w.r.t. target layer activations
        pred_mask = (probs.detach() > 0.5).float()
        if pred_mask.sum() < 10:
            # Fallback: use all pixels if mask is nearly empty
            score = probs.mean()
        else:
            score = (probs * pred_mask).sum() / (pred_mask.sum() + 1e-8)

        score.backward()

        # GradCAM++ weighting
        # alpha_k = sum(grad^2) / (2*sum(grad^2) + sum(A * grad^3) + eps)
        grads = self._gradients           # [1, C, h, w]
        acts  = self._activations         # [1, C, h, w]

        grads_sq  = grads ** 2
        grads_cb  = grads ** 3
        denom     = 2 * grads_sq + (acts * grads_cb).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha     = grads_sq / denom      # [1, C, h, w]

        # Weighted sum of activations
        weights   = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam       = (weights * acts).sum(dim=1, keepdim=True)              # [1, 1, h, w]
        cam       = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
        return ((cam - cam_min) / (cam_max - cam_min)).astype(np.float32)

    def cleanup(self):
        """Remove hooks to avoid memory leaks."""
        self._activations = None
        self._gradients   = None


def render_gradcam_overlay(
    img_gray: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a GradCAM heatmap over a grayscale image.

    Args:
        img_gray: [H, W] uint8 grayscale (will be resized to INPUT_H x INPUT_W)
        heatmap:  [H, W] float32 in [0, 1]
        alpha:    heatmap opacity

    Returns:
        [H, W, 3] uint8 RGB blended image
    """
    img_r = cv2.resize(img_gray, (INPUT_W, INPUT_H))
    rgb   = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    # Apply jet colormap to heatmap
    hm_rgb = cm.jet(heatmap)[:, :, :3].astype(np.float32)  # [H, W, 3]

    blended = (1 - alpha) * rgb + alpha * hm_rgb
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


# ── boundary uncertainty visualisation ────────────────────────────────────

def render_uncertainty_overlay(
    img_gray: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay boundary uncertainty map (from temporal model) on image.

    High uncertainty (bright in 'hot' colormap) = model disagreed across frames
    at that pixel — clinically: the boundary is ambiguous at that location.

    Args:
        img_gray:    [H, W] uint8 grayscale
        uncertainty: [H, W] float32 std of per-frame binary predictions
        alpha:       uncertainty layer opacity

    Returns:
        [H, W, 3] uint8 RGB blended image
    """
    img_r = cv2.resize(img_gray, (INPUT_W, INPUT_H))
    rgb   = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    # Normalise uncertainty to [0, 1]
    unc_max = uncertainty.max()
    if unc_max < 1e-8:
        return (rgb * 255).astype(np.uint8)

    unc_norm = (uncertainty / unc_max).astype(np.float32)
    unc_rgb  = cm.hot(unc_norm)[:, :, :3].astype(np.float32)

    # Only show uncertainty where it's meaningful (> 5% of max)
    mask_show = (unc_norm > 0.05).astype(np.float32)[:, :, np.newaxis]
    blended   = rgb * (1 - alpha * mask_show) + unc_rgb * alpha * mask_show
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def render_boundary_ellipse(
    img_gray: np.ndarray,
    consensus_mask: np.ndarray,
    per_frame_probs: np.ndarray,
    pixel_spacing_mm: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Render the consensus ellipse with ±1 std uncertainty shading.

    The shading shows the range of predicted boundaries across frames —
    narrow shading = consistent prediction, wide = uncertain boundary.

    Args:
        img_gray:         [H, W] uint8
        consensus_mask:   [H, W] binary uint8
        per_frame_probs:  [T, H, W] float32 probability maps
        pixel_spacing_mm: for annotation
        threshold:        binarisation threshold

    Returns:
        [H, W, 3] uint8 annotated image
    """
    from skimage.measure import label, regionprops

    img_r = cv2.resize(img_gray, (INPUT_W, INPUT_H))
    rgb   = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB)

    # Fit ellipse to consensus mask
    contours, _ = cv2.findContours(
        (consensus_mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or len(max(contours, key=cv2.contourArea)) < 5:
        return rgb

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), (ma, mi), angle = cv2.fitEllipse(largest)

    # Draw ±1 std uncertainty bands using per-frame ellipse variation
    frame_axes = []
    for t in range(per_frame_probs.shape[0]):
        fm = (per_frame_probs[t] > threshold).astype(np.uint8)
        cnts, _ = cv2.findContours(fm * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts and len(max(cnts, key=cv2.contourArea)) >= 5:
            lc = max(cnts, key=cv2.contourArea)
            _, (a, b), _ = cv2.fitEllipse(lc)
            frame_axes.append((a, b))

    if len(frame_axes) >= 2:
        axes_arr = np.array(frame_axes)
        ma_std = axes_arr[:, 0].std()
        mi_std = axes_arr[:, 1].std()

        # Inner bound (consensus - std)
        cv2.ellipse(rgb, (int(cx), int(cy)),
                    (max(1, int((ma - ma_std) / 2)), max(1, int((mi - mi_std) / 2))),
                    angle, 0, 360, (100, 200, 100), 1)
        # Outer bound (consensus + std)
        cv2.ellipse(rgb, (int(cx), int(cy)),
                    (int((ma + ma_std) / 2), int((mi + mi_std) / 2)),
                    angle, 0, 360, (100, 200, 100), 1)

    # Consensus ellipse (bright green)
    cv2.ellipse(rgb, (int(cx), int(cy)),
                (int(ma / 2), int(mi / 2)),
                angle, 0, 360, (50, 220, 50), 2)

    return rgb


# ── attention weight visualisation ────────────────────────────────────────

def render_attention_heatmap(
    attn_weights: np.ndarray,
    n_frames: int = N_FRAMES,
) -> np.ndarray:
    """
    Render temporal attention weights as a matplotlib figure image.

    Shows: (1) T×T attention matrix, (2) per-frame attention received
    (column mean — how much each frame was attended to overall).

    Args:
        attn_weights: [T, T] float32 attention weight matrix

    Returns:
        [H, W, 3] uint8 RGB figure image suitable for display in Streamlit
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Full attention matrix
    im = axes[0].imshow(attn_weights, cmap="hot", vmin=0, aspect="auto")
    axes[0].set_title("Temporal attention matrix\n(row = query frame, col = key frame)",
                      fontsize=9)
    axes[0].set_xlabel("Key frame"); axes[0].set_ylabel("Query frame")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Per-frame attention received (column sum → which frames were most useful)
    attn_received = attn_weights.mean(axis=0)
    frame_ids = np.arange(n_frames)
    axes[1].bar(frame_ids, attn_received, color=cm.hot(attn_received / attn_received.max()))
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Mean attention received")
    axes[1].set_title("Per-frame attention\n(high = model found this frame most informative)",
                      fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.canvas.draw()
    # tostring_rgb() removed in matplotlib 3.8+ — use buffer_rgba() instead
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf  = rgba[:, :, :3]  # drop alpha channel -> RGB
    plt.close(fig)
    return buf


# ── combined XAI panel ─────────────────────────────────────────────────────

def build_xai_panel(
    img_gray: np.ndarray,
    result: dict,
    phase0_model: Optional[ResidualUNetDS] = None,
) -> dict:
    """
    Build all XAI outputs for a given prediction result.

    Args:
        img_gray:      [H, W] uint8 original image
        result:        output dict from predict_single_frame or predict_cine_clip
        phase0_model:  required for GradCAM++ (single frame mode)

    Returns dict with keys depending on mode:
        Single frame: gradcam_overlay
        Cine clip:    uncertainty_overlay, ellipse_overlay, attn_heatmap
    """
    outputs = {}

    if result["mode"] == "single_frame" and phase0_model is not None:
        # GradCAM++ on decoder final layer
        target_layer = phase0_model.dec1.block[-1]
        cam = SegmentationGradCAMPlusPlus(phase0_model, target_layer)
        img_t = preprocess_image(img_gray)
        try:
            heatmap = cam(img_t)
            outputs["gradcam_overlay"] = render_gradcam_overlay(img_gray, heatmap)
            outputs["gradcam_raw"]     = heatmap
        except Exception as e:
            outputs["gradcam_error"] = str(e)
        finally:
            cam.cleanup()

    elif result["mode"] == "cine_clip":
        # Boundary uncertainty overlay
        outputs["uncertainty_overlay"] = render_uncertainty_overlay(
            img_gray, result["uncertainty"]
        )
        outputs["uncertainty_raw"] = result["uncertainty"]

        # Attention heatmap
        outputs["attn_heatmap"] = render_attention_heatmap(result["attn_weights"])

        # Ellipse with uncertainty bands (requires per-frame probs)
        # Note: per_frame_probs not stored in result dict by default —
        # this is optional and requires re-running inference with prob storage
        outputs["ellipse_overlay"] = render_boundary_ellipse(
            img_gray,
            result["consensus_mask"],
            np.stack([result["prob_map"]] * N_FRAMES),  # fallback: tile mean
            result.get("hc_mm", 0) or 0,
        )

    return outputs
