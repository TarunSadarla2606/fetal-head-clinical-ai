"""XAI computation backing the /findings/{id}/* API endpoints.

Three explanations are exposed:

1. **GradCAM++** — class-activation map on the U-Net decoder's last
   convolution. Highlights the spatial regions that drove the segmentation
   decision. Works for both static (ResidualUNetDS / PrunedResidualUNetDS)
   and temporal (TemporalFetaSegNet) models — for the temporal case we
   use the spatial backbone.
2. **Uncertainty heatmap** — pixel-wise variance of the predicted
   foreground probability across multiple stochastic forward passes.
   For static models we use input perturbation (small Gaussian noise);
   for temporal models we reuse the per-frame disagreement that the cine
   pipeline already computes.
3. **OOD analysis** — combines the existing :func:`validate_input` checks
   with image statistics (intensity, contrast, edge density, Laplacian
   variance) to surface a structured set of out-of-distribution reasons.

All three return single-channel float arrays in [0, 1] for heatmaps, and
either uint8 RGB blends for overlays or JSON for OOD.
"""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

from app.inference import (
    INPUT_H,
    INPUT_W,
    PrunedResidualUNetDS,
    ResidualUNetDS,
    TemporalFetaSegNet,
    preprocess_image,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _spatial_backbone(model) -> torch.nn.Module:
    """Return the static spatial sub-model used for GradCAM/uncertainty.

    Temporal models wrap a :class:`ResidualUNetDS` backbone; we run XAI
    against that backbone directly, which avoids the 5-D clip plumbing
    and gives a per-frame explanation that mirrors the consensus mask.
    """
    if isinstance(model, TemporalFetaSegNet):
        return model.backbone
    return model


def _gradcam_target_layer(model) -> torch.nn.Module:
    """Pick the last decoder convolution as the GradCAM target layer.

    ``dec1.block[-1]`` is the conv that produces the final pre-logit
    feature map; gradients on its activations correlate strongly with
    pixels that influenced the final segmentation.
    """
    backbone = _spatial_backbone(model)
    if isinstance(backbone, (ResidualUNetDS, PrunedResidualUNetDS)):
        return backbone.dec1.block[-1]
    raise TypeError(f"Unsupported model type for GradCAM: {type(backbone).__name__}")


def _colormap_overlay(
    img_gray: np.ndarray,
    heatmap_norm: np.ndarray,
    cmap_name: str = "jet",
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a normalised [0, 1] heatmap over a grayscale image.

    Output shape matches the input grayscale image so the caller doesn't
    have to track resize state.
    """
    h, w = img_gray.shape
    heatmap_resized = cv2.resize(heatmap_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    cmap = cm.get_cmap(cmap_name)
    hm_rgb = cmap(np.clip(heatmap_resized, 0.0, 1.0))[:, :, :3].astype(np.float32)

    blended = (1 - alpha) * rgb + alpha * hm_rgb
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


# ── 1. GradCAM++ ──────────────────────────────────────────────────────────────


def compute_gradcam(model, img_gray: np.ndarray) -> np.ndarray:
    """Compute a GradCAM++ overlay (uint8 RGB) for *img_gray*.

    Returns an array shaped like the input grayscale (H, W, 3).
    """
    backbone = _spatial_backbone(model)
    target = _gradcam_target_layer(model)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    fh = target.register_forward_hook(lambda _m, _i, o: activations.append(o))
    bh = target.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))

    try:
        backbone.eval()
        device = next(backbone.parameters()).device
        img_t = preprocess_image(img_gray).to(device)
        img_t.requires_grad_(True)
        backbone.zero_grad(set_to_none=True)

        logits = backbone(img_t)  # [1, 1, H, W]
        probs = torch.sigmoid(logits)
        pred_mask = (probs.detach() > 0.5).float()
        if pred_mask.sum() < 10:
            score = probs.mean()
        else:
            score = (probs * pred_mask).sum() / (pred_mask.sum() + 1e-8)
        score.backward()

        acts = activations[-1].detach()  # [1, C, h, w]
        grads = gradients[-1].detach()
        grads_sq = grads**2
        grads_cb = grads**3
        denom = 2 * grads_sq + (acts * grads_cb).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha = grads_sq / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min < 1e-8:
            cam_norm = np.zeros_like(cam_np)
        else:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
    finally:
        fh.remove()
        bh.remove()

    return _colormap_overlay(img_gray, cam_norm, cmap_name="jet", alpha=0.45)


# ── 2. Uncertainty (MC via input perturbation) ────────────────────────────────


def compute_uncertainty(
    model,
    img_gray: np.ndarray,
    n_samples: int = 8,
    noise_sigma: float = 0.04,
) -> np.ndarray:
    """Compute a pixel-wise uncertainty heatmap (uint8 RGB overlay).

    Strategy:
      * Run *n_samples* forward passes with small Gaussian noise added
        to the input. The variance of the resulting probability maps
        approximates predictive uncertainty without requiring dropout
        layers (which are absent in the pruned variants).
      * For temporal models we still use the spatial backbone — a single
        frame is the unit of explanation we care about for the demo.
    """
    backbone = _spatial_backbone(model)
    backbone.eval()
    device = next(backbone.parameters()).device

    base = preprocess_image(img_gray).to(device)
    rng = np.random.default_rng(0)

    probs_stack = []
    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.from_numpy(
                rng.normal(0.0, noise_sigma, base.shape).astype(np.float32)
            ).to(device)
            logits = backbone(base + noise)
            probs_stack.append(torch.sigmoid(logits).cpu().numpy()[0, 0])

    arr = np.stack(probs_stack, axis=0)
    var = arr.var(axis=0)  # [H, W]
    v_max = float(var.max())
    var_norm = var / v_max if v_max > 1e-8 else var
    return _colormap_overlay(img_gray, var_norm.astype(np.float32), cmap_name="hot", alpha=0.55)


def uncertainty_variance(model, img_gray: np.ndarray, n_samples: int = 8) -> float:
    """Return the maximum pixel variance across MC samples — used by tests
    to assert the heatmap has non-trivial signal.
    """
    backbone = _spatial_backbone(model)
    backbone.eval()
    device = next(backbone.parameters()).device

    base = preprocess_image(img_gray).to(device)
    rng = np.random.default_rng(0)

    probs_stack = []
    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.from_numpy(rng.normal(0.0, 0.04, base.shape).astype(np.float32)).to(
                device
            )
            logits = backbone(base + noise)
            probs_stack.append(torch.sigmoid(logits).cpu().numpy()[0, 0])

    arr = np.stack(probs_stack, axis=0)
    return float(arr.var(axis=0).max())


# ── 3. OOD analysis ───────────────────────────────────────────────────────────


def analyze_ood(img_gray: np.ndarray, prior_validation: dict) -> dict:
    """Combine input validation with image statistics into a structured OOD report.

    Args:
      img_gray: original grayscale image (uint8, 2-D)
      prior_validation: result of :func:`app.inference.validate_input`,
        passed in so we don't double-run the checks.

    Returns:
      dict with keys ``ood_flag``, ``reasons`` (list of structured findings),
      ``score`` (float in [0, 1] — higher = more OOD), and ``stats``
      (raw image statistics for the UI to render).
    """
    reasons: list[dict] = []

    # 1. Carry over validation checks
    if not prior_validation.get("valid", True):
        for warning in prior_validation.get("warnings", []):
            reasons.append({"category": "input_validation", "detail": warning})

    # 2. Image statistics — independent of validate_input so we always populate
    arr = img_gray.astype(np.float32)
    mean_int = float(arr.mean())
    std_int = float(arr.std())

    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = float(edges.sum() / (255.0 * edges.size))

    lap_var = float(cv2.Laplacian(arr, cv2.CV_32F).var())

    stats = {
        "mean_intensity": mean_int,
        "std_intensity": std_int,
        "edge_density": edge_density,
        "laplacian_variance": lap_var,
    }

    # 3. OOD signals beyond the validation pass
    if std_int < 15:
        reasons.append(
            {
                "category": "low_contrast",
                "detail": f"Standard deviation of intensity is very low (std={std_int:.1f}). "
                "Real ultrasound frames typically have std ≥ 30.",
            }
        )
    if lap_var < 30 and prior_validation.get("checks", {}).get("has_texture", True):
        reasons.append(
            {
                "category": "low_texture_borderline",
                "detail": f"Laplacian variance ({lap_var:.1f}) is below typical ultrasound range (>50).",
            }
        )
    if edge_density < 0.005:
        reasons.append(
            {
                "category": "low_edge_density",
                "detail": f"Edge density ({edge_density:.4f}) is very low — image may be smooth or synthetic.",
            }
        )
    if mean_int < 30 or mean_int > 220:
        reasons.append(
            {
                "category": "extreme_brightness",
                "detail": f"Mean intensity ({mean_int:.1f}) is outside the typical ultrasound range [40, 200].",
            }
        )

    # 4. Aggregate score: fraction of triggered checks (rough but explainable)
    total_checks = 4
    triggered_extras = sum(
        1
        for r in reasons
        if r["category"]
        in {"low_contrast", "low_texture_borderline", "low_edge_density", "extreme_brightness"}
    )
    extras_score = triggered_extras / total_checks
    base_score = 0.0 if prior_validation.get("valid", True) else 0.5
    score = min(1.0, base_score + 0.5 * extras_score)

    return {
        "ood_flag": bool(reasons),
        "reasons": reasons,
        "score": float(score),
        "stats": stats,
    }


# ── shape helpers used by tests ───────────────────────────────────────────────


def gradcam_overlay_shape(img_gray: np.ndarray) -> tuple[int, int, int]:
    """Predicted shape of the GradCAM overlay for a given input image."""
    h, w = img_gray.shape
    return (h, w, 3)
