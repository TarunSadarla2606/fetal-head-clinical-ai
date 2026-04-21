"""Pseudo-LDDM v2: Physics-inspired synthetic cine-loop generation.

Converts static HC18 frames into 16-frame cine sequences for temporal model
training without requiring proprietary cine datasets.

Key fix over v1 (course project):
  v1 used sinusoidal motion → all frames nearly identical → temporal HC std ≈ 0
  → TAM had nothing to learn from inter-frame variation.

  v2 adds cross-sectional mask variation (per-frame ellipse axis perturbation)
  → temporal HC std = 10.33 px → TAM learns meaningful inter-frame attention.

Four fidelity stages:
  Stage 1: OU motion only (no noise/shadowing)
  Stage 2: + Rician speckle + depth attenuation
  Stage 3: + acoustic shadowing behind skull
  Stage 4: + TGC drift between frames  (806 clips generated at this stage)

Output format: .npz files with keys:
  'frames': uint8 [T, H, W]  (T=16)
  'masks':  uint8 [T, H, W]  solid binary masks per frame
  'hc_gt':  float  HC in mm for each frame
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

N_FRAMES = 16
INPUT_H  = 256
INPUT_W  = 384
SEED     = 42

# Ornstein-Uhlenbeck parameters (probe motion)
OU_TX     = {"theta": 0.15, "sigma": 2.0}   # translation x (pixels)
OU_TY     = {"theta": 0.15, "sigma": 1.5}   # translation y (pixels)
OU_ROT    = {"theta": 0.20, "sigma": 0.40}  # rotation (degrees)


def ornstein_uhlenbeck(
    n: int,
    theta: float,
    sigma: float,
    mu: float = 0.0,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate an Ornstein-Uhlenbeck mean-reverting stochastic process.

    dX = theta*(mu - X)*dt + sigma*sqrt(dt)*N(0,1)

    Mean-reversion property keeps probe drift within anatomically plausible
    range while remaining non-periodic and stochastic.

    Args:
        n:     Number of time steps.
        theta: Mean-reversion rate.
        sigma: Noise amplitude (standard deviation of per-step increment).
        mu:    Long-run mean (default: 0.0).
        dt:    Time step size (default: 1.0).
        rng:   NumPy random generator for reproducibility.

    Returns:
        Array of shape (n,) with OU process values.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    x = np.zeros(n)
    for i in range(1, n):
        noise = rng.normal(0, 1)
        x[i]  = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * noise
    return x


def get_ellipse_params(mask: np.ndarray) -> Optional[Dict]:
    """Fit ellipse to filled skull mask.

    Args:
        mask: [H, W] uint8 solid binary mask.

    Returns:
        Dict with keys cx, cy, a, b, angle (semi-axes in pixels) or None.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    (cx, cy), (ma, mi), angle = cv2.fitEllipse(largest)
    return {"cx": cx, "cy": cy, "a": ma / 2, "b": mi / 2, "angle": angle}


def generate_cine(
    img_gray:  np.ndarray,
    mask:      np.ndarray,
    n_frames:  int = N_FRAMES,
    seed:      int = SEED,
    stage:     int = 4,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic 16-frame cine-loop from a static ultrasound frame.

    Pipeline per frame:
      1. Sample OU probe motion (tx, ty, rotation)
      2. Perturb ellipse semi-axes for cross-sectional variation  ← critical fix
      3. Warp affine transform (cv2.warpAffine)
      4. Rician speckle: sqrt(real² + imag²)                       (Stage 2+)
      5. Depth-dependent intensity attenuation: exp(−k×depth)      (Stage 2+)
      6. Acoustic shadowing behind skull boundary                  (Stage 3+)
      7. TGC (time-gain compensation) drift between frames         (Stage 4)

    Args:
        img_gray: Static ultrasound frame [H, W] uint8.
        mask:     Solid skull mask [H, W] uint8 (for axis perturbation + shadowing).
        n_frames: Number of output frames (default: 16).
        seed:     Random seed.
        stage:    Fidelity stage 1–4 (Stage 4 = full clinical fidelity).

    Returns:
        Dict with:
          'frames': uint8 [T, H, W]
          'masks':  uint8 [T, H, W]  per-frame solid masks (after axis perturbation)
          'hc_gt':  float array [T]  per-frame HC in pixels
    """
    # TODO: populate from notebooks/fetal_head_phase1_lddm_v2.ipynb
    raise NotImplementedError


def generate_dataset(
    image_paths: List[Path],
    mask_paths:  List[Path],
    output_dir:  Path,
    n_frames:    int = N_FRAMES,
    stage:       int = 4,
    seed:        int = SEED,
) -> None:
    """Generate cine clips for a list of HC18 image/mask pairs and save as .npz.

    Used to produce the 806 training clips for Phase 2.
    Output structure: output_dir/{stem}_cine.npz  per input image.

    Args:
        image_paths: List of HC18 ultrasound image paths.
        mask_paths:  Corresponding solid mask paths (fill_hollow_mask applied).
        output_dir:  Directory to save .npz files.
        n_frames:    Frames per clip.
        stage:       Simulation fidelity stage.
        seed:        Base seed (each clip gets seed + idx for reproducibility).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # TODO: populate from notebook
        raise NotImplementedError
