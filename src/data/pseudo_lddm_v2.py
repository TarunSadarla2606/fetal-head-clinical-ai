"""Pseudo-LDDM v2: Physics-inspired synthetic cine-loop generation.

Converts static HC18 frames into 16-frame cine sequences for temporal
model training, without requiring proprietary cine datasets.

Key improvements over v1 (course project):

  v1 issues:
  - Sinusoidal probe motion: periodic, fully predictable
  - No cross-sectional variation: all 16 frames identical HC → trivial temporal task
  - Temporal HC std ≈ 0.0 px → attention module had nothing to learn

  v2 fixes:
  - Ornstein-Uhlenbeck motion: mean-reverting, stochastic, non-periodic
  - Per-frame ellipse axis perturbation: non-trivial HC variation across frames
  - Rician speckle: sqrt(real^2 + imag^2), physically correct US noise
  - Depth-dependent attenuation: exp(-k * depth_fraction)
  - Acoustic shadowing behind skull boundary
  - TGC (time-gain compensation) drift between frames

  Result: Mean temporal HC std = 10.33 px  (vs ~0.0 in v1)
  806 clips generated at Stage 4 full clinical fidelity
"""

import numpy as np
from typing import List


def ornstein_uhlenbeck(
    n: int,
    theta: float = 0.15,
    sigma: float = 2.0,
    mu: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate Ornstein-Uhlenbeck mean-reverting stochastic process.

    Used for probe translation (tx, ty) and rotation across frames.
    Mean-reverting property keeps probe drift anatomically plausible.

    Args:
        n: Number of time steps (frames).
        theta: Mean-reversion rate.
        sigma: Noise amplitude.
        mu: Long-run mean (default: 0.0).
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape (n,) with the OU process values.
    """
    # TODO: populate from notebook
    raise NotImplementedError


def generate_cine(
    img_gray: np.ndarray,
    mask: np.ndarray,
    n_frames: int = 16,
    seed: int = 42,
    stage: int = 4,
) -> List[np.ndarray]:
    """Generate a synthetic cine-loop from a single static ultrasound frame.

    Simulates realistic ultrasound probe motion and image degradation over
    16 frames using physics-inspired models.

    Pipeline per frame:
      1. Sample OU probe motion (tx, ty, rotation)
      2. Perturb ellipse axes for cross-sectional variation (key fix)
      3. Apply warp affine transform (cv2.warpAffine)
      4. Apply Rician speckle noise
      5. Apply depth-dependent intensity attenuation
      6. Apply acoustic shadowing behind skull (Stage 4+)
      7. Apply TGC drift (Stage 4)

    Args:
        img_gray: Single ultrasound frame [H, W] uint8.
        mask: Skull boundary mask [H, W] uint8 (for shadowing).
        n_frames: Number of output frames (default: 16).
        seed: Random seed for reproducibility.
        stage: Simulation fidelity stage (1-4). Stage 4 = full clinical fidelity.

    Returns:
        List of n_frames uint8 frames, each [H, W].
    """
    # TODO: populate from notebook
    raise NotImplementedError
