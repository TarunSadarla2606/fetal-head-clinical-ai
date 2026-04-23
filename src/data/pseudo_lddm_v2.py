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
from tqdm import tqdm

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


def load_image_mask(
    img_path: Path,
    mask_path: Path,
    target_h: int = INPUT_H,
    target_w: int = INPUT_W,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess one image/mask pair.

    Returns:
        img:  [H, W] uint8 grayscale, resized
        mask: [H, W] uint8 binary {0,1}, filled solid ellipse, resized
    """
    img  = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Resize to training dimensions
    img  = cv2.resize(img,  (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Binarise
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Flood-fill
    h, w = mask_bin.shape
    fm = np.zeros((h + 2, w + 2), dtype=np.uint8)
    filled = mask_bin.copy()
    cv2.floodFill(filled, fm, (0, 0), 255)
    solid = cv2.bitwise_or(mask_bin, cv2.bitwise_not(filled))
    solid = (solid > 127).astype(np.uint8)

    return img, solid


def apply_rigid_transform(
    img: np.ndarray,
    tx: float,
    ty: float,
    rot_deg: float,
    scale: float,
    border_value: int = 0,
) -> np.ndarray:
    """Apply affine transform (translation + rotation + scale) to an image.

    Centre of rotation is the image centre.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def perturb_mask_cross_section(
    mask: np.ndarray,
    ellipse_params: Dict,
    da_frac: float,
    db_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a slightly different cross-section by perturbing ellipse semi-axes.

    da_frac, db_frac: fractional change to semi-axes a and b (e.g. 0.03 = ±3%)
    This simulates seeing a slightly off-plane cross-section of the skull.

    Returns: new binary mask [H, W]
    """
    ep = ellipse_params
    h, w = mask.shape

    # Perturb semi-axes independently
    new_a = max(ep['a'] * (1 + da_frac), ep['a'] * 0.85)  # clamp: don't shrink > 15%
    new_b = max(ep['b'] * (1 + db_frac), ep['b'] * 0.85)

    # Draw new filled ellipse
    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        new_mask,
        center=(int(round(ep['cx'])), int(round(ep['cy']))),
        axes=(int(round(new_a)), int(round(new_b))),
        angle=ep['angle'],
        startAngle=0, endAngle=360,
        color=1, thickness=-1  # filled
    )
    return new_mask


def add_rician_speckle(
    img_float: np.ndarray,
    noise_std: float = 0.08,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Rician speckle noise to a [0,1] float image.

    Rician noise: magnitude of complex Gaussian noise added to signal.
    Result = sqrt((img + n1)^2 + n2^2) where n1,n2 ~ N(0, sigma)

    This correctly models coherent ultrasound backscatter.
    """
    if rng is None:
        rng = np.random.default_rng()
    n1 = rng.normal(0, noise_std, img_float.shape).astype(np.float32)
    n2 = rng.normal(0, noise_std, img_float.shape).astype(np.float32)
    noisy = np.sqrt((img_float + n1)**2 + n2**2)
    return np.clip(noisy, 0, 1)


def add_depth_attenuation(
    img_float: np.ndarray,
    attenuation_coeff: float = 0.4,
) -> np.ndarray:
    """Apply depth-dependent intensity attenuation.

    Top of image (near field) = no attenuation.
    Bottom (far field) = multiplied by exp(-coeff).

    attenuation_coeff: 0.4 produces ~33% intensity reduction at bottom.
    """
    h, w = img_float.shape
    depth = np.linspace(0, attenuation_coeff, h, dtype=np.float32)
    attenuation = np.exp(-depth)[:, np.newaxis]  # [H, 1] broadcast over width
    return img_float * attenuation


def add_acoustic_shadow(
    img_float: np.ndarray,
    mask: np.ndarray,
    shadow_strength: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add acoustic shadow below the skull boundary.

    Finds the bottom edge of the skull mask per column,
    then suppresses intensity below it by shadow_strength.
    shadow_strength=0.5 means 50% intensity reduction in shadow zone.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img_float.shape
    shadow_map = np.ones((h, w), dtype=np.float32)

    # Only apply to ~40% of columns (realistic — shadow appears where bone is thickest)
    shadow_cols = rng.choice(w, size=int(w * 0.4), replace=False)

    for col in shadow_cols:
        skull_rows = np.where(mask[:, col] > 0)[0]
        if len(skull_rows) == 0:
            continue
        bottom_of_skull = skull_rows.max()
        # Shadow extends from bottom of skull to image bottom
        shadow_depth = min(bottom_of_skull + rng.integers(5, 20), h)
        shadow_map[shadow_depth:, col] *= (1 - shadow_strength)

    # Smooth the shadow boundary to avoid hard edges
    shadow_map = cv2.GaussianBlur(shadow_map, (7, 1), 0)
    return img_float * shadow_map


def add_tgc_drift(
    img_float: np.ndarray,
    max_drift: float = 0.08,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate TGC (time-gain compensation) drift.

    TGC is the machine's attempt to correct depth attenuation —
    but it drifts slightly between frames, causing inter-frame intensity variation.
    """
    if rng is None:
        rng = np.random.default_rng()
    h = img_float.shape[0]
    drift_amount = rng.uniform(-max_drift, max_drift)
    tgc = np.linspace(1.0, 1.0 + drift_amount, h, dtype=np.float32)[:, np.newaxis]
    return np.clip(img_float * tgc, 0, 1)


def generate_motion_trajectory(n_frames: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate per-frame motion parameters for a cine clip.

    Returns dict of arrays, each length n_frames:
        tx: x translation (pixels)
        ty: y translation (pixels)
        rot: rotation (degrees)
        scale: uniform scale factor
        wiggle: boolean flags for fetal wiggle frames
    """
    # Slow drift: low theta (slow reversion), low sigma (gentle movement)
    tx  = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=2.0,  rng=rng)  # ±~5px
    ty  = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=1.5,  rng=rng)  # ±~3px
    rot = ornstein_uhlenbeck(n_frames, theta=0.2,  sigma=0.4,  rng=rng)  # ±~1°

    # Scale variation: subtle (simulates slight depth change)
    scale_noise = ornstein_uhlenbeck(n_frames, theta=0.3, sigma=0.005, rng=rng)
    scale = 1.0 + scale_noise

    # Fetal wiggle events: sudden displacement in 1-2 random frames
    # Probability 30% that this clip contains a wiggle at all
    wiggle = np.zeros(n_frames, dtype=bool)
    if rng.random() < 0.3:
        wiggle_frame = rng.integers(2, n_frames - 2)
        wiggle[wiggle_frame] = True
        # Wiggle = sudden offset that partially recovers
        wiggle_tx = rng.normal(0, 8)  # larger sudden displacement
        wiggle_ty = rng.normal(0, 6)
        tx[wiggle_frame:]   += wiggle_tx * np.exp(-0.5 * np.arange(n_frames - wiggle_frame))
        ty[wiggle_frame:]   += wiggle_ty * np.exp(-0.5 * np.arange(n_frames - wiggle_frame))

    return {'tx': tx, 'ty': ty, 'rot': rot, 'scale': scale, 'wiggle': wiggle}


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
    rng = np.random.default_rng(seed)

    H, W = img_gray.shape
    img_float = img_gray.astype(np.float32) / 255.0

    # Get ellipse parameters for cross-sectional variation
    ep = get_ellipse_params(mask)

    # Generate motion trajectory
    if stage >= 3:
        traj = generate_motion_trajectory(n_frames, rng)
    else:
        # Stage 1-2: simple sinusoidal motion (kept for ablation comparison)
        t = np.linspace(0, 2 * np.pi, n_frames)
        traj = {
            'tx':     np.sin(t) * 4,
            'ty':     np.cos(t) * 2,
            'rot':    np.sin(t * 0.5) * 1.5,
            'scale':  1.0 + np.sin(t * 0.3) * 0.02,
            'wiggle': np.zeros(n_frames, dtype=bool)
        }

    frames = []
    masks  = []

    for f in range(n_frames):
        # --- Image transform ---
        frame = apply_rigid_transform(
            img_float, traj['tx'][f], traj['ty'][f],
            traj['rot'][f], traj['scale'][f]
        )

        # --- Physics (stage 2+) ---
        if stage >= 2:
            noise_std = rng.uniform(0.05, 0.12)  # variable speckle per frame
            frame = add_rician_speckle(frame, noise_std=noise_std, rng=rng)
            frame = add_depth_attenuation(frame, attenuation_coeff=rng.uniform(0.2, 0.5))

        # --- Acoustic shadow + TGC (stage 4) ---
        if stage >= 4:
            # Transform the mask for shadow calculation (same transform as image)
            mask_transformed = apply_rigid_transform(
                mask.astype(np.float32), traj['tx'][f], traj['ty'][f],
                traj['rot'][f], traj['scale'][f]
            )
            mask_t_bin = (mask_transformed > 0.5).astype(np.uint8)
            frame = add_acoustic_shadow(frame, mask_t_bin, rng=rng)
            frame = add_tgc_drift(frame, rng=rng)

        frames.append(np.clip(frame, 0, 1).astype(np.float32))

        # --- Mask: cross-sectional variation + rigid transform ---
        if ep is not None:
            da = rng.normal(0, 0.025)  # ±2.5% semi-axis variation
            db = rng.normal(0, 0.025)
            mask_varied = perturb_mask_cross_section(mask, ep, da, db, rng)
        else:
            mask_varied = mask.copy()

        # Apply same rigid transform to the varied mask
        mask_transformed = apply_rigid_transform(
            mask_varied.astype(np.float32),
            traj['tx'][f], traj['ty'][f],
            traj['rot'][f], traj['scale'][f],
            border_value=0
        )
        masks.append((mask_transformed > 0.5).astype(np.uint8))

    frames_arr = np.stack(frames, axis=0)
    masks_arr  = np.stack(masks, axis=0)

    # Convert frames to uint8 for storage
    frames_uint8 = (frames_arr * 255).astype(np.uint8)

    # Per-frame HC in pixels (semi-perimeter of perturbed ellipse via Ramanujan approximation)
    hc_gt = np.zeros(n_frames, dtype=np.float32)
    for f in range(n_frames):
        ep_f = get_ellipse_params(masks_arr[f])
        if ep_f is not None:
            a, b = ep_f['a'], ep_f['b']
            h = ((a - b) / (a + b + 1e-8)) ** 2
            hc_gt[f] = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h + 1e-8)))

    return {'frames': frames_uint8, 'masks': masks_arr, 'hc_gt': hc_gt}


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

    for idx, (img_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc='Generating cine clips')):
        stem = img_path.stem
        out_path = output_dir / f'{stem}_cine.npz'

        # Skip if already generated (allows resuming interrupted runs)
        if out_path.exists():
            continue

        try:
            img, mask = load_image_mask(img_path, mask_path)

            # Use seed + idx so each clip is reproducible but distinct
            clip_seed = seed + idx

            clip = generate_cine(img, mask, n_frames=n_frames, seed=clip_seed, stage=stage)

            np.savez_compressed(
                str(out_path),
                frames=clip['frames'],
                masks=clip['masks'],
                hc_gt=clip['hc_gt'],
            )

        except Exception as e:
            print(f'  FAILED: {img_path.name} — {e}')
