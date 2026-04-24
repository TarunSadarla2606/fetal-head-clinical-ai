"""Evaluation metrics for fetal head segmentation.

Metrics:
  - Dice coefficient (Sorensen-Dice)
  - IoU (Jaccard index)
  - HC MAE and RMSE (mm)
  - R-squared
  - Reliability score (temporal: inter-frame HC consistency)

HC computation: Ramanujan ellipse approximation on the largest connected
component of the binary mask, scaled from model input space to mm via
pixel_spacing_mm and the HC18 original image width (800px).

GA computation: Hadlock (1984) cubic polynomial, clipped to [10, 42] weeks.
"""

import csv
import numpy as np
from pathlib import Path
from skimage.measure import label, regionprops
from typing import Optional, Dict, List

INPUT_W = 384   # model input width
ORIG_W  = 800   # HC18 original image width (for pixel-spacing correction)


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sorensen-Dice between two binary masks.

    Args:
        pred: Predicted binary mask [H, W].
        gt:   Ground-truth binary mask [H, W].

    Returns:
        Dice score in [0, 1].
    """
    pred = (pred > 0).astype(bool)
    gt   = (gt   > 0).astype(bool)
    tp   = (pred & gt).sum()
    return float(2 * tp) / float(pred.sum() + gt.sum() + 1e-8)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Jaccard index (Intersection over Union)."""
    pred = (pred > 0).astype(bool)
    gt   = (gt   > 0).astype(bool)
    return float((pred & gt).sum()) / float((pred | gt).sum() + 1e-8)


def estimate_hc_mm(
    mask: np.ndarray,
    pixel_spacing_mm: float,
    orig_w: int = ORIG_W,
    input_w: int = INPUT_W,
) -> Optional[float]:
    """Estimate HC in mm from a binary segmentation mask.

    Finds the largest connected component, extracts its semi-axes via
    regionprops, and applies Ramanujan's ellipse perimeter approximation.
    Pixel count is then scaled to mm using pixel_spacing and the
    HC18 original-to-model resize ratio.

    Ramanujan approximation:
        h  = ((a − b) / (a + b))²
        HC = π(a + b) × [1 + (3h) / (10 + √(4 − 3h))]

    Args:
        mask:             Binary mask [H, W] (any dtype, thresholded at 0).
        pixel_spacing_mm: mm per pixel (default 0.070 mm/px for HC18).
        orig_w:           HC18 original image width (800 px).
        input_w:          Model input width (384 px).

    Returns:
        HC in mm, or None if no component found.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    labeled  = label(mask_bin)
    regions  = regionprops(labeled)
    if not regions:
        return None

    largest = max(regions, key=lambda r: r.area)
    a = largest.axis_major_length / 2.0  # semi-major axis (pixels)
    b = largest.axis_minor_length / 2.0  # semi-minor axis (pixels)
    if a <= 0 or b <= 0:
        return None

    h     = ((a - b) / (a + b)) ** 2
    hc_px = np.pi * (a + b) * (1.0 + (3 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h)))

    # Scale: px in model space → px in original HC18 space → mm
    hc_mm = hc_px * pixel_spacing_mm * (orig_w / input_w)
    return float(hc_mm)


def hadlock_ga(hc_mm: float) -> Dict:
    """Estimate gestational age from HC using Hadlock (1984) polynomial.

    Formula: GA = 8.96 + 0.540×(HC/10) − 0.0040×(HC/10)² + 0.000399×(HC/10)³
    Validated range: 14–42 weeks. Clipped to [10, 42].

    Reference: Hadlock FP et al. AJR 1984;143:97-100.

    Args:
        hc_mm: Head circumference in mm.

    Returns:
        Dict with keys:
          'ga_weeks'  (float),
          'ga_str'    ("XwYd" string),
          'trimester' ("Early (<20w)", "Mid (20–30w)", or "Late (>30w)")
    """
    x      = hc_mm / 10.0
    ga_raw = 8.96 + 0.540 * x - 0.0040 * x**2 + 0.000399 * x**3
    ga     = float(np.clip(ga_raw, 10.0, 42.0))

    weeks = int(ga)
    days  = round((ga - weeks) * 7)
    ga_str = f"{weeks}w{days}d"

    if ga < 20:
        trimester = "Early (<20w)"
    elif ga < 30:
        trimester = "Mid (20–30w)"
    else:
        trimester = "Late (>30w)"

    return {"ga_weeks": ga, "ga_str": ga_str, "trimester": trimester}


def reliability_score(per_frame_hc: List[float]) -> float:
    """Temporal HC reliability: how consistent are per-frame measurements?

    reliability = max(0, 1 − std(HCs) / mean(HCs))

    Mapping:
      ≥ 0.97  → HIGH CONFIDENCE
      ≥ 0.92  → MODERATE CONFIDENCE
      < 0.92  → LOW — VERIFY MANUALLY

    Args:
        per_frame_hc: HC values (mm) across T frames.

    Returns:
        Reliability score in [0, 1].
    """
    hcs  = [h for h in per_frame_hc if h is not None and h > 0]
    if len(hcs) < 2:
        return 1.0
    return float(max(0.0, 1.0 - np.std(hcs) / (np.mean(hcs) + 1e-8)))


def load_pixel_spacing_csv(csv_path: Path | str) -> Dict[str, float]:
    """Load per-image pixel spacings from the HC18 metadata CSV.

    Args:
        csv_path: Path to training_set_pixel_size_and_HC.csv.

    Returns:
        Dict mapping image stem (e.g. '001_HC') → pixel_size_mm (float).
    """
    spacing: Dict[str, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["filename"]).stem
            spacing[stem] = float(row["pixel_size(mm)"])
    return spacing


def evaluate_predictions(
    pred_masks:       List[np.ndarray],
    gt_masks:         List[np.ndarray],
    pixel_spacing_mm: float = 0.070,
    pixel_spacings:   Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute full evaluation suite over a list of predictions.

    Args:
        pred_masks:       List of predicted binary masks [H, W].
        gt_masks:         List of ground-truth binary masks [H, W].
        pixel_spacing_mm: Fallback scalar spacing in mm (used when pixel_spacings is None).
        pixel_spacings:   Per-image pixel spacings in mm. When provided, overrides
                          pixel_spacing_mm for each image individually. Load from the
                          HC18 CSV with load_pixel_spacing_csv().

    Returns:
        Dict with keys: dice_mean, dice_std, iou_mean, mae_mm, rmse_mm, r2.
    """
    dice_scores, iou_scores, pred_hcs, gt_hcs = [], [], [], []

    for i, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
        ps = pixel_spacings[i] if pixel_spacings is not None else pixel_spacing_mm
        dice_scores.append(dice_coefficient(pred, gt))
        iou_scores.append(iou(pred, gt))
        ph = estimate_hc_mm(pred, ps)
        gh = estimate_hc_mm(gt,   ps)
        if ph is not None and gh is not None:
            pred_hcs.append(ph)
            gt_hcs.append(gh)

    pred_arr = np.array(pred_hcs)
    gt_arr   = np.array(gt_hcs)
    errors   = np.abs(pred_arr - gt_arr)

    ss_res = ((pred_arr - gt_arr) ** 2).sum()
    ss_tot = ((gt_arr - gt_arr.mean()) ** 2).sum()
    r2     = float(1.0 - ss_res / (ss_tot + 1e-8))

    return {
        "dice_mean": float(np.mean(dice_scores)),
        "dice_std":  float(np.std(dice_scores)),
        "iou_mean":  float(np.mean(iou_scores)),
        "mae_mm":    float(errors.mean()),
        "rmse_mm":   float(np.sqrt((errors**2).mean())),
        "r2":        r2,
    }
