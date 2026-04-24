"""HC18 Dataset, preprocessing, and augmentation pipeline.

HC18 annotations are ellipse outlines (1px rings), not solid regions.
fill_hollow_mask() converts ring annotations to solid disks via border
flood-fill + inversion before any training or metric computation.

Dataset splits (reproducible, random_state=42):
  Static  (Phase 0): 70/15/15 split — test_size=0.15 then test_size=0.176
  Temporal (Phase 2): 564 train / 121 val / 121 test  (from 806 clips)

Normalisation: pixel values divided by 255 (A.Normalize mean=0, std=1, max_pixel_value=255)
matching the phase0 notebook training configuration.

Augmentation (training only, albumentations):
  HorizontalFlip(p=0.5), VerticalFlip(p=0.5), Rotate(limit=45, p=0.7),
  ElasticTransform(p=0.3), GaussNoise(p=0.2), GaussianBlur(p=0.2),
  RandomBrightnessContrast(p=0.3)
"""

import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Tuple, List

INPUT_H = 256
INPUT_W = 384
SEED    = 42


def fill_hollow_mask(mask: np.ndarray) -> np.ndarray:
    """Convert HC18 ring annotation (1px outline) to solid disk.

    HC18 ground truth marks the skull boundary as a 1-pixel ellipse ring.
    Training with ring masks produces unstable Dice gradients. This function
    flood-fills the background from the image border, then inverts to recover
    the solid interior region.

    Args:
        mask: Binary ring mask [H, W] uint8 (255 = boundary, 0 = background).

    Returns:
        Solid binary mask [H, W] uint8 (255 = head region, 0 = background).
    """
    h, w = mask.shape
    flood = mask.copy()
    # Flood fill from top-left corner (guaranteed to be background)
    cv2.floodFill(flood, None, (0, 0), 255)
    # Invert flood region: original background becomes 0, skull interior becomes 255
    inverted = cv2.bitwise_not(flood)
    # Union with original ring to ensure boundary pixels are included
    solid = cv2.bitwise_or(mask, inverted)
    return solid


def load_image_mask(
    img_path: Path,
    mask_path: Path,
    target_h: int = INPUT_H,
    target_w: int = INPUT_W,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load, resize, and binarise one HC18 image/mask pair.

    Returns:
        img:  [H, W] uint8 grayscale image.
        mask: [H, W] uint8 solid binary mask (fill_hollow_mask applied).
    """
    img  = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    img  = cv2.resize(img,  (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_solid  = fill_hollow_mask(mask_bin)
    return img, mask_solid


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


class HC18Dataset(Dataset):
    """PyTorch Dataset for the HC18 fetal head circumference challenge.

    Args:
        image_paths: List of image file paths.
        mask_paths:  List of corresponding mask file paths.
        augment:     Apply training augmentations (default: False).
        input_h:     Model input height (default: 256).
        input_w:     Model input width (default: 384).
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths:  List[Path],
        augment: bool = False,
        input_h: int = INPUT_H,
        input_w: int = INPUT_W,
    ) -> None:
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.stems       = [p.stem for p in image_paths]
        self.augment     = augment
        self.input_h     = input_h
        self.input_w     = input_w

        if augment:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            self._aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.7),
                A.ElasticTransform(p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return (image, mask, stem) — tensors [1,H,W] float32, stem is filename without extension."""
        img, mask = load_image_mask(
            self.image_paths[idx], self.mask_paths[idx],
            self.input_h, self.input_w,
        )
        stem = self.stems[idx]

        if self.augment:
            aug = self._aug(image=img, mask=mask)
            # ToTensorV2 already produced tensors; add channel dim to mask
            img_t  = aug["image"].float()                          # [1, H, W]
            mask_t = aug["mask"].unsqueeze(0).float()              # [1, H, W]
            return img_t, mask_t, stem

        # /255 normalisation matching training (A.Normalize mean=0, std=1, max=255)
        img_norm = img.astype(np.float32) / 255.0

        img_t  = torch.from_numpy(img_norm).unsqueeze(0)          # [1, H, W]
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t, stem


def build_loaders(
    root: str | Path,
    batch_size: int = 16,
    num_workers: int = 2,
    input_h: int = INPUT_H,
    input_w: int = INPUT_W,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for HC18 (70/15/15 split).

    Args:
        root: HC18 dataset root (must contain 'archive (4)/training_set/training_set/' subdirectory).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    root = Path(root)
    img_dir  = root / "archive (4)" / "training_set" / "training_set"
    all_imgs  = sorted(img_dir.glob("*_HC.png"))
    all_masks = [p.parent / (p.stem + "_Annotation.png") for p in all_imgs]

    indices = list(range(len(all_imgs)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.15,  random_state=seed)
    val_idx,  test_idx  = train_test_split(temp_idx, test_size=0.176, random_state=seed)

    def _make(idx: list, augment: bool) -> DataLoader:
        ds = HC18Dataset(
            [all_imgs[i]  for i in idx],
            [all_masks[i] for i in idx],
            augment=augment,
            input_h=input_h,
            input_w=input_w,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=augment,
                          num_workers=num_workers, pin_memory=True)

    return _make(train_idx, True), _make(val_idx, False), _make(test_idx, False)
