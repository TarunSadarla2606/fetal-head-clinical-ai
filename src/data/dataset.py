"""HC18 Dataset and augmentation pipeline.

HC18 annotations are ellipse outlines (rings), not solid masks.
fill_hollow_mask() converts ring annotations to solid disks via
border flood-fill + inversion before training.

Augmentation (training only):
  - Random horizontal/vertical flip
  - Random rotation (±15 degrees)
  - Elastic deformation (alpha=200, sigma=10)
  - Rician noise injection (physically correct US noise model)
  - Coarse dropout (random rectangular region blackout)
  - Random brightness/contrast jitter

Dataset splits: 75% train / 20% val / 5% test (stratified by HC range).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


def fill_hollow_mask(mask: np.ndarray) -> np.ndarray:
    """Convert HC18 ring annotation to solid disk.

    HC18 ground truth marks the skull boundary as a 1px ring.
    Models need a solid region for Dice loss computation.
    Flood-fill from border with 255, then invert to recover interior.

    Args:
        mask: Binary ring mask [H, W] uint8.

    Returns:
        Solid binary mask [H, W] uint8.
    """
    # TODO: populate from notebook
    raise NotImplementedError


class HC18Dataset(Dataset):
    """PyTorch Dataset for the HC18 fetal head ultrasound challenge.

    Args:
        root: Path to HC18 dataset root (contains training_set/, test_set/).
        split: One of 'train', 'val', 'test'.
        augment: Apply training augmentations (ignored for val/test).
        input_h: Model input height (default: 256).
        input_w: Model input width (default: 384).
        img_mean: Normalization mean (default: 0.2).
        img_std: Normalization std (default: 0.15).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = True,
        input_h: int = 256,
        input_w: int = 384,
        img_mean: float = 0.2,
        img_std: float = 0.15,
    ) -> None:
        # TODO: populate from notebook
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (image, mask) tensors, both [1, H, W] float32."""
        raise NotImplementedError


def build_loaders(
    root: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    input_h: int = 256,
    input_w: int = 384,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for HC18.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # TODO: populate from notebook
    raise NotImplementedError
