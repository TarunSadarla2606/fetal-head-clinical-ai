"""Shared pytest fixtures for fetal-head-clinical-ai."""

import numpy as np
import pytest


@pytest.fixture
def sample_us_image() -> np.ndarray:
    """256x256 uint8 grayscale image simulating an ultrasound frame."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (256, 256), dtype=np.uint8)


@pytest.fixture
def sample_batch_tensor():
    """Batch of 1 normalised image tensor (1, 1, 256, 256)."""
    import torch  # deferred so non-torch tests skip the import

    return torch.randn(1, 1, 256, 256)


@pytest.fixture
def sample_binary_mask() -> np.ndarray:
    """256x256 binary mask with an ellipse (simulates HC segmentation output)."""
    from skimage.draw import ellipse

    mask = np.zeros((256, 256), dtype=np.uint8)
    rr, cc = ellipse(128, 128, 80, 60, shape=mask.shape)
    mask[rr, cc] = 1
    return mask
