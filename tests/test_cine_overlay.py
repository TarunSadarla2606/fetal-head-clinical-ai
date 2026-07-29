"""Cine-loop synthesis + animated segmentation-overlay GIF (app/cine.py)."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from app import cine


def _frame(h: int = 256, w: int = 384) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(20, 200, (h, w), dtype=np.uint8)


def _ellipse_mask(h: int = 256, w: int = 384, cx: int | None = None) -> np.ndarray:
    import cv2

    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (cx if cx is not None else w // 2, h // 2), (90, 60), 0, 0, 360, 1, -1)
    return m


def _decode(data_uri: str) -> bytes:
    assert data_uri.startswith("data:image/gif;base64,")
    return base64.b64decode(data_uri.split(",", 1)[1])


# ── synthesis ────────────────────────────────────────────────────────────────


def test_synthesize_returns_requested_frame_count():
    frames = cine.synthesize_cine_frames(_frame(), n_frames=16)
    assert len(frames) == 16
    assert all(f.shape == (cine.INPUT_H, cine.INPUT_W) for f in frames)
    assert all(f.dtype == np.uint8 for f in frames)


def test_synthesize_frames_actually_differ():
    """The whole point of the loop — identical frames animate to nothing."""
    frames = cine.synthesize_cine_frames(_frame(), n_frames=16)
    assert not np.array_equal(frames[0], frames[8])


def test_synthesize_is_deterministic_for_a_seed():
    a = cine.synthesize_cine_frames(_frame(), n_frames=8, seed=42)
    b = cine.synthesize_cine_frames(_frame(), n_frames=8, seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_synthesize_survives_very_short_clips(n):
    """n_frames below the OU warp-event window must not IndexError."""
    assert len(cine.synthesize_cine_frames(_frame(), n_frames=n)) == n


# ── overlay rendering ────────────────────────────────────────────────────────


def test_contour_overlay_marks_the_boundary():
    frame = _frame()
    out = cine.make_contour_overlay(frame, _ellipse_mask())
    assert out.shape == (256, 384, 3)
    # The contour colour should appear somewhere it did not before.
    assert (out[:, :, 1] > out[:, :, 0] + 40).any()


def test_contour_overlay_with_empty_mask_is_the_plain_frame():
    frame = _frame()
    out = cine.make_contour_overlay(frame, np.zeros((256, 384), dtype=np.uint8))
    assert np.array_equal(out[:, :, 0], out[:, :, 1])


def test_contour_overlay_resizes_a_mismatched_mask():
    out = cine.make_contour_overlay(_frame(256, 384), _ellipse_mask(128, 192))
    assert out.shape == (256, 384, 3)


# ── GIF assembly ─────────────────────────────────────────────────────────────


def test_build_overlay_gif_returns_a_data_uri():
    frames = cine.synthesize_cine_frames(_frame(), n_frames=16)
    masks = [_ellipse_mask(cx=190 + i) for i in range(16)]
    uri = cine.build_overlay_gif(frames, masks)
    assert uri is not None
    assert _decode(uri).startswith(b"GIF89a")


def test_build_overlay_gif_stays_under_budget():
    frames = cine.synthesize_cine_frames(_frame(), n_frames=30)
    masks = [_ellipse_mask(cx=190 + i) for i in range(30)]
    uri = cine.build_overlay_gif(frames, masks)
    assert len(_decode(uri)) <= cine.MAX_GIF_BYTES


def test_build_overlay_gif_caps_frame_count():
    """A long clip is subsampled, not encoded whole."""
    n = 120
    frames = cine.synthesize_cine_frames(_frame(), n_frames=n)
    masks = [_ellipse_mask() for _ in range(n)]
    data = _decode(cine.build_overlay_gif(frames, masks))

    from PIL import Image
    import io

    im = Image.open(io.BytesIO(data))
    assert im.n_frames <= cine.MAX_GIF_FRAMES


def test_build_overlay_gif_returns_none_for_degenerate_input():
    assert cine.build_overlay_gif([], []) is None
    assert cine.build_overlay_gif(_frame()[None], []) is None
    # A single frame is a still, not a loop — the caller already has overlay_b64.
    assert cine.build_overlay_gif([_frame()], [_ellipse_mask()]) is None


def test_build_overlay_gif_tolerates_a_short_mask_list():
    """A frame whose mask extraction failed animates bare rather than crashing."""
    frames = cine.synthesize_cine_frames(_frame(), n_frames=8)
    uri = cine.build_overlay_gif(frames, [_ellipse_mask(), _ellipse_mask()])
    assert uri is not None


def test_build_overlay_gif_never_raises_on_garbage():
    assert cine.build_overlay_gif(["not an image", "also not"], [None, None]) is None
