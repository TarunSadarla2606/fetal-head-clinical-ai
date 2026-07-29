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

    import io

    from PIL import Image

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


# ── frame annotation (labels + key-frame marker) ─────────────────────────────


def test_annotate_frame_preserves_shape_and_dtype():
    rgb = np.dstack([_frame()] * 3)
    out = cine.annotate_frame(rgb, 0, 16, 245.3, False)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_annotate_frame_does_not_mutate_its_input():
    rgb = np.dstack([_frame()] * 3)
    before = rgb.copy()
    cine.annotate_frame(rgb, 3, 16, 245.3, True)
    assert np.array_equal(rgb, before)


def test_annotate_frame_key_marker_draws_amber_border():
    rgb = np.zeros((256, 384, 3), dtype=np.uint8)
    plain = cine.annotate_frame(rgb, 0, 16, None, False)
    keyed = cine.annotate_frame(rgb, 0, 16, None, True)
    # Amber border on the top row of the key frame only.
    assert keyed[0, 200, 0] > 200 and keyed[0, 200, 2] < 150
    assert plain[0, 200].sum() == 0


def test_annotate_frame_without_hc_still_labels_the_index():
    rgb = np.zeros((256, 384, 3), dtype=np.uint8)
    out = cine.annotate_frame(rgb, 0, 16, None, False)
    # Something was drawn in the bottom-left counter area.
    assert out[230:, :80].any()


# ── raw loop GIF ─────────────────────────────────────────────────────────────


def test_build_loop_gif_returns_a_data_uri():
    frames = cine.synthesize_cine_frames(_frame(), n_frames=16)
    uri = cine.build_loop_gif(frames)
    assert uri is not None
    assert _decode(uri).startswith(b"GIF89a")


def test_build_loop_gif_respects_its_smaller_budget():
    frames = cine.synthesize_cine_frames(_frame(), n_frames=30)
    assert len(_decode(cine.build_loop_gif(frames))) <= cine.LOOP_GIF_BYTES


def test_build_loop_gif_returns_none_for_degenerate_input():
    assert cine.build_loop_gif([]) is None
    assert cine.build_loop_gif([_frame()]) is None


def test_build_loop_gif_never_raises_on_garbage():
    assert cine.build_loop_gif(["nope", "also nope"]) is None


# ── overlay GIF with labels ──────────────────────────────────────────────────


def test_overlay_gif_accepts_per_frame_hc_and_key_index():
    frames = cine.synthesize_cine_frames(_frame(), n_frames=16)
    masks = [_ellipse_mask(cx=190 + i) for i in range(16)]
    hc = [245.0 + i * 0.1 for i in range(16)]
    uri = cine.build_overlay_gif(frames, masks, per_frame_hc=hc, key_frame_index=8)
    assert uri is not None
    assert len(_decode(uri)) <= cine.OVERLAY_GIF_BYTES


def test_overlay_gif_tolerates_none_entries_in_per_frame_hc():
    """A frame whose HC could not be estimated must not break labelling."""
    frames = cine.synthesize_cine_frames(_frame(), n_frames=8)
    masks = [_ellipse_mask() for _ in range(8)]
    hc = [245.0, None, 246.0, None, None, 247.0, None, 248.0]
    assert cine.build_overlay_gif(frames, masks, per_frame_hc=hc, key_frame_index=4) is not None


def test_overlay_gif_tolerates_a_short_per_frame_hc_list():
    frames = cine.synthesize_cine_frames(_frame(), n_frames=8)
    masks = [_ellipse_mask() for _ in range(8)]
    assert cine.build_overlay_gif(frames, masks, per_frame_hc=[245.0]) is not None


def test_overlay_gif_key_index_out_of_range_is_harmless():
    frames = cine.synthesize_cine_frames(_frame(), n_frames=8)
    masks = [_ellipse_mask() for _ in range(8)]
    assert cine.build_overlay_gif(frames, masks, key_frame_index=999) is not None


def test_both_gifs_together_stay_within_a_sane_response_budget():
    """The two animations ship in one JSON response — cap the combined cost."""
    frames = cine.synthesize_cine_frames(_frame(), n_frames=16)
    masks = [_ellipse_mask(cx=190 + i) for i in range(16)]
    total = len(_decode(cine.build_overlay_gif(frames, masks))) + len(
        _decode(cine.build_loop_gif(frames))
    )
    assert total <= cine.OVERLAY_GIF_BYTES + cine.LOOP_GIF_BYTES
