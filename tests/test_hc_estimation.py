"""HC estimation from a segmentation mask — especially when the ring is broken.

`estimate_hc_mm` fitted its ellipse to the largest connected component. When
the calvarium is segmented with a gap, `fill_hollow_mask` cannot close it, the
skull survives as two arcs, and the larger arc is half a head — reported
confidently at roughly half the true circumference. Measured on real demo
subjects at 160 mm against a 322 mm reference.

The tests below pin both halves of the fix: unchanged behaviour on an intact
mask, and recovery on a fragmented one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference import INTACT_RING_MIN_SHARE, estimate_hc_mm  # noqa: E402

H, W = 256, 384
SPACING = 0.1


def _ring(cx=192, cy=128, rx=80, ry=64, thickness=6, gap_deg=None) -> np.ndarray:
    """An elliptical skull ring, optionally with an arc cut out of it."""
    yy, xx = np.ogrid[:H, :W]
    norm = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    outer = norm <= 1.0
    inner = norm <= (1.0 - thickness / min(rx, ry)) ** 2
    mask = (outer & ~inner).astype(np.uint8)
    if gap_deg:
        ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
        for centre in (0, 180):  # two opposing gaps -> two arcs
            lo, hi = centre - gap_deg / 2, centre + gap_deg / 2
            mask[((ang >= lo % 360) | (ang <= hi % 360)) if lo < 0 else ((ang >= lo) & (ang <= hi))] = 0
    return mask


def _filled(cx=192, cy=128, rx=80, ry=64) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.uint8)


def _largest_share(mask: np.ndarray) -> float:
    from skimage.measure import label, regionprops

    regs = regionprops(label(mask))
    total = sum(r.area for r in regs)
    return max(r.area for r in regs) / total


# ── the fix must not disturb healthy masks ───────────────────────────────────


def test_an_intact_ring_is_measured_exactly_as_before():
    """The fallback must be a no-op wherever the mask is one component."""
    mask = _ring()
    assert _largest_share(mask) >= INTACT_RING_MIN_SHARE

    from skimage.measure import label, regionprops

    largest = max(regionprops(label(mask)), key=lambda r: r.area)
    a, b = largest.axis_major_length / 2, largest.axis_minor_length / 2
    h = ((a - b) / (a + b + 1e-8)) ** 2
    expected_px = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h + 1e-8)))

    from app.inference import INPUT_W, ORIG_W

    assert estimate_hc_mm(mask, SPACING) == pytest.approx(
        expected_px * SPACING * (ORIG_W / INPUT_W)
    )


def test_a_filled_ellipse_is_unaffected():
    assert estimate_hc_mm(_filled(), SPACING) is not None


def test_small_speckle_does_not_trigger_the_fallback():
    """Stray blobs are exactly what the largest-component filter is for."""
    mask = _ring()
    mask[10:14, 10:14] = 1
    mask[240:244, 370:374] = 1
    intact = estimate_hc_mm(_ring(), SPACING)
    assert estimate_hc_mm(mask, SPACING) == pytest.approx(intact)


# ── the fix must rescue fragmented rings ─────────────────────────────────────


def test_a_ring_broken_into_arcs_is_not_measured_as_one_arc():
    """The real failure: two arcs, the larger of which is half a head."""
    whole = _ring()
    broken = _ring(gap_deg=40)
    assert _largest_share(broken) < INTACT_RING_MIN_SHARE, "fixture must actually fragment"

    intact_hc = estimate_hc_mm(whole, SPACING)
    broken_hc = estimate_hc_mm(broken, SPACING)

    # Removing two small arcs must not halve the measurement.
    assert broken_hc == pytest.approx(intact_hc, rel=0.15), (
        f"a fragmented ring measured {broken_hc:.1f} against {intact_hc:.1f} for the "
        "same skull — this is the half-a-head failure the fallback exists to prevent"
    )


def test_the_old_behaviour_would_have_failed_this_case():
    """Guards the test itself: confirm the fixture reproduces the original bug."""
    from skimage.measure import label, regionprops

    from app.inference import INPUT_W, ORIG_W, _ellipse_perimeter_mm

    broken = _ring(gap_deg=40)
    largest = max(regionprops(label(broken)), key=lambda r: r.area)
    old = _ellipse_perimeter_mm(
        largest.axis_major_length, largest.axis_minor_length, SPACING, INPUT_W, ORIG_W
    )
    intact = estimate_hc_mm(_ring(), SPACING)
    assert old < intact * 0.85, (
        "the fixture no longer reproduces the original under-measurement, so the "
        "test above would pass even with the fix reverted"
    )


# ── edges ────────────────────────────────────────────────────────────────────


def test_an_empty_mask_returns_none():
    assert estimate_hc_mm(np.zeros((H, W), dtype=np.uint8), SPACING) is None


def test_a_degenerate_speck_returns_none_rather_than_a_tiny_hc():
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[100:101, 100:101] = 1
    assert estimate_hc_mm(mask, SPACING) is None


def test_hc_scales_linearly_with_pixel_spacing():
    """A wrong spacing scales the result — the model card says so; pin it."""
    mask = _ring()
    assert estimate_hc_mm(mask, 0.2) == pytest.approx(2 * estimate_hc_mm(mask, 0.1))
