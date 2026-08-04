"""Release gates for the fetal-HC models — the single place thresholds live.

Editing a number here changes what CI will allow to ship. Nothing else
hard-codes a gate value; ``scripts/evaluate.py`` writes these into its JSON
output so an archived run stays interpretable after the thresholds move.

These are **regression** gates, not accuracy claims. They bound how far the
deployed inference path may drift from its measured behaviour on the 12 demo
subjects. They are not comparable to the HC18 test-set figures in the README,
which were produced under different conditions on data not redistributed here.

Every gate is measurable from data in this repository. That constraint is
deliberate — see ``DICE_UNAVAILABLE_REASON``.
"""

from __future__ import annotations

# ── default gates, applied to any variant without an override ────────────────

DEFAULT_GATES: dict[str, tuple[str, float]] = {
    "hc_mae_mm": ("<=", 12.0),
    "hc_max_error_mm": ("<=", 40.0),
    "measurable_rate": (">=", 1.0),
    "mean_latency_ms": ("<=", 25_000.0),
}
"""metric -> (comparison, bound).

``measurable_rate`` is 1.0 because a model that cannot produce any measurement
on a clean demo image is broken outright, whatever its error elsewhere.

``mean_latency_ms`` is generous: temporal variants run 16 frames per image on a
CPU runner and measure around 11 s.
"""

# ── per-variant overrides ────────────────────────────────────────────────────

VARIANT_GATES: dict[str, dict[str, tuple[str, float]]] = {
    # Both static variants were fitting the HC ellipse to half a skull when the
    # calvarium segmented with a gap — see KNOWN_ISSUES and the fallback in
    # estimate_hc_mm. phase4a went 32.4 -> 11.8 mm MAE once that was fixed, so
    # its bound is tightened from 45 to 18: comfortably clear of the measured
    # value without sitting on the default 12.0, which it now only just clears
    # and would flap against.
    "phase4a": {
        "hc_mae_mm": ("<=", 18.0),
        "hc_max_error_mm": ("<=", 60.0),
    },
    # phase0's weights are LFS pointers outside CI, so its post-fix number is
    # not yet known — it measured 41.9 mm *before* the fix with the same
    # fragmentation signature. Left wide deliberately rather than guessed at.
    # TIGHTEN THIS once the next main run reports the new value.
    "phase0": {
        "hc_mae_mm": ("<=", 55.0),
        "hc_max_error_mm": ("<=", 250.0),
    },
}


def gates_for(variant: str) -> dict[str, tuple[str, float]]:
    """The effective gates for one variant: defaults with any override applied."""
    merged = dict(DEFAULT_GATES)
    merged.update(VARIANT_GATES.get(variant, {}))
    return merged


# ── metrics reported but deliberately not gated ──────────────────────────────

UNGATED_METRICS = ("isuog_pass_rate",)
"""Tracked in the model card, not enforced.

On 12 images this rate moves in 8.3% steps, so gating it would fail builds on
sampling granularity rather than on regressions.
"""

# ── known issues the gates are currently accommodating ───────────────────────

KNOWN_ISSUES = {
    "phase0": (
        "Measured 41.9 mm MAE with a 208 mm worst case before the fragmented-ring "
        "fix in estimate_hc_mm, which took phase4a from 32.4 to 11.8 mm on the "
        "same failure signature. phase0's weights are LFS pointers outside CI so "
        "its post-fix number is not yet known; this bound is the pre-fix one and "
        "should be tightened as soon as a main run reports the new value. Leaving "
        "it wide is a temporary, visible compromise, not a judgement that 55 mm "
        "is acceptable."
    ),
}

# ── what cannot be gated here, and why ───────────────────────────────────────

DICE_UNAVAILABLE_REASON = (
    "No ground-truth segmentation masks are present in this repository. "
    "training_set_pixel_size_and_HC.csv supplies a reference head circumference "
    "per image but no per-pixel annotation, so Dice cannot be computed here. "
    "The Dice figures quoted in the README come from the full HC18 test set, "
    "which is not redistributed with this repo. Add HC18 annotation masks under "
    "demo_subjects/masks/ to enable a Dice gate."
)
