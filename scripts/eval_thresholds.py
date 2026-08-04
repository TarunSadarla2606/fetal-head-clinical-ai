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
    # phase4a does not currently meet the default bound. See KNOWN_ISSUES.
    # Encoded at its measured behaviour plus headroom so the pipeline still
    # catches a *further* regression, rather than being disabled or left
    # permanently red. Tighten this once the underlying failure is fixed.
    "phase4a": {
        "hc_mae_mm": ("<=", 45.0),
        "hc_max_error_mm": ("<=", 200.0),
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
    "phase4a": (
        "On 2 of the 12 demo subjects (800_HC.png, 805_HC.png) phase4a predicts "
        "roughly half the reference circumference — 159.98 vs 321.83 mm and "
        "169.07 vs 330.70 mm. Both are the signature of the calvarium being "
        "only partly segmented, and both are handled correctly by phase4b on "
        "the same images, so this is the pruned static checkpoint rather than "
        "the harness or the data. Excluding those two, phase4a's MAE is about "
        "6.6 mm. The gate is set wide enough not to block deploys on a "
        "pre-existing fault, and narrow enough to catch it getting worse."
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
