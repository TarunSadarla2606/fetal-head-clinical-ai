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
    # phase0 measured 13.5 mm MAE / 64.2 mm worst case in CI after the fix,
    # down from 41.9 / 208.0. Tightened from the stale 55/250 accordingly. It
    # still misses the 12 mm default, so it keeps an override rather than
    # having the default relaxed for everyone.
    "phase0": {
        "hc_mae_mm": ("<=", 20.0),
        "hc_max_error_mm": ("<=", 90.0),
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
    "static_variants": (
        "After the fragmented-ring fix in estimate_hc_mm, phase0 measures 13.5 mm "
        "MAE (was 41.9) and phase4a 11.8 mm (was 32.4). Both remain roughly 3x "
        "the temporal variants' 4.3 mm on the same 12 images. That residual gap "
        "is no longer a measurement bug — it is the single-frame path having no "
        "equivalent of the consensus averaging cine mode gets across 16 "
        "synthesised frames, which acts as test-time augmentation. Worth noting "
        "when comparing static and temporal numbers: some of the temporal "
        "advantage is the averaging, not the architecture."
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
