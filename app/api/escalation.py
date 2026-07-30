"""Agentic reliability escalation — decide whether to trust a measurement.

A measurement with no self-assessment is the dangerous kind. This module reads
the signals the pipeline already produces (temporal reliability, per-frame HC
spread) and decides one of three things:

* ``ACCEPT`` — the frames agree; report the number.
* ``RE_CHECK`` — borderline; re-run against the *other* checkpoint and compare.
* ``FLAG_FOR_REVIEW`` — the frames disagree enough that a sonographer should
  confirm or re-acquire.

The agentic part is that ``RE_CHECK`` is a *decision to use a tool*, not a
pipeline stage. The re-run is expensive (a second full forward pass over a
16-frame clip), so it happens only when the uncertainty signal warrants it. The
final verdict then depends on what the tool returned.

Thresholds live in named constants at the top and are tuned by editing them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

log = logging.getLogger(__name__)


# ── tuning constants ─────────────────────────────────────────────────────────
#
# Reliability is 1 - (std / mean) over per-frame HC, so it is a coefficient of
# variation expressed as agreement. The bands mirror `confidence_label()` in
# app/inference.py; keep them in step or the badge and the verdict will
# disagree with each other in the UI.

RELIABILITY_ACCEPT_MIN = 0.97
"""At or above this, inter-frame agreement is strong enough to accept outright."""

RELIABILITY_FLAG_BELOW = 0.92
"""Below this, escalate to human review without spending a re-check."""

HC_STD_ACCEPT_MAX_MM = 2.0
"""Per-frame HC standard deviation must also be under this to accept outright."""

HC_STD_FLAG_ABOVE_MM = 5.0
"""Spread above this is flagged regardless of the ratio-based reliability score."""

CHECKPOINT_AGREEMENT_MAX_MM = 3.0
"""Two checkpoints must agree within this to clear a re-check.

3 mm is the ISUOG biometry threshold this project measures itself against, so a
cross-checkpoint disagreement wider than it is clinically material rather than
merely numerical.
"""

MIN_FRAMES_FOR_RELIABILITY = 2
"""Fewer measurable frames than this and the reliability score is not evidence."""

# The other checkpoint of the same family: pruned <-> unpruned. Comparing a
# static model against a temporal one would confound architecture with
# compression, so pairs stay within their own type.
ALTERNATE_CHECKPOINT = {
    "phase0": "phase4a",
    "phase4a": "phase0",
    "phase2": "phase4b",
    "phase4b": "phase2",
}


class Decision(StrEnum):
    ACCEPT = "ACCEPT"
    RE_CHECK = "RE_CHECK"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"


BADGE_COLOR = {
    Decision.ACCEPT: "green",
    Decision.RE_CHECK: "amber",
    Decision.FLAG_FOR_REVIEW: "red",
}


@dataclass
class ToolCall:
    """One tool the agent chose to invoke, and what came back."""

    tool: str
    reason: str
    result: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class EscalationOutcome:
    decision: Decision
    badge_color: str
    signals: dict
    tool_calls: list[ToolCall] = field(default_factory=list)
    rationale: str = ""
    justification: str | None = None
    justification_error: str | None = None


# ── tools the agent can call ─────────────────────────────────────────────────


def tool_rerun_alternate_checkpoint(
    record,
    run_inference: Callable[[str, object, float, float], dict],
) -> dict:
    """Re-run this exact input through the paired checkpoint.

    ``run_inference`` is injected rather than imported so this stays testable
    without loading real weights, and so the endpoint controls how a model is
    resolved.
    """
    alternate = ALTERNATE_CHECKPOINT.get(record.model_variant)
    if alternate is None:
        raise ValueError(f"No paired checkpoint is defined for '{record.model_variant}'.")

    result = run_inference(
        alternate,
        record.img_gray,
        record.pixel_spacing_mm,
        record.threshold,
    )
    return {
        "variant": alternate,
        "hc_mm": result.get("hc_mm"),
        "reliability": result.get("reliability"),
        "hc_std_mm": result.get("hc_std_mm"),
    }


def tool_compare_measurements(primary_hc: float | None, alternate: dict) -> dict:
    """Compare two checkpoints' HC and say whether they agree.

    ``agrees`` is None when either side could not measure HC — absence of a
    number is not agreement, and must not read as one.
    """
    alt_hc = alternate.get("hc_mm")
    if primary_hc is None or alt_hc is None:
        return {
            "primary_hc_mm": primary_hc,
            "alternate_hc_mm": alt_hc,
            "delta_mm": None,
            "agrees": None,
            "threshold_mm": CHECKPOINT_AGREEMENT_MAX_MM,
        }
    delta = abs(float(primary_hc) - float(alt_hc))
    return {
        "primary_hc_mm": round(float(primary_hc), 2),
        "alternate_hc_mm": round(float(alt_hc), 2),
        "delta_mm": round(delta, 2),
        "agrees": delta <= CHECKPOINT_AGREEMENT_MAX_MM,
        "threshold_mm": CHECKPOINT_AGREEMENT_MAX_MM,
    }


# ── the decision agent ───────────────────────────────────────────────────────


def _collect_signals(findings: dict) -> dict:
    """Read the uncertainty evidence, and say plainly when there is none.

    Single-frame inference hard-codes ``reliability = 1.0`` and
    ``hc_std_mm = 0.0``: placeholders so the response shape matches cine mode,
    not measurements. Treating them as evidence would produce a confident
    ACCEPT for every static result — a safety feature reporting certainty it
    never established. ``has_consistency_signal`` marks that difference so the
    agent reaches for a tool instead.
    """
    mode = findings.get("mode", "unknown")
    per_frame = [h for h in (findings.get("per_frame_hc") or []) if h is not None]
    has_signal = mode == "cine_clip" and len(per_frame) >= MIN_FRAMES_FOR_RELIABILITY

    return {
        "mode": mode,
        "hc_mm": findings.get("hc_mm"),
        "reliability": findings.get("reliability") if has_signal else None,
        "hc_std_mm": findings.get("hc_std_mm") if has_signal else None,
        "hc_range_mm": (
            round(float(np.max(per_frame) - np.min(per_frame)), 2) if has_signal else None
        ),
        "measurable_frames": len(per_frame),
        "has_consistency_signal": has_signal,
    }


def _triage(signals: dict) -> tuple[Decision, str]:
    """First pass: decide from the self-consistency signal alone."""
    if signals["hc_mm"] is None:
        return (
            Decision.FLAG_FOR_REVIEW,
            "No head circumference could be measured from this image, so there is "
            "nothing to accept.",
        )

    if not signals["has_consistency_signal"]:
        return (
            Decision.RE_CHECK,
            f"This is a {signals['mode']} result with no inter-frame agreement to "
            "measure, so self-consistency cannot be assessed from it. Independent "
            "evidence is needed before the number can be accepted.",
        )

    reliability = float(signals["reliability"])
    hc_std = float(signals["hc_std_mm"])

    if reliability < RELIABILITY_FLAG_BELOW or hc_std > HC_STD_FLAG_ABOVE_MM:
        return (
            Decision.FLAG_FOR_REVIEW,
            f"Inter-frame agreement is poor (reliability {reliability:.3f}, HC spread "
            f"{hc_std:.2f} mm). A second checkpoint would not resolve a disagreement "
            "this wide — the frames themselves are inconsistent.",
        )

    if reliability >= RELIABILITY_ACCEPT_MIN and hc_std <= HC_STD_ACCEPT_MAX_MM:
        return (
            Decision.ACCEPT,
            f"The frames agree closely (reliability {reliability:.3f}, HC spread "
            f"{hc_std:.2f} mm, both within the accept thresholds).",
        )

    return (
        Decision.RE_CHECK,
        f"Borderline: reliability {reliability:.3f} and HC spread {hc_std:.2f} mm fall "
        "between the accept and flag thresholds. A second opinion from the paired "
        "checkpoint should settle it.",
    )


def decide(
    record,
    run_inference: Callable[[str, object, float, float], dict],
) -> EscalationOutcome:
    """Assess the measurement, optionally use a tool, and return a verdict.

    The tool is invoked only when triage returns ``RE_CHECK``. A clear accept or
    a clear flag is already decided; spending a second forward pass on either
    would be a fixed pipeline pretending to be a decision.
    """
    signals = _collect_signals(record.findings or {})
    decision, rationale = _triage(signals)
    tool_calls: list[ToolCall] = []

    if decision is not Decision.RE_CHECK:
        return EscalationOutcome(
            decision=decision,
            badge_color=BADGE_COLOR[decision],
            signals=signals,
            rationale=rationale,
        )

    rerun = ToolCall(tool="rerun_alternate_checkpoint", reason=rationale)
    try:
        rerun.result = tool_rerun_alternate_checkpoint(record, run_inference)
    except Exception as exc:  # noqa: BLE001 — a failed tool must not 500
        rerun.error = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("escalation: re-check tool failed — %s", rerun.error, exc_info=True)
        tool_calls.append(rerun)
        # The tool was reached for because the evidence was insufficient. It
        # failed, so the evidence is still insufficient — escalate rather than
        # fall back to accepting an unverified borderline number.
        return EscalationOutcome(
            decision=Decision.FLAG_FOR_REVIEW,
            badge_color=BADGE_COLOR[Decision.FLAG_FOR_REVIEW],
            signals=signals,
            tool_calls=tool_calls,
            rationale=(
                f"{rationale} The re-check could not be completed ({rerun.error}), so the "
                "measurement remains unverified and is referred for review."
            ),
        )
    tool_calls.append(rerun)

    compare = ToolCall(
        tool="compare_measurements",
        reason="Quantify whether the two checkpoints agree within the ISUOG threshold.",
    )
    compare.result = tool_compare_measurements(signals["hc_mm"], rerun.result)
    tool_calls.append(compare)

    agrees = compare.result["agrees"]
    alt = rerun.result["variant"]

    if agrees is True:
        return EscalationOutcome(
            decision=Decision.ACCEPT,
            badge_color=BADGE_COLOR[Decision.ACCEPT],
            signals=signals,
            tool_calls=tool_calls,
            rationale=(
                f"{rationale} The {alt} checkpoint measured "
                f"{compare.result['alternate_hc_mm']} mm, within "
                f"{compare.result['delta_mm']} mm of the primary result and inside the "
                f"{CHECKPOINT_AGREEMENT_MAX_MM} mm agreement threshold."
            ),
        )

    if agrees is None:
        return EscalationOutcome(
            decision=Decision.FLAG_FOR_REVIEW,
            badge_color=BADGE_COLOR[Decision.FLAG_FOR_REVIEW],
            signals=signals,
            tool_calls=tool_calls,
            rationale=(
                f"{rationale} The {alt} checkpoint could not measure a head "
                "circumference on this image, so the borderline result is unconfirmed."
            ),
        )

    return EscalationOutcome(
        decision=Decision.FLAG_FOR_REVIEW,
        badge_color=BADGE_COLOR[Decision.FLAG_FOR_REVIEW],
        signals=signals,
        tool_calls=tool_calls,
        rationale=(
            f"{rationale} The {alt} checkpoint measured "
            f"{compare.result['alternate_hc_mm']} mm — a "
            f"{compare.result['delta_mm']} mm disagreement, wider than the "
            f"{CHECKPOINT_AGREEMENT_MAX_MM} mm threshold. Two checkpoints disagreeing "
            "clinically on the same image warrants sonographer confirmation or a "
            "repeat acquisition."
        ),
    )
