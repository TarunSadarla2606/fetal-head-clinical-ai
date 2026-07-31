"""``POST /findings/{id}/escalate`` — agentic reliability check on a measurement.

Separate from ``/infer`` on purpose. The agent may decide to re-run inference
against a second checkpoint, which on a 16-frame clip costs about as much as the
original request. Running that inside ``/infer`` would make every measurement pay
for the minority that need it, so the check is a follow-up call the client makes
once results are on screen.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from . import escalation, findings_store, inference_wrapper, model_manager
from .deps import verify_api_key
from .rag_endpoints import DISCLAIMER, _sanitize_error, get_api_key
from .schemas import EscalationResponse, ToolCallOut

log = logging.getLogger(__name__)

router = APIRouter(tags=["Reliability"])

LLM_MODEL = "claude-haiku-4-5-20251001"

_JUSTIFICATION_SYSTEM_PROMPT = (
    "You write the one-paragraph justification a quality-control system attaches "
    "to an automated fetal head-circumference measurement, explaining why it was "
    "accepted or referred for human review. The audience is a sonographer.\n\n"
    "RULES:\n"
    "1. Explain ONLY the decision described in the input. Do not re-decide, "
    "soften, or second-guess it, and do not add caveats it does not contain.\n"
    "2. Use the actual numbers given. Never invent a measurement, threshold or "
    "comparison that is not in the input.\n"
    "3. Never state or imply a clinical finding about the fetus. You are "
    "explaining a confidence assessment of a measurement, not interpreting it.\n"
    "4. Plain language a busy clinician can read in one pass. 2-3 sentences. No "
    "markdown, no bullet points, no preamble, no self-reference as an AI."
)


def _run_inference(variant: str, img_gray, pixel_spacing_mm: float, threshold: float) -> dict:
    """Run one checkpoint over an image, matching what /infer does for its type."""
    from app import cine
    from app.inference import N_FRAMES, TemporalFetaSegNet

    model = model_manager.get_model(variant)
    if model is None:
        raise RuntimeError(f"Checkpoint '{variant}' is not loaded on this server.")

    if isinstance(model, TemporalFetaSegNet):
        frames = cine.synthesize_cine_frames(img_gray, n_frames=N_FRAMES)
        return inference_wrapper.predict_cine_clip(
            model=model,
            frames=frames,
            pixel_spacing_mm=pixel_spacing_mm,
            threshold=threshold,
        )
    return inference_wrapper.predict_single_frame(
        model=model,
        img_gray=img_gray,
        pixel_spacing_mm=pixel_spacing_mm,
        threshold=threshold,
    )


def _build_prompt(outcome: escalation.EscalationOutcome) -> str:
    lines = [
        f"DECISION: {outcome.decision.value}",
        "",
        "EVIDENCE:",
        f"- mode: {outcome.signals['mode']}",
        f"- head circumference: {outcome.signals['hc_mm']} mm",
    ]
    if outcome.signals["has_consistency_signal"]:
        lines += [
            f"- inter-frame reliability: {outcome.signals['reliability']:.3f} "
            f"(accept at or above {escalation.RELIABILITY_ACCEPT_MIN}, "
            f"flag below {escalation.RELIABILITY_FLAG_BELOW})",
            f"- per-frame HC standard deviation: {outcome.signals['hc_std_mm']:.2f} mm "
            f"(accept at or below {escalation.HC_STD_ACCEPT_MAX_MM} mm, "
            f"flag above {escalation.HC_STD_FLAG_ABOVE_MM} mm)",
            f"- per-frame HC range: {outcome.signals['hc_range_mm']} mm "
            f"across {outcome.signals['measurable_frames']} measurable frames",
        ]
    else:
        lines.append(
            "- no inter-frame consistency signal exists for this result; the "
            "reliability field is a fixed placeholder, not a measurement"
        )

    for call in outcome.tool_calls:
        lines.append("")
        lines.append(f"TOOL USED: {call.tool} — {call.reason}")
        if call.error:
            lines.append(f"  failed: {call.error}")
        else:
            for k, v in call.result.items():
                lines.append(f"  {k}: {v}")

    lines += [
        "",
        "RULE-BASED RATIONALE (restate this in plain language, do not contradict it):",
        outcome.rationale,
        "",
        "Write the justification now.",
    ]
    return "\n".join(lines)


def _call_llm(api_key: str, prompt: str, max_tokens: int = 300) -> tuple[str | None, str | None]:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=_JUSTIFICATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        if not r.content:
            return None, "Empty response: the model returned no content blocks."
        return r.content[0].text.strip(), None
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        detail = _sanitize_error(exc)
        log.warning("escalation: justification call failed — %s", detail, exc_info=True)
        return None, detail


@router.post(
    "/findings/{finding_id}/escalate",
    response_model=EscalationResponse,
    summary="Decide whether this measurement can be trusted, and say why",
)
def escalate(finding_id: str, _: None = Depends(verify_api_key)) -> EscalationResponse:
    record = findings_store.get(finding_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown or expired finding_id. Re-run the analysis.",
        )

    outcome = escalation.decide(record, _run_inference)

    # The decision and its rule-based rationale stand on their own. The model
    # only rewrites them for a reader, so a failed call costs phrasing, never
    # the verdict.
    api_key, key_problem = get_api_key()
    if api_key:
        outcome.justification, outcome.justification_error = _call_llm(
            api_key, _build_prompt(outcome)
        )
    else:
        outcome.justification_error = key_problem or (
            "No ANTHROPIC_API_KEY configured on the server."
        )

    return EscalationResponse(
        finding_id=finding_id,
        decision=outcome.decision.value,
        badge_color=outcome.badge_color,
        rationale=outcome.rationale,
        justification=outcome.justification,
        justification_error=outcome.justification_error,
        used_llm=outcome.justification is not None,
        signals=outcome.signals,
        tool_calls=[
            ToolCallOut(
                tool=c.tool,
                reason=c.reason,
                result=c.result,
                error=c.error,
            )
            for c in outcome.tool_calls
        ],
        thresholds={
            "reliability_accept_min": escalation.RELIABILITY_ACCEPT_MIN,
            "reliability_flag_below": escalation.RELIABILITY_FLAG_BELOW,
            "hc_std_accept_max_mm": escalation.HC_STD_ACCEPT_MAX_MM,
            "hc_std_flag_above_mm": escalation.HC_STD_FLAG_ABOVE_MM,
            "checkpoint_agreement_max_mm": escalation.CHECKPOINT_AGREEMENT_MAX_MM,
        },
        disclaimer=DISCLAIMER,
    )
