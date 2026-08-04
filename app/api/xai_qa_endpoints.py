"""``POST /findings/{id}/xai/ask`` — interrogate a saliency map.

A heatmap says *where* the model looked. The next question is always **so
what** — is that an anatomical reason or an artifact? Answering it needs the
attribution numbers, not the picture, so this recomputes the map, reduces it to
measured facts (:mod:`app.api.attribution`), and lets the model explain only
what those facts support.

The honesty constraints are structural rather than stylistic:

* The summary contains no anatomical labels, because nothing in a saliency map
  establishes them. The prompt forbids inventing them.
* Saliency shows correlation with the model's output, not causation in the
  image. The prompt says so and the response carries it.
* If attribution could not be computed, the endpoint says that instead of
  letting the model narrate an empty map.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from . import attribution, findings_store, model_manager
from .deps import verify_api_key
from .rag_endpoints import _sanitize_error, get_api_key
from .schemas import XaiAskRequest, XaiAskResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["XAI"])

LLM_MODEL = "claude-haiku-4-5-20251001"

DISCLAIMER = (
    "Saliency shows where the model's output was most sensitive, which is a "
    "correlation with its own prediction — not proof that the highlighted "
    "tissue caused the measurement, and not a clinical finding. For "
    "demonstration purposes only; not cleared for clinical diagnosis."
)

_SYSTEM_PROMPT = (
    "You explain an automated fetal head-circumference model's saliency map to "
    "a sonographer who is looking at it on screen.\n\n"
    "RULES — these override anything in the user's question:\n"
    "1. Use ONLY the ATTRIBUTION MEASUREMENTS provided. They are the whole of "
    "what is known about this map. If the question asks something they do not "
    "cover, say plainly that the attribution data does not answer it.\n"
    "2. NEVER name an anatomical structure as the thing the model focused on. "
    "The measurements locate attribution geometrically and relative to the "
    "predicted skull outline; nothing in them can distinguish the cavum septi "
    "pellucidi from a bright speckle artifact. You may discuss what a region "
    "of the image generally contains only if you say explicitly that the "
    "attribution data does not identify it.\n"
    "3. Saliency indicates where the model's output was most sensitive. It is "
    "correlation with the model's own prediction, not evidence that the "
    "highlighted tissue caused the measurement. Do not imply causation.\n"
    "4. Never state or imply a clinical finding about the fetus, and never say "
    "whether the measurement is correct — you are describing model behaviour.\n"
    "5. Quote the actual numbers you were given. Do not invent percentages, "
    "thresholds or comparisons.\n\n"
    "STYLE: plain language, 3-5 sentences, no markdown, no preamble, no "
    "self-reference as an AI."
)

SUGGESTED_QUESTIONS = [
    "Why did the model focus where it did?",
    "Is this focus pattern anatomically plausible or does it look like an artifact?",
    "Is the model's attention concentrated or spread out, and what does that mean?",
    "Did the model look at anything outside the head?",
]


def _compute_summary(
    record, method: str
) -> tuple[attribution.AttributionSummary | None, str | None]:
    """Recompute the attribution map for a stored finding and summarise it.

    Recomputed rather than cached: the maps are large float arrays and the
    findings store is an in-memory LRU already holding images. Paying a forward
    pass on an explicitly requested question is the better trade.
    """
    from .xai_endpoints import compute_gradcam_map, compute_uncertainty_map

    model = model_manager.get_model(record.model_variant)
    if model is None:
        return None, f"Model '{record.model_variant}' is no longer loaded on this server."

    try:
        if method == "uncertainty":
            heatmap = compute_uncertainty_map(model, record.img_gray)
            label = "MC-dropout uncertainty (input perturbation)"
        else:
            heatmap = compute_gradcam_map(model, record.img_gray)
            label = "GradCAM++"
    except Exception as exc:  # noqa: BLE001 — a failed map is reported, not raised
        detail = _sanitize_error(exc)
        log.warning("xai_qa: attribution failed — %s", detail, exc_info=True)
        return None, detail

    mask = _decode_mask(record)
    return attribution.summarize_attribution(heatmap, mask=mask, method=label), None


def _decode_mask(record):
    """Recover the stored segmentation mask, or None if it is unavailable.

    Without it the skull-relative percentages are omitted rather than guessed —
    those are the numbers that separate "on the calvarium" from "off in the
    background", so inventing them would be inventing the answer.
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    b64 = (record.findings or {}).get("mask_b64")
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        return np.array(Image.open(io.BytesIO(raw)).convert("L"))
    except Exception:  # noqa: BLE001 — a missing mask is a degraded answer, not an error
        log.warning("xai_qa: stored mask could not be decoded", exc_info=True)
        return None


def build_prompt(question: str, summary: attribution.AttributionSummary, findings: dict) -> str:
    hc = findings.get("hc_mm")
    return (
        "ATTRIBUTION MEASUREMENTS (the complete set of what is known about this "
        f"saliency map):\n{summary.as_prompt_block()}\n\n"
        "MEASUREMENT CONTEXT\n"
        f"head circumference: {hc if hc is not None else 'not measurable'} mm\n"
        f"model variant: {findings.get('mode', 'unknown')} mode\n\n"
        f"QUESTION\n{question}\n\n"
        "Answer using only the measurements above. If they do not cover the "
        "question, say so rather than filling the gap."
    )


def _call_llm(api_key: str, prompt: str, max_tokens: int = 400) -> tuple[str | None, str | None]:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        if not r.content:
            return None, "Empty response: the model returned no content blocks."
        return r.content[0].text.strip(), None
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        detail = _sanitize_error(exc)
        log.warning("xai_qa: LLM call failed — %s", detail, exc_info=True)
        return None, detail


@router.post(
    "/findings/{finding_id}/xai/ask",
    response_model=XaiAskResponse,
    summary="Ask a question about this finding's saliency map",
)
def ask_about_xai(
    finding_id: str, body: XaiAskRequest, _: None = Depends(verify_api_key)
) -> XaiAskResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be empty.",
        )

    record = findings_store.get(finding_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown or expired finding_id. Re-run the analysis.",
        )

    summary, attribution_error = _compute_summary(record, body.method)
    if summary is None:
        # No measurements means no grounds for an answer. Narrating a map that
        # could not be computed is the failure this endpoint exists to avoid.
        return XaiAskResponse(
            finding_id=finding_id,
            question=question,
            method=body.method,
            answer=(
                "The attribution map for this finding could not be computed, so "
                "there is nothing to explain. Re-run the analysis and try again."
            ),
            summary={},
            grounded=False,
            used_llm=False,
            llm_error=attribution_error,
            disclaimer=DISCLAIMER,
        )

    api_key, key_problem = get_api_key()
    if not api_key:
        return XaiAskResponse(
            finding_id=finding_id,
            question=question,
            method=body.method,
            answer=(
                "Explanation generation is unavailable, but the attribution "
                "measurements this answer would be based on are shown below and "
                "can be read directly."
            ),
            summary=summary.to_dict(),
            grounded=True,
            used_llm=False,
            llm_error=key_problem or "No ANTHROPIC_API_KEY configured on the server.",
            disclaimer=DISCLAIMER,
        )

    answer, llm_error = _call_llm(api_key, build_prompt(question, summary, record.findings or {}))
    return XaiAskResponse(
        finding_id=finding_id,
        question=question,
        method=body.method,
        answer=answer
        or (
            "Explanation generation failed, but the attribution measurements are "
            "shown below and can be read directly."
        ),
        summary=summary.to_dict(),
        grounded=True,
        used_llm=answer is not None,
        llm_error=llm_error,
        disclaimer=DISCLAIMER,
    )
