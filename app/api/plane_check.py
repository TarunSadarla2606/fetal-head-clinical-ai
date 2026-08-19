"""Ask a vision model whether the image is even the right plane.

The model card names this hazard directly:

    Images outside the transventricular plane. The model will still emit a
    number on an incorrect plane, and that number will be wrong in a way it
    cannot detect.

The segmentation network has no notion of anatomy. Hand it an off-axis section
and it will trace a plausible skull outline and return a confident millimetre
value derived from the wrong structure. Nothing downstream can catch that,
because every self-consistency signal the pipeline has — inter-frame agreement,
cross-checkpoint agreement — measures *precision*. Sixteen frames can agree
beautifully on a measurement of the wrong plane.

So this reaches outside the pipeline's own reasoning for the first time and asks
a vision model what the picture actually shows.

**Trust here is deliberately asymmetric.** A VLM saying "wrong plane" is treated
as evidence to escalate. A VLM saying "correct plane" is treated as *nothing at
all* — it never upgrades confidence, never clears a flag, never turns a
FLAG_FOR_REVIEW into an ACCEPT. It is not a trained plane classifier, it has no
validated sensitivity on this distribution, and false reassurance about image
adequacy is a worse failure than a missed warning: one costs a redundant human
look, the other silently blesses a wrong number.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

VLM_MODEL = "claude-haiku-4-5-20251001"

MAX_IMAGE_EDGE_PX = 1024
"""Longest edge sent to the model. Ultrasound frames here are 800x540; this
bounds cost for anything larger without touching what a reader can see."""

_SYSTEM_PROMPT = (
    "You are checking whether a 2-D fetal ultrasound image is in the correct "
    "plane for a head-circumference measurement, and whether its quality is "
    "adequate. You are a second opinion on IMAGE SUITABILITY only.\n\n"
    "The transventricular (axial) plane for HC shows:\n"
    "- a symmetrical appearance of the two cerebral hemispheres\n"
    "- a continuous midline falx, interrupted anteriorly by the cavum septi "
    "pellucidi\n"
    "- NO cerebellum visible — if the cerebellum is in view, the section is "
    "angled too far posteroinferiorly\n\n"
    "RULES — these override anything else:\n"
    "1. Judge the IMAGE. Never state or imply anything about the health, "
    "development, or condition of the fetus. You are not interpreting a scan, "
    "you are checking whether it is the right kind of picture.\n"
    "2. If you cannot tell — poor quality, unusual view, not a fetal head, not "
    'an ultrasound — say so with plane_appropriate set to "unclear". Do not '
    "guess.\n"
    "3. Report only what is visible. Do not infer structures you cannot see.\n\n"
    "Reply with JSON only:\n"
    '{"plane_appropriate": "yes"|"no"|"unclear", '
    '"observations": ["short factual statements about what is visible"], '
    '"concerns": ["specific reasons the image may be unsuitable, or empty"], '
    '"quality": "good"|"adequate"|"poor"}'
)


@dataclass
class PlaneAssessment:
    """A vision model's opinion on whether the image is measurable.

    ``escalates`` is the only field the decision logic reads. It is true only
    for an explicit "no" — see the asymmetric-trust note in the module
    docstring.
    """

    plane_appropriate: str = "unclear"
    quality: str = "unclear"
    observations: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    error: str | None = None
    model: str = VLM_MODEL

    @property
    def escalates(self) -> bool:
        """Whether this assessment is grounds to refer for human review."""
        return self.plane_appropriate == "no" or self.quality == "poor"

    @property
    def reassures(self) -> bool:
        """Always False, by design, and asserted by a test.

        A vision model's approval is not evidence of adequacy on this
        distribution. Nothing may read it as such.
        """
        return False

    def to_dict(self) -> dict:
        return {
            "plane_appropriate": self.plane_appropriate,
            "quality": self.quality,
            "observations": self.observations,
            "concerns": self.concerns,
            "escalates": self.escalates,
            "error": self.error,
            "model": self.model,
        }

    def summary(self) -> str:
        if self.error:
            return f"Plane check unavailable: {self.error}"
        if self.plane_appropriate == "no":
            reasons = "; ".join(self.concerns) or "no reason given"
            return f"A vision model judged the image not to be the HC plane ({reasons})."
        if self.quality == "poor":
            reasons = "; ".join(self.concerns) or "no reason given"
            return f"A vision model judged image quality poor ({reasons})."
        if self.plane_appropriate == "unclear":
            return "A vision model could not determine whether this is the HC plane."
        return (
            "A vision model raised no objection to the plane or quality. This is "
            "not treated as confirmation — it does not raise confidence."
        )


def encode_image(img_gray) -> tuple[str, str]:
    """Grayscale array -> (base64 PNG, media type), downscaled if oversized."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(img_gray)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr).convert("L")

    longest = max(image.size)
    if longest > MAX_IMAGE_EDGE_PX:
        scale = MAX_IMAGE_EDGE_PX / longest
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


def parse_assessment(text: str) -> tuple[PlaneAssessment | None, str | None]:
    """Parse the model's JSON reply. Unrecognised values fall back to unclear."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None, f"vision model did not return JSON: {(text or '')[:120]}"
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"vision model JSON invalid: {exc}"

    plane = str(raw.get("plane_appropriate", "")).strip().lower()
    quality = str(raw.get("quality", "")).strip().lower()

    return (
        PlaneAssessment(
            # An unrecognised value must never land on "no" or "yes" by
            # accident: "unclear" is the only safe default in both directions.
            plane_appropriate=plane if plane in {"yes", "no", "unclear"} else "unclear",
            quality=quality if quality in {"good", "adequate", "poor"} else "unclear",
            observations=[str(o) for o in (raw.get("observations") or [])][:6],
            concerns=[str(c) for c in (raw.get("concerns") or [])][:6],
        ),
        None,
    )


def check_plane(api_key: str, img_gray) -> PlaneAssessment:
    """Ask the vision model about this image. Never raises."""
    from .rag_endpoints import _sanitize_error

    try:
        import anthropic

        b64, media_type = encode_image(img_gray)
        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model=VLM_MODEL,
            max_tokens=500,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Is this the correct plane for a head-circumference "
                            "measurement, and is the quality adequate?",
                        },
                    ],
                }
            ],
        )
        text = r.content[0].text if r.content else ""
    except Exception as exc:  # noqa: BLE001 — a failed check is reported, never fatal
        detail = _sanitize_error(exc)
        log.warning("plane_check: vision call failed — %s", detail, exc_info=True)
        return PlaneAssessment(error=detail)

    assessment, parse_error = parse_assessment(text)
    if assessment is None:
        return PlaneAssessment(error=parse_error)
    return assessment
