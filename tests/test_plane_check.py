"""VLM plane check — and the asymmetry that makes it safe to add.

The model card names the hazard this closes: an off-plane image still yields a
confident millimetre value, and nothing downstream can detect it, because every
other signal in the pipeline measures *precision*. Sixteen frames can agree
beautifully on a measurement of the wrong plane.

The tests below are mostly about what the check is NOT allowed to do. A vision
model is not a validated plane classifier on this distribution, so its approval
must never raise confidence — only its objection may lower it.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.api import escalation, findings_store, plane_check
from app.api.plane_check import PlaneAssessment, encode_image, parse_assessment


def _record(**overrides):
    findings = {
        "hc_mm": 220.0,
        "reliability": 0.995,
        "hc_std_mm": 0.8,
        "per_frame_hc": [220.1, 219.8, 220.4],
        "mode": "cine_clip",
    }
    findings.update(overrides)
    fid = findings_store.store(
        img_gray=np.zeros((256, 384), dtype=np.uint8),
        model_variant="phase2",
        pixel_spacing_mm=0.0691,
        threshold=0.5,
        findings=findings,
    )
    return findings_store.get(fid)


def _returns(**kw):
    return lambda _img: PlaneAssessment(**kw)


def _no_rerun(*_a, **_k):
    raise AssertionError("the re-check tool must not run on this path")


# ── the asymmetry ────────────────────────────────────────────────────────────


def test_an_objection_escalates_an_otherwise_clean_accept():
    """The whole point: consistent numbers on an unmeasurable image."""
    out = escalation.decide(
        _record(),
        _no_rerun,
        check_plane=_returns(
            plane_appropriate="no", quality="good", concerns=["cerebellum is visible"]
        ),
    )
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW
    assert "cerebellum is visible" in out.rationale
    assert [c.tool for c in out.tool_calls] == ["check_plane"]


def test_approval_does_not_raise_confidence_it_only_fails_to_lower_it():
    """A 'yes' must be worth exactly nothing."""
    with_check = escalation.decide(
        _record(), _no_rerun, check_plane=_returns(plane_appropriate="yes", quality="good")
    )
    without_check = escalation.decide(_record(), _no_rerun)
    assert with_check.decision is without_check.decision is escalation.Decision.ACCEPT
    assert with_check.rationale == without_check.rationale, (
        "an approving plane check changed the reasoning, which means it was "
        "treated as evidence"
    )


def test_the_assessment_never_reports_itself_as_reassuring():
    for plane in ("yes", "no", "unclear"):
        assert PlaneAssessment(plane_appropriate=plane, quality="good").reassures is False


def test_a_plane_check_cannot_clear_an_existing_flag():
    """Approval must not rescue a result the numbers already condemned."""
    out = escalation.decide(
        _record(reliability=0.80, hc_std_mm=12.0),
        _no_rerun,
        check_plane=_returns(plane_appropriate="yes", quality="good"),
    )
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW


def test_the_check_is_not_spent_on_an_already_flagged_result():
    """It can only escalate, so on a flag there is nothing left to learn."""
    called = []
    escalation.decide(
        _record(reliability=0.80, hc_std_mm=12.0),
        _no_rerun,
        check_plane=lambda img: called.append(img) or PlaneAssessment(),
    )
    assert called == []


def test_poor_quality_escalates_even_when_the_plane_is_right():
    out = escalation.decide(
        _record(),
        _no_rerun,
        check_plane=_returns(
            plane_appropriate="yes", quality="poor", concerns=["heavy shadowing"]
        ),
    )
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW


# ── uncertainty and failure must not escalate ────────────────────────────────


def test_an_unclear_verdict_leaves_the_decision_alone():
    """'I cannot tell' is not an objection; treating it as one cries wolf."""
    out = escalation.decide(
        _record(), _no_rerun, check_plane=_returns(plane_appropriate="unclear")
    )
    assert out.decision is escalation.Decision.ACCEPT


def test_a_failed_vision_call_leaves_the_decision_alone():
    """Unlike the re-check tool, whose failure escalates.

    That tool was reached for because the evidence was thin. This one runs on a
    result whose own evidence is already strong, so losing it returns us to
    where we started rather than leaving a gap.
    """
    out = escalation.decide(
        _record(), _no_rerun, check_plane=_returns(error="APIConnectionError: Connection error.")
    )
    assert out.decision is escalation.Decision.ACCEPT
    assert out.tool_calls[0].result["error"]


def test_an_exception_in_the_checker_does_not_500():
    def boom(_img):
        raise RuntimeError("vision service exploded")

    out = escalation.decide(_record(), _no_rerun, check_plane=boom)
    assert out.decision is escalation.Decision.ACCEPT
    assert "vision service exploded" in out.tool_calls[0].error


def test_omitting_the_checker_reproduces_the_previous_behaviour_exactly():
    """No key must cost a check, never change a verdict."""
    a = escalation.decide(_record(), _no_rerun)
    b = escalation.decide(_record(), _no_rerun, check_plane=None)
    assert a.decision is b.decision and a.rationale == b.rationale
    assert a.tool_calls == b.tool_calls == []


# ── parsing ──────────────────────────────────────────────────────────────────


def test_a_well_formed_reply_parses():
    a, err = parse_assessment(
        '{"plane_appropriate": "no", "quality": "adequate", '
        '"observations": ["cerebellum visible"], "concerns": ["angled too far posteriorly"]}'
    )
    assert err is None
    assert a.plane_appropriate == "no" and a.escalates is True
    assert a.concerns == ["angled too far posteriorly"]


def test_json_wrapped_in_prose_still_parses():
    a, err = parse_assessment('Sure! {"plane_appropriate": "yes", "quality": "good"} Hope that helps.')
    assert err is None and a.plane_appropriate == "yes"


def test_an_unrecognised_verdict_falls_back_to_unclear_not_to_an_objection():
    """A garbled reply must not escalate, and must not reassure either."""
    a, _ = parse_assessment('{"plane_appropriate": "probably fine", "quality": "meh"}')
    assert a.plane_appropriate == "unclear" and a.quality == "unclear"
    assert a.escalates is False


def test_a_non_json_reply_is_an_error_not_a_verdict():
    a, err = parse_assessment("I'm not able to assess medical images.")
    assert a is None and "did not return JSON" in err


def test_malformed_json_is_an_error():
    a, err = parse_assessment('{"plane_appropriate": "no",,}')
    assert a is None and "JSON invalid" in err


def test_runaway_lists_are_capped():
    a, _ = parse_assessment(
        '{"plane_appropriate": "no", "quality": "poor", '
        f'"observations": {[f"o{i}" for i in range(30)]!r}, '
        f'"concerns": {[f"c{i}" for i in range(30)]!r}'.replace("'", '"') + "}"
    )
    assert len(a.observations) <= 6 and len(a.concerns) <= 6


# ── image encoding ───────────────────────────────────────────────────────────


def test_an_oversized_image_is_downscaled_before_being_sent():
    b64, media = encode_image(np.zeros((2000, 3000), dtype=np.uint8))
    assert media == "image/png"

    import base64
    import io

    from PIL import Image

    assert max(Image.open(io.BytesIO(base64.b64decode(b64))).size) <= plane_check.MAX_IMAGE_EDGE_PX


def test_a_normal_frame_is_sent_at_its_own_size():
    import base64
    import io

    from PIL import Image

    b64, _ = encode_image(np.zeros((540, 800), dtype=np.uint8))
    assert Image.open(io.BytesIO(base64.b64decode(b64))).size == (800, 540)


def test_a_float_image_is_coerced_rather_than_rejected():
    b64, _ = encode_image(np.full((64, 64), 0.5, dtype=np.float32))
    assert b64


# ── the prompt's own constraints ─────────────────────────────────────────────


def test_the_prompt_forbids_saying_anything_about_the_fetus():
    low = plane_check._SYSTEM_PROMPT.lower()
    assert "never state or imply anything about the health" in low
    assert "you are not interpreting a scan" in low


def test_the_prompt_offers_an_explicit_unclear_escape():
    """Without one, a model asked a binary question will guess."""
    assert '"unclear"' in plane_check._SYSTEM_PROMPT
    assert "do not guess" in plane_check._SYSTEM_PROMPT.lower()


def test_the_prompt_states_the_transventricular_criteria():
    low = plane_check._SYSTEM_PROMPT.lower()
    for landmark in ["falx", "cavum septi pellucidi", "cerebellum", "symmetrical"]:
        assert landmark in low


def test_a_failed_vision_call_is_reported_not_raised():
    with patch("anthropic.Anthropic", side_effect=RuntimeError("APIConnectionError")):
        a = plane_check.check_plane("sk-test", np.zeros((64, 64), dtype=np.uint8))
    assert a.error and "APIConnectionError" in a.error
    assert a.escalates is False, "a failed check must not escalate"


def test_the_summary_says_approval_is_not_confirmation():
    summary = PlaneAssessment(plane_appropriate="yes", quality="good").summary()
    assert "not treated as confirmation" in summary


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"plane_appropriate": "no", "quality": "good"}, True),
        ({"plane_appropriate": "yes", "quality": "poor"}, True),
        ({"plane_appropriate": "yes", "quality": "good"}, False),
        ({"plane_appropriate": "unclear", "quality": "adequate"}, False),
        ({"error": "boom"}, False),
    ],
)
def test_escalates_only_on_an_explicit_objection(kwargs, expected):
    assert PlaneAssessment(**kwargs).escalates is expected
