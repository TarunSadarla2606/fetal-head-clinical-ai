"""Agentic reliability escalation — decision paths, tool use, and abstention.

This is a safety feature, so the tests are written around what it must never
do: never report confidence it did not establish, never treat a missing number
as agreement, and never quietly accept a borderline result whose verification
failed.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import escalation, findings_store
from app.api.main import app

client = TestClient(app)


def _store(**overrides) -> str:
    findings = {
        "hc_mm": 220.0,
        "ga_str": "23w 2d",
        "ga_weeks": 23.3,
        "trimester": "Second trimester (14–28w)",
        "reliability": 0.995,
        "hc_std_mm": 0.8,
        "per_frame_hc": [220.1, 219.8, 220.4, 219.9, 220.2],
        "confidence_label": "HIGH CONFIDENCE",
        "mode": "cine_clip",
    }
    findings.update(overrides)
    return findings_store.store(
        img_gray=np.zeros((256, 384), dtype=np.uint8),
        model_variant="phase2",
        pixel_spacing_mm=0.0691,
        threshold=0.5,
        findings=findings,
    )


def _record(**overrides):
    return findings_store.get(_store(**overrides))


def _never_called(*_args, **_kwargs):
    raise AssertionError("the re-check tool must not run on this path")


def _returns(hc_mm):
    def run(_variant, _img, _spacing, _threshold):
        return {"hc_mm": hc_mm, "reliability": 0.99, "hc_std_mm": 0.5}

    return run


# ── ACCEPT ───────────────────────────────────────────────────────────────────


def test_consistent_frames_are_accepted_without_spending_a_re_check():
    """A clear accept is already decided — re-running would be a fixed pipeline."""
    out = escalation.decide(_record(reliability=0.995, hc_std_mm=0.8), _never_called)
    assert out.decision is escalation.Decision.ACCEPT
    assert out.badge_color == "green"
    assert out.tool_calls == []


# ── FLAG ─────────────────────────────────────────────────────────────────────


def test_poor_inter_frame_agreement_is_flagged_without_a_re_check():
    """A second checkpoint cannot fix frames that disagree with each other."""
    out = escalation.decide(
        _record(reliability=0.80, hc_std_mm=12.0, per_frame_hc=[200.0, 225.0, 210.0]),
        _never_called,
    )
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW
    assert out.badge_color == "red"
    assert out.tool_calls == []


def test_wide_hc_spread_is_flagged_even_when_the_ratio_score_looks_acceptable():
    """A large fetus makes std/mean small; absolute spread must still bite."""
    signals_ok_ratio = dict(reliability=0.975, hc_std_mm=8.0)
    out = escalation.decide(_record(**signals_ok_ratio), _never_called)
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW


def test_an_unmeasurable_image_is_flagged_rather_than_accepted():
    out = escalation.decide(_record(hc_mm=None), _never_called)
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW
    assert "nothing to accept" in out.rationale


# ── RE_CHECK: the agentic path ───────────────────────────────────────────────


def test_a_borderline_result_triggers_the_re_check_tool():
    out = escalation.decide(_record(reliability=0.95, hc_std_mm=3.5), _returns(221.0))
    assert [c.tool for c in out.tool_calls] == [
        "rerun_alternate_checkpoint",
        "compare_measurements",
    ]
    assert out.tool_calls[0].reason, "the agent must record why it reached for the tool"


def test_agreeing_checkpoints_clear_a_borderline_result():
    out = escalation.decide(_record(reliability=0.95, hc_std_mm=3.5), _returns(221.0))
    assert out.decision is escalation.Decision.ACCEPT
    assert out.tool_calls[1].result["delta_mm"] == 1.0
    assert out.tool_calls[1].result["agrees"] is True


def test_disagreeing_checkpoints_escalate_to_review():
    """Beyond the ISUOG 3 mm threshold the disagreement is clinically material."""
    out = escalation.decide(_record(reliability=0.95, hc_std_mm=3.5), _returns(232.0))
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW
    assert out.tool_calls[1].result["agrees"] is False
    assert out.tool_calls[1].result["delta_mm"] == 12.0


def test_the_re_check_uses_the_paired_checkpoint_not_an_arbitrary_one():
    seen = {}

    def run(variant, _img, _spacing, _threshold):
        seen["variant"] = variant
        return {"hc_mm": 220.5}

    escalation.decide(_record(reliability=0.95, hc_std_mm=3.5), run)
    assert seen["variant"] == "phase4b", "phase2's pair is phase4b"


def test_a_failed_re_check_escalates_rather_than_accepting_unverified():
    """The tool was needed because the evidence was thin. It still is."""

    def run(*_a, **_k):
        raise RuntimeError("checkpoint not loaded")

    out = escalation.decide(_record(reliability=0.95, hc_std_mm=3.5), run)
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW
    assert out.tool_calls[0].error and "checkpoint not loaded" in out.tool_calls[0].error


def test_an_unmeasurable_alternate_is_not_treated_as_agreement():
    """Absence of a second number is not confirmation of the first."""
    out = escalation.decide(_record(reliability=0.95, hc_std_mm=3.5), _returns(None))
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW
    assert out.tool_calls[1].result["agrees"] is None


# ── the single-frame trap ────────────────────────────────────────────────────


def test_single_frame_reliability_is_not_treated_as_evidence():
    """predict_single_frame hard-codes reliability=1.0 as a shape placeholder.

    Reading it as a measurement would mean every static result got a confident
    green ACCEPT that nothing had verified.
    """
    out = escalation.decide(
        _record(mode="single_frame", reliability=1.0, hc_std_mm=0.0, per_frame_hc=None),
        _returns(220.4),
    )
    assert out.signals["has_consistency_signal"] is False
    assert out.signals["reliability"] is None, "the placeholder must not be reported as a signal"
    assert out.tool_calls, "with no self-consistency signal the agent must seek evidence"
    assert out.decision is escalation.Decision.ACCEPT  # cleared by cross-checkpoint agreement


def test_a_thin_cine_clip_reports_its_frame_count_without_faking_a_range():
    """inference.py leaves reliability at 0.0 below MIN_FRAMES_FOR_RELIABILITY.

    So a thin clip flags on its own numbers rather than being special-cased —
    the fixture here mirrors what inference.py actually produces for one
    measurable frame, which the previous version of this test did not.
    """
    out = escalation.decide(
        _record(per_frame_hc=[220.1], reliability=0.0, hc_std_mm=0.0), _never_called
    )
    assert out.signals["measurable_frames"] == 1
    assert out.signals["hc_range_mm"] is None, "a range needs at least two points"
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW


# ── endpoint ─────────────────────────────────────────────────────────────────


def test_escalate_returns_the_verdict_with_its_evidence_and_thresholds():
    fid = _store(reliability=0.995, hc_std_mm=0.8)
    with patch.dict(os.environ, {}, clear=True):
        d = client.post(f"/findings/{fid}/escalate").json()
    assert d["decision"] == "ACCEPT"
    assert d["badge_color"] == "green"
    assert d["rationale"], "the rule-based rationale must never depend on the LLM"
    assert d["thresholds"]["checkpoint_agreement_max_mm"] == 3.0
    assert d["signals"]["reliability"] == pytest.approx(0.995)
    assert "not cleared for clinical diagnosis" in d["disclaimer"]


def test_escalate_survives_a_failed_justification_call():
    """A verdict is a safety output; losing prose must not lose the verdict."""
    fid = _store(reliability=0.80, hc_std_mm=12.0)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.escalation_endpoints._call_llm",
            return_value=(None, "APIConnectionError: Connection error."),
        ):
            d = client.post(f"/findings/{fid}/escalate").json()
    assert d["decision"] == "FLAG_FOR_REVIEW"
    assert d["used_llm"] is False
    assert d["justification"] is None
    assert "Connection error" in d["justification_error"]
    assert d["rationale"]


def test_escalate_reports_why_the_justification_is_missing_without_a_key():
    fid = _store()
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        d = client.post(f"/findings/{fid}/escalate").json()
    assert d["used_llm"] is False
    assert "ANTHROPIC_API_KEY" in d["justification_error"]


def test_escalate_uses_the_llm_only_to_rephrase_the_decision():
    fid = _store(reliability=0.80, hc_std_mm=12.0)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.escalation_endpoints._call_llm", return_value=("Flagged because…", None)
        ) as llm:
            d = client.post(f"/findings/{fid}/escalate").json()
    prompt = llm.call_args[0][1]
    assert "DECISION: FLAG_FOR_REVIEW" in prompt
    assert "12.0" in prompt, "the actual numbers must be in the prompt, not invented"
    assert d["used_llm"] is True and d["justification"] == "Flagged because…"


def test_escalate_404s_on_an_expired_finding():
    assert client.post("/findings/does-not-exist/escalate").status_code == 404


def test_the_justification_prompt_forbids_re_deciding_and_inventing_numbers():
    from app.api.escalation_endpoints import _JUSTIFICATION_SYSTEM_PROMPT as p

    low = p.lower()
    assert "do not re-decide" in low
    assert "never invent" in low
    assert "never state or imply a clinical finding" in low


def test_badge_colors_cover_every_decision():
    for decision in escalation.Decision:
        assert escalation.BADGE_COLOR[decision] in {"green", "amber", "red"}


def test_every_variant_has_a_distinct_paired_checkpoint():
    for variant, alternate in escalation.ALTERNATE_CHECKPOINT.items():
        assert alternate != variant
        assert escalation.ALTERNATE_CHECKPOINT[alternate] == variant, "pairing must be symmetric"


# ── regressions found in production ──────────────────────────────────────────
#
# Both of these shipped green because every test injected a fake run_inference
# and hand-built the findings dict. Nothing exercised the real code path or the
# real stored shape, so the module's own wiring was never checked.


def test_run_inference_resolves_its_imports():
    """The real _run_inference, not an injected stub.

    It shipped with `from . import cine` — cine lives at app/cine.py, so every
    re-check died with ImportError. Reaching the "checkpoint not loaded" guard
    proves the imports resolve; loading real weights is not the point.
    """
    from app.api.escalation_endpoints import _run_inference

    with pytest.raises(RuntimeError, match="not loaded"):
        _run_inference("phase0", np.zeros((256, 384), dtype=np.uint8), 0.07, 0.5)


def test_a_cine_result_uses_its_reliability_signal_rather_than_calling_a_tool():
    """has_consistency_signal keyed on per_frame_hc, which /infer never stored.

    Every cine result therefore reported "no signal" and spent a second full
    inference — the agent ignoring its own primary evidence, which is precisely
    the fixed-pipeline behaviour this design exists to avoid.
    """
    out = escalation.decide(
        _record(mode="cine_clip", reliability=0.995, hc_std_mm=0.8, per_frame_hc=None),
        _never_called,
    )
    assert out.signals["has_consistency_signal"] is True
    assert out.signals["reliability"] == pytest.approx(0.995)
    assert out.decision is escalation.Decision.ACCEPT
    assert out.tool_calls == []


def test_infer_stores_the_per_frame_series_the_verdict_reports():
    """The evidence panel showed measurable_frames: 0 on a 16-frame clip."""
    import inspect

    from app.api import main

    source = inspect.getsource(main.infer)
    assert '"per_frame_hc"' in source, (
        "/infer must persist per_frame_hc or the verdict loses its evidence"
    )


def test_a_cine_run_with_no_measurable_frames_still_flags():
    """It flags on reliability 0.0 from inference.py, not by being special-cased."""
    out = escalation.decide(
        _record(mode="cine_clip", reliability=0.0, hc_std_mm=0.0, per_frame_hc=[]),
        _never_called,
    )
    assert out.decision is escalation.Decision.FLAG_FOR_REVIEW
    assert out.signals["measurable_frames"] == 0
