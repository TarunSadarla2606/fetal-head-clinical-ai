"""Report narrative generation must not claim LLM prose it did not produce.

Every narrative site is written ``_call_llm(...) or _rule_...``, so a failed
call is invisible in the output — the report renders either way. Paired with a
"Report type" caption derived from key *presence*, that produced reports
labelled LLM-generated whose every paragraph came from a template.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from app import report


def test_a_failed_call_is_recorded_rather_than_swallowed():
    with patch("anthropic.Anthropic", side_effect=RuntimeError("APIConnectionError")):
        with report.track_llm_calls() as run:
            out = report._call_llm("sk-test", "prompt")
    assert out is None, "the caller still falls back to its template"
    assert run.attempts == 1 and run.failures == 1
    assert run.used_llm is False
    assert run.fully_degraded is True
    assert "APIConnectionError" in run.last_error


def test_a_successful_call_reports_used_llm():
    with patch("anthropic.Anthropic") as A:
        A.return_value.messages.create.return_value.content = [
            type("B", (), {"text": "Narrative prose."})()
        ]
        with report.track_llm_calls() as run:
            out = report._call_llm("sk-test", "prompt")
    assert out == "Narrative prose."
    assert run.used_llm is True and run.failures == 0


def test_a_partially_degraded_run_still_counts_as_llm_generated():
    """One template paragraph among four does not make the report rule-based."""
    with report.track_llm_calls() as run:
        with patch("anthropic.Anthropic") as A:
            A.return_value.messages.create.return_value.content = [
                type("B", (), {"text": "ok"})()
            ]
            report._call_llm("sk-test", "p")
        with patch("anthropic.Anthropic", side_effect=RuntimeError("boom")):
            report._call_llm("sk-test", "p")
    assert run.attempts == 2 and run.failures == 1
    assert run.used_llm is True
    assert run.fully_degraded is False


def test_the_error_is_redacted_before_it_is_recorded():
    secret = "sk-ant-DONOTLEAK0123456789"
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": secret}):
        with patch("anthropic.Anthropic", side_effect=RuntimeError(f"bad key {secret}")):
            with report.track_llm_calls() as run:
                report._call_llm(secret, "prompt")
    assert "DONOTLEAK" not in run.last_error


def test_telemetry_is_isolated_between_concurrent_reports():
    """Report rendering runs in FastAPI's threadpool; counters must not bleed."""
    import threading

    results: dict[str, report.LlmTelemetry] = {}

    def render(name: str, fail: bool):
        with report.track_llm_calls() as run:
            if fail:
                with patch("anthropic.Anthropic", side_effect=RuntimeError("boom")):
                    report._call_llm("sk-test", "p")
            else:
                with patch("anthropic.Anthropic") as A:
                    A.return_value.messages.create.return_value.content = [
                        type("B", (), {"text": "ok"})()
                    ]
                    report._call_llm("sk-test", "p")
            results[name] = run

    threads = [
        threading.Thread(target=render, args=("good", False)),
        threading.Thread(target=render, args=("bad", True)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["good"].used_llm is True
    assert results["bad"].used_llm is False


def test_calls_outside_a_tracking_block_do_not_crash():
    """report.py is also imported by Streamlit, which may not wrap calls."""
    with patch("anthropic.Anthropic", side_effect=RuntimeError("boom")):
        assert report._call_llm("sk-test", "prompt") is None


def test_the_api_reports_template_prose_as_template_when_every_call_fails():
    """The mislabel this fixes: report_mode='llm' + a key returned used_llm=True.

    It shipped that way because _llm_static_narrative never raises — it
    substitutes templates internally — so the endpoint's try/except was dead
    code and the flag was effectively hard-coded True.
    """
    from app.api.reports_endpoints import _generate_narratives

    with patch("anthropic.Anthropic", side_effect=RuntimeError("APIConnectionError")):
        (p1, p2, p3, impression), used_llm = _generate_narratives(
            hc_mm=245.3,
            ga_str="21w 0d",
            ga_weeks=21.0,
            trimester="Second trimester (14–28w)",
            reliability=0.99,
            hc_std_mm=0.5,
            model_variant="phase0",
            elapsed_ms=120.0,
            api_key="sk-test",
            report_mode="llm",
        )
    assert used_llm is False, "template prose must not be labelled LLM-generated"
    assert p1 and impression, "the report still renders from templates"


def test_the_api_reports_llm_prose_as_llm_when_the_call_works():
    from app.api.reports_endpoints import _generate_narratives

    with patch("anthropic.Anthropic") as A:
        A.return_value.messages.create.return_value.content = [
            type("B", (), {"text": "Generated clinical narrative."})()
        ]
        _, used_llm = _generate_narratives(
            hc_mm=245.3,
            ga_str="21w 0d",
            ga_weeks=21.0,
            trimester="Second trimester (14–28w)",
            reliability=0.99,
            hc_std_mm=0.5,
            model_variant="phase0",
            elapsed_ms=120.0,
            api_key="sk-test",
            report_mode="llm",
        )
    assert used_llm is True
