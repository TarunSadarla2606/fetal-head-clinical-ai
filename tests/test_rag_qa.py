"""Retrieval-grounded Q&A — knowledge_base indexing + /findings/{id}/ask.

The grounding guarantees are the point of this feature, so they are asserted
directly: no retrieval means no LLM call, retrieved chunks are returned to the
client so an answer can be checked against its evidence, and off-topic
questions are refused rather than answered from model memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import findings_store, knowledge_base, rag_endpoints
from app.api.main import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _fresh_index():
    knowledge_base.reset_retriever()
    yield
    knowledge_base.reset_retriever()


def _store_finding(**overrides) -> str:
    findings = {
        "hc_mm": 92.1,
        "ga_str": "13w 6d",
        "ga_weeks": 13.9,
        "trimester": "First trimester (<14w)",
        "reliability": 0.9965,
        "hc_std_mm": 0.321,
        "confidence_label": "HIGH CONFIDENCE",
        "mode": "cine_clip",
        "validation": {"quality_label": "good", "quality_score": 0.71},
    }
    findings.update(overrides)
    return findings_store.store(
        img_gray=np.zeros((256, 384), dtype=np.uint8),
        model_variant="phase4b",
        pixel_spacing_mm=0.0691,
        threshold=0.5,
        findings=findings,
    )


# ── knowledge base ───────────────────────────────────────────────────────────


def test_knowledge_dir_exists_and_indexes():
    r = knowledge_base.get_retriever()
    assert r.ready, "knowledge/ failed to index — Q&A would be permanently unavailable"
    assert len(r.chunks) >= 15


def test_readme_is_never_indexed():
    """knowledge/README.md documents the corpus; citing it would be nonsense."""
    files = {c.source_file for c in knowledge_base.get_retriever().chunks}
    assert not any(f.lower() == "readme.md" for f in files)


def test_chunks_carry_citation_and_provenance():
    chunks = knowledge_base.get_retriever().chunks
    for c in chunks:
        assert " § " in c.citation
        assert c.text.strip()
        # Provenance blockquotes must be stripped from the indexed body.
        assert not c.text.lstrip().startswith(">")


def test_provisional_flag_tracks_the_todo_marker():
    chunks = knowledge_base.get_retriever().chunks
    provisional = [c for c in chunks if c.provisional]
    assert provisional, "expected TODO(verbatim) sections to be flagged"
    # project_metrics.md describes this repo's own code and is fully sourced.
    assert all(c.source_file != "project_metrics.md" for c in provisional)


def test_parse_markdown_splits_on_h2_only(tmp_path: Path):
    f = tmp_path / "sample.md"
    f.write_text(
        "# Title\npreamble\n\n"
        "## First heading\n> Source: somewhere\nbody one\n### sub\nstill one\n\n"
        "## Second heading\nbody two\n",
        encoding="utf-8",
    )
    chunks = knowledge_base.parse_markdown(f)
    assert [c.heading for c in chunks] == ["First heading", "Second heading"]
    assert "still one" in chunks[0].text  # ### stays inside its parent
    assert "preamble" not in " ".join(c.text for c in chunks)
    assert chunks[0].source_note == "Source: somewhere"


def test_missing_knowledge_dir_degrades_to_empty(tmp_path: Path):
    assert knowledge_base.load_chunks(tmp_path / "nope") == []


def test_retrieval_finds_the_reliability_definition():
    """Regression: word-only TF-IDF scored these at zero for 'reliable'."""
    hits = knowledge_base.retrieve("how reliable is this measurement?")
    assert hits, "no chunks retrieved for a core question"
    assert "reliability" in hits[0].chunk.heading.lower()


def test_retrieval_is_topical_for_each_reference_file():
    cases = {
        "what does the Dice coefficient measure?": "project_metrics.md",
        "which plane should head circumference be measured in?": "isuog_hc_measurement.md",
        "how is gestational age calculated from HC?": "hadlock_ga.md",
    }
    for question, expected_file in cases.items():
        hits = knowledge_base.retrieve(question)
        assert hits, f"nothing retrieved for {question!r}"
        assert expected_file in {h.chunk.source_file for h in hits}, question


@pytest.mark.parametrize(
    "question",
    ["what is the capital of France?", "write me a poem about cats", "how do I bake bread"],
)
def test_off_topic_questions_retrieve_nothing(question):
    """The threshold must reject noise, or off-topic answers get 'grounded'."""
    assert knowledge_base.retrieve(question) == []


# ── /findings/{id}/ask ───────────────────────────────────────────────────────


def test_ask_404_for_unknown_finding():
    r = client.post("/findings/does-not-exist/ask", json={"question": "how reliable is this?"})
    assert r.status_code == 404


def test_ask_422_for_empty_question():
    fid = _store_finding()
    assert client.post(f"/findings/{fid}/ask", json={"question": "   "}).status_code == 422
    assert client.post(f"/findings/{fid}/ask", json={"question": ""}).status_code == 422


def test_ask_refuses_when_nothing_retrieved_and_never_calls_the_llm():
    """The core guarantee: no evidence → no model call → no ungrounded answer."""
    fid = _store_finding()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("app.api.rag_endpoints._call_llm") as llm:
            r = client.post(f"/findings/{fid}/ask", json={"question": "capital of France?"})
    assert r.status_code == 200
    d = r.json()
    assert d["grounded"] is False
    assert d["used_llm"] is False
    assert d["citations"] == [] and d["chunks"] == []
    llm.assert_not_called()


def test_ask_returns_retrieved_chunks_so_the_answer_can_be_audited():
    fid = _store_finding()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.rag_endpoints._call_llm",
            return_value=(
                "Reliability is inter-frame agreement "
                "[project_metrics.md § Temporal reliability score — definition].",
                None,
            ),
        ):
            r = client.post(
                f"/findings/{fid}/ask", json={"question": "how reliable is this measurement?"}
            )
    d = r.json()
    assert d["grounded"] is True and d["used_llm"] is True
    assert d["chunks"], "retrieved evidence must be returned to the client"
    for c in d["chunks"]:
        assert c["text"] and c["citation"] and "score" in c


def test_ask_reports_only_the_citations_the_answer_used():
    """Listing every retrieved chunk as a citation would overstate the sourcing."""
    fid = _store_finding()
    used = "project_metrics.md § Temporal reliability score — definition"
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.rag_endpoints._call_llm", return_value=(f"Answer text [{used}].", None)
        ):
            r = client.post(
                f"/findings/{fid}/ask", json={"question": "how reliable is this measurement?"}
            )
    d = r.json()
    assert d["citations"] == [used]
    assert len(d["chunks"]) > 1, "fixture should retrieve more than the one cited"


def test_ask_passes_the_measurement_numbers_into_the_prompt():
    fid = _store_finding(hc_mm=245.3, ga_str="21w 0d")
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("app.api.rag_endpoints._call_llm", return_value=("ok", None)) as llm:
            client.post(f"/findings/{fid}/ask", json={"question": "how reliable is this?"})
    prompt = llm.call_args[0][1]
    assert "245.3 mm" in prompt
    assert "21w 0d" in prompt
    assert "REFERENCE EXCERPTS" in prompt and "QUESTION" in prompt


def test_ask_marks_provisional_when_any_chunk_is_unverified():
    fid = _store_finding()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("app.api.rag_endpoints._call_llm", return_value=("ok", None)):
            r = client.post(
                f"/findings/{fid}/ask",
                json={"question": "which plane should HC be measured in?"},
            )
    d = r.json()
    assert d["any_provisional"] is True
    assert any(c["provisional"] for c in d["chunks"])


def test_ask_degrades_to_excerpts_without_an_api_key():
    fid = _store_finding()
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        r = client.post(
            f"/findings/{fid}/ask", json={"question": "how reliable is this measurement?"}
        )
    d = r.json()
    assert d["grounded"] is True and d["used_llm"] is False
    assert d["chunks"], "excerpts should still be returned so the user can read them"


def test_ask_degrades_when_the_llm_call_fails():
    fid = _store_finding()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.rag_endpoints._call_llm", return_value=(None, "APIError: upstream failure")
        ):
            r = client.post(
                f"/findings/{fid}/ask", json={"question": "how reliable is this measurement?"}
            )
    assert r.status_code == 200
    d = r.json()
    assert d["used_llm"] is False and d["chunks"]
    # The reason must reach the client — a bare "generation failed" is
    # undiagnosable, which is exactly the bug this signature change fixes.
    assert d["llm_error"] and "upstream failure" in d["llm_error"]


def test_ask_always_carries_the_disclaimer():
    fid = _store_finding()
    for question in ["how reliable is this measurement?", "capital of France?"]:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch("app.api.rag_endpoints._call_llm", return_value=("ok", None)):
                r = client.post(f"/findings/{fid}/ask", json={"question": question})
        assert "not cleared for clinical diagnosis" in r.json()["disclaimer"]


def test_system_prompt_forbids_diagnosis_and_ungrounded_answers():
    """These constraints are the product requirement, not stylistic advice."""
    from app.api.rag_endpoints import _SYSTEM_PROMPT

    lowered = _SYSTEM_PROMPT.lower()
    assert "only" in lowered and "reference excerpts" in lowered
    assert "diagnosis" in lowered
    assert "cite" in lowered


def test_llm_call_uses_the_same_model_as_the_report_narratives():
    """Drift between the two Claude call sites would be silent otherwise."""
    from app.api import rag_endpoints

    fake = MagicMock()
    fake.messages.create.return_value.content = [MagicMock(text="answer")]
    with patch("anthropic.Anthropic", return_value=fake):
        rag_endpoints._call_llm("sk-test", "prompt")
    assert fake.messages.create.call_args.kwargs["model"] == "claude-haiku-4-5-20251001"


# ── /knowledge/status ────────────────────────────────────────────────────────


def test_knowledge_status_reports_the_index():
    d = client.get("/knowledge/status").json()
    assert d["ready"] is True
    assert d["chunk_count"] >= 15
    assert "project_metrics.md" in d["files"]
    assert isinstance(d["provisional_chunks"], list)


# ── LLM failure diagnostics ──────────────────────────────────────────────────
#
# "Answer generation failed" with no reason is unactionable — the cause sat in
# the server log where the person clicking the button cannot see it. These
# assert the reason reaches the client, and that credentials never do.


def test_call_llm_returns_the_error_instead_of_swallowing_it():
    from app.api import rag_endpoints

    with patch("anthropic.Anthropic") as A:
        A.return_value.messages.create.side_effect = Exception("boom (401)")
        answer, error = rag_endpoints._call_llm("sk-test", "prompt")
    assert answer is None
    assert error and "boom" in error


def test_sanitize_error_redacts_anything_key_shaped():
    from app.api.rag_endpoints import _sanitize_error

    out = _sanitize_error(Exception("bad key sk-ant-SECRETVALUE123456 rejected"))
    assert "SECRETVALUE" not in out
    assert "sk-***" in out


def test_sanitize_error_caps_length():
    from app.api.rag_endpoints import _sanitize_error

    assert len(_sanitize_error(Exception("x" * 5000))) <= 600


def test_ask_surfaces_the_failure_reason_to_the_client():
    fid = _store_finding()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.rag_endpoints._call_llm",
            return_value=(None, "AuthenticationError: invalid x-api-key"),
        ):
            r = client.post(
                f"/findings/{fid}/ask", json={"question": "how reliable is this measurement?"}
            )
    d = r.json()
    assert d["used_llm"] is False
    assert d["llm_error"] == "AuthenticationError: invalid x-api-key"
    assert "invalid x-api-key" in d["answer"], "reason must be visible in the answer text"
    assert d["chunks"], "excerpts are still returned so the feature stays useful"


def test_ask_has_no_llm_error_on_success():
    fid = _store_finding()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("app.api.rag_endpoints._call_llm", return_value=("An answer.", None)):
            r = client.post(
                f"/findings/{fid}/ask", json={"question": "how reliable is this measurement?"}
            )
    d = r.json()
    assert d["used_llm"] is True and d["llm_error"] is None


def test_llm_status_reports_a_missing_key():
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        d = client.get("/llm/status").json()
    assert d["key_present"] is False and d["ok"] is False
    assert "ANTHROPIC_API_KEY" in d["hint"]


def test_llm_status_reports_success():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("anthropic.Anthropic") as A:
            A.return_value.messages.create.return_value = MagicMock()
            d = client.get("/llm/status").json()
    assert d["ok"] is True and d["error"] is None
    assert d["latency_ms"] is not None


@pytest.mark.parametrize(
    "raised,expect_in_hint",
    [
        ("AuthenticationError: invalid x-api-key 401", "not valid"),
        ("NotFoundError: 404 model does not exist", "model access"),
        ("BillingError: insufficient credit balance", "credit"),
        ("APIConnectionError: could not resolve host", "outbound network"),
    ],
)
def test_llm_status_maps_common_failures_to_a_remedy(raised, expect_in_hint):
    # probe_network is stubbed out: this covers the text-matching fallback, and
    # a live probe would both reach the real network and mask _diagnose().
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("app.api.rag_endpoints.probe_network", return_value=None):
            with patch("anthropic.Anthropic") as A:
                A.return_value.messages.create.side_effect = Exception(raised)
                d = client.get("/llm/status").json()
    assert d["ok"] is False
    assert d["hint"] and expect_in_hint in d["hint"]


def test_llm_status_never_leaks_the_key():
    secret = "sk-ant-DONOTLEAK0123456789"
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": secret}):
        with patch("app.api.rag_endpoints.probe_network", return_value=None):
            with patch("anthropic.Anthropic") as A:
                A.return_value.messages.create.side_effect = Exception(f"rejected key={secret}")
                d = client.get("/llm/status").json()
    assert "DONOTLEAK" not in str(d)


def test_rag_and_report_narratives_stay_on_the_same_model():
    """They share a key and model; drift would make diagnostics misleading."""
    import re as _re

    from app.api.rag_endpoints import LLM_MODEL

    report_src = (Path(__file__).resolve().parent.parent / "app" / "report.py").read_text()
    models = set(_re.findall(r'model="(claude-[^"]+)"', report_src))
    assert models == {LLM_MODEL}, f"report.py uses {models}, rag uses {LLM_MODEL}"


# ── network diagnostics ──────────────────────────────────────────────────────
#
# The live Space reported `APIConnectionError: Connection error.` with no
# further detail, which is unactionable. These cover the two fixes: reading the
# chained cause, and probing each network layer separately.


def test_sanitize_error_reports_the_chained_cause_not_just_the_wrapper():
    """APIConnectionError stringifies to "Connection error." and nothing else."""

    class APIConnectionError(Exception):
        pass

    try:
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError as root:
            raise APIConnectionError("Connection error.") from root
    except APIConnectionError as exc:
        msg = rag_endpoints._sanitize_error(exc)

    assert "Connection error." in msg
    assert "ConnectionRefusedError" in msg, "the actual reason must survive"
    assert "Connection refused" in msg


def test_sanitize_error_redacts_keys_anywhere_in_the_chain():
    try:
        try:
            raise ValueError("auth failed for sk-ant-api03-SECRETVALUE123")
        except ValueError as root:
            raise RuntimeError("wrapped") from root
    except RuntimeError as exc:
        msg = rag_endpoints._sanitize_error(exc)

    assert "SECRETVALUE123" not in msg
    assert "sk-***" in msg


def test_sanitize_error_survives_a_self_referential_cause_chain():
    """A cycle must not hang the diagnostic."""
    a = ValueError("a")
    b = ValueError("b")
    a.__cause__ = b
    b.__cause__ = a
    msg = rag_endpoints._sanitize_error(a)
    assert "a" in msg and "b" in msg


def test_probe_reports_dns_as_the_failed_layer_when_resolution_fails():
    with patch("socket.getaddrinfo", side_effect=OSError("Name or service not known")):
        probe = rag_endpoints.probe_network("api.anthropic.com")
    assert probe.dns_ok is False
    assert probe.failed_layer == "dns"
    assert probe.tcp_ok is None, "layers below a failure are not attempted"


def test_probe_reports_tcp_as_the_failed_layer_when_the_socket_is_refused():
    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("1.2.3.4", 443))]):
        with patch("socket.create_connection", side_effect=ConnectionRefusedError("refused")):
            probe = rag_endpoints.probe_network()
    assert probe.dns_ok is True
    assert probe.resolved_ips == ["1.2.3.4"]
    assert probe.tcp_ok is False
    assert probe.failed_layer == "tcp"


def test_probe_redacts_credentials_embedded_in_a_proxy_url():
    env = {"HTTPS_PROXY": "http://user:hunter2@proxy.internal:8080"}
    with patch.dict(os.environ, env):
        with patch("socket.getaddrinfo", side_effect=OSError("boom")):
            probe = rag_endpoints.probe_network()
    assert "hunter2" not in probe.proxy_env["HTTPS_PROXY"]
    assert "***" in probe.proxy_env["HTTPS_PROXY"]


def test_a_configured_proxy_is_named_as_the_prime_suspect():
    probe = rag_endpoints.NetworkProbe(
        host="api.anthropic.com",
        dns_ok=True,
        tcp_ok=False,
        failed_layer="tcp",
        proxy_env={"HTTPS_PROXY": "http://proxy.internal:8080"},
    )
    hint = rag_endpoints._diagnose_network(probe)
    assert "proxy" in hint.lower() and "HTTPS_PROXY" in hint


def test_working_raw_https_points_at_the_sdk_rather_than_the_network():
    probe = rag_endpoints.NetworkProbe(
        host="api.anthropic.com", dns_ok=True, tcp_ok=True, tls_ok=True, https_get_ok=True
    )
    hint = rag_endpoints._diagnose_network(probe)
    assert "SDK" in hint


def test_llm_status_probes_the_network_on_a_connection_error():
    class APIConnectionError(Exception):
        pass

    fake_probe = rag_endpoints.NetworkProbe(
        host="api.anthropic.com", dns_ok=True, tcp_ok=False, failed_layer="tcp"
    )
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("anthropic.Anthropic", side_effect=APIConnectionError("Connection error.")):
            with patch("app.api.rag_endpoints.probe_network", return_value=fake_probe) as probe:
                r = client.get("/llm/status")
    probe.assert_called_once()
    d = r.json()
    assert d["ok"] is False
    assert d["network"]["failed_layer"] == "tcp"
    assert "blocked at the network level" in d["hint"]


def test_llm_status_skips_the_network_probe_for_an_auth_failure():
    """An invalid key says nothing about connectivity — probing would mislead."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("anthropic.Anthropic", side_effect=Exception("AuthenticationError: 401")):
            with patch("app.api.rag_endpoints.probe_network") as probe:
                r = client.get("/llm/status")
    probe.assert_not_called()
    assert r.json()["network"] is None
