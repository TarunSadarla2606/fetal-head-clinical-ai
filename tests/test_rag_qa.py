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

from app.api import findings_store, knowledge_base
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
            return_value="Reliability is inter-frame agreement "
            "[project_metrics.md § Temporal reliability score — definition].",
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
        with patch("app.api.rag_endpoints._call_llm", return_value=f"Answer text [{used}]."):
            r = client.post(
                f"/findings/{fid}/ask", json={"question": "how reliable is this measurement?"}
            )
    d = r.json()
    assert d["citations"] == [used]
    assert len(d["chunks"]) > 1, "fixture should retrieve more than the one cited"


def test_ask_passes_the_measurement_numbers_into_the_prompt():
    fid = _store_finding(hc_mm=245.3, ga_str="21w 0d")
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("app.api.rag_endpoints._call_llm", return_value="ok") as llm:
            client.post(f"/findings/{fid}/ask", json={"question": "how reliable is this?"})
    prompt = llm.call_args[0][1]
    assert "245.3 mm" in prompt
    assert "21w 0d" in prompt
    assert "REFERENCE EXCERPTS" in prompt and "QUESTION" in prompt


def test_ask_marks_provisional_when_any_chunk_is_unverified():
    fid = _store_finding()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("app.api.rag_endpoints._call_llm", return_value="ok"):
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
        with patch("app.api.rag_endpoints._call_llm", return_value=None):
            r = client.post(
                f"/findings/{fid}/ask", json={"question": "how reliable is this measurement?"}
            )
    assert r.status_code == 200
    d = r.json()
    assert d["used_llm"] is False and d["chunks"]


def test_ask_always_carries_the_disclaimer():
    fid = _store_finding()
    for question in ["how reliable is this measurement?", "capital of France?"]:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch("app.api.rag_endpoints._call_llm", return_value="ok"):
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
