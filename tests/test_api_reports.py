"""Batch 6: Reports API tests — POST/GET reports, PDF, sign-off, audit.

Spec items covered:
  6.1  POST /studies/{id}/reports — calls Claude Haiku, writes Report row
       (LLM is mocked here; we assert the row was persisted with the narrative).
  6.2  GET /studies/{id}/reports + GET /reports/{id} — retrieval test.
  6.3  PDF generation server-side via reportlab — bytes valid, opens cleanly.
  6.7  Audit log: who signed, when, IP/user-agent → AuditLog table.

Spec 6.4–6.6 (frontend tabs / sign-off dialog / watermark visibility) are
tested in the webapp repo; here we only verify the watermark transition
indirectly via the X-Report-Signed response header on GET /reports/{id}/pdf.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """Each test gets its own SQLite file — keeps tests order-independent and
    leaves no state behind in the project root."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    monkeypatch.setenv("REPORTS_DB_PATH", path)
    # Reload the module that snapshots DB_PATH at import time.
    from app.api import reports_db

    reports_db.DB_PATH = path
    reports_db.init_db(path)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def client():
    from app.api.main import app

    return TestClient(app)


def _payload(**overrides):
    base = {
        "patient_name": "Test Patient",
        "study_date": "2026-05-04",
        "model": "phase0",
        "pixel_spacing_mm": 0.07,
        "hc_mm": 220.5,
        "ga_str": "24w 0d",
        "ga_weeks": 24.0,
        "trimester": "Mid (20–30w)",
        "reliability": 0.92,
        "confidence_label": "HIGH",
        "elapsed_ms": 412.0,
        "report_mode": "template",
    }
    base.update(overrides)
    return base


def _mock_llm():
    """Patch the underlying Anthropic call so tests don't hit the network.

    _llm_static_narrative now returns (p1, p2, p3, impression) — 4 calls.
    Supply plenty of values for any number of paragraphs.
    """
    return patch(
        "app.report._call_llm",
        side_effect=[
            "LLM-generated biometric paragraph with HC and GA context.",
            "LLM-generated activation map paragraph mentioning calvarium.",
            None,  # p3 (only for pruned models — None triggers rule-based fallback)
            "LLM-generated impression for the referring physician.",
        ]
        * 8,  # enough for any call count
    )


# ── 6.1 + 6.2 ─────────────────────────────────────────────────────────────────


def test_create_report_persists_row_and_returns_narrative(client, monkeypatch):
    """LLM mode requires report_mode='llm' + API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with _mock_llm():
        r = client.post("/studies/study-abc/reports", json=_payload(report_mode="llm"))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["study_id"] == "study-abc"
    assert body["patient_name"] == "Test Patient"
    assert body["hc_mm"] == 220.5
    assert body["used_llm"] is True
    assert body["narrative_p1"]
    assert "biometric" in body["narrative_p1"].lower()
    assert body["is_signed"] is False
    assert body["id"].startswith("rep_")
    # New fields: accession number and impression
    assert body["accession_number"] and body["accession_number"].startswith("FHC-")
    assert body["report_mode"] == "llm"


def test_create_report_falls_back_to_template_without_api_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/studies/study-abc/reports", json=_payload())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["used_llm"] is False
    assert body["narrative_p1"]  # rule-based fallback still populates a paragraph
    assert body["narrative_impression"]  # impression also rule-based


def test_create_report_template_mode_ignores_api_key(client, monkeypatch):
    """report_mode='template' must NOT use the LLM even when API key is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with _mock_llm():
        r = client.post("/studies/study-abc/reports", json=_payload(report_mode="template"))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["used_llm"] is False  # template mode never calls LLM
    assert body["report_mode"] == "template"


def test_create_report_accession_number_format(client, monkeypatch):
    """Accession numbers must match FHC-YYYYMMDD-HHMMSS format."""
    import re

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/studies/s/reports", json=_payload())
    assert r.status_code == 201, r.text
    accession = r.json()["accession_number"]
    assert accession is not None
    assert re.match(r"FHC-\d{8}-\d{6}", accession), f"Bad accession format: {accession}"


def test_create_report_clinical_fields_persisted(client, monkeypatch):
    """Referring physician, LMP, clinical indication etc. must survive the round-trip."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    payload = _payload(
        referring_physician="Dr. Jane Smith",
        lmp="2025-11-01",
        clinical_indication="Routine 24-week dating scan",
        us_approach="transabdominal",
        image_quality="optimal",
        patient_id="MRN-12345",
    )
    r = client.post("/studies/s/reports", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["referring_physician"] == "Dr. Jane Smith"
    assert body["lmp"] == "2025-11-01"
    assert body["clinical_indication"] == "Routine 24-week dating scan"
    assert body["us_approach"] == "transabdominal"
    assert body["image_quality"] == "optimal"
    assert body["patient_id"] == "MRN-12345"


def test_create_report_requires_hc_or_finding(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = _payload()
    payload.pop("hc_mm")
    with _mock_llm():
        r = client.post("/studies/study-abc/reports", json=payload)
    assert r.status_code == 400
    assert "hc_mm" in r.text


def test_list_and_get_reports_for_study(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with _mock_llm():
        r1 = client.post("/studies/study-1/reports", json=_payload(patient_name="A"))
        r2 = client.post("/studies/study-1/reports", json=_payload(patient_name="B"))
        client.post("/studies/study-2/reports", json=_payload(patient_name="C"))
    assert r1.status_code == 201 and r2.status_code == 201

    list_resp = client.get("/studies/study-1/reports")
    assert list_resp.status_code == 200
    rows = list_resp.json()
    assert len(rows) == 2
    names = {row["patient_name"] for row in rows}
    assert names == {"A", "B"}

    rid = r1.json()["id"]
    one = client.get(f"/reports/{rid}")
    assert one.status_code == 200
    assert one.json()["id"] == rid


# ── 6.3 ───────────────────────────────────────────────────────────────────────


def test_get_report_pdf_returns_valid_pdf_bytes(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with _mock_llm():
        r = client.post("/studies/s/reports", json=_payload())
    rid = r.json()["id"]

    pdf_resp = client.get(f"/reports/{rid}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    body = pdf_resp.content
    assert body[:4] == b"%PDF"
    assert b"%%EOF" in body[-1024:]
    assert pdf_resp.headers["X-Report-Signed"] == "0"


def test_get_pdf_404_on_missing_report(client):
    r = client.get("/reports/rep_does_not_exist/pdf")
    assert r.status_code == 404


# ── 6.5 + 6.6 + 6.7 ───────────────────────────────────────────────────────────


def test_sign_report_mutates_is_signed_and_writes_audit(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with _mock_llm():
        r = client.post("/studies/s/reports", json=_payload())
    rid = r.json()["id"]

    sign_resp = client.post(
        f"/reports/{rid}/sign",
        json={"signed_by": "Dr. Eleanor Vance", "signoff_note": "Concur with measurement."},
        headers={"User-Agent": "FetalScan-Test/1.0", "X-Forwarded-For": "203.0.113.7"},
    )
    assert sign_resp.status_code == 200, sign_resp.text
    signed = sign_resp.json()
    assert signed["is_signed"] is True
    assert signed["signed_by"] == "Dr. Eleanor Vance"
    assert signed["signoff_note"] == "Concur with measurement."
    assert signed["signed_at"]

    # Audit row exists with the signature metadata
    audit_resp = client.get(f"/reports/{rid}/audit")
    assert audit_resp.status_code == 200
    rows = audit_resp.json()
    actions = [row["action"] for row in rows]
    assert actions == ["created", "signed"]
    sign_row = rows[-1]
    assert sign_row["actor"] == "Dr. Eleanor Vance"
    assert sign_row["ip"] == "203.0.113.7"
    assert "FetalScan-Test" in (sign_row["user_agent"] or "")


def test_sign_report_rejects_double_sign(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with _mock_llm():
        r = client.post("/studies/s/reports", json=_payload())
    rid = r.json()["id"]
    client.post(f"/reports/{rid}/sign", json={"signed_by": "Dr. A"})
    again = client.post(f"/reports/{rid}/sign", json={"signed_by": "Dr. B"})
    assert again.status_code == 409


def test_signed_report_pdf_drops_watermark_header_flag(client, monkeypatch):
    """6.6: signed reports get watermark removed; draft keeps it.

    We don't pixel-diff the PDFs; we assert the X-Report-Signed header flips
    to "1" after signing, which is the same flag the renderer uses to skip
    the DRAFT watermark callback.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with _mock_llm():
        r = client.post("/studies/s/reports", json=_payload())
    rid = r.json()["id"]

    draft_pdf = client.get(f"/reports/{rid}/pdf")
    assert draft_pdf.headers["X-Report-Signed"] == "0"

    client.post(f"/reports/{rid}/sign", json={"signed_by": "Dr. A"})

    signed_pdf = client.get(f"/reports/{rid}/pdf")
    assert signed_pdf.headers["X-Report-Signed"] == "1"
    # The signed PDF embeds the clinician name in the sign-off block; the
    # draft does not. Bytes should differ.
    assert draft_pdf.content != signed_pdf.content


def test_audit_404_on_missing_report(client):
    r = client.get("/reports/rep_nope/audit")
    assert r.status_code == 404
