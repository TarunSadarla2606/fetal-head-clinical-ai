"""Batch 7 tests — FHIR R4 + DICOM SR exports.

Spec items covered:
  7.1  GET /reports/{id}/fhir → FHIR R4 Bundle (DiagnosticReport + Patient +
       one Observation per measurement). Status flips preliminary → final
       on sign-off.
  7.2  GET /reports/{id}/dicom → Comprehensive SR Storage (.dcm) with the
       'DICM' magic at offset 128, valid SOP class, LOINC-coded numeric
       content items, sign-off populates VerifyingObserverSequence and
       flips PreliminaryFlag → FINAL.
"""

from __future__ import annotations

import io
import tempfile

import pydicom
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    monkeypatch.setenv("REPORTS_DB_PATH", path)
    from app.api import reports_db

    reports_db.DB_PATH = path
    reports_db.init_db()
    yield


@pytest.fixture
def client():
    from app.api.main import app

    return TestClient(app)


def _create_report():
    """Helper: write one report row directly via reports_db (bypasses the
    LLM-mocked POST /reports flow used in test_api_reports.py)."""
    from app.api import reports_db

    return reports_db.create_report(
        study_id="study-1",
        finding_id=None,
        patient_name="Jane Doe",
        study_date="2026-05-05",
        model="phase4a",
        hc_mm=245.3,
        ga_str="21w 0d",
        ga_weeks=21.0,
        trimester="Second trimester (14–28w)",
        reliability=0.92,
        confidence_label="HIGH CONFIDENCE",
        pixel_spacing_mm=0.154,
        elapsed_ms=460.0,
        narrative_p1="",
        narrative_p2="",
        narrative_p3=None,
        narrative_impression="Normal fetal anatomy at 21 weeks.",
        used_llm=False,
        referring_physician="Dr. Sarah Chen",
        patient_id="MRN-001",
        patient_dob="1996-04-15",
        bpd_mm=58.4,
        fetal_presentation="cephalic",
    )


# ── 7.1  FHIR ────────────────────────────────────────────────────────────────


def test_fhir_export_returns_collection_bundle(client):
    r = _create_report()
    resp = client.get(f"/reports/{r.id}/fhir")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/fhir+json")

    bundle = resp.json()
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"

    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Patient" in types
    assert "DiagnosticReport" in types
    assert types.count("Observation") == 3  # HC + BPD + GA


def test_fhir_observations_use_correct_loinc_codes(client):
    r = _create_report()
    bundle = client.get(f"/reports/{r.id}/fhir").json()
    obs = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"]
    loinc_codes = {o["code"]["coding"][0]["code"] for o in obs}
    assert loinc_codes == {"11779-6", "11820-2", "18185-9"}


def test_fhir_status_flips_to_final_on_sign(client):
    r = _create_report()
    b_pre = client.get(f"/reports/{r.id}/fhir").json()
    dr_pre = next(
        e["resource"] for e in b_pre["entry"] if e["resource"]["resourceType"] == "DiagnosticReport"
    )
    assert dr_pre["status"] == "preliminary"

    client.post(f"/reports/{r.id}/sign", json={"signed_by": "Dr. Test"})
    b_post = client.get(f"/reports/{r.id}/fhir").json()
    dr_post = next(
        e["resource"]
        for e in b_post["entry"]
        if e["resource"]["resourceType"] == "DiagnosticReport"
    )
    assert dr_post["status"] == "final"


def test_fhir_404_when_report_missing(client):
    resp = client.get("/reports/rep_does_not_exist/fhir")
    assert resp.status_code == 404


# ── 7.2  DICOM SR ────────────────────────────────────────────────────────────


def test_dicom_sr_export_is_valid_dicom(client):
    r = _create_report()
    resp = client.get(f"/reports/{r.id}/dicom")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/dicom"
    # 128-byte preamble then 'DICM' magic
    assert resp.content[128:132] == b"DICM"


def test_dicom_sr_carries_correct_measurements(client):
    r = _create_report()
    resp = client.get(f"/reports/{r.id}/dicom")
    ds = pydicom.dcmread(io.BytesIO(resp.content))
    assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.33"  # Comprehensive SR
    assert ds.Modality == "SR"
    assert ds.PatientID == "MRN-001"

    nums = {
        item.ConceptNameCodeSequence[0].CodeValue: float(item.MeasuredValueSequence[0].NumericValue)
        for item in ds.ContentSequence
        if item.ValueType == "NUM"
    }
    assert nums == {"11779-6": 245.3, "11820-2": 58.4, "18185-9": 21.0}


def test_dicom_sr_sign_flips_completion_and_adds_verifier(client):
    r = _create_report()

    ds_pre = pydicom.dcmread(io.BytesIO(client.get(f"/reports/{r.id}/dicom").content))
    assert ds_pre.CompletionFlag == "PARTIAL"
    assert ds_pre.VerificationFlag == "UNVERIFIED"
    assert ds_pre.PreliminaryFlag == "PRELIMINARY"
    assert "VerifyingObserverSequence" not in ds_pre

    client.post(f"/reports/{r.id}/sign", json={"signed_by": "Dr. Tester"})

    ds_post = pydicom.dcmread(io.BytesIO(client.get(f"/reports/{r.id}/dicom").content))
    assert ds_post.CompletionFlag == "COMPLETE"
    assert ds_post.VerificationFlag == "VERIFIED"
    assert ds_post.PreliminaryFlag == "FINAL"
    assert ds_post.VerifyingObserverSequence[0].VerifyingObserverName == "Dr. Tester"


def test_dicom_sr_404_when_report_missing(client):
    resp = client.get("/reports/rep_does_not_exist/dicom")
    assert resp.status_code == 404


# ── 7.5  Mock C-STORE ────────────────────────────────────────────────────────


def test_cstore_round_trip_logs_received_event(client):
    """Generate an SR, upload it back via /cstore, confirm the log row
    captures the SOP UIDs + patient identifiers."""
    r = _create_report()
    dcm_bytes = client.get(f"/reports/{r.id}/dicom").content

    up = client.post("/cstore", files={"file": ("test.dcm", dcm_bytes, "application/dicom")})
    assert up.status_code == 201
    body = up.json()
    assert body["status"] == "received"
    assert body["sop_class_uid"] == "1.2.840.10008.5.1.4.1.1.88.33"
    assert body["sop_instance_uid"]
    assert body["patient_id"] == "MRN-001"
    assert body["file_size"] == len(dcm_bytes)

    log = client.get("/cstore/log")
    assert log.status_code == 200
    rows = log.json()
    assert len(rows) == 1
    assert rows[0]["sop_instance_uid"] == body["sop_instance_uid"]


def test_cstore_rejects_non_dicom(client):
    resp = client.post("/cstore", files={"file": ("oops.txt", b"plain text", "text/plain")})
    assert resp.status_code == 400


def test_cstore_log_limit_validation(client):
    assert client.get("/cstore/log?limit=0").status_code == 400
    assert client.get("/cstore/log?limit=501").status_code == 400
    assert client.get("/cstore/log?limit=10").status_code == 200


# ── 7.6  Longitudinal patient reports ────────────────────────────────────────


def test_list_reports_for_patient_groups_across_studies(client):
    """Two studies, same patient_id → both reports come back, sorted by study_date."""
    from app.api import reports_db

    reports_db.create_report(
        study_id="visit-1",
        finding_id=None,
        patient_name="Long Term",
        study_date="2026-01-15",
        model="phase4a",
        hc_mm=180.0,
        ga_str="17w 0d",
        ga_weeks=17.0,
        trimester="Second trimester (14–28w)",
        reliability=0.9,
        confidence_label="HIGH",
        pixel_spacing_mm=0.154,
        elapsed_ms=400.0,
        narrative_p1="",
        narrative_p2="",
        narrative_p3=None,
        narrative_impression=None,
        used_llm=False,
        patient_id="MRN-LONG",
    )
    reports_db.create_report(
        study_id="visit-2",
        finding_id=None,
        patient_name="Long Term",
        study_date="2026-04-10",
        model="phase4a",
        hc_mm=255.0,
        ga_str="22w 0d",
        ga_weeks=22.0,
        trimester="Second trimester (14–28w)",
        reliability=0.9,
        confidence_label="HIGH",
        pixel_spacing_mm=0.154,
        elapsed_ms=400.0,
        narrative_p1="",
        narrative_p2="",
        narrative_p3=None,
        narrative_impression=None,
        used_llm=False,
        patient_id="MRN-LONG",
    )

    resp = client.get("/patients/MRN-LONG/reports")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    # Oldest first per the endpoint contract
    assert rows[0]["study_date"] == "2026-01-15"
    assert rows[1]["study_date"] == "2026-04-10"


def test_list_reports_for_patient_empty_for_unknown_id(client):
    resp = client.get("/patients/MRN-NOT-A-REAL-PATIENT/reports")
    assert resp.status_code == 200
    assert resp.json() == []


# ── 8.2  Demo seed ───────────────────────────────────────────────────────────


def test_demo_seed_inserts_ten_reports():
    """seed_demo_reports() inserts exactly ten reports the first time and
    is a no-op on a second call (idempotent)."""
    from app.api import demo_seed, reports_db

    n_first = demo_seed.seed_demo_reports()
    assert n_first == 10
    n_second = demo_seed.seed_demo_reports()
    assert n_second == 0

    # All ten study IDs are populated
    for i in range(1, 11):
        sid = f"demo-{i:03d}"
        rows = reports_db.list_reports_for_study(sid)
        assert len(rows) == 1


def test_demo_seed_includes_three_abnormal_cases():
    """The three abnormal cases (microcephaly / macrocephaly / IUGR) must
    end up well outside the population HC norm so the report flags fire."""
    from app.api import demo_seed, reports_db
    from app.inference import hadlock_ga

    demo_seed.seed_demo_reports()

    abnormal_ids = ["demo-002", "demo-005", "demo-008"]
    flagged = 0
    for sid in abnormal_ids:
        rep = reports_db.list_reports_for_study(sid)[0]
        assert rep.hc_mm is not None
        assert rep.ga_weeks is not None
        # Population mean HC for this GA via inverse Hadlock — same logic
        # the demo seed uses to position the abnormal points.
        ga_at_mean, _ = hadlock_ga(rep.hc_mm)
        delta_weeks = abs(ga_at_mean - rep.ga_weeks)
        # Abnormal cases should be >1.5 weeks off the mean curve. IUGR cases
        # (mildest of the three) sit at HC <10th percentile ≈ -1.3 SD, which
        # corresponds to ~2 weeks of GA delta — micro/macrocephaly are more.
        assert delta_weeks > 1.5, f"{sid} only {delta_weeks:.2f} wk off mean"
        flagged += 1
    assert flagged == 3
