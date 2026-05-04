"""Reports API — Batch 6.

Endpoints
---------
POST   /studies/{study_id}/reports        Generate a clinical report (calls Claude
                                          Haiku for narrative paragraphs, persists
                                          a Report row, audits as "created").
GET    /studies/{study_id}/reports        List all reports for a study.
GET    /reports/{report_id}               Fetch one report's JSON.
GET    /reports/{report_id}/pdf           Render and download the PDF.
                                          Unsigned reports get a DRAFT watermark.
POST   /reports/{report_id}/sign          Mark the report as signed off; appends
                                          signed_by + signoff_note + timestamps
                                          and writes an audit row capturing
                                          actor IP and user-agent.
GET    /reports/{report_id}/audit         List audit entries for the report.

Storage
-------
- Reports + audit log live in SQLite (see reports_db.py).
- Sign-off freezes the narrative; the PDF is re-rendered on each download from
  the stored fields, so the watermark transition (DRAFT → signed) is fully
  driven by is_signed without re-running the LLM.
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response

from app.report import generate_cine_report, generate_static_report

from . import findings_store, model_manager, reports_db, xai_endpoints
from .schemas import (
    AuditEntryResponse,
    CreateReportRequest,
    ReportResponse,
    SignReportRequest,
)

router = APIRouter(tags=["Reports"])


_MODEL_VARIANT_TO_REPORT_NAME = {
    "phase0": "Phase 0 — Static baseline",
    "phase4a": "Phase 4a — Compressed static",
    "phase2": "Phase 2 — Temporal baseline",
    "phase4b": "Phase 4b — Compressed temporal",
}

_TEMPORAL_VARIANTS = {"phase2", "phase4b"}


def _generate_narratives(
    *,
    model_variant: str,
    hc_mm: float,
    ga_str: str,
    ga_weeks: float,
    trimester: str,
    reliability: float,
    hc_std_mm: float,
    elapsed_ms: float,
    api_key: str | None,
    report_mode: str = "template",
) -> tuple[tuple[str, str, str | None, str | None], bool]:
    """Generate narrative paragraphs.

    Returns ((p1, p2, p3, impression), used_llm).
    LLM is used only when report_mode=='llm' AND api_key is present.
    Falls back to rule-based paragraphs otherwise.
    """
    from app.report import (
        _llm_cine_narrative,
        _llm_static_narrative,
        _rule_cine_p1,
        _rule_cine_p2,
        _rule_compression_note,
        _rule_impression,
        _rule_static_p1,
        _rule_static_p2,
    )

    model_name = _MODEL_VARIANT_TO_REPORT_NAME[model_variant]
    is_temporal = model_variant in _TEMPORAL_VARIANTS
    use_llm = report_mode == "llm" and bool(api_key)

    if use_llm:
        try:
            if is_temporal:
                p1, p2, p3, impression = _llm_cine_narrative(
                    hc_mm,
                    ga_str,
                    ga_weeks,
                    trimester,
                    reliability,
                    hc_std_mm,
                    16,
                    model_name,
                    elapsed_ms,
                    api_key,
                )
            else:
                p1, p2, p3, impression = _llm_static_narrative(
                    hc_mm,
                    ga_str,
                    ga_weeks,
                    trimester,
                    True,
                    model_name,
                    elapsed_ms,
                    api_key,
                )
            return (p1, p2, p3, impression), True
        except Exception:
            pass  # fall through to rule-based on any LLM failure

    if is_temporal:
        p1 = _rule_cine_p1(hc_mm, ga_str, ga_weeks, trimester, reliability, hc_std_mm)
        p2 = _rule_cine_p2(reliability, hc_std_mm, 16)
    else:
        p1 = _rule_static_p1(hc_mm, ga_str, ga_weeks, trimester)
        p2 = _rule_static_p2(True)
    p3 = _rule_compression_note(model_name, elapsed_ms)
    impression = _rule_impression(hc_mm, ga_str, ga_weeks, trimester)
    return (p1, p2, p3, impression), False


def _extract_images_from_store(
    finding_id: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Pull original, overlay, and GradCAM images from the findings_store.

    Returns (original_b64, overlay_b64, gradcam_b64). Any entry may be None
    if the finding is no longer cached or the model is unloaded.
    """
    if not finding_id:
        return None, None, None
    rec = findings_store.get(finding_id)
    if rec is None:
        return None, None, None

    # Overlay is pre-computed by /infer and stored in the findings dict
    overlay_b64 = rec.findings.get("overlay_b64")

    # Convert grayscale numpy array → PNG → base64
    original_b64: str | None = None
    try:
        img = rec.img_gray
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", img)
        original_b64 = base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass

    # Compute GradCAM using the loaded model
    gradcam_b64: str | None = None
    try:
        model = model_manager.get_model(rec.model_variant)
        if model is not None:
            overlay_arr = xai_endpoints.compute_gradcam(model, rec.img_gray)
            _, buf = cv2.imencode(".png", overlay_arr)
            gradcam_b64 = base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass

    return original_b64, overlay_b64, gradcam_b64


def _render_pdf(report: reports_db.Report) -> bytes:
    """Render the stored Report into a PDF, applying watermark / sign-off
    sections based on is_signed."""
    model_name = _MODEL_VARIANT_TO_REPORT_NAME.get(report.model, "Phase 0 — Static baseline")
    is_temporal = report.model in _TEMPORAL_VARIANTS

    result = {
        "hc_mm": report.hc_mm or 0.0,
        "ga_str": report.ga_str or "—",
        "ga_weeks": report.ga_weeks or 0.0,
        "trimester": report.trimester or "—",
        "reliability": report.reliability or 0.0,
        "hc_std_mm": 0.0,
        "confidence_label": report.confidence_label or "—",
        "elapsed_ms": report.elapsed_ms or 0.0,
        "mode": "cine_clip" if is_temporal else "phase0",
        "gradcam_ok": bool(report.gradcam_image_b64),
    }
    narrative = (
        report.narrative_p1 or "",
        report.narrative_p2 or "",
        report.narrative_p3,
        report.narrative_impression,
    )
    signed_meta = None
    if report.is_signed:
        signed_meta = {
            "signed_by": report.signed_by,
            "signed_at": report.signed_at,
            "signoff_note": report.signoff_note,
        }
    builder = generate_cine_report if is_temporal else generate_static_report
    return builder(
        result,
        api_key=None,
        use_llm=False,
        model_name=model_name,
        pixel_spacing=report.pixel_spacing_mm or 0.070,
        narrative=narrative,
        draft=not report.is_signed,
        signed_meta=signed_meta,
        report=report,
    )


def _coalesce_finding(req: CreateReportRequest) -> dict:
    """Merge fields pulled from findings_store with explicit body fields,
    preferring explicit body fields when both are present."""
    found = {}
    if req.finding_id:
        rec = findings_store.get(req.finding_id)
        if rec is not None:
            f = rec.findings
            found = {
                "hc_mm": f.get("hc_mm"),
                "ga_str": f.get("ga_str"),
                "ga_weeks": f.get("ga_weeks"),
                "trimester": f.get("trimester"),
                "reliability": f.get("reliability"),
                "hc_std_mm": f.get("hc_std_mm", 0.0),
                "confidence_label": f.get("confidence_label"),
                "elapsed_ms": f.get("elapsed_ms"),
            }
    explicit = {
        "hc_mm": req.hc_mm,
        "ga_str": req.ga_str,
        "ga_weeks": req.ga_weeks,
        "trimester": req.trimester,
        "reliability": req.reliability,
        "confidence_label": req.confidence_label,
        "elapsed_ms": req.elapsed_ms,
    }
    merged = {**found, **{k: v for k, v in explicit.items() if v is not None}}
    return merged


def _request_meta(request: Request) -> tuple[str | None, str | None]:
    """Extract caller IP + user-agent for the audit log. Falls back to
    X-Forwarded-For when present (HF Spaces / Vercel proxies)."""
    fwd = request.headers.get("x-forwarded-for")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else None)
    ua = request.headers.get("user-agent")
    return ip, ua


@router.post(
    "/studies/{study_id}/reports",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a clinical report (LLM-authored, persisted)",
)
def create_report_endpoint(
    study_id: str,
    body: CreateReportRequest,
    request: Request,
):
    if not study_id or len(study_id) > 200:
        raise HTTPException(400, "study_id must be 1-200 chars")

    fields = _coalesce_finding(body)
    if fields.get("hc_mm") is None:
        raise HTTPException(
            400,
            "hc_mm required — pass a finding_id from a /infer call or "
            "supply hc_mm/ga_str/etc. in the request body.",
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    (p1, p2, p3, impression), used_llm = _generate_narratives(
        model_variant=body.model,
        hc_mm=fields["hc_mm"] or 0.0,
        ga_str=fields.get("ga_str") or "—",
        ga_weeks=fields.get("ga_weeks") or 0.0,
        trimester=fields.get("trimester") or "—",
        reliability=fields.get("reliability") or 0.0,
        hc_std_mm=fields.get("hc_std_mm") or 0.0,
        elapsed_ms=fields.get("elapsed_ms") or 0.0,
        api_key=api_key,
        report_mode=body.report_mode,
    )

    original_b64, overlay_b64, gradcam_b64 = _extract_images_from_store(body.finding_id)

    report = reports_db.create_report(
        study_id=study_id,
        finding_id=body.finding_id,
        patient_name=body.patient_name,
        study_date=body.study_date,
        model=body.model,
        hc_mm=fields.get("hc_mm"),
        ga_str=fields.get("ga_str"),
        ga_weeks=fields.get("ga_weeks"),
        trimester=fields.get("trimester"),
        reliability=fields.get("reliability"),
        confidence_label=fields.get("confidence_label"),
        pixel_spacing_mm=body.pixel_spacing_mm,
        elapsed_ms=fields.get("elapsed_ms"),
        narrative_p1=p1,
        narrative_p2=p2,
        narrative_p3=p3,
        narrative_impression=impression,
        used_llm=used_llm,
        referring_physician=body.referring_physician,
        patient_id=body.patient_id,
        patient_dob=body.patient_dob,
        lmp=body.lmp,
        ordering_facility=body.ordering_facility,
        sonographer_name=body.sonographer_name,
        clinical_indication=body.clinical_indication,
        us_approach=body.us_approach,
        image_quality=body.image_quality,
        pixel_spacing_dicom_derived=body.pixel_spacing_dicom_derived,
        report_mode=body.report_mode,
        original_image_b64=original_b64,
        overlay_image_b64=overlay_b64,
        gradcam_image_b64=gradcam_b64,
        fetal_presentation=body.fetal_presentation,
        bpd_mm=body.bpd_mm,
    )
    ip, ua = _request_meta(request)
    reports_db.add_audit(
        report_id=report.id,
        action="created",
        actor=None,
        ip=ip,
        user_agent=ua,
        details=f"model={body.model}; mode={body.report_mode}; used_llm={used_llm}",
    )
    return ReportResponse(**report.to_dict())


@router.get(
    "/studies/{study_id}/reports",
    response_model=list[ReportResponse],
    summary="List reports for a study",
)
def list_reports_for_study_endpoint(study_id: str):
    rows = reports_db.list_reports_for_study(study_id)
    return [ReportResponse(**r.to_dict()) for r in rows]


@router.get(
    "/reports/{report_id}",
    response_model=ReportResponse,
    summary="Fetch one report's JSON",
)
def get_report_endpoint(report_id: str):
    report = reports_db.get_report(report_id)
    if report is None:
        raise HTTPException(404, "report not found")
    return ReportResponse(**report.to_dict())


@router.get(
    "/reports/{report_id}/pdf",
    summary="Download the report PDF (DRAFT watermark on unsigned reports)",
    responses={200: {"content": {"application/pdf": {}}}},
)
def get_report_pdf_endpoint(report_id: str):
    report = reports_db.get_report(report_id)
    if report is None:
        raise HTTPException(404, "report not found")
    pdf_bytes = _render_pdf(report)
    suffix = "signed" if report.is_signed else "draft"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="report-{report.id}-{suffix}.pdf"',
            "X-Report-Signed": "1" if report.is_signed else "0",
        },
    )


@router.post(
    "/reports/{report_id}/sign",
    response_model=ReportResponse,
    summary="Sign-off on the report (audit-logged)",
)
def sign_report_endpoint(
    report_id: str,
    body: SignReportRequest,
    request: Request,
):
    existing = reports_db.get_report(report_id)
    if existing is None:
        raise HTTPException(404, "report not found")
    if existing.is_signed:
        raise HTTPException(409, "report already signed")

    updated = reports_db.sign_report(
        report_id,
        body.signed_by,
        body.signoff_note,
    )
    if updated is None:
        # Race: another sign happened between get_report and sign_report.
        raise HTTPException(409, "report already signed")

    ip, ua = _request_meta(request)
    reports_db.add_audit(
        report_id=report_id,
        action="signed",
        actor=body.signed_by,
        ip=ip,
        user_agent=ua,
        details=body.signoff_note or None,
    )
    return ReportResponse(**updated.to_dict())


@router.get(
    "/reports/{report_id}/audit",
    response_model=list[AuditEntryResponse],
    summary="Audit log entries for the report",
)
def get_report_audit_endpoint(report_id: str):
    if reports_db.get_report(report_id) is None:
        raise HTTPException(404, "report not found")
    rows = reports_db.list_audit_for_report(report_id)
    return [AuditEntryResponse(**a.to_dict()) for a in rows]
