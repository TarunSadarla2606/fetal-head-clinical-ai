"""Pydantic request/response schemas for the FetalScan AI inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelVariant = Literal["phase0", "phase4a", "phase2", "phase4b"]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_available: list[str]
    device: str


class ValidationResult(BaseModel):
    valid: bool
    warnings: list[str]
    checks: dict[str, bool]


class InferResponse(BaseModel):
    finding_id: str = Field(
        description="UUID for this inference; pass to /findings/{id}/* for XAI overlays."
    )
    hc_mm: float | None = None
    ga_str: str | None = None
    ga_weeks: float | None = None
    trimester: str = "Unknown"
    reliability: float = 0.0
    hc_std_mm: float = 0.0
    confidence_label: str
    confidence_color: str
    elapsed_ms: float
    mode: str
    validation: ValidationResult
    ood_flag: bool
    ood_reasons: list[str]
    mask_b64: str = Field(description="Base64-encoded PNG of the segmentation mask")
    overlay_b64: str = Field(description="Base64-encoded PNG of the HC overlay image")


class OodReason(BaseModel):
    category: str
    detail: str


class OodResponse(BaseModel):
    ood_flag: bool
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[OodReason]
    stats: dict[str, float]


# ── Reports (Batch 6) ─────────────────────────────────────────────────────────


class CreateReportRequest(BaseModel):
    """Body for POST /studies/{study_id}/reports.

    Either `finding_id` (preferred — pulls findings from the in-memory
    findings_store written by /infer) OR an explicit set of biometric values
    must be provided. The endpoint hydrates from the store first and lets
    explicit fields override.
    """

    finding_id: str | None = None
    patient_name: str
    study_date: str
    model: ModelVariant
    pixel_spacing_mm: float | None = 0.070
    # explicit overrides — used when no finding_id (synthetic / external data)
    hc_mm: float | None = None
    ga_str: str | None = None
    ga_weeks: float | None = None
    trimester: str | None = None
    reliability: float | None = None
    confidence_label: str | None = None
    elapsed_ms: float | None = None
    # ACR/AIUM/ESR-compliant clinical fields (all optional for backwards compat)
    referring_physician: str | None = None
    patient_id: str | None = None
    patient_dob: str | None = None
    lmp: str | None = None  # ISO date — used for EDD cross-check / GA discordance flag
    ordering_facility: str | None = None
    sonographer_name: str | None = None
    clinical_indication: str | None = None
    us_approach: Literal["transabdominal", "transvaginal"] | None = None
    image_quality: Literal["optimal", "suboptimal", "limited"] | None = None
    pixel_spacing_dicom_derived: bool = False
    pixel_spacing_source: Literal["DICOM", "CSV", "USER"] | None = None
    report_mode: Literal["template", "llm"] = "template"
    fetal_presentation: Literal["cephalic", "breech", "transverse", "not_assessed"] | None = (
        "not_assessed"
    )
    bpd_mm: float | None = None  # optional secondary biometric parameter
    prior_biometry: str | None = (
        None  # free-text prior measurement summary, e.g. "HC 198 mm @ 2024-12-01"
    )


class SignReportRequest(BaseModel):
    signed_by: str = Field(min_length=1, max_length=200)
    signoff_note: str | None = Field(default=None, max_length=2000)


class ReportResponse(BaseModel):
    id: str
    study_id: str
    finding_id: str | None
    patient_name: str
    study_date: str
    model: str
    hc_mm: float | None
    ga_str: str | None
    ga_weeks: float | None
    trimester: str | None
    reliability: float | None
    confidence_label: str | None
    pixel_spacing_mm: float | None
    elapsed_ms: float | None
    narrative_p1: str | None
    narrative_p2: str | None
    narrative_p3: str | None
    narrative_impression: str | None = None
    used_llm: bool
    is_signed: bool
    signed_by: str | None
    signed_at: str | None
    signoff_note: str | None
    created_at: str
    # Extended clinical fields
    referring_physician: str | None = None
    patient_id: str | None = None
    patient_dob: str | None = None
    lmp: str | None = None
    ordering_facility: str | None = None
    sonographer_name: str | None = None
    clinical_indication: str | None = None
    us_approach: str | None = None
    image_quality: str | None = None
    pixel_spacing_dicom_derived: bool = False
    pixel_spacing_source: Literal["DICOM", "CSV", "USER"] | None = None
    report_mode: str = "template"
    accession_number: str | None = None
    original_image_b64: str | None = None
    overlay_image_b64: str | None = None
    gradcam_image_b64: str | None = None
    fetal_presentation: str | None = None
    bpd_mm: float | None = None
    prior_biometry: str | None = None


class AuditEntryResponse(BaseModel):
    id: str
    report_id: str
    action: str
    actor: str | None
    ip: str | None
    user_agent: str | None
    details: str | None
    timestamp: str
