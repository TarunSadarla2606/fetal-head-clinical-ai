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
    # Batch 8.3 — composite image-quality score + label + raw blur metric
    quality_score: float = 0.0
    quality_label: Literal["poor", "suboptimal", "good", "excellent"] = "good"
    blur_score: float = 0.0


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
    cine_overlay_gif: str | None = Field(
        default=None,
        description=(
            "Animated GIF as a `data:image/gif;base64,...` URI showing the predicted "
            "segmentation contour on every frame of the cine-loop, each frame labelled "
            "with its index and its own HC, and the key frame marked. Populated only when "
            "`mode == 'cine_clip'` (temporal variants phase2 / phase4b); null otherwise, "
            "and null if the animation could not be built."
        ),
    )
    cine_loop_gif: str | None = Field(
        default=None,
        description=(
            "Animated GIF `data:` URI of the raw synthesised cine-loop with no prediction "
            "drawn on it — what the temporal model was actually fed. Same null semantics "
            "as `cine_overlay_gif`."
        ),
    )
    cine_per_frame_hc: list[float | None] | None = Field(
        default=None,
        description=(
            "Per-frame HC in mm, one entry per frame of the cine-loop, aligned with frame "
            "index. `null` entries are frames where HC could not be estimated. Drives the "
            "HC-stability sparkline."
        ),
    )
    cine_frame_count: int | None = Field(
        default=None, description="Number of frames in the synthesised cine-loop."
    )
    cine_key_frame_index: int | None = Field(
        default=None,
        description=(
            "Index of the frame the static `overlay_b64` still is rendered from, marked "
            "inside `cine_overlay_gif`."
        ),
    )


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


class CombinedFinding(BaseModel):
    """One per-model entry inside a combined-report request.

    `finding_id` is preferred — the server hydrates HC / GA / images from the
    in-memory findings_store. Explicit fields override the hydrated ones.
    """

    model: ModelVariant
    finding_id: str | None = None
    hc_mm: float | None = None
    ga_str: str | None = None
    ga_weeks: float | None = None
    trimester: str | None = None
    reliability: float | None = None
    confidence_label: str | None = None
    elapsed_ms: float | None = None


class CreateCombinedReportRequest(BaseModel):
    """Body for POST /studies/{study_id}/reports/combined.

    Same patient/exam fields as the single-model variant, plus a list of
    per-model findings (2–4 entries). The server computes consensus values
    across the supplied findings and renders the multi-model PDF.
    """

    findings: list[CombinedFinding] = Field(min_length=2, max_length=4)
    patient_name: str
    study_date: str
    pixel_spacing_mm: float | None = 0.070
    referring_physician: str | None = None
    patient_id: str | None = None
    patient_dob: str | None = None
    lmp: str | None = None
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
    bpd_mm: float | None = None
    prior_biometry: str | None = None


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
    is_combined: bool = False
    combined_models_json: str | None = None


class CStoreReceiveResponse(BaseModel):
    """Response for the mock C-STORE upload (Batch 7.5)."""

    id: str
    sop_class_uid: str | None
    sop_instance_uid: str | None
    patient_id: str | None
    patient_name: str | None
    study_date: str | None
    file_size: int | None
    received_at: str
    status: str = "received"


class CStoreLogEntryResponse(BaseModel):
    id: str
    sop_class_uid: str | None
    sop_instance_uid: str | None
    patient_id: str | None
    patient_name: str | None
    study_date: str | None
    file_size: int | None
    actor_ip: str | None
    user_agent: str | None
    received_at: str


class AuditEntryResponse(BaseModel):
    id: str
    report_id: str
    action: str
    actor: str | None
    ip: str | None
    user_agent: str | None
    details: str | None
    timestamp: str


# ── Retrieval-grounded Q&A ────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Free-text question about this measurement.",
    )
    top_k: int = Field(default=4, ge=1, le=8, description="How many reference chunks to retrieve.")


class RetrievedChunkOut(BaseModel):
    citation: str = Field(description="Human-readable label, e.g. 'file.md § Heading'.")
    source_file: str
    heading: str
    text: str = Field(description="The chunk text the model was given, verbatim.")
    score: float = Field(description="Cosine similarity against the question.")
    provisional: bool = Field(
        description="True when the chunk still carries a TODO(verbatim) marker — its "
        "reference text has not been verified against the primary source."
    )
    source_note: str | None = Field(
        default=None, description="The chunk's '> Source:' provenance line, if present."
    )


class AskResponse(BaseModel):
    finding_id: str
    question: str
    answer: str
    citations: list[str] = Field(
        description="Citations the answer actually referenced, not everything retrieved."
    )
    chunks: list[RetrievedChunkOut] = Field(
        description="Every chunk supplied to the model, so the answer can be checked "
        "against its evidence."
    )
    grounded: bool = Field(
        description="False when retrieval found nothing; the model is not called in that "
        "case and the answer is a refusal."
    )
    used_llm: bool = Field(
        description="False when the answer came from a fallback path (no API key, or the "
        "LLM call failed) rather than the model."
    )
    any_provisional: bool = Field(
        description="True when any supplied chunk is provisional — surface this in the UI."
    )
    disclaimer: str
