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
