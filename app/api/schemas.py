"""Pydantic request/response schemas for the FetalScan AI inference API."""
from __future__ import annotations

from typing import Literal, Optional

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
    hc_mm: Optional[float] = None
    ga_str: Optional[str] = None
    ga_weeks: Optional[float] = None
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
