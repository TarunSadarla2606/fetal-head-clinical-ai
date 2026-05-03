"""FetalScan AI — FastAPI clinical inference API.

Routes
------
GET  /health          System status, loaded model list, device
POST /infer           Single-frame HC measurement
GET  /api/openapi.json  OpenAPI schema (auto-generated)
"""
from __future__ import annotations

import base64
import io
import logging

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import model_manager
from .deps import verify_api_key
from .inference_wrapper import predict_single_frame, validate_input
from .schemas import HealthResponse, InferResponse, ModelVariant, ValidationResult

log = logging.getLogger(__name__)

APP_VERSION = "2.1.0"

app = FastAPI(
    title="FetalScan AI — Clinical Inference API",
    version=APP_VERSION,
    description=(
        "Fetal head circumference measurement from 2D ultrasound. "
        "Research use only — not cleared for clinical diagnosis."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fetal-head-webapp.vercel.app",
        "https://*.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _decode_upload(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to a grayscale uint8 numpy array."""
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.array(img)


def _encode_png_b64(arr: np.ndarray) -> str:
    """Encode a uint8 numpy array as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Return API status, version, available models, and compute device."""
    import torch  # deferred so tests without torch skip this safely

    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        models_available=model_manager.available_variants(),
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )


@app.post(
    "/infer",
    response_model=InferResponse,
    tags=["Inference"],
    summary="Single-frame fetal head circumference measurement",
)
def infer(
    image: UploadFile = File(..., description="Ultrasound image (JPEG / PNG / DICOM)"),
    model_variant: ModelVariant = Form(default="phase0"),
    pixel_spacing_mm: float = Form(
        default=0.2,
        description="mm per pixel from DICOM tag (0028,0030). Default 0.2 ≈ HC18 dataset.",
    ),
    threshold: float = Form(default=0.5, description="Segmentation probability threshold"),
    _: None = Depends(verify_api_key),
) -> InferResponse:
    """Run the selected model on a single ultrasound frame and return HC + GA."""
    # 1. Load model (returns None if weights not configured)
    model = model_manager.get_model(model_variant)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{model_variant}' is not available. "
                f"Set the WEIGHT_{model_variant.upper()} environment variable to the weight path."
            ),
        )

    # 2. Decode image
    raw_bytes = image.file.read()
    try:
        img_gray = _decode_upload(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    # 3. Input validation / OOD check
    val_result = validate_input(img_gray)

    # 4. Inference
    result = predict_single_frame(
        model=model,
        img_gray=img_gray,
        pixel_spacing_mm=pixel_spacing_mm,
        threshold=threshold,
    )

    # 5. Encode outputs
    mask_b64    = _encode_png_b64(result["mask"] * 255)
    overlay_b64 = _encode_png_b64(result["overlay"])

    return InferResponse(
        hc_mm=result["hc_mm"],
        ga_str=result["ga_str"],
        ga_weeks=result["ga_weeks"],
        trimester=result["trimester"],
        reliability=result["reliability"],
        hc_std_mm=result["hc_std_mm"],
        confidence_label=result["confidence_label"],
        confidence_color=result["confidence_color"],
        elapsed_ms=result["elapsed_ms"],
        mode=result["mode"],
        validation=ValidationResult(**val_result),
        ood_flag=not val_result["valid"],
        ood_reasons=val_result["warnings"],
        mask_b64=mask_b64,
        overlay_b64=overlay_b64,
    )
