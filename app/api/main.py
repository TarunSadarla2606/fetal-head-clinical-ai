"""FetalScan AI — FastAPI clinical inference API.

Routes
------
GET  /                    Redirect to interactive API docs
GET  /health              System status, loaded model list, device
GET  /demo/list           List filenames in demo_subjects directory
GET  /demo/{filename}     Serve a demo subject image
POST /infer               Single-frame HC measurement
GET  /api/openapi.json    OpenAPI schema (auto-generated)
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image

from . import inference_wrapper, model_manager
from .deps import verify_api_key
from .schemas import HealthResponse, InferResponse, ModelVariant, ValidationResult

log = logging.getLogger(__name__)

APP_VERSION = "2.2.0"

_DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "demo_subjects"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

app = FastAPI(
    title="FetalScan AI — Clinical Inference API",
    version=APP_VERSION,
    description=(
        "Fetal head circumference measurement from 2D ultrasound. "
        "For demonstration purposes only — not cleared for clinical diagnosis."
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


# ── helpers ────────────────────────────────────────────────────────────────────────────────────
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


# ── routes ─────────────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/api/docs")


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


@app.get("/demo/list", tags=["Demo"])
def list_demo_subjects() -> dict:
    """Return sorted list of image filenames in the demo_subjects directory."""
    if not _DEMO_DIR.is_dir():
        return {"files": []}
    names = [f.name for f in _DEMO_DIR.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS]
    return {"files": sorted(names)}


@app.get("/demo/{filename}", tags=["Demo"])
def get_demo_subject(filename: str) -> FileResponse:
    """Serve a single demo subject image by filename."""
    if not _DEMO_DIR.is_dir():
        raise HTTPException(status_code=404, detail="Demo subjects directory not found")
    file_path = (_DEMO_DIR / filename).resolve()
    if not str(file_path).startswith(str(_DEMO_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename!r} not found")
    return FileResponse(str(file_path))


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
    model = model_manager.get_model(model_variant)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{model_variant}' is not available. "
                f"Set the WEIGHT_{model_variant.upper()} environment variable to the weight path."
            ),
        )

    raw_bytes = image.file.read()
    try:
        img_gray = _decode_upload(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    val_result = inference_wrapper.validate_input(img_gray)

    result = inference_wrapper.predict_single_frame(
        model=model,
        img_gray=img_gray,
        pixel_spacing_mm=pixel_spacing_mm,
        threshold=threshold,
    )

    mask_b64 = _encode_png_b64(result["mask"] * 255)
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
