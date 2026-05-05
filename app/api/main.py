"""FetalScan AI — FastAPI clinical inference API.

Routes
------
GET  /                              Redirect to interactive API docs
GET  /health                        System status, loaded model list, device
GET  /demo/list                     List filenames in demo_subjects directory
GET  /demo/{filename}/metadata      Pixel spacing + HC reference from HC18 CSV
GET  /demo/{filename}               Serve a demo subject image
POST /infer                         Single-frame HC measurement
GET  /findings/{id}/gradcam         GradCAM++ overlay PNG (Batch 5)
GET  /findings/{id}/uncertainty     MC uncertainty heatmap PNG (Batch 5)
GET  /findings/{id}/ood             OOD flag + structured reasons JSON (Batch 5)
GET  /api/openapi.json              OpenAPI schema (auto-generated)
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image

from app.inference import N_FRAMES, TemporalFetaSegNet

from . import (
    findings_store,
    inference_wrapper,
    model_manager,
    reports_db,
    reports_endpoints,
    xai_endpoints,
)
from .deps import verify_api_key
from .schemas import (
    HealthResponse,
    InferResponse,
    ModelVariant,
    OodResponse,
    ValidationResult,
)

log = logging.getLogger(__name__)

APP_VERSION = "2.5.0"

_DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "demo_subjects"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# HC18 CSV: filename → (pixel_spacing_mm, hc_reference_mm)
# Loaded once at startup; absent file is handled gracefully.
def _load_hc18_csv() -> dict[str, dict]:
    csv_path = _DEMO_DIR.parent / "training_set_pixel_size_and_HC.csv"
    result: dict[str, dict] = {}
    if not csv_path.is_file():
        return result
    try:
        import csv

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                fn = (row.get("filename") or "").strip()
                ps = (row.get("pixel size(mm)") or "").strip()
                hc = (row.get("head circumference (mm)") or "").strip()
                if fn and ps:
                    result[fn] = {
                        "pixel_spacing_mm": float(ps),
                        "hc_reference_mm": float(hc) if hc else None,
                    }
    except Exception:
        pass
    return result


_HC18_META: dict[str, dict] = _load_hc18_csv()

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
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialise the SQLite reports DB on startup so the first request doesn't
# race the schema bootstrap.
reports_db.init_db()
app.include_router(reports_endpoints.router)


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


@app.get("/demo/{filename}/metadata", tags=["Demo"])
def get_demo_metadata(filename: str) -> dict:
    """Return pixel spacing and HC reference from the HC18 CSV for one demo subject.

    Returns 404 when the filename is not in the CSV (non-HC18 demo images).
    Frontend uses this to auto-apply the correct pixel spacing and display
    the ground-truth HC reference alongside the AI prediction.
    """
    meta = _HC18_META.get(filename)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"No HC18 metadata for {filename!r}")
    return meta


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
    """Run the selected model on a single ultrasound frame and return HC + GA.

    Temporal models (phase2, phase4b) accept single-frame input by tiling the
    frame to N_FRAMES so the cine-loop architecture can be exercised without a
    full video clip.
    """
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

    # Temporal models expect a 5-D clip [B, T, C, H, W]. Tile the single frame
    # so single-image demo subjects work with phase2 and phase4b.
    if isinstance(model, TemporalFetaSegNet):
        result = inference_wrapper.predict_cine_clip(
            model=model,
            frames=[img_gray] * N_FRAMES,
            pixel_spacing_mm=pixel_spacing_mm,
            threshold=threshold,
        )
    else:
        result = inference_wrapper.predict_single_frame(
            model=model,
            img_gray=img_gray,
            pixel_spacing_mm=pixel_spacing_mm,
            threshold=threshold,
        )

    mask_b64 = _encode_png_b64(result["mask"] * 255)
    overlay_b64 = _encode_png_b64(result["overlay"])

    finding_id = findings_store.store(
        img_gray=img_gray,
        model_variant=model_variant,
        pixel_spacing_mm=pixel_spacing_mm,
        threshold=threshold,
        findings={
            "hc_mm": result["hc_mm"],
            "ga_str": result["ga_str"],
            "ga_weeks": result["ga_weeks"],
            "trimester": result["trimester"],
            "reliability": result["reliability"],
            "hc_std_mm": result["hc_std_mm"],
            "confidence_label": result["confidence_label"],
            "elapsed_ms": result["elapsed_ms"],
            "mode": result["mode"],
            "validation": val_result,
            "mask_b64": mask_b64,
            "overlay_b64": overlay_b64,
        },
    )

    return InferResponse(
        finding_id=finding_id,
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


# ── XAI endpoints (Batch 5) ────────────────────────────────────────────────────


def _load_finding_or_404(finding_id: str) -> findings_store.FindingRecord:
    record = findings_store.get(finding_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Finding {finding_id!r} not found. Findings expire after 1 hour or "
                "when the LRU cache fills — re-run /infer to generate a fresh ID."
            ),
        )
    return record


def _png_response(rgb: np.ndarray) -> Response:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encoding failed")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get(
    "/findings/{finding_id}/gradcam",
    tags=["XAI"],
    summary="GradCAM++ overlay PNG for a stored finding",
    responses={200: {"content": {"image/png": {}}}},
)
def get_gradcam(
    finding_id: str,
    _: None = Depends(verify_api_key),
) -> Response:
    record = _load_finding_or_404(finding_id)
    model = model_manager.get_model(record.model_variant)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{record.model_variant}' is no longer loaded.",
        )
    overlay = xai_endpoints.compute_gradcam(model, record.img_gray)
    return _png_response(overlay)


@app.get(
    "/findings/{finding_id}/uncertainty",
    tags=["XAI"],
    summary="MC uncertainty heatmap PNG for a stored finding",
    responses={200: {"content": {"image/png": {}}}},
)
def get_uncertainty(
    finding_id: str,
    _: None = Depends(verify_api_key),
) -> Response:
    record = _load_finding_or_404(finding_id)
    model = model_manager.get_model(record.model_variant)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{record.model_variant}' is no longer loaded.",
        )
    overlay = xai_endpoints.compute_uncertainty(model, record.img_gray)
    return _png_response(overlay)


@app.get(
    "/findings/{finding_id}/ood",
    response_model=OodResponse,
    tags=["XAI"],
    summary="OOD flag + structured reasons for a stored finding",
)
def get_ood(
    finding_id: str,
    _: None = Depends(verify_api_key),
) -> OodResponse:
    record = _load_finding_or_404(finding_id)
    val_result = record.findings.get("validation") or inference_wrapper.validate_input(
        record.img_gray
    )
    report = xai_endpoints.analyze_ood(record.img_gray, val_result)
    return OodResponse(**report)
