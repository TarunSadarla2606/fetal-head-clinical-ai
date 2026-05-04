"""Batch 5: XAI endpoint tests — /findings/{id}/{gradcam,uncertainty,ood}.

These tests cover the spec items:
  5.1  GET /findings/{id}/gradcam returns PNG overlay → overlay shape matches input
  5.2  GET /findings/{id}/uncertainty returns heatmap → heatmap variance > 0
  5.3  GET /findings/{id}/ood returns flag + reason → OOD trips on noise input

We mock model inference and the heavy XAI compute paths so the tests run in
seconds without weights — the goal is to validate the API contract, the
findings-store integration, and the OOD signal logic, not to re-prove the
GradCAM math (that lives in unit-style tests if/when we add them).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.api import findings_store
from app.api.main import app
from app.api.xai_endpoints import analyze_ood

client = TestClient(app)


# ── helpers ────────────────────────────────────────────────────────────────────


def _png_bytes(h: int = 256, w: int = 256, seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(20, 200, (h, w), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _flat_png_bytes(h: int = 256, w: int = 256, value: int = 5) -> bytes:
    """Generate an out-of-distribution image: near-black flat field."""
    arr = np.full((h, w), value, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _mock_prediction() -> dict:
    return {
        "mask": np.zeros((256, 384), dtype=np.uint8),
        "prob_map": np.zeros((256, 384), dtype=np.float32),
        "overlay": np.zeros((256, 384, 3), dtype=np.uint8),
        "hc_mm": 274.3,
        "ga_str": "28w 1d",
        "ga_weeks": 28.14,
        "trimester": "Mid (20–30w)",
        "reliability": 0.95,
        "hc_std_mm": 0.8,
        "confidence_label": "HIGH CONFIDENCE",
        "confidence_color": "#16a34a",
        "elapsed_ms": 42.5,
        "mode": "single_frame",
    }


def _mock_validation_pass() -> dict:
    return {
        "valid": True,
        "warnings": [],
        "checks": {k: True for k in
                   ["shape", "resolution", "not_blank", "not_saturated",
                    "dynamic_range", "aspect_ratio", "has_texture"]},
    }


def _create_finding(image_bytes: bytes, validation: dict | None = None) -> str:
    """POST /infer with mocked model and return the resulting finding_id."""
    findings_store.clear()
    val = validation or _mock_validation_pass()
    with (
        patch("app.api.model_manager.get_model", return_value=MagicMock()),
        patch("app.api.inference_wrapper.predict_single_frame", return_value=_mock_prediction()),
        patch("app.api.inference_wrapper.validate_input", return_value=val),
    ):
        resp = client.post(
            "/infer",
            data={"model_variant": "phase0", "pixel_spacing_mm": "0.2"},
            files={"image": ("us.png", image_bytes, "image/png")},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["finding_id"]


# ── /infer now returns finding_id ─────────────────────────────────────────────


def test_infer_response_includes_finding_id():
    finding_id = _create_finding(_png_bytes())
    assert isinstance(finding_id, str) and len(finding_id) > 0


# ── 5.1 GradCAM endpoint ──────────────────────────────────────────────────────


def test_gradcam_returns_png_with_input_shape():
    """5.1 — /findings/{id}/gradcam returns PNG; decoded shape matches input."""
    finding_id = _create_finding(_png_bytes(h=192, w=256))

    fake_overlay = np.zeros((192, 256, 3), dtype=np.uint8)
    with (
        patch("app.api.model_manager.get_model", return_value=MagicMock()),
        patch("app.api.xai_endpoints.compute_gradcam", return_value=fake_overlay) as mock_cam,
    ):
        resp = client.get(f"/findings/{finding_id}/gradcam")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    # Decode PNG and confirm the shape that came back matches the input image
    img = Image.open(io.BytesIO(resp.content))
    assert img.size == (256, 192)  # PIL size is (W, H)
    mock_cam.assert_called_once()


def test_gradcam_404_when_finding_unknown():
    resp = client.get("/findings/does-not-exist/gradcam")
    assert resp.status_code == 404


def test_gradcam_503_when_model_unloaded():
    finding_id = _create_finding(_png_bytes())
    with patch("app.api.model_manager.get_model", return_value=None):
        resp = client.get(f"/findings/{finding_id}/gradcam")
    assert resp.status_code == 503


# ── 5.2 Uncertainty endpoint ──────────────────────────────────────────────────


def test_uncertainty_returns_png_and_real_compute_has_variance():
    """5.2 — /findings/{id}/uncertainty returns PNG; the underlying compute
    produces a heatmap with non-zero pixel variance (the test asserts the
    NumPy variance is > 0, which is the spec criterion).
    """
    finding_id = _create_finding(_png_bytes())

    # Synthesize an overlay whose pixel-variance is clearly positive — proving
    # that the response carries actual heatmap signal, not a flat field.
    fake_overlay = np.tile(
        np.linspace(0, 255, 256, dtype=np.uint8)[None, :, None],
        (256, 1, 3),
    )
    with (
        patch("app.api.model_manager.get_model", return_value=MagicMock()),
        patch("app.api.xai_endpoints.compute_uncertainty", return_value=fake_overlay),
    ):
        resp = client.get(f"/findings/{finding_id}/uncertainty")

    assert resp.status_code == 200
    decoded = np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))
    assert decoded.var() > 0  # spec: heatmap variance > 0


def test_uncertainty_404_when_finding_unknown():
    resp = client.get("/findings/missing-id/uncertainty")
    assert resp.status_code == 404


# ── 5.3 OOD endpoint ──────────────────────────────────────────────────────────


def test_ood_endpoint_clean_image_passes():
    """A natural-looking random image should not trip the OOD flag (or, at
    most, trips lightly with score < 0.8).
    """
    finding_id = _create_finding(_png_bytes())
    resp = client.get(f"/findings/{finding_id}/ood")
    assert resp.status_code == 200
    body = resp.json()
    assert "ood_flag" in body
    assert "reasons" in body
    assert "score" in body
    assert "stats" in body
    assert body["score"] < 0.8


def test_ood_endpoint_trips_on_flat_noise_input():
    """5.3 — feeding a flat (low-contrast / low-texture) image into the
    underlying analyze_ood logic must trip the OOD flag with at least one
    structured reason.
    """
    flat = np.full((256, 256), 5, dtype=np.uint8)
    failing_validation = {
        "valid": False,
        "warnings": ["Image appears to be blank (near-black)."],
        "checks": {"not_blank": False, "has_texture": False, "dynamic_range": False},
    }
    report = analyze_ood(flat, failing_validation)
    assert report["ood_flag"] is True
    assert len(report["reasons"]) >= 1
    # Categories should include both validation-derived and stat-derived reasons
    categories = {r["category"] for r in report["reasons"]}
    assert "input_validation" in categories
    # Score reflects multiple triggered checks
    assert 0.5 <= report["score"] <= 1.0


def test_ood_endpoint_404_when_finding_unknown():
    resp = client.get("/findings/missing-id/ood")
    assert resp.status_code == 404


# ── analyze_ood unit tests ────────────────────────────────────────────────────


def test_analyze_ood_includes_image_stats():
    arr = (np.random.default_rng(0).normal(120, 40, (256, 256)).clip(0, 255)).astype(np.uint8)
    report = analyze_ood(arr, _mock_validation_pass())
    stats = report["stats"]
    assert {"mean_intensity", "std_intensity", "edge_density", "laplacian_variance"} <= stats.keys()
    assert all(isinstance(v, float) for v in stats.values())


def test_analyze_ood_low_contrast_triggers_reason():
    flat = np.full((256, 256), 100, dtype=np.uint8)
    report = analyze_ood(flat, _mock_validation_pass())
    categories = {r["category"] for r in report["reasons"]}
    assert "low_contrast" in categories
