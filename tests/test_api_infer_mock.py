"""Batch 1: /infer endpoint tests with mocked model — no weights required."""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app


def _png_bytes(h: int = 256, w: int = 256) -> bytes:
    rng = np.random.default_rng(42)
    arr = rng.integers(20, 200, (h, w), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _mock_prediction() -> dict:
    mask = np.zeros((256, 384), dtype=np.uint8)
    overlay = np.zeros((256, 384, 3), dtype=np.uint8)
    return {
        "mask": mask,
        "prob_map": np.zeros((256, 384), dtype=np.float32),
        "overlay": overlay,
        "hc_mm": 274.3,
        "ga_str": "28w 1d",
        "ga_weeks": 28.14,
        "trimester": "Third trimester (≥28w)",
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
        "checks": {
            "shape": True,
            "resolution": True,
            "not_blank": True,
            "not_saturated": True,
            "dynamic_range": True,
            "aspect_ratio": True,
            "has_texture": True,
        },
    }


client = TestClient(app)


def _patched_client():
    mock_model = MagicMock()
    return (
        patch("app.api.model_manager.get_model", return_value=mock_model),
        patch("app.api.inference_wrapper.predict_single_frame", return_value=_mock_prediction()),
        patch("app.api.inference_wrapper.validate_input", return_value=_mock_validation_pass()),
    )


def test_infer_returns_200():
    p1, p2, p3 = _patched_client()
    with p1, p2, p3:
        resp = client.post(
            "/infer",
            data={"model_variant": "phase0", "pixel_spacing_mm": "0.2"},
            files={"image": ("us.png", _png_bytes(), "image/png")},
        )
    assert resp.status_code == 200


def test_infer_hc_and_ga_in_response():
    p1, p2, p3 = _patched_client()
    with p1, p2, p3:
        data = client.post(
            "/infer",
            data={"model_variant": "phase0"},
            files={"image": ("us.png", _png_bytes(), "image/png")},
        ).json()
    assert abs(data["hc_mm"] - 274.3) < 0.01
    assert data["ga_str"] == "28w 1d"
    assert data["trimester"] == "Third trimester (≥28w)"


def test_infer_overlay_b64_is_valid_base64():
    p1, p2, p3 = _patched_client()
    with p1, p2, p3:
        data = client.post(
            "/infer",
            data={"model_variant": "phase0"},
            files={"image": ("us.png", _png_bytes(), "image/png")},
        ).json()
    assert len(data["overlay_b64"]) > 0
    base64.b64decode(data["overlay_b64"])  # raises if invalid


def test_infer_mask_b64_is_valid_base64():
    p1, p2, p3 = _patched_client()
    with p1, p2, p3:
        data = client.post(
            "/infer",
            data={"model_variant": "phase0"},
            files={"image": ("us.png", _png_bytes(), "image/png")},
        ).json()
    base64.b64decode(data["mask_b64"])


def test_infer_validation_field_present():
    p1, p2, p3 = _patched_client()
    with p1, p2, p3:
        data = client.post(
            "/infer",
            data={"model_variant": "phase0"},
            files={"image": ("us.png", _png_bytes(), "image/png")},
        ).json()
    assert "validation" in data
    assert data["validation"]["valid"] is True
    assert isinstance(data["ood_flag"], bool)


def test_infer_503_when_model_unavailable():
    with patch("app.api.model_manager.get_model", return_value=None):
        resp = client.post(
            "/infer",
            data={"model_variant": "phase0"},
            files={"image": ("us.png", _png_bytes(), "image/png")},
        )
    assert resp.status_code == 503


def test_infer_400_on_invalid_image_bytes():
    with patch("app.api.model_manager.get_model", return_value=MagicMock()):
        resp = client.post(
            "/infer",
            data={"model_variant": "phase0"},
            files={"image": ("bad.png", b"not-an-image", "image/png")},
        )
    assert resp.status_code == 400


def test_infer_ood_flag_true_when_validation_fails():
    mock_model = MagicMock()
    fail_val = {
        "valid": False,
        "warnings": ["Image appears to be blank."],
        "checks": {"not_blank": False},
    }
    with (
        patch("app.api.model_manager.get_model", return_value=mock_model),
        patch("app.api.inference_wrapper.validate_input", return_value=fail_val),
        patch("app.api.inference_wrapper.predict_single_frame", return_value=_mock_prediction()),
    ):
        data = client.post(
            "/infer",
            data={"model_variant": "phase0"},
            files={"image": ("us.png", _png_bytes(), "image/png")},
        ).json()
    assert data["ood_flag"] is True
    assert len(data["ood_reasons"]) > 0
