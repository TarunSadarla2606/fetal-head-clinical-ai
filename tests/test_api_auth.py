"""Batch 1: API key authentication tests — no model weights required."""
from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app


def _png_bytes(h: int = 64, w: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_no_key_configured_allows_request(monkeypatch):
    monkeypatch.delenv("FETALSCAN_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post(
        "/infer",
        data={"model_variant": "phase0"},
        files={"image": ("us.png", _png_bytes(), "image/png")},
    )
    # No key configured → auth passes; model not loaded → 503
    assert resp.status_code != 401


def test_wrong_key_returns_401(monkeypatch):
    monkeypatch.setenv("FETALSCAN_API_KEY", "secret-abc-123")
    client = TestClient(app)
    resp = client.post(
        "/infer",
        headers={"X-API-Key": "wrong-key"},
        data={"model_variant": "phase0"},
        files={"image": ("us.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 401


def test_no_key_header_returns_401_when_key_configured(monkeypatch):
    monkeypatch.setenv("FETALSCAN_API_KEY", "secret-abc-123")
    client = TestClient(app)
    resp = client.post(
        "/infer",
        data={"model_variant": "phase0"},
        files={"image": ("us.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 401


def test_correct_key_passes_auth(monkeypatch):
    monkeypatch.setenv("FETALSCAN_API_KEY", "secret-abc-123")
    client = TestClient(app)
    resp = client.post(
        "/infer",
        headers={"X-API-Key": "secret-abc-123"},
        data={"model_variant": "phase0"},
        files={"image": ("us.png", _png_bytes(), "image/png")},
    )
    # Auth passed → no 401; model not loaded → 503
    assert resp.status_code != 401


def test_health_endpoint_requires_no_key(monkeypatch):
    """Health check must always be accessible regardless of auth config."""
    monkeypatch.setenv("FETALSCAN_API_KEY", "secret-abc-123")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
