"""Batch 1: /health endpoint tests — no model weights required."""

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_has_required_fields():
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "models_available" in data
    assert "device" in data


def test_health_models_available_is_list():
    data = client.get("/health").json()
    assert isinstance(data["models_available"], list)


def test_health_version_value():
    data = client.get("/health").json()
    assert data["version"] == "2.5.0"


def test_health_device_is_string():
    data = client.get("/health").json()
    assert isinstance(data["device"], str)
    assert len(data["device"]) > 0


def test_openapi_schema_accessible():
    resp = client.get("/api/openapi.json")
    assert resp.status_code == 200


def test_openapi_schema_has_health_and_infer():
    schema = client.get("/api/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/infer" in schema["paths"]
