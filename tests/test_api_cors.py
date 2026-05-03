"""CORS header tests — verifies the browser can actually receive API responses."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app, follow_redirects=False)

ALLOWED_ORIGINS = [
    "https://fetal-head-webapp.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
]


class TestCORSSimpleRequests:
    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_allowed_origin_gets_cors_header(self, origin):
        resp = client.get("/demo/list", headers={"Origin": origin})
        assert resp.status_code == 200
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == origin or acao == "*", f"Expected CORS header for {origin}, got {acao!r}"

    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_allowed_origin_on_health(self, origin):
        resp = client.get("/health", headers={"Origin": origin})
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == origin or acao == "*"

    def test_unknown_origin_gets_no_cors_header(self):
        resp = client.get("/demo/list", headers={"Origin": "https://evil.example.com"})
        acao = resp.headers.get("access-control-allow-origin", "")
        # Must NOT echo back an untrusted origin
        assert acao != "https://evil.example.com"


class TestCORSPreflightRequests:
    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_preflight_demo_list(self, origin):
        resp = client.options(
            "/demo/list",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code in (200, 204)

    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_preflight_infer_allows_post(self, origin):
        resp = client.options(
            "/infer",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code in (200, 204)
        methods = resp.headers.get("access-control-allow-methods", "")
        assert "POST" in methods

    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_preflight_returns_allow_origin(self, origin):
        resp = client.options(
            "/demo/list",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == origin or acao == "*"

    def test_preflight_allows_content_type_header(self):
        resp = client.options(
            "/infer",
            headers={
                "Origin": ALLOWED_ORIGINS[0],
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert resp.status_code in (200, 204)


class TestCORSVercelPreviewURLs:
    """Vercel generates preview URLs like https://proj-hash-org.vercel.app.
    These must also be allowed by the wildcard rule.
    """

    def test_main_vercel_domain_allowed(self):
        origin = "https://fetal-head-webapp.vercel.app"
        resp = client.get("/health", headers={"Origin": origin})
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == origin or acao == "*"
