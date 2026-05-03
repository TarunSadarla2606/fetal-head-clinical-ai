"""Comprehensive tests for /demo/list, /demo/{filename}, and GET / redirect."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app, follow_redirects=False)


# ── GET / redirect ──────────────────────────────────────────────────────────────────────────────
class TestRootRedirect:
    def test_root_returns_redirect(self):
        resp = client.get("/")
        assert resp.status_code in (301, 302, 307, 308)

    def test_root_redirects_to_api_docs(self):
        resp = client.get("/")
        location = resp.headers.get("location", "")
        assert "/api/docs" in location

    def test_api_docs_accessible_after_redirect(self):
        client_follow = TestClient(app, follow_redirects=True)
        resp = client_follow.get("/")
        assert resp.status_code == 200


# ── GET /demo/list ───────────────────────────────────────────────────────────────────────────────
class TestDemoList:
    def test_returns_200(self):
        resp = client.get("/demo/list")
        assert resp.status_code == 200

    def test_response_is_json(self):
        resp = client.get("/demo/list")
        assert resp.headers["content-type"].startswith("application/json")

    def test_response_has_files_key(self):
        data = client.get("/demo/list").json()
        assert "files" in data

    def test_files_is_a_list(self):
        data = client.get("/demo/list").json()
        assert isinstance(data["files"], list)

    def test_files_contains_only_strings(self):
        data = client.get("/demo/list").json()
        assert all(isinstance(f, str) for f in data["files"])

    def test_files_are_sorted(self):
        data = client.get("/demo/list").json()
        assert data["files"] == sorted(data["files"])

    def test_files_have_image_extensions(self):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        for fname in client.get("/demo/list").json()["files"]:
            assert Path(fname).suffix.lower() in valid_exts, f"{fname!r} has non-image extension"

    def test_no_directories_in_list(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "scan.png").write_bytes(b"x")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            data = client.get("/demo/list").json()
            assert "subdir" not in data["files"]

    def test_non_image_files_excluded(self, tmp_path):
        (tmp_path / "model.pth").write_bytes(b"x")
        (tmp_path / "notes.txt").write_bytes(b"x")
        (tmp_path / "data.csv").write_bytes(b"x")
        (tmp_path / "scan.png").write_bytes(b"x")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            data = client.get("/demo/list").json()
            assert data["files"] == ["scan.png"]

    def test_returns_empty_list_when_dir_missing(self, tmp_path):
        with patch("app.api.main._DEMO_DIR", tmp_path / "nonexistent"):
            resp = client.get("/demo/list")
            assert resp.status_code == 200
            assert resp.json() == {"files": []}

    def test_multiple_image_types_included(self, tmp_path):
        for name in ["a.jpg", "b.jpeg", "c.png", "d.bmp"]:
            (tmp_path / name).write_bytes(b"x")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            files = client.get("/demo/list").json()["files"]
            assert set(files) == {"a.jpg", "b.jpeg", "c.png", "d.bmp"}

    def test_filenames_do_not_contain_path_separator(self, tmp_path):
        (tmp_path / "scan.png").write_bytes(b"x")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            for fname in client.get("/demo/list").json()["files"]:
                assert "/" not in fname
                assert "\\" not in fname

    def test_case_insensitive_extension_match(self, tmp_path):
        (tmp_path / "UPPER.PNG").write_bytes(b"x")
        (tmp_path / "lower.jpg").write_bytes(b"x")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            files = client.get("/demo/list").json()["files"]
            names = [f.lower() for f in files]
            assert "upper.png" in names
            assert "lower.jpg" in names


# ── GET /demo/{filename} ─────────────────────────────────────────────────────────────────────────
class TestDemoGet:
    def test_200_for_existing_file(self, tmp_path):
        (tmp_path / "scan.png").write_bytes(b"PNG_DATA")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            resp = client.get("/demo/scan.png")
            assert resp.status_code == 200

    def test_response_body_matches_file_content(self, tmp_path):
        content = b"\x89PNG\r\n\x1a\n" + b"fake_image_payload"
        (tmp_path / "scan.png").write_bytes(content)
        with patch("app.api.main._DEMO_DIR", tmp_path):
            resp = client.get("/demo/scan.png")
            assert resp.content == content

    def test_404_for_missing_file(self, tmp_path):
        with patch("app.api.main._DEMO_DIR", tmp_path):
            resp = client.get("/demo/does_not_exist.png")
            assert resp.status_code == 404

    def test_404_when_demo_dir_missing(self):
        with patch("app.api.main._DEMO_DIR", Path("/definitely_nonexistent_xyz123")):
            resp = client.get("/demo/any.png")
            assert resp.status_code == 404

    def test_404_detail_mentions_filename(self, tmp_path):
        with patch("app.api.main._DEMO_DIR", tmp_path):
            resp = client.get("/demo/ghost.png")
            assert resp.status_code == 404
            assert "ghost.png" in resp.json()["detail"]

    def test_path_traversal_blocked(self, tmp_path):
        # Attempt to escape the demo dir
        with patch("app.api.main._DEMO_DIR", tmp_path):
            resp = client.get("/demo/..%2Frequirements.txt")
            assert resp.status_code in (400, 404)

    def test_double_dot_in_filename_blocked(self, tmp_path):
        with patch("app.api.main._DEMO_DIR", tmp_path):
            resp = client.get("/demo/../../etc/passwd")
            assert resp.status_code in (400, 404)

    def test_serves_jpg_file(self, tmp_path):
        (tmp_path / "scan.jpg").write_bytes(b"JFIF_DATA")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            resp = client.get("/demo/scan.jpg")
            assert resp.status_code == 200

    def test_response_has_content_type(self, tmp_path):
        (tmp_path / "scan.png").write_bytes(b"x")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            resp = client.get("/demo/scan.png")
            assert "content-type" in resp.headers

    def test_multiple_files_accessible(self, tmp_path):
        for name in ["a.png", "b.png", "c.jpg"]:
            (tmp_path / name).write_bytes(b"data")
        with patch("app.api.main._DEMO_DIR", tmp_path):
            for name in ["a.png", "b.png", "c.jpg"]:
                assert client.get(f"/demo/{name}").status_code == 200
