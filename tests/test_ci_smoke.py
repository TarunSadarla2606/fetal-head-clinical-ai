"""Batch 0 smoke tests — no model weights required, fast CI."""

from pathlib import Path

import tomllib

ROOT = Path(__file__).parent.parent


def test_pyproject_parseable():
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["name"] == "fetal-head-clinical-ai"
    assert data["project"]["requires-python"] == ">=3.11"


def test_requirements_nonempty():
    req = ROOT / "requirements.txt"
    assert req.exists()
    lines = [
        line.strip()
        for line in req.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(lines) > 5


def test_app_directory_structure():
    app = ROOT / "app"
    for fname in ("app.py", "inference.py", "report.py", "xai.py"):
        assert (app / fname).exists(), f"missing app/{fname}"


def test_streamlit_config_valid():
    cfg = ROOT / ".streamlit" / "config.toml"
    assert cfg.exists()
    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    assert "theme" in data
    assert data["theme"]["primaryColor"] == "#0D7680"


def test_tests_init_exists():
    assert (ROOT / "tests" / "__init__.py").exists()


def test_no_secrets_in_requirements():
    content = (ROOT / "requirements.txt").read_text().lower()
    for token in ("sk-", "api_key=", "secret=", "password="):
        assert token not in content, f"Possible secret in requirements.txt: {token!r}"


def test_gitignore_covers_model_weights():
    gitignore = (ROOT / ".gitignore").read_text()
    assert any(pattern in gitignore for pattern in ("*.pt", "*.pth", "*.bin"))
