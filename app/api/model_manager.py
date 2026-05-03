"""Lazy model loading and cache for the FastAPI inference API."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment variables pointing to each model's weight file on disk.
_WEIGHT_ENVS: dict[str, str] = {
    "phase0": "WEIGHT_PHASE0",
    "phase4a": "WEIGHT_PHASE4A",
    "phase2": "WEIGHT_PHASE2",
    "phase4b": "WEIGHT_PHASE4B",
}

_cache: dict[str, object] = {}


def _find_weight_path(variant: str) -> str:
    """Scan common directories for a .pth file whose name contains *variant*."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    search_dirs = [
        repo_root,
        Path.cwd(),
        repo_root / "models",
        repo_root / "weights",
    ]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for pth_file in sorted(search_dir.glob("*.pth")):
            if variant.lower() in pth_file.name.lower():
                log.info("Auto-detected weight file for %s: %s", variant, pth_file)
                return str(pth_file)
    return ""


def get_model(variant: str) -> object | None:
    """Return cached model for *variant*, loading it on first call.

    Returns None if the weight file is not configured or does not exist.
    """
    if variant in _cache:
        return _cache[variant]

    env_key = _WEIGHT_ENVS.get(variant)
    if not env_key:
        raise ValueError(f"Unknown model variant: {variant!r}")

    weight_path = os.getenv(env_key, "") or _find_weight_path(variant)
    if not weight_path or not Path(weight_path).exists():
        log.warning(
            "Weight file for %s not found (env %s=%r, auto-detect returned %r)",
            variant, env_key, os.getenv(env_key, ""), weight_path,
        )
        return None

    try:
        from app.inference import (  # noqa: PLC0415
            load_phase0,
            load_phase2,
            load_phase4a,
            load_phase4b,
        )

        loaders = {
            "phase0": load_phase0,
            "phase4a": load_phase4a,
            "phase2": load_phase2,
            "phase4b": load_phase4b,
        }
        log.info("Loading %s from %s", variant, weight_path)
        model = loaders[variant](weight_path, device=DEVICE)
        _cache[variant] = model
        return model
    except Exception:
        log.exception("Failed to load model %s", variant)
        return None


def available_variants() -> list[str]:
    """Return variant names whose weight files exist on disk."""
    result = []
    for variant, env_key in _WEIGHT_ENVS.items():
        path = os.getenv(env_key, "") or _find_weight_path(variant)
        if path and Path(path).exists():
            result.append(variant)
    return result
