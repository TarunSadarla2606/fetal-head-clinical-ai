"""FastAPI dependencies: API-key verification."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status

_ENV_KEY = "FETALSCAN_API_KEY"


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """Check X-API-Key header. If no key is configured, access is open (dev mode)."""
    expected = os.getenv(_ENV_KEY, "")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
