"""In-memory LRU store for finding records, keyed by UUID.

A *finding record* bundles everything an XAI endpoint needs to recompute
explanations for an inference call: the original grayscale image, the model
variant, the threshold and pixel spacing, and the previous result.

The store is bounded (default 128 entries) and evicts the oldest record
when full. Records expire after :data:`TTL_SECONDS` to keep memory steady
under load. This is intentionally simple — for production you would back
this with Redis or similar, but for a demo Space the in-memory dict is
sufficient and avoids any external dependencies.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

MAX_ENTRIES = 128
TTL_SECONDS = 60 * 60  # 1 hour


@dataclass
class FindingRecord:
    finding_id: str
    img_gray: np.ndarray
    model_variant: str
    pixel_spacing_mm: float
    threshold: float
    findings: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


_store: OrderedDict[str, FindingRecord] = OrderedDict()
_lock = threading.Lock()


def _evict_expired() -> None:
    """Drop entries older than TTL_SECONDS. Caller must hold _lock."""
    now = time.time()
    expired = [k for k, v in _store.items() if now - v.created_at > TTL_SECONDS]
    for k in expired:
        _store.pop(k, None)


def store(
    img_gray: np.ndarray,
    model_variant: str,
    pixel_spacing_mm: float,
    threshold: float,
    findings: dict,
) -> str:
    """Persist a record and return its UUID."""
    finding_id = uuid.uuid4().hex
    record = FindingRecord(
        finding_id=finding_id,
        img_gray=img_gray.copy(),
        model_variant=model_variant,
        pixel_spacing_mm=pixel_spacing_mm,
        threshold=threshold,
        findings=findings,
    )
    with _lock:
        _evict_expired()
        _store[finding_id] = record
        while len(_store) > MAX_ENTRIES:
            _store.popitem(last=False)
    return finding_id


def get(finding_id: str) -> FindingRecord | None:
    """Return the record for *finding_id* or None if missing/expired."""
    with _lock:
        _evict_expired()
        return _store.get(finding_id)


def clear() -> None:
    """Drop all records — used by tests."""
    with _lock:
        _store.clear()
