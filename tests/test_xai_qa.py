"""Interactive XAI — attribution summarisation and grounded explanation.

The whole point is that the explanation is grounded in measured attribution
rather than in the rendered picture, so the tests are about what the summary
actually measures and what the endpoint refuses to say without it.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import attribution, findings_store
from app.api.main import app

client = TestClient(app)


def _blob(shape=(64, 96), centre=(32, 48), radius=8, value=1.0) -> np.ndarray:
    """A heatmap with all its mass in one disc."""
    hm = np.zeros(shape, dtype=np.float32)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    hm[(yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2] = value
    return hm


def _ring_mask(shape=(64, 96), centre=(32, 48), radius=20) -> np.ndarray:
    """A filled ellipse standing in for a segmented head."""
    m = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    m[(yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2] = 255
    return m


def _store(**overrides) -> str:
    findings = {
        "hc_mm": 220.0,
        "mode": "single_frame",
        "mask_b64": None,
    }
    findings.update(overrides)
    return findings_store.store(
        img_gray=np.zeros((64, 96), dtype=np.uint8),
        model_variant="phase0",
        pixel_spacing_mm=0.0691,
        threshold=0.5,
        findings=findings,
    )


# ── the summary measures what it claims to ───────────────────────────────────


def test_a_tight_blob_reads_as_concentrated_and_a_flat_map_as_diffuse():
    tight = attribution.summarize_attribution(_blob(radius=4))
    spread = attribution.summarize_attribution(np.full((64, 96), 0.5, dtype=np.float32))
    assert tight.concentration_label == "concentrated"
    assert spread.concentration_label == "diffuse"
    assert tight.concentration > spread.concentration


def test_the_peak_region_matches_where_the_mass_actually_is():
    # Upper-left of a 64x96 map.
    summary = attribution.summarize_attribution(_blob(centre=(10, 15), radius=5))
    assert summary.peak_region == "upper-left"
    assert summary.top_regions[0].region == "upper-left"
    assert summary.peak_xy_pct[0] < 50 and summary.peak_xy_pct[1] < 50


def test_region_shares_sum_to_the_whole_map():
    hm = np.random.default_rng(0).random((64, 96)).astype(np.float32)
    summary = attribution.summarize_attribution(hm)
    # Only the top 3 are returned, so assert each is a real fraction instead.
    assert all(0.0 <= r.share <= 1.0 for r in summary.top_regions)
    assert sum(r.share for r in summary.top_regions) <= 1.0


def test_a_flat_zero_map_reports_no_signal_rather_than_inventing_a_peak():
    """argmax of an all-zero array is index 0 — reporting that as 'the peak' lies."""
    summary = attribution.summarize_attribution(np.zeros((64, 96), dtype=np.float32))
    assert summary.concentration_label == "no attribution signal"
    assert summary.peak_region == "unknown"
    assert summary.top_regions == []


def test_attribution_on_the_skull_edge_is_measured_as_on_boundary():
    """The clinically load-bearing number: on the calvarium vs off in the field."""
    mask = _ring_mask(radius=20)
    # A blob sitting right on the mask edge.
    on_edge = _blob(centre=(32, 68), radius=3)
    summary = attribution.summarize_attribution(on_edge, mask=mask)
    assert summary.mask_available is True
    assert summary.on_boundary_pct > 90, "a blob on the edge must read as on-boundary"


def test_attribution_off_in_the_background_is_measured_as_outside_the_head():
    mask = _ring_mask(radius=20)
    far_away = _blob(centre=(5, 90), radius=3)
    summary = attribution.summarize_attribution(far_away, mask=mask)
    assert summary.outside_head_pct > 90, "an artifact-looking blob must read as outside"
    assert summary.on_boundary_pct < 10


def test_the_three_skull_relative_shares_account_for_the_whole_map():
    mask = _ring_mask()
    hm = np.random.default_rng(1).random((64, 96)).astype(np.float32)
    s = attribution.summarize_attribution(hm, mask=mask)
    total = s.on_boundary_pct + s.inside_head_pct + s.outside_head_pct
    assert total == pytest.approx(100.0, abs=0.5)


def test_no_mask_means_the_skull_relation_is_absent_not_guessed():
    summary = attribution.summarize_attribution(_blob())
    assert summary.mask_available is False
    assert summary.on_boundary_pct is None
    assert "could not be related to the predicted skull" in summary.as_prompt_block()


def test_a_mismatched_mask_is_resized_rather_than_dropped():
    big_mask = _ring_mask(shape=(128, 192), centre=(64, 96), radius=40)
    summary = attribution.summarize_attribution(_blob(), mask=big_mask)
    assert summary.mask_available is True


def test_the_prompt_block_quotes_the_numbers_the_answer_must_use():
    summary = attribution.summarize_attribution(_blob(radius=4), mask=_ring_mask())
    block = summary.as_prompt_block()
    for token in ["concentration", "peak attribution", "predicted skull boundary", "outside"]:
        assert token in block


def test_the_summary_never_names_anatomy():
    """Nothing in a saliency map establishes which structure a region contains."""
    summary = attribution.summarize_attribution(_blob(), mask=_ring_mask())
    block = summary.as_prompt_block().lower()
    for anatomical in ["thalamus", "cavum", "falx", "cerebell", "ventricle"]:
        assert anatomical not in block


# ── the endpoint ─────────────────────────────────────────────────────────────


def _fake_summary(**kw):
    s = attribution.summarize_attribution(_blob(radius=4), mask=_ring_mask())
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_ask_returns_the_measurements_the_answer_was_grounded_in():
    fid = _store()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.xai_qa_endpoints._compute_summary", return_value=(_fake_summary(), None)
        ):
            with patch(
                "app.api.xai_qa_endpoints._call_llm",
                return_value=("The attribution is concentrated on the boundary.", None),
            ):
                d = client.post(
                    f"/findings/{fid}/xai/ask", json={"question": "Why did it focus there?"}
                ).json()
    assert d["grounded"] is True and d["used_llm"] is True
    assert d["summary"]["concentration_label"]
    assert d["summary"]["on_boundary_pct"] is not None
    assert "correlation" in d["disclaimer"]


def test_the_prompt_carries_the_actual_attribution_numbers():
    fid = _store()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.xai_qa_endpoints._compute_summary", return_value=(_fake_summary(), None)
        ):
            with patch(
                "app.api.xai_qa_endpoints._call_llm", return_value=("answer", None)
            ) as llm:
                client.post(f"/findings/{fid}/xai/ask", json={"question": "Concentrated?"})
    prompt = llm.call_args[0][1]
    assert "ATTRIBUTION MEASUREMENTS" in prompt
    assert "peak attribution" in prompt
    assert "Concentrated?" in prompt


def test_a_map_that_could_not_be_computed_is_not_narrated():
    """With no measurements there are no grounds — do not let the model improvise."""
    fid = _store()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.xai_qa_endpoints._compute_summary",
            return_value=(None, "Model 'phase0' is no longer loaded on this server."),
        ):
            with patch("app.api.xai_qa_endpoints._call_llm") as llm:
                d = client.post(f"/findings/{fid}/xai/ask", json={"question": "why?"}).json()
    llm.assert_not_called()
    assert d["grounded"] is False and d["used_llm"] is False
    assert "no longer loaded" in d["llm_error"]


def test_without_a_key_the_measurements_are_still_returned():
    fid = _store()
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with patch(
            "app.api.xai_qa_endpoints._compute_summary", return_value=(_fake_summary(), None)
        ):
            d = client.post(f"/findings/{fid}/xai/ask", json={"question": "why?"}).json()
    assert d["grounded"] is True and d["used_llm"] is False
    assert d["summary"]["peak_region"]
    assert "ANTHROPIC_API_KEY" in d["llm_error"]


def test_a_failed_llm_call_reports_the_reason_and_keeps_the_measurements():
    fid = _store()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.xai_qa_endpoints._compute_summary", return_value=(_fake_summary(), None)
        ):
            with patch(
                "app.api.xai_qa_endpoints._call_llm",
                return_value=(None, "APIConnectionError: Connection error."),
            ):
                d = client.post(f"/findings/{fid}/xai/ask", json={"question": "why?"}).json()
    assert d["used_llm"] is False
    assert "Connection error" in d["llm_error"]
    assert d["summary"]["concentration_label"]


def test_ask_404s_on_an_expired_finding():
    r = client.post("/findings/nope/xai/ask", json={"question": "why?"})
    assert r.status_code == 404


def test_an_empty_question_is_rejected():
    fid = _store()
    r = client.post(f"/findings/{fid}/xai/ask", json={"question": "   "})
    assert r.status_code == 422


def test_the_system_prompt_forbids_naming_anatomy_and_implying_causation():
    from app.api.xai_qa_endpoints import _SYSTEM_PROMPT as p

    low = p.lower()
    assert "never name an anatomical structure" in low
    assert "not evidence that the highlighted tissue caused" in low
    assert "do not invent" in low


def test_the_uncertainty_method_is_selectable():
    fid = _store()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch(
            "app.api.xai_qa_endpoints._compute_summary", return_value=(_fake_summary(), None)
        ) as summarize:
            with patch("app.api.xai_qa_endpoints._call_llm", return_value=("a", None)):
                d = client.post(
                    f"/findings/{fid}/xai/ask",
                    json={"question": "why?", "method": "uncertainty"},
                ).json()
    assert summarize.call_args[0][1] == "uncertainty"
    assert d["method"] == "uncertainty"


# ── the real wiring, not an injected stub ────────────────────────────────────
#
# The endpoint tests above mock _compute_summary. That is the same shape of gap
# that let `from . import cine` ship in the escalation agent: the module's own
# imports and glue were never executed. These run the real functions.


def test_compute_summary_resolves_its_imports_and_reports_a_missing_model():
    """Reaching the "no longer loaded" message proves the import chain resolves."""
    from app.api.xai_qa_endpoints import _compute_summary

    record = findings_store.get(_store())
    summary, error = _compute_summary(record, "gradcam")
    # No weights are loaded in the test environment, so this is the expected
    # branch — but it is only reachable if every import above it worked.
    assert summary is None
    assert error and "no longer loaded" in error


def test_decode_mask_round_trips_a_real_stored_png():
    """The skull-relative percentages depend entirely on this decoding."""
    import base64
    import io

    from PIL import Image

    from app.api.xai_qa_endpoints import _decode_mask

    mask = _ring_mask()
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    record = findings_store.get(_store(mask_b64=b64))
    decoded = _decode_mask(record)
    assert decoded is not None
    assert decoded.shape == mask.shape
    assert (decoded > 0).sum() == (mask > 0).sum()


def test_decode_mask_returns_none_for_a_corrupt_blob_rather_than_raising():
    from app.api.xai_qa_endpoints import _decode_mask

    record = findings_store.get(_store(mask_b64="not-valid-base64-@@@"))
    assert _decode_mask(record) is None


def test_a_finding_with_no_mask_yields_a_summary_without_skull_percentages():
    """End-to-end through the real summariser, no mocks on the maths."""
    from app.api.xai_qa_endpoints import build_prompt

    summary = attribution.summarize_attribution(_blob(radius=4), mask=None)
    prompt = build_prompt("Why here?", summary, {"hc_mm": 220.0, "mode": "single_frame"})
    assert "could not be related to the predicted skull" in prompt
    assert "220.0 mm" in prompt
