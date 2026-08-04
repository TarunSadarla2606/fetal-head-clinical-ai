"""Release-gate logic — what CI will and will not allow to ship.

These are tests about failure direction. A gate that errs toward passing is
worse than no gate, because it converts "we did not check" into "we checked and
it was fine".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_thresholds import check_variant  # noqa: E402
from scripts.eval_thresholds import (  # noqa: E402
    DEFAULT_GATES,
    DICE_UNAVAILABLE_REASON,
    KNOWN_ISSUES,
    VARIANT_GATES,
    gates_for,
)
from scripts.render_model_card import BEGIN, END, render_block, splice  # noqa: E402


def _entry(**metrics) -> dict:
    base = {
        "hc_mae_mm": 4.0,
        "hc_max_error_mm": 15.0,
        "measurable_rate": 1.0,
        "mean_latency_ms": 900.0,
    }
    base.update(metrics)
    return {"loaded": True, "error": None, "metrics": base, "per_image": []}


# ── the gate's failure direction ─────────────────────────────────────────────


def test_a_healthy_variant_passes():
    assert check_variant("phase4b", _entry()) == []


def test_a_regressed_mae_fails():
    failures = check_variant("phase4b", _entry(hc_mae_mm=99.0))
    assert failures and "hc_mae_mm" in failures[0]


def test_one_blown_case_fails_even_when_the_mean_looks_fine():
    """A good mean can hide a catastrophic single image — that is the point of the max."""
    failures = check_variant("phase4b", _entry(hc_mae_mm=4.0, hc_max_error_mm=500.0))
    assert any("hc_max_error_mm" in f for f in failures)


def test_a_checkpoint_that_did_not_load_fails_rather_than_being_skipped():
    entry = {"loaded": False, "error": "weight file missing", "metrics": {}}
    failures = check_variant("phase0", entry)
    assert failures and "did not load" in failures[0]


def test_an_unmeasured_metric_fails_rather_than_passing_vacuously():
    """None must never satisfy a bound. 'We could not measure it' is not 'it is fine'."""
    failures = check_variant("phase4b", _entry(hc_mae_mm=None))
    assert any("was not measured" in f for f in failures)


def test_a_model_that_measures_nothing_fails():
    failures = check_variant("phase4b", _entry(measurable_rate=0.5))
    assert any("measurable_rate" in f for f in failures)


# ── per-variant overrides ────────────────────────────────────────────────────


def test_overrides_apply_only_to_their_own_variant():
    assert gates_for("phase4a")["hc_mae_mm"][1] == VARIANT_GATES["phase4a"]["hc_mae_mm"][1]
    assert gates_for("phase4b")["hc_mae_mm"][1] == DEFAULT_GATES["hc_mae_mm"][1]


def test_an_override_still_inherits_the_ungiven_defaults():
    """A loosened MAE bound must not quietly drop the measurable-rate gate."""
    gates = gates_for("phase4a")
    assert gates["measurable_rate"] == DEFAULT_GATES["measurable_rate"]
    assert set(gates) == set(DEFAULT_GATES)


def test_every_known_issue_names_a_variant_that_has_an_override():
    """A documented exception with no corresponding gate change is a stale note."""
    for variant in KNOWN_ISSUES:
        assert variant in VARIANT_GATES, f"{variant} is documented but not actually excepted"


def test_the_phase4a_gate_admits_its_measured_value_and_rejects_the_old_one():
    """Tightened after the fragmented-ring fix took phase4a to 11.8 mm.

    The pre-fix 32.4 mm must now FAIL — a gate that still admits the broken
    behaviour has not been tightened, it has just been relabelled.
    """
    assert check_variant("phase4a", _entry(hc_mae_mm=11.8, hc_max_error_mm=39.4)) == []
    assert check_variant("phase4a", _entry(hc_mae_mm=32.4, hc_max_error_mm=162.0)) != []


def test_the_phase0_gate_is_still_the_pre_fix_bound_and_says_so():
    """Left wide on purpose until CI measures it; the note must admit that."""
    assert check_variant("phase0", _entry(hc_mae_mm=41.9, hc_max_error_mm=208.0)) == []
    assert "not yet known" in KNOWN_ISSUES["phase0"]


# ── the CLI, end to end ──────────────────────────────────────────────────────


def _run_gate(tmp_path: Path, report: dict, *args: str) -> subprocess.CompletedProcess:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(report))
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_thresholds.py"), str(path), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_the_cli_exits_nonzero_on_a_regression(tmp_path):
    result = _run_gate(tmp_path, {"variants": {"phase4b": _entry(hc_mae_mm=99.0)}})
    assert result.returncode == 1
    assert "::error::" in result.stdout


def test_the_cli_exits_zero_when_everything_passes(tmp_path):
    result = _run_gate(tmp_path, {"variants": {"phase4b": _entry()}})
    assert result.returncode == 0


def test_an_empty_report_is_refused_rather_than_passing(tmp_path):
    """Zero variants trivially satisfies 'all variants pass'. Refuse it."""
    result = _run_gate(tmp_path, {"variants": {}})
    assert result.returncode == 1
    assert "vacuous" in result.stdout


def test_a_required_variant_missing_from_the_report_fails(tmp_path):
    """Catches a checkpoint that silently stopped being evaluated."""
    result = _run_gate(tmp_path, {"variants": {"phase4b": _entry()}}, "--require", "phase0")
    assert result.returncode == 1
    assert "absent from the eval report" in result.stdout


# ── model card rendering ─────────────────────────────────────────────────────


def _report() -> dict:
    return {
        "generated_at": "2026-08-04T12:00:00+00:00",
        "variants": {
            "phase4b": {**_entry(), "metrics": {**_entry()["metrics"], "n_images": 12,
                                                "isuog_pass_rate": 0.5,
                                                "dice": None,
                                                "dice_unavailable_reason": DICE_UNAVAILABLE_REASON}},
        },
    }


def test_the_rendered_block_carries_the_actual_numbers():
    block = render_block(_report())
    assert "4" in block and "12 demo subjects" in block
    assert BEGIN in block and END in block


def test_the_card_states_why_there_is_no_dice_figure():
    """An omitted metric reads as an oversight; a stated absence is a claim."""
    block = render_block(_report())
    assert "no ground-truth segmentation masks" in block.lower()


def test_a_variant_that_did_not_load_is_named_as_such_not_left_blank():
    report = _report()
    report["variants"]["phase0"] = {"loaded": False, "error": "missing", "metrics": {}}
    assert "checkpoint not loaded" in render_block(report)


def test_splice_refuses_a_card_without_markers():
    """Better to fail than to overwrite hand-written limitations prose."""
    with pytest.raises(ValueError, match="markers"):
        splice("# Card\n\nNo markers here.\n", "block")


def test_splice_preserves_everything_outside_the_markers():
    card = f"HEAD\n{BEGIN}\nold\n{END}\nTAIL\n"
    out = splice(card, f"{BEGIN}\nnew\n{END}")
    assert out.startswith("HEAD") and out.endswith("TAIL\n")
    assert "old" not in out and "new" in out


def test_the_real_model_card_has_the_markers_and_the_disclaimer():
    card = (REPO_ROOT / "MODEL_CARD.md").read_text()
    assert BEGIN in card and END in card
    assert "Not FDA-cleared" in card
    assert "Not for clinical use" in card
    # The limitations that matter most are hand-written, not generated.
    # Normalised because the source wraps these sentences across lines.
    flat = " ".join(card.split())
    assert "placeholders (1.0 and 0.0), not measurements" in flat
    assert "correlation with the model's own prediction" in flat
    assert "no ground-truth segmentation masks ship with this repository" in flat.lower()


# ── the workflow must not swallow the gate's verdict ─────────────────────────
#
# The first live run measured phase0 at 41.9 mm MAE against a 12 mm bound and
# deployed anyway: the step ran `check_thresholds.py | tee gate.log`, and a
# pipeline's exit status is the last command's, so tee's 0 masked the gate's 1.
# `bash -e` does not cover that. A gate that cannot fail is decorative.


def _gate_step() -> dict:
    import yaml

    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "deploy-hf.yml").read_text())
    for step in workflow["jobs"]["evaluate"]["steps"]:
        if "check_thresholds.py" in (step.get("run") or ""):
            return step
    raise AssertionError("no step runs check_thresholds.py — the gate is not wired in")


def test_the_gate_step_sets_pipefail_because_it_pipes():
    run = _gate_step()["run"]
    # Comments are stripped before matching: an earlier version of this test
    # passed against the *unfixed* workflow because the explanatory comment
    # above the fix contains the word "pipefail". Matching on prose rather than
    # on the executed line is how a test ends up unable to fail.
    code = "\n".join(
        line for line in run.splitlines() if not line.strip().startswith("#")
    )
    if "|" in code:
        assert any(
            directive in code for directive in ("set -o pipefail", "set -euo pipefail")
        ), (
            "the gate's exit code is piped into another command; without "
            "pipefail the pipeline reports that command's status and every "
            "regression passes"
        )


def test_the_deploy_job_depends_on_the_gate():
    import yaml

    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "deploy-hf.yml").read_text())
    needs = workflow["jobs"]["deploy"].get("needs")
    needs = [needs] if isinstance(needs, str) else (needs or [])
    assert "evaluate" in needs, "deploy must not run unless the gate passed"


def test_the_gate_requires_every_variant_explicitly():
    """Otherwise a checkpoint that stops being evaluated shrinks the gate."""
    run = _gate_step()["run"]
    assert "--require" in run
    for variant in ("phase0", "phase4a", "phase2", "phase4b"):
        assert variant in run, f"{variant} is not named in --require"
