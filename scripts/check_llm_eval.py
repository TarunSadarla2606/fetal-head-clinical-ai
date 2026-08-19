#!/usr/bin/env python3
"""Enforce the LLM answer-quality gates. Exit 1 blocks the build.

Separate from ``eval_llm.py`` for the same reason ``check_thresholds.py`` is
separate from ``evaluate.py``: a non-zero exit here means answer quality
regressed, a non-zero exit there means the harness broke. Conflating them makes
a crashed harness look like a bad model.

Judged metrics are null without an API key. Those are reported as **skipped**,
never as passed — but the deterministic gates always run, so a build with no key
still catches invented citations and answers to off-topic questions.

Usage::

    python scripts/check_llm_eval.py llm_eval_results.json
    python scripts/check_llm_eval.py llm_eval_results.json --require-judged
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.llm_eval_thresholds import DETERMINISTIC_GATES, GATES  # noqa: E402

COMPARATORS = {
    "<=": lambda value, bound: value <= bound,
    ">=": lambda value, bound: value >= bound,
}


def evaluate_gates(report: dict, require_judged: bool) -> tuple[list[str], list[str]]:
    """Return (failures, skipped)."""
    metrics = report.get("metrics") or {}
    failures: list[str] = []
    skipped: list[str] = []

    for metric, (op, bound) in sorted(GATES.items()):
        value = metrics.get(metric)
        deterministic = metric in DETERMINISTIC_GATES

        if value is None:
            if deterministic:
                # These need no model, so a null here means the harness did not
                # run — a failure, not an absence of signal.
                failures.append(
                    f"{metric} was not measured, but it needs no API key — the "
                    "eval harness did not run properly"
                )
            elif require_judged:
                failures.append(
                    f"{metric} requires a judge and was not measured. "
                    f"{report.get('judge_unavailable_reason') or 'No reason given.'}"
                )
            else:
                skipped.append(f"{metric} (no API key — judged metrics unavailable)")
            continue

        if not COMPARATORS[op](value, bound):
            failures.append(f"{metric} = {value}, required {op} {bound}")

    return failures, skipped


def _report_offenders(report: dict) -> None:
    """Print what actually went wrong, not just which gate moved."""
    for entry in report.get("grounded_answers", []):
        if entry.get("invented_citations"):
            print(f"\n  invented citation(s) in answer to {entry['question']!r}:")
            for label in entry["invented_citations"]:
                print(f"    [{label}] matches no retrieved chunk")
        if entry.get("supported") is False:
            print(f"\n  unsupported claim(s) in answer to {entry['question']!r}:")
            for claim in entry.get("unsupported_claims") or []:
                print(f"    {claim}")

    for entry in report.get("adversarial", []):
        if entry.get("refused") is False:
            print(f"\n  ADVERSARIAL PROMPT NOT REFUSED: {entry['prompt']!r}")
            print(f"    judge: {entry.get('reason')}")
            print(f"    answer: {(entry.get('answer') or '')[:300]}")

    for entry in report.get("off_topic", []):
        if not entry.get("declined"):
            print(f"\n  off-topic question was answered: {entry['question']!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument(
        "--require-judged",
        action="store_true",
        help="Fail if the judged metrics are missing, rather than skipping them. "
        "Use where ANTHROPIC_API_KEY is expected to be present.",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.results).read_text())
    metrics = report.get("metrics") or {}

    print(
        f"judged: {report.get('judged')} | questions {metrics.get('n_questions')} | "
        f"judge errors {metrics.get('judge_errors')}"
    )
    for metric, (op, bound) in sorted(GATES.items()):
        value = metrics.get(metric)
        tag = "det" if metric in DETERMINISTIC_GATES else "judged"
        print(f"  [{tag:>6}] {metric}: {value}  (required {op} {bound})")

    failures, skipped = evaluate_gates(report, args.require_judged)
    _report_offenders(report)

    for note in skipped:
        print(f"\n::warning::skipped {note}")

    if failures:
        print()
        for failure in failures:
            print(f"::error::{failure}")
        print(f"\n{len(failures)} gate failure(s).")
        return 1

    print("\nAll enforced gates passed." + (" Judged gates skipped." if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
