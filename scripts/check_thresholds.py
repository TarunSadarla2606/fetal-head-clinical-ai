#!/usr/bin/env python3
"""Enforce the release gates against an eval run. Exit 1 blocks the deploy.

Kept separate from ``evaluate.py`` so the two failure modes stay
distinguishable: a non-zero exit here means the model regressed, a non-zero
exit there means the evaluation could not run. Conflating them would make a
crashed harness look like a bad model, or worse, the reverse.

Usage::

    python scripts/check_thresholds.py eval_results.json
    python scripts/check_thresholds.py eval_results.json --require phase0 phase2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_thresholds import KNOWN_ISSUES, gates_for  # noqa: E402

COMPARATORS = {
    "<=": lambda value, bound: value <= bound,
    ">=": lambda value, bound: value >= bound,
}


def check_variant(name: str, entry: dict) -> list[str]:
    """Return a list of failure messages for one variant. Empty means pass."""
    failures: list[str] = []

    if not entry.get("loaded"):
        return [f"{name}: checkpoint did not load — {entry.get('error') or 'no reason given'}"]

    metrics = entry.get("metrics") or {}
    for metric, (op, bound) in gates_for(name).items():
        value = metrics.get(metric)
        if value is None:
            # An unmeasurable metric is a failure, not a skip. A gate that
            # passes because nothing was measured protects nothing.
            failures.append(f"{name}: {metric} was not measured, so the gate cannot be evaluated")
            continue
        if not COMPARATORS[op](value, bound):
            failures.append(f"{name}: {metric} = {value}, required {op} {bound}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="Path to evaluate.py's JSON output")
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help="Variants that must be present and pass. Defaults to every variant "
        "in the report that loaded; pass explicitly in CI to catch a silently "
        "missing checkpoint.",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.results).read_text())
    variants = report.get("variants") or {}

    required = args.require if args.require is not None else sorted(variants)
    if not required:
        print("::error::No variants were evaluated — refusing to pass a vacuous gate.")
        return 1

    all_failures: list[str] = []
    for name in required:
        entry = variants.get(name)
        if entry is None:
            all_failures.append(f"{name}: required variant is absent from the eval report")
            continue
        failures = check_variant(name, entry)
        metrics = entry.get("metrics") or {}
        status = "FAIL" if failures else "PASS"
        print(
            f"[{status}] {name}: MAE {metrics.get('hc_mae_mm')} mm, "
            f"max error {metrics.get('hc_max_error_mm')} mm, "
            f"measurable {metrics.get('measurable_rate')}, "
            f"ISUOG pass {metrics.get('isuog_pass_rate')}"
        )
        all_failures.extend(failures)

    print()
    for name in required:
        for gate, (op, bound) in sorted(gates_for(name).items()):
            print(f"gate[{name}]: {gate} {op} {bound}")
        if name in KNOWN_ISSUES:
            print(f"note[{name}]: {KNOWN_ISSUES[name]}")

    if all_failures:
        print()
        for failure in all_failures:
            print(f"::error::{failure}")
        print(f"\n{len(all_failures)} gate failure(s) — deployment blocked.")
        return 1

    print(f"\nAll gates passed for: {', '.join(required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
