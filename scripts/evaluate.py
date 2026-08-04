#!/usr/bin/env python3
"""Evaluate the checkpoints on the demo subjects and emit metrics as JSON.

Runs the *deployed* inference path — the same ``predict_single_frame`` /
``predict_cine_clip`` the API calls — so a regression in preprocessing, the
ellipse fit or the pixel-spacing handling trips the gate, not just a change in
model weights.

Ground truth is the head circumference in ``training_set_pixel_size_and_HC.csv``
(HC18 reference annotations). There are no ground-truth *masks* in this
repository, so Dice is reported as null with a reason rather than invented.

Usage::

    python scripts/evaluate.py --out eval_results.json
    python scripts/evaluate.py --variants phase0 phase4a --out results.json

Exit status is 0 whenever the evaluation itself ran; threshold enforcement is
``scripts/check_thresholds.py``, kept separate so a failing gate is
distinguishable from a crashed evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Repo root only. Putting app/ on the path makes `import app` resolve to the
# Streamlit module app/app.py instead of the package, which drags in streamlit.
sys.path.insert(0, str(REPO_ROOT))

ISUOG_THRESHOLD_MM = 3.0
DEFAULT_VARIANTS = ["phase0", "phase4a", "phase2", "phase4b"]


def load_ground_truth() -> dict[str, dict[str, float]]:
    """filename -> {pixel_spacing_mm, hc_reference_mm} from the HC18 CSV."""
    csv_path = REPO_ROOT / "training_set_pixel_size_and_HC.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found at {csv_path}")

    truth: dict[str, dict[str, float]] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("filename") or "").strip()
            try:
                truth[name] = {
                    "pixel_spacing_mm": float(row["pixel size(mm)"]),
                    "hc_reference_mm": float(row["head circumference (mm)"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return truth


def evaluate_variant(variant: str, truth: dict[str, dict[str, float]]) -> dict:
    """Run one checkpoint over every demo subject that has a reference HC."""
    from app.api import model_manager

    result: dict = {
        "variant": variant,
        "n_images": 0,
        "per_image": [],
        "loaded": False,
        "error": None,
    }

    model = model_manager.get_model(variant)
    if model is None:
        # Reported, never treated as a pass. CI decides what to do about a
        # checkpoint it could not load; this script does not decide for it.
        result["error"] = (
            f"Checkpoint '{variant}' could not be loaded. Its weight file is "
            "missing or the WEIGHT_* environment variable is unset."
        )
        return result
    result["loaded"] = True

    from app import cine
    from app.api import inference_wrapper
    from app.inference import N_FRAMES, TemporalFetaSegNet

    demo_dir = REPO_ROOT / "demo_subjects"
    is_temporal = isinstance(model, TemporalFetaSegNet)

    for path in sorted(demo_dir.glob("*.png")):
        gt = truth.get(path.name)
        if gt is None:
            continue

        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            result["per_image"].append({"file": path.name, "error": "unreadable"})
            continue

        started = time.perf_counter()
        try:
            if is_temporal:
                frames = cine.synthesize_cine_frames(img, n_frames=N_FRAMES)
                pred = inference_wrapper.predict_cine_clip(
                    model=model,
                    frames=frames,
                    pixel_spacing_mm=gt["pixel_spacing_mm"],
                    threshold=0.5,
                )
            else:
                pred = inference_wrapper.predict_single_frame(
                    model=model,
                    img_gray=img,
                    pixel_spacing_mm=gt["pixel_spacing_mm"],
                    threshold=0.5,
                )
        except Exception as exc:  # noqa: BLE001 — one bad image must not lose the run
            result["per_image"].append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        latency_ms = (time.perf_counter() - started) * 1000

        hc = pred.get("hc_mm")
        entry = {
            "file": path.name,
            "hc_pred_mm": round(float(hc), 2) if hc is not None else None,
            "hc_reference_mm": round(gt["hc_reference_mm"], 2),
            "abs_error_mm": round(abs(float(hc) - gt["hc_reference_mm"]), 2)
            if hc is not None
            else None,
            "latency_ms": round(latency_ms, 1),
        }
        result["per_image"].append(entry)

    result["n_images"] = len(result["per_image"])
    return result


def summarise(run: dict) -> dict:
    """Aggregate per-image results into the gated metrics."""
    from scripts.eval_thresholds import DICE_UNAVAILABLE_REASON

    measured = [e for e in run["per_image"] if e.get("abs_error_mm") is not None]
    errors = [e["abs_error_mm"] for e in measured]
    latencies = [e["latency_ms"] for e in run["per_image"] if e.get("latency_ms") is not None]
    n = len(run["per_image"])

    metrics: dict = {
        "n_images": n,
        "n_measured": len(measured),
        "measurable_rate": round(len(measured) / n, 4) if n else 0.0,
        "hc_mae_mm": round(sum(errors) / len(errors), 3) if errors else None,
        "hc_max_error_mm": round(max(errors), 3) if errors else None,
        "isuog_pass_rate": round(sum(1 for e in errors if e <= ISUOG_THRESHOLD_MM) / len(errors), 4)
        if errors
        else None,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        # Explicitly null, with the reason attached. An omitted key reads as an
        # oversight; a null with a reason is a statement about the data.
        "dice": None,
        "dice_unavailable_reason": DICE_UNAVAILABLE_REASON,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval_results.json", help="Where to write the JSON")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = parser.parse_args()

    truth = load_ground_truth()
    print(f"Ground truth: {len(truth)} annotated images in the HC18 CSV", file=sys.stderr)

    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "isuog_threshold_mm": ISUOG_THRESHOLD_MM,
        "ground_truth_source": "training_set_pixel_size_and_HC.csv (HC18 reference HC)",
        "variants": {},
    }

    for variant in args.variants:
        print(f"Evaluating {variant} ...", file=sys.stderr)
        run = evaluate_variant(variant, truth)
        entry = {"loaded": run["loaded"], "error": run["error"], "per_image": run["per_image"]}
        entry["metrics"] = summarise(run) if run["loaded"] else {}
        report["variants"][variant] = entry
        if run["loaded"]:
            m = entry["metrics"]
            print(
                f"  {variant}: MAE {m['hc_mae_mm']} mm over {m['n_measured']}/{m['n_images']} "
                f"images, ISUOG pass {m['isuog_pass_rate']}",
                file=sys.stderr,
            )
        else:
            print(f"  {variant}: {run['error']}", file=sys.stderr)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
