# This system's own metrics — definitions

> Scope: what each number this system reports actually means. Unlike the
> guideline files, everything here is derived directly from the source in this
> repository and is accurate as written — no TODO(verbatim) placeholders.

## How head circumference is computed from the segmentation

> Source: `app/inference.py::estimate_hc_mm`.

The model outputs a binary segmentation of the calvarium. The largest connected
component is taken, an ellipse is fitted to it, and the ellipse perimeter is
computed with **Ramanujan's approximation** from the semi-major and semi-minor
axes. That perimeter, in pixels, is converted to millimetres using the supplied
pixel spacing and a scale factor correcting for the difference between the
network input width and the original image width. If no component is found, HC
is reported as unavailable rather than guessed.

## Temporal reliability score — definition

> Source: `app/inference.py::predict_cine_clip`.

In cine mode the temporal model produces a segmentation for each of the 16
frames, and an HC is measured on each frame independently. The reliability
score is

    reliability = max(0, 1 − standard_deviation(per_frame_HC) / mean(per_frame_HC))

that is, one minus the coefficient of variation of HC across the loop. A value
near 1.0 means every frame produced almost the same measurement.

## What the reliability score does and does not tell you

> Source: derived from the definition in `app/inference.py::predict_cine_clip`.

Reliability measures **self-consistency across frames only**. It is a precision
statistic, not an accuracy statistic. A model that segments the wrong structure
identically on all 16 frames scores near 1.0. High reliability therefore means
the measurement is stable, not that it is correct — it cannot substitute for
visual confirmation that the contour follows the calvarium. Conversely, low
reliability is informative: it indicates the boundary moved between frames and
the measurement should be verified manually.

## Confidence labels and their thresholds

> Source: `app/inference.py::confidence_label`.

The reliability score is banded into three labels: **HIGH CONFIDENCE** at 0.97
and above, **MODERATE CONFIDENCE** from 0.92 up to 0.97, and **LOW — VERIFY
MANUALLY** below 0.92. These thresholds are this project's own convention, set
against observed behaviour on the HC18 dataset. They are not a clinical
standard and carry no regulatory meaning.

## Single-frame mode reports a reliability of 1.0 that means nothing

> Source: `app/inference.py::predict_single_frame`.

The static variants (`phase0`, `phase4a`) process one frame, so there is no
inter-frame spread to measure. They return a hard-coded reliability of 1.0 and
a standard deviation of 0.0. This is a placeholder, not a measurement of
quality — a single-frame result carries no reliability information at all, and
its 1.0 must not be read as higher confidence than a cine result scoring 0.96.

## HC mean absolute error (MAE)

> Source: project evaluation on the HC18 test set; see README model table.

MAE is the mean of the absolute differences between predicted HC and reference
HC across the test set, in millimetres. It summarises typical error magnitude
but says nothing about the distribution's tail — a model with a good MAE can
still fail badly on individual atypical images. It is a dataset-level
aggregate and does not describe the error on any one measurement.

## Dice coefficient

> Source: standard segmentation metric; project evaluation on the HC18 test set.

Dice measures spatial overlap between the predicted and reference masks:
twice the intersection area divided by the sum of the two areas, ranging 0 to 1.
It is dominated by the interior of a large filled region, so for a structure
like the calvarium a high Dice can coexist with a boundary that is
systematically offset by a small distance. Because HC is a **perimeter**
measurement, boundary placement matters more to HC accuracy than Dice suggests,
which is why HC MAE is reported alongside it.

## Reported model performance on the HC18 test set

> Source: project README model-variants table; HC18 test set (Radboud UMC).

Four variants are available. Phase 0 (static baseline) reports 97.75% Dice and
1.65 mm MAE. Phase 4a (compressed static) reports 97.64% Dice and 1.76 mm MAE.
Phase 2 (temporal baseline) reports 95.95% Dice and 2.10 mm MAE. Phase 4b
(compressed temporal) reports 96.00% Dice and 2.06 mm MAE. These are test-set
aggregates, not guarantees for any individual image.

## What the out-of-distribution (OOD) flag checks

> Source: `app/inference.py::validate_input` and `app/api/xai_endpoints.py::analyze_ood`.

The OOD flag is raised by heuristic checks on the input image — resolution,
blankness, saturation, dynamic range, aspect ratio and texture presence — not
by a learned novelty detector. It catches images that are obviously unlike
training data, such as a blank frame, a screenshot, or a non-ultrasound
photograph. It will not catch a genuine ultrasound image of the wrong anatomy
or the wrong plane, so a clear OOD result is not confirmation that the input
was appropriate.

## The cine loop is synthesised, not acquired

> Source: `app/cine.py::synthesize_cine_frames`.

This system accepts a single static frame. In cine mode the 16-frame loop is
**generated from that one frame** by Pseudo-LDDM v2 — Ornstein-Uhlenbeck probe
motion, Rician speckle and depth attenuation applied as synthetic
perturbations. No real temporal information is present. The reliability score
in cine mode therefore measures robustness to these synthetic perturbations,
which is a weaker claim than stability across a genuine acquired cine sweep.

## Image quality score

> Source: `app/inference.py::_compute_quality_score`.

The quality score is a weighted composite in the range 0–1: blur measured by
Laplacian variance contributes 60%, intensity standard deviation as a contrast
proxy 20%, mean-intensity distance from mid-grey 10%, and short-axis resolution
10%. It is banded as poor below 0.35, suboptimal below 0.60, good below 0.85,
and excellent at or above 0.85. These weights and bands are this project's
heuristic, not a validated image-quality standard.
