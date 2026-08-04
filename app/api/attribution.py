"""Turn a saliency heatmap into structured, text-friendly signal.

A rendered GradCAM PNG cannot be reasoned over — an LLM handed only the image
caption will produce plausible anatomy-flavoured prose with nothing behind it.
This module reduces the raw attribution array to a compact set of measured
facts so an explanation can be grounded in what the map actually contains.

The load-bearing measurement is **where the attribution sits relative to the
predicted skull**. "The model looked at the upper left" is trivia; "82% of the
attribution mass falls on the predicted calvarial boundary" is the difference
between an anatomically plausible explanation and an artifact, which is the
question a clinician actually asks of a saliency map.

Deliberately NOT computed: named anatomical structures. Nothing here can
establish that a hot region is the cavum septi pellucidi rather than a bright
speckle artifact, and a summary that named structures would invite exactly the
confident, unfalsifiable prose this design exists to prevent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# Fraction of peak attribution above which a pixel counts as "focused on".
# 0.5 of the normalised max is the conventional saliency reading threshold.
FOCUS_THRESHOLD = 0.5

# Attribution mass inside this fraction of pixels defines "concentrated".
TOP_FRACTION = 0.10

# Concentration bands: share of total attribution held by the top 10% of pixels.
CONCENTRATED_ABOVE = 0.45
DIFFUSE_BELOW = 0.25

# Half-width of the boundary band around the predicted mask edge, in pixels.
# The calvarium is a few pixels thick at network resolution, so a band this
# wide captures "on the skull edge" without swallowing the interior.
BOUNDARY_BAND_PX = 6

# Grid used for coarse spatial description. 3x3 keeps the vocabulary small
# enough to state plainly; finer grids produce descriptions no reader uses.
GRID_ROWS, GRID_COLS = 3, 3

_ROW_NAMES = ("upper", "middle", "lower")
_COL_NAMES = ("left", "centre", "right")


@dataclass
class RegionShare:
    """One grid cell's share of the total attribution."""

    region: str
    share: float
    mean_intensity: float


@dataclass
class AttributionSummary:
    """Measured description of a saliency map. Every field is computed."""

    method: str
    concentration: float = 0.0
    concentration_label: str = "unknown"
    focused_area_pct: float = 0.0
    peak_region: str = "unknown"
    peak_xy_pct: tuple[float, float] = (0.0, 0.0)
    top_regions: list[RegionShare] = field(default_factory=list)
    # Relation to the predicted skull — null when no mask was supplied.
    on_boundary_pct: float | None = None
    inside_head_pct: float | None = None
    outside_head_pct: float | None = None
    mask_available: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["peak_xy_pct"] = list(self.peak_xy_pct)
        return d

    def as_prompt_block(self) -> str:
        """Render the measurements as lines an LLM can quote back."""
        lines = [
            f"attribution method: {self.method}",
            f"focus concentration: {self.concentration:.2f} "
            f"({self.concentration_label}) — share of total attribution held by "
            f"the strongest {int(TOP_FRACTION * 100)}% of pixels",
            f"area above {int(FOCUS_THRESHOLD * 100)}% of peak intensity: "
            f"{self.focused_area_pct:.1f}% of the image",
            f"peak attribution located in the {self.peak_region} region, at "
            f"{self.peak_xy_pct[0]:.0f}% across and {self.peak_xy_pct[1]:.0f}% down",
        ]
        if self.top_regions:
            ranked = ", ".join(f"{r.region} ({r.share * 100:.0f}%)" for r in self.top_regions)
            lines.append(f"attribution by region, strongest first: {ranked}")
        if self.mask_available:
            lines += [
                f"attribution falling on the predicted skull boundary "
                f"(within {BOUNDARY_BAND_PX} px of the segmented edge): "
                f"{self.on_boundary_pct:.1f}%",
                f"attribution inside the predicted head, away from the edge: "
                f"{self.inside_head_pct:.1f}%",
                f"attribution outside the predicted head entirely: {self.outside_head_pct:.1f}%",
            ]
        else:
            lines.append(
                "no segmentation mask was available, so attribution could not be "
                "related to the predicted skull"
            )
        return "\n".join(lines)


def _region_name(row: int, col: int) -> str:
    if row == 1 and col == 1:
        return "centre"
    if row == 1:
        return f"middle-{_COL_NAMES[col]}"
    if col == 1:
        return f"{_ROW_NAMES[row]}-centre"
    return f"{_ROW_NAMES[row]}-{_COL_NAMES[col]}"


def _concentration_label(value: float) -> str:
    if value >= CONCENTRATED_ABOVE:
        return "concentrated"
    if value <= DIFFUSE_BELOW:
        return "diffuse"
    return "moderately focused"


def _boundary_band(mask: np.ndarray) -> np.ndarray:
    """Pixels within BOUNDARY_BAND_PX of the mask edge.

    Implemented with a distance transform rather than morphological dilation so
    the band is symmetric about the edge — attribution just outside the skull
    line is as much "on the boundary" as attribution just inside it.
    """
    import cv2

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0 or binary.all():
        return np.zeros_like(binary, dtype=bool)

    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    signed = np.where(binary > 0, inside, outside)
    return signed <= BOUNDARY_BAND_PX


def summarize_attribution(
    heatmap: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "GradCAM++",
) -> AttributionSummary:
    """Reduce a [0, 1] attribution map to measured, quotable facts.

    ``mask`` is the predicted segmentation at the same resolution. When absent,
    the skull-relative fields are None rather than guessed — an explanation
    should say the relationship is unknown, not assume it.
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be 2-D, got shape {hm.shape}")

    summary = AttributionSummary(method=method)

    total = float(hm.sum())
    if total <= 0 or not np.isfinite(total):
        # A flat or empty map carries no signal; saying so beats reporting
        # a peak region that is really just array index 0.
        summary.concentration_label = "no attribution signal"
        return summary

    flat = np.sort(hm.ravel())[::-1]
    k = max(1, int(len(flat) * TOP_FRACTION))
    summary.concentration = round(float(flat[:k].sum() / total), 4)
    summary.concentration_label = _concentration_label(summary.concentration)

    peak_value = float(hm.max())
    focused = hm >= (FOCUS_THRESHOLD * peak_value)
    summary.focused_area_pct = round(float(focused.mean() * 100), 2)

    peak_idx = int(np.argmax(hm))
    py, px = np.unravel_index(peak_idx, hm.shape)
    h, w = hm.shape
    summary.peak_xy_pct = (round(float(px) / w * 100, 1), round(float(py) / h * 100, 1))
    summary.peak_region = _region_name(
        min(int(py / h * GRID_ROWS), GRID_ROWS - 1),
        min(int(px / w * GRID_COLS), GRID_COLS - 1),
    )

    regions: list[RegionShare] = []
    rows = np.array_split(np.arange(h), GRID_ROWS)
    cols = np.array_split(np.arange(w), GRID_COLS)
    for r, row_idx in enumerate(rows):
        for c, col_idx in enumerate(cols):
            cell = hm[np.ix_(row_idx, col_idx)]
            regions.append(
                RegionShare(
                    region=_region_name(r, c),
                    share=round(float(cell.sum() / total), 4),
                    mean_intensity=round(float(cell.mean()), 4),
                )
            )
    regions.sort(key=lambda r: r.share, reverse=True)
    summary.top_regions = regions[:3]

    if mask is not None:
        m = np.asarray(mask)
        if m.shape != hm.shape:
            import cv2

            m = cv2.resize(
                m.astype(np.uint8), (hm.shape[1], hm.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        binary = m > 0
        if binary.any() and not binary.all():
            band = _boundary_band(binary.astype(np.uint8))
            interior = binary & ~band
            exterior = ~binary & ~band
            summary.on_boundary_pct = round(float(hm[band].sum() / total * 100), 2)
            summary.inside_head_pct = round(float(hm[interior].sum() / total * 100), 2)
            summary.outside_head_pct = round(float(hm[exterior].sum() / total * 100), 2)
            summary.mask_available = True

    return summary
