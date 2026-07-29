"""Cine-loop synthesis and animated segmentation-overlay rendering.

Two concerns live here:

1. **Pseudo-LDDM v2 cine synthesis** — turning a single ultrasound frame into
   a plausible N-frame cine-loop (Ornstein-Uhlenbeck probe motion + Rician
   speckle + depth attenuation). This was previously inlined in the Streamlit
   ``app/app.py``; it moved here so the FastAPI ``/infer`` route can produce
   the same loop the Streamlit cine tab does.

2. **Animated overlay rendering** — drawing the per-frame predicted
   segmentation onto its own frame and encoding the sequence as a single
   animated GIF, returned as a ``data:`` URI. Nothing is written to disk: the
   serverless / Space filesystem is ephemeral and the response has to be
   self-contained.

The GIF is deliberately budgeted. ``build_overlay_gif`` subsamples frames and
downscales until the encoded payload fits ``MAX_GIF_BYTES``, so a long clip
cannot blow the JSON response size limit.
"""

from __future__ import annotations

import base64
import io
import logging

import cv2
import numpy as np
from PIL import Image

try:  # package import — FastAPI (PYTHONPATH=/code, `app.api.main:app`)
    from app.inference import INPUT_H, INPUT_W, N_FRAMES
except ImportError as _exc:  # flat import — Streamlit runs with app/ on sys.path
    if _exc.name not in ("app", "app.inference"):
        raise  # a genuine missing dependency — don't mask it behind the fallback
    from inference import INPUT_H, INPUT_W, N_FRAMES

log = logging.getLogger(__name__)

# ── GIF budget ────────────────────────────────────────────────────────────────
# The overlay GIF rides inside the /infer JSON response, so it has a hard
# ceiling. These are the knobs build_overlay_gif turns down, in order, until
# the payload fits.
MAX_GIF_FRAMES = 24  # never animate more than this many frames
MAX_GIF_BYTES = 300_000  # ~300 KB of GIF → ~400 KB as base64
DEFAULT_FPS = 8
GIF_COLORS = 64  # palette size; 64 is indistinguishable from 256 here

# Quality ladder, walked in order until the payload fits MAX_GIF_BYTES. Width
# comes down first (the receiving panel is ~288 px, so 256 already displays
# near 1:1), then frame count, then width again as a last resort. Ultrasound
# speckle is close to incompressible, so a full-resolution 16-frame loop is
# several times over budget and something has to give.
_LADDER: tuple[tuple[int, int], ...] = (
    # (width, max_frames)
    (320, 24),
    (288, 24),
    (256, 24),
    (256, 12),
    (224, 12),
    (192, 12),
    (192, 8),
    (160, 8),
)

# Contour colour (RGB). Bright cyan-green survives palette quantisation and
# sits well off the grayscale ultrasound ramp, so the skull outline stays
# legible even after the GIF is reduced to 256 colours.
CONTOUR_RGB = (60, 240, 180)
CONTOUR_THICKNESS = 2
# A very light fill under the contour gives the eye something to track between
# frames without hiding the skull echo it is drawn over.
FILL_RGB = (60, 240, 180)
# Kept low deliberately: the head interior is near-black, so even a light tint
# reads strongly, and anything heavier hides the falx / midline echo.
FILL_ALPHA = 0.07


# ── Pseudo-LDDM v2 cine synthesis ─────────────────────────────────────────────


def ornstein_uhlenbeck(
    n: int, theta: float = 0.3, sigma: float = 1.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Mean-reverting random walk — models hand-held probe drift."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (0 - x[t - 1]) + sigma * rng.normal(0, 1)
    return x


def add_rician_speckle(
    img: np.ndarray, std: float = 0.08, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Rician-distributed speckle — the characteristic ultrasound noise model."""
    if rng is None:
        rng = np.random.default_rng()
    n1 = rng.normal(0, std, img.shape).astype(np.float32)
    n2 = rng.normal(0, std, img.shape).astype(np.float32)
    return np.clip(np.sqrt((img + n1) ** 2 + n2**2), 0, 1)


def add_depth_attenuation(img: np.ndarray, coeff: float = 0.35) -> np.ndarray:
    """Exponential intensity falloff with depth."""
    h = img.shape[0]
    return img * np.exp(-np.linspace(0, coeff, h, dtype=np.float32))[:, np.newaxis]


def synthesize_cine_frames(
    img_gray: np.ndarray, n_frames: int = N_FRAMES, seed: int = 42
) -> list[np.ndarray]:
    """Synthesise an ``n_frames`` cine-loop from a single frame (Pseudo-LDDM v2).

    Returns a list of ``[INPUT_H, INPUT_W]`` uint8 frames. Deterministic for a
    given ``seed`` so repeat requests on the same image return the same loop.
    """
    n_frames = max(1, int(n_frames))
    rng = np.random.default_rng(seed)
    img_r = cv2.resize(img_gray, (INPUT_W, INPUT_H))
    img_f = img_r.astype(np.float32) / 255.0

    tx = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=2.0, rng=rng)
    ty = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=1.5, rng=rng)
    rot = ornstein_uhlenbeck(n_frames, theta=0.20, sigma=0.40, rng=rng)

    # Occasional sharp probe correction that decays away, as a sonographer
    # re-centres the head mid-sweep. Needs room either side to be meaningful.
    if n_frames > 6 and rng.random() < 0.3:
        wf = int(rng.integers(3, n_frames - 3))
        wtx, wty = rng.normal(0, 8), rng.normal(0, 6)
        tx[wf:] += wtx * np.exp(-0.5 * np.arange(n_frames - wf))
        ty[wf:] += wty * np.exp(-0.5 * np.arange(n_frames - wf))

    cx, cy = INPUT_W / 2, INPUT_H / 2
    frames = []
    for i in range(n_frames):
        M = cv2.getRotationMatrix2D((cx, cy), float(rot[i]), 1.0)
        M[0, 2] += float(tx[i])
        M[1, 2] += float(ty[i])
        w = cv2.warpAffine(
            img_f,
            M,
            (INPUT_W, INPUT_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        w = add_rician_speckle(w, std=float(rng.uniform(0.04, 0.10)), rng=rng)
        w = add_depth_attenuation(w, coeff=float(rng.uniform(0.20, 0.45)))
        frames.append((np.clip(w, 0, 1) * 255).astype(np.uint8))
    return frames


# ── overlay rendering ─────────────────────────────────────────────────────────


def make_contour_overlay(frame_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the predicted segmentation boundary on one frame. Returns RGB uint8.

    A traced contour plus a faint fill, rather than a solid mask fill: on
    ultrasound a heavy fill hides the very skull echo a reader needs to check
    the boundary against.
    """
    if frame_gray.ndim == 3:
        rgb = frame_gray.astype(np.uint8)
    else:
        rgb = cv2.cvtColor(frame_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    if mask is None:
        return rgb

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.shape != rgb.shape[:2]:
        mask_u8 = cv2.resize(
            mask_u8, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    if not mask_u8.any():
        return rgb

    out = rgb.copy()
    fill = out.copy()
    fill[mask_u8 > 0] = FILL_RGB
    out = cv2.addWeighted(out, 1 - FILL_ALPHA, fill, FILL_ALPHA, 0)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, CONTOUR_RGB, CONTOUR_THICKNESS)
    return out


def _subsample(items: list, limit: int) -> list:
    """Evenly pick at most ``limit`` items, always keeping the first and last."""
    if limit <= 0 or len(items) <= limit:
        return list(items)
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx.tolist())]


def _smooth_for_display(frame_gray: np.ndarray) -> np.ndarray:
    """Light denoise + gray-level quantisation, applied to the GIF background only.

    The speckle this removes is the *synthetic* Rician noise
    ``synthesize_cine_frames`` just added, not acquired image detail, and the
    diagnostic stills (``overlay_b64``) are untouched. Dropping it roughly
    halves the encoded GIF, which is the difference between showing the loop
    at 256 px and showing it at 160 px.
    """
    smoothed = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    return (smoothed // 16 * 16).astype(np.uint8)  # 16 gray levels


def _encode_gif(frames_rgb: list[np.ndarray], width: int, fps: int) -> bytes:
    """Palette-quantise, downscale to ``width``, and encode as an animated GIF."""
    pil_frames = []
    for f in frames_rgb:
        im = Image.fromarray(f, mode="RGB")
        if im.width > width:
            height = max(1, round(im.height * width / im.width))
            im = im.resize((width, height), Image.LANCZOS)
        # Quantise explicitly rather than letting Pillow re-decide per frame,
        # which bloats the per-frame local colour tables.
        pil_frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=GIF_COLORS))

    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(20, round(1000 / max(1, fps))),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return buf.getvalue()


def build_overlay_gif(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    fps: int = DEFAULT_FPS,
    max_frames: int = MAX_GIF_FRAMES,
    max_bytes: int = MAX_GIF_BYTES,
) -> str | None:
    """Overlay each mask on its frame and encode the loop as a GIF data URI.

    Returns ``data:image/gif;base64,...`` or ``None`` when there is nothing
    worth animating (no frames, a single frame, or encoding failed). Callers
    should treat ``None`` as "no animation available" and fall back to the
    static consensus overlay — this never raises.
    """
    try:
        if not frames or not masks:
            return None

        # Pair frames with masks; a short mask list (a frame whose HC/mask
        # extraction failed upstream) just means those frames animate bare.
        pairs = [(frames[i], masks[i] if i < len(masks) else None) for i in range(len(frames))]

        # A one-frame "loop" is a still image — the caller already has
        # overlay_b64 for that, so don't pay the payload cost twice.
        if len(pairs) < 2:
            return None

        pairs = _subsample(pairs, max_frames)

        rendered: list[np.ndarray] = []
        for frame, mask in pairs:
            try:
                rendered.append(make_contour_overlay(_smooth_for_display(frame), mask))
            except Exception:  # noqa: BLE001 — one bad frame must not kill the loop
                log.warning("cine overlay: frame render failed, using raw frame", exc_info=True)
                try:
                    rendered.append(
                        cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
                    )
                except Exception:  # noqa: BLE001
                    continue

        if len(rendered) < 2:
            return None

        # Walk the quality ladder until the payload fits. If even the last rung
        # overshoots we ship it anyway — an oversized animation is better than
        # none, and the ladder's floor is well inside any sane response limit.
        data = b""
        for width, frame_cap in _LADDER:
            data = _encode_gif(_subsample(rendered, min(frame_cap, max_frames)), width, fps)
            if len(data) <= max_bytes:
                break
        else:
            log.warning(
                "cine overlay: GIF is %d bytes, over the %d budget at the lowest rung",
                len(data),
                max_bytes,
            )

        return "data:image/gif;base64," + base64.b64encode(data).decode("ascii")
    except Exception:  # noqa: BLE001 — the animation is a nicety, never a 500
        log.warning("cine overlay: GIF build failed", exc_info=True)
        return None
