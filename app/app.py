"""
app.py — Fetal Head Circumference Clinical AI
Production-grade Streamlit application.

Tab structure:
  1. Static Analysis     — Phase 0 (baseline) or Phase 4a (compressed)
  2. Cine Analysis       — Phase 2 (baseline) or Phase 4b (compressed)
  3. Head-to-Head        — All four models on one image simultaneously
  4. Model Card          — Responsible AI documentation

Key production features over the original deployment:
  - All four model variants loadable and selectable at runtime
  - Inference latency measured and displayed for every prediction
  - Input validation with clinical sanity checks before inference
  - Optional ground-truth mask upload for Dice/MAE verification
  - Confidence badge (HIGH / MODERATE / LOW) from reliability score
  - Session analytics counter (analyses run, average latency)
  - System information panel in sidebar
  - Three-colour GT comparison overlay (TP=yellow, FP=red, FN=green)
  - Structured clinical report card inline (not PDF-only)
"""

import io
import os
import platform
import time

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import streamlit as st
from inference import (
    # constants
    DEVICE,
    INPUT_H,
    INPUT_W,
    N_FRAMES,
    compute_gt_metrics,
    confidence_label,
    get_model_info,
    # loaders
    load_phase0,
    load_phase2,
    load_phase4a,
    load_phase4b,
    make_comparison_overlay,
    predict_cine_clip,
    # prediction
    predict_single_frame,
    # utilities
    validate_input,
)
from model_card import render_model_card
from PIL import Image
from report import generate_cine_report, generate_comparison_report, generate_static_report
from xai import build_xai_panel

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Head Circumference AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)

# ── session state initialisation ──────────────────────────────────────────────
if "analyses_run"     not in st.session_state:
    st.session_state.analyses_run     = 0
if "total_latency_ms" not in st.session_state:
    st.session_state.total_latency_ms = 0.0
if "load_start"       not in st.session_state:
    st.session_state.load_start       = time.perf_counter()
# 3-source pixel spacing: DICOM > HC18 CSV > user (default = HC18 median 0.154 mm/px)
if "ps_input"  not in st.session_state:
    st.session_state.ps_input  = 0.154
if "ps_source" not in st.session_state:
    st.session_state.ps_source = "user"
# Persisted inference results — survive form-field reruns so patient form stays visible
if "static_result" not in st.session_state:
    st.session_state.static_result = None
if "cine_result" not in st.session_state:
    st.session_state.cine_result = None
if "static_pdf" not in st.session_state:
    st.session_state.static_pdf = None
if "cine_pdf" not in st.session_state:
    st.session_state.cine_pdf = None


# ── Pseudo-LDDM v2 cine synthesis ─────────────────────────────────────────────

def ornstein_uhlenbeck(n, theta=0.3, sigma=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] + theta * (0 - x[t-1]) + sigma * rng.normal(0, 1)
    return x

def add_rician_speckle(img, std=0.08, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n1 = rng.normal(0, std, img.shape).astype(np.float32)
    n2 = rng.normal(0, std, img.shape).astype(np.float32)
    return np.clip(np.sqrt((img + n1) ** 2 + n2 ** 2), 0, 1)

def add_depth_attenuation(img, coeff=0.35):
    h = img.shape[0]
    return img * np.exp(-np.linspace(0, coeff, h, dtype=np.float32))[:, np.newaxis]

def generate_cine(img_gray, n_frames=N_FRAMES, seed=42):
    rng   = np.random.default_rng(seed)
    img_r = cv2.resize(img_gray, (INPUT_W, INPUT_H))
    img_f = img_r.astype(np.float32) / 255.0
    tx    = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=2.0,  rng=rng)
    ty    = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=1.5,  rng=rng)
    rot   = ornstein_uhlenbeck(n_frames, theta=0.20, sigma=0.40, rng=rng)
    if rng.random() < 0.3:
        wf  = rng.integers(3, n_frames - 3)
        wtx, wty = rng.normal(0, 8), rng.normal(0, 6)
        tx[wf:] += wtx * np.exp(-0.5 * np.arange(n_frames - wf))
        ty[wf:] += wty * np.exp(-0.5 * np.arange(n_frames - wf))
    cx, cy = INPUT_W / 2, INPUT_H / 2
    frames = []
    for i in range(n_frames):
        M = cv2.getRotationMatrix2D((cx, cy), float(rot[i]), 1.0)
        M[0, 2] += float(tx[i])
        M[1, 2] += float(ty[i])
        w = cv2.warpAffine(img_f, M, (INPUT_W, INPUT_H),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        w = add_rician_speckle(w, std=float(rng.uniform(0.04, 0.10)), rng=rng)
        w = add_depth_attenuation(w, coeff=float(rng.uniform(0.20, 0.45)))
        frames.append((np.clip(w, 0, 1) * 255).astype(np.uint8))
    return frames

def frames_to_gif(frames, fps=8):
    buf = io.BytesIO()
    imageio.mimsave(buf,
                    [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in frames],
                    format="GIF", fps=fps, loop=0)
    buf.seek(0)
    return buf.read()


# ── pixel spacing helpers (3-source pipeline) ─────────────────────────────────

@st.cache_data
def _load_hc18_csv() -> dict:
    """Load filename → pixel spacing from training_set_pixel_size_and_HC.csv."""
    csv_candidates = [
        Path(__file__).parent.parent / "training_set_pixel_size_and_HC.csv",
        Path(__file__).parent / "training_set_pixel_size_and_HC.csv",
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            try:
                import csv
                result = {}
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fn = (row.get("filename") or "").strip()
                        ps = (row.get("pixel size(mm)") or "").strip()
                        if fn and ps:
                            result[fn] = float(ps)
                return result
            except Exception:
                pass
    return {}


def _dicom_pixel_spacing(file_bytes: bytes) -> float | None:
    """Extract pixel spacing from DICOM tag (0028,0030) using pydicom."""
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        ps = ds.PixelSpacing
        return float(ps[0])
    except Exception:
        return None


# ── model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models — please wait...")
def get_all_models():
    t0  = time.perf_counter()
    m0  = load_phase0("phase0_model.pth",  device=DEVICE)
    m4a = load_phase4a("4a_best_pruned_ft_v10.pth", device=DEVICE)
    m2  = load_phase2("phase2_model.pth",  device=DEVICE)
    m4b = load_phase4b("4b_best_pruned_ft_v10.pth", device=DEVICE)
    elapsed = (time.perf_counter() - t0) * 1000
    return m0, m4a, m2, m4b, elapsed

phase0_model, phase4a_model, phase2_model, phase4b_model, model_load_ms = get_all_models()

MODEL_INFO = {
    "Phase 0 — Static baseline":   get_model_info(phase0_model),
    "Phase 4a — Compressed static": get_model_info(phase4a_model),
    "Phase 2 — Temporal baseline":  get_model_info(phase2_model),
    "Phase 4b — Compressed temporal": get_model_info(phase4b_model),
}

REPORTED_METRICS = {
    "Phase 0 — Static baseline": {
        "dice": "97.75%", "mae": "1.65 mm", "params": "8.11M",
        "flops": "21.58 GMACs", "latency": "11.6 ms", "isuog": "PASS",
    },
    "Phase 4a — Compressed static": {
        "dice": "97.64%", "mae": "1.76 mm", "params": "4.57M",
        "flops": "16.56 GMACs", "latency": "9.9 ms", "isuog": "PASS",
    },
    "Phase 2 — Temporal baseline": {
        "dice": "95.95%", "mae": "2.10 mm", "params": "8.90M",
        "flops": "21.58 GMACs", "latency": "182.7 ms (16 frames)", "isuog": "PASS",
    },
    "Phase 4b — Compressed temporal": {
        "dice": "96.00%", "mae": "2.06 mm", "params": "5.20M",
        "flops": "16.44 GMACs", "latency": "171.5 ms (16 frames)", "isuog": "PASS",
    },
}


# ── demo subjects ─────────────────────────────────────────────────────────────

DEMO_DIR = Path("demo_subjects")

def get_demo_subjects():
    if not DEMO_DIR.exists():
        return []
    return sorted([f.name for f in DEMO_DIR.glob("*.png")])

def load_demo_image(filename):
    return np.array(Image.open(DEMO_DIR / filename).convert("L"))


# ── shared UI helpers ─────────────────────────────────────────────────────────

def render_confidence_badge(label: str, color: str):
    st.markdown(
        f"<div style='display:inline-block;padding:6px 14px;border-radius:6px;"
        f"background:{color}22;border:1.5px solid {color};color:{color};"
        f"font-weight:600;font-size:0.82em;letter-spacing:0.04em;'>"
        f"⬤ {label}</div>",
        unsafe_allow_html=True,
    )

def render_metrics_card(result: dict, model_name: str, pixel_spacing: float):
    """Inline structured clinical report card (not PDF)."""
    hc   = result.get("hc_mm")
    ms   = result.get("elapsed_ms", 0.0)
    rel  = result.get("reliability", 1.0)
    conf_lbl, conf_col = confidence_label(rel)

    st.markdown("#### Clinical Measurement Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    if hc:
        c1.metric("Head Circumference", f"{hc:.1f} mm")
        c2.metric("Gestational Age", result.get("ga_str") or "—", "±2 weeks (Hadlock 1984)")
        c3.metric("Trimester", result.get("trimester") or "—")
    c4.metric("Inference Time", f"{ms:.1f} ms")
    c5.metric("Reliability", f"{rel:.3f}" if result.get("mode") == "cine_clip" else "N/A (single frame)")

    st.markdown("")
    render_confidence_badge(conf_lbl, conf_col)
    st.markdown("")

    m = REPORTED_METRICS.get(model_name, {})
    st.caption(
        f"Model: **{model_name}** · "
        f"Reported Dice: {m.get('dice','—')} · "
        f"Reported MAE: {m.get('mae','—')} · "
        f"Parameters: {m.get('params','—')} · "
        f"ISUOG: {m.get('isuog','—')} · "
        f"Pixel spacing: {pixel_spacing:.4f} mm/px"
    )

def update_session_stats(elapsed_ms: float):
    st.session_state.analyses_run     += 1
    st.session_state.total_latency_ms += elapsed_ms

def render_gt_section(result: dict, pixel_spacing: float, key_prefix: str):
    """Optional ground truth upload and comparison panel."""
    with st.expander("📐 Compare against ground truth mask (optional)", expanded=False):
        st.caption(
            "Upload the corresponding HC18 ground-truth annotation (PNG, white ellipse on black). "
            "The app will compute Dice and MAE between the model prediction and your mask."
        )
        gt_file = st.file_uploader(
            "Upload ground truth mask", type=["png", "jpg", "jpeg"],
            key=f"gt_{key_prefix}"
        )
        if gt_file:
            gt_img  = np.array(Image.open(gt_file).convert("L"))
            gt_bin  = (cv2.resize(gt_img, (INPUT_W, INPUT_H)) > 127).astype(np.uint8)
            gt_metrics = compute_gt_metrics(result["mask"], gt_bin, pixel_spacing)
            comp_overlay = make_comparison_overlay(
                cv2.resize(
                    np.array(Image.open(gt_file).convert("L")), (INPUT_W, INPUT_H)
                ),
                result["mask"], gt_bin
            )
            col1, col2, col3 = st.columns(3)
            col1.image(result["overlay"], caption="Model prediction", use_column_width=True)
            col2.image(comp_overlay,
                       caption="Comparison (yellow=TP, red=FP, green=FN)",
                       use_column_width=True)
            col3.image(gt_bin * 255, caption="Ground truth mask", use_column_width=True)

            gc1, gc2 = st.columns(2)
            gc1.metric("Dice vs Ground Truth", f"{gt_metrics['dice']*100:.2f}%")
            if gt_metrics["mae_mm"] is not None:
                gc2.metric("MAE vs Ground Truth", f"{gt_metrics['mae_mm']:.2f} mm")


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    pixel_spacing = st.number_input(
        "Pixel spacing (mm/pixel)",
        min_value=0.010, max_value=0.500, step=0.001, format="%.4f",
        key="ps_input",
        help="Physical size of one pixel. Found in DICOM tag (0028,0030). "
             "Default 0.154 mm/px = HC18 dataset median. "
             "Incorrect spacing directly affects HC and GA accuracy.",
    )
    _ps_source = st.session_state.get("ps_source", "user")
    if _ps_source == "dicom":
        st.caption("✅ DICOM-derived — pixel spacing verified from tag (0028,0030)")
    elif _ps_source == "csv":
        st.caption("✅ HC18 CSV — spacing from training dataset record")
    else:
        if not (0.030 <= pixel_spacing <= 0.500):
            st.caption(f"⚠️ Value {pixel_spacing:.4f} mm/px is outside typical range (0.030–0.500)")
        else:
            st.caption("⚠️ User-supplied pixel spacing — verify before clinical use")

    # Confirmation checkbox only needed when spacing is user-supplied (not CSV/DICOM)
    if _ps_source == "user":
        pixel_spacing_confirmed = st.checkbox(
            "I confirm the pixel spacing matches the DICOM tag (0028,0030)",
            value=False,
            help=(
                "HC and GA both depend on pixel spacing being correct. "
                "Check the source DICOM before generating a clinical report."
            ),
            key="pixel_spacing_confirmed",
        )
        if not pixel_spacing_confirmed:
            st.caption(
                "⚠️ Confirm pixel spacing above before generating a clinical report."
            )
    else:
        pixel_spacing_confirmed = True  # CSV/DICOM sources are pre-verified

    use_llm = st.checkbox(
        "Enable LLM clinical summary",
        value=True if ANTHROPIC_API_KEY else False,
        disabled=(ANTHROPIC_API_KEY is None),
        help=(
            "Uses Claude Haiku to write clinical paragraphs in medical language."
            if ANTHROPIC_API_KEY
            else "ANTHROPIC_API_KEY not configured — rule-based template active."
        ),
    )
    if not ANTHROPIC_API_KEY:
        st.caption("ℹ️ Rule-based report template active.")
    elif use_llm:
        st.caption("✅ LLM clinical summary active.")
    else:
        st.caption("📋 Rule-based template active.")

    st.markdown("---")
    st.markdown("**Model performance (HC18 test set)**")
    st.markdown(
        "| Model | Dice | MAE | Params |\n"
        "|-------|------|-----|--------|\n"
        "| Phase 0 | 97.75% | 1.65 mm | 8.11M |\n"
        "| Phase 4a ✂️ | 97.64% | 1.76 mm | 4.57M |\n"
        "| Phase 2 | 95.95% | 2.10 mm | 8.90M |\n"
        "| Phase 4b ✂️ | 96.00% | 2.06 mm | 5.20M |"
    )

    st.markdown("---")

    # Session analytics
    n_run = st.session_state.analyses_run
    avg_ms = (st.session_state.total_latency_ms / n_run) if n_run > 0 else 0.0
    st.markdown("**Session**")
    st.caption(f"Analyses run: **{n_run}** · Avg latency: **{avg_ms:.0f} ms**")

    st.markdown("---")

    # System info
    with st.expander("🖥️ System info"):
        mi = MODEL_INFO["Phase 0 — Static baseline"]
        st.caption(
            f"Device: **{str(DEVICE).upper()}**\n\n"
            f"PyTorch: **{torch.__version__}**\n\n"
            f"Python: **{platform.python_version()}**\n\n"
            f"Model load time: **{model_load_ms:.0f} ms**\n\n"
            f"Phase 0 size: **{MODEL_INFO['Phase 0 — Static baseline']['size_mb']} MB**\n\n"
            f"Phase 4a size: **{MODEL_INFO['Phase 4a — Compressed static']['size_mb']} MB**\n\n"
            f"Phase 2 size: **{MODEL_INFO['Phase 2 — Temporal baseline']['size_mb']} MB**\n\n"
            f"Phase 4b size: **{MODEL_INFO['Phase 4b — Compressed temporal']['size_mb']} MB**"
        )

    st.markdown("---")
    st.markdown("**Dataset:** HC18 · Radboud UMC, Netherlands")
    st.markdown("**Author:** Tarun Sadarla · MS AI · UNT 2026")
    st.markdown("---")
    st.caption("⚠️ Research prototype · Not FDA-cleared\nRequires sonographer verification")


# ── header ────────────────────────────────────────────────────────────────────
st.title("🧬 Fetal Head Circumference Clinical AI")
st.caption(
    "Automated HC measurement · Gestational age estimation · "
    "GradCAM++ XAI · Temporal uncertainty quantification · "
    "Structural pruning compression · Clinical report generation"
)
st.markdown("---")

tab_static, tab_cine, tab_compare, tab_card = st.tabs([
    "📷 Static Analysis",
    "🎬 Cine Analysis",
    "⚖️ Head-to-Head",
    "📋 Model Card",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STATIC (Phase 0 / Phase 4a)
# ══════════════════════════════════════════════════════════════════════════════
with tab_static:
    st.subheader("Static Analysis — Single Frame Segmentation")

    # Model selector
    static_model_choice = st.radio(
        "Model variant",
        ["Phase 0 — Static baseline (8.11M params)",
         "Phase 4a — Compressed static (4.57M params, −43.7%)"],
        horizontal=True, key="static_model_choice",
        help="Phase 4a is the structurally-pruned version. "
             "Same architecture, 43.7% fewer parameters, 1.17× faster inference.",
    )
    use_phase4a = "4a" in static_model_choice
    active_static_model = phase4a_model if use_phase4a else phase0_model
    active_static_name  = (
        "Phase 4a — Compressed static" if use_phase4a
        else "Phase 0 — Static baseline"
    )

    if use_phase4a:
        st.info(
            "✂️ **Compressed model active.** Phase 4a is 43.7% smaller than the baseline "
            "(4.57M vs 8.11M params) with <0.12pp Dice difference. "
            "Pruned using Hybrid Crossover channel merging + Knowledge Distillation recovery.",
            icon=None,
        )

    st.caption(
        "Upload a fetal head ultrasound image. The model performs skull boundary "
        "segmentation and estimates HC and gestational age. "
        "GradCAM++ shows which image regions drove the boundary prediction."
    )

    demo_subjects = get_demo_subjects()
    input_mode_s = st.radio(
        "Input mode",
        ["Use demo subject", "Upload your own image"] if demo_subjects else ["Upload your own image"],
        horizontal=True, key="mode_s",
    )

    _hc18_csv = _load_hc18_csv()
    img_gray = None
    if input_mode_s == "Use demo subject" and demo_subjects:
        selected_demo_s = st.selectbox(
            "Select demo subject", demo_subjects,
            format_func=lambda x: x.replace(".png", "").replace("_", " "),
            key="demo_s",
        )
        _csv_ps = _hc18_csv.get(selected_demo_s)
        if _csv_ps:
            _col_a, _col_b = st.columns([3, 1])
            _col_a.caption(f"HC18 CSV pixel spacing for this image: **{_csv_ps:.4f} mm/px**")
            if _col_b.button("Apply", key="apply_csv_s"):
                st.session_state.ps_input = _csv_ps
                st.session_state.ps_source = "csv"
                st.rerun()
        if st.button("Run analysis on demo subject", key="run_demo_s"):
            img_gray = load_demo_image(selected_demo_s)
    else:
        uploaded_s = st.file_uploader(
            "Upload ultrasound image (PNG or JPG, or .dcm DICOM)",
            type=["png", "jpg", "jpeg", "dcm"], key="sf",
        )
        if uploaded_s:
            _raw = uploaded_s.read()
            if uploaded_s.name.lower().endswith(".dcm"):
                _dicom_ps = _dicom_pixel_spacing(_raw)
                if _dicom_ps:
                    st.session_state.ps_input = _dicom_ps
                    st.session_state.ps_source = "dicom"
                    st.info(f"DICOM pixel spacing extracted: **{_dicom_ps:.4f} mm/px** ✅")
                img_gray = np.array(Image.open(io.BytesIO(_raw)).convert("L"))
            else:
                st.session_state.ps_source = "user"
                img_gray = np.array(Image.open(io.BytesIO(_raw)).convert("L"))

    if img_gray is not None:

        # Input validation
        validation = validate_input(img_gray)
        if validation["warnings"]:
            for w in validation["warnings"]:
                st.warning(f"⚠️ {w}")
            if not validation["valid"]:
                st.error(
                    "Input failed clinical sanity checks. "
                    "Inference blocked to prevent misleading results. "
                    "Please upload a valid grayscale ultrasound image."
                )
                st.stop()

        with st.spinner(f"Running segmentation + GradCAM++ ({active_static_name})..."):
            result_s = predict_single_frame(
                active_static_model, img_gray, pixel_spacing, device=DEVICE
            )
            xai_s = build_xai_panel(img_gray, result_s, phase0_model=active_static_model)

        update_session_stats(result_s["elapsed_ms"])
        # Persist result so the patient form survives form-field reruns
        st.session_state.static_result = result_s
        st.session_state.static_pdf = None  # invalidate any previous PDF

        # Image panels
        c1, c2, c3 = st.columns(3)
        c1.image(cv2.resize(img_gray, (INPUT_W, INPUT_H)),
                 caption="Input ultrasound", clamp=True, use_column_width=True)
        c2.image(result_s["overlay"],
                 caption="Segmentation overlay (red = predicted skull boundary)",
                 clamp=True, use_column_width=True)
        if "gradcam_overlay" in xai_s:
            c3.image(xai_s["gradcam_overlay"],
                     caption="GradCAM++ — boundary attention (red/yellow = high influence)",
                     clamp=True, use_column_width=True)
        else:
            c3.image(result_s["mask"] * 255, caption="Binary mask",
                     clamp=True, use_column_width=True)

        st.markdown("---")

        if result_s["hc_mm"] is not None:
            render_metrics_card(result_s, active_static_name, pixel_spacing)
            st.markdown("---")
            render_gt_section(result_s, pixel_spacing, key_prefix="static")
        else:
            st.warning("Could not estimate HC. Check pixel spacing and image quality.")

        with st.expander("ℹ️ Technical details"):
            m = REPORTED_METRICS[active_static_name]
            st.markdown(
                f"- **Model:** {'Compressed Residual U-Net (Phase 4a — Hybrid Crossover pruned)' if use_phase4a else 'Residual U-Net with deep supervision (Phase 0 baseline)'}\n"
                f"- **Parameters:** {m['params']} · **FLOPs:** {m['flops']}\n"
                f"- **Reported Dice:** {m['dice']} · **Reported MAE:** {m['mae']} · **ISUOG:** {m['isuog']}\n"
                f"- **Runtime inference:** {result_s['elapsed_ms']:.1f} ms on {str(DEVICE).upper()}\n"
                f"- **XAI:** GradCAM++ on final decoder layer (custom, no external packages)\n"
                f"- **HC formula:** Ramanujan ellipse perimeter approximation\n"
                f"- **GA formula:** Hadlock et al. AJR 1984 (±2 weeks CI)\n"
                f"- **Pixel spacing:** {pixel_spacing:.4f} mm/px"
                + ("\n- **Pruning:** Hybrid Crossover channel merging + KD recovery · 3 prune-FT cycles · 49.1% channel compression · Wilcoxon p=0.0049" if use_phase4a else "")
            )


    # ── Persistent report-generation section (survives form-field reruns) ──────
    _stored_s = st.session_state.get("static_result")
    if _stored_s and _stored_s.get("hc_mm"):
        from datetime import date as _date_s, timedelta as _td_s
        st.markdown("---")
        st.success(
            f"✓ Analysis results loaded — HC **{_stored_s['hc_mm']:.1f} mm** · "
            f"GA **{_stored_s.get('ga_str', '—')}** · "
            f"Confidence **{_stored_s.get('confidence_label', '—')}**"
        )

        st.subheader("Generate Clinical Report")

        # Demo Mode
        demo_mode_s = st.toggle("Enable Demo Mode (pre-fill clinical scenario)", key="demo_mode_s")
        if demo_mode_s:
            _today_s = _date_s.today()
            _scenario_s = st.selectbox(
                "Select clinical scenario",
                [
                    "A — Normal 2nd trimester",
                    "B — LMP discordance (preterm risk)",
                    "C — IUGR / BPD mismatch",
                ],
                key="scenario_s",
            )
            if _scenario_s.startswith("A"):
                _dp_name, _dp_id   = "Demo Patient A", "HC18-DEMO-001"
                _dp_lmp            = (_today_s - _td_s(days=137)).strftime("%Y-%m-%d")
                _dp_ref, _dp_fac   = "Dr. Sarah Chen, OB/GYN", "City General Hospital"
                _dp_indic          = "Routine 2nd trimester anatomy scan"
                _dp_appr, _dp_qual = "transabdominal", "optimal"
                _dp_bpd, _dp_pres  = 0.0, "cephalic"
            elif _scenario_s.startswith("B"):
                _dp_name, _dp_id   = "Demo Patient B", "HC18-DEMO-002"
                _dp_lmp            = (_today_s - _td_s(days=104)).strftime("%Y-%m-%d")
                _dp_ref, _dp_fac   = "Dr. James Park, MFM", "University Medical Center"
                _dp_indic          = "LMP-size discordance — rule out growth restriction"
                _dp_appr, _dp_qual = "transabdominal", "suboptimal"
                _dp_bpd, _dp_pres  = 0.0, "cephalic"
            else:
                _dp_name, _dp_id   = "Demo Patient C", "HC18-DEMO-003"
                _dp_lmp            = (_today_s - _td_s(days=168)).strftime("%Y-%m-%d")
                _dp_ref, _dp_fac   = "Dr. Maria Santos, MFM", "Perinatology Associates"
                _dp_indic          = "Suspected IUGR — detailed biometry"
                _dp_appr, _dp_qual = "transabdominal", "suboptimal"
                _dp_bpd, _dp_pres  = 52.0, "cephalic"
        else:
            _dp_name = _dp_id = _dp_lmp = _dp_ref = _dp_fac = _dp_indic = ""
            _dp_appr, _dp_qual = "transabdominal", "optimal"
            _dp_bpd, _dp_pres  = 0.0, "cephalic"

        with st.expander("Patient & Exam Information", expanded=True):
            _sc1, _sc2 = st.columns(2)
            _sp_name  = _sc1.text_input("Patient Name",        value=_dp_name,  key="sp_name_s")
            _sp_id    = _sc2.text_input("Patient ID / MRN",    value=_dp_id,    key="sp_id_s")
            _sp_lmp   = _sc1.text_input("LMP (YYYY-MM-DD)",    value=_dp_lmp,   key="sp_lmp_s")
            _sp_ref   = _sc2.text_input("Referring Physician",  value=_dp_ref,   key="sp_ref_s")
            _sp_fac   = _sc1.text_input("Ordering Facility",   value=_dp_fac,   key="sp_fac_s")
            _sp_indic = _sc2.text_input("Clinical Indication",  value=_dp_indic, key="sp_indic_s")
            _appr_opts = ["transabdominal", "transvaginal"]
            _qual_opts = ["optimal", "suboptimal", "limited"]
            _pres_opts = ["cephalic", "breech", "transverse", "not_assessed"]
            _sa1, _sa2, _sa3 = st.columns(3)
            _sp_appr = _sa1.selectbox("US Approach",         _appr_opts, index=_appr_opts.index(_dp_appr), key="sp_appr_s")
            _sp_qual = _sa2.selectbox("Image Quality",       _qual_opts, index=_qual_opts.index(_dp_qual), key="sp_qual_s")
            _sp_pres = _sa3.selectbox("Fetal Presentation",  _pres_opts, index=_pres_opts.index(_dp_pres), key="sp_pres_s")
            _sb1, _sb2 = st.columns(2)
            _sp_bpd   = _sb1.number_input("BPD (mm, optional)", value=_dp_bpd, min_value=0.0, max_value=150.0, step=0.1, key="sp_bpd_s")
            _sp_prior = _sb2.text_input("Prior Biometry", value="", key="sp_prior_s")
            # Read-only auto-filled analysis results
            st.markdown("**AI analysis results (read-only)**")
            _ra1, _ra2, _ra3 = st.columns(3)
            _ra1.text_input("HC (mm)",    value=f"{_stored_s['hc_mm']:.1f}",            disabled=True, key="ra_hc_s")
            _ra2.text_input("GA",         value=_stored_s.get("ga_str", "—"),            disabled=True, key="ra_ga_s")
            _ra3.text_input("Confidence", value=_stored_s.get("confidence_label", "—"), disabled=True, key="ra_conf_s")

        _ps_src_s = st.session_state.get("ps_source", "user")
        _needs_confirm_s = _ps_src_s == "user" and not pixel_spacing_confirmed
        if _needs_confirm_s:
            st.warning("Confirm pixel spacing in the sidebar before generating a clinical report.")

        if st.button(
            "Generate Clinical Report (PDF)",
            disabled=_needs_confirm_s,
            type="primary",
            key="gen_report_s",
            use_container_width=True,
        ):
            class _PatientProxy_S:  # noqa: N801
                patient_name        = _sp_name or "—"
                patient_id          = _sp_id or None
                patient_dob         = None
                lmp                 = _sp_lmp or None
                ordering_facility   = _sp_fac or None
                referring_physician = _sp_ref or None
                sonographer_name    = None
                clinical_indication = _sp_indic or None
                study_date          = _date_s.today().strftime("%Y-%m-%d")
                us_approach         = _sp_appr
                image_quality       = _sp_qual
                pixel_spacing_mm    = pixel_spacing
                pixel_spacing_dicom_derived = (_ps_src_s == "dicom")
                pixel_spacing_source = _ps_src_s.upper()
                bpd_mm              = _sp_bpd if _sp_bpd > 0 else None
                fetal_presentation  = _sp_pres
                prior_biometry      = _sp_prior or None
                original_image_b64  = None
                overlay_image_b64   = None
                gradcam_image_b64   = None
                accession_number    = None
                report_mode         = "template"
            with st.spinner("Generating clinical report..."):
                st.session_state.static_pdf = generate_static_report(
                    _stored_s,
                    api_key=ANTHROPIC_API_KEY,
                    use_llm=use_llm,
                    model_name=active_static_name,
                    pixel_spacing=pixel_spacing,
                    report=_PatientProxy_S(),
                    pixel_spacing_source=_ps_src_s.upper(),
                )
            _rmode = "LLM-generated" if (use_llm and ANTHROPIC_API_KEY) else "Rule-based template"
            st.caption(f"Report type: {_rmode}")

        if st.session_state.get("static_pdf"):
            st.download_button(
                "⬇️ Download Clinical Report (PDF)",
                st.session_state.static_pdf,
                "fetal_hc_static_report.pdf",
                "application/pdf",
                use_container_width=True,
                key="dl_static_s",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CINE (Phase 2 / Phase 4b)
# ══════════════════════════════════════════════════════════════════════════════
with tab_cine:
    st.subheader("Cine Analysis — Temporal Attention Segmentation")

    # Model selector
    cine_model_choice = st.radio(
        "Model variant",
        ["Phase 2 — Temporal baseline (8.90M params)",
         "Phase 4b — Compressed temporal (5.20M params, −41.6%)"],
        horizontal=True, key="cine_model_choice",
        help="Phase 4b is the pruned version with the backbone compressed by 41.6%. "
             "The TemporalAttentionModule is preserved intact.",
    )
    use_phase4b = "4b" in cine_model_choice
    active_cine_model = phase4b_model if use_phase4b else phase2_model
    active_cine_name  = (
        "Phase 4b — Compressed temporal" if use_phase4b
        else "Phase 2 — Temporal baseline"
    )

    if use_phase4b:
        st.info(
            "✂️ **Compressed model active.** Phase 4b backbone is 41.6% smaller "
            "(5.20M vs 8.90M params). Pruned Dice (96.00%) marginally exceeds baseline (95.95%) — "
            "FT recovery rate was 103.8% (student surpassed teacher post-KD).",
            icon=None,
        )

    st.caption(
        "Upload a fetal head ultrasound image. The app generates a 16-frame synthetic "
        "cine-loop using Pseudo-LDDM v2 (Ornstein-Uhlenbeck motion + Rician speckle + "
        "depth attenuation), then runs the temporal attention model on the full sequence."
    )

    demo_subjects_c = get_demo_subjects()
    input_mode_c = st.radio(
        "Input mode",
        ["Use demo subject", "Upload your own image"] if demo_subjects_c else ["Upload your own image"],
        horizontal=True, key="mode_c",
    )

    _hc18_csv_c = _load_hc18_csv()
    img_gray_c = None
    if input_mode_c == "Use demo subject" and demo_subjects_c:
        selected_demo_c = st.selectbox(
            "Select demo subject", demo_subjects_c,
            format_func=lambda x: x.replace(".png", "").replace("_", " "),
            key="demo_c",
        )
        _csv_ps_c = _hc18_csv_c.get(selected_demo_c)
        if _csv_ps_c:
            _col_ca, _col_cb = st.columns([3, 1])
            _col_ca.caption(f"HC18 CSV pixel spacing for this image: **{_csv_ps_c:.4f} mm/px**")
            if _col_cb.button("Apply", key="apply_csv_c"):
                st.session_state.ps_input = _csv_ps_c
                st.session_state.ps_source = "csv"
                st.rerun()
        if st.button("Run cine analysis on demo subject", key="run_demo_c"):
            img_gray_c = load_demo_image(selected_demo_c)
    else:
        uploaded_c = st.file_uploader(
            "Upload ultrasound image (PNG or JPG, or .dcm DICOM)",
            type=["png", "jpg", "jpeg", "dcm"], key="cine",
        )
        if uploaded_c:
            _raw_c = uploaded_c.read()
            if uploaded_c.name.lower().endswith(".dcm"):
                _dicom_ps_c = _dicom_pixel_spacing(_raw_c)
                if _dicom_ps_c:
                    st.session_state.ps_input = _dicom_ps_c
                    st.session_state.ps_source = "dicom"
                    st.info(f"DICOM pixel spacing extracted: **{_dicom_ps_c:.4f} mm/px** ✅")
                img_gray_c = np.array(Image.open(io.BytesIO(_raw_c)).convert("L"))
            else:
                st.session_state.ps_source = "user"
                img_gray_c = np.array(Image.open(io.BytesIO(_raw_c)).convert("L"))

    if img_gray_c is not None:

        # Input validation
        validation_c = validate_input(img_gray_c)
        if validation_c["warnings"]:
            for w in validation_c["warnings"]:
                st.warning(f"⚠️ {w}")
            if not validation_c["valid"]:
                st.error(
                    "Input failed clinical sanity checks. "
                    "Please upload a valid grayscale ultrasound image."
                )
                st.stop()

        with st.spinner("Generating synthetic cine-loop (Pseudo-LDDM v2)..."):
            cine_frames = generate_cine(img_gray_c, n_frames=N_FRAMES, seed=42)
            gif_bytes   = frames_to_gif(cine_frames, fps=8)

        st.markdown("#### Synthetic cine-loop (Pseudo-LDDM v2 — 16 frames)")
        st.image(gif_bytes,
                 caption="OU probe motion + Rician speckle + depth attenuation",
                 use_column_width=True)
        st.markdown("---")

        with st.spinner(f"Running temporal attention inference ({active_cine_name})..."):
            result_c = predict_cine_clip(
                active_cine_model, cine_frames, pixel_spacing, device=DEVICE
            )
            xai_c = build_xai_panel(img_gray_c, result_c)

        update_session_stats(result_c["elapsed_ms"])
        st.session_state.cine_result = result_c
        st.session_state.cine_pdf = None

        st.markdown("#### Segmentation results")
        c1, c2, c3 = st.columns(3)
        c1.image(cv2.resize(img_gray_c, (INPUT_W, INPUT_H)),
                 caption="Input image", clamp=True, use_column_width=True)
        c2.image(result_c["overlay"],
                 caption="Temporal consensus segmentation (mean across 16 frames)",
                 clamp=True, use_column_width=True)
        if "uncertainty_overlay" in xai_c:
            c3.image(xai_c["uncertainty_overlay"],
                     caption="Boundary uncertainty (bright = high inter-frame disagreement)",
                     clamp=True, use_column_width=True)
        st.markdown("---")

        if result_c["hc_mm"] is not None:
            render_metrics_card(result_c, active_cine_name, pixel_spacing)
            st.markdown("---")

            # Temporal attention heatmap
            if "attn_heatmap" in xai_c:
                st.markdown("#### Temporal attention analysis")
                st.image(xai_c["attn_heatmap"],
                         caption="Left: T×T attention matrix · Right: per-frame attention received",
                         use_column_width=True)
                st.markdown("---")

            # Per-frame HC stability chart
            if result_c.get("per_frame_hc"):
                st.markdown("#### Per-frame HC stability")
                hc  = result_c["hc_mm"]
                std = result_c["hc_std_mm"]
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(result_c["per_frame_hc"], marker="o", markersize=4,
                        color="#2563eb", linewidth=1.5, label="Per-frame HC")
                ax.axhline(hc, color="#dc2626", linestyle="--", linewidth=1.5,
                           label=f"Consensus: {hc:.1f} mm")
                ax.fill_between(range(len(result_c["per_frame_hc"])),
                                hc - std, hc + std, alpha=0.2, color="#2563eb",
                                label=f"±1 std ({std:.2f} mm)")
                ax.set_xlabel("Frame")
                ax.set_ylabel("HC (mm)")
                ax.set_title("HC stability across the cine-loop")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
                st.markdown("---")

            render_gt_section(result_c, pixel_spacing, key_prefix="cine")
        else:
            st.warning("Could not estimate HC from consensus. Check pixel spacing.")

        with st.expander("ℹ️ Technical details — cine pipeline"):
            m = REPORTED_METRICS[active_cine_name]
            st.markdown(
                f"- **Cine generation:** Pseudo-LDDM v2 — OU motion + Rician speckle + depth attenuation ({N_FRAMES} frames)\n"
                f"- **Model:** {'Compressed backbone + TemporalAttentionModule (TAM preserved, backbone pruned 41.6%)' if use_phase4b else '2D Residual U-Net encoder + temporal self-attention (8 heads, 256-dim)'}\n"
                f"- **Parameters:** {m['params']} · **FLOPs/frame:** {m['flops']}\n"
                f"- **Reported Dice:** {m['dice']} · **Reported MAE:** {m['mae']} · **ISUOG:** {m['isuog']}\n"
                f"- **Runtime inference:** {result_c['elapsed_ms']:.1f} ms on {str(DEVICE).upper()}\n"
                f"- **Ablation (no attention):** Dice 81.48% · MAE 19.37mm\n"
                f"- **Pixel spacing:** {pixel_spacing:.4f} mm/px"
                + ("\n- **Pruning:** Hybrid Crossover + surgical decoder reconstruction + KD · 49.6% channel compression · Wilcoxon p=0.1013 (NS — statistically indistinguishable from baseline)" if use_phase4b else "")
            )

    # ── Persistent cine report-generation section ────────────────────────────
    _stored_c = st.session_state.get("cine_result")
    if _stored_c and _stored_c.get("hc_mm"):
        from datetime import date as _date_c, timedelta as _td_c
        st.markdown("---")
        st.success(
            f"✓ Cine analysis results loaded — HC **{_stored_c['hc_mm']:.1f} mm** · "
            f"GA **{_stored_c.get('ga_str', '—')}** · "
            f"Confidence **{_stored_c.get('confidence_label', '—')}**"
        )

        st.subheader("Generate Cine Clinical Report")

        demo_mode_c = st.toggle("Enable Demo Mode (pre-fill clinical scenario)", key="demo_mode_c")
        if demo_mode_c:
            _today_c = _date_c.today()
            _scenario_c = st.selectbox(
                "Select clinical scenario",
                [
                    "A — Normal 2nd trimester",
                    "B — LMP discordance (preterm risk)",
                    "C — IUGR / BPD mismatch",
                ],
                key="scenario_c",
            )
            if _scenario_c.startswith("A"):
                _cp_name, _cp_id   = "Demo Patient A", "HC18-DEMO-001"
                _cp_lmp            = (_today_c - _td_c(days=137)).strftime("%Y-%m-%d")
                _cp_ref, _cp_fac   = "Dr. Sarah Chen, OB/GYN", "City General Hospital"
                _cp_indic          = "Routine 2nd trimester anatomy scan"
                _cp_appr, _cp_qual = "transabdominal", "optimal"
                _cp_bpd, _cp_pres  = 0.0, "cephalic"
            elif _scenario_c.startswith("B"):
                _cp_name, _cp_id   = "Demo Patient B", "HC18-DEMO-002"
                _cp_lmp            = (_today_c - _td_c(days=104)).strftime("%Y-%m-%d")
                _cp_ref, _cp_fac   = "Dr. James Park, MFM", "University Medical Center"
                _cp_indic          = "LMP-size discordance — rule out growth restriction"
                _cp_appr, _cp_qual = "transabdominal", "suboptimal"
                _cp_bpd, _cp_pres  = 0.0, "cephalic"
            else:
                _cp_name, _cp_id   = "Demo Patient C", "HC18-DEMO-003"
                _cp_lmp            = (_today_c - _td_c(days=168)).strftime("%Y-%m-%d")
                _cp_ref, _cp_fac   = "Dr. Maria Santos, MFM", "Perinatology Associates"
                _cp_indic          = "Suspected IUGR — detailed biometry"
                _cp_appr, _cp_qual = "transabdominal", "suboptimal"
                _cp_bpd, _cp_pres  = 52.0, "cephalic"
        else:
            _cp_name = _cp_id = _cp_lmp = _cp_ref = _cp_fac = _cp_indic = ""
            _cp_appr, _cp_qual = "transabdominal", "optimal"
            _cp_bpd, _cp_pres  = 0.0, "cephalic"

        with st.expander("Patient & Exam Information", expanded=True):
            _cc1, _cc2 = st.columns(2)
            _cp_name_f  = _cc1.text_input("Patient Name",        value=_cp_name,  key="cp_name_c")
            _cp_id_f    = _cc2.text_input("Patient ID / MRN",    value=_cp_id,    key="cp_id_c")
            _cp_lmp_f   = _cc1.text_input("LMP (YYYY-MM-DD)",    value=_cp_lmp,   key="cp_lmp_c")
            _cp_ref_f   = _cc2.text_input("Referring Physician",  value=_cp_ref,   key="cp_ref_c")
            _cp_fac_f   = _cc1.text_input("Ordering Facility",   value=_cp_fac,   key="cp_fac_c")
            _cp_indic_f = _cc2.text_input("Clinical Indication",  value=_cp_indic, key="cp_indic_c")
            _ca1, _ca2, _ca3 = st.columns(3)
            _appr_opts_c = ["transabdominal", "transvaginal"]
            _qual_opts_c = ["optimal", "suboptimal", "limited"]
            _pres_opts_c = ["cephalic", "breech", "transverse", "not_assessed"]
            _cp_appr_f = _ca1.selectbox("US Approach",        _appr_opts_c, index=_appr_opts_c.index(_cp_appr), key="cp_appr_c")
            _cp_qual_f = _ca2.selectbox("Image Quality",      _qual_opts_c, index=_qual_opts_c.index(_cp_qual), key="cp_qual_c")
            _cp_pres_f = _ca3.selectbox("Fetal Presentation", _pres_opts_c, index=_pres_opts_c.index(_cp_pres), key="cp_pres_c")
            _cb1, _cb2 = st.columns(2)
            _cp_bpd_f   = _cb1.number_input("BPD (mm, optional)", value=_cp_bpd, min_value=0.0, max_value=150.0, step=0.1, key="cp_bpd_c")
            _cp_prior_f = _cb2.text_input("Prior Biometry", value="", key="cp_prior_c")
            st.markdown("**AI analysis results (read-only)**")
            _ra1c, _ra2c, _ra3c = st.columns(3)
            _ra1c.text_input("HC (mm)",    value=f"{_stored_c['hc_mm']:.1f}",             disabled=True, key="ra_hc_c")
            _ra2c.text_input("GA",         value=_stored_c.get("ga_str", "—"),             disabled=True, key="ra_ga_c")
            _ra3c.text_input("Confidence", value=_stored_c.get("confidence_label", "—"),  disabled=True, key="ra_conf_c")

        _ps_src_c = st.session_state.get("ps_source", "user")
        _needs_confirm_c = _ps_src_c == "user" and not pixel_spacing_confirmed
        if _needs_confirm_c:
            st.warning("Confirm pixel spacing in the sidebar before generating a clinical report.")

        if st.button(
            "Generate Cine Clinical Report (PDF)",
            disabled=_needs_confirm_c,
            type="primary",
            key="gen_report_c",
            use_container_width=True,
        ):
            class _PatientProxy_C:  # noqa: N801
                patient_name        = _cp_name_f or "—"
                patient_id          = _cp_id_f or None
                patient_dob         = None
                lmp                 = _cp_lmp_f or None
                ordering_facility   = _cp_fac_f or None
                referring_physician = _cp_ref_f or None
                sonographer_name    = None
                clinical_indication = _cp_indic_f or None
                study_date          = _date_c.today().strftime("%Y-%m-%d")
                us_approach         = _cp_appr_f
                image_quality       = _cp_qual_f
                pixel_spacing_mm    = pixel_spacing
                pixel_spacing_dicom_derived = (_ps_src_c == "dicom")
                pixel_spacing_source = _ps_src_c.upper()
                bpd_mm              = _cp_bpd_f if _cp_bpd_f > 0 else None
                fetal_presentation  = _cp_pres_f
                prior_biometry      = _cp_prior_f or None
                original_image_b64  = None
                overlay_image_b64   = None
                gradcam_image_b64   = None
                accession_number    = None
                report_mode         = "template"
            with st.spinner("Generating cine clinical report..."):
                st.session_state.cine_pdf = generate_cine_report(
                    _stored_c,
                    api_key=ANTHROPIC_API_KEY,
                    use_llm=use_llm,
                    model_name=active_cine_name,
                    pixel_spacing=pixel_spacing,
                    report=_PatientProxy_C(),
                    pixel_spacing_source=_ps_src_c.upper(),
                )
            _rmode_c = "LLM-generated" if (use_llm and ANTHROPIC_API_KEY) else "Rule-based template"
            st.caption(f"Report type: {_rmode_c}")

        if st.session_state.get("cine_pdf"):
            st.download_button(
                "⬇️ Download Cine Clinical Report (PDF)",
                st.session_state.cine_pdf,
                "fetal_hc_cine_report.pdf",
                "application/pdf",
                use_container_width=True,
                key="dl_cine_c",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEAD-TO-HEAD COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Head-to-Head Model Comparison")
    st.caption(
        "Run all four model variants on the same image simultaneously. "
        "This demonstrates compression efficiency: identical input, different "
        "model sizes, comparable accuracy."
    )

    demo_subjects_cmp = get_demo_subjects()
    input_mode_cmp = st.radio(
        "Input mode",
        ["Use demo subject", "Upload your own image"] if demo_subjects_cmp else ["Upload your own image"],
        horizontal=True, key="mode_cmp",
    )

    img_gray_cmp = None
    if input_mode_cmp == "Use demo subject" and demo_subjects_cmp:
        selected_demo_cmp = st.selectbox(
            "Select demo subject", demo_subjects_cmp,
            format_func=lambda x: x.replace(".png", "").replace("_", " "),
            key="demo_cmp",
        )
        if st.button("Run comparison on demo subject", key="run_demo_cmp"):
            img_gray_cmp = load_demo_image(selected_demo_cmp)
    else:
        uploaded_cmp = st.file_uploader(
            "Upload ultrasound image (PNG or JPG)",
            type=["png", "jpg", "jpeg"], key="cmp",
        )
        if uploaded_cmp:
            img_gray_cmp = np.array(Image.open(uploaded_cmp).convert("L"))

    if img_gray_cmp is not None:

        validation_cmp = validate_input(img_gray_cmp)
        if not validation_cmp["valid"]:
            for w in validation_cmp["warnings"]:
                st.warning(f"⚠️ {w}")
            st.error("Input failed sanity checks. Please upload a valid ultrasound image.")
            st.stop()

        with st.spinner("Running all four models — this takes ~5s on CPU..."):
            cmp_frames = generate_cine(img_gray_cmp, n_frames=N_FRAMES, seed=42)

            res_p0  = predict_single_frame(phase0_model,  img_gray_cmp, pixel_spacing, DEVICE)
            res_p4a = predict_single_frame(phase4a_model, img_gray_cmp, pixel_spacing, DEVICE)
            res_p2  = predict_cine_clip(phase2_model,  cmp_frames, pixel_spacing, DEVICE)
            res_p4b = predict_cine_clip(phase4b_model, cmp_frames, pixel_spacing, DEVICE)

        update_session_stats(
            res_p0["elapsed_ms"] + res_p4a["elapsed_ms"]
            + res_p2["elapsed_ms"] + res_p4b["elapsed_ms"]
        )

        # Segmentation panels
        st.markdown("#### Segmentation overlays")
        col1, col2, col3, col4 = st.columns(4)
        col1.image(res_p0["overlay"],  caption="Phase 0 baseline",    use_column_width=True)
        col2.image(res_p4a["overlay"], caption="Phase 4a compressed ✂️", use_column_width=True)
        col3.image(res_p2["overlay"],  caption="Phase 2 baseline",    use_column_width=True)
        col4.image(res_p4b["overlay"], caption="Phase 4b compressed ✂️", use_column_width=True)

        st.markdown("---")
        st.markdown("#### Measurement comparison")

        def _fmt(v, unit=""):
            return f"{v:.1f}{unit}" if v is not None else "—"

        rows = [
            ("Head Circumference (mm)",
             _fmt(res_p0["hc_mm"]), _fmt(res_p4a["hc_mm"]),
             _fmt(res_p2["hc_mm"]), _fmt(res_p4b["hc_mm"])),
            ("Gestational Age",
             res_p0.get("ga_str") or "—", res_p4a.get("ga_str") or "—",
             res_p2.get("ga_str") or "—",  res_p4b.get("ga_str") or "—"),
            ("Inference latency (ms)",
             f"{res_p0['elapsed_ms']:.1f}",  f"{res_p4a['elapsed_ms']:.1f}",
             f"{res_p2['elapsed_ms']:.1f}",  f"{res_p4b['elapsed_ms']:.1f}"),
        ]

        m_p0  = REPORTED_METRICS["Phase 0 — Static baseline"]
        m_p4a = REPORTED_METRICS["Phase 4a — Compressed static"]
        m_p2  = REPORTED_METRICS["Phase 2 — Temporal baseline"]
        m_p4b = REPORTED_METRICS["Phase 4b — Compressed temporal"]

        st.markdown(
            "| Metric | Phase 0 | Phase 4a ✂️ | Phase 2 | Phase 4b ✂️ |\n"
            "|--------|---------|------------|---------|------------|\n"
            f"| HC (this image) | {_fmt(res_p0['hc_mm'])} mm | {_fmt(res_p4a['hc_mm'])} mm | {_fmt(res_p2['hc_mm'])} mm | {_fmt(res_p4b['hc_mm'])} mm |\n"
            f"| GA (this image) | {res_p0.get('ga_str') or '—'} | {res_p4a.get('ga_str') or '—'} | {res_p2.get('ga_str') or '—'} | {res_p4b.get('ga_str') or '—'} |\n"
            f"| Inference (ms) | {res_p0['elapsed_ms']:.1f} | {res_p4a['elapsed_ms']:.1f} | {res_p2['elapsed_ms']:.1f} | {res_p4b['elapsed_ms']:.1f} |\n"
            f"| Reported Dice | {m_p0['dice']} | {m_p4a['dice']} | {m_p2['dice']} | {m_p4b['dice']} |\n"
            f"| Reported MAE | {m_p0['mae']} | {m_p4a['mae']} | {m_p2['mae']} | {m_p4b['mae']} |\n"
            f"| Parameters | {m_p0['params']} | {m_p4a['params']} | {m_p2['params']} | {m_p4b['params']} |\n"
            f"| Compression | — | −43.7% ✂️ | — | −41.6% ✂️ |\n"
            f"| FLOPs | {m_p0['flops']} | {m_p4a['flops']} | {m_p2['flops']} | {m_p4b['flops']} |\n"
            f"| ISUOG ≤3mm | {m_p0['isuog']} | {m_p4a['isuog']} | {m_p2['isuog']} | {m_p4b['isuog']} |"
        )

        st.markdown("---")
        st.info(
            "**Interpretation:** Phase 4a and 4b achieve near-identical measurements to their "
            "full-size counterparts with 41–44% fewer parameters and 23% fewer FLOPs. "
            "Both compressed models satisfy the ISUOG ≤3mm clinical MAE threshold. "
            "Latency improvement is visible in the inference column above.",
            icon=None,
        )

        st.markdown("---")
        st.markdown("#### Comparative Clinical Report")
        st.caption(
            "Download a single PDF report covering all four models — includes four-model "
            "measurement table, clinical recommendation on static vs cine, compression "
            "deployment significance, and ablation context. "
            + ("LLM clinical narrative active." if (use_llm and ANTHROPIC_API_KEY) else "Rule-based template active.")
        )
        with st.spinner("Generating comparative clinical report..."):
            pdf_cmp = generate_comparison_report(
                results={
                    "phase0":  res_p0,
                    "phase4a": res_p4a,
                    "phase2":  res_p2,
                    "phase4b": res_p4b,
                },
                api_key=ANTHROPIC_API_KEY,
                use_llm=use_llm,
                pixel_spacing=pixel_spacing,
            )
        st.download_button(
            "⬇️ Download Comparative Clinical Report (PDF)", pdf_cmp,
            "fetal_hc_comparative_report.pdf", "application/pdf",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL CARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_card:
    render_model_card()


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.82em;'>"
    "⚠️ Research prototype · Not FDA-cleared · Requires sonographer verification before clinical use<br>"
    "SaMD Class II candidate · HC18 dataset, Radboud UMC, Netherlands · "
    "Tarun Sadarla — MS Artificial Intelligence, University of North Texas, 2026"
    "</div>",
    unsafe_allow_html=True,
)
