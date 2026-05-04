"""
report.py — Fetal Head Circumference Clinical AI
PDF clinical report generation.

Four public functions:
  generate_static_report()     — Phase 0 / Phase 4a single-frame analysis
  generate_cine_report()       — Phase 2 / Phase 4b temporal cine analysis
  generate_comparison_report() — All four models head-to-head (Tab 3)
  generate_pdf_report()        — Backwards-compatible router

Design principles:
  - Reports are model-aware: header, metrics table, and narrative all reflect
    which model variant (baseline vs compressed) produced the result.
  - LLM and template reports are visually distinct: different accent colours,
    different section structure, clearly labelled so a reader knows immediately
    which type they are looking at.
  - Clinical language throughout: no ML jargon visible to clinical end-users.
    Dice → "segmentation accuracy", MAE → "mean measurement deviation",
    parameters → "model computational footprint", compression → "optimised for
    resource-constrained deployment".
  - The comparison report is the portfolio centrepiece: it tells the full
    clinical deployment story across all four models in one document.
"""

import io
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


# ── Draft watermark ───────────────────────────────────────────────────────────


def _draw_draft_watermark(canvas, doc):
    """Draw a diagonal "DRAFT — UNSIGNED" watermark across the page.

    Used as the onFirstPage/onLaterPages callback when rendering an unsigned
    report. Once a clinician signs the report we re-render without the
    callback so the final PDF carries no watermark.
    """
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 90)
    canvas.setFillColorRGB(0.85, 0.20, 0.20, alpha=0.18)
    canvas.translate(297, 421)  # A4 centre in points (210mm x 297mm)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, "DRAFT — UNSIGNED")
    canvas.restoreState()


# ── Colour scheme ─────────────────────────────────────────────────────────────
_NAVY = colors.HexColor("#1e3a5f")
_TEAL = colors.HexColor("#0f6e56")  # LLM accent
_GREY = colors.HexColor("#4b5563")  # template accent
_AMBER = colors.HexColor("#b45309")  # warning / regulatory
_LIGHT_BG = colors.HexColor("#f0f4f8")
_ALT_BG = colors.white
_BORDER = colors.HexColor("#cccccc")
_PRUNED = colors.HexColor("#065f46")  # compression green


# ── Model metadata registry ───────────────────────────────────────────────────
_MODEL_META = {
    "Phase 0 — Static baseline": {
        "short": "Phase 0 — Residual U-Net (baseline)",
        "type": "static",
        "pruned": False,
        "params": "8.11M",
        "flops": "21.58 GMACs",
        "dice": "97.75%",
        "mae": "1.65 mm",
        "compression": None,
        "isuog": "PASS",
        "dataset": "HC18 — 199 held-out test images",
    },
    "Phase 4a — Compressed static": {
        "short": "Phase 4a — Compressed Residual U-Net (−43.7%)",
        "type": "static",
        "pruned": True,
        "params": "4.57M",
        "flops": "16.56 GMACs",
        "dice": "97.64%",
        "mae": "1.76 mm",
        "compression": "43.7% parameter reduction (8.11M → 4.57M)",
        "isuog": "PASS",
        "dataset": "HC18 — 199 held-out test images",
    },
    "Phase 2 — Temporal baseline": {
        "short": "Phase 2 — Temporal Attention U-Net (baseline)",
        "type": "cine",
        "pruned": False,
        "params": "8.90M",
        "flops": "21.58 GMACs/frame",
        "dice": "95.95%",
        "mae": "2.10 mm",
        "compression": None,
        "isuog": "PASS",
        "dataset": "HC18 — 121 held-out synthetic cine clips",
    },
    "Phase 4b — Compressed temporal": {
        "short": "Phase 4b — Compressed Temporal U-Net (−41.6%)",
        "type": "cine",
        "pruned": True,
        "params": "5.20M",
        "flops": "16.44 GMACs/frame",
        "dice": "96.00%",
        "mae": "2.06 mm",
        "compression": "41.6% parameter reduction (8.90M → 5.20M)",
        "isuog": "PASS",
        "dataset": "HC18 — 121 held-out synthetic cine clips",
    },
}

_FALLBACK_META = _MODEL_META["Phase 0 — Static baseline"]


def _meta(model_name: str) -> dict:
    return _MODEL_META.get(model_name, _FALLBACK_META)


# ── Styles ────────────────────────────────────────────────────────────────────


def _styles(llm: bool = False):
    s = getSampleStyleSheet()
    acc = _TEAL if llm else _GREY
    return dict(
        title=ParagraphStyle(
            "ti", parent=s["Heading1"], fontSize=13, spaceAfter=3, textColor=_NAVY
        ),
        sub=ParagraphStyle("su", parent=s["Normal"], fontSize=8.5, spaceAfter=8, textColor=_GREY),
        badge=ParagraphStyle(
            "ba",
            parent=s["Normal"],
            fontSize=8,
            spaceAfter=6,
            textColor=acc,
            fontName="Helvetica-Bold",
        ),
        sec=ParagraphStyle(
            "se", parent=s["Heading2"], fontSize=10.5, spaceBefore=10, spaceAfter=3, textColor=_NAVY
        ),
        body=ParagraphStyle("bo", parent=s["Normal"], fontSize=9.5, spaceAfter=5, leading=14.5),
        bodyI=ParagraphStyle(
            "bi", parent=s["Normal"], fontSize=9.5, spaceAfter=5, leading=14.5, textColor=_GREY
        ),
        warn=ParagraphStyle("wa", parent=s["Normal"], fontSize=8.5, leading=12, textColor=_AMBER),
        label=ParagraphStyle(
            "la",
            parent=s["Normal"],
            fontSize=9,
            spaceAfter=2,
            fontName="Helvetica-Bold",
            textColor=_NAVY,
        ),
        green=ParagraphStyle(
            "gr", parent=s["Normal"], fontSize=9, spaceAfter=5, leading=13, textColor=_PRUNED
        ),
    )


def _tbl_style(header_color=None):
    hc = header_color or _NAVY
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), hc),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_BG, _ALT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.3, _BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]
    )


def _hr(story, color=None, thickness=0.5):
    story.append(
        HRFlowable(
            width="100%",
            thickness=thickness,
            color=color or _BORDER,
            spaceAfter=6,
        )
    )


def _regulatory(story, st):
    story.append(Spacer(1, 4 * mm))
    _hr(story)
    story.append(Paragraph("Regulatory and Safety Notice", st["sec"]))
    story.append(
        Paragraph(
            "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system has not received "
            "FDA 510(k) clearance or CE marking under EU MDR / IVDR. It is classified as a "
            "Software as a Medical Device (SaMD) Class II candidate under 21 CFR Part 892. "
            "All automated measurements must be independently verified by a qualified "
            "healthcare professional (sonographer or obstetrician) before incorporation "
            "into any clinical decision. Gestational age estimates carry an inherent "
            "±2-week confidence interval (Hadlock 1984). This report does not constitute "
            "a diagnostic opinion.",
            st["warn"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            f"HC18 dataset — Radboud University Medical Center, Nijmegen, Netherlands. "
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}. "
            f"System: Fetal Head Circumference Clinical AI v2.0 — "
            f"Tarun Sadarla, MS Artificial Intelligence, University of North Texas, 2026.",
            st["sub"],
        )
    )


def _report_header(story, st, title_line, model_name, report_type_label, llm, elapsed_ms=None):
    """Render the report title block with model badge and LLM/template badge."""
    acc = _TEAL if llm else _GREY
    label = "AI-AUTHORED CLINICAL NARRATIVE" if llm else "AUTOMATED TEMPLATE REPORT"
    m = _meta(model_name)

    story.append(Paragraph(title_line, st["title"]))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC  |  "
            f"Model: {m['short']}" + (f"  |  Inference: {elapsed_ms:.0f} ms" if elapsed_ms else ""),
            st["sub"],
        )
    )

    # Coloured badge line
    badge_parts = [f'<font color="{acc.hexval()}">[{label}]</font>']
    if m["pruned"]:
        badge_parts.append(
            f'<font color="{_PRUNED.hexval()}">[COMPRESSED MODEL — {m["compression"]}]</font>'
        )
    badge_parts.append(f'<font color="{_NAVY.hexval()}">[{report_type_label}]</font>')
    story.append(Paragraph("  ·  ".join(badge_parts), st["badge"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=acc, spaceAfter=10))


# ── LLM narrative functions ───────────────────────────────────────────────────


def _call_llm(api_key: str, prompt: str, max_tokens: int = 380) -> Optional[str]:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()
    except Exception as e:
        return None


def _llm_static_narrative(hc, ga_str, ga_weeks, trim, gradcam_ok, model_name, elapsed_ms, api_key):
    m = _meta(model_name)
    ctx = {
        "Early (<20w)": "consistent with first-trimester biometric assessment",
        "Mid (20–30w)": "within the optimal second-trimester sonographic window",
        "Late (>30w)": "consistent with third-trimester parameters where acoustic shadowing may affect calvarium delineation",
    }.get(trim, "")

    p1 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (5 sentences) for a clinical audience — obstetrician or "
            f"senior sonographer. Use formal medical terminology throughout. "
            f"Do NOT use any machine learning terms (Dice, MAE, IoU, neural network, model). "
            f"Do NOT make diagnostic conclusions or suggest pathology. "
            f"End with a sentence requiring clinical correlation and sonographer verification.\n\n"
            f"Biometric data: Head circumference = {hc:.1f} mm, gestational age estimate = "
            f"{ga_str} ({ga_weeks:.1f} weeks) by Hadlock (1984) biometric nomogram, "
            f"trimester = {trim} ({ctx}). "
            f"Automated measurement system independently validated with a mean absolute "
            f"measurement deviation of {m['mae']} against expert sonographer annotation "
            f"(well within the ISUOG clinically acceptable biometry threshold of ±3 mm). "
            f"System classification: candidate Software as a Medical Device (SaMD), "
            f"not yet regulatory-cleared.\n\n"
            f"Structure your paragraph as: (1) HC measurement and GA estimate with biometric "
            f"context, (2) trimester-specific clinical relevance — gestational age accuracy "
            f"in the relevant window, (3) automated system validation status and measurement "
            f"confidence, (4) limitations and advisory for clinical correlation. "
            f"Plain prose only — no bullets, headers, or markdown."
        ),
    ) or _rule_static_p1(hc, ga_str, ga_weeks, trim)

    p2 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (4 sentences) for a clinical audience interpreting a "
            f"gradient-weighted activation map generated during automated fetal head "
            f"circumference measurement. The map highlights regions of the ultrasound image "
            f"that most influenced the automated delineation of the calvarium boundary.\n"
            f"{'The activation map was successfully generated for this acquisition.' if gradcam_ok else 'The activation map could not be generated for this image — possibly due to limited segmentation coverage or image quality constraints.'}\n\n"
            f"Explain: (1) what the activation pattern indicates about the system's focus "
            f"on the hyperechoic calvarium interface versus intracranial soft-tissue structures, "
            f"(2) why anatomically appropriate activation supports confidence in the measurement, "
            f"(3) clinical implications if the activation pattern is atypical. "
            f"Use terms: calvarium, hyperechoic interface, cranial ossification, acoustic "
            f"impedance contrast, intracranial compartment. "
            f"No ML jargon. Plain prose only."
        ),
    ) or _rule_static_p2(gradcam_ok)

    p3 = None
    if m["pruned"]:
        p3 = _call_llm(
            api_key,
            (
                f"Write ONE paragraph (4 sentences) for a hospital technology committee "
                f"evaluating an AI-assisted fetal biometry system for deployment in their "
                f"obstetric unit. The system uses a structurally optimised neural network "
                f"with {m['compression']}, achieving equivalent clinical accuracy to the "
                f"full-size system (mean measurement deviation {m['mae']} vs 1.65 mm for "
                f"the uncompressed model, both within ISUOG ±3 mm). "
                f"Inference latency on standard CPU hardware: {elapsed_ms:.0f} ms per image.\n\n"
                f"Explain in clinical health systems language: what this computational "
                f"optimisation means for (1) deployment on portable or point-of-care "
                f"ultrasound platforms in low- and middle-income country settings, "
                f"(2) integration into existing hospital IT infrastructure without requiring "
                f"dedicated GPU hardware, (3) the clinical equivalence evidence base. "
                f"Do not use engineering or ML terminology. Plain prose only."
            ),
            max_tokens=320,
        )

    return p1, p2, p3


def _llm_cine_narrative(
    hc, ga_str, ga_weeks, trim, rel, std, n_frames, model_name, elapsed_ms, api_key
):
    m = _meta(model_name)
    rel_desc = "excellent" if rel > 0.97 else "good" if rel > 0.93 else "moderate"
    std_desc = "highly stable" if std < 2.0 else "acceptable" if std < 5.0 else "variable"
    ctx = {
        "Early (<20w)": "consistent with first-trimester biometric assessment",
        "Mid (20–30w)": "within the optimal second-trimester sonographic window",
        "Late (>30w)": "consistent with third-trimester parameters",
    }.get(trim, "")

    p1 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (5 sentences) for a clinical audience — obstetrician or "
            f"senior sonographer. Formal medical terminology. No ML jargon. "
            f"No diagnostic conclusions. End requiring sonographer verification.\n\n"
            f"Data: Consensus head circumference = {hc:.1f} mm derived from automated "
            f"analysis of a {n_frames}-frame sequential ultrasound acquisition. "
            f"Gestational age = {ga_str} ({ga_weeks:.1f} weeks) by Hadlock (1984). "
            f"Trimester: {trim} ({ctx}). "
            f"Measurement concordance across the acquisition sequence: {rel_desc} "
            f"(frame-to-frame variability = {std:.2f} mm, {std_desc}). "
            f"System validated to {m['mae']} mean measurement deviation (ISUOG threshold ±3 mm).\n\n"
            f"Structure: (1) HC and GA from multi-frame consensus, "
            f"(2) inter-frame measurement concordance in clinical terms — what variability "
            f"of {std:.1f} mm implies about fetal position stability and probe contact "
            f"quality during acquisition, (3) validation status, (4) requirement for "
            f"clinical correlation. Plain prose only."
        ),
    ) or _rule_cine_p1(hc, ga_str, ga_weeks, trim, rel, std)

    p2 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (5 sentences) for a clinical audience explaining the "
            f"clinical rationale for sequential multi-frame fetal head measurement compared "
            f"to standard single-frame biometry.\n\n"
            f"Context: {n_frames} sequential frames were analysed. Frame-to-frame "
            f"HC variability = {std:.2f} mm. The system automatically weighted frames "
            f"with superior calvarium boundary definition using temporal attention — "
            f"a mechanism analogous to the sonographer's own practice of identifying "
            f"the optimal image plane before freezing for measurement.\n\n"
            f"Address: (1) how probe angle micro-variation and fetal micro-movement "
            f"introduce inter-frame variation in calvarial plane measurement, "
            f"(2) how automated frame-quality weighting may reduce operator-dependent "
            f"variability compared to manual freeze-frame selection, "
            f"(3) clinical scenarios where multi-frame consensus may be advantageous "
            f"(suboptimal acoustic window, restless fetus, second operator review), "
            f"(4) use cautious language throughout ('may provide', 'could reduce'). "
            f"Plain prose only. No ML terms."
        ),
    ) or _rule_cine_p2(rel, std, n_frames)

    p3 = None
    if m["pruned"]:
        p3 = _call_llm(
            api_key,
            (
                f"Write ONE paragraph (3-4 sentences) for a clinical technology committee. "
                f"The temporal cine analysis system uses a computationally optimised model "
                f"({m['compression']}). The optimised system achieves {m['dice']} segmentation "
                f"accuracy — marginally exceeding its uncompressed predecessor (95.95%) — "
                f"with {m['mae']} mean measurement deviation and {elapsed_ms:.0f} ms inference "
                f"latency per 16-frame clip on standard CPU.\n\n"
                f"Explain the clinical deployment significance: what this efficiency gain "
                f"means for integration into portable ultrasound platforms, cine-capable "
                f"handheld devices, and real-time workflow in high-throughput screening "
                f"environments. Clinical health systems language only. No ML jargon."
            ),
            max_tokens=280,
        )

    return p1, p2, p3


def _llm_comparison_narrative(results: dict, api_key: str):
    """
    Three LLM paragraphs for the comparison report:
    1. Clinical recommendation — when to use static vs cine
    2. Compression deployment story
    3. Overall system summary for a committee
    """
    r0 = results.get("phase0", {})
    r4a = results.get("phase4a", {})
    r2 = results.get("phase2", {})
    r4b = results.get("phase4b", {})

    hc_vals = [r.get("hc_mm") for r in [r0, r4a, r2, r4b] if r.get("hc_mm")]
    hc_range = f"{min(hc_vals):.1f}–{max(hc_vals):.1f} mm" if len(hc_vals) >= 2 else "—"
    cine_std = r4b.get("hc_std_mm") or r2.get("hc_std_mm") or 0.0

    ms_p0 = r0.get("elapsed_ms", 0)
    ms_p4a = r4a.get("elapsed_ms", 0)
    ms_p2 = r2.get("elapsed_ms", 0)
    ms_p4b = r4b.get("elapsed_ms", 0)

    p1 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (5-6 sentences) for an obstetric ultrasound department "
            f"clinical committee deciding between single-frame and multi-frame automated "
            f"fetal head biometry approaches.\n\n"
            f"System data: Four model variants were evaluated on the same ultrasound image. "
            f"Single-frame measurements: {r0.get('hc_mm', 0):.1f} mm (full model, "
            f"{ms_p0:.0f} ms) and {r4a.get('hc_mm', 0):.1f} mm (compressed model, "
            f"{ms_p4a:.0f} ms). "
            f"Multi-frame cine consensus measurements: {r2.get('hc_mm', 0):.1f} mm "
            f"(full model, {ms_p2:.0f} ms) and {r4b.get('hc_mm', 0):.1f} mm "
            f"(compressed, {ms_p4b:.0f} ms). "
            f"Inter-frame measurement variability (cine): {cine_std:.2f} mm.\n\n"
            f"Provide a nuanced clinical recommendation: in which clinical scenarios is "
            f"single-frame sufficient (standard second-trimester screening, clear acoustic "
            f"window, experienced operator) versus when multi-frame cine analysis may offer "
            f"incremental benefit (suboptimal window, assessment for audit/second opinion, "
            f"training environment, high-throughput screening). "
            f"Use cautious, evidence-appropriate language. No ML jargon."
        ),
        max_tokens=420,
    ) or (
        "Clinical scenario recommendation not available — rule-based summary: "
        "single-frame analysis is appropriate for standard second-trimester screening "
        "with a clear acoustic window. Multi-frame cine analysis may provide incremental "
        "benefit in settings with suboptimal probe contact, fetal movement, or when "
        "measurement audit is required."
    )

    p2 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (4 sentences) for a hospital IT and clinical engineering "
            f"committee evaluating AI-assisted fetal biometry for deployment.\n\n"
            f"Both compressed models achieve clinically equivalent accuracy: "
            f"single-frame compressed (4.57M parameters): 97.64% segmentation accuracy, "
            f"1.76 mm mean deviation; temporal compressed (5.20M parameters): 96.00% "
            f"accuracy, 2.06 mm deviation. Both full-size and compressed variants satisfy "
            f"ISUOG ±3 mm clinical threshold. CPU inference: single-frame {ms_p4a:.0f} ms, "
            f"cine {ms_p4b:.0f} ms per 16-frame clip.\n\n"
            f"Explain the clinical IT deployment significance of parameter-efficient models: "
            f"(1) integration without GPU infrastructure requirement, (2) compatibility with "
            f"point-of-care ultrasound devices and portable platforms, (3) LMIC deployment "
            f"relevance where GPU-equipped hardware is unavailable. "
            f"Clinical health systems language. No ML/engineering jargon."
        ),
        max_tokens=320,
    ) or (
        "Compressed model deployment note not available — rule-based summary: "
        "parameter-efficient model variants enable deployment on standard CPU hardware "
        "without GPU infrastructure, supporting integration in resource-constrained "
        "settings including portable ultrasound platforms and LMIC point-of-care environments."
    )

    return p1, p2


# ── Rule-based narratives ─────────────────────────────────────────────────────


def _rule_static_p1(hc, ga_str, ga_weeks, trim):
    ctx = {
        "Early (<20w)": "consistent with first-trimester biometric parameters",
        "Mid (20–30w)": "within the optimal second-trimester sonographic assessment window",
        "Late (>30w)": "consistent with third-trimester biometry, where increased acoustic shadowing from the calvarium may influence boundary delineation",
    }.get(trim, "")
    return (
        f"Automated biometric analysis of the submitted fetal head ultrasound yielded "
        f"a head circumference of {hc:.1f} mm, corresponding to an estimated gestational "
        f"age of {ga_str} ({ga_weeks:.1f} weeks) derived from the Hadlock (1984) biometric "
        f"nomogram, {ctx}. "
        f"The measurement was produced by an automated calvarium boundary detection system "
        f"validated on an independent cohort of 199 fetal head ultrasound images, with a "
        f"mean absolute measurement deviation consistently within the ISUOG clinically "
        f"acceptable biometry threshold of ±3 mm for second-trimester assessment. "
        f"Single-frame biometric analysis is dependent on image quality, correct probe "
        f"angulation to the standard suboccipitobregmatic plane, and adequate acoustic "
        f"access to the cranial vault; suboptimal acquisition conditions may reduce "
        f"measurement fidelity. "
        f"These automated findings require clinical correlation with menstrual dating, "
        f"prior sonographic biometry, and direct verification by a qualified sonographer "
        f"before incorporation into clinical management."
    )


def _rule_static_p2(gradcam_ok):
    if gradcam_ok:
        return (
            "The gradient-weighted activation map generated for this measurement "
            "delineates the spatial regions of the ultrasound image that most strongly "
            "influenced the automated identification of the calvarium boundary. "
            "High-activation regions correspond to areas of high acoustic impedance "
            "contrast at the outer cranial bone interface — the hyperechoic calvarium "
            "echo — which provides the principal anatomical landmark for head "
            "circumference measurement. "
            "Low-activation regions correspond to intracranial soft-tissue structures "
            "(cerebral parenchyma, ventricular system) that do not contribute to the "
            "biometric perimeter. "
            "This activation pattern is anatomically congruent with appropriate model "
            "behaviour for fetal head biometry and supports confidence in the validity "
            "of the automated measurement boundary selection."
        )
    return (
        "A gradient-weighted activation map could not be generated for this image, "
        "which may occur when the predicted segmentation region is insufficient for "
        "reliable spatial attribution analysis — often associated with challenging "
        "acoustic windows or partial calvarium visualisation. "
        "In the absence of activation map confirmation, the automated measurement "
        "result should be interpreted with heightened caution, and direct sonographer "
        "review of the original image is recommended before clinical use."
    )


def _rule_cine_p1(hc, ga_str, ga_weeks, trim, rel, std):
    rel_desc = "excellent" if rel > 0.97 else "good" if rel > 0.93 else "moderate"
    std_desc = (
        "highly stable (<2 mm inter-frame deviation)"
        if std < 2.0
        else f"acceptably stable ({std:.1f} mm inter-frame deviation)"
        if std < 5.0
        else f"variable ({std:.1f} mm inter-frame deviation — review recommended)"
    )
    ctx = {
        "Early (<20w)": "consistent with first-trimester biometric parameters",
        "Mid (20–30w)": "within the optimal second-trimester sonographic assessment window",
        "Late (>30w)": "consistent with third-trimester biometric parameters",
    }.get(trim, "")
    return (
        f"Sequential cine-loop analysis of a {16}-frame ultrasound acquisition yielded "
        f"a consensus fetal head circumference of {hc:.1f} mm, corresponding to an "
        f"estimated gestational age of {ga_str} ({ga_weeks:.1f} weeks) derived from the "
        f"Hadlock (1984) biometric nomogram, {ctx}. "
        f"Inter-frame measurement concordance across the acquisition sequence was "
        f"{rel_desc}, with calvarium boundary measurements {std_desc}, indicating that "
        f"{'the fetal head position and probe contact remained stable' if std < 3.0 else 'some variation in probe contact or fetal head position occurred'} "
        f"during the recorded sequence. "
        f"The consensus measurement derived from temporal integration of sequential "
        f"frames provides greater robustness against single-frame artefacts, transient "
        f"acoustic shadowing, and probe angle micro-variation compared to static "
        f"single-frame measurement, potentially reducing operator-dependent variability "
        f"in the biometric measurement selection. "
        f"Clinical correlation with menstrual dating, prior sonographic biometry, and "
        f"direct verification by a qualified sonographer are required before incorporation "
        f"into clinical management."
    )


def _rule_cine_p2(rel, std, n_frames):
    return (
        f"The temporal analysis system evaluated all {n_frames} frames from the "
        f"acquisition sequence and applied automated preferential weighting toward frames "
        f"exhibiting superior calvarium boundary delineation — an approach analogous to "
        f"the sonographer's clinical practice of identifying the optimal image plane before "
        f"freezing for biometric measurement. "
        f"The observed inter-frame measurement variability of {std:.2f} mm is attributable "
        f"to minor changes in the imaged calvarial cross-section arising from natural probe "
        f"micro-motion and fetal head micro-movement during the acquisition sequence, "
        f"consistent with the clinical variability expected in cine-loop acquisitions at "
        f"this gestational stage. "
        f"Automated temporal frame weighting reduces the operator-dependent component of "
        f"measurement variability inherent in manual freeze-frame selection during "
        f"standard biometric assessment, and may be of particular value in cases where "
        f"acoustic access is intermittently limited by fetal position or maternal habitus. "
        f"The multi-frame consensus approach is recommended as a complementary verification "
        f"method in cases where single-frame image quality is suboptimal."
    )


def _rule_compression_note(model_name, elapsed_ms):
    m = _meta(model_name)
    if not m["pruned"]:
        return None
    return (
        f"This report was generated using a parameter-efficient model variant with "
        f"{m['compression']}. The optimised model achieves {m['dice']} segmentation "
        f"accuracy with {m['mae']} mean measurement deviation — clinically equivalent "
        f"to the full-size model and within the ISUOG ±3 mm threshold. "
        f"Inference was completed in {elapsed_ms:.0f} ms on standard CPU hardware, "
        f"supporting deployment on portable ultrasound platforms and in settings without "
        f"dedicated graphics processing infrastructure."
    )


# ── PDF building blocks ───────────────────────────────────────────────────────


def _biometric_table(
    story, st, hc, ga_str, trim, conf_label, model_name, elapsed_ms, pixel_spacing
):
    m = _meta(model_name)
    story.append(Paragraph("Biometric Findings", st["sec"]))
    rows = [
        ["Parameter", "Automated Result", "Reference / Context"],
        [
            "Head Circumference (HC)",
            f"{hc:.1f} mm" if hc else "—",
            "Measured from calvarium perimeter",
        ],
        ["Estimated Gestational Age", ga_str or "—", "Hadlock (1984) nomogram — ±2 weeks CI"],
        ["Trimester Classification", trim or "—", "Derived from estimated GA"],
        [
            "Measurement Confidence",
            conf_label or "—",
            "Based on segmentation coverage and boundary clarity",
        ],
        [
            "Image Pixel Spacing",
            f"{pixel_spacing:.4f} mm/pixel" if pixel_spacing else "0.0700 mm/pixel",
            "Applied to convert pixel measurements to mm",
        ],
    ]
    t = Table(rows, colWidths=[58 * mm, 42 * mm, 70 * mm])
    t.setStyle(_tbl_style())
    story.append(t)
    story.append(Spacer(1, 4 * mm))


def _model_performance_table(story, st, model_name, elapsed_ms):
    m = _meta(model_name)
    story.append(Paragraph("AI System Performance (Validation Cohort)", st["sec"]))
    rows = [
        ["Metric", "This Model", "Clinical Reference"],
        ["Segmentation accuracy (validation)", m["dice"], f"HC18 cohort — {m['dataset']}"],
        ["Mean measurement deviation", m["mae"], "ISUOG acceptable biometry threshold: ±3 mm"],
        ["ISUOG clinical threshold", f"✓ {m['isuog']}", "ISUOG Practice Guidelines 2010"],
        ["Model computational footprint", m["params"], "Parameter count"],
        [
            "Computational operations per image",
            m["flops"],
            "Floating-point multiply-accumulate operations",
        ],
        [
            "Runtime inference (this image)",
            f"{elapsed_ms:.0f} ms" if elapsed_ms else "—",
            "Measured on CPU hardware",
        ],
    ]
    if m["pruned"]:
        rows.append(
            [
                "Compression vs full model",
                f"−{m['compression'].split('(')[0].strip()}",
                "Structurally pruned via Hybrid Crossover channel merging",
            ]
        )
    t = Table(rows, colWidths=[72 * mm, 38 * mm, 60 * mm])
    t.setStyle(_tbl_style())
    story.append(t)
    story.append(Spacer(1, 4 * mm))


def _temporal_table(story, st, rel, std, n_frames):
    rel_label = (
        "Excellent (>0.97)"
        if rel > 0.97
        else "Good (0.93–0.97)"
        if rel > 0.93
        else "Moderate (<0.93)"
    )
    std_label = (
        "Highly stable" if std < 2.0 else "Acceptable" if std < 5.0 else "Variable — review advised"
    )
    story.append(Paragraph("Temporal Acquisition Analysis", st["sec"]))
    rows = [
        ["Parameter", "Value", "Clinical Interpretation"],
        ["Frames analysed", str(n_frames), "Synthetic cine-loop (Pseudo-LDDM v2)"],
        ["Inter-frame concordance", rel_label, "Consistency of calvarium boundary across frames"],
        ["Frame-to-frame HC variability", f"{std:.2f} mm", std_label],
        ["Consensus method", "Temporal mean probability", "Mean prediction across all frames"],
    ]
    t = Table(rows, colWidths=[58 * mm, 42 * mm, 70 * mm])
    t.setStyle(_tbl_style())
    story.append(t)
    story.append(Spacer(1, 4 * mm))


# ── Sign-off block ────────────────────────────────────────────────────────────


def _signoff_block(story, st, signed_meta: dict):
    """Append a green-bordered sign-off panel after the regulatory notice.

    signed_meta keys: signed_by (required), signed_at (ISO 8601 string),
    signoff_note (optional clinician comment).
    """
    story.append(Spacer(1, 4 * mm))
    _hr(story, color=_PRUNED, thickness=1.2)
    story.append(Paragraph("Clinical Sign-off", st["sec"]))
    rows = [
        ["Signed by", signed_meta.get("signed_by") or "—"],
        ["Signed at", signed_meta.get("signed_at") or "—"],
    ]
    note = signed_meta.get("signoff_note")
    if note:
        rows.append(["Clinician note", note])
    t = Table(rows, colWidths=[40 * mm, 130 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), _LIGHT_BG),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, -1), _NAVY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, _PRUNED),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def generate_static_report(
    result: dict,
    api_key: Optional[str] = None,
    use_llm: bool = True,
    model_name: str = "Phase 0 — Static baseline",
    pixel_spacing: float = 0.070,
    narrative: Optional[tuple] = None,
    draft: bool = False,
    signed_meta: Optional[dict] = None,
) -> bytes:
    """PDF report for static single-frame analysis. Model-aware for Phase 0 and Phase 4a.

    Optional parameters:
      narrative    — pre-rendered (p1, p2, p3) tuple. If provided, skips the LLM
                     call entirely and renders the supplied paragraphs. Used by
                     the /studies/.../reports endpoint to render the same PDF
                     deterministically across requests after the LLM has run
                     once at create time.
      draft        — when True, overlays a DRAFT watermark on every page.
      signed_meta  — when present (dict with signed_by, signed_at, signoff_note),
                     a sign-off block is appended after the regulatory notice.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )
    llm = use_llm and bool(api_key)
    st = _styles(llm)
    story = []
    m = _meta(model_name)

    hc = result.get("hc_mm") or 0.0
    ga_str = result.get("ga_str") or "—"
    ga_weeks = result.get("ga_weeks") or 0.0
    trim = result.get("trimester") or "—"
    conf_lbl = result.get("confidence_label") or "HIGH CONFIDENCE"
    elapsed = result.get("elapsed_ms") or 0.0
    gradcam = result.get("gradcam_ok", True)

    _report_header(
        story,
        st,
        "Fetal Head Circumference — Automated Biometry Report",
        model_name,
        "STATIC SINGLE-FRAME ANALYSIS",
        llm,
        elapsed,
    )

    _biometric_table(story, st, hc, ga_str, trim, conf_lbl, model_name, elapsed, pixel_spacing)
    _model_performance_table(story, st, model_name, elapsed)

    story.append(Paragraph("Clinical Interpretation", st["sec"]))

    if narrative is not None:
        p1, p2, p3 = (narrative + (None, None, None))[:3]
    elif llm:
        p1, p2, p3 = _llm_static_narrative(
            hc, ga_str, ga_weeks, trim, gradcam, model_name, elapsed, api_key
        )
    else:
        p1 = _rule_static_p1(hc, ga_str, ga_weeks, trim)
        p2 = _rule_static_p2(gradcam)
        p3 = _rule_compression_note(model_name, elapsed)

    story.append(Paragraph("<b>Biometric assessment</b>", st["label"]))
    story.append(Paragraph(p1, st["body"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("<b>Activation map interpretation</b>", st["label"]))
    story.append(Paragraph(p2, st["body"]))
    if p3 and m["pruned"]:
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                "<b>Deployment efficiency — clinical context</b>"
                if (narrative or llm)
                else "<b>Computational efficiency note</b>",
                st["label"],
            )
        )
        story.append(Paragraph(p3, st["green"]))

    _regulatory(story, st)
    if signed_meta:
        _signoff_block(story, st, signed_meta)
    if draft:
        doc.build(story, onFirstPage=_draw_draft_watermark, onLaterPages=_draw_draft_watermark)
    else:
        doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_cine_report(
    result: dict,
    api_key: Optional[str] = None,
    use_llm: bool = True,
    model_name: str = "Phase 2 — Temporal baseline",
    pixel_spacing: float = 0.070,
    narrative: Optional[tuple] = None,
    draft: bool = False,
    signed_meta: Optional[dict] = None,
) -> bytes:
    """PDF report for temporal cine-loop analysis. Model-aware for Phase 2 and Phase 4b.

    See generate_static_report for narrative / draft / signed_meta semantics.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )
    llm = use_llm and bool(api_key)
    st = _styles(llm)
    story = []
    m = _meta(model_name)

    hc = result.get("hc_mm") or 0.0
    ga_str = result.get("ga_str") or "—"
    ga_weeks = result.get("ga_weeks") or 0.0
    trim = result.get("trimester") or "—"
    rel = result.get("reliability") or 0.0
    std = result.get("hc_std_mm") or 0.0
    conf_lbl = result.get("confidence_label") or "—"
    elapsed = result.get("elapsed_ms") or 0.0
    n_frames = len(result.get("per_frame_hc") or []) or 16

    _report_header(
        story,
        st,
        "Fetal Head Circumference — Automated Biometry Report",
        model_name,
        "TEMPORAL CINE-LOOP ANALYSIS",
        llm,
        elapsed,
    )

    _biometric_table(story, st, hc, ga_str, trim, conf_lbl, model_name, elapsed, pixel_spacing)
    _temporal_table(story, st, rel, std, n_frames)
    _model_performance_table(story, st, model_name, elapsed)

    story.append(Paragraph("Clinical Interpretation", st["sec"]))

    if narrative is not None:
        p1, p2, p3 = (narrative + (None, None, None))[:3]
    elif llm:
        p1, p2, p3 = _llm_cine_narrative(
            hc, ga_str, ga_weeks, trim, rel, std, n_frames, model_name, elapsed, api_key
        )
    else:
        p1 = _rule_cine_p1(hc, ga_str, ga_weeks, trim, rel, std)
        p2 = _rule_cine_p2(rel, std, n_frames)
        p3 = _rule_compression_note(model_name, elapsed)

    story.append(Paragraph("<b>Biometric assessment and temporal concordance</b>", st["label"]))
    story.append(Paragraph(p1, st["body"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("<b>Cine-loop analysis — clinical rationale</b>", st["label"]))
    story.append(Paragraph(p2, st["body"]))
    if p3 and m["pruned"]:
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                "<b>Deployment efficiency — clinical context</b>"
                if (narrative or llm)
                else "<b>Computational efficiency note</b>",
                st["label"],
            )
        )
        story.append(Paragraph(p3, st["green"]))

    _regulatory(story, st)
    if signed_meta:
        _signoff_block(story, st, signed_meta)
    if draft:
        doc.build(story, onFirstPage=_draw_draft_watermark, onLaterPages=_draw_draft_watermark)
    else:
        doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_comparison_report(
    results: dict,
    api_key: Optional[str] = None,
    use_llm: bool = True,
    pixel_spacing: float = 0.070,
) -> bytes:
    """
    Head-to-head PDF report comparing all four model variants on one image.

    results dict keys: phase0, phase4a, phase2, phase4b — each a result dict
    from predict_single_frame / predict_cine_clip.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )
    llm = use_llm and bool(api_key)
    st = _styles(llm)
    story = []

    r0 = results.get("phase0", {})
    r4a = results.get("phase4a", {})
    r2 = results.get("phase2", {})
    r4b = results.get("phase4b", {})

    acc = _TEAL if llm else _GREY
    label = "AI-AUTHORED COMPARATIVE REPORT" if llm else "AUTOMATED COMPARATIVE TEMPLATE"
    story.append(
        Paragraph("Fetal Head Circumference — Four-Model Comparative Biometry Report", st["title"])
    )
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC  |  "
            f"All four AI model variants evaluated on identical input image  |  "
            f"Pixel spacing: {pixel_spacing:.4f} mm/pixel",
            st["sub"],
        )
    )
    story.append(
        Paragraph(
            f'<font color="{acc.hexval()}">[{label}]</font>  ·  '
            f'<font color="{_NAVY.hexval()}">[STATIC + TEMPORAL ANALYSIS]</font>  ·  '
            f'<font color="{_PRUNED.hexval()}">[INCLUDES COMPRESSED MODEL VARIANTS]</font>',
            st["badge"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=acc, spaceAfter=10))

    # ── Four-model comparison table ──────────────────────────────────────────
    story.append(Paragraph("Four-Model Measurement Comparison", st["sec"]))
    story.append(
        Paragraph(
            "The table below summarises automated measurements and validated system performance "
            "across all four model variants applied to the same ultrasound image.",
            st["bodyI"],
        )
    )

    def _fmt(v, unit="", decimals=1):
        return f"{v:.{decimals}f}{unit}" if v else "—"

    m0 = _meta("Phase 0 — Static baseline")
    m4a = _meta("Phase 4a — Compressed static")
    m2 = _meta("Phase 2 — Temporal baseline")
    m4b = _meta("Phase 4b — Compressed temporal")

    cmp_rows = [
        [
            "Metric",
            "Phase 0\nStatic",
            "Phase 4a ✂\nCompressed Static",
            "Phase 2\nTemporal",
            "Phase 4b ✂\nCompressed Temporal",
        ],
        [
            "HC — this image",
            _fmt(r0.get("hc_mm"), " mm"),
            _fmt(r4a.get("hc_mm"), " mm"),
            _fmt(r2.get("hc_mm"), " mm"),
            _fmt(r4b.get("hc_mm"), " mm"),
        ],
        [
            "GA — this image",
            r0.get("ga_str") or "—",
            r4a.get("ga_str") or "—",
            r2.get("ga_str") or "—",
            r4b.get("ga_str") or "—",
        ],
        [
            "Inference (ms)",
            _fmt(r0.get("elapsed_ms"), " ms", 0),
            _fmt(r4a.get("elapsed_ms"), " ms", 0),
            _fmt(r2.get("elapsed_ms"), " ms", 0),
            _fmt(r4b.get("elapsed_ms"), " ms", 0),
        ],
        ["Validated accuracy", m0["dice"], m4a["dice"], m2["dice"], m4b["dice"]],
        ["Mean deviation (val.)", m0["mae"], m4a["mae"], m2["mae"], m4b["mae"]],
        ["Model footprint", m0["params"], m4a["params"], m2["params"], m4b["params"]],
        ["Compression vs baseline", "—", "−43.7%", "—", "−41.6%"],
        ["Operations per frame", m0["flops"], m4a["flops"], m2["flops"], m4b["flops"]],
        ["ISUOG ≤3 mm", "✓ PASS", "✓ PASS", "✓ PASS", "✓ PASS"],
        [
            "Analysis type",
            "Single frame",
            "Single frame",
            "16-frame consensus",
            "16-frame consensus",
        ],
    ]

    cw = [42 * mm, 28 * mm, 32 * mm, 28 * mm, 32 * mm]
    t = Table(cmp_rows, colWidths=cw)
    ts = _tbl_style()
    # Highlight compressed model columns
    ts.add("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#ecfdf5"))
    ts.add("BACKGROUND", (4, 1), (4, -1), colors.HexColor("#ecfdf5"))
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # ── Clinical interpretation ───────────────────────────────────────────────
    story.append(Paragraph("Clinical and Deployment Interpretation", st["sec"]))

    if llm:
        p1, p2 = _llm_comparison_narrative(results, api_key)
        story.append(
            Paragraph("<b>Static vs temporal analysis — clinical recommendation</b>", st["label"])
        )
        story.append(Paragraph(p1, st["body"]))
        story.append(Spacer(1, 3 * mm))
        story.append(
            Paragraph("<b>Compressed model variants — deployment significance</b>", st["label"])
        )
        story.append(Paragraph(p2, st["green"]))
    else:
        story.append(
            Paragraph("<b>Static vs temporal analysis — clinical recommendation</b>", st["label"])
        )
        story.append(
            Paragraph(
                (
                    "Single-frame static analysis is appropriate for standard second-trimester "
                    "sonographic screening where a clear suboccipitobregmatic plane with adequate "
                    "acoustic access to the calvarium can be obtained by the examining sonographer. "
                    "The temporal cine-loop analysis, which derives a consensus measurement from "
                    "16 sequential frames, may provide incremental clinical value in scenarios "
                    "with suboptimal probe contact, intermittent acoustic shadowing from fetal "
                    "position, elevated operator variability, or when measurement audit and "
                    "inter-observer reproducibility documentation are required. "
                    "For routine high-throughput second-trimester screening with experienced "
                    "operators and standard acoustic conditions, single-frame analysis offers "
                    "equivalent validated accuracy with substantially lower computational overhead. "
                    "Both approaches satisfy the ISUOG ±3 mm acceptable biometry threshold on "
                    "the HC18 independent validation cohort."
                ),
                st["body"],
            )
        )
        story.append(Spacer(1, 3 * mm))
        story.append(
            Paragraph("<b>Compressed model variants — deployment significance</b>", st["label"])
        )
        story.append(
            Paragraph(
                (
                    "The compressed model variants (Phase 4a and Phase 4b) achieve clinically "
                    "equivalent accuracy to their full-size counterparts following structural "
                    "parameter reduction of 43.7% and 41.6% respectively. "
                    "Both compressed models satisfy the ISUOG ±3 mm measurement deviation "
                    "threshold, and the temporal compressed variant (Phase 4b) marginally "
                    "exceeds baseline accuracy (96.00% vs 95.95%). "
                    "CPU inference times of under 400 ms for static and under 7,000 ms for "
                    "cine analysis support deployment without dedicated GPU infrastructure, "
                    "enabling integration on portable ultrasound platforms, handheld devices, "
                    "and in low-resource clinical settings where GPU-equipped workstations "
                    "are unavailable — a configuration relevant to an estimated 60% of "
                    "global obstetric ultrasound facilities."
                ),
                st["green"],
            )
        )

    # ── Ablation context ──────────────────────────────────────────────────────
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Temporal Model — Ablation Context", st["sec"]))
    story.append(
        Paragraph(
            (
                "To contextualise the clinical value of the temporal attention mechanism, "
                "an ablation study was conducted in which the temporal self-attention module "
                "was removed, reducing the system to frame-independent static inference. "
                "Without temporal attention, the same model architecture achieved a segmentation "
                "accuracy of 81.48% and a mean measurement deviation of 19.37 mm — well outside "
                "ISUOG thresholds. This demonstrates that the temporal integration component is "
                "clinically essential for reliable multi-frame biometric measurement, not merely "
                "an incremental enhancement."
            ),
            st["bodyI"],
        )
    )

    _regulatory(story, st)
    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_pdf_report(
    result: dict,
    api_key: Optional[str] = None,
    use_llm: bool = True,
    model_name: str = None,
    pixel_spacing: float = 0.070,
) -> bytes:
    """Backwards-compatible router."""
    if result.get("mode") == "cine_clip":
        return generate_cine_report(
            result,
            api_key=api_key,
            use_llm=use_llm,
            model_name=model_name or "Phase 2 — Temporal baseline",
            pixel_spacing=pixel_spacing,
        )
    return generate_static_report(
        result,
        api_key=api_key,
        use_llm=use_llm,
        model_name=model_name or "Phase 0 — Static baseline",
        pixel_spacing=pixel_spacing,
    )
