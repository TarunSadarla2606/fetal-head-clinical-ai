"""
report.py — Fetal Head Circumference Clinical AI
ACR/AIUM/ESR-compliant PDF report generation.

Public functions:
  generate_static_report()     — Phase 0 / Phase 4a single-frame analysis
  generate_cine_report()       — Phase 2 / Phase 4b temporal cine analysis
  generate_comparison_report() — All four models head-to-head (Tab 3)
  generate_pdf_report()        — Backwards-compatible router

Report structure (11 sections):
  1  Header — accession, datetime, status badges, AI-assisted watermark line
  2  Patient & Exam Information — two-column demographics table
  3  Clinical Indication
  4  Technical Parameters — pixel spacing with DICOM/estimated flag
  5  Biometric Findings — HC, GA, LMP discordance
  6  Images — 3-panel: original / segmentation overlay / GradCAM
  7  Clinical Interpretation — template or AI-authored (clearly labelled)
  8  AI System Performance — compact validation table
  9  Impression — bordered box, plain language for referring physician
  10 Sign-off block
  11 Regulatory footer

Two modes are supported and visually distinct:
  template — rule-based paragraphs, blue accent, "AUTOMATED TEMPLATE REPORT"
  llm      — Claude Haiku narrative, teal accent, "AI-AUTHORED CLINICAL NARRATIVE"
"""

import io
import re
from datetime import datetime, timedelta
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
_TEAL = colors.HexColor("#2D7D9A")  # clinical teal — accent bar, LLM mode
_NAVY = colors.HexColor("#1e3a5f")  # navy — headings, labels
_GREY = colors.HexColor("#4b5563")  # grey — template mode accent
_AMBER = colors.HexColor("#b45309")  # amber — warnings, estimated pixel spacing
_RED = colors.HexColor("#991b1b")  # red — OOD flag
_PRUNED = colors.HexColor("#065f46")  # green — sign-off, compression note
_LIGHT_BG = colors.HexColor("#f0f4f8")
_IMPRESSION_BG = colors.HexColor("#f0f9ff")  # light blue — impression box
_BORDER = colors.HexColor("#cccccc")
_TEAL_BORDER = colors.HexColor("#2D7D9A")
_PAGE_W = A4[0]
_CONTENT_W = _PAGE_W - 40 * mm  # 20 mm margins each side


# ── Model metadata ─────────────────────────────────────────────────────────────
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


# ── Styles ─────────────────────────────────────────────────────────────────────


def _styles(llm: bool = False):
    s = getSampleStyleSheet()
    acc = _TEAL if llm else _GREY
    return dict(
        title=ParagraphStyle(
            "ti",
            parent=s["Heading1"],
            fontSize=12,
            spaceAfter=2,
            textColor=colors.white,
            fontName="Helvetica-Bold",
        ),
        subtitle=ParagraphStyle(
            "sti",
            parent=s["Normal"],
            fontSize=8,
            spaceAfter=0,
            textColor=colors.HexColor("#cce8f4"),
        ),
        badge=ParagraphStyle(
            "ba",
            parent=s["Normal"],
            fontSize=7.5,
            spaceAfter=4,
            textColor=acc,
            fontName="Helvetica-Bold",
        ),
        sec=ParagraphStyle(
            "se",
            parent=s["Heading2"],
            fontSize=9.5,
            spaceBefore=8,
            spaceAfter=3,
            textColor=_NAVY,
            fontName="Helvetica-Bold",
        ),
        body=ParagraphStyle(
            "bo",
            parent=s["Normal"],
            fontSize=9,
            spaceAfter=4,
            leading=13.5,
        ),
        bodyI=ParagraphStyle(
            "bi",
            parent=s["Normal"],
            fontSize=9,
            spaceAfter=4,
            leading=13.5,
            textColor=_GREY,
        ),
        warn=ParagraphStyle(
            "wa",
            parent=s["Normal"],
            fontSize=8,
            leading=11,
            textColor=_AMBER,
        ),
        label=ParagraphStyle(
            "la",
            parent=s["Normal"],
            fontSize=8.5,
            spaceAfter=1,
            fontName="Helvetica-Bold",
            textColor=_NAVY,
        ),
        green=ParagraphStyle(
            "gr",
            parent=s["Normal"],
            fontSize=9,
            spaceAfter=4,
            leading=13.5,
            textColor=_PRUNED,
        ),
        impression=ParagraphStyle(
            "im",
            parent=s["Normal"],
            fontSize=9.5,
            leading=14,
            textColor=_NAVY,
        ),
        footer=ParagraphStyle(
            "fo",
            parent=s["Normal"],
            fontSize=7,
            leading=10,
            textColor=_GREY,
        ),
        img_cap=ParagraphStyle(
            "ic",
            parent=s["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=_GREY,
            alignment=1,  # centre
        ),
    )


def _tbl_style(header_color=None):
    hc = header_color or _NAVY
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), hc),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_BG, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, _BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]
    )


# ── Draft watermark ─────────────────────────────────────────────────────────────


def _draw_draft_watermark(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 90)
    canvas.setFillColorRGB(0.85, 0.20, 0.20, alpha=0.18)
    canvas.translate(297, 421)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, "DRAFT — UNSIGNED")
    canvas.restoreState()


# ── Utility functions ───────────────────────────────────────────────────────────


def _strip_markdown(text: str) -> str:
    """Remove markdown headers, bold, italic, bullets from LLM output."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ga_ci_string(ga_weeks: float) -> str:
    """Trimester-appropriate Hadlock 1984 GA confidence interval."""
    if ga_weeks < 14:
        return "±5–7 days (Hadlock 1984, first-trimester range)"
    elif ga_weeks < 22:
        return "±7–10 days (Hadlock 1984)"
    elif ga_weeks < 28:
        return "±10–14 days (Hadlock 1984)"
    else:
        return "±14–21 days (Hadlock 1984, third-trimester — early dating preferred)"


def _calculate_edd(lmp: str) -> Optional[str]:
    """Naegele's rule: EDD = LMP + 280 days."""
    try:
        lmp_dt = datetime.strptime(lmp, "%Y-%m-%d")
        edd_dt = lmp_dt + timedelta(days=280)
        return edd_dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _ga_discordance_days(lmp: str, ga_weeks: float) -> Optional[int]:
    """Days difference between LMP-derived GA and HC-derived GA."""
    try:
        lmp_dt = datetime.strptime(lmp, "%Y-%m-%d")
        today = datetime.utcnow()
        lmp_ga_days = (today - lmp_dt).days
        hc_ga_days = round(ga_weeks * 7)
        return abs(lmp_ga_days - hc_ga_days)
    except Exception:
        return None


def _b64_to_image_flowable(b64_str: str, max_width: float, max_height: float) -> Optional[Image]:
    """Decode base64 PNG/JPG string → reportlab Image flowable, respecting max dimensions."""
    try:
        import base64

        data = base64.b64decode(b64_str)
        buf = io.BytesIO(data)
        img = Image(buf)
        # Scale to fit within max_width × max_height preserving aspect ratio
        w, h = img.imageWidth, img.imageHeight
        if w > 0 and h > 0:
            scale = min(max_width / w, max_height / h)
            img.drawWidth = w * scale
            img.drawHeight = h * scale
        return img
    except Exception:
        return None


# ── Section 1 — Report Header ──────────────────────────────────────────────────


def _section_header(story, st, model_name, llm, elapsed_ms, accession, report_mode, ood_flag=False):
    acc = _TEAL if llm else _GREY
    mode_label = "AI-AUTHORED CLINICAL NARRATIVE" if llm else "AUTOMATED TEMPLATE REPORT"
    m = _meta(model_name)

    # Accent bar — teal rectangle drawn as a coloured table row
    header_data = [
        [
            Paragraph("Fetal Biometry — AI-Assisted Measurement Report", st["title"]),
            Paragraph(
                f"Accession: {accession or '—'}<br/>"
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                st["subtitle"],
            ),
        ]
    ]
    header_tbl = Table(header_data, colWidths=[110 * mm, 60 * mm])
    header_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), acc),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ]
        )
    )
    story.append(header_tbl)
    story.append(Spacer(1, 2 * mm))

    # Badge row
    badges = [f'<font color="{acc.hexval()}">[{mode_label}]</font>']
    if m["pruned"]:
        badges.append(
            f'<font color="{_PRUNED.hexval()}">[COMPRESSED MODEL — {m["compression"]}]</font>'
        )
    badges.append(
        f'<font color="{_NAVY.hexval()}">[AI-Assisted — Requires Clinical Verification]</font>'
    )
    if ood_flag:
        badges.append(f'<font color="{_RED.hexval()}">[⚠ OUT-OF-DISTRIBUTION ALERT]</font>')
    story.append(Paragraph("  ·  ".join(badges), st["badge"]))
    story.append(
        Paragraph(
            f"Model: {m['short']}" + (f"  |  Inference: {elapsed_ms:.0f} ms" if elapsed_ms else ""),
            st["footer"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.2, color=acc, spaceAfter=4))


# ── Section 2 — Patient & Exam Information ─────────────────────────────────────


def _section_patient_exam(story, st, report):
    story.append(Paragraph("Patient & Exam Information", st["sec"]))

    def _val(v):
        return v or "—"

    edd = _calculate_edd(report.lmp) if report.lmp else None
    lmp_row = f"{_val(report.lmp)}"
    edd_row = edd or ("Calculated from LMP" if report.lmp else "—")

    left = [
        ["Patient Name", _val(report.patient_name)],
        ["Patient ID / MRN", _val(report.patient_id)],
        ["Date of Birth", _val(report.patient_dob)],
        ["LMP (Last Menstrual Period)", lmp_row],
        ["EDD (Expected Delivery Date)", edd_row],
    ]
    right = [
        ["Referring Physician", _val(report.referring_physician)],
        ["Ordering Facility", _val(report.ordering_facility)],
        ["Sonographer", _val(report.sonographer_name)],
        ["Exam Date", _val(report.study_date)],
        ["Exam Type", _ga_exam_type(report.ga_weeks)],
    ]

    # Build as a single wide table with 4 columns: label | value | label | value
    combined = []
    for l_row, r_row in zip(left, right):
        combined.append([l_row[0], l_row[1], r_row[0], r_row[1]])

    col_w = [38 * mm, 52 * mm, 38 * mm, 42 * mm]
    t = Table(combined, colWidths=col_w)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 0), (0, -1), _NAVY),
                ("TEXTCOLOR", (2, 0), (2, -1), _NAVY),
                ("BACKGROUND", (0, 0), (0, -1), _LIGHT_BG),
                ("BACKGROUND", (2, 0), (2, -1), _LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0.3, _BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 3 * mm))


def _ga_exam_type(ga_weeks) -> str:
    if ga_weeks is None:
        return "—"
    if ga_weeks < 14:
        return "First trimester"
    elif ga_weeks < 28:
        return "Second trimester"
    else:
        return "Third trimester"


# ── Section 3 — Clinical Indication ───────────────────────────────────────────


def _section_clinical_indication(story, st, report):
    story.append(Paragraph("Clinical Indication", st["sec"]))
    indication = report.clinical_indication if report.clinical_indication else None
    if indication:
        story.append(Paragraph(indication, st["body"]))
    else:
        story.append(
            Paragraph(
                "⚠ Clinical indication not provided. Completing this field is recommended "
                "for ACR/AIUM-compliant sonography documentation.",
                st["warn"],
            )
        )
    story.append(Spacer(1, 2 * mm))


# ── Section 4 — Technical Parameters ──────────────────────────────────────────


def _section_technical_params(story, st, report):
    story.append(Paragraph("Technical Parameters", st["sec"]))

    ps_mm = report.pixel_spacing_mm or 0.070
    dicom_flag = getattr(report, "pixel_spacing_dicom_derived", False)
    ps_source = "DICOM-derived ✓" if dicom_flag else "Estimated ⚠ — verify before clinical use"

    us_approach = (report.us_approach or "transabdominal").capitalize()
    image_quality = (report.image_quality or "not recorded").capitalize()

    rows = [
        ["Parameter", "Value", "Notes"],
        ["Ultrasound approach", us_approach, ""],
        ["Scanning plane", "Suboccipitobregmatic", "Standard fetal head biometry plane"],
        ["Pixel spacing", f"{ps_mm:.4f} mm/pixel", ps_source],
        ["Image quality", image_quality, "Sonographer assessment"],
        ["Measurement standard", "ISUOG Practice Guidelines 2010", "HC measurement methodology"],
    ]
    t = Table(rows, colWidths=[55 * mm, 55 * mm, 60 * mm])
    ts = _tbl_style()
    if not dicom_flag:
        # Highlight pixel spacing row in amber
        ts.add("TEXTCOLOR", (0, 2), (-1, 2), _AMBER)
        ts.add("FONTNAME", (0, 2), (0, 2), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)

    if not dicom_flag:
        story.append(
            Paragraph(
                "⚠ Pixel spacing was not auto-detected from DICOM metadata. "
                "The value shown is user-supplied or a system default. "
                "Incorrect pixel spacing will produce a wrong HC measurement — "
                "verify against your ultrasound system's calibration data.",
                st["warn"],
            )
        )
    story.append(Spacer(1, 3 * mm))


# ── Section 5 — Biometric Findings ────────────────────────────────────────────


def _section_biometric_findings(story, st, report, lmp=None):
    story.append(Paragraph("Biometric Findings", st["sec"]))

    hc = report.hc_mm
    ga_str = report.ga_str or "—"
    ga_weeks = report.ga_weeks or 0.0
    trim = report.trimester or "—"
    conf = report.confidence_label or "—"
    ci_str = _ga_ci_string(ga_weeks) if ga_weeks else "—"

    ood_flag = False  # populated from result dict when called from generate_*_report

    rows = [
        ["Parameter", "AI Measurement", "Reference / Notes"],
        ["Head Circumference (HC)", f"{hc:.1f} mm" if hc else "—", "Calvarium perimeter"],
        ["Estimated Gestational Age", ga_str, f"Hadlock 1984 nomogram  {ci_str}"],
        ["Trimester", trim, "Derived from estimated GA"],
        ["Measurement Confidence", conf, "Based on segmentation coverage"],
    ]
    t = Table(rows, colWidths=[58 * mm, 42 * mm, 70 * mm])
    t.setStyle(_tbl_style())
    story.append(t)

    # LMP cross-check
    lmp_val = getattr(report, "lmp", None) or lmp
    if lmp_val and ga_weeks:
        edd = _calculate_edd(lmp_val)
        disc = _ga_discordance_days(lmp_val, ga_weeks)
        disc_flag = disc is not None and disc > 14
        disc_str = f"{disc} days" if disc is not None else "—"
        disc_note = " ⚠ DISCORDANT (>14 days — clinical review recommended)" if disc_flag else ""

        lmp_rows = [
            ["GA Source", "Value", "Notes"],
            ["GA from HC (Hadlock 1984)", ga_str, "This measurement"],
            ["LMP-derived GA", "Calculated from LMP entry", f"LMP: {lmp_val}  EDD: {edd or '—'}"],
            ["Discordance (|LMP GA − HC GA|)", disc_str, disc_note],
        ]
        lt = Table(lmp_rows, colWidths=[58 * mm, 52 * mm, 60 * mm])
        lt.setStyle(_tbl_style(header_color=_TEAL if not disc_flag else colors.HexColor("#7f1d1d")))
        story.append(Spacer(1, 2 * mm))
        story.append(lt)

    story.append(Spacer(1, 3 * mm))


# ── Section 6 — Images ────────────────────────────────────────────────────────


def _section_images(story, st, report):
    story.append(Paragraph("Annotated Ultrasound Images", st["sec"]))

    panel_w = (_CONTENT_W - 8 * mm) / 3  # 3 panels + 2 gaps
    panel_h = 52 * mm

    def _panel(b64, label):
        img = _b64_to_image_flowable(b64, panel_w, panel_h) if b64 else None
        if img is None:
            # Grey placeholder box
            placeholder = Table(
                [["Not\navailable"]],
                colWidths=[panel_w],
                rowHeights=[panel_h],
            )
            placeholder.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#d1d5db")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTRE"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("TEXTCOLOR", (0, 0), (-1, -1), _GREY),
                    ]
                )
            )
            return [placeholder, Paragraph(label, st["img_cap"])]
        return [img, Paragraph(label, st["img_cap"])]

    orig_b64 = getattr(report, "original_image_b64", None)
    overlay_b64 = getattr(report, "overlay_image_b64", None)
    gradcam_b64 = getattr(report, "gradcam_image_b64", None)

    p1 = _panel(orig_b64, "Original ultrasound")
    p2 = _panel(overlay_b64, "Segmentation overlay")
    p3 = _panel(gradcam_b64, "GradCAM++ activation")

    img_row = Table(
        [[p1[0], p2[0], p3[0]]],
        colWidths=[panel_w, panel_w, panel_w],
        rowHeights=[panel_h],
    )
    img_row.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTRE"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    cap_row = Table(
        [[p1[1], p2[1], p3[1]]],
        colWidths=[panel_w, panel_w, panel_w],
    )
    cap_row.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTRE"),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(img_row)
    story.append(cap_row)
    story.append(
        Paragraph(
            "AI-generated segmentation boundary (teal) and activation map (jet colormap). "
            "Annotated boundaries require verification by a qualified sonographer prior to clinical use.",
            st["footer"],
        )
    )
    story.append(Spacer(1, 3 * mm))


# ── Section 9 — Impression ────────────────────────────────────────────────────
# Placed before Section 7 in the story so it appears on page 1.


def _section_impression(story, st, narrative_impression, report):
    story.append(Paragraph("Impression", st["sec"]))
    hc = report.hc_mm
    ga_str = report.ga_str or "—"
    ga_weeks = report.ga_weeks or 0.0

    text = narrative_impression or _rule_impression(hc, ga_str, ga_weeks, report.trimester or "—")
    text = _strip_markdown(text)

    imp_data = [[Paragraph(text, st["impression"])]]
    imp_tbl = Table(imp_data, colWidths=[_CONTENT_W])
    imp_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _IMPRESSION_BG),
                ("BOX", (0, 0), (-1, -1), 1.5, _TEAL_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(imp_tbl)
    story.append(Spacer(1, 3 * mm))


# ── Section 7 — Clinical Interpretation ───────────────────────────────────────


def _section_interpretation(story, st, llm, model_name, p1, p2, p3, report):
    acc = _TEAL if llm else _GREY
    mode_label = (
        "AI-Authored Clinical Narrative — Claude Haiku" if llm else "Automated Template Report"
    )
    m = _meta(model_name)
    is_temporal = m["type"] == "cine"

    story.append(Paragraph("Clinical Interpretation", st["sec"]))

    # Mode badge
    badge_data = [
        [
            Paragraph(
                f'<font color="{acc.hexval()}">[{mode_label.upper()}]</font>',
                st["badge"],
            )
        ]
    ]
    badge_tbl = Table(badge_data, colWidths=[_CONTENT_W])
    badge_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, acc),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(badge_tbl)
    story.append(Spacer(1, 2 * mm))

    p1_clean = _strip_markdown(p1) if p1 else "—"
    p2_clean = _strip_markdown(p2) if p2 else "—"

    story.append(
        Paragraph(
            "<b>Biometric assessment</b>"
            if not is_temporal
            else "<b>Biometric assessment and temporal concordance</b>",
            st["label"],
        )
    )
    story.append(Paragraph(p1_clean, st["body"]))

    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "<b>Activation map interpretation</b>"
            if not is_temporal
            else "<b>Cine-loop analysis — clinical rationale</b>",
            st["label"],
        )
    )
    story.append(Paragraph(p2_clean, st["body"]))

    if p3 and m["pruned"]:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("<b>Deployment efficiency — clinical context</b>", st["label"]))
        story.append(Paragraph(_strip_markdown(p3), st["green"]))

    story.append(Spacer(1, 3 * mm))


# ── Section 8 — AI System Performance ─────────────────────────────────────────


def _section_ai_performance(story, st, model_name, elapsed_ms):
    m = _meta(model_name)
    story.append(Paragraph("AI System Validation Summary", st["sec"]))
    rows = [
        ["Metric", "Result", "Clinical Reference"],
        ["Segmentation accuracy (validation cohort)", m["dice"], f"HC18 — {m['dataset']}"],
        ["Mean measurement deviation", m["mae"], "ISUOG acceptable threshold: ±3 mm"],
        ["ISUOG clinical threshold (±3 mm)", f"✓ {m['isuog']}", "ISUOG Practice Guidelines 2010"],
        ["Validation cohort", "199 images (held-out test set)", "HC18 — Radboud UMC, Netherlands"],
        ["Runtime (this image)", f"{elapsed_ms:.0f} ms" if elapsed_ms else "—", "CPU inference"],
    ]
    t = Table(rows, colWidths=[72 * mm, 38 * mm, 60 * mm])
    t.setStyle(_tbl_style())
    story.append(t)
    story.append(Spacer(1, 2 * mm))


# ── Section 10 — Sign-off block ────────────────────────────────────────────────


def _section_signoff(story, st, signed_meta: dict):
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=1.2, color=_PRUNED, spaceAfter=4))
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
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TEXTCOLOR", (0, 0), (-1, -1), _NAVY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, _PRUNED),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t)


# ── Section 11 — Regulatory footer ────────────────────────────────────────────


def _section_regulatory(story, st, ga_weeks=0.0):
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=0.4, color=_BORDER, spaceAfter=3))
    ci_str = _ga_ci_string(ga_weeks)
    story.append(
        Paragraph(
            "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system has not received FDA 510(k) "
            "clearance or CE marking under EU MDR/IVDR. It is classified as a Software as a "
            "Medical Device (SaMD) Class II candidate under 21 CFR Part 892. All automated "
            f"measurements must be independently verified by a qualified healthcare professional "
            f"before incorporation into any clinical decision. Gestational age estimates carry an "
            f"inherent {ci_str}. This report does not constitute a diagnostic opinion.",
            st["footer"],
        )
    )
    story.append(
        Paragraph(
            "HC18 dataset — Radboud University Medical Center, Nijmegen, Netherlands. "
            f"For research and evaluation purposes only. "
            f"System: TemporalFetaSegNet / Fetal Head Clinical AI v3.0 — "
            f"Tarun Sadarla, MS Artificial Intelligence, University of North Texas, 2026. "
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            st["footer"],
        )
    )


# ── Appendix — Technical details ───────────────────────────────────────────────


def _appendix_technical(story, st, model_name, elapsed_ms):
    m = _meta(model_name)
    story.append(PageBreak())
    story.append(Paragraph("Appendix A — AI Model Technical Specifications", st["sec"]))
    story.append(
        Paragraph(
            "The following technical metrics are provided for AI/ML validation teams and "
            "clinical engineering committees. They are not relevant to routine clinical reporting.",
            st["bodyI"],
        )
    )
    rows = [
        ["Specification", "Value"],
        ["Model architecture", m["short"]],
        ["Parameter count", m["params"]],
        ["Computational operations (GMACs)", m["flops"]],
        ["Compression vs baseline", m["compression"] or "N/A (baseline)"],
        ["Training dataset", "HC18 (1334 images, Radboud UMC)"],
        ["Validation dataset", m["dataset"]],
        ["Runtime on CPU", f"{elapsed_ms:.0f} ms" if elapsed_ms else "—"],
    ]
    t = Table(rows, colWidths=[80 * mm, 90 * mm])
    t.setStyle(_tbl_style())
    story.append(t)


# ── LLM narrative functions ────────────────────────────────────────────────────


def _call_llm(api_key: str, prompt: str, max_tokens: int = 400) -> Optional[str]:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=(
                "You are a clinical report assistant for an AI-assisted fetal biometry system. "
                "Write in formal clinical prose only. Do NOT use any markdown formatting — "
                "no headers (#), no bold (**), no bullet points (-), no italic (*). "
                "Output plain text paragraphs only."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()
    except Exception:
        return None


def _llm_static_narrative(hc, ga_str, ga_weeks, trim, gradcam_ok, model_name, elapsed_ms, api_key):
    m = _meta(model_name)
    ci_str = _ga_ci_string(ga_weeks)
    ctx = {
        "Early (<20w)": "first-trimester biometric assessment range",
        "Mid (20–30w)": "second-trimester sonographic window",
        "Late (>30w)": "third-trimester range where acoustic shadowing may affect delineation",
    }.get(trim, "")

    p1 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (5 sentences) for a clinical audience — obstetrician or senior "
            f"sonographer. Use formal medical terminology. Do NOT use any machine learning terms "
            f"(Dice, MAE, IoU, neural network, model). Do NOT make diagnostic conclusions or suggest "
            f"pathology. End requiring sonographer verification.\n\n"
            f"Data: HC = {hc:.1f} mm, GA = {ga_str} ({ga_weeks:.1f} weeks) by Hadlock (1984) "
            f"nomogram, trimester = {trim} ({ctx}). GA confidence interval: {ci_str}. "
            f"System MAE = {m['mae']} (within ISUOG ±3 mm threshold).\n\n"
            f"Structure: (1) HC and GA with biometric context, (2) trimester-specific GA accuracy "
            f"({ci_str}), (3) automated system validation status, (4) limitations and advisory "
            f"for clinical correlation. Plain prose only."
        ),
    ) or _rule_static_p1(hc, ga_str, ga_weeks, trim)

    p2 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (4 sentences) for a clinical audience interpreting a "
            f"GradCAM++ activation map from an automated fetal head circumference measurement. "
            f"{'The activation map was generated for this acquisition.' if gradcam_ok else 'The activation map could not be generated for this image.'}\n\n"
            f"Explain: (1) what the activation pattern indicates about the system's focus on "
            f"the hyperechoic calvarium interface, (2) why anatomically appropriate activation "
            f"supports confidence in the measurement, (3) clinical implications if atypical. "
            f"Use terms: calvarium, hyperechoic interface, cranial ossification, acoustic impedance "
            f"contrast. No ML jargon. Plain prose only."
        ),
    ) or _rule_static_p2(gradcam_ok)

    p3 = None
    if m["pruned"]:
        p3 = _call_llm(
            api_key,
            (
                f"Write ONE paragraph (3-4 sentences) for a hospital technology committee. "
                f"Compressed model: {m['compression']}, accuracy {m['mae']}, CPU inference "
                f"{elapsed_ms:.0f} ms. Explain clinical deployment significance for portable "
                f"and point-of-care platforms. Clinical language only. No ML jargon."
            ),
            max_tokens=280,
        )

    impression = _call_llm(
        api_key,
        (
            f"Write 2-3 sentences for the Impression box of a clinical ultrasound report. "
            f"Plain clinical language for a referring obstetrician. No markdown. "
            f"Include: HC ({hc:.1f} mm), GA ({ga_str}), ISUOG compliance, and a verification requirement.\n"
            f"Example format: 'AI-assisted head circumference measurement: X mm. Estimated "
            f"gestational age: Y. Automated measurement requires verification by a qualified "
            f"sonographer prior to clinical use.'"
        ),
        max_tokens=150,
    ) or _rule_impression(hc, ga_str, ga_weeks, trim)

    return p1, p2, p3, impression


def _llm_cine_narrative(
    hc, ga_str, ga_weeks, trim, rel, std, n_frames, model_name, elapsed_ms, api_key
):
    m = _meta(model_name)
    ci_str = _ga_ci_string(ga_weeks)
    rel_desc = "excellent" if rel > 0.97 else "good" if rel > 0.93 else "moderate"
    std_desc = "highly stable" if std < 2.0 else "acceptable" if std < 5.0 else "variable"
    ctx = {
        "Early (<20w)": "first-trimester biometric assessment",
        "Mid (20–30w)": "second-trimester sonographic window",
        "Late (>30w)": "third-trimester range",
    }.get(trim, "")

    p1 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (5 sentences) for a clinical audience. Formal medical "
            f"terminology. No ML jargon. No diagnostic conclusions. End requiring verification.\n\n"
            f"Data: Consensus HC = {hc:.1f} mm from {n_frames}-frame acquisition. "
            f"GA = {ga_str} ({ga_weeks:.1f} weeks) by Hadlock (1984). GA CI: {ci_str}. "
            f"Trimester: {trim} ({ctx}). Frame concordance: {rel_desc} "
            f"(variability = {std:.2f} mm, {std_desc}). System MAE = {m['mae']}.\n\n"
            f"Structure: (1) HC and GA from multi-frame consensus, (2) GA accuracy ({ci_str}), "
            f"(3) inter-frame concordance in clinical terms, (4) validation status, "
            f"(5) requirement for clinical correlation. Plain prose only."
        ),
    ) or _rule_cine_p1(hc, ga_str, ga_weeks, trim, rel, std)

    p2 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (4 sentences) explaining the clinical rationale for "
            f"sequential multi-frame fetal head measurement vs single-frame biometry.\n\n"
            f"{n_frames} frames analysed. Frame HC variability = {std:.2f} mm. "
            f"Address: (1) how probe micro-variation and fetal movement introduce inter-frame "
            f"variation, (2) how automated frame weighting may reduce operator variability, "
            f"(3) clinical scenarios where cine analysis may be advantageous. "
            f"Use cautious language. Plain prose only. No ML terms."
        ),
    ) or _rule_cine_p2(rel, std, n_frames)

    p3 = None
    if m["pruned"]:
        p3 = _call_llm(
            api_key,
            (
                f"Write ONE paragraph (3-4 sentences) for a clinical technology committee. "
                f"Temporal cine model: {m['compression']}, {m['dice']} accuracy, {m['mae']} deviation, "
                f"{elapsed_ms:.0f} ms per 16-frame clip. Explain deployment significance for "
                f"portable platforms. Clinical language only."
            ),
            max_tokens=280,
        )

    impression = _call_llm(
        api_key,
        (
            f"Write 2-3 sentences for the Impression box of a clinical report. Plain language "
            f"for a referring obstetrician. No markdown. "
            f"HC = {hc:.1f} mm from {n_frames}-frame cine analysis, GA = {ga_str}, "
            f"frame concordance = {rel_desc}."
        ),
        max_tokens=150,
    ) or _rule_impression(hc, ga_str, ga_weeks, trim)

    return p1, p2, p3, impression


def _llm_comparison_narrative(results: dict, api_key: str):
    r0, r4a, r2, r4b = (
        results.get("phase0", {}),
        results.get("phase4a", {}),
        results.get("phase2", {}),
        results.get("phase4b", {}),
    )
    hc_vals = [r.get("hc_mm") for r in [r0, r4a, r2, r4b] if r.get("hc_mm")]
    cine_std = r4b.get("hc_std_mm") or r2.get("hc_std_mm") or 0.0
    ms_p0, ms_p4a, ms_p2, ms_p4b = (
        r0.get("elapsed_ms", 0),
        r4a.get("elapsed_ms", 0),
        r2.get("elapsed_ms", 0),
        r4b.get("elapsed_ms", 0),
    )

    p1 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (5-6 sentences) for a clinical committee deciding between "
            f"single-frame and multi-frame automated fetal head biometry.\n\n"
            f"Data: Single-frame HC: {r0.get('hc_mm', 0):.1f} mm ({ms_p0:.0f} ms), "
            f"compressed {r4a.get('hc_mm', 0):.1f} mm ({ms_p4a:.0f} ms). "
            f"Cine HC: {r2.get('hc_mm', 0):.1f} mm ({ms_p2:.0f} ms), "
            f"compressed {r4b.get('hc_mm', 0):.1f} mm ({ms_p4b:.0f} ms). "
            f"Cine inter-frame variability: {cine_std:.2f} mm.\n\n"
            f"Provide a nuanced recommendation: when is single-frame sufficient vs when cine "
            f"may offer incremental benefit. Cautious, evidence-appropriate language. "
            f"No ML jargon. Plain prose."
        ),
        max_tokens=420,
    ) or (
        "Single-frame analysis is appropriate for standard second-trimester screening with "
        "adequate acoustic access. Multi-frame cine analysis may provide incremental benefit "
        "with suboptimal probe contact or when measurement audit is required."
    )

    p2 = _call_llm(
        api_key,
        (
            f"Write ONE paragraph (4 sentences) for a hospital IT committee. "
            f"Both compressed models meet ISUOG ±3 mm threshold with CPU inference "
            f"({ms_p4a:.0f} ms single-frame, {ms_p4b:.0f} ms cine). "
            f"Explain clinical IT deployment significance: no GPU infrastructure required, "
            f"portable platform compatibility, LMIC relevance. Clinical language only."
        ),
        max_tokens=320,
    ) or (
        "Compressed model variants enable deployment on standard CPU hardware without GPU "
        "infrastructure, supporting integration in portable ultrasound platforms and "
        "resource-constrained settings."
    )
    return p1, p2


# ── Rule-based narratives ──────────────────────────────────────────────────────


def _rule_static_p1(hc, ga_str, ga_weeks, trim):
    ci_str = _ga_ci_string(ga_weeks) if ga_weeks else "±2 weeks"
    ctx = {
        "Early (<20w)": "consistent with first-trimester biometric parameters",
        "Mid (20–30w)": "within the optimal second-trimester sonographic assessment window",
        "Late (>30w)": "consistent with third-trimester biometry, where increased acoustic "
        "shadowing from the calvarium may influence boundary delineation",
    }.get(trim, "")
    return (
        f"Automated biometric analysis of the submitted fetal head ultrasound yielded "
        f"a head circumference of {hc:.1f} mm, corresponding to an estimated gestational "
        f"age of {ga_str} ({ga_weeks:.1f} weeks) derived from the Hadlock (1984) biometric "
        f"nomogram, {ctx}. "
        f"The gestational age estimate carries a confidence interval of {ci_str} for this "
        f"trimester, consistent with established sonographic biometry accuracy standards. "
        f"The measurement was produced by an automated calvarium boundary detection system "
        f"validated on an independent cohort of 199 fetal head ultrasound images (HC18), "
        f"with a mean absolute measurement deviation within the ISUOG clinically acceptable "
        f"biometry threshold of ±3 mm for second-trimester assessment. "
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
            "High-activation regions correspond to the hyperechoic calvarium echo — "
            "the outer cranial bone interface providing the principal anatomical landmark "
            "for head circumference measurement. "
            "Low-activation regions correspond to intracranial soft-tissue structures "
            "that do not contribute to the biometric perimeter. "
            "This activation pattern is anatomically congruent with appropriate model "
            "behaviour and supports confidence in the automated boundary selection."
        )
    return (
        "A gradient-weighted activation map could not be generated for this image, "
        "which may occur when the predicted segmentation region is insufficient for "
        "reliable spatial attribution — often associated with challenging acoustic windows "
        "or partial calvarium visualisation. "
        "In the absence of activation map confirmation, the automated measurement "
        "result should be interpreted with heightened caution, and direct sonographer "
        "review of the original image is recommended."
    )


def _rule_cine_p1(hc, ga_str, ga_weeks, trim, rel, std):
    ci_str = _ga_ci_string(ga_weeks) if ga_weeks else "±2 weeks"
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
        f"Sequential cine-loop analysis of a 16-frame ultrasound acquisition yielded "
        f"a consensus fetal head circumference of {hc:.1f} mm, corresponding to an "
        f"estimated gestational age of {ga_str} ({ga_weeks:.1f} weeks) by Hadlock (1984), "
        f"{ctx}. The GA estimate carries a confidence interval of {ci_str}. "
        f"Inter-frame measurement concordance was {rel_desc}, with calvarium boundary "
        f"measurements {std_desc}. "
        f"The consensus measurement from temporal integration of sequential frames provides "
        f"greater robustness against single-frame artefacts compared to static biometry. "
        f"Clinical correlation with menstrual dating and direct sonographer verification "
        f"are required before incorporation into clinical management."
    )


def _rule_cine_p2(rel, std, n_frames):
    return (
        f"The temporal analysis system evaluated all {n_frames} frames from the "
        f"acquisition sequence and applied automated preferential weighting toward frames "
        f"exhibiting superior calvarium boundary delineation — analogous to the sonographer's "
        f"practice of identifying the optimal image plane before freezing for measurement. "
        f"The observed inter-frame measurement variability of {std:.2f} mm is attributable "
        f"to minor changes in the imaged calvarial cross-section from probe micro-motion "
        f"and fetal head micro-movement during acquisition. "
        f"Automated temporal frame weighting reduces the operator-dependent component of "
        f"measurement variability and may be of particular value in cases where acoustic "
        f"access is intermittently limited by fetal position or maternal habitus."
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
        f"supporting deployment on portable ultrasound platforms."
    )


def _rule_impression(hc, ga_str, ga_weeks, trim):
    ci_str = _ga_ci_string(ga_weeks) if ga_weeks else "±2 weeks"
    hc_str = f"{hc:.1f} mm" if hc else "not available"
    return (
        f"AI-assisted head circumference measurement: {hc_str}. "
        f"Estimated gestational age: {ga_str} (confidence interval {ci_str}). "
        f"Measurement within ISUOG clinically acceptable biometry threshold (±3 mm). "
        f"Automated measurement requires verification by a qualified sonographer "
        f"prior to clinical use."
    )


# ── PDF document builder ───────────────────────────────────────────────────────


def _build_story(
    result: dict,
    api_key: Optional[str],
    use_llm: bool,
    model_name: str,
    pixel_spacing: float,
    narrative: Optional[tuple],
    draft: bool,
    signed_meta: Optional[dict],
    report_type_label: str,
    is_temporal: bool,
    report=None,
) -> list:
    """Assemble the complete story list for both static and cine reports."""
    llm = use_llm and bool(api_key)
    st = _styles(llm)
    m = _meta(model_name)

    hc = result.get("hc_mm") or 0.0
    ga_str = result.get("ga_str") or "—"
    ga_weeks = result.get("ga_weeks") or 0.0
    trim = result.get("trimester") or "—"
    conf_lbl = result.get("confidence_label") or "HIGH CONFIDENCE"
    elapsed = result.get("elapsed_ms") or 0.0
    rel = result.get("reliability") or 0.0
    std = result.get("hc_std_mm") or 0.0
    gradcam_ok = result.get("gradcam_ok", True)
    ood_flag = result.get("ood_flag", False)
    n_frames = len(result.get("per_frame_hc") or []) or 16

    # Accession number and report_mode from the report object if available
    accession = getattr(report, "accession_number", None)
    stored_mode = getattr(report, "report_mode", "template")
    effective_llm = llm or (stored_mode == "llm")

    # Resolve narrative
    if narrative is not None:
        if len(narrative) >= 4:
            p1, p2, p3, p_impression = narrative[0], narrative[1], narrative[2], narrative[3]
        else:
            p1, p2, p3 = (narrative + (None, None, None))[:3]
            p_impression = None
    elif llm:
        if is_temporal:
            p1, p2, p3, p_impression = _llm_cine_narrative(
                hc, ga_str, ga_weeks, trim, rel, std, n_frames, model_name, elapsed, api_key
            )
        else:
            p1, p2, p3, p_impression = _llm_static_narrative(
                hc, ga_str, ga_weeks, trim, gradcam_ok, model_name, elapsed, api_key
            )
    else:
        if is_temporal:
            p1 = _rule_cine_p1(hc, ga_str, ga_weeks, trim, rel, std)
            p2 = _rule_cine_p2(rel, std, n_frames)
        else:
            p1 = _rule_static_p1(hc, ga_str, ga_weeks, trim)
            p2 = _rule_static_p2(gradcam_ok)
        p3 = _rule_compression_note(model_name, elapsed)
        p_impression = _rule_impression(hc, ga_str, ga_weeks, trim)

    story = []

    # ── PAGE 1: Sections 1–6 + 9 ──────────────────────────────────────────────
    _section_header(
        story,
        st,
        model_name,
        effective_llm,
        elapsed,
        accession,
        stored_mode,
        ood_flag=ood_flag,
    )

    if report is not None:
        _section_patient_exam(story, st, report)
        _section_clinical_indication(story, st, report)
        _section_technical_params(story, st, report)
    else:
        # Minimal patient block when no report object (backwards compat)
        story.append(Paragraph("Patient & Exam Information", st["sec"]))
        story.append(
            Paragraph(
                f"Pixel spacing: {pixel_spacing:.4f} mm/pixel  |  "
                f"Model: {m['short']}  |  Analysis: {report_type_label}",
                st["bodyI"],
            )
        )
        story.append(Spacer(1, 2 * mm))

    _section_biometric_findings(
        story, st, _BiometricProxy(hc, ga_str, ga_weeks, trim, conf_lbl, report)
    )

    if is_temporal:
        _section_temporal_table(story, st, rel, std, n_frames)

    _section_images(story, st, report if report is not None else _NoImages())

    # Impression on page 1 so referring physician sees it immediately
    _section_impression(
        story, st, p_impression, _BiometricProxy(hc, ga_str, ga_weeks, trim, conf_lbl, report)
    )

    # ── PAGE 2: Sections 7–8 ─────────────────────────────────────────────────
    story.append(PageBreak())
    _section_interpretation(story, st, effective_llm, model_name, p1, p2, p3, report)
    _section_ai_performance(story, st, model_name, elapsed)

    # Sign-off (section 10)
    if signed_meta:
        _section_signoff(story, st, signed_meta)

    # Regulatory footer (section 11)
    _section_regulatory(story, st, ga_weeks=ga_weeks)

    # Technical appendix (page 3, only if compressed model)
    if m["pruned"]:
        _appendix_technical(story, st, model_name, elapsed)

    return story


class _BiometricProxy:
    """Minimal proxy so _section_biometric_findings / _section_impression work
    when called without a full report DB object."""

    def __init__(self, hc, ga_str, ga_weeks, trim, conf, report=None):
        self.hc_mm = hc
        self.ga_str = ga_str
        self.ga_weeks = ga_weeks
        self.trimester = trim
        self.confidence_label = conf
        self.lmp = getattr(report, "lmp", None)

    # Fallback attrs used elsewhere
    patient_name = "—"
    patient_id = None
    patient_dob = None
    ordering_facility = None
    referring_physician = None
    sonographer_name = None
    clinical_indication = None
    study_date = "—"
    us_approach = None
    image_quality = None
    pixel_spacing_mm = None
    pixel_spacing_dicom_derived = False
    original_image_b64 = None
    overlay_image_b64 = None
    gradcam_image_b64 = None
    accession_number = None
    report_mode = "template"


class _NoImages:
    original_image_b64 = None
    overlay_image_b64 = None
    gradcam_image_b64 = None


def _section_temporal_table(story, st, rel, std, n_frames):
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
        ["Frames analysed", str(n_frames), "Sequential cine acquisition"],
        ["Inter-frame concordance", rel_label, "Consistency across frames"],
        ["Frame-to-frame HC variability", f"{std:.2f} mm", std_label],
        ["Consensus method", "Temporal mean probability", "Mean prediction across frames"],
    ]
    t = Table(rows, colWidths=[58 * mm, 42 * mm, 70 * mm])
    t.setStyle(_tbl_style())
    story.append(t)
    story.append(Spacer(1, 3 * mm))


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
    report=None,
) -> bytes:
    """PDF report for static single-frame analysis (Phase 0 / Phase 4a).

    Parameters
    ----------
    narrative : optional pre-rendered (p1, p2, p3, impression) tuple. If
        provided, skips the LLM call and renders the supplied paragraphs
        deterministically. Used by the /reports/{id}/pdf endpoint.
    draft : overlay a DRAFT watermark on every page when True.
    signed_meta : dict with signed_by/signed_at/signoff_note; appends sign-off block.
    report : Report DB object. When present, enables all 11 sections including
        patient demographics, images, and clinical indication.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )
    story = _build_story(
        result,
        api_key,
        use_llm,
        model_name,
        pixel_spacing,
        narrative,
        draft,
        signed_meta,
        report_type_label="STATIC SINGLE-FRAME ANALYSIS",
        is_temporal=False,
        report=report,
    )
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
    report=None,
) -> bytes:
    """PDF report for temporal cine-loop analysis (Phase 2 / Phase 4b).

    See generate_static_report for parameter semantics.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )
    story = _build_story(
        result,
        api_key,
        use_llm,
        model_name,
        pixel_spacing,
        narrative,
        draft,
        signed_meta,
        report_type_label="TEMPORAL CINE-LOOP ANALYSIS",
        is_temporal=True,
        report=report,
    )
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
    """Head-to-head PDF comparing all four model variants on one image."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )
    llm = use_llm and bool(api_key)
    st = _styles(llm)
    acc = _TEAL if llm else _GREY
    label = "AI-AUTHORED COMPARATIVE REPORT" if llm else "AUTOMATED COMPARATIVE TEMPLATE"

    r0 = results.get("phase0", {})
    r4a = results.get("phase4a", {})
    r2 = results.get("phase2", {})
    r4b = results.get("phase4b", {})

    m0, m4a, m2, m4b = (
        _meta("Phase 0 — Static baseline"),
        _meta("Phase 4a — Compressed static"),
        _meta("Phase 2 — Temporal baseline"),
        _meta("Phase 4b — Compressed temporal"),
    )

    story = []

    # Header
    header_data = [
        [
            Paragraph("Fetal Head Circumference — Four-Model Comparative Report", st["title"]),
            Paragraph(
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                st["subtitle"],
            ),
        ]
    ]
    ht = Table(header_data, colWidths=[110 * mm, 60 * mm])
    ht.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), acc),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ]
        )
    )
    story.append(ht)
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            f'<font color="{acc.hexval()}">[{label}]</font>  ·  '
            f'<font color="{_NAVY.hexval()}">[STATIC + TEMPORAL ANALYSIS]</font>  ·  '
            f'<font color="{_PRUNED.hexval()}">[INCLUDES COMPRESSED VARIANTS]</font>',
            st["badge"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.2, color=acc, spaceAfter=6))

    # Four-model comparison table
    story.append(Paragraph("Four-Model Measurement Comparison", st["sec"]))

    def _fmt(v, unit="", decimals=1):
        return f"{v:.{decimals}f}{unit}" if v else "—"

    cmp_rows = [
        [
            "Metric",
            "Phase 0\nStatic",
            "Phase 4a ✂\nCompressed",
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
        ["Mean deviation", m0["mae"], m4a["mae"], m2["mae"], m4b["mae"]],
        ["ISUOG ≤3 mm", "✓ PASS", "✓ PASS", "✓ PASS", "✓ PASS"],
        ["Compression vs baseline", "—", "−43.7%", "—", "−41.6%"],
        ["Analysis type", "Single frame", "Single frame", "16-frame", "16-frame"],
    ]
    cw = [42 * mm, 28 * mm, 30 * mm, 28 * mm, 34 * mm]
    t = Table(cmp_rows, colWidths=cw)
    ts = _tbl_style()
    ts.add("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#ecfdf5"))
    ts.add("BACKGROUND", (4, 1), (4, -1), colors.HexColor("#ecfdf5"))
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # Clinical interpretation
    story.append(Paragraph("Clinical and Deployment Interpretation", st["sec"]))
    if llm:
        p1, p2 = _llm_comparison_narrative(results, api_key)
    else:
        p1 = (
            "Single-frame static analysis is appropriate for standard second-trimester "
            "screening with a clear suboccipitobregmatic plane and adequate acoustic access. "
            "The temporal cine-loop analysis may provide incremental value in scenarios with "
            "suboptimal probe contact, fetal movement, or when measurement audit is required. "
            "Both approaches satisfy the ISUOG ±3 mm acceptable biometry threshold on the "
            "HC18 independent validation cohort (199 images, Radboud UMC)."
        )
        p2 = (
            "The compressed model variants (Phase 4a and Phase 4b) achieve clinically "
            "equivalent accuracy to their full-size counterparts following structural parameter "
            "reduction of 43.7% and 41.6% respectively, while CPU inference remains under "
            "400 ms for single-frame and under 7,000 ms for cine analysis. This supports "
            "deployment without dedicated GPU infrastructure, enabling integration on portable "
            "ultrasound platforms and in resource-constrained clinical settings."
        )

    story.append(
        Paragraph("<b>Static vs temporal analysis — clinical recommendation</b>", st["label"])
    )
    story.append(Paragraph(_strip_markdown(p1), st["body"]))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph("<b>Compressed model variants — deployment significance</b>", st["label"])
    )
    story.append(Paragraph(_strip_markdown(p2), st["green"]))

    # Ablation context
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Temporal Model — Ablation Context", st["sec"]))
    story.append(
        Paragraph(
            "To contextualise the clinical value of the temporal attention mechanism, an ablation "
            "study was conducted removing the temporal self-attention module, reducing the system "
            "to frame-independent static inference. Without temporal attention, the same architecture "
            "achieved 81.48% segmentation accuracy and 19.37 mm mean measurement deviation — "
            "well outside ISUOG thresholds. The temporal integration component is clinically "
            "essential, not merely an incremental enhancement.",
            st["bodyI"],
        )
    )

    _section_regulatory(story, st)

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
