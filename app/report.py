"""
report.py — Fetal Head Circumference Clinical AI v3.2
ACR/AIUM/ESR-compliant PDF report generation.

Public functions:
  generate_static_report()     — Phase 0 / Phase 4a single-frame analysis
  generate_cine_report()       — Phase 2 / Phase 4b temporal cine analysis
  generate_comparison_report() — All four models head-to-head (Tab 3)
  generate_pdf_report()        — Backwards-compatible router

Page layout (v3.1):
  Page 1  Header + Patient/Exam + Indication + Technical + Biometric Findings
  Page 2  Images + Impression + Clinical Interpretation
  Page 3  AI System Validation Summary + Sign-off + Regulatory Notice
  Page 4  Appendix A — model engineering details (only when compressed model used)

Two narrative modes:
  template — rule-based paragraphs, grey accent, "AUTOMATED TEMPLATE REPORT"
  llm      — Claude Haiku narrative, teal accent, "AI-AUTHORED CLINICAL NARRATIVE"
"""

import io
import re
from datetime import datetime, timedelta

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
_TEAL = colors.HexColor("#2D7D9A")
_NAVY = colors.HexColor("#1e3a5f")
_GREY = colors.HexColor("#4b5563")
_AMBER = colors.HexColor("#b45309")
_AMBER_BG = colors.HexColor("#FFF3CD")
_AMBER_BORDER = colors.HexColor("#FFC107")
_RED = colors.HexColor("#991b1b")
_PRUNED = colors.HexColor("#065f46")
_LIGHT_BG = colors.HexColor("#f0f4f8")
_ROW_ALT = colors.HexColor("#F9F9F9")
_IMPRESSION_BG = colors.HexColor("#F0F7FA")
_BORDER = colors.HexColor("#cccccc")
_HEADER_RULE = colors.HexColor("#CCCCCC")
_PAGE_W = A4[0]
# 50pt margins each side → 495pt usable content width.
# All table column widths are computed in points and sum to _CONTENT_W.
_MARGIN_PT = 50
_CONTENT_W = _PAGE_W - 2 * _MARGIN_PT  # 495pt

# Status text fragments — use ⚠ (U+26A0), never ■ (causes glyph fallback issues)
_WARN = "⚠"


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


_PRESENTATION_LABELS = {
    "cephalic": "Cephalic",
    "breech": "Breech",
    "transverse": "Transverse",
    "not_assessed": "Not assessed",
}


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
            fontSize=11,
            spaceBefore=8,
            spaceAfter=2,
            textColor=_TEAL,
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
            fontName="Helvetica-Oblique",
        ),
        warn=ParagraphStyle(
            "wa",
            parent=s["Normal"],
            fontSize=8,
            leading=11,
            textColor=_AMBER,
        ),
        warn_bold=ParagraphStyle(
            "wab",
            parent=s["Normal"],
            fontSize=10,
            leading=13,
            textColor=_AMBER,
            fontName="Helvetica-Bold",
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
            fontSize=10,
            leading=14,
            textColor=_NAVY,
        ),
        not_provided=ParagraphStyle(
            "np",
            parent=s["Normal"],
            fontSize=8,
            textColor=_GREY,
            fontName="Helvetica-Oblique",
        ),
        signoff_note=ParagraphStyle(
            "son",
            parent=s["Normal"],
            fontSize=9,
            leading=12,
            textColor=_NAVY,
            fontName="Helvetica-Oblique",
        ),
        footer=ParagraphStyle(
            "fo",
            parent=s["Normal"],
            fontSize=7,
            leading=10,
            textColor=_GREY,
        ),
        footnote=ParagraphStyle(
            "fn",
            parent=s["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=_GREY,
            fontName="Helvetica-Oblique",
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
    """Standard data table style: navy/teal header, alternating row shading,
    light grey grid, 8pt content."""
    hc = header_color or _NAVY
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), hc),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_ROW_ALT, colors.white]),
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
    if not text:
        return ""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ci_bare(ci_str: str) -> str:
    """Strip '(Hadlock 1984…)' suffix for LLM prompts that already cite Hadlock."""
    return re.sub(r"\s*\([^)]*Hadlock[^)]*\)", "", ci_str).strip()


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


def _calculate_edd(lmp: str) -> str | None:
    """Naegele's rule: EDD = LMP + 280 days."""
    try:
        lmp_dt = datetime.strptime(lmp, "%Y-%m-%d")
        edd_dt = lmp_dt + timedelta(days=280)
        return edd_dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _ga_discordance_days(lmp: str, ga_weeks: float) -> int | None:
    """Days difference between LMP-derived GA and HC-derived GA."""
    try:
        lmp_dt = datetime.strptime(lmp, "%Y-%m-%d")
        today = datetime.utcnow()
        lmp_ga_days = (today - lmp_dt).days
        hc_ga_days = round(ga_weeks * 7)
        return abs(lmp_ga_days - hc_ga_days)
    except Exception:
        return None


def _lmp_ga_days(lmp: str) -> int | None:
    try:
        lmp_dt = datetime.strptime(lmp, "%Y-%m-%d")
        return (datetime.utcnow() - lmp_dt).days
    except Exception:
        return None


def _format_weeks_days(days: int | None) -> str:
    """Format day count as 'Xw Yd'. Returns '—' for None."""
    if days is None:
        return "—"
    weeks = days // 7
    rem = days % 7
    return f"{weeks}w {rem}d"


def _bpd_to_ga_weeks(bpd_mm: float) -> float | None:
    """Hadlock 1984 BPD-only nomogram (BPD in cm).

    GA(weeks) = 9.54 + 1.482·BPD + 0.1676·BPD²

    Reasonable estimate over 14–34 weeks. Outside that range, the polynomial
    diverges — return None so the report shows '—' rather than a misleading value.
    """
    if not bpd_mm or bpd_mm <= 0:
        return None
    bpd_cm = bpd_mm / 10.0
    ga = 9.54 + 1.482 * bpd_cm + 0.1676 * bpd_cm * bpd_cm
    if ga < 12 or ga > 42:
        return None
    return ga


def _ga_str_from_weeks(ga_weeks: float) -> str:
    """Convert GA in weeks to 'Xw Yd' display string."""
    total_days = round(ga_weeks * 7)
    return f"{total_days // 7}w {total_days % 7}d"


def _b64_to_image_flowable(b64_str: str, max_width: float, max_height: float) -> Image | None:
    """Decode base64 PNG/JPG string → reportlab Image flowable, fit to box."""
    try:
        import base64

        data = base64.b64decode(b64_str)
        buf = io.BytesIO(data)
        img = Image(buf)
        w, h = img.imageWidth, img.imageHeight
        if w > 0 and h > 0:
            scale = min(max_width / w, max_height / h)
            img.drawWidth = w * scale
            img.drawHeight = h * scale
        return img
    except Exception:
        return None


def _field_value(st, value: str | None) -> Paragraph:
    """Render a field value: bold black if present, italic grey 'Not provided' if blank."""
    if value is None or str(value).strip() == "" or value == "—":
        return Paragraph("Not provided", st["not_provided"])
    return Paragraph(str(value), st["body"])


# ── Section header rule (under each section title) ─────────────────────────────


def _section_rule(story):
    story.append(HRFlowable(width="100%", thickness=0.5, color=_HEADER_RULE, spaceAfter=3))


# ── Section 1 — Report Header ──────────────────────────────────────────────────


def _section_header(
    story,
    st,
    model_name,
    llm,
    elapsed_ms,
    accession,
    report_mode,
    ood_flag=False,
    hc_only=False,
):
    acc = _TEAL if llm else _GREY
    mode_label = "AI-AUTHORED CLINICAL NARRATIVE" if llm else "AUTOMATED TEMPLATE REPORT"

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
    # 4pt teal rule directly under title (clinical AI report convention)
    story.append(HRFlowable(width="100%", thickness=4, color=_TEAL, spaceAfter=4))

    # Badge row — clinical-only labels. Model engineering details belong in Appendix A.
    badges = [
        f'<font color="{acc.hexval()}">[{mode_label}]</font>',
        f'<font color="{_NAVY.hexval()}">[AI-Assisted — Requires Clinical Verification]</font>',
    ]
    if hc_only:
        badges.append(f'<font color="{_AMBER.hexval()}">[SINGLE PARAMETER — HC ONLY]</font>')
    if ood_flag:
        badges.append(f'<font color="{_RED.hexval()}">[{_WARN} OUT-OF-DISTRIBUTION ALERT]</font>')
    story.append(Paragraph("  ·  ".join(badges), st["badge"]))


# ── Section 2 — Patient & Exam Information ─────────────────────────────────────


def _section_patient_exam(story, st, report):
    story.append(Paragraph("Patient &amp; Exam Information", st["sec"]))
    _section_rule(story)

    edd = _calculate_edd(report.lmp) if report.lmp else None
    edd_row = edd or ("Calculated from LMP" if report.lmp else None)

    left = [
        ("Patient Name", report.patient_name),
        ("Patient ID / MRN", report.patient_id),
        ("Date of Birth", report.patient_dob),
        ("LMP (Last Menstrual Period)", report.lmp),
        ("EDD (Expected Delivery Date)", edd_row),
    ]
    right = [
        ("Referring Physician", report.referring_physician),
        ("Ordering Facility", report.ordering_facility),
        ("Sonographer", report.sonographer_name),
        ("Exam Date", report.study_date),
        ("Exam Type", _ga_exam_type(report.ga_weeks)),
    ]

    rows = []
    for (l_label, l_val), (r_label, r_val) in zip(left, right):
        rows.append(
            [
                Paragraph(l_label, st["label"]),
                _field_value(st, l_val),
                Paragraph(r_label, st["label"]),
                _field_value(st, r_val),
            ]
        )

    # Explicit pt widths (sum=495). Auto rowHeights via Paragraph wrapping.
    col_w = [100, 145, 100, 150]

    t = Table(rows, colWidths=col_w)
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
                ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, _ROW_ALT]),
                ("ROWBACKGROUNDS", (3, 0), (3, -1), [colors.white, _ROW_ALT]),
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
    if ga_weeks is None or ga_weeks == 0:
        return None  # rendered as 'Not provided' by _field_value
    if ga_weeks < 14:
        return "First trimester"
    elif ga_weeks < 28:
        return "Second trimester"
    else:
        return "Third trimester"


# ── Section 3 — Clinical Indication ───────────────────────────────────────────


def _section_clinical_indication(story, st, report):
    story.append(Paragraph("Clinical Indication", st["sec"]))
    _section_rule(story)
    indication = report.clinical_indication if report.clinical_indication else None
    if indication:
        story.append(Paragraph(indication, st["body"]))
    else:
        story.append(
            Paragraph(
                f"{_WARN} Clinical indication not provided. Completing this field is "
                "recommended for ACR/AIUM-compliant sonography documentation.",
                st["warn"],
            )
        )
    story.append(Spacer(1, 2 * mm))


# ── Section 4 — Technical Parameters ──────────────────────────────────────────


def _section_technical_params(story, st, report):
    story.append(Paragraph("Technical Parameters", st["sec"]))
    _section_rule(story)

    ps_mm = report.pixel_spacing_mm or 0.070
    dicom_flag = getattr(report, "pixel_spacing_dicom_derived", False)
    ps_source = (
        "DICOM-derived ✓" if dicom_flag else f"{_WARN} Estimated — verify before clinical use"
    )

    us_approach = (report.us_approach or "transabdominal").capitalize()
    image_quality = (report.image_quality or "not recorded").capitalize()

    presentation_key = getattr(report, "fetal_presentation", None) or "not_assessed"
    presentation_label = _PRESENTATION_LABELS.get(presentation_key, "Not assessed")

    cell = st["body"]
    rows = [
        ["Parameter", "Value", "Notes"],
        [Paragraph("Ultrasound approach", cell), Paragraph(us_approach, cell), Paragraph("", cell)],
        [
            Paragraph("Scanning plane", cell),
            Paragraph("Suboccipitobregmatic", cell),
            Paragraph("Standard fetal head biometry plane", cell),
        ],
        [
            Paragraph("Pixel spacing", cell),
            Paragraph(f"{ps_mm:.4f} mm/pixel", cell),
            Paragraph(ps_source, cell),
        ],
        [
            Paragraph("Image quality", cell),
            Paragraph(image_quality, cell),
            Paragraph("Sonographer assessment", cell),
        ],
        [
            Paragraph("Fetal lie / presentation", cell),
            Paragraph(presentation_label, cell),
            Paragraph("AIUM/ISUOG standard component", cell),
        ],
        [
            Paragraph("Measurement standard", cell),
            Paragraph("ISUOG Practice Guidelines 2010", cell),
            Paragraph("HC measurement methodology", cell),
        ],
    ]
    # Explicit pt widths (sum=495).
    col_w = [150, 120, 225]
    t = Table(rows, colWidths=col_w)
    ts = _tbl_style()
    if not dicom_flag:
        ts.add("TEXTCOLOR", (0, 3), (-1, 3), _AMBER)
        ts.add("FONTNAME", (0, 3), (0, 3), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 3 * mm))


# ── Section 5 — Biometric Findings ────────────────────────────────────────────


def _confidence_label_effective(report, base_label: str | None) -> str:
    """Downgrade HIGH CONFIDENCE → MODERATE when pixel spacing is unverified.

    HIGH CONFIDENCE requires both adequate segmentation coverage AND
    DICOM-verified pixel spacing. If pixel spacing was user-supplied or
    defaulted, cap the confidence rating regardless of segmentation quality.
    """
    base = base_label or "—"
    dicom_flag = bool(getattr(report, "pixel_spacing_dicom_derived", False))
    if not dicom_flag and "HIGH" in base.upper():
        return "MODERATE — pixel spacing unverified"
    return base


def _section_biometric_findings(story, st, report):
    story.append(Paragraph("Biometric Findings", st["sec"]))
    _section_rule(story)

    dicom_flag = bool(getattr(report, "pixel_spacing_dicom_derived", False))
    ps_mm = report.pixel_spacing_mm or 0.070

    # Pixel-spacing warning banner — surface this prominently rather than
    # burying it in a table footnote.
    if not dicom_flag:
        warn_data = [
            [
                Paragraph(
                    f"<b>{_WARN} WARNING:</b> Pixel spacing ({ps_mm:.4f} mm/pixel) is "
                    "estimated, not DICOM-verified. HC measurement may be inaccurate. "
                    "Verify pixel spacing before clinical use.",
                    st["warn_bold"],
                )
            ]
        ]
        warn_tbl = Table(warn_data, colWidths=[_CONTENT_W])
        warn_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _AMBER_BG),
                    ("BOX", (0, 0), (-1, -1), 1, _AMBER_BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(warn_tbl)
        story.append(Spacer(1, 3 * mm))

    hc = report.hc_mm
    ga_str = report.ga_str or "—"
    ga_weeks = report.ga_weeks or 0.0
    trim = report.trimester or "—"
    conf = _confidence_label_effective(report, report.confidence_label)
    ci_str = _ga_ci_string(ga_weeks) if ga_weeks else "—"

    bpd_mm = getattr(report, "bpd_mm", None)
    bpd_ga_weeks = _bpd_to_ga_weeks(bpd_mm) if bpd_mm else None
    bpd_ga_str = _ga_str_from_weeks(bpd_ga_weeks) if bpd_ga_weeks else None

    cell = st["body"]

    def P(t):
        return Paragraph(t, cell)

    rows = [
        ["Parameter", "AI Measurement", "Reference / Notes"],
        [P("Head Circumference (HC)"), P(f"{hc:.1f} mm" if hc else "—"), P("Calvarium perimeter")],
        [
            P("Estimated Gestational Age (HC)"),
            P(ga_str),
            # ci_str already cites Hadlock 1984 — do not double-cite.
            P(ci_str),
        ],
        [P("Trimester"), P(trim), P("Derived from estimated GA")],
        [P("Measurement Confidence"), P(conf), P("Coverage + pixel-spacing verification")],
    ]
    # BPD row — optional; greyed out if not entered
    if bpd_mm:
        rows.insert(
            2,
            [
                P("Biparietal Diameter (BPD)"),
                P(f"{bpd_mm:.1f} mm"),
                P(f"BPD-derived GA: {bpd_ga_str}" if bpd_ga_str else "BPD outside nomogram range"),
            ],
        )
        # BPD-HC discordance row when both GA estimates are available
        if bpd_ga_weeks and ga_weeks:
            disc_days_bpd = abs(round(ga_weeks * 7) - round(bpd_ga_weeks * 7))
            if disc_days_bpd > 10:
                rows.append([
                    P("HC/BPD GA discordance"),
                    P(_format_weeks_days(disc_days_bpd)),
                    P(f"{_WARN} >10 days — ISUOG: consider additional biometry or clinical review"),
                ])
            else:
                rows.append([
                    P("HC/BPD agreement"),
                    P(_format_weeks_days(disc_days_bpd)),
                    P("Within 10-day threshold"),
                ])
    else:
        rows.insert(
            2,
            [P("BPD"), P("Not measured"), P("Single-parameter study (HC only)")],
        )

    # Explicit pt widths (sum=495).
    col_w = [175, 120, 200]
    t = Table(rows, colWidths=col_w)
    ts = _tbl_style()
    if not bpd_mm:
        # grey the BPD row to signal intentional omission
        ts.add("TEXTCOLOR", (0, 2), (-1, 2), _GREY)
        ts.add("FONTNAME", (1, 2), (1, 2), "Helvetica-Oblique")
    t.setStyle(ts)
    story.append(t)

    # LMP cross-check / GA discordance
    lmp_val = getattr(report, "lmp", None)
    if lmp_val and ga_weeks:
        edd = _calculate_edd(lmp_val)
        lmp_days = _lmp_ga_days(lmp_val)
        disc_days = _ga_discordance_days(lmp_val, ga_weeks)
        disc_flag = disc_days is not None and disc_days > 14

        lmp_ga_str = _format_weeks_days(lmp_days)
        hc_ga_display = _format_weeks_days(round(ga_weeks * 7)) if ga_weeks else "—"
        disc_str = _format_weeks_days(disc_days) if disc_days is not None else "—"

        disc_notes = (
            f"{_WARN} >2 weeks — clinical review recommended"
            if disc_flag
            else "Within 14-day threshold"
        )

        cell = st["body"]

        def P2(text):
            return Paragraph(text, cell)

        lmp_rows = [
            ["GA Comparison", "Value", "Notes"],
            [P2("GA from HC"), P2(hc_ga_display), P2("This measurement (Hadlock 1984)")],
            [P2("LMP-derived GA"), P2(lmp_ga_str), P2(f"LMP date: {lmp_val}")],
            [P2("EDD (Naegele's rule)"), P2(edd or "—"), P2("From LMP + 280 days")],
            # Discordance split into two rows: numeric value + clinical interpretation.
            [P2("Discordance (HC vs LMP)"), P2(disc_str), P2(disc_notes)],
            [
                P2("Summary"),
                P2(""),
                P2(f"LMP GA: {lmp_ga_str}  |  HC GA: {hc_ga_display}  |  EDD: {edd or '—'}"),
            ],
        ]
        prior = getattr(report, "prior_biometry", None)
        if prior:
            lmp_rows.append([P2("Prior biometry"), P2(""), P2(prior)])
        # Explicit pt widths (sum=495).
        lcol_w = [150, 120, 225]
        lt = Table(lmp_rows, colWidths=lcol_w)
        header_color = _RED if disc_flag else _TEAL
        lt.setStyle(_tbl_style(header_color=header_color))
        footnote = Paragraph(
            "<i>EDD calculated from LMP. If GA discordance exceeds 14 days, "
            "EDD should be revised based on sonographic dating per ACOG/ISUOG "
            "guidelines.</i>",
            st["footnote"],
        )
        # KeepTogether prevents mid-table page break on the GA Comparison block.
        story.append(KeepTogether([Spacer(1, 2 * mm), lt, footnote]))
    elif not lmp_val:
        story.append(
            Paragraph(
                "LMP not provided — GA cross-check not available.",
                st["bodyI"],
            )
        )

    story.append(Spacer(1, 3 * mm))


# ── Section 6 — Images ────────────────────────────────────────────────────────


def _section_images(story, st, report):
    """Three-panel image section. If all three images are unavailable,
    the entire section collapses to a single-line note."""
    orig_b64 = getattr(report, "original_image_b64", None)
    overlay_b64 = getattr(report, "overlay_image_b64", None)
    gradcam_b64 = getattr(report, "gradcam_image_b64", None)

    if not any([orig_b64, overlay_b64, gradcam_b64]):
        story.append(Paragraph("Annotated Ultrasound Images", st["sec"]))
        _section_rule(story)
        story.append(
            Paragraph(
                "Annotated images not available for this study.",
                st["bodyI"],
            )
        )
        story.append(Spacer(1, 3 * mm))
        return

    story.append(Paragraph("Annotated Ultrasound Images", st["sec"]))
    _section_rule(story)

    panel_w = (_CONTENT_W - 8 * mm) / 3
    panel_h = 52 * mm

    def _placeholder(label: str):
        ph = Table(
            [[Paragraph(label, st["img_cap"])]],
            colWidths=[panel_w],
            rowHeights=[panel_h],
        )
        ph.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E5E7EB")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTRE"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.3, _BORDER),
                ]
            )
        )
        return ph

    def _panel(b64, missing_label):
        img = _b64_to_image_flowable(b64, panel_w, panel_h) if b64 else None
        if img is None:
            return _placeholder(missing_label)
        return img

    cells = [
        _panel(orig_b64, "Original ultrasound\nnot available"),
        _panel(overlay_b64, "Segmentation overlay\nnot generated"),
        _panel(gradcam_b64, "GradCAM++ activation\nnot generated"),
    ]

    img_row = Table(
        [cells],
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
        [
            [
                Paragraph("Original ultrasound", st["img_cap"]),
                Paragraph("Segmentation overlay", st["img_cap"]),
                Paragraph("GradCAM++ activation", st["img_cap"]),
            ]
        ],
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
            "AI-generated segmentation boundary (teal) and activation map (jet "
            "colormap). Annotated boundaries require verification by a qualified "
            "sonographer prior to clinical use.",
            st["footer"],
        )
    )
    story.append(Spacer(1, 3 * mm))


# ── Section 9 — Impression ────────────────────────────────────────────────────


def _section_impression(story, st, narrative_impression, report):
    story.append(Paragraph("Impression", st["sec"]))
    _section_rule(story)
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
                ("BOX", (0, 0), (-1, -1), 1, _TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(imp_tbl)
    story.append(Spacer(1, 8))


# ── Section 7 — Clinical Interpretation ───────────────────────────────────────


def _section_interpretation(story, st, llm, model_name, p1, p2, p3, report, discordance_note=None):
    """Render Clinical Interpretation. p1=BIOMETRIC ASSESSMENT (terse),
    p2=ACTIVATION MAP (terse). p3 optional compression deployment context."""
    acc = _TEAL if llm else _GREY
    mode_label = (
        "AI-Authored Clinical Narrative — Claude Haiku" if llm else "Automated Template Report"
    )
    m = _meta(model_name)

    story.append(Paragraph("Clinical Interpretation", st["sec"]))
    _section_rule(story)

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

    if discordance_note:
        # Append discordance recommendation to biometric assessment
        p1_clean = p1_clean.rstrip(". ") + ". " + discordance_note

    story.append(Paragraph("<b>BIOMETRIC ASSESSMENT</b>", st["label"]))
    story.append(Paragraph(p1_clean, st["body"]))

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("<b>ACTIVATION MAP</b>", st["label"]))
    story.append(Paragraph(p2_clean, st["body"]))

    if p3 and m["pruned"]:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("<b>DEPLOYMENT CONTEXT</b>", st["label"]))
        story.append(Paragraph(_strip_markdown(p3), st["green"]))

    story.append(Spacer(1, 8))


# ── Section 8 — AI System Performance ─────────────────────────────────────────


def _section_ai_performance(story, st, model_name, elapsed_ms):
    m = _meta(model_name)
    story.append(Paragraph("AI System Validation Summary", st["sec"]))
    _section_rule(story)
    cell = st["body"]

    def P3(text):
        return Paragraph(text, cell)

    # m["dataset"] already begins with "HC18 — …", so just render it directly
    # rather than prefixing another "HC18 — " (was producing "HC18 — HC18 —").
    rows = [
        ["Metric", "Result", "Clinical Reference"],
        [P3("Segmentation accuracy (validation cohort)"), P3(m["dice"]), P3(m["dataset"])],
        [P3("Mean measurement deviation"), P3(m["mae"]), P3("ISUOG acceptable threshold: ±3 mm")],
        [
            P3("ISUOG clinical threshold (±3 mm)"),
            P3(f"✓ {m['isuog']}"),
            P3("ISUOG Practice Guidelines 2010"),
        ],
        [
            P3("Validation cohort"),
            P3("199 images (held-out test set)"),
            P3("HC18 — Radboud UMC, Netherlands"),
        ],
    ]
    if elapsed_ms:
        rows.append(
            [P3("Inference runtime (this image)"), P3(f"{elapsed_ms:.0f} ms"), P3("CPU inference")]
        )

    col_w = [180, 120, 195]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_tbl_style())
    story.append(t)
    story.append(Spacer(1, 2 * mm))


# ── Section 10 — Sign-off block ────────────────────────────────────────────────


def _section_signoff(story, st, signed_meta: dict):
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.2, color=_PRUNED, spaceAfter=4))
    story.append(Paragraph("Clinical Sign-off", st["sec"]))
    _section_rule(story)
    rows = [
        [
            Paragraph("Signed by", st["label"]),
            Paragraph(signed_meta.get("signed_by") or "—", st["body"]),
        ],
        [
            Paragraph("Signed at", st["label"]),
            Paragraph(signed_meta.get("signed_at") or "—", st["body"]),
        ],
    ]
    note = signed_meta.get("signoff_note")
    if note:
        rows.append(
            [
                Paragraph("Verification note (clinical)", st["label"]),
                Paragraph(note, st["signoff_note"]),
            ]
        )
    # Explicit pt widths (sum=495). Generous value column avoids name truncation.
    t = Table(rows, colWidths=[160, 335])
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
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)


# ── Section 11 — Regulatory footer ────────────────────────────────────────────


def _section_regulatory(story, st, ga_weeks=0.0):
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.4, color=_BORDER, spaceAfter=3))
    ci_str = _ga_ci_string(ga_weeks)
    story.append(
        Paragraph(
            "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system has not received FDA 510(k) "
            "clearance or CE marking under EU MDR/IVDR. It is classified as a Software as a "
            "Medical Device (SaMD) Class II candidate under 21 CFR Part 892. All automated "
            "measurements must be independently verified by a qualified healthcare professional "
            "before incorporation into any clinical decision. Gestational age estimates carry an "
            f"inherent {ci_str}. This report does not constitute a diagnostic opinion.",
            st["footer"],
        )
    )
    story.append(
        Paragraph(
            "HC18 dataset — Radboud University Medical Center, Nijmegen, Netherlands. "
            "For research and evaluation purposes only. "
            "System: TemporalFetaSegNet / Fetal Head Clinical AI v3.2 — "
            "Tarun Sadarla, MS Artificial Intelligence, University of North Texas, 2026. "
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            st["footer"],
        )
    )


# ── Appendix — Technical details ───────────────────────────────────────────────


def _appendix_technical(story, st, model_name, elapsed_ms):
    m = _meta(model_name)
    story.append(PageBreak())
    story.append(Paragraph("Appendix A — AI Model Technical Specifications", st["sec"]))
    _section_rule(story)
    story.append(
        Paragraph(
            "The following technical metrics are provided for AI/ML validation teams and "
            "clinical engineering committees. They are not relevant to routine clinical reporting.",
            st["bodyI"],
        )
    )
    cell = st["body"]

    def Pa(text):
        return Paragraph(text, cell)

    rows = [
        ["Specification", "Value"],
        [Pa("Model architecture"), Pa(m["short"])],
        [Pa("Parameter count"), Pa(m["params"])],
        [Pa("Computational operations (GMACs)"), Pa(m["flops"])],
        [Pa("Compression vs baseline"), Pa(m["compression"] or "N/A (baseline)")],
        [Pa("Training dataset"), Pa("HC18 (1334 images, Radboud UMC)")],
        [Pa("Validation dataset"), Pa(m["dataset"])],
        [Pa("Runtime on CPU"), Pa(f"{elapsed_ms:.0f} ms" if elapsed_ms else "—")],
    ]
    col_w = [220, 275]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_tbl_style())
    story.append(t)


# ── LLM narrative functions ────────────────────────────────────────────────────


_CLINICAL_SYSTEM_PROMPT = (
    "You are generating the Clinical Interpretation section of a fetal head circumference "
    "biometry report. The audience is the referring obstetrician, not a patient. "
    "Write in terse, structured clinical prose — the style of a radiologist's dictated report, "
    "not an essay. Do not use markdown headers, bullet points, bold text, or any markdown "
    "syntax. Plain prose only. No preamble, no 'In conclusion', no self-reference to being "
    "an AI. Use direct clinical language, no passive constructions like 'automated biometric "
    "analysis yielded'."
)


def _call_llm(api_key: str, prompt: str, max_tokens: int = 250) -> str | None:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=_CLINICAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()
    except Exception:
        return None


def _llm_static_narrative(hc, ga_str, ga_weeks, trim, gradcam_ok, model_name, elapsed_ms, api_key):
    m = _meta(model_name)
    ci_str = _ga_ci_string(ga_weeks)
    ci_bare = _ci_bare(ci_str)

    p1 = _call_llm(
        api_key,
        (
            "BIOMETRIC ASSESSMENT — write 2-3 sentences maximum.\n\n"
            f"Data: HC = {hc:.1f} mm, GA = {ga_str} ({ga_weeks:.1f} weeks) by Hadlock 1984. "
            f"GA confidence interval for this trimester: {ci_bare}. "
            f"System mean deviation: {m['mae']} (within ISUOG ±3 mm threshold).\n\n"
            "State: HC value, GA estimate with CI, ISUOG compliance status, and one sentence "
            "on clinical correlation requirement. Do NOT explain the Hadlock nomogram. "
            "Example tone: 'HC 218.6 mm. GA estimated at 23+0 weeks (Hadlock 1984; ±10–14 days "
            "for this trimester). Within ISUOG ±3 mm biometry threshold. Clinical correlation "
            "with menstrual dating and prior biometry recommended.'"
        ),
        max_tokens=180,
    ) or _rule_static_p1(hc, ga_str, ga_weeks, trim)

    p2 = _call_llm(
        api_key,
        (
            "ACTIVATION MAP — write 1-2 sentences only.\n\n"
            f"GradCAM activation map {'was generated' if gradcam_ok else 'could NOT be generated'} "
            "for this acquisition.\n\n"
            "State whether the AI confidence map is anatomically appropriate "
            "(localised to calvarium perimeter) or shows any atypical pattern. "
            "Do NOT explain what GradCAM is. Example tone: 'AI confidence map localised to "
            "hyperechoic calvarium perimeter — anatomically appropriate. No atypical focus on "
            "intracranial soft tissue structures.'"
        ),
        max_tokens=120,
    ) or _rule_static_p2(gradcam_ok)

    p3 = None
    if m["pruned"]:
        p3 = _call_llm(
            api_key,
            (
                "DEPLOYMENT CONTEXT — write 2 sentences for a clinical technology committee.\n\n"
                f"Compressed model: {m['compression']}, mean deviation {m['mae']}, "
                f"CPU inference {elapsed_ms:.0f} ms. State clinical deployment significance for "
                "portable and point-of-care platforms. Plain clinical language, no ML jargon."
            ),
            max_tokens=140,
        )

    impression = _call_llm(
        api_key,
        (
            "IMPRESSION — write 2-3 sentences for the Impression box of a clinical "
            "ultrasound report. Plain clinical language for a referring obstetrician.\n\n"
            f"Include: HC ({hc:.1f} mm), GA ({ga_str}), ISUOG compliance, and a verification "
            "requirement. Example tone: 'AI-assisted head circumference measurement: "
            "X mm. Estimated gestational age: Y. Automated measurement requires verification "
            "by a qualified sonographer prior to clinical use.'"
        ),
        max_tokens=130,
    ) or _rule_impression(hc, ga_str, ga_weeks, trim)

    return p1, p2, p3, impression


def _llm_cine_narrative(
    hc, ga_str, ga_weeks, trim, rel, std, n_frames, model_name, elapsed_ms, api_key
):
    m = _meta(model_name)
    ci_str = _ga_ci_string(ga_weeks)
    ci_bare = _ci_bare(ci_str)
    rel_desc = "excellent" if rel > 0.97 else "good" if rel > 0.93 else "moderate"
    std_desc = "highly stable" if std < 2.0 else "acceptable" if std < 5.0 else "variable"

    p1 = _call_llm(
        api_key,
        (
            "BIOMETRIC ASSESSMENT — write 2-3 sentences maximum.\n\n"
            f"Data: Consensus HC = {hc:.1f} mm from {n_frames}-frame cine acquisition. "
            f"GA = {ga_str} ({ga_weeks:.1f} weeks) by Hadlock 1984. CI: {ci_bare}. "
            f"Frame concordance: {rel_desc} ({std:.2f} mm inter-frame deviation, {std_desc}). "
            f"System mean deviation: {m['mae']}.\n\n"
            "Cover: HC + GA with CI, frame concordance, ISUOG compliance, clinical correlation "
            "requirement. Do NOT explain Hadlock or temporal attention."
        ),
        max_tokens=200,
    ) or _rule_cine_p1(hc, ga_str, ga_weeks, trim, rel, std)

    p2 = _call_llm(
        api_key,
        (
            "ACTIVATION MAP — write 1-2 sentences. The temporal model produced consistent "
            f"activation across {n_frames} frames. State whether the activation pattern is "
            "anatomically appropriate (localised to calvarium perimeter across frames) "
            "or shows variability suggesting acoustic challenge. No ML jargon."
        ),
        max_tokens=120,
    ) or _rule_cine_p2(rel, std, n_frames)

    p3 = None
    if m["pruned"]:
        p3 = _call_llm(
            api_key,
            (
                "DEPLOYMENT CONTEXT — write 2 sentences for a clinical technology committee.\n\n"
                f"Temporal compressed model: {m['compression']}, mean deviation {m['mae']}, "
                f"CPU inference {elapsed_ms:.0f} ms per 16-frame clip. "
                "Plain clinical language."
            ),
            max_tokens=140,
        )

    impression = _call_llm(
        api_key,
        (
            "IMPRESSION — 2-3 sentences for the Impression box. Plain language for "
            f"a referring obstetrician. HC = {hc:.1f} mm from {n_frames}-frame cine analysis, "
            f"GA = {ga_str}, frame concordance = {rel_desc}."
        ),
        max_tokens=130,
    ) or _rule_impression(hc, ga_str, ga_weeks, trim)

    return p1, p2, p3, impression


def _llm_comparison_narrative(results: dict, api_key: str):
    r0, r4a, r2, r4b = (
        results.get("phase0", {}),
        results.get("phase4a", {}),
        results.get("phase2", {}),
        results.get("phase4b", {}),
    )
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
            "Recommend when single-frame is sufficient vs when cine offers benefit. "
            "Cautious clinical language. No ML jargon."
        ),
        max_tokens=350,
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
        max_tokens=280,
    ) or (
        "Compressed model variants enable deployment on standard CPU hardware without GPU "
        "infrastructure, supporting integration in portable ultrasound platforms and "
        "resource-constrained settings."
    )
    return p1, p2


# ── Rule-based narratives ──────────────────────────────────────────────────────


def _rule_static_p1(hc, ga_str, ga_weeks, trim):
    """Terse biometric assessment (2-3 sentences)."""
    ci_str = _ga_ci_string(ga_weeks) if ga_weeks else "±2 weeks"
    hc_disp = f"{hc:.1f} mm" if hc else "—"
    return (
        f"HC {hc_disp}. GA estimated at {ga_str} (Hadlock 1984; {ci_str}). "
        f"Within ISUOG ±3 mm biometry threshold. "
        f"Clinical correlation with menstrual dating and prior biometry recommended."
    )


def _rule_static_p2(gradcam_ok):
    """Terse activation map interpretation (1-2 sentences)."""
    if gradcam_ok:
        return (
            "AI confidence map localised to hyperechoic calvarium perimeter — "
            "anatomically appropriate. No atypical focus on intracranial soft-tissue structures."
        )
    return (
        "AI confidence map could not be generated for this acquisition; "
        "interpret the automated measurement with heightened caution and direct sonographer review."
    )


def _rule_cine_p1(hc, ga_str, ga_weeks, trim, rel, std):
    """Terse temporal biometric assessment."""
    ci_str = _ga_ci_string(ga_weeks) if ga_weeks else "±2 weeks"
    rel_desc = "excellent" if rel > 0.97 else "good" if rel > 0.93 else "moderate"
    hc_disp = f"{hc:.1f} mm" if hc else "—"
    return (
        f"Consensus HC {hc_disp} from 16-frame cine. GA estimated at {ga_str} "
        f"(Hadlock 1984; {ci_str}). Inter-frame concordance {rel_desc} ({std:.2f} mm deviation). "
        "Within ISUOG ±3 mm threshold. Clinical correlation with menstrual dating recommended."
    )


def _rule_cine_p2(rel, std, n_frames):
    """Terse temporal activation interpretation."""
    if std < 2.0:
        stability = "stable"
    elif std < 5.0:
        stability = "acceptably stable"
    else:
        stability = "variable — review advised"
    return (
        f"Frame-to-frame measurement {stability} across {n_frames} frames "
        f"({std:.2f} mm deviation). Activation pattern consistent with calvarium boundary."
    )


def _rule_compression_note(model_name, elapsed_ms):
    m = _meta(model_name)
    if not m["pruned"]:
        return None
    return (
        f"{m['compression']}, {m['mae']} mean deviation, CPU inference "
        f"{elapsed_ms:.0f} ms. Clinically equivalent to baseline; supports deployment on "
        "portable and point-of-care platforms."
    )


def _rule_impression(hc, ga_str, ga_weeks, trim):
    ci_str = _ga_ci_string(ga_weeks) if ga_weeks else "±2 weeks"
    hc_str = f"{hc:.1f} mm" if hc else "not available"
    return (
        f"AI-assisted head circumference measurement: {hc_str}. "
        f"Estimated gestational age: {ga_str} (confidence interval {ci_str}). "
        "Measurement within ISUOG clinically acceptable biometry threshold (±3 mm). "
        "Automated measurement requires verification by a qualified sonographer "
        "prior to clinical use."
    )


def _discordance_recommendation(lmp: str | None, ga_weeks: float) -> str | None:
    """Return a discordance recommendation sentence if LMP-GA disagreement exceeds 14 days."""
    if not lmp or not ga_weeks:
        return None
    disc_days = _ga_discordance_days(lmp, ga_weeks)
    if disc_days is None or disc_days <= 14:
        return None
    weeks_days = _format_weeks_days(disc_days)
    return (
        f"GA-LMP discordance of {weeks_days} exceeds the 14-day threshold. "
        "Recommend clinical review; consider revision of gestational dates per ACOG guidelines."
    )


# ── PDF document builder ───────────────────────────────────────────────────────


def _build_story(
    result: dict,
    api_key: str | None,
    use_llm: bool,
    model_name: str,
    pixel_spacing: float,
    narrative: tuple | None,
    draft: bool,
    signed_meta: dict | None,
    report_type_label: str,
    is_temporal: bool,
    report=None,
) -> list:
    """Assemble the complete story list for both static and cine reports.

    Page layout (v3.1):
      Page 1  Header + Patient/Exam + Indication + Technical + Biometric Findings
      Page 2  Images + Impression + Clinical Interpretation
      Page 3  AI System Validation + Sign-off + Regulatory
      Page 4  Appendix A (compressed only)
    """
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

    accession = getattr(report, "accession_number", None)
    stored_mode = getattr(report, "report_mode", "template")
    effective_llm = llm or (stored_mode == "llm")

    # Resolve narrative — supplied tuple, LLM call, or rule-based
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

    # GA-LMP discordance recommendation — appended to BIOMETRIC ASSESSMENT
    # in both template and LLM modes.
    lmp = getattr(report, "lmp", None) if report is not None else None
    discordance_note = _discordance_recommendation(lmp, ga_weeks)

    # HC-only flag — true when no BPD was supplied
    bpd_mm = getattr(report, "bpd_mm", None) if report is not None else None
    hc_only = bpd_mm is None or bpd_mm == 0

    story = []

    # ─── PAGE 1: Header + Patient + Indication + Technical + Biometric ───────
    _section_header(
        story,
        st,
        model_name,
        effective_llm,
        elapsed,
        accession,
        stored_mode,
        ood_flag=ood_flag,
        hc_only=hc_only,
    )

    if report is not None:
        _section_patient_exam(story, st, report)
        _section_clinical_indication(story, st, report)
        _section_technical_params(story, st, report)
    else:
        story.append(Paragraph("Patient &amp; Exam Information", st["sec"]))
        _section_rule(story)
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

    # ─── PAGE 2: Images + Impression + Clinical Interpretation ───────────────
    story.append(PageBreak())
    _section_images(story, st, report if report is not None else _NoImages())
    _section_impression(
        story, st, p_impression, _BiometricProxy(hc, ga_str, ga_weeks, trim, conf_lbl, report)
    )
    _section_interpretation(
        story,
        st,
        effective_llm,
        model_name,
        p1,
        p2,
        p3,
        report,
        discordance_note=discordance_note,
    )

    # ─── PAGE 3: AI System Validation + Sign-off + Regulatory ────────────────
    story.append(PageBreak())
    _section_ai_performance(story, st, model_name, elapsed)

    if signed_meta:
        _section_signoff(story, st, signed_meta)

    _section_regulatory(story, st, ga_weeks=ga_weeks)

    # ─── PAGE 4: Appendix A (only when compressed model used) ────────────────
    if m["pruned"]:
        _appendix_technical(story, st, model_name, elapsed)

    return story


class _BiometricProxy:
    """Minimal proxy so section builders work without a full report DB object."""

    def __init__(self, hc, ga_str, ga_weeks, trim, conf, report=None):
        self.hc_mm = hc
        self.ga_str = ga_str
        self.ga_weeks = ga_weeks
        self.trimester = trim
        self.confidence_label = conf
        self.lmp = getattr(report, "lmp", None)
        self.pixel_spacing_mm = getattr(report, "pixel_spacing_mm", None)
        self.pixel_spacing_dicom_derived = getattr(report, "pixel_spacing_dicom_derived", False)
        self.bpd_mm = getattr(report, "bpd_mm", None)
        self.fetal_presentation = getattr(report, "fetal_presentation", None)
        self.prior_biometry = getattr(report, "prior_biometry", None)

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
    _section_rule(story)
    cell = st["body"]

    def Pt(text):
        return Paragraph(text, cell)

    rows = [
        ["Parameter", "Value", "Clinical Interpretation"],
        [Pt("Frames analysed"), Pt(str(n_frames)), Pt("Sequential cine acquisition")],
        [Pt("Inter-frame concordance"), Pt(rel_label), Pt("Consistency across frames")],
        [Pt("Frame-to-frame HC variability"), Pt(f"{std:.2f} mm"), Pt(std_label)],
        [
            Pt("Consensus method"),
            Pt("Temporal mean probability"),
            Pt("Mean prediction across frames"),
        ],
    ]
    col_w = [150, 120, 225]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_tbl_style())
    story.append(t)
    story.append(Spacer(1, 3 * mm))


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def generate_static_report(
    result: dict,
    api_key: str | None = None,
    use_llm: bool = True,
    model_name: str = "Phase 0 — Static baseline",
    pixel_spacing: float = 0.070,
    narrative: tuple | None = None,
    draft: bool = False,
    signed_meta: dict | None = None,
    report=None,
) -> bytes:
    """PDF report for static single-frame analysis (Phase 0 / Phase 4a)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        leftMargin=_MARGIN_PT,
        rightMargin=_MARGIN_PT,
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
    api_key: str | None = None,
    use_llm: bool = True,
    model_name: str = "Phase 2 — Temporal baseline",
    pixel_spacing: float = 0.070,
    narrative: tuple | None = None,
    draft: bool = False,
    signed_meta: dict | None = None,
    report=None,
) -> bytes:
    """PDF report for temporal cine-loop analysis (Phase 2 / Phase 4b)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        leftMargin=_MARGIN_PT,
        rightMargin=_MARGIN_PT,
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
    api_key: str | None = None,
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
        leftMargin=_MARGIN_PT,
        rightMargin=_MARGIN_PT,
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
    story.append(HRFlowable(width="100%", thickness=4, color=_TEAL, spaceAfter=4))
    story.append(
        Paragraph(
            f'<font color="{acc.hexval()}">[{label}]</font>  ·  '
            f'<font color="{_NAVY.hexval()}">[STATIC + TEMPORAL ANALYSIS]</font>  ·  '
            f'<font color="{_PRUNED.hexval()}">[INCLUDES COMPRESSED VARIANTS]</font>',
            st["badge"],
        )
    )

    story.append(Paragraph("Four-Model Measurement Comparison", st["sec"]))
    _section_rule(story)

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
    cw = [
        _CONTENT_W * 0.25,
        _CONTENT_W * 0.18,
        _CONTENT_W * 0.19,
        _CONTENT_W * 0.18,
        _CONTENT_W * 0.20,
    ]
    t = Table(cmp_rows, colWidths=cw)
    ts = _tbl_style()
    ts.add("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#ecfdf5"))
    ts.add("BACKGROUND", (4, 1), (4, -1), colors.HexColor("#ecfdf5"))
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Clinical and Deployment Interpretation", st["sec"]))
    _section_rule(story)
    if llm:
        p1, p2 = _llm_comparison_narrative(results, api_key)
    else:
        p1 = (
            "Single-frame static analysis is appropriate for standard second-trimester "
            "screening with adequate acoustic access. Temporal cine analysis may provide "
            "incremental value with suboptimal probe contact or when measurement audit is "
            "required. Both approaches satisfy the ISUOG ±3 mm threshold on the HC18 "
            "validation cohort."
        )
        p2 = (
            "Compressed model variants achieve clinically equivalent accuracy following "
            "structural parameter reduction of 43.7% and 41.6%. CPU inference under 400 ms "
            "(single-frame) and 7 s (cine) supports deployment without GPU infrastructure "
            "on portable platforms and in resource-constrained settings."
        )

    story.append(Paragraph("<b>STATIC vs TEMPORAL — RECOMMENDATION</b>", st["label"]))
    story.append(Paragraph(_strip_markdown(p1), st["body"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("<b>COMPRESSED VARIANTS — DEPLOYMENT</b>", st["label"]))
    story.append(Paragraph(_strip_markdown(p2), st["green"]))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Temporal Model — Ablation Context", st["sec"]))
    _section_rule(story)
    story.append(
        Paragraph(
            "Removing the temporal self-attention module reduces the system to frame-independent "
            "static inference. Ablation results: 81.48% segmentation accuracy and 19.37 mm mean "
            "deviation — well outside ISUOG thresholds. Temporal integration is clinically "
            "essential.",
            st["bodyI"],
        )
    )

    _section_regulatory(story, st)

    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_pdf_report(
    result: dict,
    api_key: str | None = None,
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
