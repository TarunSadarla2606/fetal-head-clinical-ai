"""Demo data seeding — Batch 8.2.

Idempotently inserts 10 fabricated patient reports into the SQLite store
so reviewers see a populated worklist + Reports tab on first open. Three
of the ten cases are abnormal so the demo also exercises clinically-
interesting flags (microcephaly / macrocephaly / IUGR).

Mapping
-------
- Study IDs `demo-001` … `demo-010` mirror the worklist ordering used by
  the webapp (see WorkstationView.loadDemoStudies()), so when a reviewer
  clicks the matching study the seeded report appears in the Reports
  panel without any extra wiring.
- Each study gets one report. Three of the ten carry an explicit
  abnormal-finding indication so the inter-observer threshold flags
  surface in the PDF.

Idempotency
-----------
- `seed_demo_reports()` is a no-op if any seeded report already exists
  for any of the demo-* study IDs. Safe to call on every container
  startup; safe to call multiple times.
"""

from __future__ import annotations

from . import reports_db


def _hadlock_mean_hc(ga_weeks: float) -> float:
    """Approximate population-mean HC at a given GA via the inverse Hadlock
    polynomial (used only to make the abnormal-finding HC values physically
    plausible — not for any live measurement)."""
    # Coarse inverse: brute-force search across the same HC grid the
    # server-side growth chart uses.
    from app.inference import hadlock_ga as _to_ga

    best_hc = 200.0
    best_diff = 1e9
    for hc in range(60, 401, 2):
        ga, _ = _to_ga(float(hc))
        diff = abs(ga - ga_weeks)
        if diff < best_diff:
            best_diff = diff
            best_hc = float(hc)
    return best_hc


# Curated 10-patient demo cohort. 7 normal-for-GA + 3 abnormal cases.
# GA weeks → HC mm calibrated against Hadlock 1984: normals sit on the
# mean curve, abnormals are pushed >2 SD off (±4 % of mean per the
# growth-chart approximation we use everywhere else).
_SEED_PATIENTS: list[dict] = [
    # --- 7 normals ---------------------------------------------------------
    {
        "study_id": "demo-001",
        "patient_name": "Sarah Thompson",
        "patient_id": "MRN-DEMO-001",
        "patient_dob": "1995-03-12",
        "study_date": "2026-04-29",
        "lmp": "2025-12-08",
        "ga_weeks": 21.0,
        "hc_mm": _hadlock_mean_hc(21.0),
        "model": "phase4a",
        "indication": "Routine 2nd-trimester anatomy survey.",
        "image_quality": "optimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "cephalic",
        "narrative_impression": (
            "HC measurement within population norm for stated gestational age. "
            "No abnormality flagged."
        ),
    },
    {
        "study_id": "demo-002",
        "patient_name": "Maria Santos",
        "patient_id": "MRN-DEMO-002",
        "patient_dob": "1991-08-22",
        "study_date": "2026-04-28",
        "lmp": "2025-11-26",
        "ga_weeks": 22.4,
        # ABNORMAL: HC ~12% below mean → microcephaly flag
        "hc_mm": _hadlock_mean_hc(22.4) * 0.88,
        "model": "phase0",
        "indication": (
            "MFM referral — sonographic micrognathia / suspected microcephaly. Detailed biometry."
        ),
        "image_quality": "suboptimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "cephalic",
        "narrative_impression": (
            "Head circumference falls more than 2 SD below the population mean "
            "for stated gestational age. Findings consistent with microcephaly; "
            "MFM consultation and follow-up biometry recommended."
        ),
        "abnormal": True,
    },
    {
        "study_id": "demo-003",
        "patient_name": "Aisha Patel",
        "patient_id": "MRN-DEMO-003",
        "patient_dob": "1994-01-04",
        "study_date": "2026-04-27",
        "lmp": "2026-01-09",
        "ga_weeks": 16.5,
        "hc_mm": _hadlock_mean_hc(16.5),
        "model": "phase4a",
        "indication": "Dating scan — early-pregnancy biometry.",
        "image_quality": "optimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "not_assessed",
        "narrative_impression": (
            "Early-pregnancy biometry consistent with stated gestational age."
        ),
    },
    {
        "study_id": "demo-004",
        "patient_name": "Linda Chen",
        "patient_id": "MRN-DEMO-004",
        "patient_dob": "1989-11-30",
        "study_date": "2026-04-26",
        "lmp": "2025-11-08",
        "ga_weeks": 24.1,
        "hc_mm": _hadlock_mean_hc(24.1),
        "model": "phase2",
        "indication": "Routine fetal anatomy + growth assessment.",
        "image_quality": "optimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "cephalic",
        "narrative_impression": "Biometry within normal population range.",
    },
    {
        "study_id": "demo-005",
        "patient_name": "Jessica Brown",
        "patient_id": "MRN-DEMO-005",
        "patient_dob": "1986-06-17",
        "study_date": "2026-04-25",
        "lmp": "2025-08-21",
        "ga_weeks": 34.5,
        # ABNORMAL: HC ~12% above mean → macrocephaly flag
        "hc_mm": _hadlock_mean_hc(34.5) * 1.12,
        "model": "phase0",
        "indication": (
            "Late-third-trimester scan — head circumference greater than expected "
            "by clinical fundal-height measurement."
        ),
        "image_quality": "suboptimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "cephalic",
        "narrative_impression": (
            "Head circumference exceeds population mean by more than 2 SD for "
            "stated gestational age. Findings consistent with macrocephaly; "
            "consider intracranial imaging and MFM consultation."
        ),
        "abnormal": True,
    },
    {
        "study_id": "demo-006",
        "patient_name": "Emily Davis",
        "patient_id": "MRN-DEMO-006",
        "patient_dob": "1996-04-04",
        "study_date": "2026-04-24",
        "lmp": "2026-01-02",
        "ga_weeks": 16.9,
        "hc_mm": _hadlock_mean_hc(16.9),
        "model": "phase4a",
        "indication": "Initial dating scan.",
        "image_quality": "optimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "not_assessed",
        "narrative_impression": "Biometry within normal population range.",
    },
    {
        "study_id": "demo-007",
        "patient_name": "Olivia Johnson",
        "patient_id": "MRN-DEMO-007",
        "patient_dob": "1992-09-19",
        "study_date": "2026-04-23",
        "lmp": "2025-10-12",
        "ga_weeks": 28.0,
        "hc_mm": _hadlock_mean_hc(28.0),
        "model": "phase2",
        "indication": "Third-trimester growth assessment.",
        "image_quality": "optimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "cephalic",
        "narrative_impression": "Biometry within normal population range.",
    },
    {
        "study_id": "demo-008",
        "patient_name": "Priya Kumar",
        "patient_id": "MRN-DEMO-008",
        "patient_dob": "1990-12-01",
        "study_date": "2026-04-22",
        "lmp": "2025-11-04",
        "ga_weeks": 24.6,
        # ABNORMAL: HC ~7% below mean → IUGR flag
        "hc_mm": _hadlock_mean_hc(24.6) * 0.93,
        "model": "phase4b",
        "indication": (
            "MFM referral — clinically small for dates. Rule out fetal growth "
            "restriction; Doppler studies pending."
        ),
        "image_quality": "suboptimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "cephalic",
        "narrative_impression": (
            "Head circumference falls below the 5th percentile for stated "
            "gestational age. Findings consistent with intrauterine growth "
            "restriction (IUGR); umbilical artery Doppler and weekly biophysical "
            "profile recommended."
        ),
        "abnormal": True,
    },
    {
        "study_id": "demo-009",
        "patient_name": "Hannah Garcia",
        "patient_id": "MRN-DEMO-009",
        "patient_dob": "1993-02-15",
        "study_date": "2026-04-21",
        "lmp": "2025-10-19",
        "ga_weeks": 26.7,
        "hc_mm": _hadlock_mean_hc(26.7),
        "model": "phase4a",
        "indication": "Routine third-trimester biometry.",
        "image_quality": "optimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "cephalic",
        "narrative_impression": "Biometry within normal population range.",
    },
    {
        "study_id": "demo-010",
        "patient_name": "Grace Williams",
        "patient_id": "MRN-DEMO-010",
        "patient_dob": "1988-07-26",
        "study_date": "2026-04-20",
        "lmp": "2025-09-14",
        "ga_weeks": 31.3,
        "hc_mm": _hadlock_mean_hc(31.3),
        "model": "phase2",
        "indication": "Routine third-trimester anatomy and growth.",
        "image_quality": "optimal",
        "us_approach": "transabdominal",
        "fetal_presentation": "cephalic",
        "narrative_impression": "Biometry within normal population range.",
    },
]


def _ga_str_from_weeks(ga_weeks: float) -> str:
    weeks = int(ga_weeks)
    days = round((ga_weeks - weeks) * 7)
    if days == 7:
        weeks += 1
        days = 0
    return f"{weeks}w {days}d"


def _trimester(ga_weeks: float) -> str:
    if ga_weeks < 14:
        return "First trimester (<14w)"
    if ga_weeks < 28:
        return "Second trimester (14–28w)"
    return "Third trimester (≥28w)"


def seed_demo_reports(db_path: str | None = None, *, force: bool = False) -> int:
    """Idempotently insert the 10 demo reports.

    Returns the number of newly-inserted rows. Skips any study_id that
    already has a report unless `force=True` (in which case the row is
    inserted regardless — useful for fresh demo containers that wipe the
    DB on every boot).
    """
    inserted = 0
    for spec in _SEED_PATIENTS:
        if not force:
            existing = reports_db.list_reports_for_study(spec["study_id"], db_path=db_path)
            if existing:
                continue

        ga = float(spec["ga_weeks"])
        hc = float(spec["hc_mm"])
        ga_str = _ga_str_from_weeks(ga)
        trim = _trimester(ga)

        reports_db.create_report(
            study_id=spec["study_id"],
            finding_id=None,
            patient_name=spec["patient_name"],
            study_date=spec["study_date"],
            model=spec["model"],
            hc_mm=round(hc, 1),
            ga_str=ga_str,
            ga_weeks=round(ga, 2),
            trimester=trim,
            reliability=0.86 if spec.get("abnormal") else 0.94,
            confidence_label="MODERATE CONFIDENCE" if spec.get("abnormal") else "HIGH CONFIDENCE",
            pixel_spacing_mm=0.154,
            elapsed_ms=420.0,
            narrative_p1=(
                f"AI segmentation produced an HC of {hc:.1f} mm, corresponding to "
                f"{ga_str} by Hadlock 1984. " + (spec.get("narrative_impression", ""))
            ),
            narrative_p2="",
            narrative_p3=None,
            narrative_impression=spec["narrative_impression"],
            used_llm=False,
            referring_physician="Dr. Demo Reviewer",
            patient_id=spec["patient_id"],
            patient_dob=spec["patient_dob"],
            lmp=spec.get("lmp"),
            ordering_facility="FetalScan AI Demo Hospital",
            sonographer_name="Demo Sonographer",
            clinical_indication=spec["indication"],
            us_approach=spec["us_approach"],
            image_quality=spec["image_quality"],
            pixel_spacing_dicom_derived=False,
            pixel_spacing_source="CSV",
            report_mode="template",
            fetal_presentation=spec.get("fetal_presentation"),
            db_path=db_path,
        )
        inserted += 1
    return inserted
