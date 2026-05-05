"""SQLite persistence for clinical reports and the audit log.

The schema is intentionally narrow:

  reports     — one row per Report. Stores the snapshot of biometric values
                used to render the PDF, plus the LLM-authored narrative
                paragraphs (frozen at create time so re-rendering the PDF is
                deterministic and free of new LLM calls). Sign-off mutates
                is_signed / signed_by / signed_at / signoff_note in place.

  audit_log   — append-only history of actions taken on a report (created,
                viewed, signed). Captures actor name + IP + user-agent so a
                downstream audit query can reconstruct who did what when.

Reports are keyed by UUID and grouped by an arbitrary study_id string. We
intentionally do not enforce a foreign key on study_id — studies live in the
front-end's worklist (demo-001, demo-002, … or upload-derived IDs) and are
not yet first-class server-side resources.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

DB_PATH = os.environ.get("REPORTS_DB_PATH", "reports.db")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id                       TEXT PRIMARY KEY,
    study_id                 TEXT NOT NULL,
    finding_id               TEXT,
    patient_name             TEXT NOT NULL,
    study_date               TEXT NOT NULL,
    model                    TEXT NOT NULL,
    hc_mm                    REAL,
    ga_str                   TEXT,
    ga_weeks                 REAL,
    trimester                TEXT,
    reliability              REAL,
    confidence_label         TEXT,
    pixel_spacing_mm         REAL,
    elapsed_ms               REAL,
    narrative_p1             TEXT,
    narrative_p2             TEXT,
    narrative_p3             TEXT,
    narrative_impression     TEXT,
    used_llm                 INTEGER NOT NULL DEFAULT 0,
    is_signed                INTEGER NOT NULL DEFAULT 0,
    signed_by                TEXT,
    signed_at                TEXT,
    signoff_note             TEXT,
    created_at               TEXT NOT NULL,
    referring_physician      TEXT,
    patient_id               TEXT,
    patient_dob              TEXT,
    lmp                      TEXT,
    ordering_facility        TEXT,
    sonographer_name         TEXT,
    clinical_indication      TEXT,
    us_approach              TEXT,
    image_quality            TEXT,
    pixel_spacing_dicom_derived INTEGER NOT NULL DEFAULT 0,
    pixel_spacing_source     TEXT,
    report_mode              TEXT NOT NULL DEFAULT 'template',
    accession_number         TEXT,
    original_image_b64       TEXT,
    overlay_image_b64        TEXT,
    gradcam_image_b64        TEXT,
    fetal_presentation       TEXT,
    bpd_mm                   REAL,
    prior_biometry           TEXT,
    is_combined              INTEGER NOT NULL DEFAULT 0,
    combined_models_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_reports_study   ON reports(study_id);
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    report_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    actor       TEXT,
    ip          TEXT,
    user_agent  TEXT,
    details     TEXT,
    timestamp   TEXT NOT NULL,
    FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audit_report ON audit_log(report_id);
"""


_MIGRATION_COLUMNS = [
    "narrative_impression     TEXT",
    "referring_physician      TEXT",
    "patient_id               TEXT",
    "patient_dob              TEXT",
    "lmp                      TEXT",
    "ordering_facility        TEXT",
    "sonographer_name         TEXT",
    "clinical_indication      TEXT",
    "us_approach              TEXT",
    "image_quality            TEXT",
    "pixel_spacing_dicom_derived INTEGER NOT NULL DEFAULT 0",
    "pixel_spacing_source     TEXT",
    "report_mode              TEXT NOT NULL DEFAULT 'template'",
    "accession_number         TEXT",
    "original_image_b64       TEXT",
    "overlay_image_b64        TEXT",
    "gradcam_image_b64        TEXT",
    "fetal_presentation       TEXT",
    "bpd_mm                   REAL",
    "prior_biometry           TEXT",
    "is_combined              INTEGER NOT NULL DEFAULT 0",
    "combined_models_json     TEXT",
]


@dataclass
class Report:
    id: str
    study_id: str
    finding_id: str | None
    patient_name: str
    study_date: str
    model: str
    hc_mm: float | None
    ga_str: str | None
    ga_weeks: float | None
    trimester: str | None
    reliability: float | None
    confidence_label: str | None
    pixel_spacing_mm: float | None
    elapsed_ms: float | None
    narrative_p1: str | None
    narrative_p2: str | None
    narrative_p3: str | None
    narrative_impression: str | None
    used_llm: bool
    is_signed: bool
    signed_by: str | None
    signed_at: str | None
    signoff_note: str | None
    created_at: str
    referring_physician: str | None = None
    patient_id: str | None = None
    patient_dob: str | None = None
    lmp: str | None = None
    ordering_facility: str | None = None
    sonographer_name: str | None = None
    clinical_indication: str | None = None
    us_approach: str | None = None
    image_quality: str | None = None
    pixel_spacing_dicom_derived: bool = False
    pixel_spacing_source: str | None = None
    report_mode: str = "template"
    accession_number: str | None = None
    original_image_b64: str | None = None
    overlay_image_b64: str | None = None
    gradcam_image_b64: str | None = None
    fetal_presentation: str | None = None
    bpd_mm: float | None = None
    prior_biometry: str | None = None
    is_combined: bool = False
    combined_models_json: str | None = None  # JSON-serialised list of per-model dicts

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "study_id": self.study_id,
            "finding_id": self.finding_id,
            "patient_name": self.patient_name,
            "study_date": self.study_date,
            "model": self.model,
            "hc_mm": self.hc_mm,
            "ga_str": self.ga_str,
            "ga_weeks": self.ga_weeks,
            "trimester": self.trimester,
            "reliability": self.reliability,
            "confidence_label": self.confidence_label,
            "pixel_spacing_mm": self.pixel_spacing_mm,
            "elapsed_ms": self.elapsed_ms,
            "narrative_p1": self.narrative_p1,
            "narrative_p2": self.narrative_p2,
            "narrative_p3": self.narrative_p3,
            "narrative_impression": self.narrative_impression,
            "used_llm": self.used_llm,
            "is_signed": self.is_signed,
            "signed_by": self.signed_by,
            "signed_at": self.signed_at,
            "signoff_note": self.signoff_note,
            "created_at": self.created_at,
            "referring_physician": self.referring_physician,
            "patient_id": self.patient_id,
            "patient_dob": self.patient_dob,
            "lmp": self.lmp,
            "ordering_facility": self.ordering_facility,
            "sonographer_name": self.sonographer_name,
            "clinical_indication": self.clinical_indication,
            "us_approach": self.us_approach,
            "image_quality": self.image_quality,
            "pixel_spacing_dicom_derived": self.pixel_spacing_dicom_derived,
            "pixel_spacing_source": self.pixel_spacing_source,
            "report_mode": self.report_mode,
            "accession_number": self.accession_number,
            "original_image_b64": self.original_image_b64,
            "overlay_image_b64": self.overlay_image_b64,
            "gradcam_image_b64": self.gradcam_image_b64,
            "fetal_presentation": self.fetal_presentation,
            "bpd_mm": self.bpd_mm,
            "prior_biometry": self.prior_biometry,
            "is_combined": self.is_combined,
            "combined_models_json": self.combined_models_json,
        }


@dataclass
class AuditEntry:
    id: str
    report_id: str
    action: str
    actor: str | None
    ip: str | None
    user_agent: str | None
    details: str | None
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "report_id": self.report_id,
            "action": self.action,
            "actor": self.actor,
            "ip": self.ip,
            "user_agent": self.user_agent,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@contextmanager
def _conn(db_path: str | None = None):
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with _conn(db_path) as c:
        c.executescript(_SCHEMA)
        # Idempotent migration: add any columns that didn't exist in older DBs.
        for col_def in _MIGRATION_COLUMNS:
            try:
                c.execute(f"ALTER TABLE reports ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists


def _now() -> str:
    # ISO 8601 with milliseconds in UTC
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row_get(row: sqlite3.Row, key: str, default=None):
    """Safe column access — returns default for columns not yet migrated."""
    try:
        return row[key]
    except IndexError:
        return default


def _row_to_report(row: sqlite3.Row) -> Report:
    return Report(
        id=row["id"],
        study_id=row["study_id"],
        finding_id=row["finding_id"],
        patient_name=row["patient_name"],
        study_date=row["study_date"],
        model=row["model"],
        hc_mm=row["hc_mm"],
        ga_str=row["ga_str"],
        ga_weeks=row["ga_weeks"],
        trimester=row["trimester"],
        reliability=row["reliability"],
        confidence_label=row["confidence_label"],
        pixel_spacing_mm=row["pixel_spacing_mm"],
        elapsed_ms=row["elapsed_ms"],
        narrative_p1=row["narrative_p1"],
        narrative_p2=row["narrative_p2"],
        narrative_p3=row["narrative_p3"],
        narrative_impression=_row_get(row, "narrative_impression"),
        used_llm=bool(row["used_llm"]),
        is_signed=bool(row["is_signed"]),
        signed_by=row["signed_by"],
        signed_at=row["signed_at"],
        signoff_note=row["signoff_note"],
        created_at=row["created_at"],
        referring_physician=_row_get(row, "referring_physician"),
        patient_id=_row_get(row, "patient_id"),
        patient_dob=_row_get(row, "patient_dob"),
        lmp=_row_get(row, "lmp"),
        ordering_facility=_row_get(row, "ordering_facility"),
        sonographer_name=_row_get(row, "sonographer_name"),
        clinical_indication=_row_get(row, "clinical_indication"),
        us_approach=_row_get(row, "us_approach"),
        image_quality=_row_get(row, "image_quality"),
        pixel_spacing_dicom_derived=bool(_row_get(row, "pixel_spacing_dicom_derived", 0)),
        pixel_spacing_source=_row_get(row, "pixel_spacing_source"),
        report_mode=_row_get(row, "report_mode", "template") or "template",
        accession_number=_row_get(row, "accession_number"),
        original_image_b64=_row_get(row, "original_image_b64"),
        overlay_image_b64=_row_get(row, "overlay_image_b64"),
        gradcam_image_b64=_row_get(row, "gradcam_image_b64"),
        fetal_presentation=_row_get(row, "fetal_presentation"),
        bpd_mm=_row_get(row, "bpd_mm"),
        prior_biometry=_row_get(row, "prior_biometry"),
        is_combined=bool(_row_get(row, "is_combined", 0)),
        combined_models_json=_row_get(row, "combined_models_json"),
    )


def _row_to_audit(row: sqlite3.Row) -> AuditEntry:
    return AuditEntry(
        id=row["id"],
        report_id=row["report_id"],
        action=row["action"],
        actor=row["actor"],
        ip=row["ip"],
        user_agent=row["user_agent"],
        details=row["details"],
        timestamp=row["timestamp"],
    )


def _make_accession(created: str) -> str:
    """Generate FHC-YYYYMMDD-HHMMSS accession number from ISO timestamp."""
    digits = created.replace("-", "").replace("T", "-").replace(":", "").replace("Z", "")
    # digits = "YYYYMMDD-HHMMSS"
    return f"FHC-{digits[:8]}-{digits[9:15]}"


def create_report(
    *,
    study_id: str,
    finding_id: str | None,
    patient_name: str,
    study_date: str,
    model: str,
    hc_mm: float | None,
    ga_str: str | None,
    ga_weeks: float | None,
    trimester: str | None,
    reliability: float | None,
    confidence_label: str | None,
    pixel_spacing_mm: float | None,
    elapsed_ms: float | None,
    narrative_p1: str | None,
    narrative_p2: str | None,
    narrative_p3: str | None,
    narrative_impression: str | None,
    used_llm: bool,
    referring_physician: str | None = None,
    patient_id: str | None = None,
    patient_dob: str | None = None,
    lmp: str | None = None,
    ordering_facility: str | None = None,
    sonographer_name: str | None = None,
    clinical_indication: str | None = None,
    us_approach: str | None = None,
    image_quality: str | None = None,
    pixel_spacing_dicom_derived: bool = False,
    pixel_spacing_source: str | None = None,
    report_mode: str = "template",
    original_image_b64: str | None = None,
    overlay_image_b64: str | None = None,
    gradcam_image_b64: str | None = None,
    fetal_presentation: str | None = None,
    bpd_mm: float | None = None,
    prior_biometry: str | None = None,
    is_combined: bool = False,
    combined_models_json: str | None = None,
    db_path: str | None = None,
) -> Report:
    rid = f"rep_{uuid.uuid4().hex[:16]}"
    created = _now()
    accession = _make_accession(created)
    with _conn(db_path) as c:
        c.execute(
            """
            INSERT INTO reports (
                id, study_id, finding_id, patient_name, study_date, model,
                hc_mm, ga_str, ga_weeks, trimester, reliability,
                confidence_label, pixel_spacing_mm, elapsed_ms,
                narrative_p1, narrative_p2, narrative_p3, narrative_impression,
                used_llm, is_signed, created_at,
                referring_physician, patient_id, patient_dob, lmp,
                ordering_facility, sonographer_name, clinical_indication,
                us_approach, image_quality,
                pixel_spacing_dicom_derived, pixel_spacing_source,
                report_mode, accession_number,
                original_image_b64, overlay_image_b64, gradcam_image_b64,
                fetal_presentation, bpd_mm, prior_biometry,
                is_combined, combined_models_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                rid,
                study_id,
                finding_id,
                patient_name,
                study_date,
                model,
                hc_mm,
                ga_str,
                ga_weeks,
                trimester,
                reliability,
                confidence_label,
                pixel_spacing_mm,
                elapsed_ms,
                narrative_p1,
                narrative_p2,
                narrative_p3,
                narrative_impression,
                int(used_llm),
                created,
                referring_physician,
                patient_id,
                patient_dob,
                lmp,
                ordering_facility,
                sonographer_name,
                clinical_indication,
                us_approach,
                image_quality,
                int(pixel_spacing_dicom_derived),
                pixel_spacing_source,
                report_mode,
                accession,
                original_image_b64,
                overlay_image_b64,
                gradcam_image_b64,
                fetal_presentation,
                bpd_mm,
                prior_biometry,
                int(is_combined),
                combined_models_json,
            ),
        )
    return get_report(rid, db_path=db_path)  # type: ignore[return-value]


def get_report(report_id: str, db_path: str | None = None) -> Report | None:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return _row_to_report(row) if row else None


def list_reports_for_study(study_id: str, db_path: str | None = None) -> list[Report]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM reports WHERE study_id = ? ORDER BY created_at DESC",
            (study_id,),
        ).fetchall()
    return [_row_to_report(r) for r in rows]


def list_all_reports(db_path: str | None = None) -> list[Report]:
    with _conn(db_path) as c:
        rows = c.execute("SELECT * FROM reports ORDER BY created_at DESC").fetchall()
    return [_row_to_report(r) for r in rows]


def sign_report(
    report_id: str,
    signed_by: str,
    signoff_note: str | None,
    db_path: str | None = None,
) -> Report | None:
    signed_at = _now()
    with _conn(db_path) as c:
        cur = c.execute(
            """
            UPDATE reports
            SET is_signed = 1, signed_by = ?, signed_at = ?, signoff_note = ?
            WHERE id = ? AND is_signed = 0
            """,
            (signed_by, signed_at, signoff_note, report_id),
        )
        if cur.rowcount == 0:
            return None
    return get_report(report_id, db_path=db_path)


def add_audit(
    *,
    report_id: str,
    action: str,
    actor: str | None,
    ip: str | None,
    user_agent: str | None,
    details: str | None = None,
    db_path: str | None = None,
) -> AuditEntry:
    aid = f"aud_{uuid.uuid4().hex[:16]}"
    ts = _now()
    with _conn(db_path) as c:
        c.execute(
            """
            INSERT INTO audit_log
                (id, report_id, action, actor, ip, user_agent, details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (aid, report_id, action, actor, ip, user_agent, details, ts),
        )
    return AuditEntry(
        id=aid,
        report_id=report_id,
        action=action,
        actor=actor,
        ip=ip,
        user_agent=user_agent,
        details=details,
        timestamp=ts,
    )


def list_audit_for_report(report_id: str, db_path: str | None = None) -> list[AuditEntry]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM audit_log WHERE report_id = ? ORDER BY timestamp ASC",
            (report_id,),
        ).fetchall()
    return [_row_to_audit(r) for r in rows]


def clear_all(db_path: str | None = None) -> None:
    """Test-only: wipe both tables. No-op if the DB doesn't exist."""
    with _conn(db_path) as c:
        c.execute("DELETE FROM audit_log")
        c.execute("DELETE FROM reports")
