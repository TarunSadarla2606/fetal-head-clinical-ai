"""DICOM Structured Report (SR) export — Batch 7.

Builds a Comprehensive SR Storage object (SOP class
1.2.840.10008.5.1.4.1.1.88.33) for one stored Report. The content tree
follows the spirit of TID 5000 (Imaging Measurement Report) → TID 1411
(Numeric Measurement) for fetal biometry:

  Container: Imaging Report (LOINC 11526-1)
   ├── NUM: Head Circumference (LOINC 11779-6) = X mm
   ├── NUM: Biparietal Diameter (LOINC 11820-2) = X mm  (optional)
   ├── NUM: Gestational Age (LOINC 18185-9) = X wk
   └── TEXT: Finding (DCM 121071) = narrative impression  (optional)

For fully TID-conformant production output you'd typically use
`highdicom.sr.Container` to enforce the relationship-type and
content-item rules end-to-end. This minimal hand-rolled SR opens cleanly
in DCMTK (`dcmdump`), Weasis, OHIF, and any standard DICOM viewer.

Verifying-observer block populated when `is_signed`, mapping our sign-off
audit fields onto the standard VerifyingObserverSequence.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from . import reports_db

# Comprehensive SR Storage — supports container + numeric measurements
_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.88.33"


def _code_item(code_value: str, scheme: str, meaning: str) -> Dataset:
    """One CodedEntry sub-dataset (CodeValue / CodingSchemeDesignator / CodeMeaning)."""
    item = Dataset()
    item.CodeValue = code_value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return item


def _numeric_measurement(value: float, unit_ucum: str, loinc_code: str, meaning: str) -> Dataset:
    """One NUM content-tree item for a biometric measurement."""
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "NUM"
    item.ConceptNameCodeSequence = [_code_item(loinc_code, "LN", meaning)]

    measured = Dataset()
    measured.NumericValue = str(round(float(value), 2))
    measured.MeasurementUnitsCodeSequence = [_code_item(unit_ucum, "UCUM", unit_ucum)]
    item.MeasuredValueSequence = [measured]
    return item


def _text_item(code_value: str, scheme: str, meaning: str, text: str) -> Dataset:
    """One TEXT content-tree item (used for the conclusion / impression)."""
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = [_code_item(code_value, scheme, meaning)]
    item.TextValue = text
    return item


def _format_dicom_date(iso: str | None) -> str:
    """ISO 'YYYY-MM-DD' (or 'YYYY-MM-DDThh:mm:ssZ') → DICOM 'YYYYMMDD'."""
    if not iso:
        return ""
    return iso.replace("-", "").split("T")[0][:8]


def _format_dicom_datetime(iso: str | None) -> str:
    """ISO 'YYYY-MM-DDThh:mm:ssZ' → DICOM 'YYYYMMDDhhmmss'."""
    if not iso:
        return ""
    cleaned = iso.replace("-", "").replace(":", "").replace("T", "").replace("Z", "")
    return cleaned[:14]


def report_to_dicom_sr(report: reports_db.Report) -> bytes:
    """Serialise a stored Report into a DICOM SR (Comprehensive SR Storage) byte string."""

    # ── File meta ──────────────────────────────────────────────────────────
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = _SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "FetalScanAI_1.0"

    ds = FileDataset(
        filename_or_obj="",
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    # ── Patient module ─────────────────────────────────────────────────────
    ds.PatientName = report.patient_name or "Unknown^Patient"
    ds.PatientID = report.patient_id or "UNKNOWN"
    if report.patient_dob:
        ds.PatientBirthDate = _format_dicom_date(report.patient_dob)
    ds.PatientSex = ""

    # ── Study / Series modules ─────────────────────────────────────────────
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.Modality = "SR"
    # AccessionNumber is VR=SH with a 16-character ceiling; our internal
    # FHC-YYYYMMDD-HHMMSS format is 19 chars, so truncate the seconds.
    ds.AccessionNumber = (report.accession_number or "")[:16]
    ds.StudyDate = _format_dicom_date(report.study_date)
    ds.StudyTime = "120000"
    ds.StudyDescription = "Fetal biometry — AI-assisted"
    ds.SeriesDescription = "AI Structured Report"
    ds.ReferringPhysicianName = report.referring_physician or ""
    ds.Manufacturer = "FetalScan AI"
    ds.ManufacturerModelName = report.model or "phase0"

    # ── SOP Common ─────────────────────────────────────────────────────────
    ds.SOPClassUID = _SOP_CLASS_UID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    # ISO_IR 192 = UTF-8 — covers em-dashes, accented chars, "±" etc that
    # routinely appear in our trimester / narrative strings without
    # round-trip data loss.
    ds.SpecificCharacterSet = "ISO_IR 192"

    # ── SR Document General ────────────────────────────────────────────────
    now = datetime.now(UTC)
    ds.ContentDate = ds.StudyDate or now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.CompletionFlag = "COMPLETE" if report.is_signed else "PARTIAL"
    ds.VerificationFlag = "VERIFIED" if report.is_signed else "UNVERIFIED"
    ds.PreliminaryFlag = "FINAL" if report.is_signed else "PRELIMINARY"

    if report.is_signed and report.signed_by:
        verifier = Dataset()
        verifier.VerifyingObserverName = report.signed_by
        verifier.VerificationDateTime = _format_dicom_datetime(report.signed_at)
        verifier.VerifyingOrganization = "FetalScan AI"
        ds.VerifyingObserverSequence = [verifier]

    # ── SR Document Content (the content tree root) ────────────────────────
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = [_code_item("11526-1", "LN", "Imaging Report")]

    items: list[Dataset] = []
    if report.hc_mm is not None:
        items.append(_numeric_measurement(report.hc_mm, "mm", "11779-6", "Head Circumference"))
    if report.bpd_mm is not None:
        items.append(_numeric_measurement(report.bpd_mm, "mm", "11820-2", "Biparietal Diameter"))
    if report.ga_weeks is not None:
        items.append(_numeric_measurement(report.ga_weeks, "wk", "18185-9", "Gestational Age"))
    if report.narrative_impression:
        items.append(_text_item("121071", "DCM", "Finding", report.narrative_impression))
    ds.ContentSequence = items

    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()
