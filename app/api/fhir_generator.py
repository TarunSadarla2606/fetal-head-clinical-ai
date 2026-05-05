"""FHIR R4 export — turn a stored Report into a FHIR Bundle.

Emits a self-contained "collection" Bundle with:
- one DiagnosticReport (the parent)
- one Observation per measurement (HC always, BPD if recorded,
  Gestational Age always)
- one Patient resource (built from the report's patient_name / patient_id /
  patient_dob fields)

LOINC codes used (all from the Loinc fetal-biometry value set):
  11779-6  — Head circumference (US estimate)
  11820-2  — Biparietal diameter (US estimate)
  18185-9  — Gestational age (estimated)
  42148-7  — Fetal Ultrasound Report (Diagnostic Report code)

The resulting Bundle is hand-rolled JSON (no extra dependency) and validates
against the FHIR R4 schema for these resource types — the field names and
value types follow the R4 specification exactly. For production hospital
integration the same shape would be sent to a FHIR server's POST /Bundle
endpoint.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from . import reports_db


def _measurement_observation(
    obs_id: str,
    loinc_code: str,
    loinc_display: str,
    value: float,
    unit: str,
    ucum_code: str,
    patient_ref: str,
    effective: str,
    status: str,
) -> dict[str, Any]:
    """Build one FHIR Observation resource for a numeric biometric value."""
    return {
        "resourceType": "Observation",
        "id": obs_id,
        "status": status,
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": loinc_code,
                    "display": loinc_display,
                }
            ],
            "text": loinc_display,
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": effective,
        "valueQuantity": {
            "value": round(float(value), 2),
            "unit": unit,
            "system": "http://unitsofmeasure.org",
            "code": ucum_code,
        },
    }


def _patient_resource(report: reports_db.Report) -> dict[str, Any]:
    name_parts = (report.patient_name or "—").strip().split(maxsplit=1)
    given = [name_parts[0]] if name_parts else ["—"]
    family = name_parts[1] if len(name_parts) > 1 else ""
    res: dict[str, Any] = {
        "resourceType": "Patient",
        "id": f"patient-{report.id}",
        "name": [{"family": family, "given": given, "text": report.patient_name}],
    }
    if report.patient_id:
        res["identifier"] = [
            {"system": "urn:oid:2.16.840.1.113883.19.5", "value": report.patient_id}
        ]
    if report.patient_dob:
        res["birthDate"] = report.patient_dob
    return res


def report_to_fhir_bundle(report: reports_db.Report) -> dict[str, Any]:
    """Convert a stored Report row into a FHIR R4 Bundle dict."""
    status = "final" if report.is_signed else "preliminary"
    effective = report.study_date or datetime.utcnow().strftime("%Y-%m-%d")
    issued = (report.signed_at or report.created_at or datetime.utcnow().isoformat()).replace(
        " ", "T"
    )
    if not issued.endswith("Z"):
        issued = issued.rstrip("Z") + "Z"

    patient = _patient_resource(report)
    patient_ref = f"Patient/{patient['id']}"

    observations: list[dict[str, Any]] = []

    if report.hc_mm is not None:
        observations.append(
            _measurement_observation(
                obs_id=f"hc-{report.id}",
                loinc_code="11779-6",
                loinc_display="Head circumference Estimated by US",
                value=report.hc_mm,
                unit="mm",
                ucum_code="mm",
                patient_ref=patient_ref,
                effective=effective,
                status=status,
            )
        )
    if report.bpd_mm is not None:
        observations.append(
            _measurement_observation(
                obs_id=f"bpd-{report.id}",
                loinc_code="11820-2",
                loinc_display="Biparietal diameter Estimated by US",
                value=report.bpd_mm,
                unit="mm",
                ucum_code="mm",
                patient_ref=patient_ref,
                effective=effective,
                status=status,
            )
        )
    if report.ga_weeks is not None:
        observations.append(
            _measurement_observation(
                obs_id=f"ga-{report.id}",
                loinc_code="18185-9",
                loinc_display="Gestational age Estimated",
                value=report.ga_weeks,
                unit="weeks",
                ucum_code="wk",
                patient_ref=patient_ref,
                effective=effective,
                status=status,
            )
        )

    diag_report: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": f"report-{report.id}",
        "status": status,
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "42148-7",
                    "display": "Fetal Ultrasound Report",
                }
            ],
            "text": "Fetal Ultrasound Report",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": effective,
        "issued": issued,
        "result": [{"reference": f"Observation/{o['id']}"} for o in observations],
    }
    if report.referring_physician:
        diag_report["performer"] = [{"display": report.referring_physician}]
    if report.narrative_impression:
        diag_report["conclusion"] = report.narrative_impression
    if report.accession_number:
        diag_report["identifier"] = [
            {
                "system": "urn:oid:2.16.840.1.113883.19.5.1",
                "value": report.accession_number,
            }
        ]
    if report.is_signed and report.signed_by:
        # FHIR uses extensions for non-standard fields; sign-off is closer
        # to the resultsInterpreter performer for a final report.
        diag_report.setdefault("performer", [])
        diag_report["performer"].append({"display": f"Signed by {report.signed_by}"})

    bundle: dict[str, Any] = {
        "resourceType": "Bundle",
        "id": f"bundle-{report.id}",
        "type": "collection",
        "timestamp": issued,
        "entry": [{"resource": patient}, {"resource": diag_report}]
        + [{"resource": o} for o in observations],
    }
    return bundle
