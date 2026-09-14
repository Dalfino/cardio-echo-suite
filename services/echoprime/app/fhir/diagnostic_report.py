"""FHIR R4 DiagnosticReport builder for EchoPrime outputs.

EchoPrime produces three things:
1. View classification (e.g. A4C, A2C, PLAX, PSAX)
2. Structural measurements (EF, LV dimensions, LA size, etc.)
3. A draft echo report (free text)

We map these into a FHIR R4 DiagnosticReport with:
- `code` = LOINC 26520-0 (Echocardiography study)
- `conclusion` = the draft report text
- `conclusionCode` = view classification codes
- `result` = references to contained Observation resources for measurements
- `media` = link to the source video (DICOM SOP Instance UID)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOINC_ECHO_STUDY = "26520-0"
LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"

# View classification SNOMED CT codes (subset — extend per institution)
VIEW_SNOMED = {
    "A4C":   {"code": "25100000", "display": "Apical 4-chamber view"},
    "A2C":   {"code": "25100008", "display": "Apical 2-chamber view"},
    "A5C":   {"code": "25100009", "display": "Apical 5-chamber view"},
    "PLAX":  {"code": "25100001", "display": "Parasternal long-axis view"},
    "PSAX":  {"code": "25100002", "display": "Parasternal short-axis view"},
    "SAX-AV": {"code": "25100003", "display": "PSAX at aortic valve level"},
    "SAX-PM": {"code": "25100004", "display": "PSAX at papillary muscle level"},
    "SAX-AP": {"code": "25100005", "display": "PSAX at apical level"},
    "SC-4C": {"code": "25100006", "display": "Subcostal 4-chamber view"},
    "SC-IVC": {"code": "25100007", "display": "Subcostal IVC view"},
    "SUPRA": {"code": "25100010", "display": "Suprasternal view"},
}

# Measurement LOINC codes
MEASUREMENT_LOINC = {
    "ef":        {"code": "10230-1", "display": "Ejection fraction"},
    "lvidd":     {"code": "76534-3", "display": "LV internal diameter, diastole"},
    "lvids":     {"code": "76535-0", "display": "LV internal diameter, systole"},
    "ivsd":      {"code": "76536-8", "display": "Interventricular septum, diastole"},
    "pwdd":      {"code": "76537-6", "display": "Posterior wall thickness, diastole"},
    "la_size":   {"code": "76538-4", "display": "Left atrial size"},
    "a_peak_v":  {"code": "76539-2", "display": "Aortic valve peak velocity"},
    "m_peak_v":  {"code": "76540-0", "display": "Mitral valve peak velocity"},
    "e_wave":    {"code": "76541-8", "display": "Mitral E wave velocity"},
    "a_wave":    {"code": "76542-6", "display": "Mitral A wave velocity"},
    "e_e_ratio": {"code": "76543-4", "display": "E/e' ratio"},
    "tr_v":      {"code": "76544-2", "display": "Tricuspid regurgitation velocity"},
}


def _contained_observation(
    measurement_key: str, value: float, unit: str, patient_id: str
) -> Dict[str, Any]:
    """Build a contained Observation for a single measurement."""
    loinc = MEASUREMENT_LOINC[measurement_key]
    return {
        "resourceType": "Observation",
        "id": f"obs-{measurement_key}",
        "status": "preliminary",
        "code": {
            "coding": [{"system": LOINC_SYSTEM, "code": loinc["code"], "display": loinc["display"]}],
            "text": loinc["display"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "performer": [{"display": "EchoPrime AI"}],
        "valueQuantity": {
            "value": round(float(value), 2),
            "unit": unit,
            "system": "http://unitsofmeasure.org",
            "code": unit,
        },
        "note": [{"text": "AI-drafted from EchoPrime; verify with sonographer."}],
    }


def build_diagnostic_report(
    views: List[str],
    measurements: Dict[str, float],
    draft_report: str,
    patient_id: str = "unknown",
    encounter_id: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    performer: str = "EchoPrime AI",
    status: str = "preliminary",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 DiagnosticReport from EchoPrime output.

    Args:
        views: list of view codes (e.g. ["A4C", "A2C", "PLAX"]).
        measurements: dict of measurement_key -> value. Keys must be in
            `MEASUREMENT_LOINC` (e.g. "ef", "lvidd", "e_e_ratio").
        draft_report: free-text AI-drafted report.
        patient_id, encounter_id, study_instance_uid: FHIR/DICOM references.
        performer: who/what issued the report.
        status: "preliminary" (AI draft) by default. Set to "final" once a
            physician signs the report.
        timestamp: observation time (defaults to now).

    Returns:
        FHIR R4 DiagnosticReport dict with contained Observations.
    """
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()

    contained: List[Dict[str, Any]] = []
    result_refs: List[Dict[str, str]] = []

    # Add an EF observation specifically (key measurement)
    for key, value in measurements.items():
        if key not in MEASUREMENT_LOINC:
            continue
        unit = "%" if key in ("ef",) else (
            "cm" if key in ("lvidd", "lvids", "ivsd", "pwdd", "la_size") else (
                "m/s" if key in ("a_peak_v", "m_peak_v", "e_wave", "a_wave", "tr_v") else ""
            )
        )
        contained.append(_contained_observation(key, value, unit, patient_id))
        result_refs.append({"reference": f"#obs-{key}"})

    # View classification codes
    conclusion_codes = []
    for v in views:
        v_upper = v.upper().replace("_", "").replace("-", "").replace(" ", "")
        if v_upper in VIEW_SNOMED:
            conclusion_codes.append({
                "coding": [{"system": SNOMED_SYSTEM, **VIEW_SNOMED[v_upper]}],
                "text": VIEW_SNOMED[v_upper]["display"],
            })
        else:
            conclusion_codes.append({"text": v})

    report: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "status": status,
        "category": [
            {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "RAD",
                    "display": "Radiology",
                }]
            }
        ],
        "code": {
            "coding": [{
                "system": LOINC_SYSTEM,
                "code": LOINC_ECHO_STUDY,
                "display": "Echocardiography study",
            }],
            "text": "Echocardiography (AI-drafted)",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": performer}],
        "resultsInterpreter": [{"display": f"{performer} (preliminary draft)"}],
        "conclusion": draft_report or "AI-drafted echo report (preliminary)",
        "conclusionCode": conclusion_codes,
        "contained": contained,
        "result": result_refs,
    }

    if encounter_id:
        report["encounter"] = {"reference": f"Encounter/{encounter_id}"}

    if study_instance_uid:
        report.setdefault("identifier", []).append({
            "system": "urn:dicom:uid",
            "value": f"urn:oid:{study_instance_uid}",
        })
        report["media"] = [{
            "comment": "Source DICOM study for this AI-drafted report",
            "link": {
                "reference": "ImagingStudy/" + study_instance_uid.split(".")[-1],
                "display": "Source DICOM ImagingStudy",
            },
        }]

    return report
