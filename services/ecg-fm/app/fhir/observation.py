"""FHIR R4 Observation builder for ECG-FM outputs.

ECG-FM produces:
1. **Arrhythmia classification** — multi-label probabilities across ~30 SNOMED CT rhythm codes
2. **Interval measurements** — PR, QRS, QT, QTc, RR intervals in ms
3. **STEMI detection** — boolean + anatomical localization (anterior/inferior/lateral)

We map these into FHIR R4 Observation resources with appropriate LOINC codes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"
UNITS_SYSTEM = "http://unitsofmeasure.org"

# Interval measurements (LOINC codes)
INTERVAL_LOINC: Dict[str, Dict[str, Any]] = {
    "pr_interval":   {"code": "44967-8", "display": "PR interval", "unit": "ms"},
    "qrs_duration":  {"code": "44577-8", "display": "QRS duration", "unit": "ms"},
    "qt_interval":   {"code": "44996-7", "display": "QT interval", "unit": "ms"},
    "qtc_interval":  {"code": "44978-5", "display": "QTc interval", "unit": "ms"},
    "rr_interval":   {"code": "44980-1", "display": "RR interval", "unit": "ms"},
    "heart_rate":    {"code": "8867-4",  "display": "Heart rate", "unit": "/min"},
}

# Common arrhythmia SNOMED codes (subset — extend per institution)
ARRHYTHMIA_SNOMED: Dict[str, Dict[str, Any]] = {
    "sinus_rhythm":          {"code": "426783004", "display": "Sinus rhythm"},
    "sinus_tachycardia":     {"code": "427073009", "display": "Sinus tachycardia"},
    "sinus_bradycardia":     {"code": "426996005", "display": "Sinus bradycardia"},
    "sinus_arrhythmia":      {"code": "426784000", "display": "Sinus arrhythmia"},
    "afib":                  {"code": "164873001", "display": "Atrial fibrillation"},
    "aflutter":              {"code": "164864005", "display": "Atrial flutter"},
    "svt":                   {"code": "61721007",  "display": "Supraventricular tachycardia"},
    "av_block_1":            {"code": "233917008", "display": "First-degree AV block"},
    "av_block_2_mobitz1":    {"code": "17434006",  "display": "Second-degree AV block (Wenckebach)"},
    "av_block_2_mobitz2":    {"code": "174341005", "display": "Second-degree AV block (Mobitz II)"},
    "av_block_3":            {"code": "27885002",  "display": "Complete AV block"},
    "rbbb":                  {"code": "251115004", "display": "Right bundle branch block"},
    "lbbb":                  {"code": "251120005", "display": "Left bundle branch block"},
    "pac":                   {"code": "61119008",  "display": "Premature atrial contraction"},
    "pvc":                   {"code": "416456002", "display": "Premature ventricular contraction"},
    "vt":                    {"code": "71908006",  "display": "Ventricular tachycardia"},
    "vfib":                  {"code": "426788005", "display": "Ventricular fibrillation"},
    "stemi_anterior":        {"code": "401303003", "display": "Anterior ST-elevation myocardial infarction"},
    "stemi_inferior":        {"code": "401301005", "display": "Inferior ST-elevation myocardial infarction"},
    "stemi_lateral":         {"code": "401302003", "display": "Lateral ST-elevation myocardial infarction"},
    "nstemi":                {"code": "399266000", "display": "Non-ST elevation myocardial infarction"},
}


def build_interval_observation(
    interval_key: str,
    value_ms: float,
    patient_id: str = "unknown",
    encounter_id: Optional[str] = None,
    performer: str = "ECG-FM AI",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 Observation for an ECG interval measurement."""
    if interval_key not in INTERVAL_LOINC:
        raise KeyError(f"Unknown interval: {interval_key}")
    meta = INTERVAL_LOINC[interval_key]
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()

    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": f"obs-{interval_key}",
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "procedure",
                "display": "Procedure",
            }]
        }],
        "code": {
            "coding": [{"system": LOINC_SYSTEM, "code": meta["code"], "display": meta["display"]}],
            "text": meta["display"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": performer}],
        "valueQuantity": {
            "value": round(float(value_ms), 1),
            "unit": meta["unit"],
            "system": UNITS_SYSTEM,
            "code": meta["unit"],
        },
        "note": [{"text": "AI-drafted by ECG-FM; verify with cardiologist."}],
    }
    if encounter_id:
        obs["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    return obs


def build_arrhythmia_observation(
    arrhythmia_key: str,
    probability: float,
    predicted: bool,
    patient_id: str = "unknown",
    encounter_id: Optional[str] = None,
    performer: str = "ECG-FM AI",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 Observation for an arrhythmia finding."""
    if arrhythmia_key not in ARRHYTHMIA_SNOMED:
        raise KeyError(f"Unknown arrhythmia: {arrhythmia_key}")
    meta = ARRHYTHMIA_SNOMED[arrhythmia_key]
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()

    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": f"obs-arr-{arrhythmia_key}",
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "procedure",
                "display": "Procedure",
            }]
        }],
        "code": {
            "coding": [{
                "system": SNOMED_SYSTEM,
                "code": "439401001",
                "display": "Determination of cardiac rhythm by ECG",
            }],
            "text": "Cardiac rhythm (AI)",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": performer}],
        "valueCodeableConcept": {
            "coding": [{
                "system": SNOMED_SYSTEM,
                "code": meta["code"],
                "display": meta["display"],
            }],
            "text": meta["display"],
        },
        "interpretation": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": "POS" if predicted else "NEG",
            }],
            "text": f"{'Predicted' if predicted else 'Not predicted'} (p={probability:.2%})",
        }],
        "note": [{"text": "AI-drafted by ECG-FM; verify with cardiologist."}],
    }
    if encounter_id:
        obs["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    return obs


def build_ecg_diagnostic_report(
    intervals: Dict[str, float],
    arrhythmia_predictions: Dict[str, float],
    threshold: float = 0.5,
    patient_id: str = "unknown",
    encounter_id: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    performer: str = "ECG-FM AI",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 DiagnosticReport for an ECG-FM interpretation.

    Args:
        intervals: dict of interval_key -> value_ms (e.g. {"pr_interval": 160.0, ...})
        arrhythmia_predictions: dict of arrhythmia_key -> probability
        threshold: probability threshold for "predicted" flag
        patient_id, encounter_id, study_instance_uid: FHIR/DICOM references

    Returns:
        FHIR R4 DiagnosticReport with contained Observations.
    """
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()

    contained: List[Dict[str, Any]] = []
    result_refs: List[Dict[str, str]] = []

    for key, val in intervals.items():
        if key in INTERVAL_LOINC:
            contained.append(build_interval_observation(
                key, val, patient_id, encounter_id, performer, timestamp
            ))
            result_refs.append({"reference": f"#obs-{key}"})

    # Pick the most likely arrhythmia as the conclusion
    critical_arrhythmias = []
    for key, prob in arrhythmia_predictions.items():
        if key in ARRHYTHMIA_SNOMED:
            predicted = prob >= threshold
            contained.append(build_arrhythmia_observation(
                key, prob, predicted, patient_id, encounter_id, performer, timestamp
            ))
            result_refs.append({"reference": f"#obs-arr-{key}"})
            if predicted:
                critical_arrhythmias.append((key, prob, ARRHYTHMIA_SNOMED[key]["display"]))

    # Conclusion text
    if not critical_arrhythmias:
        conclusion = "Sinus rhythm (AI assessment). No significant arrhythmia detected."
    else:
        top = sorted(critical_arrhythmias, key=lambda x: -x[1])[0]
        conclusion = f"AI assessment: {top[2]} (p={top[1]:.2%}). Verify with cardiologist."

    report: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "EC",
                "display": "Electrocardiac",
            }]
        }],
        "code": {
            "coding": [{
                "system": LOINC_SYSTEM,
                "code": "11524-6",
                "display": "EKG study",
            }],
            "text": "12-lead ECG (AI-interpreted)",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": performer}],
        "resultsInterpreter": [{"display": f"{performer} (preliminary draft)"}],
        "conclusion": conclusion,
        "conclusionCode": [
            {
                "coding": [{
                    "system": SNOMED_SYSTEM,
                    "code": ARRHYTHMIA_SNOMED[k]["code"],
                    "display": ARRHYTHMIA_SNOMED[k]["display"],
                }]
            }
            for k, _, _ in critical_arrhythmias[:3]  # top 3 in conclusion
        ],
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

    return report
