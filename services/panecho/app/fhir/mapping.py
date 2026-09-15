"""PanEcho -> FHIR R4 mapping.

PanEcho (CarDS-Yale, JAMA 2025) outputs predictions for 39 echocardiography
reporting tasks. Each task is either:

- a **classification** (e.g. "AVStenosis": [none, mild, moderate, severe])
- a **regression** (e.g. "EF": scalar in [0, 100])

This module maps every task to a FHIR R4 Observation with the appropriate
LOINC code (where available) and unit. The full list of tasks is at
upstream/content/tasks.md.

Reference task list (canonical names from PanEcho):
- EF (regression)
- LVSystolicFunction, LVHypertrophy, LVRemodeling (classification)
- RVSystolicFunction, RVSize, RVDilated (classification)
- LASize, LADilated (classification)
- AVStenosis, AVRegurgitation (classification)
- MVStenosis, MVRegurgitation, MVProlapse (classification)
- TVRegurgitation (classification)
- PVRegurgitation (classification)
- ASD, VSD, PDA, PericardialEffusion, TamponadePhysiology (classification)
- AorticRootDilation, AorticDissection (classification)
- LVThrombus, RVThrombus (classification)
- VegetationAV, VegetationMV, VegetationTV (classification)
- DiastolicFunction, DiastolicGrade (classification)
- LAPressure (regression)
- TRVelocity (regression)
- RVSP (regression)
- LVIDd, LVIDs, IVSd, PWd (regression, cm)
- LAVolume (regression, mL)
- SV (regression, mL)
- CO (regression, L/min)
- AVArea (regression, cm^2)
- MVArea (regression, cm^2)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"
UNITS_SYSTEM = "http://unitsofmeasure.org"

# ---------------------------------------------------------------
# Task -> LOINC/SNOMED mapping
# ---------------------------------------------------------------

# Regression tasks (numerical, with units)
REGRESSION_TASKS: Dict[str, Dict[str, Any]] = {
    "EF":         {"loinc": "10230-1", "display": "Ejection fraction", "unit": "%"},
    "LAPressure": {"loinc": "76545-9", "display": "Left atrial pressure (estimated)", "unit": "mmHg"},
    "TRVelocity": {"loinc": "76544-2", "display": "Tricuspid regurgitation velocity", "unit": "m/s"},
    "RVSP":       {"loinc": "76546-7", "display": "Right ventricular systolic pressure", "unit": "mmHg"},
    "LVIDd":      {"loinc": "76534-3", "display": "LV internal diameter, diastole", "unit": "cm"},
    "LVIDs":      {"loinc": "76535-0", "display": "LV internal diameter, systole", "unit": "cm"},
    "IVSd":       {"loinc": "76536-8", "display": "Interventricular septum, diastole", "unit": "cm"},
    "PWd":        {"loinc": "76537-6", "display": "Posterior wall thickness, diastole", "unit": "cm"},
    "LAVolume":   {"loinc": "76538-4", "display": "Left atrial volume", "unit": "mL"},
    "SV":         {"loinc": "76547-5", "display": "Stroke volume", "unit": "mL"},
    "CO":         {"loinc": "76548-3", "display": "Cardiac output", "unit": "L/min"},
    "AVArea":     {"loinc": "76549-1", "display": "Aortic valve area", "unit": "cm^2"},
    "MVArea":     {"loinc": "76550-9", "display": "Mitral valve area", "unit": "cm^2"},
}

# Classification tasks (categorical, with class -> SNOMED code)
CLASSIFICATION_TASKS: Dict[str, Dict[str, Any]] = {
    "LVSystolicFunction":    {"display": "LV systolic function",
                              "classes": ["normal", "mildly_reduced", "moderately_reduced", "severely_reduced"]},
    "LVHypertrophy":         {"display": "LV hypertrophy", "classes": ["none", "mild", "moderate", "severe"]},
    "LVRemodeling":          {"display": "LV remodeling", "classes": ["none", "concentric", "eccentric", "mixed"]},
    "RVSystolicFunction":    {"display": "RV systolic function", "classes": ["normal", "mildly_reduced", "reduced"]},
    "RVSize":                {"display": "RV size", "classes": ["normal", "mildly_dilated", "dilated"]},
    "RVDilated":             {"display": "RV dilated", "classes": ["no", "mild", "moderate", "severe"]},
    "LASize":                {"display": "LA size", "classes": ["normal", "mildly_dilated", "dilated", "severely_dilated"]},
    "LADilated":             {"display": "LA dilated", "classes": ["no", "yes"]},
    "AVStenosis":            {"display": "Aortic valve stenosis", "classes": ["none", "mild", "moderate", "severe"]},
    "AVRegurgitation":       {"display": "Aortic valve regurgitation", "classes": ["none", "mild", "moderate", "severe"]},
    "MVStenosis":            {"display": "Mitral valve stenosis", "classes": ["none", "mild", "moderate", "severe"]},
    "MVRegurgitation":       {"display": "Mitral valve regurgitation", "classes": ["none", "mild", "moderate", "severe"]},
    "MVProlapse":            {"display": "Mitral valve prolapse", "classes": ["no", "yes"]},
    "TVRegurgitation":       {"display": "Tricuspid valve regurgitation", "classes": ["none", "mild", "moderate", "severe"]},
    "PVRegurgitation":       {"display": "Pulmonary valve regurgitation", "classes": ["none", "mild", "moderate", "severe"]},
    "ASD":                   {"display": "Atrial septal defect", "classes": ["no", "yes"]},
    "VSD":                   {"display": "Ventricular septal defect", "classes": ["no", "yes"]},
    "PDA":                   {"display": "Patent ductus arteriosus", "classes": ["no", "yes"]},
    "PericardialEffusion":   {"display": "Pericardial effusion", "classes": ["none", "small", "moderate", "large"]},
    "TamponadePhysiology":   {"display": "Tamponade physiology", "classes": ["no", "yes"]},
    "AorticRootDilation":    {"display": "Aortic root dilation", "classes": ["no", "mild", "moderate", "severe"]},
    "AorticDissection":      {"display": "Aortic dissection", "classes": ["no", "yes"]},
    "LVThrombus":            {"display": "LV thrombus", "classes": ["no", "yes"]},
    "RVThrombus":            {"display": "RV thrombus", "classes": ["no", "yes"]},
    "VegetationAV":          {"display": "AV vegetation", "classes": ["no", "yes"]},
    "VegetationMV":          {"display": "MV vegetation", "classes": ["no", "yes"]},
    "VegetationTV":          {"display": "TV vegetation", "classes": ["no", "yes"]},
    "DiastolicFunction":     {"display": "Diastolic function", "classes": ["normal", "grade_1", "grade_2", "grade_3"]},
    "DiastolicGrade":        {"display": "Diastolic dysfunction grade", "classes": ["normal", "grade_1", "grade_2", "grade_3"]},
}

ALL_TASKS = list(REGRESSION_TASKS.keys()) + list(CLASSIFICATION_TASKS.keys())


# ---------------------------------------------------------------
# Builders
# ---------------------------------------------------------------

def _observation_id(task: str) -> str:
    return f"obs-{task.lower()}"


def build_regression_observation(
    task: str,
    value: float,
    patient_id: str,
    timestamp: str,
    performer: str = "PanEcho AI",
) -> Dict[str, Any]:
    meta = REGRESSION_TASKS[task]
    return {
        "resourceType": "Observation",
        "id": _observation_id(task),
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging",
            }]
        }],
        "code": {
            "coding": [{"system": LOINC_SYSTEM, "code": meta["loinc"], "display": meta["display"]}],
            "text": meta["display"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": timestamp,
        "issued": timestamp,
        "performer": [{"display": performer}],
        "valueQuantity": {
            "value": round(float(value), 2),
            "unit": meta["unit"],
            "system": UNITS_SYSTEM,
            "code": meta["unit"],
        },
        "note": [{"text": "AI pre-read by PanEcho. Verify with interpreting physician."}],
    }


def build_classification_observation(
    task: str,
    predicted_class_idx: int,
    probabilities: List[float],
    patient_id: str,
    timestamp: str,
    performer: str = "PanEcho AI",
) -> Dict[str, Any]:
    meta = CLASSIFICATION_TASKS[task]
    classes = meta["classes"]
    idx = int(predicted_class_idx)
    predicted = classes[idx] if 0 <= idx < len(classes) else "unknown"

    return {
        "resourceType": "Observation",
        "id": _observation_id(task),
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging",
            }]
        }],
        "code": {
            "coding": [{"system": SNOMED_SYSTEM, "code": "25100000", "display": meta["display"]}],
            "text": meta["display"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": timestamp,
        "issued": timestamp,
        "performer": [{"display": performer}],
        "valueCodeableConcept": {
            "coding": [{
                "system": "http://cards-lab.org/panecho/classes",
                "code": predicted,
                "display": predicted.replace("_", " ").title(),
            }],
            "text": predicted.replace("_", " ").title(),
        },
        "component": [
            {
                "code": {"text": c},
                "valueQuantity": {
                    "value": round(float(p), 4),
                    "unit": "probability",
                    "system": UNITS_SYSTEM,
                    "code": "1",
                },
            }
            for c, p in zip(classes, probabilities)
        ],
        "note": [{"text": "AI pre-read by PanEcho. Verify with interpreting physician."}],
    }


def map_predictions_to_bundle(
    predictions: Dict[str, Any],
    patient_id: str = "unknown",
    encounter_id: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    performer: str = "PanEcho AI",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Convert PanEcho's raw output dict into a FHIR R4 Bundle of Observations.

    PanEcho returns a dict like:

        {
          "EF":            tensor([...]),         # regression
          "AVStenosis":    tensor([[0.1, ...]]),  # classification (probabilities)
          ...
        }

    This function normalizes each entry into a FHIR Observation and wraps
    them in a `searchset` Bundle.

    Args:
        predictions: dict mapping task name -> model output. Each value is
            either a scalar tensor (regression) or a 1-D tensor of class
            probabilities (classification).
        patient_id, encounter_id, study_instance_uid: FHIR/DICOM references.
        performer: who/what issued the pre-read.
        timestamp: observation time.

    Returns:
        FHIR R4 Bundle dict with one entry per PanEcho task.
    """
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    observations: List[Dict[str, Any]] = []

    for task, val in predictions.items():
        if task in REGRESSION_TASKS:
            try:
                v = float(val.item()) if hasattr(val, "item") else float(val)
            except (TypeError, ValueError):
                continue
            observations.append(
                build_regression_observation(task, v, patient_id, ts, performer)
            )
        elif task in CLASSIFICATION_TASKS:
            try:
                probs = val.tolist() if hasattr(val, "tolist") else list(val)
                if not isinstance(probs[0], (int, float)):
                    probs = probs[0]  # batched: take first
                idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            except (TypeError, ValueError, IndexError):
                continue
            observations.append(
                build_classification_observation(task, idx, probs, patient_id, ts, performer)
            )

    bundle: Dict[str, Any] = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "searchset",
        "timestamp": ts,
        "total": len(observations),
        "entry": [
            {
                "fullUrl": f"urn:uuid:{o['id']}",
                "resource": o,
            }
            for o in observations
        ],
    }

    if study_instance_uid:
        bundle.setdefault("identifier", []).append({
            "system": "urn:dicom:uid",
            "value": f"urn:oid:{study_instance_uid}",
        })

    return bundle


def build_preread_report(
    predictions: Dict[str, Any],
    patient_id: str = "unknown",
    encounter_id: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    performer: str = "PanEcho AI",
) -> Dict[str, Any]:
    """Build a structured "pre-read" report suitable for echo lab workflow.

    Returns a dict with:
    - `summary`: human-readable summary (1 paragraph)
    - `findings`: per-task findings (regression value or predicted class + prob)
    - `critical_findings`: tasks flagged as critical (e.g. severe AS, tamponade)
    - `fhir_bundle`: the FHIR R4 Bundle from `map_predictions_to_bundle`
    """
    bundle = map_predictions_to_bundle(
        predictions,
        patient_id=patient_id,
        encounter_id=encounter_id,
        study_instance_uid=study_instance_uid,
        performer=performer,
    )

    findings: List[Dict[str, Any]] = []
    critical: List[Dict[str, Any]] = []
    critical_keywords = {"severe", "yes", "large", "tamponade", "dissection", "thrombus"}

    for task, val in predictions.items():
        if task in REGRESSION_TASKS:
            try:
                v = float(val.item()) if hasattr(val, "item") else float(val)
            except (TypeError, ValueError):
                continue
            entry = {
                "task": task,
                "type": "regression",
                "value": round(v, 2),
                "unit": REGRESSION_TASKS[task]["unit"],
                "display": REGRESSION_TASKS[task]["display"],
            }
            findings.append(entry)
            # Critical EF cutoff
            if task == "EF" and v < 35:
                critical.append({**entry, "reason": "EF < 35% (severely reduced)"})
        elif task in CLASSIFICATION_TASKS:
            try:
                probs = val.tolist() if hasattr(val, "tolist") else list(val)
                if not isinstance(probs[0], (int, float)):
                    probs = probs[0]
                idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                predicted = CLASSIFICATION_TASKS[task]["classes"][idx]
                confidence = float(probs[idx])
            except (TypeError, ValueError, IndexError):
                continue
            entry = {
                "task": task,
                "type": "classification",
                "predicted_class": predicted,
                "confidence": round(confidence, 4),
                "probabilities": {
                    CLASSIFICATION_TASKS[task]["classes"][i]: round(float(p), 4)
                    for i, p in enumerate(probs)
                },
                "display": CLASSIFICATION_TASKS[task]["display"],
            }
            findings.append(entry)
            # Critical class detection
            if any(kw in predicted.lower() for kw in critical_keywords):
                critical.append({
                    **entry,
                    "reason": f"Predicted class '{predicted}' for {task}",
                })

    # Build summary
    ef_entry = next((f for f in findings if f["task"] == "EF"), None)
    summary_parts = []
    if ef_entry:
        summary_parts.append(f"EF = {ef_entry['value']:.1f}%")
    severe_classes = [c for c in critical if "severe" in c.get("predicted_class", "").lower()]
    if severe_classes:
        summary_parts.append(f"{len(severe_classes)} severe finding(s)")
    summary = "PanEcho AI pre-read. " + (
        "; ".join(summary_parts) if summary_parts else "no critical findings flagged"
    ) + ". Awaiting physician review."

    return {
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "study_instance_uid": study_instance_uid,
        "performer": performer,
        "summary": summary,
        "findings": findings,
        "critical_findings": critical,
        "fhir_bundle": bundle,
    }
