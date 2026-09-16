"""FHIR R4 utilities shared across services.

Provides:
- `build_observation()` — generic FHIR Observation builder
- `build_bundle()` — FHIR Bundle wrapper
- `build_composition()` — FHIR Composition for cross-modality reports
- LOINC/SNOMED code constants commonly used across services
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"
UNITS_SYSTEM = "http://unitsofmeasure.org"


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def build_observation(
    code: str,
    system: str,
    display: str,
    value: Any,
    value_type: str = "quantity",
    patient_id: str = "unknown",
    encounter_id: Optional[str] = None,
    performer: str = "cardio-echo-suite AI",
    status: str = "preliminary",
    timestamp: Optional[str] = None,
    note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generic FHIR R4 Observation builder.

    Args:
        code: LOINC/SNOMED code (e.g. "10230-1")
        system: code system URL (use LOINC_SYSTEM or SNOMED_SYSTEM)
        display: human-readable display
        value: the observation value. For "quantity" type, this is a dict
            {value, unit, system, code}. For "codeable" type, a dict
            {coding: [{system, code, display}], text}.
        value_type: "quantity" | "codeable" | "string"
        patient_id, encounter_id: FHIR references
        performer: who/what issued the observation
        status: "preliminary" (default for AI) | "final" | "entered-in-error"
        timestamp: ISO 8601 timestamp (defaults to now)
        note: optional note text
        extra: additional fields to merge into the observation
    """
    ts = timestamp or now_iso()
    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": status,
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging",
            }]
        }],
        "code": {
            "coding": [{"system": system, "code": code, "display": display}],
            "text": display,
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": performer}],
    }

    if value_type == "quantity":
        obs["valueQuantity"] = value
    elif value_type == "codeable":
        obs["valueCodeableConcept"] = value
    elif value_type == "string":
        obs["valueString"] = value

    if encounter_id:
        obs["encounter"] = {"reference": f"Encounter/{encounter_id}"}

    if note:
        obs["note"] = [{"text": note}]

    if extra:
        obs.update(extra)

    return obs


def build_bundle(
    resources: List[Dict[str, Any]],
    bundle_type: str = "searchset",
    identifier: Optional[Dict[str, str]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap a list of FHIR resources in a Bundle."""
    bundle: Dict[str, Any] = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": bundle_type,
        "timestamp": timestamp or now_iso(),
        "total": len(resources),
        "entry": [
            {
                "fullUrl": f"urn:uuid:{r.get('id', str(uuid.uuid4()))}",
                "resource": r,
            }
            for r in resources
        ],
    }
    if identifier:
        bundle["identifier"] = [identifier]
    return bundle


def build_composition(
    title: str,
    patient_id: str,
    sections: List[Dict[str, Any]],
    encounter_id: Optional[str] = None,
    author: str = "cardio-echo-suite orchestrator",
    status: str = "preliminary",
    timestamp: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 Composition that ties multiple reports together.

    Used by the orchestrator when a patient has both echo + ECG studies, to
    emit a single Composition that references both DiagnosticReports.
    """
    ts = timestamp or now_iso()
    composition: Dict[str, Any] = {
        "resourceType": "Composition",
        "id": str(uuid.uuid4()),
        "status": status,
        "type": {
            "coding": [{
                "system": LOINC_SYSTEM,
                "code": "11526-1",
                "display": "Pathology study",
            }],
            "text": title,
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": ts,
        "author": [{"display": author}],
        "title": title,
        "section": sections,
    }
    if encounter_id:
        composition["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    if study_instance_uid:
        composition.setdefault("identifier", []).append({
            "system": "urn:dicom:uid",
            "value": f"urn:oid:{study_instance_uid}",
        })
    return composition
