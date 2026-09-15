"""FHIR R4 ImagingStudy + Observation builders for MedSAM2 outputs."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cardio_echo_core.fhir import LOINC_SYSTEM, SNOMED_SYSTEM, build_observation


def build_segmentation_observation(
    structure: str,
    volume_ml: Optional[float] = None,
    surface_area_mm2: Optional[float] = None,
    n_slices: int = 0,
    patient_id: str = "unknown",
    performer: str = "MedSAM2 AI",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 Observation summarizing a 3D segmentation."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging",
            }]
        }],
        "code": {
            "coding": [{
                "system": SNOMED_SYSTEM,
                "code": "118565006",
                "display": f"Segmentation of {structure}",
            }],
            "text": f"{structure} segmentation (AI)",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": performer}],
        "component": [],
        "note": [{"text": "AI segmentation by MedSAM2. Verify with radiologist."}],
    }
    if volume_ml is not None:
        obs["component"].append({
            "code": {"text": "Volume"},
            "valueQuantity": {"value": round(float(volume_ml), 2), "unit": "mL",
                              "system": "http://unitsofmeasure.org", "code": "mL"},
        })
    if surface_area_mm2 is not None:
        obs["component"].append({
            "code": {"text": "Surface area"},
            "valueQuantity": {"value": round(float(surface_area_mm2), 2), "unit": "mm2",
                              "system": "http://unitsofmeasure.org", "code": "mm2"},
        })
    obs["component"].append({
        "code": {"text": "Slices segmented"},
        "valueQuantity": {"value": int(n_slices), "unit": "slices",
                          "system": "http://unitsofmeasure.org", "code": "1"},
    })
    return obs
