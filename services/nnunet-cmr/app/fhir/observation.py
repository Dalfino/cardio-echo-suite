"""FHIR R4 Observation builders for CMR segmentation outputs.

nnU-Net CMR outputs three label maps:
- Label 1: LV endocardium (blood pool)
- Label 2: LV myocardium
- Label 3: RV endocardium

We compute per-structure volume and mass (where applicable) and emit a
FHIR R4 Observation per structure.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def build_cmr_observation(
    structure: str,
    volume_ml: float,
    n_slices: int,
    patient_id: str = "unknown",
    mass_g: Optional[float] = None,
    performer: str = "nnU-Net CMR AI",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 Observation for a CMR segmentation structure.

    Args:
        structure: "lv_blood_pool" | "lv_myocardium" | "rv_blood_pool"
        volume_ml: structure volume in mL
        n_slices: number of slices segmented
        mass_g: myocardial mass in grams (only for lv_myocardium)
    """
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
                "system": "http://snomed.info/sct",
                "code": "241615005",
                "display": f"Cardiac MRI {structure} measurement",
            }],
            "text": f"{structure} (AI)",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": performer}],
        "component": [
            {
                "code": {"text": "Volume"},
                "valueQuantity": {
                    "value": round(float(volume_ml), 2),
                    "unit": "mL",
                    "system": "http://unitsofmeasure.org",
                    "code": "mL",
                },
            },
            {
                "code": {"text": "Slices segmented"},
                "valueQuantity": {
                    "value": int(n_slices),
                    "unit": "slices",
                    "system": "http://unitsofmeasure.org",
                    "code": "1",
                },
            },
        ],
        "note": [{"text": "AI segmentation by nnU-Net. Verify with radiologist."}],
    }
    if mass_g is not None:
        obs["component"].append({
            "code": {"text": "Mass"},
            "valueQuantity": {
                "value": round(float(mass_g), 2),
                "unit": "g",
                "system": "http://unitsofmeasure.org",
                "code": "g",
            },
        })
    return obs


def build_cmr_bundle(
    structures: Dict[str, Dict[str, Any]],
    patient_id: str = "unknown",
    study_instance_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a FHIR Bundle of CMR Observations.

    Args:
        structures: dict of structure_name -> {volume_ml, mass_g?, n_slices}
    """
    observations = []
    for name, vals in structures.items():
        observations.append(build_cmr_observation(
            structure=name,
            volume_ml=vals.get("volume_ml", 0.0),
            n_slices=vals.get("n_slices", 0),
            mass_g=vals.get("mass_g"),
            patient_id=patient_id,
        ))

    bundle: Dict[str, Any] = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "searchset",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(observations),
        "entry": [{"fullUrl": f"urn:uuid:{o['id']}", "resource": o} for o in observations],
    }
    if study_instance_uid:
        bundle["identifier"] = [{
            "system": "urn:dicom:uid",
            "value": f"urn:oid:{study_instance_uid}",
        }]
    return bundle
