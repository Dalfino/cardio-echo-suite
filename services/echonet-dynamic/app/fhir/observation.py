"""FHIR R4 Observation builder for EchoNet-Dynamic EF predictions.

Generates a FHIR R4 Observation resource conforming to the
`http://hl7.org/fhir/StructureDefinition/Observation` profile, with
the ejection-fraction LOINC code (10230-1) and a reference range
consistent with ASE guidelines.

Reference:
- LOINC 10230-1: "Ejection fraction"
- ASE 2015 chamber quantification guidelines
  (normal EF: >= 52% male / >= 54% female; mildly reduced: 41-51%)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


LOINC_EF = "10230-1"
LOINC_SYSTEM = "http://loinc.org"
UNITS_SYSTEM = "http://unitsofmeasure.org"

EF_INTERPRETATION = {
    "normal": "Normal (>= 52%)",
    "mildly_reduced": "Mildly reduced (41-51%)",
    "moderately_reduced": "Moderately reduced (30-40%)",
    "severely_reduced": "Severely reduced (< 30%)",
}


def _ef_category(ef: float) -> str:
    if ef >= 52:
        return "normal"
    if ef >= 41:
        return "mildly_reduced"
    if ef >= 30:
        return "moderately_reduced"
    return "severely_reduced"


def build_ef_observation(
    ef_percent: float,
    patient_id: str = "unknown",
    encounter_id: Optional[str] = None,
    performer: str = "EchoNet-Dynamic AI",
    device_id: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    vendor: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 Observation for an EF prediction."""
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    category = _ef_category(ef_percent)

    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
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
                    "system": LOINC_SYSTEM,
                    "code": LOINC_EF,
                    "display": "Ejection fraction",
                }
            ],
            "text": "Left ventricular ejection fraction (AI-estimated)",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [
            {
                "reference": "#ai-performer",
                "display": performer,
            }
        ],
        "contained": [
            {
                "resourceType": "Device",
                "id": "ai-performer",
                "deviceName": [
                    {
                        "name": performer,
                        "type": "model-name",
                    }
                ],
                "version": [
                    {"value": "EchoNet-Dynamic (Nature 2020)"},
                ],
                "type": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "462547000",
                            "display": "Artificial intelligence interpretation software",
                        }
                    ]
                },
                **({"udi": [device_id]} if device_id else {}),
            }
        ],
        "valueQuantity": {
            "value": round(float(ef_percent), 1),
            "unit": "%",
            "system": UNITS_SYSTEM,
            "code": "%",
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": {
                            "normal": "N",
                            "mildly_reduced": "L",
                            "moderately_reduced": "L",
                            "severely_reduced": "L",
                        }[category],
                    }
                ],
                "text": EF_INTERPRETATION[category],
            }
        ],
        "referenceRange": [
            {
                "low": {"value": 52, "unit": "%", "system": UNITS_SYSTEM, "code": "%"},
                "high": {"value": 100, "unit": "%", "system": UNITS_SYSTEM, "code": "%"},
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/referencerange-meaning",
                            "code": "normal",
                            "display": "Normal Range",
                        }
                    ],
                    "text": "Normal (ASE 2015)",
                },
            }
        ],
        "note": [
            {
                "text": (
                    f"EF estimated by EchoNet-Dynamic deep learning model "
                    f"(vendor={vendor or 'unknown'}). "
                    f"Research use only; not a substitute for physician interpretation."
                )
            }
        ],
    }

    if encounter_id:
        obs["encounter"] = {"reference": f"Encounter/{encounter_id}"}

    if study_instance_uid:
        obs.setdefault("identifier", []).append(
            {
                "system": "urn:dicom:uid",
                "value": f"urn:oid:{study_instance_uid}",
            }
        )

    return obs


def build_segmentation_observation(
    n_frames_segmented: int,
    mean_lv_area_px: float,
    patient_id: str = "unknown",
    performer: str = "EchoNet-Dynamic AI",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a FHIR R4 Observation summarizing an LV segmentation run."""
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
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
                    "system": LOINC_SYSTEM,
                    "code": "76846-1",
                    "display": "Left ventricular end diastolic volume",
                }
            ],
            "text": "LV segmentation summary (AI)",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": performer}],
        "component": [
            {
                "code": {
                    "coding": [
                        {
                            "system": LOINC_SYSTEM,
                            "code": "76846-1",
                            "display": "LV segmentation frames",
                        }
                    ]
                },
                "valueQuantity": {
                    "value": int(n_frames_segmented),
                    "unit": "frames",
                    "system": UNITS_SYSTEM,
                    "code": "1",
                },
            },
            {
                "code": {
                    "coding": [
                        {
                            "system": LOINC_SYSTEM,
                            "code": "76846-1",
                            "display": "Mean LV area",
                        }
                    ]
                },
                "valueQuantity": {
                    "value": round(float(mean_lv_area_px), 1),
                    "unit": "px^2",
                    "system": UNITS_SYSTEM,
                    "code": "px^2",
                },
            },
        ],
        "note": [
            {
                "text": (
                    "LV segmentation performed by EchoNet-Dynamic deep learning model. "
                    "Research use only."
                )
            }
        ],
    }
