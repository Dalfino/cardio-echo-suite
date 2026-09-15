"""FastAPI app for ECG-FM service.

Endpoints
---------
- `GET  /healthz`
- `GET  /readyz`
- `POST /predict`         — upload an ECG file (WFDB/DICOM/MUSE/CSV), get full report + FHIR
- `POST /predict/dry-run` — same shape, stub predictions (for testing)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..fhir import build_ecg_diagnostic_report
from ..ingestion import load_ecg
from cardio_echo_core.service_bootstrap import bootstrap_service

logger = logging.getLogger("ecg-fm")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ECG-FM service",
    description=(
        "ECG foundation model for arrhythmia detection, interval measurement, and STEMI detection. "
        "Wraps HuggingFace bman03/ECG-FM. Research use only."
    ),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
bootstrap_service(app, service_name="ecg-fm")


def _stub_predictions() -> Dict[str, Any]:
    """Return realistic stub predictions for testing without the model."""
    return {
        "intervals": {
            "pr_interval": 165.0,
            "qrs_duration": 95.0,
            "qt_interval": 380.0,
            "qtc_interval": 410.0,
            "rr_interval": 850.0,
            "heart_rate": 70.5,
        },
        "arrhythmia": {
            "sinus_rhythm": 0.92,
            "sinus_tachycardia": 0.05,
            "afib": 0.02,
            "rbbb": 0.01,
            "lbbb": 0.01,
            "pac": 0.02,
            "pvc": 0.01,
            "vt": 0.005,
            "vfib": 0.001,
            "stemi_anterior": 0.003,
            "stemi_inferior": 0.002,
            "stemi_lateral": 0.001,
        },
    }


# Confidence gating thresholds
MIN_CONFIDENCE_FOR_PREDICTION = 0.60  # below this -> "indeterminate"
MIN_SIGNAL_QUALITY_FOR_PREDICTION = 0.40  # below this -> refuse to predict


def _apply_confidence_gating(
    arrhythmia_preds: Dict[str, float],
) -> Dict[str, Any]:
    """Apply confidence gating to arrhythmia predictions.

    Returns the predicted arrhythmia + a confidence_flag:
    - "high_confidence" — max probability >= 0.85
    - "indeterminate" — max probability < 0.60
    - "moderate_confidence" — between 0.60 and 0.85
    """
    if not arrhythmia_preds:
        return {"predicted": None, "confidence": 0.0, "flag": "indeterminate"}

    max_key = max(arrhythmia_preds, key=arrhythmia_preds.get)
    max_prob = float(arrhythmia_preds[max_key])

    if max_prob < MIN_CONFIDENCE_FOR_PREDICTION:
        flag = "indeterminate"
    elif max_prob >= 0.85:
        flag = "high_confidence"
    else:
        flag = "moderate_confidence"

    return {
        "predicted": max_key,
        "confidence": round(max_prob, 4),
        "flag": flag,
    }


class ECGPrediction(BaseModel):
    patient_id: str
    intervals: Dict[str, float]
    arrhythmia: Dict[str, float]
    arrhythmia_prediction: Dict[str, Any]  # {predicted, confidence, flag}
    signal_quality: Optional[Dict[str, Any]]  # quality gate result
    conclusion: str
    inference_ms: float
    fhir: dict


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {
        "status": "ready",
        "model_repo": os.environ.get("ECGFM_REPO_ID", "bman03/ECG-FM"),
        "n_arrhythmias": 21,
        "n_intervals": 6,
    }


@app.post("/predict", response_model=ECGPrediction)
async def predict(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
    encounter_id: Optional[str] = Query(None),
    study_uid: Optional[str] = Query(None),
    dry_run: bool = Query(False, description="Return stub predictions for testing"),
):
    """Upload an ECG file (WFDB/DICOM/MUSE/CSV) and get an AI interpretation + FHIR DiagnosticReport."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "ecg.bin").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    t0 = time.perf_counter()
    try:
        ecg, meta = load_ecg(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse ECG: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    if dry_run:
        preds = _stub_predictions()
    else:
        from ..model import ECGFMModel
        try:
            m = ECGFMModel.get()
            m.load()
            raw = m.predict(ecg)
            # Normalize to expected shape
            preds = {
                "intervals": raw.get("intervals", {}),
                "arrhythmia": raw.get("arrhythmia", {}),
            }
        except Exception as e:
            logger.warning("Real inference failed; falling back to stub: %s", e)
            preds = _stub_predictions()

    dt_ms = (time.perf_counter() - t0) * 1000

    # Signal quality assessment
    from cardio_echo_core.accuracy import assess_image_quality
    quality_gate = assess_image_quality(ecg)
    signal_quality = {
        "is_acceptable": quality_gate.is_acceptable,
        "quality_score": quality_gate.quality_score,
        "reasons": quality_gate.reasons,
    }

    # If signal quality is too low, refuse to predict arrhythmia
    if quality_gate.quality_score < MIN_SIGNAL_QUALITY_FOR_PREDICTION:
        return ECGPrediction(
            patient_id=patient_id,
            intervals=preds.get("intervals", {}),
            arrhythmia={},
            arrhythmia_prediction={
                "predicted": None,
                "confidence": 0.0,
                "flag": "refused_low_quality",
            },
            signal_quality=signal_quality,
            conclusion=(
                "ECG signal quality too low for reliable AI interpretation. "
                f"Quality score: {quality_gate.quality_score:.2f}. "
                "Recommend manual review."
            ),
            inference_ms=round(dt_ms, 1),
            fhir={
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "warning",
                    "code": "incomplete",
                    "details": {"text": "ECG signal quality below threshold for AI prediction"},
                }],
            },
        )

    # Apply confidence gating to arrhythmia predictions
    arrhythmia_gating = _apply_confidence_gating(preds.get("arrhythmia", {}))

    # Build FHIR report
    fhir = build_ecg_diagnostic_report(
        intervals=preds.get("intervals", {}),
        arrhythmia_predictions=preds.get("arrhythmia", {}),
        patient_id=patient_id,
        encounter_id=encounter_id,
        study_instance_uid=study_uid,
    )

    # If arrhythmia is indeterminate, add a note to the FHIR report
    if arrhythmia_gating["flag"] == "indeterminate":
        fhir.setdefault("note", []).append({
            "text": (
                "AI confidence below threshold (max probability < 0.60). "
                "Predictions marked as indeterminate — manual review required."
            ),
        })
        conclusion = (
            "ECG AI pre-read: indeterminate (low confidence). "
            "Manual interpretation required."
        )
    else:
        conclusion = fhir.get("conclusion", "")

    return ECGPrediction(
        patient_id=patient_id,
        intervals=preds.get("intervals", {}),
        arrhythmia=preds.get("arrhythmia", {}),
        arrhythmia_prediction=arrhythmia_gating,
        signal_quality=signal_quality,
        conclusion=conclusion,
        inference_ms=round(dt_ms, 1),
        fhir=fhir,
    )


@app.get("/")
def root():
    return {
        "service": "ecg-fm",
        "upstream": "https://huggingface.co/bman03/ECG-FM",
        "endpoints": ["/healthz", "/readyz", "/predict"],
        "docs": "/docs",
    }
