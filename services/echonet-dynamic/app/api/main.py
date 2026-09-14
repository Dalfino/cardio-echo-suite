"""FastAPI app for EchoNet-Dynamic.

Endpoints
---------
- `GET  /healthz`        — liveness probe
- `GET  /readyz`         — model loaded?
- `POST /predict/ef`     — upload a DICOM/AVI/MP4 and get EF + FHIR Observation
- `POST /predict/segment` — same input, returns LV segmentation summary

Run:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..ingestion import load_any
from ..fhir import build_ef_observation, build_segmentation_observation
from ..model import EchoNetDynamicModel

logger = logging.getLogger("echonet-dynamic")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="EchoNet-Dynamic service",
    description=(
        "AI ejection-fraction estimation from echocardiogram videos. "
        "Wraps upstream `echonet/dynamic` (Ouyang et al., Nature 2020). "
        "Research use only."
    ),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Optional[EchoNetDynamicModel] = None


def get_model() -> EchoNetDynamicModel:
    global _model
    if _model is None:
        _model = EchoNetDynamicModel.get(
            device=os.environ.get("ECHONET_DEVICE"),
        )
        # Don't auto-load — let /readyz trigger it so we can start
        # the API before weights download (which can be slow).
    return _model


# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    m = get_model()
    if not m._loaded:
        try:
            m.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    return {"status": "ready", "device": m.device, "weights_loaded": m._loaded}


# ---------------------------------------------------------------
# Response models
# ---------------------------------------------------------------

class EFPrediction(BaseModel):
    ef_percent: float
    interpretation: str
    vendor: Optional[str]
    n_frames: int
    inference_ms: float
    fhir: dict


class SegmentationResult(BaseModel):
    n_frames_segmented: int
    mean_lv_area_px: float
    inference_ms: float
    fhir: dict


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------

@app.post("/predict/ef", response_model=EFPrediction)
async def predict_ef(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown", description="FHIR Patient ID"),
    encounter_id: Optional[str] = Query(None),
    study_instance_uid: Optional[str] = Query(None, description="DICOM StudyInstanceUID"),
):
    """Upload a DICOM/AVI/MP4 echo and get an EF prediction + FHIR Observation."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "echo.bin").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        video, meta = load_any(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse input: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    m = get_model()
    if not m._loaded:
        try:
            m.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    t0 = time.perf_counter()
    try:
        ef = m.predict_ef(video)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    dt_ms = (time.perf_counter() - t0) * 1000

    fhir = build_ef_observation(
        ef_percent=ef,
        patient_id=patient_id,
        encounter_id=encounter_id,
        study_instance_uid=study_instance_uid,
        vendor=meta.vendor,
    )

    from ..fhir.observation import _ef_category
    return EFPrediction(
        ef_percent=round(ef, 2),
        interpretation=fhir["interpretation"][0]["text"],
        vendor=meta.vendor,
        n_frames=meta.n_frames,
        inference_ms=round(dt_ms, 1),
        fhir=fhir,
    )


@app.post("/predict/segment", response_model=SegmentationResult)
async def predict_segment(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
):
    """Run LV segmentation across the video and return summary stats + FHIR."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "echo.bin").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        video, meta = load_any(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse input: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    m = get_model()
    if not m._loaded:
        m.load()

    t0 = time.perf_counter()
    try:
        masks, areas = m.predict_segmentation(video)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")
    dt_ms = (time.perf_counter() - t0) * 1000

    fhir = build_segmentation_observation(
        n_frames_segmented=int(masks.shape[1]),
        mean_lv_area_px=float(areas.mean()),
        patient_id=patient_id,
    )
    return SegmentationResult(
        n_frames_segmented=int(masks.shape[1]),
        mean_lv_area_px=round(float(areas.mean()), 2),
        inference_ms=round(dt_ms, 1),
        fhir=fhir,
    )


@app.get("/")
def root():
    return {
        "service": "echonet-dynamic",
        "upstream": "https://github.com/echonet/dynamic",
        "endpoints": ["/healthz", "/readyz", "/predict/ef", "/predict/segment"],
        "docs": "/docs",
    }
