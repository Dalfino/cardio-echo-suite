"""FastAPI app for PanEcho.

Endpoints
---------
- `GET  /healthz`
- `GET  /readyz`
- `POST /predict`        — upload a video, get the full pre-read report + FHIR bundle
- `POST /predict/dry-run` — same shape but with stub predictions (for testing)
- `GET  /tasks`          — list all PanEcho tasks (39)
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

from ..fhir import ALL_TASKS, CLASSIFICATION_TASKS, REGRESSION_TASKS, build_preread_report
from ..cli import predict as cli_predict
from cardio_echo_core.service_bootstrap import bootstrap_service

logger = logging.getLogger("panecho")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="PanEcho service",
    description="Multi-task echo AI pre-read (39 tasks). Wraps CarDS-Yale/PanEcho (JAMA 2025). Research use only.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
bootstrap_service(app, service_name="panecho")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    # We don't preload — torch.hub.load is expensive.
    return {
        "status": "ready",
        "model_repo": os.environ.get("PANECHO_REPO", "CarDS-Yale/PanEcho"),
        "n_tasks": len(ALL_TASKS),
        "n_regression": len(REGRESSION_TASKS),
        "n_classification": len(CLASSIFICATION_TASKS),
    }


@app.get("/tasks")
def list_tasks():
    return {
        "regression": REGRESSION_TASKS,
        "classification": CLASSIFICATION_TASKS,
        "total": len(ALL_TASKS),
    }


class PanEchoReport(BaseModel):
    patient_id: str
    encounter_id: Optional[str]
    study_instance_uid: Optional[str]
    summary: str
    findings: list[Dict[str, Any]]
    critical_findings: list[Dict[str, Any]]
    fhir_bundle: Dict[str, Any]
    inference_ms: float


@app.post("/predict", response_model=PanEchoReport)
async def predict_endpoint(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
    encounter_id: Optional[str] = Query(None),
    study_uid: Optional[str] = Query(None),
    clip_len: int = Query(16, ge=4, le=64),
    dry_run: bool = Query(False, description="Return stub predictions for testing"),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "echo.mp4").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    t0 = time.perf_counter()
    try:
        report = cli_predict(
            tmp_path,
            patient_id=patient_id,
            encounter_id=encounter_id,
            study_uid=study_uid,
            clip_len=clip_len,
            dry_run=dry_run,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)
    dt_ms = (time.perf_counter() - t0) * 1000

    return PanEchoReport(
        patient_id=report["patient_id"],
        encounter_id=report["encounter_id"],
        study_instance_uid=report["study_instance_uid"],
        summary=report["summary"],
        findings=report["findings"],
        critical_findings=report["critical_findings"],
        fhir_bundle=report["fhir_bundle"],
        inference_ms=round(dt_ms, 1),
    )


@app.get("/")
def root():
    return {
        "service": "panecho",
        "upstream": "https://github.com/CarDS-Yale/PanEcho",
        "endpoints": ["/healthz", "/readyz", "/tasks", "/predict"],
        "docs": "/docs",
    }
