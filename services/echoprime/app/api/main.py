"""FastAPI app for EchoPrime.

Endpoints
---------
- `GET  /healthz`
- `GET  /readyz`
- `POST /analyze`             — upload an echo video, get views + measurements + draft report
- `POST /finetune/start`      — kick off a LoRA fine-tune job (async)
- `GET  /finetune/status/{id}` — poll fine-tune job status
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..fhir import build_diagnostic_report
from ..model import EchoPrimeModel
from cardio_echo_core.service_bootstrap import bootstrap_service

logger = logging.getLogger("echoprime")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="EchoPrime service",
    description="Vision-language echo AI — view classification, measurements, draft reports. Wraps HuggingFace digital-echo/EchoPrime. Research use only.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
bootstrap_service(app, service_name="echoprime")

_model: Optional[EchoPrimeModel] = None
_finetune_jobs: Dict[str, Dict[str, Any]] = {}


def get_model() -> EchoPrimeModel:
    global _model
    if _model is None:
        _model = EchoPrimeModel.get(
            repo_id=os.environ.get("ECHOPRIME_REPO_ID", "digital-echo/EchoPrime"),
        )
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
    return {
        "status": "ready" if m.weights_available else "degraded",
        "device": m.device,
        "repo_id": m.repo_id,
        "weights_available": m.weights_available,
        "fallback_active": not m.weights_available,
        "fallback_message": (
            "EchoPrime weights not publicly released yet. "
            "Echo studies should be routed to PanEcho (/v1/echo/full with skip=echoprime)."
            if not m.weights_available else None
        ),
    }


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------

class EchoAnalysisResult(BaseModel):
    views: list[str]
    measurements: Dict[str, float]
    draft_report: str
    fhir: dict


@app.post("/analyze", response_model=EchoAnalysisResult)
async def analyze(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
    encounter_id: Optional[str] = Query(None),
    study_instance_uid: Optional[str] = Query(None),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "echo.mp4").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    m = get_model()
    if not m._loaded:
        try:
            m.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    # If EchoPrime weights are unavailable, return a structured fallback response
    # telling the caller to use PanEcho instead.
    if not m.weights_available:
        tmp_path.unlink(missing_ok=True)
        return EchoAnalysisResult(
            views=["unknown"],
            measurements={},
            draft_report=(
                "EchoPrime weights are not yet publicly released. "
                "Please use PanEcho (/v1/echo/full with skip=echonet,echoprime) "
                "for echo analysis until the EchoPrime team publishes weights."
            ),
            fhir={
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "warning",
                    "code": "not-supported",
                    "details": {"text": "EchoPrime weights unavailable — use PanEcho fallback"},
                }],
            },
        )

    try:
        out = m.analyze(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Normalise output (placeholder until EchoPrime public release)
    views = out.get("views", []) or ["unknown"]
    measurements = out.get("measurements", {}) or {}
    draft = out.get("draft_report", "") or out.get("error", "EchoPrime analysis pending public release.")

    fhir = build_diagnostic_report(
        views=views,
        measurements=measurements,
        draft_report=draft,
        patient_id=patient_id,
        encounter_id=encounter_id,
        study_instance_uid=study_instance_uid,
    )

    return EchoAnalysisResult(
        views=views,
        measurements=measurements,
        draft_report=draft,
        fhir=fhir,
    )


# ---------------------------------------------------------------
# Fine-tuning jobs (async stub)
# ---------------------------------------------------------------

class FineTuneRequest(BaseModel):
    train_jsonl: str
    val_jsonl: Optional[str] = None
    output_dir: str = "checkpoints/echoprime-lora"
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 5
    learning_rate: float = 2e-4


@app.post("/finetune/start")
def finetune_start(req: FineTuneRequest):
    """Schedule a LoRA fine-tune job. Returns a job_id immediately.

    NOTE: This is a stub that records the job. The actual training must be
    run on a GPU node via `python -m app.finetune.train ...`. In a real
    deployment this would dispatch to a Celery/Ray job queue.
    """
    job_id = str(uuid.uuid4())
    _finetune_jobs[job_id] = {
        "status": "queued",
        "request": req.model_dump(),
    }
    return {"job_id": job_id, "status": "queued", "message": (
        "Fine-tune job recorded. Run `python -m app.finetune.train` on a GPU "
        "node with the same arguments to actually execute."
    )}


@app.get("/finetune/status/{job_id}")
def finetune_status(job_id: str):
    if job_id not in _finetune_jobs:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return _finetune_jobs[job_id]


@app.get("/")
def root():
    return {
        "service": "echoprime",
        "upstream": "https://huggingface.co/digital-echo/EchoPrime",
        "endpoints": ["/healthz", "/readyz", "/analyze", "/finetune/start", "/finetune/status/{id}"],
        "docs": "/docs",
    }
