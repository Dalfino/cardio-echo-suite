"""Top-level orchestrator for cardio-echo-suite.

Routes incoming studies to all 7 services:
- EchoNet-Dynamic  → EF + LV segmentation      (port 8001)
- EchoPrime        → views + measurements + draft report (port 8002)
- PanEcho          → 39-task pre-read          (port 8003)
- ECG-FM           → ECG rhythm + intervals    (port 8004)
- MedSAM2          → promptable segmentation   (port 8005)
- nnU-Net CMR      → automatic CMR seg         (port 8006)
- NeuroKit2        → HRV + R-peak + quality    (port 8007)

Endpoints:
- GET  /healthz            — aggregated downstream health
- POST /v1/echo/full       — fan out to echo services (echonet + echoprime + panecho)
- POST /v1/ecg/full        — fan out to ECG services (ecg-fm + neurokit)
- POST /v1/imaging/full    — fan out to imaging services (medsam2 + nnunet-cmr)
- POST /v1/composition     — run all relevant services for a patient, return FHIR Composition
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO)

# Service URLs (configurable via env)
SERVICES = {
    "echonet-dynamic": os.environ.get("ECHONET_URL", "http://echonet-dynamic:8001"),
    "echoprime":       os.environ.get("ECHOPRIME_URL", "http://echoprime:8002"),
    "panecho":         os.environ.get("PANECHO_URL", "http://panecho:8003"),
    "ecg-fm":          os.environ.get("ECGFM_URL", "http://ecg-fm:8004"),
    "medsam2":         os.environ.get("MEDSAM2_URL", "http://medsam2:8005"),
    "nnunet-cmr":      os.environ.get("NNUNET_CMR_URL", "http://nnunet-cmr:8006"),
    "neurokit":        os.environ.get("NEUROKIT_URL", "http://neurokit:8007"),
}
TIMEOUT_S = float(os.environ.get("ORCH_TIMEOUT", "120"))

app = FastAPI(
    title="cardio-echo-suite orchestrator",
    description="Unified entry point for the cardio-echo-suite AI services (7 services). Research use only.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    """Aggregated health check across all 7 downstream services."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        results = {}
        for name, url in SERVICES.items():
            try:
                r = await client.get(f"{url}/healthz")
                results[name] = r.json().get("status", "?") if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as e:
                results[name] = f"down: {type(e).__name__}"
        return results


@app.get("/services")
def list_services():
    """List all configured downstream services."""
    return {"services": SERVICES, "version": "0.2.0"}


async def _forward(
    client: httpx.AsyncClient,
    service_name: str,
    endpoint: str,
    filename: str,
    content: bytes,
    params: Dict[str, str],
    method: str = "POST",
) -> Dict[str, Any]:
    """Forward a request to a downstream service."""
    url = f"{SERVICES[service_name]}{endpoint}"
    files = {"file": (filename, content)}
    try:
        r = await client.post(url, files=files, params=params)
        if r.status_code >= 400:
            return {"error": r.text, "status_code": r.status_code, "service": service_name}
        return r.json()
    except Exception as e:
        return {"error": str(e), "service": service_name}


class FullResult(BaseModel):
    patient_id: str
    study_uid: Optional[str]
    inference_ms: float
    results: Dict[str, Any]


@app.post("/v1/echo/full", response_model=FullResult)
async def echo_full(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
    encounter_id: Optional[str] = Query(None),
    study_uid: Optional[str] = Query(None),
    skip: Optional[str] = Query(None, description="Comma-separated service names to skip"),
):
    """Run all 3 echo models (echonet, echoprime, panecho) on an uploaded echo."""
    content = await file.read()
    filename = file.filename or "echo.mp4"
    skip_set = {s.strip() for s in skip.split(",")} if skip else set()

    params = {
        "patient_id": patient_id,
        "encounter_id": encounter_id or "",
        "study_uid": study_uid or "",
        "study_instance_uid": study_uid or "",
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        tasks = {}
        if "echonet" not in skip_set and "echonet-dynamic" not in skip_set:
            tasks["echonet"] = _forward(client, "echonet-dynamic", "/predict/ef", filename, content, params)
        if "echoprime" not in skip_set:
            tasks["echoprime"] = _forward(client, "echoprime", "/analyze", filename, content, params)
        if "panecho" not in skip_set:
            tasks["panecho"] = _forward(client, "panecho", "/predict",
                                        filename, content, {**params, "dry_run": "true"})
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out = dict(zip(tasks.keys(), results))
    dt_ms = (time.perf_counter() - t0) * 1000
    for k, v in out.items():
        if isinstance(v, Exception):
            out[k] = {"error": str(v)}

    return FullResult(patient_id=patient_id, study_uid=study_uid,
                      inference_ms=round(dt_ms, 1), results=out)


@app.post("/v1/ecg/full", response_model=FullResult)
async def ecg_full(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
    encounter_id: Optional[str] = Query(None),
    study_uid: Optional[str] = Query(None),
):
    """Run ECG-FM + NeuroKit2 on an uploaded ECG file."""
    content = await file.read()
    filename = file.filename or "ecg.csv"
    params = {
        "patient_id": patient_id,
        "encounter_id": encounter_id or "",
        "study_uid": study_uid or "",
        "dry_run": "true",  # ecg-fm dry-run by default for speed
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        tasks = {
            "ecg-fm": _forward(client, "ecg-fm", "/predict", filename, content, params),
            "neurokit-hrv": _forward(client, "neurokit", "/hrv", filename, content,
                                     {"fs": "250"}),
            "neurokit-quality": _forward(client, "neurokit", "/quality", filename, content,
                                         {"fs": "250"}),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out = dict(zip(tasks.keys(), results))
    dt_ms = (time.perf_counter() - t0) * 1000
    for k, v in out.items():
        if isinstance(v, Exception):
            out[k] = {"error": str(v)}

    return FullResult(patient_id=patient_id, study_uid=study_uid,
                      inference_ms=round(dt_ms, 1), results=out)


@app.post("/v1/imaging/full", response_model=FullResult)
async def imaging_full(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
    encounter_id: Optional[str] = Query(None),
    study_uid: Optional[str] = Query(None),
    modality: str = Query("auto", description="cmr | ct | auto"),
):
    """Run imaging services (medsam2, nnunet-cmr) on an uploaded image."""
    content = await file.read()
    filename = file.filename or "image.nii"
    params = {
        "patient_id": patient_id,
        "encounter_id": encounter_id or "",
        "study_uid": study_uid or "",
        "dry_run": "true",
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        tasks = {}
        if modality in ("cmr", "auto"):
            tasks["nnunet-cmr"] = _forward(client, "nnunet-cmr", "/predict",
                                           filename, content, params)
        if modality in ("ct", "auto"):
            tasks["medsam2"] = _forward(client, "medsam2", "/predict/auto",
                                        filename, content,
                                        {**params, "structure": "cardiac chamber"})
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out = dict(zip(tasks.keys(), results))
    dt_ms = (time.perf_counter() - t0) * 1000
    for k, v in out.items():
        if isinstance(v, Exception):
            out[k] = {"error": str(v)}

    return FullResult(patient_id=patient_id, study_uid=study_uid,
                      inference_ms=round(dt_ms, 1), results=out)


# ---------------------------------------------------------------
# FHIR Composition — ties echo + ECG reports for one patient
# ---------------------------------------------------------------

class CompositionResult(BaseModel):
    patient_id: str
    inference_ms: float
    composition: Dict[str, Any]
    echo_report: Optional[Dict[str, Any]] = None
    ecg_report: Optional[Dict[str, Any]] = None


@app.post("/v1/composition", response_model=CompositionResult)
async def composition(
    echo_file: Optional[UploadFile] = File(None, description="Echo video file"),
    ecg_file: Optional[UploadFile] = File(None, description="ECG file"),
    patient_id: str = Query("unknown"),
    encounter_id: Optional[str] = Query(None),
    study_uid: Optional[str] = Query(None),
):
    """Run echo + ECG services in parallel and emit a FHIR R4 Composition
    that ties both reports together for one patient encounter.
    """
    from cardio_echo_core.fhir import build_composition, build_bundle

    t0 = time.perf_counter()
    echo_report = None
    ecg_report = None

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        tasks = {}
        if echo_file:
            echo_content = await echo_file.read()
            echo_params = {
                "patient_id": patient_id,
                "study_uid": study_uid or "",
                "skip": "echonet,echoprime",  # PanEcho only for speed
            }
            tasks["echo"] = _forward(client, "panecho", "/predict",
                                     echo_file.filename or "echo.mp4",
                                     echo_content, {**echo_params, "dry_run": "true"})

        if ecg_file:
            ecg_content = await ecg_file.read()
            ecg_params = {"patient_id": patient_id, "study_uid": study_uid or "", "dry_run": "true"}
            tasks["ecg"] = _forward(client, "ecg-fm", "/predict",
                                    ecg_file.filename or "ecg.csv",
                                    ecg_content, ecg_params)

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            out = dict(zip(tasks.keys(), results))
            if "echo" in out and not isinstance(out["echo"], Exception):
                echo_report = out["echo"]
            if "ecg" in out and not isinstance(out["ecg"], Exception):
                ecg_report = out["ecg"]

    dt_ms = (time.perf_counter() - t0) * 1000

    # Build FHIR Composition sections
    sections = []
    if echo_report and "fhir_bundle" in (echo_report or {}):
        sections.append({
            "title": "Echocardiography AI pre-read",
            "entry": [{"reference": f"urn:uuid:{echo_report['fhir_bundle'].get('id', '')}"}],
        })
    if ecg_report and "fhir" in (ecg_report or {}):
        sections.append({
            "title": "ECG AI pre-read",
            "entry": [{"reference": f"urn:uuid:{ecg_report['fhir'].get('id', '')}"}],
        })

    composition = build_composition(
        title="Cardiac AI Suite pre-read summary",
        patient_id=patient_id,
        sections=sections,
        encounter_id=encounter_id,
        study_instance_uid=study_uid,
    )

    # Add contained resources (the actual reports)
    contained = []
    if echo_report and "fhir_bundle" in echo_report:
        contained.append(echo_report["fhir_bundle"])
    if ecg_report and "fhir" in ecg_report:
        contained.append(ecg_report["fhir"])
    composition["contained"] = contained

    return CompositionResult(
        patient_id=patient_id,
        inference_ms=round(dt_ms, 1),
        composition=composition,
        echo_report=echo_report,
        ecg_report=ecg_report,
    )


@app.get("/")
def root():
    return {
        "service": "cardio-echo-suite orchestrator v0.2.0",
        "endpoints": ["/healthz", "/services", "/v1/echo/full", "/v1/ecg/full", "/v1/imaging/full", "/v1/composition"],
        "downstream": SERVICES,
        "docs": "/docs",
    }
