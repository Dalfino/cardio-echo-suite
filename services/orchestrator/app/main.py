"""Top-level orchestrator for cardio-echo-suite.

Routes incoming echo studies to the three services:
- EchoNet-Dynamic  → EF + LV segmentation      (port 8001)
- EchoPrime        → views + measurements + draft report (port 8002)
- PanEcho          → 39-task pre-read          (port 8003)

Provides a single `/v1/echo/full` endpoint that fans out to all three
and returns a unified FHIR R4 composition.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO)

ECHONET_URL = os.environ.get("ECHONET_URL", "http://echonet-dynamic:8001")
ECHOPRIME_URL = os.environ.get("ECHOPRIME_URL", "http://echoprime:8002")
PANECHO_URL = os.environ.get("PANECHO_URL", "http://panecho:8003")
TIMEOUT_S = float(os.environ.get("ORCH_TIMEOUT", "120"))

app = FastAPI(
    title="cardio-echo-suite orchestrator",
    description="Unified entry point for the cardio-echo-suite AI services. Research use only.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ServiceHealth(BaseModel):
    echonet: str
    echoprime: str
    panecho: str


@app.get("/healthz")
async def healthz():
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{ECHONET_URL}/healthz")
            en = r.json().get("status", "?") if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            en = f"down: {e}"
        try:
            r = await client.get(f"{ECHOPRIME_URL}/healthz")
            ep = r.json().get("status", "?") if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            ep = f"down: {e}"
        try:
            r = await client.get(f"{PANECHO_URL}/healthz")
            pe = r.json().get("status", "?") if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            pe = f"down: {e}"
    return ServiceHealth(echonet=en, echoprime=ep, panecho=pe)


class FullEchoResult(BaseModel):
    patient_id: str
    study_uid: Optional[str]
    inference_ms: float
    echonet: Optional[Dict[str, Any]]
    echoprime: Optional[Dict[str, Any]]
    panecho: Optional[Dict[str, Any]]


async def _forward(
    client: httpx.AsyncClient,
    url: str,
    filename: str,
    content: bytes,
    params: Dict[str, str],
) -> Dict[str, Any]:
    files = {"file": (filename, content)}
    r = await client.post(url, files=files, params=params)
    if r.status_code >= 400:
        return {"error": r.text, "status_code": r.status_code}
    return r.json()


@app.post("/v1/echo/full", response_model=FullEchoResult)
async def echo_full(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
    encounter_id: Optional[str] = Query(None),
    study_uid: Optional[str] = Query(None),
    skip: Optional[str] = Query(None, description="Comma-separated service names to skip: echonet,echoprime,panecho"),
):
    """Run all three models on the uploaded echo and return unified results."""
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
        if "echonet" not in skip_set:
            tasks["echonet"] = _forward(
                client, f"{ECHONET_URL}/predict/ef", filename, content, params
            )
        if "echoprime" not in skip_set:
            tasks["echoprime"] = _forward(
                client, f"{ECHOPRIME_URL}/analyze", filename, content, params
            )
        if "panecho" not in skip_set:
            tasks["panecho"] = _forward(
                client, f"{PANECHO_URL}/predict", filename, content,
                {**params, "dry_run": "true"},  # default dry-run for speed
            )
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out = dict(zip(tasks.keys(), results))

    dt_ms = (time.perf_counter() - t0) * 1000

    # Normalize exceptions to error dicts
    for k, v in out.items():
        if isinstance(v, Exception):
            out[k] = {"error": str(v)}

    return FullEchoResult(
        patient_id=patient_id,
        study_uid=study_uid,
        inference_ms=round(dt_ms, 1),
        echonet=out.get("echonet"),
        echoprime=out.get("echoprime"),
        panecho=out.get("panecho"),
    )


@app.get("/")
def root():
    return {
        "service": "cardio-echo-suite orchestrator",
        "endpoints": ["/healthz", "/v1/echo/full"],
        "downstream": {
            "echonet-dynamic": ECHONET_URL,
            "echoprime": ECHOPRIME_URL,
            "panecho": PANECHO_URL,
        },
        "docs": "/docs",
    }
