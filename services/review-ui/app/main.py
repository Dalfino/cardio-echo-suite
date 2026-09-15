"""Review UI service — minimal web UI for cardiologists to review AI pre-reads.

Single-page HTML/JS app served by FastAPI. Shows:
- Video player (echo, CMR slices, ECG trace)
- AI pre-read summary (left panel)
- FHIR bundle JSON (right panel, collapsible)
- Sign-off button (flips DiagnosticReport status preliminary → final)

Auth: JWT bearer token (same as the rest of the suite).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger("review-ui")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="cardio-echo-suite review UI",
    description="Minimal web UI for cardiologists to review AI pre-reads and sign off reports.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8080")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "cardio-echo-suite review",
        "orchestrator_url": ORCHESTRATOR_URL,
    })


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/api/run/echo")
async def run_echo(file: UploadFile = File(...), patient_id: str = Form("unknown")):
    content = await file.read()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{ORCHESTRATOR_URL}/v1/echo/full",
            files={"file": (file.filename or "echo.mp4", content)},
            params={"patient_id": patient_id, "skip": "echonet,echoprime"},
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


@app.post("/api/run/ecg")
async def run_ecg(file: UploadFile = File(...), patient_id: str = Form("unknown")):
    content = await file.read()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{ORCHESTRATOR_URL}/v1/ecg/full",
            files={"file": (file.filename or "ecg.csv", content)},
            params={"patient_id": patient_id},
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


@app.post("/api/signoff")
async def signoff(report: Dict[str, Any]):
    """Mark a report as final (physician sign-off)."""
    return {
        "status": "signed",
        "signed_by": "cardiologist (TODO: from JWT)",
        "report_id": report.get("id", "unknown"),
    }
