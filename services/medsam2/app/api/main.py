"""FastAPI app for MedSAM2 — promptable medical segmentation."""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..fhir import build_segmentation_observation
from ..ingestion import load_image
from cardio_echo_core.service_bootstrap import bootstrap_service

logger = logging.getLogger("medsam2")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="MedSAM2 service",
    description="Promptable 3D medical segmentation. Wraps Bowang-lab/MedSAM2. Research use only.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
bootstrap_service(app, service_name="medsam2")


class PointPrompt(BaseModel):
    x: int
    y: int
    label: int = 1  # 1=foreground, 0=background


class BboxPrompt(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SegmentationResult(BaseModel):
    structure: str
    n_voxels: int
    volume_ml: Optional[float]
    surface_area_mm2: Optional[float]
    n_slices: int
    inference_ms: float
    backend: str
    fhir: dict


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {"status": "ready", "model_repo": os.environ.get("MEDSAM2_REPO_ID", "Bowang-lab/MedSAM2")}


@app.post("/predict/point", response_model=SegmentationResult)
async def predict_point(
    file: UploadFile = File(...),
    prompt: PointPrompt = Body(...),
    structure: str = Query("cardiac chamber", description="What structure is being segmented"),
    patient_id: str = Query("unknown"),
):
    """Segment a structure from a single point prompt."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "img.bin").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        image, _ = load_image(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Could not parse image: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    from ..model import MedSAM2Model
    m = MedSAM2Model.get()
    if not m._loaded:
        try:
            m.load()
        except Exception as e:
            raise HTTPException(503, f"Model load failed: {e}")

    t0 = time.perf_counter()
    try:
        mask = m.predict_point(image, (prompt.x, prompt.y), prompt.label)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")
    dt_ms = (time.perf_counter() - t0) * 1000

    # Compute summary stats
    n_voxels = int(mask.sum().item())
    volume_ml = float(n_voxels) / 1000.0  # placeholder; real calc needs voxel spacing
    n_slices = int(mask.shape[-1]) if mask.dim() >= 3 else 1

    fhir = build_segmentation_observation(
        structure=structure,
        volume_ml=volume_ml,
        n_slices=n_slices,
        patient_id=patient_id,
    )
    return SegmentationResult(
        structure=structure,
        n_voxels=n_voxels,
        volume_ml=round(volume_ml, 2),
        surface_area_mm2=None,
        n_slices=n_slices,
        inference_ms=round(dt_ms, 1),
        backend=m.backend,
        fhir=fhir,
    )


@app.post("/predict/bbox", response_model=SegmentationResult)
async def predict_bbox(
    file: UploadFile = File(...),
    prompt: BboxPrompt = Body(...),
    structure: str = Query("cardiac chamber"),
    patient_id: str = Query("unknown"),
):
    """Segment a structure from a bounding box prompt."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "img.bin").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        image, _ = load_image(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Could not parse image: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    from ..model import MedSAM2Model
    m = MedSAM2Model.get()
    if not m._loaded:
        m.load()

    t0 = time.perf_counter()
    try:
        mask = m.predict_bbox(image, (prompt.x1, prompt.y1, prompt.x2, prompt.y2))
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")
    dt_ms = (time.perf_counter() - t0) * 1000

    n_voxels = int(mask.sum().item())
    volume_ml = float(n_voxels) / 1000.0
    n_slices = int(mask.shape[-1]) if mask.dim() >= 3 else 1

    fhir = build_segmentation_observation(
        structure=structure,
        volume_ml=volume_ml,
        n_slices=n_slices,
        patient_id=patient_id,
    )
    return SegmentationResult(
        structure=structure,
        n_voxels=n_voxels,
        volume_ml=round(volume_ml, 2),
        surface_area_mm2=None,
        n_slices=n_slices,
        inference_ms=round(dt_ms, 1),
        backend=m.backend,
        fhir=fhir,
    )


@app.post("/predict/auto", response_model=SegmentationResult)
async def predict_auto(
    file: UploadFile = File(...),
    structure: str = Query("cardiac chamber"),
    patient_id: str = Query("unknown"),
):
    """Automatic segmentation (no prompt — uses image center)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "img.bin").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        image, _ = load_image(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Could not parse image: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    from ..model import MedSAM2Model
    m = MedSAM2Model.get()
    if not m._loaded:
        m.load()

    t0 = time.perf_counter()
    try:
        mask = m.predict_auto(image)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")
    dt_ms = (time.perf_counter() - t0) * 1000

    n_voxels = int(mask.sum().item())
    volume_ml = float(n_voxels) / 1000.0
    n_slices = int(mask.shape[-1]) if mask.dim() >= 3 else 1

    fhir = build_segmentation_observation(
        structure=structure, volume_ml=volume_ml, n_slices=n_slices, patient_id=patient_id,
    )
    return SegmentationResult(
        structure=structure, n_voxels=n_voxels, volume_ml=round(volume_ml, 2),
        surface_area_mm2=None, n_slices=n_slices, inference_ms=round(dt_ms, 1),
        backend=m.backend, fhir=fhir,
    )


@app.get("/")
def root():
    return {
        "service": "medsam2",
        "upstream": "https://huggingface.co/Bowang-lab/MedSAM2",
        "endpoints": ["/healthz", "/readyz", "/predict/point", "/predict/bbox", "/predict/auto"],
        "docs": "/docs",
    }
