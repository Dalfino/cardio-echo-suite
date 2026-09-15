"""FastAPI app for nnU-Net CMR — automatic cardiac MRI segmentation."""
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

from ..fhir import build_cmr_bundle
from ..ingestion import load_cmr

logger = logging.getLogger("nnunet-cmr")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="nnU-Net CMR service",
    description="Automatic cardiac MRI segmentation (LV, myocardium, RV). Wraps MIC-DKFZ/nnUNet. Research use only.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class CMRSegmentationResult(BaseModel):
    structures: dict
    n_slices: int
    inference_ms: float
    backend: str
    fhir: dict


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {"status": "ready"}


@app.post("/predict", response_model=CMRSegmentationResult)
async def predict(
    file: UploadFile = File(...),
    patient_id: str = Query("unknown"),
    study_uid: Optional[str] = Query(None),
    dry_run: bool = Query(False),
):
    """Upload a CMR volume (NIfTI or DICOM stack zip) and get LV/myo/RV segmentation."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "cmr.nii").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        volume, meta = load_cmr(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Could not parse CMR volume: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    if dry_run:
        # Stub: random masks of the right shape
        import torch
        D, H, W = volume.shape[-3:]
        masks = {
            "lv_blood_pool": (torch.rand(D, H, W) > 0.7).float(),
            "lv_myocardium": (torch.rand(D, H, W) > 0.8).float(),
            "rv_blood_pool": (torch.rand(D, H, W) > 0.75).float(),
        }
        backend = "stub"
    else:
        from ..model import NNUNetCMRModel
        m = NNUNetCMRModel.get()
        if not m._loaded:
            try:
                m.load()
            except Exception as e:
                raise HTTPException(503, f"Model load failed: {e}")
        try:
            masks = m.predict(volume)
        except Exception as e:
            raise HTTPException(500, f"Segmentation failed: {e}")
        backend = m.backend
    dt_ms = (time.perf_counter() - t0) * 1000

    # Compute volumes (placeholder — real calculation needs voxel spacing)
    structures = {}
    for name, mask in masks.items():
        n_voxels = int(mask.sum().item())
        structures[name] = {
            "volume_ml": round(n_voxels / 1000.0, 2),  # placeholder
            "n_slices": int(mask.shape[0]),
        }

    fhir = build_cmr_bundle(
        structures=structures,
        patient_id=patient_id,
        study_instance_uid=study_uid or meta.study_instance_uid,
    )

    return CMRSegmentationResult(
        structures=structures,
        n_slices=meta.n_slices,
        inference_ms=round(dt_ms, 1),
        backend=backend,
        fhir=fhir,
    )


@app.get("/")
def root():
    return {
        "service": "nnunet-cmr",
        "upstream": "https://github.com/MIC-DKFZ/nnUNet",
        "endpoints": ["/healthz", "/readyz", "/predict"],
        "docs": "/docs",
    }
