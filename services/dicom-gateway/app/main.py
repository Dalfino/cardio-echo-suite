"""DICOM C-STORE SCP — receives studies, routes to AI services."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pynetdicom import AE, evt, StoragePresentationContexts
from pynetdicom.sop_class import Verification

logger = logging.getLogger("dicom-gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DEFAULT_AE_TITLE = os.environ.get("DICOM_AE_TITLE", "CARDIO_ECHO_SUITE")
DEFAULT_PORT = int(os.environ.get("DICOM_PORT", "11112"))
INCOMING_DIR = Path(os.environ.get("DICOM_INCOMING_DIR", "/data/incoming"))
RESULTS_DIR = Path(os.environ.get("DICOM_RESULTS_DIR", "/data/results"))
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8080")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")

# Modality → orchestrator endpoint mapping
MODALITY_ROUTES = {
    "US": "/v1/echo/full",       # Ultrasound (echo)
    "ES": "/v1/echo/full",       # Endosonography (sometimes used for TEE)
    "ECG": "/v1/ecg/full",       # ECG (waveform)
    "MR": "/v1/imaging/full",    # MRI
    "CT": "/v1/imaging/full",    # CT
    "XA": "/v1/imaging/full",    # X-ray angiography
}


# ----------------------------------------------------------------------
# C-STORE handler
# ----------------------------------------------------------------------

class StudyBuffer:
    """Buffers incoming DICOM files for a single StudyInstanceUID.

    When a C-STORE association closes, all buffered studies are flushed
    to the appropriate AI service.
    """

    def __init__(self, incoming_dir: Path):
        self.incoming_dir = incoming_dir
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self._studies: Dict[str, List[Path]] = {}  # StudyInstanceUID -> [file paths]
        self._study_meta: Dict[str, Dict[str, Any]] = {}  # StudyInstanceUID -> metadata

    def add_file(self, ds) -> None:
        """Add a received DICOM dataset to the buffer."""
        study_uid = str(getattr(ds, "StudyInstanceUID", "unknown"))
        series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        sop_uid = str(getattr(ds, "SOPInstanceUID", "unknown"))
        patient_id = str(getattr(ds, "PatientID", "unknown"))
        modality = str(getattr(ds, "Modality", "unknown"))
        study_desc = str(getattr(ds, "StudyDescription", ""))

        # Create per-study directory
        study_dir = self.incoming_dir / study_uid
        study_dir.mkdir(parents=True, exist_ok=True)

        # Save the DICOM file
        file_path = study_dir / f"{sop_uid}.dcm"
        ds.save_as(str(file_path))

        # Track
        self._studies.setdefault(study_uid, []).append(file_path)
        self._study_meta[study_uid] = {
            "patient_id": patient_id,
            "modality": modality,
            "study_description": study_desc,
            "study_instance_uid": study_uid,
            "n_files": len(self._studies[study_uid]),
            "received_at": time.time(),
        }
        logger.info(
            "Received DICOM: study=%s series=%s sop=%s modality=%s (n=%d)",
            study_uid[-12:], series_uid[-12:], sop_uid[-12:], modality,
            self._study_meta[study_uid]["n_files"],
        )

    def flush(self) -> Dict[str, Dict[str, Any]]:
        """Return all buffered studies. Caller is responsible for processing."""
        studies = dict(self._study_meta)
        files = dict(self._studies)
        self._studies.clear()
        self._study_meta.clear()
        return {"meta": studies, "files": files}


# Module-level buffer (shared across C-STORE handlers within one association)
_buffer: Optional[StudyBuffer] = None


def get_buffer() -> StudyBuffer:
    global _buffer
    if _buffer is None:
        _buffer = StudyBuffer(INCOMING_DIR)
    return _buffer


def handle_store(event):
    """C-STORE handler — saves the dataset and acknowledges."""
    ds = event.dataset
    ds.file_meta = event.file_meta
    get_buffer().add_file(ds)
    return 0x0000  # Success


def handle_echo(event):
    """C-ECHO handler — simple acknowledgment."""
    logger.info("C-ECHO from %s", event.assoc.requestor.ae_title)
    return 0x0000


# ----------------------------------------------------------------------
# Study routing
# ----------------------------------------------------------------------

async def route_study(study_uid: str, meta: Dict[str, Any], files: List[Path]) -> Dict[str, Any]:
    """Route a completed study to the appropriate AI service via the orchestrator."""
    modality = meta.get("modality", "unknown").upper()
    patient_id = meta.get("patient_id", "unknown")

    endpoint = MODALITY_ROUTES.get(modality)
    if endpoint is None:
        logger.warning("No route for modality %s (study %s)", modality, study_uid[-12:])
        return {"status": "no_route", "modality": modality}

    # Pick the first file as representative (orchestrator will re-read)
    # In production, you'd zip the whole study directory.
    if not files:
        return {"status": "no_files", "study_uid": study_uid}
    upload_file = files[0]

    # Issue a JWT token (in production, use a real issuer)
    from cardio_echo_core.auth import issue_token
    token = issue_token(
        sub="dicom-gateway",
        scopes=["echo.predict", "ecg.predict", "segment.prompt", "report.draft"],
        secret=JWT_SECRET,
    )

    logger.info("Routing study %s to %s (modality=%s)", study_uid[-12:], endpoint, modality)
    t0 = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            params = {
                "patient_id": patient_id,
                "study_uid": study_uid,
            }
            if modality in ("MR", "CT"):
                params["modality"] = modality.lower()
            elif endpoint == "/v1/echo/full":
                # Default to dry-run for echo (Phase A will switch to real)
                params["dry_run"] = "true"

            with upload_file.open("rb") as f:
                files_payload = {"file": (upload_file.name, f.read(), "application/dicom")}
                r = await client.post(
                    f"{ORCHESTRATOR_URL}{endpoint}",
                    files=files_payload,
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )

        dt_ms = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            logger.error("Orchestrator error %d for study %s: %s",
                         r.status_code, study_uid[-12:], r.text[:200])
            return {"status": "error", "status_code": r.status_code, "body": r.text[:500]}

        result = r.json()
        result["routing_ms"] = round(dt_ms, 1)
        result["endpoint"] = endpoint
        result["modality"] = modality

        # Save result
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULTS_DIR / f"{study_uid}.json"
        result_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info("Study %s result saved to %s", study_uid[-12:], result_path)
        return result

    except Exception as e:
        logger.exception("Failed to route study %s", study_uid[-12:])
        return {"status": "exception", "error": str(e)}


async def process_completed_studies(studies: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process all completed studies after an association closes."""
    meta_dict = studies["meta"]
    files_dict = studies["files"]
    tasks = []
    for study_uid, meta in meta_dict.items():
        files = files_dict.get(study_uid, [])
        tasks.append(route_study(study_uid, meta, files))
    return await asyncio.gather(*tasks, return_exceptions=True)


# ----------------------------------------------------------------------
# AE setup
# ----------------------------------------------------------------------

def setup_ae(ae_title: str = DEFAULT_AE_TITLE) -> AE:
    """Create and configure the Application Entity."""
    ae = AE(ae_title=ae_title)
    # Support Verification (C-ECHO)
    ae.add_supported_context(Verification)
    # StoragePresentationContexts is a list of PresentationContext objects
    # Each has an abstract_syntax attribute that is the SOP Class UID
    try:
        for cx in StoragePresentationContexts:
            # PresentationContext object — extract the abstract syntax
            abstract = getattr(cx, "abstract_syntax", None)
            if abstract is None and isinstance(cx, tuple) and len(cx) > 0:
                abstract = cx[0]
            if abstract is not None:
                try:
                    ae.add_supported_context(abstract)
                except Exception:
                    pass
    except Exception:
        pass
    return ae


def run_server(port: int = DEFAULT_PORT, ae_title: str = DEFAULT_AE_TITLE) -> None:
    """Run the DICOM C-STORE SCP forever."""
    ae = setup_ae(ae_title)
    logger.info("Starting DICOM C-STORE SCP: AE=%s port=%d", ae_title, port)
    logger.info("Incoming dir: %s", INCOMING_DIR)
    logger.info("Results dir:  %s", RESULTS_DIR)
    logger.info("Orchestrator: %s", ORCHESTRATOR_URL)

    # Set up event handlers
    handlers = [
        (evt.EVT_C_STORE, handle_store),
        (evt.EVT_C_ECHO, handle_echo),
        (evt.EVT_ACCEPTED, handle_accepted),
        (evt.EVT_RELEASED, handle_released),
    ]

    try:
        ae.start_server(("0.0.0.0", port), evt_handlers=handlers)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        ae.shutdown()


def handle_accepted(event):
    """Called when an association is accepted."""
    logger.info("Association accepted from %s", event.assoc.requestor.ae_title)


async def handle_released(event):
    """Called when an association is released — process buffered studies."""
    logger.info("Association released — flushing study buffer")
    buffer = get_buffer()
    studies = buffer.flush()
    if studies["meta"]:
        logger.info("Processing %d completed study/studies", len(studies["meta"]))
        results = await process_completed_studies(studies)
        for study_uid, result in zip(studies["meta"].keys(), results):
            if isinstance(result, Exception):
                logger.error("Study %s failed: %s", study_uid[-12:], result)
            else:
                logger.info("Study %s processed: status=%s",
                            study_uid[-12:], result.get("status", "unknown"))


# pynetdicom expects sync handlers, so we need to schedule async work
_original_handle_released = handle_released


def handle_released_sync(event):
    """Sync wrapper that schedules the async handler."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_original_handle_released(event))
        else:
            loop.run_until_complete(_original_handle_released(event))
    except RuntimeError:
        # No event loop — run in a new one
        asyncio.run(_original_handle_released(event))


# Replace with sync version
handle_released = handle_released_sync


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="dicom-gateway", description="DICOM C-STORE SCP for cardio-echo-suite")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--ae-title", default=DEFAULT_AE_TITLE)
    args = p.parse_args(argv)
    run_server(port=args.port, ae_title=args.ae_title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
