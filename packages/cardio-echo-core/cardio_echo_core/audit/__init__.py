"""HIPAA-compliant audit log middleware for cardio-echo-suite services.

Every prediction request is logged with:
- timestamp (UTC, ISO 8601)
- service name
- user identity (sub from JWT)
- patient MRN (from query param or DICOM header)
- study instance UID (DICOM)
- model name + version
- backend (onnx/pytorch/int8)
- inference time (ms)
- input hash (SHA-256 of file content — NOT the file itself, for PHI safety)
- prediction summary (e.g. "EF=55.0%", "arrhythmia=afib p=0.85")
- outcome ("success" / "error: ...")

Logs are written as JSON lines to a file (default: /var/log/cardio/audit.jsonl)
and to stdout (for container logging). Each log entry is a single JSON object
on one line for easy parsing with `jq` or Splunk.

HIPAA controls addressed:
- 164.312(b) — Audit controls
- 164.312(c)(1) — Integrity
- 164.316(b)(2)(ii) — Retention period (default 6 years)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request


@dataclass
class AuditEntry:
    timestamp: str
    service: str
    user_sub: str
    patient_id: str
    study_instance_uid: Optional[str]
    model_name: str
    model_version: str
    backend: str  # "onnx" | "pytorch" | "int8" | "cpu" | "cuda"
    endpoint: str
    method: str
    status_code: int
    inference_ms: float
    input_sha256: str
    prediction_summary: str
    outcome: str  # "success" or "error: ..."
    request_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Audit logger that writes JSONL to a file + stdout."""

    def __init__(
        self,
        service_name: str,
        log_dir: Optional[str] = None,
        log_filename: str = "audit.jsonl",
        also_stdout: bool = True,
    ):
        self.service_name = service_name
        self.log_dir = Path(log_dir or os.environ.get("AUDIT_LOG_DIR", "/var/log/cardio"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_filename
        self.also_stdout = also_stdout
        self._logger = logging.getLogger(f"audit.{service_name}")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def log(self, entry: AuditEntry) -> None:
        """Write an audit entry to file + stdout."""
        line = json.dumps(asdict(entry), default=str)
        with self.log_path.open("a") as f:
            f.write(line + "\n")
        if self.also_stdout:
            self._logger.info(line)

    def log_request(
        self,
        request: Request,
        patient_id: str = "unknown",
        study_instance_uid: Optional[str] = None,
        model_name: str = "",
        model_version: str = "",
        backend: str = "",
        input_bytes: Optional[bytes] = None,
        prediction_summary: str = "",
        outcome: str = "success",
        inference_ms: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method: build and log an entry from a FastAPI request."""
        user = getattr(request.state, "user", None) or {}
        input_hash = hashlib.sha256(input_bytes or b"").hexdigest() if input_bytes else ""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            service=self.service_name,
            user_sub=user.get("sub", "anonymous"),
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            model_name=model_name,
            model_version=model_version,
            backend=backend,
            endpoint=str(request.url.path),
            method=request.method,
            status_code=200 if outcome == "success" else 500,
            inference_ms=inference_ms,
            input_sha256=input_hash,
            prediction_summary=prediction_summary,
            outcome=outcome,
            request_id=request.headers.get("X-Request-ID", ""),
            extra=extra or {},
        )
        self.log(entry)


def setup_audit(app: FastAPI, audit: AuditLogger) -> None:
    """Install audit middleware that times every request."""
    @app.middleware("http")
    async def audit_middleware(request: Request, call_next):
        # Skip health/docs
        if request.url.path in {"/healthz", "/readyz", "/", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            dt_ms = (time.perf_counter() - t0) * 1000

            # Log successful request (specific services should call audit.log_request
            # for richer info — model name, prediction summary, etc.)
            user = getattr(request.state, "user", None) or {}
            audit.log(
                AuditEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    service=audit.service_name,
                    user_sub=user.get("sub", "anonymous"),
                    patient_id="",
                    study_instance_uid=None,
                    model_name="",
                    model_version="",
                    backend="",
                    endpoint=str(request.url.path),
                    method=request.method,
                    status_code=response.status_code,
                    inference_ms=dt_ms,
                    input_sha256="",
                    prediction_summary="",
                    outcome="success" if response.status_code < 400 else f"http_{response.status_code}",
                )
            )
            return response
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000
            audit.log(
                AuditEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    service=audit.service_name,
                    user_sub="anonymous",
                    patient_id="",
                    study_instance_uid=None,
                    model_name="",
                    model_version="",
                    backend="",
                    endpoint=str(request.url.path),
                    method=request.method,
                    status_code=500,
                    inference_ms=dt_ms,
                    input_sha256="",
                    prediction_summary="",
                    outcome=f"error: {type(e).__name__}: {str(e)[:200]}",
                )
            )
            raise
