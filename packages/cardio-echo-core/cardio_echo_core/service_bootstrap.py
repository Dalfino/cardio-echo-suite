"""Service bootstrap helper — wires auth + audit into a FastAPI app.

Usage in any service:

    from cardio_echo_core.service_bootstrap import bootstrap_service

    app = FastAPI(...)
    bootstrap_service(app, service_name="echonet-dynamic")

This installs:
- JWT auth middleware (skipped for /healthz, /readyz, /docs, /openapi.json)
- Audit log middleware (records every request)
- /version endpoint (reports model version + backend)
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI

from .auth import setup_auth
from .audit import AuditLogger, setup_audit
from .model_registry import get_config


def bootstrap_service(
    app: FastAPI,
    service_name: str,
    enable_auth: Optional[bool] = None,
    enable_audit: bool = True,
    log_dir: Optional[str] = None,
) -> None:
    """Install auth + audit + version endpoint on a service's FastAPI app.

    Args:
        app: the FastAPI app
        service_name: must be in cardio_echo_core.model_registry.REGISTRY
        enable_auth: defaults to True if JWT_SECRET env var is set, else False
            (dev mode). Pass True/False to override.
        enable_audit: default True
        log_dir: where to write audit.jsonl (defaults to AUDIT_LOG_DIR env
            or /var/log/cardio)
    """
    cfg = get_config(service_name)

    if enable_auth is None:
        enable_auth = bool(os.environ.get("JWT_SECRET"))

    if enable_auth:
        setup_auth(
            app,
            jwt_secret=os.environ.get("JWT_SECRET", "dev-secret-change-me"),
            jwt_issuer=os.environ.get("JWT_ISSUER", "cardio-echo-suite"),
            jwt_audience=os.environ.get("JWT_AUDIENCE", "cardio-echo-suite"),
            allow_no_auth=os.environ.get("AUTH_ALLOW_NO_AUTH", "").lower()
                          in ("1", "true", "yes"),
        )

    if enable_audit:
        audit = AuditLogger(
            service_name=service_name,
            log_dir=log_dir or os.environ.get("AUDIT_LOG_DIR", "/var/log/cardio"),
        )
        setup_audit(app, audit)

    # /version endpoint — reports pinned upstream version
    @app.get("/version")
    def version():
        return {
            "service": service_name,
            "cardio_echo_core_version": "0.1.0",
            "upstream_repo": cfg.upstream_repo,
            "upstream_ref": cfg.upstream_ref,
            "upstream_license": cfg.upstream_license,
            "upstream_citation": cfg.upstream_citation,
            "preferred_backend": cfg.preferred_backend.value,
            "fallback_backend": cfg.fallback_backend.value,
            "quantize": cfg.quantize,
        }
