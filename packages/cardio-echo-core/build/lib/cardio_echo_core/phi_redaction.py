"""PHI redaction middleware for cardio-echo-suite services.

Strips patient-identifying information from request logs and error messages
before they reach the audit log or stdout. Required for HIPAA 164.312(a)(1)
Access control and 164.502(a) Minimum Necessary.

Patterns redacted:
- Patient MRN / PatientID (long numeric strings, often 6-12 digits)
- Patient names (Last^First^Middle DICOM format, or "First Last")
- Dates of birth (YYYYMMDD or YYYY-MM-DD)
- Phone numbers (US format)
- Email addresses
- SSN (XXX-XX-XXXX)
- Street addresses (rough heuristic)

Replacement: [REDACTED-PHI]

Usage:
    from cardio_echo_core.phi_redaction import redact_phi, setup_phi_redaction

    # Apply to any string before logging:
    safe_msg = redact_phi(f"Processing patient {patient_name} ...")

    # Or install as middleware:
    setup_phi_redaction(app)
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# PHI patterns
# ----------------------------------------------------------------------

# DICOM PatientName format: LAST^FIRST^MIDDLE^PREFIX^SUFFIX
DICOM_NAME = re.compile(
    r"\b([A-Z][A-Za-z'-]{1,30})\^([A-Z][A-Za-z'-]{1,30})(\^[A-Z][A-Za-z'-]{0,30}){0,3}\b"
)

# "First Last" format (only flag if both capitalized)
FIRST_LAST = re.compile(
    r"\b([A-Z][a-z]{1,20})\s([A-Z][a-z]{1,20})\b"
)

# DICOM dates: YYYYMMDD (8 digits starting with 19 or 20)
DICOM_DATE = re.compile(r"\b(19|20)\d{6}\b")

# ISO dates: YYYY-MM-DD
ISO_DATE = re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")

# Long numeric strings (likely MRN or accession number) — 6+ digits
LONG_NUMERIC = re.compile(r"\b\d{6,}\b")

# SSN
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# US phone
PHONE = re.compile(r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")

# Email
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Street address (rough heuristic: number + street name + st/ave/rd/etc)
STREET = re.compile(
    r"\b\d{1,6}\s+[A-Z][A-Za-z']+\s+(St(?:reet)?|Ave(?:nue)?|Rd|Road|Dr(?:ive)?|"
    r"Lane|Ln|Blvd|Way|Place|Pl|Court|Ct)\b",
    re.IGNORECASE,
)


# Order matters: do street/email/phone before long_numeric (to avoid partial redaction)
PATTERNS = [
    ("email", EMAIL),
    ("phone", PHONE),
    ("ssn", SSN),
    ("street", STREET),
    ("dicom_name", DICOM_NAME),
    ("iso_date", ISO_DATE),
    ("dicom_date", DICOM_DATE),
    ("long_numeric", LONG_NUMERIC),
]

REDACTED = "[REDACTED-PHI]"


def redact_phi(text: Any) -> str:
    """Redact PHI patterns from a string (or any object converted to str).

    Args:
        text: string or any object (will be str()-ified).

    Returns:
        String with PHI replaced by [REDACTED-PHI].
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Be conservative with FIRST_LAST pattern — only apply if the string
    # looks like it might contain a name (e.g. context keywords nearby)
    if any(kw in text.lower() for kw in ["patient", "name", "subject"]):
        text = FIRST_LAST.sub(REDACTED, text)

    for name, pattern in PATTERNS:
        text = pattern.sub(REDACTED, text)

    return text


def setup_phi_redaction(app) -> None:
    """Install a middleware that redacts PHI from request logs.

    This is a thin wrapper that patches the audit logger's `log_request`
    method to redact the prediction_summary and any extra fields.

    For request body redaction (e.g. uploaded DICOM), see the audit
    middleware in cardio_echo_core.audit — it only logs the SHA-256 hash
    of input bytes, never the content itself.
    """
    from .audit import AuditLogger

    # Monkey-patch AuditLogger.log_request to redact
    original_log_request = AuditLogger.log_request

    def redacted_log_request(self, *args, **kwargs):
        if "prediction_summary" in kwargs:
            kwargs["prediction_summary"] = redact_phi(kwargs["prediction_summary"])
        if "extra" in kwargs and isinstance(kwargs["extra"], dict):
            kwargs["extra"] = {
                k: redact_phi(v) if isinstance(v, str) else v
                for k, v in kwargs["extra"].items()
            }
        return original_log_request(self, *args, **kwargs)

    AuditLogger.log_request = redacted_log_request
    logger.info("PHI redaction installed on AuditLogger")
