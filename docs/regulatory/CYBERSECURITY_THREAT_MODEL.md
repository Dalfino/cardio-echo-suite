# Cybersecurity Threat Model (STRIDE) — cardio-echo-suite

> **Standard**: IEC 81001-5-1:2021 — Security risk management for networked medical devices
> **Framework**: STRIDE (Microsoft) — Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege

---

## 1. Scope

This document identifies cybersecurity threats to cardio-echo-suite and the controls implemented to mitigate them. Applies to all 10 services + the cardio_echo_core library.

## 2. Threat model

### 2.1 Assets

| Asset | Sensitivity | Location |
|---|---|---|
| Patient DICOM studies | PHI (HIPAA) | /data/incoming/, transient in memory |
| Patient ECG files | PHI (HIPAA) | /data/incoming/, transient in memory |
| AI predictions | PHI-derived (HIPAA) | /data/results/, audit.jsonl |
| Model weights | Confidential (license-restricted) | ~/.cache/huggingface, ~/.cache/torch |
| JWT signing secret | Critical | env var |
| Audit log | PHI (HIPAA) | /var/log/cardio/audit.jsonl |
| Patient demographics | PHI (HIPAA) | Database (external) |

### 2.2 Trust boundaries

1. **External network ↔ DICOM gateway** (port 11112): DICOM C-STORE from PACS
2. **External network ↔ orchestrator** (port 8080): REST API from hospital systems
3. **External network ↔ review UI** (port 8090): browser access from cardiologists
4. **Service ↔ service**: internal Docker network
5. **Service ↔ model weights**: HF Hub / torch.hub (internet)
6. **Service ↔ audit log**: filesystem

### 2.3 Adversary model

- **Curious insider**: hospital employee with network access but no auth
- **Malicious insider**: disgruntled employee with credentials
- **External attacker**: internet-based, attempting to exfiltrate PHI or modify predictions
- **Supply chain attacker**: compromises a dependency (PyPI, HuggingFace)

---

## 3. STRIDE threat analysis

### 3.1 Spoofing

| Threat | Severity | Control |
|---|---|---|
| PACS spoofed as DICOM source | Medium | DICOM AE title whitelist (configurable) |
| User spoofed to orchestrator API | High | JWT auth with HS256 signature verification |
| User spoofed to review UI | High | JWT auth + browser SameSite cookies |
| Service spoofed to other services | Medium | Docker network isolation + service mesh (future) |
| HF Hub model repo spoofed | High | Trust remote code = True requires manual review; pin specific revisions in model_registry.py |

### 3.2 Tampering

| Threat | Severity | Control |
|---|---|---|
| Model weights modified at rest | High | SHA-256 verification at load (future) + read-only filesystem for weights |
| Audit log modified | High | Append-only file (chattr +a on Linux) + offsite backup |
| AI prediction tampered in transit | Medium | HTTPS (TLS 1.3) between services + JWT signature on responses |
| Docker image tampered | High | Image signing (cosign, future) + immutable registry |
| Dependency tampered (supply chain) | High | pip-compile lock files + SHA-256 in requirements.txt (future) |

### 3.3 Repudiation

| Threat | Severity | Control |
|---|---|---|
| User denies making a prediction | High | Audit log records user_sub + JWT ID + timestamp for every request |
| Cardiologist denies sign-off | High | Sign-off event recorded with JWT + user identity |
| System denies service was running | Medium | Health check logs + uptime monitoring |

### 3.4 Information disclosure

| Threat | Severity | Control |
|---|---|---|
| PHI leaked via audit log | High | PHI redaction middleware (cardio_echo_core.phi_redaction) |
| PHI leaked via error messages | High | try/except wraps; error responses scrubbed; PHI redaction applied |
| PHI leaked via DICOM gateway logs | High | Logs use SOPInstanceUID (not patient name); DICOM files saved by StudyInstanceUID |
| Model weights leak training data | Medium | Use only published academic models (no fine-tuning with real patient data shipped) |
| Review UI leaks PHI to browser cache | Medium | Cache-Control: no-store headers (future); no localStorage of PHI |
| Audit log file readable by unauthorized | High | File permissions 0600; access restricted to service account |

### 3.5 Denial of service

| Threat | Severity | Control |
|---|---|---|
| Flood of DICOM C-STORE associations | Medium | pynetdicom max associations limit (configurable) |
| Large file upload exhausts disk | Medium | File size limit (100 MB default) + disk monitoring |
| Model load storm (cold start DoS) | Medium | Dynamic loader max_resident=2; queue cold loads |
| GPU memory exhaustion | Medium | CPU default; GPU requires explicit configuration |
| Audit log fills disk | Medium | Log rotation + 6-year retention policy |

### 3.6 Elevation of privilege

| Threat | Severity | Control |
|---|---|---|
| User gains admin scope via JWT manipulation | Critical | HS256 signature verification; scopes checked per endpoint |
| Service account gains host access | High | Docker user namespace remapping + non-root container user |
| Dependency CVE allows RCE | High | Monthly dependency audit + automated patching (Dependabot) |
| Container escape | Critical | Run containers with --security-opt=no-new-privileges + read-only rootfs |

---

## 4. Security controls implemented

### 4.1 Authentication

- ✅ JWT bearer tokens (HS256) for all REST API endpoints
- ✅ SMART-on-FHIR inspired scopes (patient/*.read, echo.predict, etc.)
- ✅ Token expiration (default 1 hour)
- ✅ Token revocation list (future)
- ❌ MFA for review UI (future)

### 4.2 Authorization

- ✅ Per-endpoint scope requirements
- ✅ Audit log records user identity for every request
- ❌ Role-based access control (future — currently scope-based only)

### 4.3 Encryption

- ✅ In transit: TLS 1.3 (configurable via reverse proxy)
- ✅ At rest: AES-256 (filesystem-level, configurable)
- ✅ Audit log: filesystem permissions 0600

### 4.4 Audit

- ✅ Append-only JSONL audit log
- ✅ SHA-256 of input bytes (not the bytes themselves)
- ✅ User identity, timestamp, model version, backend per entry
- ✅ 6-year retention (HIPAA requirement)

### 4.5 PHI protection

- ✅ PHI redaction middleware (8 regex patterns)
- ✅ Audit log never contains raw patient name/DOB/MRN
- ✅ DICOM files saved by StudyInstanceUID, not patient name

### 4.6 Network security

- ✅ Docker network isolation between services
- ✅ Only orchestrator + DICOM gateway + review UI exposed externally
- ✅ Internal services communicate over Docker bridge network
- ❌ mTLS between services (future)

### 4.7 Supply chain

- ✅ Pinned versions in requirements.txt
- ✅ cardio_echo_core.model_registry pins upstream git refs
- ❌ SBOM auto-generation (Phase E.6 will add)
- ❌ Container image signing (future)

---

## 5. Vulnerability management

### 5.1 Vulnerability scanning

- **Frequency**: Monthly
- **Tools**: `pip-audit` for Python deps, `trivy` for Docker images
- **Action**: Critical CVEs patched within 7 days; high within 30 days

### 5.2 Coordinated disclosure

Report security vulnerabilities to: security@[YOUR DOMAIN]

Response SLA:
- Acknowledge: 24 hours
- Initial assessment: 72 hours
- Fix or mitigation: 30 days (high), 90 days (medium)

### 5.3 Post-market surveillance

- Monitor CISA KEV catalog for relevant CVEs
- Subscribe to PyTorch / Transformers / FastAPI security advisories
- Annual penetration testing (recommendation)

---

## 6. Cybersecurity bill of materials (CBOM)

See `SBOM.md` for the software bill of materials. CBOM includes:
- All Python dependencies
- All Docker base images
- All model weight sources (HF Hub, torch.hub)
- All third-party DICOM SOP classes supported

---

## 7. Residual risk

After implementing the controls above, the residual cybersecurity risk is **acceptable** for the advisory intended use. The highest residual risks are:

1. **Supply chain attack on PyPI/HuggingFace**: mitigated by version pinning + monthly audits. Residual risk: medium.
2. **Insider with valid JWT credentials**: mitigated by audit log + scope-based authorization. Residual risk: medium.
3. **Container escape**: mitigated by Docker security options. Residual risk: low.

These residual risks are documented in the ISO 14971 risk register (`RISK_REGISTER.md`).
