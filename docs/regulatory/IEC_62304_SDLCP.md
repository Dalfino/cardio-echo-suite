# IEC 62304 Software Development Life Cycle Documentation

> **Standard**: IEC 62304:2006/AMD 1:2015 — Medical device software — Software life cycle processes
> **Device class**: Class B (non-serious injury possible) — default for AI SaMD pre-reads
> **Status**: Living document. Update on every software change.

---

## 1. Software Development Plan

### 1.1 Software safety classification

**Class**: B

**Rationale**: cardio-echo-suite produces AI pre-reads for echocardiography and ECG interpretation. The output is marked as `preliminary` and is always reviewed by a licensed cardiologist before clinical action. Incorrect AI predictions could contribute to delayed diagnosis if the cardiologist fails to override, but cannot directly cause serious injury because the AI is advisory only.

If the intended use changes to "autonomous triage without cardiologist review", safety class must be reassessed to Class C.

### 1.2 Software system structure

```
cardio-echo-suite (SYSTEM)
├── cardio_echo_core (SOFTWARE ITEM, Class B)
│   ├── auth (JWT middleware)
│   ├── audit (HIPAA log)
│   ├── fhir (R4 builders)
│   ├── dicom (loader)
│   ├── onnx (export + fallback)
│   ├── quantize (GPTQ + golden test)
│   ├── dynamic_loader (LRU cache)
│   ├── dicom_sr (FHIR → DICOM-SR)
│   ├── phi_redaction
│   ├── accuracy (consistency, calibration, quality)
│   └── service_bootstrap
├── services/echonet-dynamic (SOFTWARE UNIT, Class B)
├── services/echoprime (SOFTWARE UNIT, Class B)
├── services/panecho (SOFTWARE UNIT, Class B)
├── services/ecg-fm (SOFTWARE UNIT, Class B)
├── services/medsam2 (SOFTWARE UNIT, Class B)
├── services/nnunet-cmr (SOFTWARE UNIT, Class B)
├── services/neurokit (SOFTWARE UNIT, Class A — no clinical decision)
├── services/dicom-gateway (SOFTWARE UNIT, Class B)
├── services/orchestrator (SOFTWARE UNIT, Class B)
└── services/review-ui (SOFTWARE UNIT, Class B)
```

### 1.3 Standards applied

- IEC 62304:2006/AMD 1:2015 (software life cycle)
- ISO 14971:2019 (risk management)
- ISO 13485:2016 (quality management)
- IEC 62366-1:2015 (usability engineering)
- IEC 81001-5-1:2021 (cybersecurity for medical devices)
- FDA Guidance: "Content of Premarket Submissions for Device Software Functions" (2023)
- FDA Guidance: "Artificial Intelligence-Enabled Medical Devices" (2024)

### 1.4 Roles and responsibilities

| Role | Name | Responsibility |
|---|---|---|
| Software Project Manager | [TBD] | Overall planning, schedule, deliverables |
| Software Architect | [TBD] | System design, SOUP selection |
| Developer(s) | [TBD] | Implementation, unit testing |
| Tester | [TBD] | Integration/system testing |
| Risk Manager | [TBD] | ISO 14971 risk register maintenance |
| Clinical Lead | [TBD, MD] | Clinical requirements, label adjudication |
| Regulatory Lead | [TBD] | FDA submission, predicate analysis |

### 1.5 Software development environment

- **Languages**: Python 3.10+ (services, core), JavaScript ES2020+ (review UI), SQL (audit log queries)
- **Frameworks**: FastAPI 0.100+, PyTorch 1.13+, Transformers 4.35+, PEFT 0.7+
- **Containerization**: Docker 24+, Docker Compose 2.20+
- **CI/CD**: GitHub Actions (matrix testing across 10 packages)
- **Version control**: Git, hosted on GitHub private repo
- **Dependency management**: `pip-tools`, `uv`, pinned `requirements.txt` per service

### 1.6 SOUP (Software of Unknown Provenance)

See Appendix A for full SOUP list. Critical SOUP:

| Component | Version | Known anomalies | Mitigation |
|---|---|---|---|
| PyTorch | 1.13+ | Non-determinism on GPU | Use CPU for inference; document in IFU |
| Transformers | 4.35+ | API breaks between minor versions | Pin minor version; CI tests on every upgrade |
| onnxruntime | 1.18+ | Custom op export failures | Auto-fallback to PyTorch (see onnx/__init__.py) |
| pydicom | 2.3+ | Vendor-specific DICOM quirks | Vendor sniffing + per-vendor ingestion tests |
| FastAPI | 0.100+ | Pydantic v2 breaking changes | Pin minor version |

---

## 2. Software Requirements Specification (SRS)

### 2.1 Functional requirements

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| FR-01 | System shall accept DICOM C-STORE associations on port 11112 | High | test_gateway.py |
| FR-02 | System shall route echo studies (US modality) to EchoNet-Dynamic + PanEcho | High | test_orchestrator |
| FR-03 | System shall route ECG studies to ECG-FM + NeuroKit | High | test_orchestrator |
| FR-04 | System shall output FHIR R4 DiagnosticReport for each prediction | High | test_fhir (per service) |
| FR-05 | System shall output FHIR R4 Composition when echo + ECG exist for same patient | Medium | test_composition |
| FR-06 | System shall convert FHIR R4 to DICOM-SR on demand | Medium | test_dicom_sr |
| FR-07 | System shall gate ECG-FM predictions when confidence < 0.60 | High | test_ecg-fm |
| FR-08 | System shall reject MedSAM2 segmentations with IoU < 0.50 | High | test_medsam2 |
| FR-09 | System shall fall back to PanEcho when EchoPrime weights unavailable | High | test_echoprime |
| FR-10 | System shall redact PHI from all logs | High | test_phi_redaction |
| FR-11 | System shall authenticate all requests via JWT | High | test_auth |
| FR-12 | System shall log all predictions to audit.jsonl | High | test_audit |
| FR-13 | Review UI shall allow cardiologist sign-off | Medium | test_ui |

### 2.2 Non-functional requirements

| ID | Requirement | Verification |
|---|---|---|
| NFR-01 | Warm inference latency < 5s (CPU) | Benchmark in test_orchestrator |
| NFR-02 | Cold start latency < 30s | Manual timing |
| NFR-03 | Peak memory < 4 GB (with dynamic loading) | Monitor |
| NFR-04 | Audit log retention ≥ 6 years | Filesystem check |
| NFR-05 | All PHI redacted from logs | test_phi_redaction |
| NFR-06 | 95% of requests complete without error | Production monitoring |
| NFR-07 | No model weights redistributed (license compliance) | Build inspection |

---

## 3. Software Architectural Design

(See `README.md` for the architecture diagram.)

### 3.1 Components and interfaces

Each service exposes a FastAPI app with the following standard endpoints:
- `GET /healthz` — liveness (no auth)
- `GET /readyz` — readiness, includes model load status
- `GET /version` — model + upstream version (added by service_bootstrap)
- `POST /predict` (or variant) — main inference endpoint
- `GET /docs` — OpenAPI 3.0 schema

### 3.2 Communication

- **Inter-service**: HTTP/JSON (synchronous, via orchestrator)
- **External**: DICOM C-STORE (port 11112), HTTP REST (port 8080 for orchestrator, 8090 for review UI)
- **No persistent connections** between services (stateless REST)

### 3.3 Data flow

```
PACS → DICOM C-STORE (port 11112) → dicom-gateway
  → orchestrator /v1/echo/full (or /v1/ecg/full, /v1/imaging/full)
    → echonet-dynamic, panecho, ecg-fm (parallel)
    → results aggregated as FHIR R4 Bundle
  → optionally /v1/composition (FHIR Composition)
  → results saved to /data/results/<StudyInstanceUID>.json
  → optionally converted to DICOM-SR (cardio_echo_core.dicom_sr)
  → optionally displayed in review-ui (port 8090)
  → cardiologist sign-off (flips status preliminary → final)
```

---

## 4. Software Detailed Design

Per-service documentation lives in `services/<name>/README.md` (TBD for some — Phase E.2 will add).

### 4.1 cardio_echo_core

The shared library provides cross-cutting concerns. All services depend on it.

**Key design decisions**:
- **Lazy loading**: models are loaded on first request, not at service startup
- **ONNX fallback**: every model attempts ONNX export at load time; falls back to PyTorch on failure
- **INT8 only for EchoPrime**: GPTQ with golden-test parity gate
- **Audit log is append-only**: JSONL format, SHA-256 of inputs (not the inputs themselves)

---

## 5. Software Unit Implementation and Verification

### 5.1 Unit testing

Each software unit has a `tests/` directory with pytest tests. Current coverage:

| Unit | Tests | Status |
|---|---|---|
| cardio_echo_core | 34 | ✅ All pass |
| echonet-dynamic | 11 | ✅ All pass |
| echoprime | 9 | ✅ All pass |
| panecho | 8 | ✅ All pass |
| ecg-fm | 15 | ✅ All pass |
| medsam2 | 2 | ✅ All pass |
| nnunet-cmr | 4 | ✅ All pass |
| neurokit | 4 | ✅ All pass |
| dicom-gateway | 4 | ✅ All pass |
| review-ui | 3 | ✅ All pass |
| **Total** | **94** | ✅ |

### 5.2 CI/CD

GitHub Actions runs the full test matrix on every push and PR. See `.github/workflows/ci.yml`.

---

## 6. Software Integration and Integration Testing

### 6.1 Integration test plan

(TBD — Phase E.3 will add `tests/integration/` with cross-service tests.)

Planned integration tests:
- DICOM C-STORE → orchestrator → PanEcho → audit log (end-to-end echo)
- CSV ECG upload → orchestrator → ECG-FM + NeuroKit → FHIR Composition
- Echo + ECG → orchestrator /v1/composition → FHIR R4 Composition
- EchoPrime fallback → PanEcho substitution (when EchoPrime weights unavailable)
- Auth rejection → 401 on missing JWT
- PHI redaction → no patient name in audit log

---

## 7. Software System Testing

### 7.1 System test plan

(TBD — Phase E.4 will add `tests/system/` with full deployment tests.)

Planned system tests:
- `docker compose up` → all services healthy within 60s
- Upload 10 echo videos → all 10 produce valid FHIR R4
- Upload 10 ECGs → all 10 produce valid FHIR R4
- Upload 5 CMR volumes → all 5 produce segmentations
- Audit log contains all 25 entries with correct SHA-256
- Review UI loads and can sign off a report

---

## 8. Software Release

### 8.1 Release process

1. All tests pass on `main` branch
2. Tag release: `git tag v0.X.Y && git push --tags`
3. Build Docker images: `docker compose -f deploy/docker-compose.yml build`
4. Push to private registry
5. Update release notes in `docs/RELEASE_NOTES.md`
6. Notify deployment team

### 8.2 Version numbering

Semantic versioning: `MAJOR.MINOR.PATCH`
- MAJOR: incompatible API changes
- MINOR: new features, backward compatible
- PATCH: bug fixes

### 8.3 Release records

| Version | Date | Changes | Test report |
|---|---|---|---|
| 0.1.0 | 2026-09-15 | Initial BCA suite (3 echo models) | 28 tests pass |
| 0.2.0 | 2026-09-15 | B+ Lean (6 models + 1 utility + core) | 64 tests pass |
| 0.3.0 | 2026-09-15 | Phase C (DICOM gateway + UI + auth/audit) | 71 tests pass |
| 0.4.0 | 2026-09-15 | Phase A (DICOM-SR + PHI redaction + accuracy tier 2) | 94 tests pass |
| 0.5.0 | 2026-09-16 | Phase D (accuracy gating + IRB templates + validation scripts) | 94 tests pass |

---

## 9. Software Configuration Management

### 9.1 Version control

Git repository: https://github.com/[USER]/cardio-echo-suite (private)

### 9.2 Baselines

Each release tag is a baseline. Source code at any point can be reproduced via `git checkout <tag>`.

### 9.3 Change control

All changes via pull request. Require:
- Code review by 1+ developer
- All CI tests pass
- Updated tests for any new functionality
- Updated documentation

### 9.4 Problem resolution

GitHub Issues track bugs and feature requests. Each issue:
- Has severity (critical / major / minor)
- Has assigned owner
- Has target release

---

## 10. Software Problem Resolution

### 10.1 Problem report process

1. Issue filed on GitHub
2. Triage within 1 business day
3. Reproduce in test environment
4. Fix on feature branch
5. PR + CI tests
6. Merge to main
7. Tag patch release
8. Notify affected sites

### 10.2 Critical bug process

If a critical bug affects patient safety:
1. Immediately stop deployment at all sites
2. Notify clinical leads within 24 hours
3. Submit MedWatch report to FDA if adverse event occurred
4. Root cause analysis (RCA) within 5 business days
5. Corrective and preventive action (CAPA) per ISO 13485

---

## Appendix A: SOUP List

| Component | Version | License | Used by |
|---|---|---|---|
| Python | 3.10+ | PSF | All |
| PyTorch | 1.13+ | BSD-style | All model services |
| torchvision | 0.14+ | BSD | Image services |
| Transformers | 4.35+ | Apache 2.0 | EchoPrime, ECG-FM, MedSAM2 |
| PEFT | 0.7+ | Apache 2.0 | EchoPrime, ECG-FM |
| accelerate | 0.23+ | Apache 2.0 | EchoPrime |
| huggingface_hub | 0.17+ | Apache 2.0 | EchoPrime, ECG-FM, MedSAM2 |
| FastAPI | 0.100+ | MIT | All services |
| uvicorn | 0.22+ | BSD | All services |
| Pydantic | 2.0+ | MIT | All services |
| pydicom | 2.3+ | MIT | DICOM services |
| pynetdicom | 2.0+ | MIT | dicom-gateway |
| httpx | 0.24+ | BSD | Orchestrator, review-ui |
| numpy | 1.22+ | BSD | All |
| opencv-python-headless | 4.7+ | Apache 2.0 | Image services |
| nibabel | 5.0+ | MIT | MedSAM2, nnU-Net CMR |
| nnunetv2 | 2.2+ | Apache 2.0 | nnU-Net CMR |
| NeuroKit2 | 0.2.50+ | MIT | neurokit |
| SciPy | 1.10+ | BSD | neurokit |
| PyJWT | 2.8+ | MIT | cardio_echo_core.auth |
| onnx | 1.14+ | Apache 2.0 | cardio_echo_core.onnx |
| onnxruntime | 1.18+ | MIT | cardio_echo_core.onnx |

## Appendix B: Test Reports

(Generic template — fill in per release)

| Release | Date | Total tests | Passed | Failed | Notes |
|---|---|---|---|---|---|
| 0.5.0 | 2026-09-16 | 94 | 94 | 0 | All pass |

## Appendix C: Risk Management File Reference

See `docs/regulatory/RISK_REGISTER.md` (ISO 14971 risk register).
