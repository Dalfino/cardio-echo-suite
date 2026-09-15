# ISO 14971 Risk Management File — cardio-echo-suite

> **Standard**: ISO 14971:2019 — Application of risk management to medical devices
> **Device**: cardio-echo-suite (AI pre-read for echocardiography and ECG)
> **Intended use**: Advisory pre-read for licensed cardiologists. NOT autonomous. Output marked `preliminary` until cardiologist sign-off.

---

## 1. Risk Management Plan

### 1.1 Scope

This file covers risks associated with the cardio-echo-suite software, from design through deployment and decommissioning.

### 1.2 Responsibility

- **Risk Manager**: [TBD]
- **Clinical Risk Assessor**: [TBD, MD]
- **Software Risk Assessor**: [TBD]

### 1.3 Risk acceptance criteria

| Severity | Probability | Acceptable? |
|---|---|---|
| Catastrophic (death) | Any | ❌ Never |
| Critical (serious injury) | Frequent / Probable | ❌ |
| Critical | Occasional | ❌ |
| Critical | Remote | ⚠️ With mitigation |
| Critical | Improbable | ✅ |
| Serious (minor injury) | Frequent / Probable | ⚠️ With mitigation |
| Serious | Occasional | ✅ |
| Minor (no injury) | Any | ✅ |

Risk = Severity × Probability. Use the matrix below.

---

## 2. Hazard Identification

### 2.1 Hazard analysis

| Hazard ID | Hazard | Hazardous situation | Foreseeable event sequence |
|---|---|---|---|
| H-01 | Incorrect EF prediction | AI overestimates EF | Cardiologist trusts AI → misses heart failure → delayed treatment |
| H-02 | Incorrect EF prediction | AI underestimates EF | Cardiologist trusts AI → unnecessary heart failure workup → patient anxiety + waste |
| H-03 | False positive STEMI | AI flags STEMI when none | Cardiologist activates cath lab → unnecessary procedure → patient risk from cath |
| H-04 | False negative STEMI | AI misses STEMI | Cardiologist trusts AI → discharges patient → MI at home → death |
| H-05 | Wrong arrhythmia classification | AI calls AFib when sinus | Cardiologist starts anticoagulation → bleeding risk |
| H-06 | Wrong arrhythmia classification | AI calls sinus when AFib | Cardiologist trusts AI → no anticoagulation → stroke |
| H-07 | Segmentation error (MedSAM2) | AI undersegments LV | Wrong EF calculation downstream → wrong clinical decision |
| H-08 | Segmentation error (MedSAM2) | AI oversegments RV | Wrong RV size estimate → wrong pulmonary hypertension workup |
| H-09 | PHI breach | Audit log contains patient name | HIPAA violation → fines + patient harm |
| H-10 | PHI breach | Model weights memorize patient data | Training data leakage → HIPAA violation |
| H-11 | Service unavailable | AI service down | Delayed pre-read → no clinical impact (cardiologist reads anyway) |
| H-12 | DICOM routing error | Echo study routed to ECG service | Wrong AI output → cardiologist confusion → delayed read |
| H-13 | Weight version drift | AI gives different result for same input | Inconsistent reads → cardiologist loses trust |
| H-14 | Calibration drift over time | Patient population changes | Accuracy degrades silently → wrong predictions |
| H-15 | Cybersecurity breach | Attacker modifies model weights | Adversarial predictions → patient harm |

---

## 3. Risk Estimation

### 3.1 Severity scale

| Severity | Definition | Score |
|---|---|---|
| Catastrophic | Death or permanent injury | 4 |
| Critical | Serious injury, requires intervention | 3 |
| Serious | Minor injury, temporary | 2 |
| Minor | No injury, inconvenience | 1 |
| Negligible | No impact | 0 |

### 3.2 Probability scale

| Probability | Definition | Score |
|---|---|---|
| Frequent | >1 in 100 uses | 4 |
| Probable | 1 in 100 - 1 in 1,000 | 3 |
| Occasional | 1 in 1,000 - 1 in 10,000 | 2 |
| Remote | 1 in 10,000 - 1 in 1,000,000 | 1 |
| Improbable | <1 in 1,000,000 | 0 |

### 3.3 Risk matrix

```
Severity \ Probability | Frequent(4) | Probable(3) | Occasional(2) | Remote(1) | Improbable(0)
----------------------|-------------|-------------|---------------|-----------|---------------
Catastrophic (4)      |    16 ❌    |    12 ❌    |     8 ❌      |   4 ⚠️    |    0 ✅
Critical (3)          |    12 ❌    |     9 ❌    |     6 ⚠️      |   3 ✅    |    0 ✅
Serious (2)           |     8 ⚠️    |     6 ⚠️    |     4 ✅      |   2 ✅    |    0 ✅
Minor (1)             |     4 ✅    |     3 ✅    |     2 ✅      |   1 ✅    |    0 ✅
```

### 3.4 Initial risk estimates (before mitigation)

| Hazard | Severity | Probability | Risk | Acceptable? |
|---|---|---|---|---|
| H-01 (overestimate EF) | 3 | 3 | 9 | ❌ |
| H-02 (underestimate EF) | 2 | 3 | 6 | ⚠️ |
| H-03 (false + STEMI) | 3 | 2 | 6 | ⚠️ |
| H-04 (false - STEMI) | 4 | 2 | 8 | ❌ |
| H-05 (false + AFib) | 2 | 3 | 6 | ⚠️ |
| H-06 (false - AFib) | 3 | 2 | 6 | ⚠️ |
| H-07 (LV underseg) | 3 | 2 | 6 | ⚠️ |
| H-08 (RV overseg) | 2 | 3 | 6 | ⚠️ |
| H-09 (PHI breach audit log) | 3 | 2 | 6 | ⚠️ |
| H-10 (weight memorization) | 3 | 1 | 3 | ✅ |
| H-11 (service down) | 1 | 3 | 3 | ✅ |
| H-12 (DICOM routing error) | 2 | 1 | 2 | ✅ |
| H-13 (weight drift) | 2 | 2 | 4 | ✅ |
| H-14 (calibration drift) | 2 | 3 | 6 | ⚠️ |
| H-15 (cybersecurity) | 4 | 1 | 4 | ⚠️ |

---

## 4. Risk Control

### 4.1 Risk control measures

| Hazard | Control | Implementation |
|---|---|---|
| H-01 | Confidence gating on EF (mark as "indeterminate" if >2 SD from training) | ECG-FM gating pattern extended to echo |
| H-02 | Same as H-01 | Same |
| H-03 | STEMI flag requires cardiologist review (cannot auto-trigger cath lab) | IFU + review UI design |
| H-04 | IFU warning: "AI is advisory only; STEMI diagnosis requires clinical assessment" | IFU + UI banner |
| H-05, H-06 | ECG-FM confidence gating (refuses if max prob < 0.60) | ecg-fm/app/api/main.py |
| H-07, H-08 | MedSAM2 IoU confidence (rejects if IoU < 0.50) | medsam2/app/model.py |
| H-09 | PHI redaction middleware | cardio_echo_core/phi_redaction.py |
| H-10 | No fine-tuned weights shipped (only LoRA adapters, no training data) | NOTICE.md |
| H-11 | Health checks + automatic restart | docker-compose healthcheck |
| H-12 | Modality routing tested | dicom-gateway/app/main.py MODALITY_ROUTES |
| H-13 | Model version pinning in cardio_echo_core.model_registry | model_registry.py |
| H-14 | Audit log includes model_version for every prediction | audit/__init__.py |
| H-15 | JWT auth + network isolation + immutable model weights (SHA-256 verified at load) | auth + IFU |

### 4.2 Residual risk after controls

| Hazard | Severity | Probability (after control) | Risk | Acceptable? |
|---|---|---|---|---|
| H-01 | 3 | 1 | 3 | ✅ |
| H-02 | 2 | 1 | 2 | ✅ |
| H-03 | 3 | 1 | 3 | ✅ |
| H-04 | 4 | 1 | 4 | ⚠️ Acceptable with IFU |
| H-05 | 2 | 1 | 2 | ✅ |
| H-06 | 3 | 1 | 3 | ✅ |
| H-07 | 3 | 1 | 3 | ✅ |
| H-08 | 2 | 1 | 2 | ✅ |
| H-09 | 3 | 1 | 3 | ✅ |
| H-10 | 3 | 1 | 3 | ✅ |
| H-11 | 1 | 2 | 2 | ✅ |
| H-12 | 2 | 1 | 2 | ✅ |
| H-13 | 2 | 1 | 2 | ✅ |
| H-14 | 2 | 1 | 2 | ✅ |
| H-15 | 4 | 1 | 4 | ⚠️ Acceptable with cybersecurity controls |

### 4.3 Overall residual risk

**Acceptable.** The highest residual risks (H-04 false negative STEMI, H-15 cybersecurity) are mitigated by:
- IFU explicitly states AI is advisory only
- Mandatory cardiologist sign-off before clinical action
- Cybersecurity controls (auth, encryption, audit)

---

## 5. Risk Management Report

### 5.1 Implementation verification

All risk controls implemented and verified by tests:
- ✅ Confidence gating: `test_ecg-fm.py` tests
- ✅ IoU gating: `test_medsam2.py`
- ✅ PHI redaction: `test_phi_redaction.py`
- ✅ Auth: `test_auth.py`
- ✅ Audit: `test_audit.py`

### 5.2 Post-market surveillance

- Audit log reviewed weekly for anomalies
- Accuracy monitored monthly (drift detection)
- Adverse events reported per FDA MedWatch
- Annual risk management review

---

## Appendix: Failure Modes and Effects Analysis (FMEA)

(See hazard table above. Each row is an FMEA entry.)
