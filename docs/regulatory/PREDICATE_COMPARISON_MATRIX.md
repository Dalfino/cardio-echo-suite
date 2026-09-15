# Predicate Device Comparison Matrix

> Side-by-side comparison of cardio-echo-suite against cleared predicates. Used in 510(k) submission to demonstrate substantial equivalence.

---

## 1. Predicate devices

| Device | Manufacturer | K-number | Clearance date | Indications |
|---|---|---|---|---|
| Ultromics EchoGo Core | Ultromics Ltd | K191238 | 2019 | AI analysis of echo for HFpEF/HFrEF detection |
| Caption Health Caption Guidance | Caption Health Inc | K200843 | 2020 | AI guidance for echo acquisition |
| Cleerly LABS | Cleerly Inc | K220370 | 2022 | AI analysis of coronary CTA |
| Eko DUO ECG + Auscultation | Eko Devices | K191386 | 2020 | AI-assisted ECG + heart sound analysis |

**Primary predicate**: Ultromics EchoGo Core (K191238) — closest match for AI echo analysis
**Reference predicate for ECG**: Eko DUO (K191386)

---

## 2. Substantial equivalence comparison

| Feature | cardio-echo-suite | Ultromics EchoGo (K191238) | Eko DUO (K191386) |
|---|---|---|---|
| **Intended use** | AI pre-read for echo + ECG | AI analysis for echo HF detection | AI ECG + auscultation |
| **Advisory / autonomous** | Advisory (preliminary status) | Advisory | Advisory |
| **Patient population** | Adults ≥18 | Adults ≥18 | Adults ≥18 |
| **Modalities** | Echo (TTE), ECG, CMR, CT | Echo (TTE) | ECG, auscultation |
| **Number of AI models** | 7 (multi-model suite) | 1 (HF detection) | 1 (arrhythmia) |
| **Output format** | FHIR R4, DICOM-SR | DICOM-SR | Bluetooth to app |
| **Deployment** | On-prem Docker | Cloud SaaS | Mobile device |
| **Audit log** | HIPAA-compliant JSONL | HIPAA-compliant | HIPAA-compliant |
| **Auth** | JWT (SMART-on-FHIR scopes) | OAuth2 | App PIN |
| **PACS integration** | DICOM C-STORE SCP | DICOM Web | None |
| **Cardiologist review UI** | Yes (web) | Yes (web) | Mobile app |

---

## 3. Technological characteristics comparison

| Characteristic | cardio-echo-suite | Ultromics EchoGo | Eko DUO |
|---|---|---|---|
| **Deep learning framework** | PyTorch 1.13+ | Proprietary | TensorFlow |
| **Model architecture** | R2Plus1D, ConvNeXt-T, Transformer, SAM-2, U-Net | CNN | CNN |
| **Training data size** | Public (10k+ echo, 22k ECG) | Proprietary (50k+) | Proprietary (10k+) |
| **Input format** | DICOM, AVI, MP4, WFDB, CSV | DICOM | ECG signal via Bluetooth |
| **Inference hardware** | CPU (default), GPU (optional) | Cloud GPU | Mobile CPU |
| **Inference latency** | 2-5s CPU / 1-2s GPU | ~10s (cloud round-trip) | <1s |
| **ONNX optimization** | Yes (auto-fallback) | No | No |
| **INT8 quantization** | Yes (EchoPrime only) | No | No |

---

## 4. Clinical performance comparison

| Metric | cardio-echo-suite (proposed) | Ultromics EchoGo (K191238) | Eko DUO (K191386) |
|---|---|---|---|
| **EF MAE** | TBD (target: ≤6.0 EF%) | 4.8 EF% (internal) | N/A |
| **AFib AUROC** | TBD (target: ≥0.85) | N/A | 0.97 (internal) |
| **Multi-task AUC (echo)** | TBD (target: ≥0.80 for ≥30/39 tasks) | N/A (single-task) | N/A |
| **Calibration (Brier)** | TBD | Not reported | Not reported |

**Note**: cardio-echo-suite targets will be confirmed by local clinical validation study (see `docs/irb/PROTOCOL_TEMPLATE.md`).

---

## 5. Risk profile comparison

| Risk | cardio-echo-suite | Ultromics EchoGo | Eko DUO |
|---|---|---|---|
| **Advisory only** | ✅ | ✅ | ✅ |
| **Mandatory MD sign-off** | ✅ | ✅ | ✅ |
| **Confidence gating** | ✅ (multiple layers) | ✅ | ✅ |
| **Audit log** | ✅ HIPAA | ✅ HIPAA | ✅ HIPAA |
| **PHI redaction** | ✅ | ✅ | ✅ |
| **Cybersecurity (IEC 81001-5-1)** | ✅ | ✅ | ✅ |
| **Software class (IEC 62304)** | B | B | B |
| **Clinical risk class** | Advisory (lowest) | Advisory | Advisory |

---

## 6. Conclusion

cardio-echo-suite has the **same intended use** (advisory AI pre-read for cardiologists), **same technological characteristics** (deep learning on medical imaging), and **same risk profile** (advisory only, mandatory MD sign-off) as the predicate devices.

The primary technological differences are:
- cardio-echo-suite is multi-modal (echo + ECG + CMR + CT), whereas Ultromics is echo-only and Eko is ECG-only. This is a **broader** intended use, but each modality is individually predicate-comparable.
- cardio-echo-suite is on-premise (Docker) vs predicates are cloud/mobile. This is a **deployment** difference, not a functional difference.

We believe cardio-echo-suite is **substantially equivalent** to the predicates for each modality-specific function, and we propose to validate each modality separately in the clinical validation study.
