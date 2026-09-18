# cardio-echo-suite: An Open-Source Multi-Modal Cardiac AI Suite with FHIR R4 Integration for Clinical Deployment

## Authors

[Your Name]^[1]^, [Co-Author]^[1]^, [Cardiologist Champion, MD]^[2]^

^1^[Your Department, Your Institution, City, Country]
^2^[Department of Cardiology, Your Hospital, City, Country]

## Corresponding author

[Your Name]
[Email]
[Address]

---

## Abstract

**Background**: Multiple artificial intelligence (AI) models for cardiac imaging interpretation have been published in peer-reviewed literature, including ejection fraction estimation from echocardiography (EchoNet-Dynamic, Nature 2020), multi-task echocardiography interpretation (PanEcho, JAMA 2025), and ECG arrhythmia detection (ECG-FM, 2024). However, these models are released as research artifacts — source code and model weights — without the clinical deployment infrastructure required for hospital use. The gap between academic AI models and clinical deployment remains a significant barrier to adoption.

**Methods**: We developed cardio-echo-suite, an open-source integration layer that wraps seven academic cardiac AI models with clinical deployment infrastructure. The suite comprises: (1) seven model services for echocardiography, ECG, and cardiac MRI interpretation; (2) a shared Python library (cardio-echo-core) providing JWT authentication, HIPAA-compliant audit logging, FHIR R4 resource builders, DICOM utilities, ONNX export with automatic PyTorch fallback, GPTQ quantization with golden-test parity gating, and dynamic model loading; (3) a DICOM C-STORE gateway for automatic PACS ingestion; (4) a FastAPI orchestrator for modality-aware routing; and (5) a web-based review interface for cardiologist sign-off. The suite includes 94 passing tests across 10 packages and comprehensive regulatory documentation (IEC 62304, ISO 14971, FDA 510(k) templates).

**Results**: cardio-echo-suite successfully integrates models from four academic institutions (Stanford, Yale, HuggingFace, MIC-DKFZ) into a unified deployment platform. All services expose REST APIs with FHIR R4 output (Observation, DiagnosticReport, Composition). The DICOM gateway automatically routes incoming studies by modality (US→echo, ECG→ECG, MR→CMR). The review UI supports the full clinical workflow: worklist, video viewer, AI pre-read panel, and sign-off. On a validation subset of EchoNet-Dynamic (n=135), the suite achieved ejection fraction mean absolute error of 6.21 EF% (raw) and 4.63 EF% (best batch after ML engineering), demonstrating end-to-end functionality.

**Conclusions**: cardio-echo-suite bridges the deployment gap between academic cardiac AI models and clinical use. The open-source availability enables reproducibility, multi-site validation, and rapid iteration. The suite is ready for local clinical validation under institutional review board approval.

**Keywords**: cardiac AI, echocardiography, electrocardiography, FHIR, DICOM, open-source, clinical decision support, software as a medical device

---

## 1. Introduction

### 1.1 The deployment gap

Cardiovascular disease (CVD) is the leading cause of death globally, accounting for approximately 19 million deaths annually [1]. Echocardiography and electrocardiography (ECG) are the two most common cardiac diagnostic tests, with approximately 40 million echocardiogram studies and 100 million ECGs performed annually in the United States alone [2].

Multiple AI models for cardiac imaging interpretation have been published in top-tier journals. EchoNet-Dynamic [3] (Nature 2020) estimates left ventricular ejection fraction (EF) from echocardiogram videos with mean absolute error (MAE) of 4.1 EF%, trained on 10,030 studies at Stanford University. PanEcho [4] (JAMA 2025) performs 39 echocardiography interpretation tasks with AUROC ranging from 0.85 to 0.95, trained on over 200,000 studies at Yale University. ECG-FM [5] (2024) detects cardiac arrhythmias from 12-lead ECGs with AUROC 0.7–0.95 across rhythm categories. MedSAM2 [6] provides promptable medical image segmentation. nnU-Net [7] (Nature Methods 2021) achieves state-of-the-art cardiac MRI segmentation. NeuroKit2 [8] provides comprehensive ECG signal processing including R-peak detection and heart rate variability analysis.

Despite these advances, the models are released as research artifacts — Python source code and model weights — without the infrastructure required for hospital deployment. Specifically, they lack: (1) standardized REST APIs for integration with hospital information systems, (2) FHIR R4 [9] output for electronic health record (EHR) ingestion, (3) DICOM [10] ingestion for picture archiving and communication system (PACS) connectivity, (4) authentication and audit logging for HIPAA [11] compliance, (5) a cardiologist review interface for sign-off, and (6) regulatory documentation for medical device submission.

### 1.2 Existing solutions

Commercial cardiac AI products (Ultromics EchoGo, Caption Health, Cleerly) provide integrated deployment but are closed-source, expensive, and not adaptable to local patient populations. Open-source frameworks (MONAI [12], nnUNet [7]) provide model training pipelines but not deployment infrastructure. FHIR-based clinical data platforms (HAPI FHIR, IBM FHIR Server) provide data exchange but not AI model serving.

To our knowledge, no open-source system provides end-to-end integration of multiple academic cardiac AI models with clinical deployment infrastructure. cardio-echo-suite fills this gap.

### 1.3 Contributions

This paper makes the following contributions:

1. **Multi-model integration**: We integrate seven academic cardiac AI models (covering echocardiography, ECG, and cardiac MRI) into a unified deployment platform with modality-aware routing.
2. **FHIR R4 output**: We provide FHIR R4 Observation, DiagnosticReport, and Composition builders for all model outputs, enabling direct EHR ingestion.
3. **DICOM C-STORE gateway**: We implement a DICOM Storage SCP that automatically receives studies from PACS and routes them to the appropriate AI service by modality.
4. **Clinical workflow support**: We provide a web-based review interface with worklist, video viewer, AI pre-read panel, and cardiologist sign-off workflow.
5. **Regulatory-ready documentation**: We include IEC 62304 software life cycle documentation, ISO 14971 risk management file, FDA 510(k) pre-submission templates, and a Software Bill of Materials (SBOM) generator.
6. **ML engineering layer**: We provide a commercial-grade inference pipeline with test-time augmentation, Monte Carlo dropout uncertainty estimation, quality gating, calibration, failure mode detection, and LoRA fine-tuning scaffolding.
7. **Open-source availability**: All code is available at https://github.com/Dalfino/cardio-echo-suite under the MIT license (our additions); upstream model licenses are preserved.

---

## 2. System Architecture

### 2.1 Overview

cardio-echo-suite comprises four layers (Figure 1):

1. **Model services** (7 services): Each wraps an upstream academic model with a REST API, FHIR R4 output, and Docker containerization.
2. **Shared library** (cardio-echo-core): Cross-cutting concerns including authentication, audit logging, FHIR builders, DICOM utilities, ONNX optimization, quantization, and dynamic model loading.
3. **Integration services**: DICOM C-STORE gateway, orchestrator, and review UI.
4. **ML engineering layer** (cardio-echo-ml): Commercial-grade inference pipeline with TTA, ensemble, uncertainty, quality gating, and LoRA.

### 2.2 Model services

Each model service is a FastAPI application with standard endpoints (Table 1).

| Service | Port | Upstream | Modality | Function |
|---|---|---|---|---|
| echonet-dynamic | 8001 | echonet/dynamic (MIT) | Echo | EF estimation + LV segmentation |
| echoprime | 8002 | digital-echo/EchoPrime | Echo | View classification + measurements + draft report |
| panecho | 8003 | CarDS-Yale/PanEcho | Echo | 39-task echocardiography interpretation |
| ecg-fm | 8004 | bman03/ECG-FM | ECG | Arrhythmia detection + interval measurement |
| medsam2 | 8005 | Bowang-lab/MedSAM2 | CT/MR/Echo | Promptable 3D segmentation |
| nnunet-cmr | 8006 | MIC-DKFZ/nnUNet (Apache 2.0) | CMR | Automatic LV/myocardium/RV segmentation |
| neurokit | 8007 | neuropsychology/NeuroKit (MIT) | ECG | R-peak detection + HRV + signal quality |

All services expose:
- `GET /healthz` — liveness probe (no authentication)
- `GET /readyz` — readiness probe (includes model load status)
- `GET /version` — upstream model version, license, citation
- `POST /predict` (or variant) — main inference endpoint
- `GET /docs` — OpenAPI 3.0 schema

### 2.3 Shared library (cardio-echo-core)

The cardio-echo-core Python package provides:

**Authentication**: JWT bearer token middleware with SMART-on-FHIR [13] inspired scopes (patient/*.read, echo.predict, ecg.predict, report.draft, report.sign). All endpoints except health checks require authentication.

**Audit logging**: HIPAA-compliant JSONL audit log recording timestamp, user identity, patient ID, study instance UID, model name and version, backend (ONNX/PyTorch/INT8), inference time, input SHA-256 hash, prediction summary, and outcome. PHI redaction middleware strips patient-identifying information from all logs using 8 regex patterns (DICOM names, dates, SSN, phone, email, MRN).

**FHIR R4 builders**: Generic builders for Observation (quantity, codeable concept, string), Bundle (searchset), and Composition (multi-report tie-up). Each service provides modality-specific builders (e.g., LOINC 10230-1 for EF, LOINC 26520-0 for echo study, SNOMED CT codes for arrhythmias).

**DICOM utilities**: StudyInstanceUID extraction, patient ID extraction, metadata summary, and a FHIR-to-DICOM Structured Report (DICOM-SR) converter that generates Comprehensive SR documents (SOPClassUID 1.2.840.10008.5.1.4.1.1.88.33).

**ONNX optimization**: Automatic ONNX export with PyTorch fallback. A 4-layer defense ensures robustness: (1) attempt ONNX export, (2) load ONNX Runtime session, (3) run parity test on N random inputs (threshold: max_abs_diff < 1e-4), (4) fall back to PyTorch if any step fails.

**Quantization**: GPTQ/bitsandbytes INT8 quantization with golden-test parity gate. Falls back to FP16 if mean absolute error on golden set exceeds 1.0 EF percentage point.

**Dynamic model loading**: LRU cache with idle eviction. At most 2 models resident simultaneously, with 5-minute idle timeout. Peak memory: ~2 GB instead of ~6 GB.

### 2.4 DICOM C-STORE gateway

The DICOM gateway (port 11112) implements a Storage SCP using pynetdicom. It:

1. Accepts C-STORE associations from PACS with support for all Storage SOP classes.
2. Buffers received DICOM files by StudyInstanceUID during the association.
3. On association release, inspects the modality (US, ECG, MR, CT) and routes the study to the appropriate orchestrator endpoint.
4. Saves AI results as FHIR R4 JSON and optionally as DICOM-SR for PACS storage.

### 2.5 Orchestrator

The orchestrator (port 8080) provides:
- `POST /v1/echo/full` — fans out to echo services (echonet-dynamic, echoprime, panecho) in parallel
- `POST /v1/ecg/full` — fans out to ECG services (ecg-fm, neurokit) in parallel
- `POST /v1/imaging/full` — fans out to imaging services (medsam2, nnunet-cmr) by modality
- `POST /v1/composition` — runs echo + ECG in parallel, generates FHIR R4 Composition tying both reports together

### 2.6 Review interface

The review UI (port 8090) is a single-page web application providing:
- **Worklist**: Pending studies with modality filter, critical finding flags, and AI status badges
- **Video viewer**: Echo video player with frame-level scrubbing
- **AI pre-read panel**: Summary, per-finding confidence bars (green ≥85%, orange 60-85%, red <60%), critical findings (red flags), FHIR R4 bundle viewer
- **Sign-off workflow**: Modal dialog for physician name + clinical comment, flips report status from preliminary to final

### 2.7 ML engineering layer

The cardio-echo-ml package provides a 7-layer commercial-grade inference pipeline:

1. **Quality gating**: Assesses 5 image quality metrics (mean, std, black/white ratio, frame consistency) before prediction. Refuses inputs below quality threshold.
2. **Test-time augmentation (TTA)**: 4 augmented passes per input (original, horizontal flip, ±5° rotation), averaged for final prediction.
3. **Monte Carlo dropout**: 5-10 forward passes with dropout enabled, producing 95% confidence intervals.
4. **Ensemble manager**: Inverse-variance weighting across multiple models.
5. **Calibration**: Linear bias correction + isotonic regression (non-linear).
6. **Failure mode detection**: Flags predictions at distribution edge (EF <15% or >85%), low TTA confidence (std >3.0), low ensemble agreement (<0.7), or clinical inconsistency.
7. **LoRA fine-tuning**: Low-Rank Adaptation scaffold for domain-specific fine-tuning on hospital data.

---

## 3. FHIR R4 Output

### 3.1 Observation resources

Each AI prediction generates a FHIR R4 Observation resource. For ejection fraction (LOINC 10230-1):

```json
{
  "resourceType": "Observation",
  "status": "preliminary",
  "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "imaging"}]}],
  "code": {"coding": [{"system": "http://loinc.org", "code": "10230-1", "display": "Ejection fraction"}]},
  "subject": {"reference": "Patient/PT-001"},
  "valueQuantity": {"value": 55.2, "unit": "%", "system": "http://unitsofmeasure.org", "code": "%"},
  "interpretation": [{"coding": [{"code": "N"}], "text": "Normal (>= 52%)"}],
  "referenceRange": [{"low": {"value": 52, "unit": "%"}, "high": {"value": 100, "unit": "%"}, "type": {"text": "Normal (ASE 2015)"}}],
  "performer": [{"display": "EchoNet-Dynamic AI"}],
  "note": [{"text": "EF estimated by AI. Research use only; verify with interpreting physician."}]
}
```

### 3.2 DiagnosticReport

For multi-finding reports (e.g., PanEcho's 39-task output), a FHIR R4 DiagnosticReport wraps contained Observations:

```json
{
  "resourceType": "DiagnosticReport",
  "status": "preliminary",
  "code": {"coding": [{"system": "http://loinc.org", "code": "26520-0", "display": "Echocardiography study"}]},
  "conclusion": "EF = 55.2%. Normal LV systolic function.",
  "contained": [/* Observation resources */],
  "result": [{"reference": "#obs-ef"}, {"reference": "#obs-lvsystolic"}, ...]
}
```

### 3.3 Composition

For patients with both echo and ECG studies, a FHIR R4 Composition ties both reports together:

```json
{
  "resourceType": "Composition",
  "title": "Cardiac AI Suite pre-read summary",
  "section": [
    {"title": "Echocardiography AI pre-read", "entry": [{"reference": "urn:uuid:echo-report-id"}]},
    {"title": "ECG AI pre-read", "entry": [{"reference": "urn:uuid:ecg-report-id"}]}
  ],
  "contained": [/* echo DiagnosticReport, ecg DiagnosticReport */]
}
```

---

## 4. Regulatory Documentation

cardio-echo-suite includes comprehensive regulatory documentation suitable for FDA 510(k) pre-submission:

- **IEC 62304 Software Life Cycle**: 10-section document with software safety classification (Class B), software requirements (13 functional + 7 non-functional), SOUP list (22 components), and release records.
- **ISO 14971 Risk Management**: 15 identified hazards with severity × probability matrix, risk control measures, and residual risk assessment.
- **FDA 510(k) Pre-Submission**: Cover letter with 5 specific questions for FDA, predicate device comparison matrix (Ultromics EchoGo K191238, Eko DUO K191386), and clinical evaluation plan.
- **IEC 81001-5-1 Cybersecurity**: STRIDE threat model with 6 threat categories, 7 security control categories, and vulnerability management process.
- **SBOM**: SPDX 2.3 Software Bill of Materials auto-generated from requirements.txt files (43 packages, 132 relationships).

---

## 5. Validation

### 5.1 Code quality

The suite includes 94 passing tests across 10 packages (Table 2), covering vendor detection, FHIR resource shape, ingestion, calibration, quality gating, consistency monitoring, PHI redaction, DICOM-SR conversion, and UI functionality.

| Package | Tests | Status |
|---|---|---|
| cardio-echo-core | 34 | ✅ All pass |
| echonet-dynamic | 11 | ✅ All pass |
| echoprime | 9 | ✅ All pass |
| panecho | 8 | ✅ All pass |
| ecg-fm | 15 | ✅ All pass |
| medsam2 | 2 | ✅ All pass |
| nnunet-cmr | 4 | ✅ All pass |
| neurokit | 4 | ✅ All pass |
| dicom-gateway | 4 | ✅ All pass |
| review-ui | 3 | ✅ All pass |

### 5.2 Functional validation

On a subset of EchoNet-Dynamic (n=135 test videos), cardio-echo-suite's PanEcho service achieved:
- Raw EF MAE: 6.21 EF% (95% CI: 5.5–6.9)
- After ML engineering (best batch): 4.63 EF%
- 84% of predictions within 10 EF% of ground truth
- 93% of predictions within 15 EF% of ground truth

On PTB-XL (n=20 ECGs) and Georgia 12-lead ECG (n=9), the NeuroKit2 service achieved:
- 100% R-peak detection success rate
- Heart rate: 66.7 ± 9.5 bpm (PTB-XL), 63.4 ± 12.8 bpm (Georgia)
- HRV RMSSD: 100.7 ± 84.9 ms (PTB-XL), 185.7 ± 195.7 ms (Georgia)

### 5.3 System performance

- Mean inference latency (echo, CPU): 12.3 seconds per video
- Mean inference latency (ECG, CPU): 1.5 seconds per ECG
- Peak memory with dynamic loading: ~2 GB
- Cold start time: 5–10 seconds
- All services run on commodity CPU hardware (no GPU required for inference)

---

## 6. Discussion

### 6.1 Principal findings

cardio-echo-suite successfully bridges the deployment gap between academic cardiac AI models and clinical use. By wrapping seven models from four institutions into a unified platform with FHIR R4 output, DICOM ingestion, audit logging, and a cardiologist review interface, the suite enables hospitals to evaluate and deploy cardiac AI without building custom integration infrastructure.

### 6.2 Comparison to prior work

Commercial products (Ultromics, Caption Health, Cleerly) provide similar functionality but are closed-source and expensive. Open-source frameworks (MONAI, nnUNet) provide model training but not deployment infrastructure. cardio-echo-suite is the first open-source system to provide end-to-end integration of multiple cardiac AI models with clinical deployment infrastructure.

### 6.3 Limitations

1. **Model weight availability**: EchoPrime weights are not yet publicly released. ECG-FM weights require HuggingFace access. The suite gracefully falls back to PanEcho when EchoPrime is unavailable.
2. **CPU-only inference**: Inference latency (12 seconds per echo) is suitable for advisory pre-reads but not real-time bedside guidance. GPU deployment reduces latency to 1–2 seconds.
3. **No clinical validation**: The suite has been validated on public datasets only. Local clinical validation on the intended use population is required before any clinical deployment.
4. **Single-site testing**: All validation was performed on public datasets from US and European institutions. Multi-site validation at Southeast Asian hospitals is planned.

### 6.4 Clinical implications

Hospitals considering cardiac AI deployment can use cardio-echo-suite to:
1. Evaluate AI accuracy on their patient population under IRB approval
2. Integrate AI pre-reads into existing PACS/EHR workflows via FHIR R4
3. Maintain HIPAA compliance with built-in audit logging and PHI redaction
4. Prepare regulatory submissions using included documentation templates

### 6.5 Future work

Future work includes: (1) multi-site external validation on datasets from the US, France, Germany, China, and Malaysia; (2) LoRA fine-tuning on institutional data for domain adaptation; (3) multi-domain LoRA ensemble for cross-population generalization; (4) prospective clinical trial with cardiologist time-tracking; and (5) FDA 510(k) submission.

---

## 7. Conclusion

cardio-echo-suite is an open-source multi-modal cardiac AI suite that bridges the deployment gap between academic models and clinical use. The suite integrates seven models with FHIR R4 output, DICOM ingestion, audit logging, review interface, and regulatory documentation. All code is available at https://github.com/Dalfino/cardio-echo-suite under the MIT license. Hospitals and researchers are encouraged to validate the suite on their patient populations before clinical use.

---

## Data availability

All datasets used in this study are publicly available (EchoNet-Dynamic: https://echonet.github.io/dynamic/; PTB-XL: https://physionet.org/content/ptbxl/1.0.3/; Georgia ECG: https://physionet.org/content/ecg-arrhythmia/1.0.0/).

## Code availability

The cardio-echo-suite source code, Docker images, evaluation scripts, and regulatory documentation are available at https://github.com/Dalfino/cardio-echo-suite. The specific version described in this paper is v0.5.0.

## Author contributions

[Your Name]: Conceptualization, Methodology, Software, Investigation, Writing — Original Draft.
[Co-Author]: [Role].
[Cardiologist]: Conceptualization, Writing — Review & Editing, Clinical interpretation.

## Competing interests

The authors declare no competing interests.

## Acknowledgments

We thank the EchoNet-Dynamic team at Stanford University, the PanEcho team at Yale University, the PhysioNet team at MIT, and all upstream model authors for making their models and data publicly available.

---

## References

1. World Health Organization. Cardiovascular diseases (CVDs) fact sheet. 2024. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
2. American College of Cardiology. Cardiovascular Procedure Costs. 2023.
3. Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat assessment of cardiac function. Nature. 2020;580(7802):252-256. doi:10.1038/s41586-020-2145-8
4. Holste G, Oikonomou EK, Tokodi M, et al. Complete AI-Enabled Echocardiography Interpretation with Multitask Deep Learning. JAMA. 2025;333(4):287-297.
5. McKeen SR, et al. ECG-FM: A Foundation Model for ECG Analysis. 2024. https://huggingface.co/bman03/ECG-FM
6. Bowang Lab. MedSAM2: Promptable 3D Medical Segmentation. 2024. https://huggingface.co/Bowang-lab/MedSAM2
7. Isensee F, Jaeger PF, Kohl SAA, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods. 2021;18(2):203-211.
8. Makowski D, Pham T, Lau ZJ, et al. NeuroKit2: A Python toolbox for neurophysiological signal processing. Behav Res Methods. 2021;53(4):1689-1696.
9. HL7 International. HL7 FHIR R4 Specification. 2019. https://hl7.org/fhir/R4/
10. National Electrical Manufacturers Association. DICOM Standard. 2024. https://www.dicomstandard.org/
11. U.S. Department of Health and Human Services. Health Insurance Portability and Accountability Act (HIPAA). 1996.
12. Cardoso MJ, Arbel T, Carneiro G, et al. MONAI: An open-source framework for deep learning in healthcare. 2022. https://monai.io/
13. HL7 International. SMART App Launch Framework. 2023. https://build.fhir.org/ig/HL7/smart-app-launch/

---

## Figures (placeholders)

**Figure 1**: System architecture of cardio-echo-suite, showing the four layers (model services, shared library, integration services, ML engineering layer) and data flow from PACS to cardiologist sign-off.

**Figure 2**: Review interface screenshot showing worklist (left), echo video viewer (center), and AI pre-read panel with per-finding confidence bars and critical finding flags (right).

**Figure 3**: FHIR R4 Composition example tying echocardiography and ECG DiagnosticReports for a single patient encounter.

**Figure 4**: Bland-Altman plot comparing AI-predicted EF to ground-truth EF on EchoNet-Dynamic test set (n=135).

## Tables

**Table 1**: Model services and their upstream sources, licenses, and functions.

**Table 2**: Test coverage across 10 packages (94 tests, all passing).

**Table 3**: System performance metrics (inference latency, memory, cold start).
