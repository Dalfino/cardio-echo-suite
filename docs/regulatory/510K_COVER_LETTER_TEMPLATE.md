# 510(k) Pre-Submission Cover Letter Template

> **Instructions**: Customize all [BRACKETS]. Submit to FDA via the eSubmit program. Typical pre-submission timeline: 60-90 days for substantive feedback.

---

[Letterhead of submitter]

**Date**: [DATE]

**To**: Document Control Center (DCC)
U.S. Food and Drug Administration
10903 New Hampshire Avenue
Silver Spring, MD 20993-0002

**Re**: Pre-Submission (Q-Sub) for cardio-echo-suite — AI-enabled Echocardiography and ECG Pre-Read Software

**Submitter**:
[COMPANY NAME]
[ADDRESS]
[CITY, STATE ZIP]
Phone: [PHONE]
Email: [EMAIL]
Contact person: [NAME], [TITLE]

**Regulatory Contact** (if different):
[NAME], [TITLE]
[EMAIL], [PHONE]

---

## 1. Device Description

**cardio-echo-suite** is a software-only medical device that provides AI-assisted pre-reading of echocardiogram and ECG studies. The device consists of 7 model services (echo EF, echo VLM, echo 39-task, ECG arrhythmia, promptable segmentation, CMR segmentation, signal processing) plus a DICOM C-STORE gateway, REST API orchestrator, and review web UI.

The device is intended for use as an **advisory pre-read tool** to assist licensed cardiologists in interpreting echocardiogram and ECG studies. All AI outputs are marked `preliminary` and require cardiologist review and sign-off before clinical action. The device is NOT intended for autonomous diagnosis or triage.

**Device classification**: Class II (proposed)
**Product code**: QFM (Medical Image Communication and Management System) and/or OEB (Echocardiogram Analysis Software)
**Review division**: Office of Cardiac Devices (OHT2)

## 2. Proposed intended use

**Indications for use**: cardio-echo-suite is intended to assist licensed cardiologists in the interpretation of adult transthoracic echocardiogram and 12-lead electrocardiogram studies by providing AI-generated pre-reads including:
- Ejection fraction estimation
- 39-task echocardiography interpretation (valves, systolic/diastolic function, pericardium, etc.)
- Cardiac rhythm classification
- ECG interval measurement (PR, QRS, QT, QTc, RR, heart rate)
- STEMI detection and localization
- Cardiac chamber segmentation (echo, CT, MR)

**Limitations**:
- Advisory only — does NOT replace cardiologist interpretation
- NOT for use in pediatric patients (<18 years)
- NOT for use in emergent/STAT studies where AI delay could harm patient
- NOT for autonomous triage

## 3. Proposed predicate devices

| Predicate | Manufacturer | K-number | Indications |
|---|---|---|---|
| Ultromics EchoGo | Ultromics Ltd | K191238 | AI-assisted echocardiography analysis |
| Caption Health Caption Guidance | Caption Health | K200843 | AI-guided echocardiography acquisition |
| Cleerly LABS | Cleerly Inc | K220370 | AI analysis of coronary CT |
| Eko DUO ECG + Auscultation | Eko Devices | K191386 | AI-assisted ECG analysis |

We propose Ultromics EchoGo (K191238) as the **primary predicate** for echocardiography functions, and Eko DUO (K191386) as the **reference predicate** for ECG functions.

## 4. Questions for FDA

We respectfully request FDA feedback on the following:

### 4.1 Intended use
- Is the proposed intended use acceptable, or does FDA require modifications?
- Does FDA agree that "advisory pre-read" positioning is appropriate for Class II clearance?

### 4.2 Predicate selection
- Does FDA agree with Ultromics EchoGo (K191238) as primary predicate?
- For ECG functions, does FDA agree with Eko DUO (K191386) as reference predicate?

### 4.3 Clinical validation
- We propose a retrospective, multi-reader study with 300 cases (200 echo + 100 ECG) at [INSTITUTION NAME], comparing AI predictions to adjudicated cardiologist labels. Is this sample size acceptable?
- We propose subgroup analysis by age, sex, BMI, race, and scanner vendor. Does FDA require additional subgroups?

### 4.4 Software documentation
- We propose to provide IEC 62304 documentation, ISO 14971 risk management file, and IEC 81001-5-1 cybersecurity documentation. Is this complete?
- Our device uses 6 AI models from academic publications (EchoNet-Dynamic, PanEcho, EchoPrime, ECG-FM, MedSAM2, nnU-Net). Does FDA require separate validation for each model, or can we validate the integrated system?

### 4.5 Post-market controls
- We propose a Predetermined Change Control Plan (PCCP) per FDA's 2024 final guidance for model retraining. Does FDA agree this is appropriate?
- We propose annual accuracy audits and adverse event reporting via MedWatch. Are these sufficient?

## 5. Proposed submission timeline

- **Pre-Submission feedback**: 90 days from receipt of this Q-Sub
- **510(k) submission**: Q[X] 202[X]
- **FDA review**: 90 days (Standard) or 60 days (Special 510(k))
- **Target clearance**: Q[X] 202[X]

## 6. Enclosures

1. Device description (3 pages)
2. Proposed intended use and indications (2 pages)
3. Predicate comparison matrix (5 pages)
4. Proposed clinical validation plan (8 pages)
5. Software architecture overview (4 pages)
6. Risk management summary (3 pages)

---

**Sincerely**,

[NAME]
[TITLE]
[COMPANY NAME]

Enclosures (6)

---

*Submission format: eSubmit program per FDA Guidance "Establishment Registration and Device Listing for Device Software Functions" (2023).*
