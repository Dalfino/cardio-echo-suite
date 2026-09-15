# IRB Protocol Template — cardio-echo-suite Pilot Validation Study

> **Instructions**: This is a template. Customize sections marked [BRACKETS] for your institution. Submit to your IRB via their standard protocol submission process. Typical review timeline: 4-8 weeks for expedited review, 8-12 weeks for full board.

---

## 1. Study Title

**Validation of an Open-Source Cardiac AI Suite (cardio-echo-suite) for Echocardiography and ECG Pre-Reading at [INSTITUTION NAME]**

## 2. Principal Investigator

- **Name**: [PI NAME, MD]
- **Department**: Cardiology / Cardiovascular Medicine
- **Email**: [PI EMAIL]
- **Phone**: [PI PHONE]

## 3. Co-Investigators

- [Co-I 1, role]
- [Co-I 2, role]

## 4. Study Background and Rationale

### 4.1 Background

Cardiovascular disease remains the leading cause of death globally. Echocardiography and electrocardiography (ECG) are the two most common cardiac diagnostic tests, with ~40 million echo studies and ~100 million ECGs performed annually in the US alone. Cardiologist read times are a bottleneck: typical echo turnaround is 4-24 hours, and stat studies in the ED/ICU can wait 30-60 minutes.

Multiple AI models have been published in peer-reviewed literature for cardiac imaging interpretation:

- **EchoNet-Dynamic** (Ouyang et al., Nature 2020) — ejection fraction estimation from echo videos. Trained on 10,030 studies at Stanford. External validation MAE 4-9 EF%.
- **PanEcho** (Holste et al., JAMA 2025) — 39-task echo interpretation model. Trained on 200,000+ studies at Yale.
- **ECG-FM** (McKeen et al., 2024) — ECG foundation model for arrhythmia detection. AUROC 0.7-0.95 across arrhythmias.
- **MedSAM2** (2024) — promptable medical image segmentation.
- **nnU-Net** (Isensee et al., Nature Methods 2021) — automatic cardiac MRI segmentation.

These models are released as research artifacts (code + weights) but lack the deployment infrastructure required for hospital use: REST APIs, FHIR R4 output, DICOM ingestion, audit logging, and integration with PACS/EHR.

### 4.2 The cardio-echo-suite

`cardio-echo-suite` is an open-source integration layer that wraps these academic models with:

- 7 model services (echo EF, echo VLM, echo 39-task, ECG, promptable seg, CMR seg, signal processing)
- DICOM C-STORE gateway for automatic PACS ingestion
- FHIR R4 output (Observation, DiagnosticReport, Composition)
- HIPAA-compliant audit logging
- JWT authentication
- Review web UI for cardiologist sign-off

All upstream model code and weights are unchanged from their academic releases. cardio-echo-suite adds only the integration plumbing.

### 4.3 Rationale for this study

Before clinical deployment, we must validate that the AI outputs are accurate on our institution's patient population, scanner mix, and sonographer workflow. This study will:

1. Run the cardio-echo-suite on 300-500 de-identified echo + ECG studies
2. Compare AI predictions to ground-truth cardiologist labels
3. Measure accuracy overall and per subgroup (age, sex, BMI, scanner vendor)
4. Identify failure modes for quality improvement
5. Generate a validation report suitable for FDA 510(k) pre-submission discussions

## 5. Specific Aims

**Aim 1**: Validate EchoNet-Dynamic EF estimation on [INSTITUTION] echo studies. Target: MAE ≤ 6.0 EF%, R ≥ 0.80.

**Aim 2**: Validate PanEcho 39-task predictions on [INSTITUTION] echo studies. Target: AUC ≥ 0.80 for ≥30 of 39 tasks.

**Aim 3**: Validate ECG-FM arrhythmia detection on [INSTITUTION] ECGs. Target: AUROC ≥ 0.85 for the top 10 most common arrhythmias.

**Aim 4**: Perform subgroup analysis (age, sex, BMI, scanner vendor) to identify equity gaps. Target: no subgroup accuracy >10 percentage points below overall.

**Aim 5**: Catalog failure modes for the top 20 worst predictions per model for quality improvement.

## 6. Study Design

### 6.1 Design

Retrospective, single-site, validation study using de-identified imaging data.

### 6.2 Population

- **Inclusion criteria**: All patients ≥18 years who had an echocardiogram OR 12-lead ECG at [INSTITUTION] between [START DATE] and [END DATE].
- **Exclusion criteria**: Studies with technical artifacts (sonographer-flagged), studies where the patient withdrew consent for research use.
- **Target N**: 300 echo studies + 300 ECGs (stratified by scanner vendor, age decile, sex).

### 6.3 Data sources

- **Echo DICOM studies**: pulled from [INSTITUTION] PACS (vendor: [SECTRA/GE/PHILIPS/FUJI])
- **ECG studies**: pulled from [INSTITUTION] ECG management system (GE MUSE / Philips TC55)
- **Cardiologist labels**: extracted from final signed reports in the EHR
- **Demographics**: age, sex, BMI, race/ethnicity (from EHR)

### 6.4 De-identification

All studies will be de-identified per HIPAA Safe Harbor (45 CFR 164.514(b)) before transfer to the cardio-echo-suite environment. The following identifiers will be removed:

- Patient names
- Medical record numbers
- Dates of birth (will use age at study)
- Dates of service (will use offset dates)
- Geographic details below state level
- Phone/fax/email
- SSN
- Account numbers

A re-identification key will be held by the [INSTITUTION] Honest Broker office and destroyed after study completion.

### 6.5 Procedures

1. **Data extraction** (weeks 1-2): Honest broker pulls and de-identifies studies from PACS/EHR.
2. **Cardiologist label extraction** (weeks 3-4): Final signed report text is parsed to extract ground-truth labels (EF, rhythm, valve findings).
3. **AI inference** (weeks 5-6): cardio-echo-suite runs on de-identified studies. All predictions are logged with input hash, model version, and timestamp.
4. **Statistical analysis** (weeks 7-8): Compute per-task accuracy, calibration, subgroup analysis. Generate validation report.
5. **Failure mode review** (weeks 9-10): Cardiologist reviews top 20 worst predictions per model. Categorizes failure modes.
6. **Report generation** (weeks 11-12): Final validation report suitable for FDA pre-submission.

### 6.6 Outcomes

**Primary outcomes**:
- EchoNet-Dynamic: MAE between AI EF and cardiologist EF
- PanEcho: per-task AUC (regression: Pearson r; classification: AUROC)
- ECG-FM: per-arrhythmia AUROC

**Secondary outcomes**:
- Subgroup accuracy (age, sex, BMI, scanner vendor, race/ethnicity)
- Calibration (Brier score, reliability diagrams)
- Failure mode categorization
- Inference latency per study

## 7. Risks and Benefits

### 7.1 Risks

- **Breach of confidentiality**: low — all data is de-identified per HIPAA Safe Harbor; stored on [INSTITUTION] secured servers; access limited to study team.
- **No direct intervention risk** — this is a retrospective validation; no patient contact.

### 7.2 Benefits

- **Direct benefit to subjects**: none (retrospective).
- **Indirect benefit**: validated AI deployment at [INSTITUTION] could reduce echo read turnaround by 20-40%, improving clinical efficiency.

## 8. Data and Safety Monitoring

- All AI predictions are stored with audit logs (timestamp, model version, input hash)
- Any data breach will be reported to [INSTITUTION] Privacy Office within 24 hours per HIPAA
- Study team will meet monthly to review accrual and any adverse events

## 9. Privacy and Confidentiality

- All data stored on [INSTITUTION]-managed, access-controlled servers
- Encryption at rest (AES-256) and in transit (TLS 1.3)
- Access limited to listed study team
- Data destroyed 5 years after study closure per [INSTITUTION] policy

## 10. Data Use Agreement

A Data Use Agreement between [INSTITUTION] and the cardio-echo-suite development team is attached as Appendix A. Key terms:
- Data is shared for research purposes only
- Data is not redistributed to third parties
- Results are reported in aggregate, not per-patient
- Source code is shared openly under MIT license; model weights retain upstream licenses

## 11. References

1. Ouyang D, et al. Video-based AI for beat-to-beat assessment of cardiac function. Nature. 2020;580:252-256.
2. Holste G, et al. Complete AI-Enabled Echocardiography Interpretation with Multitask Deep Learning. JAMA. 2025.
3. McKeen SR, et al. ECG-FM: A Foundation Model for ECG Analysis. 2024.
4. Isensee F, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods. 2021;18:203-211.

---

## Appendices

- **Appendix A**: Data Use Agreement template
- **Appendix B**: Informed Consent waiver request (retrospective de-identified data — typically IRB grants waiver)
- **Appendix C**: Software Bill of Materials (SBOM)
- **Appendix D**: Validation script (in cardio-echo-suite repo: `scripts/validate.py`)
