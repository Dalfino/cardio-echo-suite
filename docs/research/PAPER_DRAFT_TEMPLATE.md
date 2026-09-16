# Paper Draft Template — cardio-echo-suite Evaluation on Public Datasets

> **Target venue**: medRxiv (preprint) → European Heart Journal - Digital Health, JAMA Cardiology, or npj Digital Medicine
>
> **How to use**: Fill in [BRACKETS] with your actual results. Delete sections that don't apply. Aim for 3,000-5,000 words.

---

## Title

**cardio-echo-suite: An Open-Source Multi-Modal Cardiac AI Suite — Evaluation on Public Echocardiography and ECG Datasets**

## Authors

[YOUR NAME]^[1]^, [CO-AUTHOR 1]^[1]^, [CARDIOLOGIST CHAMPION, MD]^[2]^

^1^[Your department/institution]
^2^[Cardiology department, your hospital]

## Corresponding author

[Your Name]
[Email]
[Address]

## Abstract

**Background**: [1-2 sentences on the problem — e.g., "Cardiovascular disease is the leading cause of death globally. AI-assisted interpretation of echocardiograms and ECGs could reduce cardiologist workload, but most published models are not integrated into deployable systems."]

**Methods**: We evaluated cardio-echo-suite, an open-source cardiac AI suite comprising [N] models for echocardiography (EchoNet-Dynamic, PanEcho) and ECG (ECG-FM) interpretation, on the EchoNet-Dynamic test set (n=[N] echo videos) and PTB-XL test set (n=[N] ECGs). Primary outcomes were [e.g., mean absolute error for EF estimation, AUROC for arrhythmia detection]. Secondary outcomes included [e.g., subgroup analysis by sex and age, calibration, inference latency].

**Results**: [3-4 sentences with key numbers — e.g., "For EF estimation, cardio-echo-suite achieved MAE [X.X] EF% (95% CI [X.X]-[X.X]) on EchoNet-Dynamic test set, with Pearson correlation r=[0.X]. For AFib detection on PTB-XL, AUROC was [0.XX] (95% CI [0.XX]-[0.XX]). Inference latency was [X.X] seconds per study on CPU. Subgroup analysis showed no significant difference between [subgroups]."]

**Conclusions**: cardio-echo-suite achieves [accuracy comparable to / slightly below / exceeding] published academic results on public datasets, with [acceptable / suboptimal] inference latency suitable for [research / clinical advisory] use. Open-source availability enables reproducibility and further development. Local clinical validation is required before deployment.

**Keywords**: cardiac AI, echocardiography, electrocardiography, open-source, FHIR, clinical decision support

---

## 1. Introduction

### 1.1 Background

Cardiovascular disease (CVD) is the leading cause of death globally, accounting for [N] million deaths annually [WHO 2024]. Echocardiography and electrocardiography (ECG) are the two most common cardiac diagnostic tests, with approximately [40 million] echo studies and [100 million] ECGs performed annually in the United States alone [cite].

Cardiologist read times represent a significant bottleneck in cardiac care delivery. Typical echo turnaround time ranges from [4-24 hours], and stat studies in the emergency department or intensive care unit can wait [30-60 minutes] for interpretation [cite]. Artificial intelligence (AI) models that pre-read these studies could reduce turnaround time and allow cardiologists to focus on complex cases.

Multiple AI models for cardiac imaging have been published in peer-reviewed literature. EchoNet-Dynamic [Ouyang 2020], trained on 10,030 echocardiograms at Stanford, estimates left ventricular ejection fraction (EF) with mean absolute error (MAE) of 4.1 EF%. PanEcho [Holste 2025] performs 39 echocardiography interpretation tasks with AUROC ranging from 0.85 to 0.95. ECG-FM [McKeen 2024] detects cardiac arrhythmias from 12-lead ECGs with AUROC 0.7-0.95 across rhythm categories.

### 1.2 The deployment gap

Despite academic progress, these models are released as research artifacts (code + weights) without the infrastructure required for hospital deployment. Specifically, they lack: (1) standardized REST APIs for integration with hospital information systems, (2) FHIR R4 output for electronic health record (EHR) ingestion, (3) DICOM ingestion for PACS connectivity, (4) authentication and audit logging for HIPAA compliance, and (5) a cardiologist review interface for sign-off.

### 1.3 cardio-echo-suite

We developed cardio-echo-suite, an open-source integration layer that wraps [N] academic cardiac AI models with deployment infrastructure. The suite includes:
- [N] model services for echocardiography, ECG, and cardiac MRI
- A shared Python library (cardio-echo-core) providing JWT authentication, HIPAA-compliant audit logging, FHIR R4 resource builders, DICOM utilities, ONNX export with PyTorch fallback, GPTQ quantization with golden-test parity gate, and dynamic model loading
- A DICOM C-STORE gateway for automatic PACS ingestion
- An orchestrator service that routes studies by modality
- A review web UI for cardiologist sign-off

### 1.4 Study objectives

In this study, we evaluated cardio-echo-suite on public datasets to assess:
1. Whether the integrated suite achieves accuracy comparable to published academic results
2. Inference latency on commodity CPU hardware
3. Subgroup performance by available demographics
4. Failure mode characteristics

**Important limitation**: This evaluation uses public datasets that overlap with the training data of the upstream models. Results should not be interpreted as estimates of real-world clinical accuracy. Local clinical validation on the intended use population is required before deployment.

---

## 2. Methods

### 2.1 Study design

Retrospective evaluation on public benchmark datasets. No human subjects were involved; all data was previously de-identified by the original dataset publishers.

### 2.2 Datasets

#### EchoNet-Dynamic

We used the EchoNet-Dynamic test set (n=2,437 apical-4-chamber echo videos) [Ouyang 2020]. Each video has a ground-truth EF label derived from cardiologist measurement. The dataset was originally de-identified per HIPAA Safe Harbor and shared under a non-commercial research data use agreement.

#### PTB-XL

We used the PTB-XL test set (n=[N] 12-lead ECGs) [Wagner 2020]. Each ECG has [N] diagnosis labels across [N] statement codes. Recordings are 10 seconds at 500 Hz (we used the 100 Hz downsampled version for faster processing).

[Add MIMIC-IV-ECG if used]

[Add ACDC if CMR models evaluated]

### 2.3 cardio-echo-suite configuration

We deployed cardio-echo-suite v0.5.0 via Docker Compose on a single machine with [CPU model, e.g., Intel Xeon E5-2680 v4 @ 2.40GHz] and [N] GB RAM. No GPU was used. All services ran with default configuration (dynamic model loading, ONNX export enabled, INT8 quantization disabled for EchoPrime due to weight unavailability).

### 2.4 Evaluation metrics

#### Echocardiography
- **Primary**: Mean absolute error (MAE) between AI-predicted EF and ground-truth EF
- **Secondary**: Pearson correlation coefficient (r), Bland-Altman analysis, [subgroup MAE by sex if available]

#### ECG
- **Primary**: Area under the receiver operating characteristic curve (AUROC) for each arrhythmia category
- **Secondary**: F1 score, calibration (Brier score, reliability diagram), [subgroup AUROC by sex/age]

#### System
- Inference latency (seconds per study)
- Peak memory usage
- Error rate

### 2.5 Statistical analysis

We report point estimates with 95% confidence intervals. For AUROC, we used the Wilson score interval. For MAE, we used bootstrap resampling (10,000 iterations). For subgroup comparisons, we used [ANOVA / Kruskal-Wallis] with Bonferroni correction. All analyses were performed in Python 3.10 using scipy 1.10 and scikit-learn 1.3. Code is available at https://github.com/[USER]/cardio-echo-suite.

### 2.6 Reproducibility

All code, configuration files, and result data are available at:
- Main repository: https://github.com/[USER]/cardio-echo-suite
- This evaluation's scripts: `scripts/public_data_shakedown.py`
- Docker images: `[DOCKER HUB]/cardio-echo-suite:v0.5.0`
- Random seeds: fixed at 42 throughout

---

## 3. Results

### 3.1 Echocardiography — EF estimation

On the EchoNet-Dynamic test set (n=2,437), cardio-echo-suite's EchoNet-Dynamic service achieved:

| Metric | Result | 95% CI |
|---|---|---|
| MAE (EF%) | [X.X] | [X.X] - [X.X] |
| Pearson r | [0.XX] | [0.XX] - [0.XX] |
| Bias (AI - GT) | [X.X] | [X.X] - [X.X] |
| Limits of agreement | [±X.X] | — |

[Include Bland-Altman plot as Figure 1]

Subgroup analysis (where EchoNet-Dynamic provides demographics):

| Subgroup | N | MAE (95% CI) |
|---|---|---|
| Overall | 2,437 | [X.X] ([X.X]-[X.X]) |
| Male | [N] | [X.X] ([X.X]-[X.X]) |
| Female | [N] | [X.X] ([X.X]-[X.X]) |
| Age <65 | [N] | [X.X] ([X.X]-[X.X]) |
| Age ≥65 | [N] | [X.X] ([X.X]-[X.X]) |
| EF <40% | [N] | [X.X] ([X.X]-[X.X]) |
| EF ≥40% | [N] | [X.X] ([X.X]-[X.X]) |

### 3.2 Echocardiography — PanEcho 39-task

On the EchoNet-Dynamic test set (using EF only, since PanEcho's 39 tasks require labels not in EchoNet-Dynamic):

[Report what was evaluated — likely EF only, plus any tasks where you can derive labels from EchoNet-Dynamic's FileList.csv]

### 3.3 ECG — arrhythmia detection

On the PTB-XL test set (n=[N]), cardio-echo-suite's ECG-FM service achieved:

| Arrhythmia | N | AUROC (95% CI) | F1 | Brier |
|---|---|---|---|---|
| Sinus rhythm | [N] | [0.XX] ([0.XX]-[0.XX]) | [0.XX] | [0.XX] |
| Atrial fibrillation | [N] | [0.XX] ([0.XX]-[0.XX]) | [0.XX] | [0.XX] |
| Atrial flutter | [N] | [0.XX] ([0.XX]-[0.XX]) | [0.XX] | [0.XX] |
| [other arrhythmias...] | | | | |

[Include reliability diagram as Figure 2]

Subgroup analysis (PTB-XL has age and sex):

| Subgroup | AFib AUROC | 95% CI |
|---|---|---|
| Overall | [0.XX] | [0.XX]-[0.XX] |
| Male | [0.XX] | [0.XX]-[0.XX] |
| Female | [0.XX] | [0.XX]-[0.XX] |
| Age <65 | [0.XX] | [0.XX]-[0.XX] |
| Age ≥65 | [0.XX] | [0.XX]-[0.XX] |

### 3.4 System performance

| Metric | Result |
|---|---|
| Mean inference latency (echo) | [X.X] seconds |
| Mean inference latency (ECG) | [X.X] seconds |
| Peak memory usage | [X] GB |
| Error rate | [X.X]% |
| Cold start time | [X.X] seconds |

### 3.5 Failure mode analysis

We manually reviewed the [N=20] worst-performing cases per modality. Common failure modes included:
- [Failure mode 1, e.g., "Poor image quality — low signal-to-noise ratio"]
- [Failure mode 2, e.g., "Arrhythmia during acquisition — irregular R-R intervals"]
- [Failure mode 3, e.g., "Unusual anatomy — congenital heart disease"]

[Include 2-3 example failure cases as figures]

---

## 4. Discussion

### 4.1 Principal findings

In this evaluation of cardio-echo-suite on public datasets, we found that [key finding 1 — e.g., "the integrated suite achieved EF estimation accuracy comparable to the original EchoNet-Dynamic publication (MAE [X.X] vs 4.1 EF%)"]. For ECG arrhythmia detection, [key finding 2 — e.g., "ECG-FM achieved AUROC [0.XX] for atrial fibrillation, slightly below the published [0.97] likely due to [reason]"].

### 4.2 Comparison to prior work

Our results are consistent with published evaluations of the underlying models (Table X). The slight degradation we observe ([X.X] vs [X.X] EF% MAE) is likely attributable to [reason — e.g., "differences in preprocessing, our use of CPU-only inference, or our ONNX export with parity threshold"].

### 4.3 Strengths

This study has several strengths. First, we evaluated an integrated deployment system rather than isolated models, providing a realistic estimate of accuracy in a hospital-deployable configuration. Second, all code is open-source, enabling full reproducibility. Third, we report subgroup analysis where demographics are available.

### 4.4 Limitations

This study has important limitations:

1. **Circular evaluation**: EchoNet-Dynamic was used to train the EchoNet-Dynamic model. Our results on its test set therefore reflect in-distribution performance and may overestimate real-world accuracy. The same applies to PTB-XL and ECG-FM.

2. **No external validation**: We did not evaluate on data from a different institution. Local clinical validation on the intended use population is required before any clinical deployment.

3. **Limited demographics**: EchoNet-Dynamic does not include race/ethnicity, BMI, or scanner vendor — critical axes for equity analysis. PTB-XL includes age and sex only.

4. **Single hardware configuration**: We tested on CPU only. GPU performance may differ.

5. **Model weight unavailability**: EchoPrime weights were not publicly available at the time of evaluation, so we could not test that service.

### 4.5 Clinical implications

Our results suggest that cardio-echo-suite is [ready for / not yet ready for] local clinical validation. The accuracy on public datasets is [comparable to / below] published benchmarks, and the inference latency is [acceptable / too slow] for [advisory / real-time] use. Hospitals considering deployment should:
1. Conduct local validation on [≥300] cases
2. Perform subgroup analysis on their patient demographics
3. Establish a cardiologist sign-off workflow before clinical use
4. Comply with local regulatory requirements (FDA in the US, MDA in Malaysia, etc.)

### 4.6 Future work

Future work should include:
- Local validation at [N] hospitals across [N] countries
- Prospective clinical trial with cardiologist time-tracking
- Fine-tuning on institutional data using LoRA adapters
- Comparison to commercial predicate devices (Ultromics, Caption Health)
- Cost-effectiveness analysis

---

## 5. Conclusions

cardio-echo-suite achieves [accuracy comparable to published academic results] on public echocardiography and ECG datasets, with [acceptable] inference latency on commodity CPU hardware. The open-source suite provides a foundation for local clinical validation and deployment. Hospitals and researchers are encouraged to validate the suite on their patient populations before any clinical use.

---

## Data availability

All datasets used in this study are publicly available (EchoNet-Dynamic: https://echonet.github.io/dynamic/; PTB-XL: https://physionet.org/content/ptbxl/1.0.3/). Code is available at https://github.com/[USER]/cardio-echo-suite under the MIT license.

## Code availability

The cardio-echo-suite source code, Docker images, and evaluation scripts are available at https://github.com/[USER]/cardio-echo-suite. The specific version evaluated in this study is tagged as v0.5.0.

## Author contributions

[Your Name]: Conceptualization, Methodology, Software, Investigation, Writing - Original Draft.
[Co-Author]: [Role].
[Cardiologist]: Conceptualization, Writing - Review & Editing, Clinical interpretation.

## Competing interests

The authors declare no competing interests.

## Acknowledgments

We thank the EchoNet-Dynamic team at Stanford University and the PTB-XL team at PhysioNet for making their datasets publicly available. We thank the upstream model authors (Ouyang et al., Holste et al., McKeen et al.) for releasing their model weights.

---

## References

1. Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat assessment of cardiac function. Nature. 2020;580(7802):252-256. doi:10.1038/s41586-020-2145-8
2. Holste G, Oikonomou EK, Tokodi M, et al. Complete AI-Enabled Echocardiography Interpretation with Multitask Deep Learning. JAMA. 2025;[volume(issue):pages]. doi:[DOI]
3. Wagner P, Strodthoff N, Bousseljot R, et al. PTB-XL, a large publicly available electrocardiography dataset. Sci Data. 2020;7(1):154. doi:10.1038/s41597-020-0495-6
4. McKeen SR, et al. [ECG-FM paper citation]
5. Isensee F, Jaeger PF, Kohl SAA, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods. 2021;18(2):203-211.
6. FDA. Artificial Intelligence-Enabled Medical Devices. Guidance for Industry and Food and Drug Administration Staff. 2024.
7. IMDRF. Software as a Medical Device: Clinical Evaluation. 2017.
8. [Add other references as needed]

---

## Figures (placeholders)

**Figure 1**: Bland-Altman plot comparing AI-predicted EF to ground-truth EF on EchoNet-Dynamic test set.

**Figure 2**: Reliability diagram for atrial fibrillation detection on PTB-XL test set, showing calibration of ECG-FM predictions.

**Figure 3**: Subgroup forest plot showing MAE by [sex, age group, EF category].

**Figure 4**: Example failure cases — three echo videos where AI predicted EF >15 EF% off from ground truth, with visual explanation.

**Figure 5**: System architecture diagram of cardio-echo-suite.

## Tables (placeholders)

**Table 1**: Dataset characteristics (EchoNet-Dynamic, PTB-XL).

**Table 2**: Primary outcomes — EF MAE and arrhythmia AUROC.

**Table 3**: Subgroup analysis.

**Table 4**: System performance metrics.

**Table 5**: Comparison to published academic results.

---

## How to use this template

1. **Week 1-2**: Run shakedown scripts on EchoNet-Dynamic + PTB-XL
2. **Week 3**: Fill in [BRACKETS] with your actual numbers
3. **Week 4**: Make figures (use matplotlib, examples in `scripts/`)
4. **Week 5**: Get co-author feedback
5. **Week 6**: Submit to medRxiv as preprint
6. **Week 7-8**: Address feedback, submit to peer-reviewed journal

**Suggested journals** (in order of impact):
- European Heart Journal - Digital Health (IF ~7)
- JAMA Cardiology (IF ~24)
- npj Digital Medicine (IF ~12)
- Circulation: Cardiovascular Quality and Outcomes (IF ~6)
- Journal of the American College of Cardiology: Clinical Electrophysiology (IF ~5)
- medRxiv preprint (no IF, but rapid dissemination)
