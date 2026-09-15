# Clinical Evaluation Plan — cardio-echo-suite

> **Standard**: FDA MEDDEV 2.7/1 rev 4 + FDA "Clinical Evaluation of AI/ML-Based SaMD" (2024 draft guidance)
> **Device**: cardio-echo-suite v0.5.0
> **Sponsor**: [SPONSOR NAME]

---

## 1. Clinical evaluation strategy

### 1.1 Pathway

We propose a **three-armed** clinical evaluation:

1. **Literature review** — leverage published validation of upstream academic models
2. **Retrospective single-site validation** — at [INSTITUTION NAME] with 300 cases
3. **Multi-reader study** — 3 cardiologists independently review subset to measure inter-rater agreement

### 1.2 Justification

The 6 AI models in cardio-echo-suite are based on peer-reviewed academic publications:
- EchoNet-Dynamic (Nature 2020) — published with 10k patient internal validation
- PanEcho (JAMA 2025) — published with 200k patient external validation
- ECG-FM (2024) — published with PTB-XL + MIMIC validation
- MedSAM2 (2024) — published with multi-organ validation
- nnU-Net (Nature Methods 2021) — published with ACDC challenge win
- NeuroKit2 — algorithmic library, no ML to validate

The cardio-echo-suite does NOT modify the upstream models — it adds deployment infrastructure. Therefore, the clinical evaluation focuses on:
- Confirming the academic accuracy holds on our patient population
- Validating the integrated deployment (no degradation from integration)
- Establishing the clinical workflow impact

---

## 2. Literature review

### 2.1 Sources

| Database | Query | Filters |
|---|---|---|
| PubMed | "EchoNet-Dynamic" OR "PanEcho" OR "ECG-FM" OR "MedSAM2" OR "nnU-Net" | 2020-2026, English, human |
| Google Scholar | Same | First 100 results |
| medRxiv | Same | 2024-2026 (preprints) |

### 2.2 Inclusion criteria

- Original research (not reviews)
- Reports at least one accuracy metric (sensitivity, specificity, AUROC, MAE, Dice)
- Sample size ≥ 100 patients OR external validation cohort

### 2.3 Data extraction

For each study:
- Model evaluated
- Patient population (age, sex, race if reported)
- Sample size
- Reference standard
- Accuracy metrics with 95% CI
- Subgroup analyses (if any)
- Failure modes reported

### 2.4 Synthesis

The literature review will be summarized in tabular form, with meta-analysis where heterogeneity allows (I² < 50%).

---

## 3. Retrospective validation study

(See `docs/irb/PROTOCOL_TEMPLATE.md` for full protocol.)

### 3.1 Study design

Retrospective, single-site, with 300 cases:
- 200 echo studies (EchoNet-Dynamic + PanEcho validation)
- 100 ECG studies (ECG-FM validation)

### 3.2 Endpoints

**Primary**:
- EchoNet-Dynamic: MAE between AI EF and cardiologist EF
- PanEcho: per-task AUROC (39 tasks)
- ECG-FM: per-arrhythmia AUROC (top 10 most common)

**Secondary**:
- Subgroup accuracy (age, sex, BMI, race, scanner vendor)
- Calibration (Brier score, reliability diagram)
- Inference latency
- Failure mode categorization

### 3.3 Sample size justification

For EchoNet-Dynamic EF MAE:
- Hypothesis: MAE ≤ 6.0 EF%
- Literature reference: Stanford internal MAE = 4.1 (Ouyang 2020)
- SD assumed: 5.0 EF%
- α = 0.05 (two-sided), β = 0.20 (power 80%)
- Required n: 44 patients
- We will enroll 200 (4.5× required) to allow subgroup analysis

For ECG-FM AUROC:
- Hypothesis: AUROC ≥ 0.85
- H0: AUROC = 0.80, H1: AUROC = 0.85
- α = 0.05 (two-sided), β = 0.20
- Required n: 71 patients per arrhythmia
- For 10 arrhythmias: need ~100 ECGs with sufficient prevalence per arrhythmia

### 3.4 Reference standard

- **Echo**: Final signed cardiologist report (extracted from EHR via NLP)
- **ECG**: Final signed cardiologist interpretation + adjudication by 2nd cardiologist for disagreement

### 3.5 Statistical analysis

- Descriptive statistics: mean, SD, median, IQR
- Continuous outcomes: paired t-test or Wilcoxon signed-rank
- Categorical outcomes: AUROC with 95% CI (Wilson method)
- Subgroup analysis: ANOVA or Kruskal-Wallis
- Calibration: Brier score + Hosmer-Lemeshow test
- Significance: α = 0.05 (two-sided), Bonferroni correction for multiple comparisons

---

## 4. Multi-reader study

### 4.1 Design

Subset of 50 echo studies + 50 ECGs from the retrospective validation set, reviewed by 3 cardiologists independently:
- Reader 1: attending cardiologist (≥5 years experience)
- Reader 2: attending cardiologist (≥5 years experience)
- Reader 3: cardiology fellow (PGY 4-6)

### 4.2 Endpoints

- Inter-rater agreement (Cohen's κ for categorical, ICC for continuous)
- AI-vs-reader agreement (Cohen's κ / ICC vs each reader)
- AI-vs-consensus agreement (consensus = majority vote of 3 readers)

### 4.3 Justification

To demonstrate that AI performance is comparable to inter-reader variability — i.e., "the AI agrees with cardiologists as well as cardiologists agree with each other."

---

## 5. Acceptance criteria

| Endpoint | Acceptance | Failure |
|---|---|---|
| EchoNet-Dynamic EF MAE | ≤ 6.0 EF% | > 8.0 EF% |
| PanEcho AUROC (per task) | ≥ 0.80 for ≥30/39 tasks | < 0.70 for >5 tasks |
| ECG-FM AUROC (top 10 arrhythmias) | ≥ 0.85 mean | < 0.80 mean |
| Calibration Brier score | ≤ 0.15 | > 0.25 |
| Subgroup equity | No subgroup >10% below overall | Any subgroup >20% below |
| Inter-reader agreement (κ) | AI-reader ≥ reader-reader - 0.10 | AI-reader < reader-reader - 0.20 |

If a metric falls in the "failure" range, the device is NOT submitted for that indication.

---

## 6. Timeline

| Phase | Activity | Duration |
|---|---|---|
| 1 | IRB approval | 8 weeks |
| 2 | Honest broker data extraction + de-identification | 4 weeks |
| 3 | Cardiologist label extraction + adjudication | 8 weeks |
| 4 | AI inference (scripts/validate.py) | 2 weeks |
| 5 | Subgroup analysis (scripts/subgroup_analysis.py) | 2 weeks |
| 6 | Multi-reader study | 6 weeks |
| 7 | Failure mode review | 2 weeks |
| 8 | Report writing | 4 weeks |
| **Total** | | **36 weeks (~9 months)** |

---

## 7. Clinical evaluation report

The final Clinical Evaluation Report (CER) will be written after study completion and will include:
1. Executive summary
2. Device description
3. Literature review summary
4. Retrospective validation study results
5. Multi-reader study results
6. Subgroup analysis
7. Failure mode analysis
8. Benefit-risk assessment
9. Conformance to acceptance criteria
10. Recommendations for IFU labeling

The CER will be signed by the Clinical Lead and submitted as part of the 510(k) package.

---

## References

1. FDA. "Clinical Evaluation of AI/ML-Based SaMD" (Draft guidance, 2024).
2. FDA. "Software as a Medical Device (SaMD): Clinical Evaluation" (Final guidance, 2017).
3. MEDDEV 2.7/1 rev 4. "Guidelines on Medical Devices — Clinical Evaluation" (2016).
4. Ouyang D et al. Nature. 2020;580:252-256.
5. Holste G et al. JAMA. 2025.
