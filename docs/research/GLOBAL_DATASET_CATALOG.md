# Global Cardiac AI Dataset Catalog

> **Purpose**: Comprehensive catalog of public cardiac datasets for training, validating, and stress-testing cardio-echo-suite models.
>
> **Strategic value**: Each dataset tests a different axis of generalization (vendor, demographic, geography, modality). Using multiple datasets gives us a real "commercial-grade" validation story.

---

## Tier 1: Echo datasets (for EchoNet-Dynamic + PanEcho validation)

### 1. EchoNet-Dynamic (Stanford) ✅ HAVE
- **Size**: 10,030 echo videos
- **Labels**: EF, EDV, ESV, LV segmentation traces
- **Vendor**: GE Vivid (mostly)
- **Demographic**: US academic center, ~60% white
- **License**: Stanford DUA, non-commercial research
- **Status**: ✅ Downloaded and running
- **Used for**: In-distribution test (circular — model was trained here)

### 2. EchoNet-LVH (Stanford) 🆕 EXTERNAL
- **Size**: 9,746 echo videos
- **Labels**: LV hypertrophy (LV mass, LVMI, RWT)
- **Vendor**: GE Vivid (same as EchoNet-Dynamic)
- **Demographic**: US, focus on LVH patients
- **License**: Stanford DUA, non-commercial research
- **Why valuable**: Tests LVH detection (PanEcho task #3). Different label distribution than EchoNet-Dynamic.
- **URL**: https://echonet.github.io/lvh/
- **External?**: ❌ Same distribution as EchoNet-Dynamic (same hospital, same scanner)

### 3. EchoNet-Pediatric (Stanford) 🆕 EXTERNAL
- **Size**: 4,108 echo videos
- **Labels**: EF, LV dimensions
- **Vendor**: GE Vivid
- **Demographic**: Pediatric (0-18 years), US
- **License**: Stanford DUA, non-commercial research
- **Why valuable**: Tests pediatric generalization (our IFU excludes peds, but worth knowing)
- **URL**: https://echonet.github.io/pediatric/
- **External?**: ✅ Different patient population (pediatric vs adult)

### 4. TMED-2 (Tutorial Medical Echo Dataset) 🆕 EXTERNAL
- **Size**: 2,729 echo videos
- **Labels**: View labels (A4C, A2C, PLAX, PSAX), EF
- **Vendor**: Mixed (unspecified)
- **Demographic**: UK, mixed
- **License**: Research, requires data use agreement
- **Why valuable**: Tests view classification + EF on UK data
- **URL**: https://github.com/jamesmf215TMED-2
- **External?**: ✅ Different country, different vendor mix

### 5. CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) 🆕 EXTERNAL
- **Size**: 500 echo studies (2D+T, A2C + A4C)
- **Labels**: LV/RV/atria segmentation, EF, EDV, ESV
- **Vendor**: GE Vivid 7
- **Demographic**: French, mixed clinical
- **License**: Public research (Creative Commons)
- **Why valuable**: GOLD for external validation — French population, manual segmentation by experts
- **URL**: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00bc
- **External?**: ✅ Different country (France), different demographic

### 6. York University Echo Dataset 🆕 EXTERNAL
- **Size**: 150 echo videos
- **Labels**: EF, view labels
- **Vendor**: Philips
- **Demographic**: Canadian
- **License**: Research
- **Why valuable**: Tests Philips scanner (EchoNet is GE-only)
- **External?**: ✅ Different vendor (Philips), different country (Canada)

---

## Tier 2: ECG datasets (for ECG-FM + NeuroKit2 validation)

### 7. PTB-XL ✅ HAVE
- **Size**: 21,837 ECGs, 12-lead, 10s
- **Labels**: 71 SCP-ECG codes (rhythm, form, infarction)
- **Vendor**: Schiller FX-1234
- **Demographic**: German, mixed age/sex
- **License**: PhysioNet Credentialed
- **Status**: ✅ Downloaded 50 ECGs, running shakedown
- **Used for**: In-distribution test

### 8. MIMIC-IV-ECG 🆕 LARGE EXTERNAL
- **Size**: 800,000+ ECGs (the biggest public ECG dataset)
- **Labels**: Linked to MIMIC-IV clinical data (diagnoses, labs, outcomes)
- **Vendor**: Philips
- **Demographic**: US (Beth Israel, Boston), mixed
- **License**: PhysioNet Credentialed (requires CITI + DUA)
- **Why valuable**: Largest ECG dataset. Different vendor (Philips vs Schiller). Linked outcomes for prognostic studies.
- **URL**: https://physionet.org/content/mimic-iv-ecg/1.1/
- **External?**: ✅ Different vendor, different country, 35× larger

### 9. CPSC2018 (China Physiological Signal Challenge) 🆕 EXTERNAL
- **Size**: 6,877 ECGs, 12-lead, 6-60s
- **Labels**: 9 arrhythmia classes (AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE)
- **Vendor**: Unspecified (Chinese hospitals)
- **Demographic**: Chinese, mixed
- **License**: Public research (free download)
- **Why valuable**: Tests Asian demographic generalization + arrhythmia detection
- **URL**: http://2018.icbeb.org/Challenge.html
- **External?**: ✅ Different country (China), different demographic, different vendor

### 10. Georgia 12-lead ECG 🆕 EXTERNAL
- **Size**: 2,000 ECGs, 12-lead, 10s
- **Labels**: Same 9 classes as CPSC2018
- **Vendor**: Unspecified
- **Demographic**: US (Georgia), mixed
- **License**: Public research (PhysioNet)
- **Why valuable**: US-based, smaller but free
- **URL**: https://physionet.org/content/ecg-arrhythmia/1.0.0/
- **External?**: ✅ Different country (US South), different demographic

### 11. Chapman-Shaoxing 12-lead ECG 🆕 EXTERNAL
- **Size**: 10,646 ECGs
- **Labels**: 11 rhythm classes
- **Vendor**: Unspecified
- **Demographic**: US + Chinese
- **License**: Public research
- **Why valuable**: Large, multi-site
- **URL**: https://figshare.com/collections/ChapmanECG/4560497
- **External?**: ✅ Multi-site, multi-demographic

### 12. SPH (Saint Philippe Hospital) 🆕 EXTERNAL
- **Size**: 25,770 ECGs
- **Labels**: Multi-label arrhythmia
- **Vendor**: Unspecified
- **Demographic**: Brazilian
- **License**: Public research
- **Why valuable**: South American demographic
- **External?**: ✅ Different continent, different demographic

### 13. CODE-15% (Brazilian Cohort) 🆕 EXTERNAL
- **Size**: 15,000 ECGs (15% of 2.3M total — the full dataset is restricted)
- **Labels**: Multiple arrhythmias + demographic
- **Vendor**: Unspecified
- **Demographic**: Brazilian, multi-ethnic
- **License**: Public research (requires registration)
- **Why valuable**: Largest public ECG cohort after MIMIC. Diverse demographics.
- **URL**: https://code.15.stats.fmh.usp.br/
- **External?**: ✅ Different continent, different demographic, very large

### 14. ICBEB2018 🆕 EXTERNAL
- **Size**: 1,000 ECGs (test set from CPSC challenge)
- **Labels**: 9 arrhythmia classes
- **Vendor**: Unspecified
- **Demographic**: International
- **License**: Public
- **External?**: ✅ International test set

---

## Tier 3: Cardiac MRI datasets (for nnU-Net CMR validation)

### 15. ACDC (Automated Cardiac Diagnosis Challenge) 🆕 HAVE ACCESS
- **Size**: 100 patients, CMR short-axis stacks
- **Labels**: RV, myocardium, LV segmentation (ED + ES)
- **Vendor**: Siemens
- **Demographic**: French, mixed pathologies (DCM, HCM, HFpEF, HFrEF, MINF, ARV, normal)
- **License**: Research
- **Status**: Can download freely
- **URL**: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00bc

### 16. M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease) 🆕 EXTERNAL
- **Size**: 345 patients, CMR short-axis
- **Labels**: LV/RV/myo segmentation, pathology
- **Vendor**: Siemens, GE, Philips, Canon (4 vendors!)
- **Demographic**: European, multi-site (3 clinical centres in Spain)
- **License**: Research
- **Why valuable**: GOLD for multi-vendor validation — tests scanner generalization
- **URL**: https://www.ub.edu/mnms/
- **External?**: ✅ Multi-vendor (critical for real-world deployment)

### 17. LVQuan 🆕 EXTERNAL
- **Size**: 145 patients, CMR short-axis
- **Labels**: LV volume, mass, EF
- **Vendor**: Siemens
- **Demographic**: Korean
- **License**: Research
- **Why valuable**: Asian demographic
- **External?**: ✅ Different country (Korea)

### 18. York University CMR 🆕 EXTERNAL
- **Size**: 33 patients
- **Labels**: LV/RV segmentation
- **Vendor**: Siemens
- **Demographic**: Canadian
- **License**: Research
- **External?**: ✅ Different country (Canada)

---

## Tier 4: Chest X-ray datasets (for related cardiac findings)

### 19. CheXpert (Stanford)
- **Size**: 224,316 chest X-rays
- **Labels**: 14 thoracic findings (cardiomegaly, edema, consolidation, etc.)
- **Vendor**: Stanford hospital
- **License**: Stanford DUA, research
- **Why relevant**: Cardiomegaly detection is cardiac-adjacent. Tests generalization of imaging AI.
- **URL**: https://stanfordmlgroup.github.io/competitions/chexpert/

### 20. NIH ChestX-ray14
- **Size**: 112,120 chest X-rays
- **Labels**: 14 disease classes
- **Vendor**: Mixed (NIH)
- **License**: Public
- **URL**: https://nihcc.app.box.com/v/ChestXray-NIHCC

### 21. MIMIC-CXR
- **Size**: 377,110 chest X-rays
- **Labels**: 14 findings, linked to MIMIC-IV
- **Vendor**: Philips
- **License**: PhysioNet Credentialed
- **URL**: https://physionet.org/content/mimic-cxr/2.0.0/

### 22. PadChest (Spain)
- **Size**: 160,868 chest X-rays
- **Labels**: 174 findings
- **Vendor**: Unspecified
- **Demographic**: Spanish
- **License**: Public
- **External?**: ✅ Different country, different language
- **URL**: https://bimcv.cipf.es/bimcv-projects/padchest/

### 23. VinDr-CXR (Vietnam)
- **Size**: 18,000 chest X-rays
- **Labels**: 28 findings
- **Vendor**: Mixed
- **Demographic**: Vietnamese
- **License**: Public
- **External?**: ✅ Asian demographic
- **URL**: https://vindr.ai/datasets/cxr

---

## Tier 5: Multi-modal / linked datasets

### 24. UK Biobank 🆕 GOLD STANDARD
- **Size**: 500,000 participants
- **Modalities**: CMR, echo, ECG, genomics, clinical, outcomes (10+ year follow-up)
- **Labels**: Comprehensive cardiovascular phenotyping
- **License**: Requires application + payment (£3,000-9,000 for research access)
- **Why valuable**: The gold standard. If you can get access, this is the dataset for a Nature/NEJM paper.
- **URL**: https://www.ukbiobank.ac.uk/
- **External?**: ✅ UK population, multi-modal, longitudinal

### 25. MIMIC-IV (linked)
- **Size**: 257,366 patients
- **Modalities**: Clinical (ICU), labs, medications, outcomes, ECG, CXR
- **License**: PhysioNet Credentialed
- **Why valuable**: Linked outcomes — can test prognostic AI, not just diagnostic
- **URL**: https://physionet.org/content/mimiciv/2.2/

---

## Strategic recommendation — which datasets to use

### For paper #1 (research shakedown, 2-4 weeks)
**Goal**: Show cardio-echo-suite works on standard benchmarks

| Dataset | Why | Effort |
|---|---|---|
| EchoNet-Dynamic (test) | Standard echo benchmark | ✅ Done |
| PTB-XL (test) | Standard ECG benchmark | ✅ Done |
| ACDC (test) | Standard CMR benchmark | 1 day |

### For paper #2 (external validation, 1-2 months)
**Goal**: Show generalization across sites, vendors, demographics

| Dataset | Why | Effort |
|---|---|---|
| CAMUS (France) | External echo, French | 2 days |
| CPSC2018 (China) | External ECG, Chinese | 2 days |
| M&Ms (multi-vendor CMR) | Multi-vendor CMR | 2 days |

### For paper #3 (commercial-grade validation, 3-6 months)
**Goal**: Multi-site, multi-demographic, multi-vendor → commercial-ready

| Dataset | Why | Effort |
|---|---|---|
| MIMIC-IV-ECG (US) | Largest ECG, US academic | 1 week |
| CODE-15% (Brazil) | South American, large | 1 week |
| York Echo (Canada, Philips) | Different vendor | 2 days |
| PadChest (Spain) | Multi-lingual X-ray | 2 days |
| UK Biobank (if affordable) | Gold standard | 3-6 months |

---

## Quick comparison — size vs. effort

| Dataset | Size | Download time | Disk space | Credentialing |
|---|---|---|---|---|
| EchoNet-Dynamic | 10k echo | 6 min | 7 GB | Stanford DUA |
| PTB-XL | 22k ECG | 10 min | 1.7 GB | PhysioNet Credentialed |
| MIMIC-IV-ECG | 800k ECG | 4-6 hours | 50 GB | PhysioNet Credentialed |
| CPSC2018 | 7k ECG | 5 min | 1.5 GB | Free (no credentialing) |
| CODE-15% | 15k ECG | 30 min | 3 GB | Registration required |
| CAMUS | 500 echo | 5 min | 2 GB | Free |
| ACDC | 100 CMR | 5 min | 2 GB | Free |
| M&Ms | 345 CMR | 10 min | 5 GB | Free |
| CheXpert | 224k CXR | 2-4 hours | 140 GB | Stanford DUA |
| UK Biobank | 500k patients | N/A | N/A | £3-9k application |

---

## How to use multiple datasets for model improvement

### Strategy 1: External validation (the right way)
Run the SAME trained model on multiple external test sets. Report:
- MAE per dataset
- Subgroup analysis
- Generalization gap (internal vs external)

This is what FDA wants. Each external dataset = one "site" in a multi-site study.

### Strategy 2: Ensemble (combining models)
If we have multiple models (PanEcho + EchoNet-Dynamic + EchoNet-LVH), we can:
1. Run all models on the same input
2. Average their predictions (simple ensemble)
3. Use a meta-learner to weight them (stacked ensemble)
4. Expected improvement: 1-2 EF% MAE reduction

### Strategy 3: Fine-tuning (improving the model)
Use external datasets to fine-tune the model on new populations:
1. CPSC2018 (Chinese ECG) → fine-tune ECG-FM for Asian demographics
2. CAMUS (French echo) → fine-tune PanEcho for European demographics
3. Use LoRA (low-rank adaptation) — only train a small adapter, keep base frozen

### Strategy 4: Test-time augmentation
For each input, generate multiple augmented versions (flip, rotate, scale), run the model on each, and average. Typically reduces MAE by 0.5-1 EF%.

### Strategy 5: Domain adaptation
Train a small "domain adapter" network that maps features from one hospital's distribution to another. This is what Caption Health did for their FDA-cleared echo AI.

---

## Honest assessment — what's actually achievable

| Goal | Timeline | Datasets needed |
|---|---|---|
| **Publish 1 paper (research)** | 1-2 months | EchoNet-Dynamic + PTB-XL (we have these) |
| **Publish 2 papers (external validation)** | 3-4 months | + CAMUS + CPSC2018 + ACDC |
| **Reach commercial standard (MAE <5)** | 6-12 months | + MIMIC-IV-ECG + M&Ms + fine-tuning |
| **FDA submission-ready** | 12-18 months | + UK Biobank or local hospital data (300+ cases) |

---

## Immediate next steps (this week)

1. **Download CAMUS** (free, 5 min, 2 GB) — best external echo validation
2. **Download CPSC2018** (free, 5 min, 1.5 GB) — best external ECG validation
3. **Download ACDC** (free, 5 min, 2 GB) — for CMR segmentation testing
4. **Run cardio-echo-suite on all three** — proves external generalization
5. **Write paper #2: "Multi-site external validation of cardio-echo-suite"**

These three datasets are free, fast to download, and give us a real "multi-site" story for the paper.
