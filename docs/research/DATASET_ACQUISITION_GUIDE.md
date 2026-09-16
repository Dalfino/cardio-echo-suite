# Public Dataset Acquisition Guide — cardio-echo-suite Research Phase

> **Purpose**: Step-by-step instructions for legally obtaining and preparing public cardiac datasets for research validation of cardio-echo-suite.
>
> **License reminder**: All datasets here are RESEARCH-ONLY. You cannot use them for commercial products. You cannot use them to make clinical decisions on real patients. You CAN use them for papers, conferences, and portfolio building.

---

## Datasets covered

| Dataset | Modality | Size | License | Time to obtain |
|---|---|---|---|---|
| EchoNet-Dynamic | Echo (TTE) | 10,030 videos (~6 GB) | Stanford DUA, non-commercial research | 1-3 days |
| PTB-XL | ECG (12-lead) | 21,837 ECGs (~3 GB) | PhysioNet Credentialed DUA | 3-7 days |
| MIMIC-IV-ECG | ECG (12-lead) | 800,000+ ECGs (~50 GB) | PhysioNet Credentialed DUA | 3-7 days |
| ACDC | Cardiac MRI | 100 patients (~2 GB) | Research-only | 1 day |
| M&Ms | Cardiac MRI | 345 patients (~5 GB) | Research-only | 1-3 days |
| CheXpert | Chest X-ray | 224,316 images (~140 GB) | Stanford DUA, research | 1-3 days (optional, not cardiac-specific) |

**Start with EchoNet-Dynamic + PTB-XL. Those cover 2 of our 7 models and are the fastest to obtain.**

---

## Step 0: Get CITI Human-Subjects Training (REQUIRED for PhysioNet)

Before you can download PTB-XL or MIMIC-IV, you need a CITI training certificate. This is free and takes ~6 hours.

1. Go to https://about.citiprogram.org/
2. Click "Register" → select "I am affiliated with an institution" (or "Not affiliated" if your hospital isn't listed)
3. Select your institution OR choose "Independent"
4. Under "Learner Group", select "Human Research — Biomedical Investigators"
5. Complete all required modules (typically 16 modules, ~6 hours total)
6. Save your completion certificate (PDF) — you'll need this for PhysioNet

**Modules to complete**:
- History and ethics
- Belmont Report
- Informed consent
- Risk/benefit analysis
- Privacy and confidentiality
- Vulnerable populations
- Data safety monitoring

**Certificate validity**: 3 years. You'll need to retake if it expires.

---

## Dataset 1: EchoNet-Dynamic (Echo)

### What you get
- 10,030 apical-4-chamber echo videos (AVI format, de-identified)
- Labels: EF (%) for each video
- Split: Train (7,465), Val (128), Test (2,437)

### Why it's useful
- Trains/validates EchoNet-Dynamic model (our service)
- Test set can be used as research validation (NOT for FDA)
- Published in Nature 2020 — high credibility

### How to get it

1. Go to https://echonet.github.io/dynamic/
2. Click "Dataset" → read the data use agreement
3. Email the EchoNet team (echonet@stanford.edu) with:
   - Your name + affiliation
   - Your research purpose (1 paragraph)
   - Your CITI training certificate
   - Agreement to the data use terms
4. They typically respond within 1-3 business days with a download link
5. Download the ZIP file (~6 GB)

### What to do once you have it

```bash
# Extract
mkdir -p /data/public/echonet-dynamic
cd /data/public/echonet-dynamic
unzip EchoNet-Dynamic.zip

# Expected structure:
# /data/public/echonet-dynamic/
# ├── Videos/                    # 10,030 AVI files
# │   ├── 0X10A28877E97DF540.avi
# │   ├── 0X129133A90A61A59D.avi
# │   └── ...
# ├── FileList.csv               # Labels (FileName, Split, EF, ...)
# └── VolumeTracings.csv         # LV segmentation traces

# Verify
ls Videos/ | wc -l               # Should be 10030
head -5 FileList.csv
```

### Sample FileList.csv

```csv
FileName,Split,EF,EDV,ESV,FrameTime,NumberOfFrames,Height,Width
0X10A28877E97DF540,TRAIN,55.0,122.5,55.1,33.333333333333336,60,616,584
0X129133A90A61A59D,VAL,68.0,148.2,47.4,33.333333333333336,60,528,528
0X132C1E8DBB715D1D,TEST,32.0,185.6,126.2,33.333333333333336,60,616,584
...
```

### Run shakedown

```bash
python scripts/public_data_shakedown.py \
    --dataset echonet-dynamic \
    --data-dir /data/public/echonet-dynamic/Videos \
    --labels /data/public/echonet-dynamic/FileList.csv \
    --orchestrator-url http://localhost:8080 \
    --output /tmp/echo_shakedown.json \
    --max-n 50 \
    --i-understand-this-is-not-validation
```

---

## Dataset 2: PTB-XL (ECG)

### What you get
- 21,837 12-lead ECGs, 10 seconds each, 500 Hz and 100 Hz versions
- 5 unique diagnoses per ECG (avg), 71 different statement codes
- Train/val/test splits provided

### Why it's useful
- Validates ECG-FM model
- Has demographics (age, sex) for subgroup analysis
- Published in Nature Scientific Data 2020

### How to get it

**Prerequisite**: PhysioNet credentialed access (see Step 0 above + below)

1. Go to https://physionet.org/register/
2. Create an account
3. Under "Credentialing", apply for "Credentialed" level
4. Upload:
   - CITI training certificate
   - Institutional affiliation letter (or supervisor confirmation)
   - Research project description (1 paragraph: "Validation of open-source cardiac AI suite")
5. Wait 3-7 business days for review
6. Once approved, go to https://physionet.org/content/ptbxl/1.0.3/
7. Click "Download" → select "zip" (full dataset) or use wget:

```bash
# Download PTB-XL (requires PhysioNet login)
wget --user=YOUR_USERNAME --password=YOUR_PASSWORD \
    https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip

# Or use the PhysioNet API
pip install wfdb
python -c "
import wfdb
wfdb.dl_database('ptb-xl', './ptbxl', ['100/'])
"
```

### What to do once you have it

```bash
mkdir -p /data/public/ptbxl
cd /data/public/ptbxl
unzip ptb-xl-*.zip

# Expected structure:
# /data/public/ptbxl/
# ├── ptbxl_database.csv          # Labels (ecg_id, patient_id, filename, rhythms, forms, ...)
# ├── records100/                 # 100 Hz version (smaller, faster for research)
# │   ├── 00000/
# │   │   ├── 00001_lr.dat
# │   │   ├── 00001_lr.hea
# │   │   └── ...
# │   ├── 01000/
# │   └── ...
# ├── records500/                 # 500 Hz version (higher quality)
# │   └── ...
# └── scp_statements.csv          # Diagnosis code definitions
```

### Sample ptbxl_database.csv

```csv
ecg_id,patient_id,filename,filename_lr,filename_hr,nurse,site,device,recording_date,report_date,investigator,heart_rate,patient_age,sex,height,weight,rhythms,forms,infarction_stadium1,...
1,0,records100/00000/00001_lr,records500/00000/00001_hr,,"1", site 1, ECG-5500,2020-10-01 12:00:00,2020-10-01 12:05:00,"Investigator 1",60,62,1,160.0,80.0,"['SR']","['SR']",Unknown,...
2,0,records100/00000/00002_lr,records500/00000/00002_hr,,"1", site 1, ECG-5500,2020-10-01 12:00:00,2020-10-01 12:05:00,"Investigator 1",75,58,0,165.0,75.0,"['SR']",['SR'],Unknown,...
```

### Run shakedown

```bash
python scripts/public_data_shakedown.py \
    --dataset ptbxl \
    --data-dir /data/public/ptbxl/records100 \
    --labels /data/public/ptbxl/ptbxl_database.csv \
    --orchestrator-url http://localhost:8080 \
    --output /tmp/ecg_shakedown.json \
    --max-n 50 \
    --i-understand-this-is-not-validation
```

---

## Dataset 3: MIMIC-IV-ECG (large-scale ECG)

### What you get
- ~800,000 12-lead ECGs from Beth Israel Deaconess Medical Center
- Linked to MIMIC-IV clinical data (diagnoses, labs, outcomes)
- 500 Hz, 10 seconds

### Why it's useful
- 35× larger than PTB-XL — better for rare arrhythmia validation
- Linked clinical outcomes (death, ICU admission, etc.)
- Different patient population than PTB-XL (US academic center)

### How to get it

**Prerequisite**: PhysioNet credentialed access (same as PTB-XL)

1. Go to https://physionet.org/content/mimic-iv-ecg/1.1/
2. Complete the MIMIC training (additional CITI module on data privacy)
3. Sign the MIMIC Data Use Agreement
4. Download via wget:

```bash
wget --user=YOUR_USERNAME --password=YOUR_PASSWORD \
    https://physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiographic-matched-subset-of-mimic-iv-1.1.zip
```

**Disk space**: ~50 GB. Make sure you have room.

### Sample usage

Same as PTB-XL — use the shakedown script. Just update `--dataset ptbxl` to point at MIMIC format (you'll need to convert the labels CSV format).

---

## Dataset 4: ACDC (Cardiac MRI segmentation)

### What you get
- 100 patients, cardiac MRI short-axis stacks
- Manual segmentations: RV, myocardium, LV (end-diastole + end-systole)
- Clinical labels: pathologies (DCM, HCM, HFpEF, HFrEF, MINF, ARV, normal)

### Why it's useful
- Standard benchmark for CMR segmentation (the "MNIST of cardiac MRI")
- Validates nnU-Net CMR service
- Published as a MICCAI 2017 challenge

### How to get it

1. Go to https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00bc
2. Register (free, instant approval for research)
3. Download the ACDC dataset ZIP (~2 GB)

### Structure

```
acdc/
├── training/
│   ├── patient001/
│   │   ├── patient001_4d.nii.gz           # 4D cine
│   │   ├── patient001_frame01.nii.gz      # ED frame
│   │   ├── patient001_frame01_gt.nii.gz   # ED segmentation
│   │   ├── patient001_frame12.nii.gz      # ES frame
│   │   ├── patient001_frame12_gt.nii.gz   # ES segmentation
│   │   └── Info.cfg
│   └── ... (patient100)
└── testing/
    └── patient101/ ... patient150/  (no GT for test)
```

### Convert to cardio-echo-suite format

```bash
# ACDC is already in NIfTI — our nnunet-cmr service reads this directly
# Just point the service at the NIfTI file:
curl -X POST "http://localhost:8006/predict?patient_id=ACDC-001&dry_run=true" \
    -F "file=@/data/public/acdc/training/patient001/patient001_frame01.nii.gz"
```

---

## Dataset 5: M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease CMR)

### What you get
- 345 patients from 4 vendors (Siemens, GE, Philips, Canon)
- CMR short-axis stacks with segmentations
- Multi-disease: HCM, HFpEF, HFrEF, DCM, ARV, MINF, normal

### Why it's useful
- Tests scanner-vendor robustness (critical for dataset shift analysis)
- Larger than ACDC
- Published as MICCAI 2020 challenge

### How to get it

1. Go to https://www.ub.edu/mnms/
2. Register
3. Download M&Ms Training Set (345 patients, ~5 GB)

---

## Legal / license quick reference

| Dataset | Commercial use? | redistribute? | Clinical use? | Research papers? |
|---|---|---|---|---|
| EchoNet-Dynamic | ❌ | ❌ | ❌ | ✅ (cite Ouyang 2020) |
| PTB-XL | ❌ | ❌ | ❌ | ✅ (cite Wagner 2020) |
| MIMIC-IV-ECG | ❌ | ❌ | ❌ | ✅ (cite Johnson 2023) |
| ACDC | ❌ | ❌ | ❌ | ✅ (cite Bernard 2018) |
| M&Ms | ❌ | ❌ | ❌ | ✅ (cite Campello 2021) |

**Rule of thumb**: If you publish a paper, you must cite the original dataset paper. You cannot redistribute the data itself — point readers to the original source.

---

## Common pitfalls

### Pitfall 1: Forgetting to cite the dataset

If you use EchoNet-Dynamic in a paper, you MUST cite:
```
Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat assessment of cardiac function.
Nature. 2020;580(7802):252-256. doi:10.1038/s41586-020-2145-8
```

Forgetting to cite = plagiarism. Always cite.

### Pitfall 2: Mixing test set into training

EchoNet-Dynamic has explicit train/val/test splits. Don't use the test set for anything other than final reporting. If you "tune" on the test set, your numbers are meaningless.

### Pitfall 3: Reporting results without confidence intervals

"MAE = 4.1" is meaningless without a confidence interval. Always report:
- Mean ± SD
- 95% confidence interval
- N (sample size)

### Pitfall 4: Not registering your study

If you plan to publish, register your study on ClinicalTrials.gov or OSF.io BEFORE running it. This prevents "p-hacking" accusations later.

### Pitfall 5: Using public data for clinical accuracy claims

This is the most dangerous pitfall. If you write "cardio-echo-suite achieves 95% accuracy" without specifying "on EchoNet-Dynamic test set, which is from Stanford and may not generalize," you're misleading readers. Always specify the data source and its limitations.

---

## What to do after downloading

1. Run the shakedown script: `scripts/public_data_shakedown.py`
2. Document your results in a research notebook (e.g., Notion, Overleaf)
3. If results are good, start drafting your paper (use `docs/research/PAPER_DRAFT_TEMPLATE.md`)
4. Submit to medRxiv as a preprint
5. Submit to a peer-reviewed journal (e.g., JAMA Cardiology, European Heart Journal - Digital Health)

---

## Time estimate

| Step | Time |
|---|---|
| CITI training | 6 hours |
| PhysioNet credentialing | 3-7 business days (waiting) |
| EchoNet-Dynamic request | 1-3 business days (waiting) |
| PTB-XL download | 30 min |
| EchoNet-Dynamic download | 1 hour (6 GB) |
| ACDC download | 30 min |
| Initial shakedown run | 2 hours |
| First paper draft | 2-4 weeks |

**Total to first paper draft**: ~4-6 weeks if you start today.
