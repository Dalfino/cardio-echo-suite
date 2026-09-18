# Southeast Asian & Asian Cardiac Dataset Catalog

> **Honest assessment**: There is NO large public Malaysian cardiac AI dataset. The best available proxies are Chinese and East Asian datasets, which share similar body habitus, disease prevalence, and scanner vendor mix.

---

## Tier 1: Direct SEA / Malaysian datasets (very limited)

### 1. Malaysia — UM Cardiac Echo Database (if exists)
- **Status**: ❌ Not publicly available
- **Note**: University of Malaya, UKM, USM may have internal datasets but none are publicly shared
- **Action**: Contact Malaysian cardiac societies (MSC, MSSH) for potential collaboration

### 2. Singapore — SingHealth ECG Repository
- **Status**: ❌ Not publicly available (institutional only)
- **Note**: Singapore General Hospital + National Heart Centre have large ECG archives
- **Action**: Contact NHCS (National Heart Centre Singapore) for research collaboration

### 3. Thailand — Thai ACS Registry
- **Status**: ⚠️ Clinical registry, not imaging data
- **Note**: Has outcomes data but no echo/ECG waveforms
- **URL**: https://www.thaiacs.org/

### 4. Indonesia — Indonesia Cardiovascular Research
- **Status**: ❌ No public cardiac AI datasets found
- **Note**: Indonesian Heart Association may have data but not publicly shared

**Honest conclusion**: There is essentially NO public Southeast Asian cardiac imaging dataset suitable for AI training. This is a gap in the literature — and an opportunity for your hospital to be the first to release one (even a small de-identified set would be impactful).

---

## Tier 2: East Asian datasets (best demographic proxies)

### 5. CPSC2018 (China Physiological Signal Challenge) ⭐ BEST ASIAN PROXY
- **Size**: 6,877 ECGs, 12-lead, 6-60 seconds
- **Labels**: 9 arrhythmia classes (AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE)
- **Vendor**: Mixed (Chinese hospitals)
- **Demographic**: Chinese, mixed age/sex
- **License**: Public research (FREE download, no credentialing!)
- **Why valuable**: 
  - Closest to Malaysian/SEA demographic (Asian body habitus)
  - Free and fast to download
  - Multi-center (from different Chinese hospitals)
  - Includes arrhythmias common in SEA (AF, PVC)
- **URL**: http://2018.icbeb.org/Challenge.html
- **Download**: https://figshare.com/articles/dataset/ECG_Arrhythmia_dataset/8272387
- **External?**: ✅ Different country, different vendor, different demographic

### 6. Chapman-Shaoxing 12-lead ECG
- **Size**: 10,646 ECGs
- **Labels**: 11 rhythm classes
- **Vendor**: Mixed
- **Demographic**: US (Missouri) + Chinese (Shaoxing) — mixed!
- **License**: Public research
- **Why valuable**: Includes Chinese patients → Asian demographic
- **URL**: https://figshare.com/collections/ChapmanECG/4560497
- **External?**: ✅ Multi-site, includes Asian demographic

### 7. Guangzhou Medical University ECG
- **Size**: ~5,000 ECGs (subset of larger internal dataset)
- **Labels**: Multi-label arrhythmia
- **Vendor**: Unspecified
- **Demographic**: Chinese (Southern China — closest to SEA genetically)
- **License**: Research (may require contact)
- **Note**: Southern Chinese population is genetically closest to SEA

### 8. Korea University LVQuan (CMR)
- **Size**: 145 patients, CMR short-axis
- **Labels**: LV volume, mass, EF
- **Vendor**: Siemens
- **Demographic**: Korean
- **License**: Research
- **Why valuable**: Korean demographic is similar to SEA for cardiac structure
- **URL**: Available via Korea University research collaboration

### 9. Japan — JACR (Japanese Association of Cardiovascular Research)
- **Status**: ❌ Not publicly available
- **Note**: Japan has rich cardiac data but strict privacy laws prevent public sharing

---

## Tier 3: South Asian datasets (secondary proxies)

### 10. India AIIMS Echo Dataset
- **Status**: ⚠️ Small datasets occasionally published on Kaggle
- **Note**: All India Institute of Medical Sciences has echo data but sharing is restricted
- **Search**: Check Kaggle for "AIIMS echo" or "Indian cardiac"

### 11. India PPG/BP datasets (Kaggle)
- **Size**: Various (100-1000 recordings)
- **Labels**: Blood pressure, heart rate
- **Modality**: PPG (not echo/ECG) — limited value
- **URL**: Search Kaggle for "PPG India"

---

## Tier 4: Diverse/multi-ethnic datasets (include some Asian patients)

### 12. MIMIC-IV-ECG (US — includes Asian patients)
- **Size**: 800,000+ ECGs
- **Demographic**: US (Boston) — includes ~5-8% Asian patients
- **License**: PhysioNet Credentialed
- **Why valuable**: Some Asian patients in the dataset
- **URL**: https://physionet.org/content/mimic-iv-ecg/1.1/

### 13. CODE-15% (Brazilian — diverse multi-ethnic)
- **Size**: 15,000 ECGs
- **Demographic**: Brazilian (multi-ethnic — includes Asian-Brazilian)
- **License**: Public research (requires registration)
- **URL**: https://code.15.stats.fmh.usp.br/

---

## Recommended Adapter A configuration for SEA relevance

```
Adapter A (multi-domain research, SEA-optimized):

  Base model: PanEcho (frozen)
  
  ECG adapters:
  ├── "cpsc2018"    — 6,877 Chinese ECGs (BEST Asian proxy)     [research]
  ├── "chapman"     — 10,646 US+Chinese ECGs                     [research]
  ├── "ptbxl"       — 21,837 German ECGs (European baseline)     [research]
  └── "georgia"     — 2,000 US ECGs (US baseline)                [research]
  
  Echo adapters:
  ├── "echonet"     — 10,030 US echo videos (Stanford)           [research]
  ├── "camus"       — 500 French echo videos                     [research]
  └── "york"        — 150 Canadian echo videos (Philips)         [research]
  
  CMR adapters:
  ├── "acdc"        — 100 French CMR (Siemens)                   [research]
  ├── "mms"         — 345 multi-vendor CMR                       [research]
  └── "lvquan"      — 145 Korean CMR (Asian demographic!)        [research]
  
  YOUR adapter (commercial):
  └── "hospital"    — Your hospital data (Malaysian)              [commercial]
```

**Routing for Malaysian patients**:
- ECG: weight CPSC2018 + Chapman highest (Asian demographic)
- Echo: weight EchoNet (GE scanner, most common in Malaysia) + CAMUS
- CMR: weight LVQuan (Korean, closest to SEA)

---

## How to download CPSC2018 (the best Asian proxy)

CPSC2018 is freely available — no credentialing needed:

```bash
# Method 1: Direct from Figshare
wget -O cpsc2018.zip "https://figshare.com/ndownloader/files/15621428"
unzip cpsc2018.zip -d /data/public/cpsc2018

# Method 2: From the challenge website
# http://2018.icbeb.org/Challenge.html

# Expected structure:
# /data/public/cpsc2018/
#   ├── TRAIN/
#   │   ├── Normal/
#   │   ├── AF/
#   │   ├── I-AVB/
#   │   ├── LBBB/
#   │   ├── RBBB/
#   │   ├── PAC/
#   │   ├── PVC/
#   │   ├── STD/
#   │   └── STE/
#   └── TEST/
#       └── (unlabeled)
```

---

## The opportunity for your hospital

**You can be the first to publish a Malaysian cardiac AI dataset.** Even a small de-identified set (100-200 echo videos with EF labels) would be:

1. **The first public Malaysian cardiac echo dataset** — highly cited
2. **Fills the SEA gap in the literature** — reviewers will love this
3. **Enables future SEA cardiac AI research** — you'd be the benchmark
4. **Strengthens your Paper 2** — "validated on the first Malaysian echo dataset"

This is a separate paper in itself: "The Malaysian Cardiac Echo Dataset (MCED): A De-Identified Open-Access Benchmark for Southeast Asian Cardiac AI"

---

## Summary: What to use for Adapter A

| Dataset | Modality | Country | Size | Download | Priority |
|---|---|---|---|---|---|
| CPSC2018 | ECG | China | 6,877 | Free, no cred | ⭐⭐⭐ (best Asian proxy) |
| Chapman | ECG | US+China | 10,646 | Free | ⭐⭐⭐ |
| EchoNet-Dynamic | Echo | US | 10,030 | ✅ Have | ⭐⭐⭐ |
| PTB-XL | ECG | Germany | 21,837 | ✅ Have | ⭐⭐ |
| Georgia | ECG | US | 2,000 | ✅ Have | ⭐⭐ |
| CAMUS | Echo | France | 500 | Free | ⭐⭐ |
| ACDC | CMR | France | 100 | Free | ⭐ |
| M&Ms | CMR | Multi | 345 | Free | ⭐ |
| LVQuan | CMR | Korea | 145 | Research | ⭐ (Asian!) |
| **Your hospital** | **Echo+ECG** | **Malaysia** | **200+** | **IRB needed** | **⭐⭐⭐ (commercial)** |

**Start with CPSC2018 + EchoNet-Dynamic + PTB-XL** (all available now).
Add CAMUS + LVQuan when downloaded.
Add your hospital data when IRB approves.
