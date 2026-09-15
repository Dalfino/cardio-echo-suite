# Modifications Applied to Upstream Projects

This document details exactly what `cardio-echo-suite` adds or changes on top of each upstream project. The upstream code is preserved verbatim under `services/<name>/upstream/` and is **not modified** — our additions live alongside it under `services/<name>/app/`.

---

## 1. EchoNet-Dynamic

**Upstream:** [echonet/dynamic](https://github.com/echonet/dynamic) (MIT license, Nature 2020)
**Location:** `services/echonet-dynamic/upstream/`

### What upstream provides
- Python package `echonet` with R2Plus1D model for EF prediction and DeepLabV3 for LV segmentation
- CLI entrypoint `python -m echonet`
- Jupyter notebook for converting DICOM → AVI
- Trained on 10,030 apical-4-chamber echo videos

### What we add
| File | Purpose |
|------|---------|
| `app/ingestion/dicom_loader.py` | Multi-vendor DICOM ingestion (GE Vivid, Philips IE33/EPIQ, Siemens Acuson, Fujifilm). Replaces the manual `ConvertDICOMToAVI.ipynb` step. Vendor detection via DICOM tags (Manufacturer, ManufacturerModelName, StationName). Vendor-aware sector cropping. Direct tensor output (1, T, 3, H, W) at the upstream's 112×112 input resolution. |
| `app/fhir/observation.py` | FHIR R4 Observation builders for EF (LOINC 10230-1) and segmentation summary. Includes contained Device resource identifying the AI performer, reference range per ASE 2015 guidelines, and interpretation codes. |
| `app/model.py` | Thin wrapper that imports the upstream `echonet` package lazily, downloads weights via `torch.hub`, and exposes `predict_ef()` + `predict_segmentation()`. |
| `app/api/main.py` | FastAPI app with `POST /predict/ef` and `POST /predict/segment`. Accepts DICOM/AVI/MP4 uploads, returns EF + FHIR Observation. |
| `Dockerfile` | CPU + GPU compatible (via `BASE` build arg). Installs upstream as `pip install -e upstream/`. |
| `tests/test_ingestion.py` | Tests for vendor sniffing and metadata extraction. No GPU or model weights required. |
| `tests/test_fhir.py` | Tests for FHIR Observation builders. |

### Suggested improvement status (from your table)
✅ "Add multi-vendor DICOM ingestion + a REST endpoint so sonographers get EF live at bedside" — **implemented**.

---

## 2. EchoPrime

**Upstream:** [digital-echo/EchoPrime](https://huggingface.co/digital-echo/EchoPrime) on HuggingFace
**Location:** `services/echoprime/` (no upstream code subtree — model is hosted on HF Hub)

### What upstream provides
- Vision-language model trained on 12M+ echo videos
- View classification, structure measurement, report drafting
- Available via `transformers.AutoModel.from_pretrained("digital-echo/EchoPrime")`

### What we add
| File | Purpose |
|------|---------|
| `app/model.py` | Singleton wrapper that downloads weights from HF Hub on first use via `transformers.AutoModel`. Supports CPU/GPU, float16 on GPU. Supports airgapped deployment via `HF_HUB_OFFLINE=1` + pre-populated cache. |
| `app/finetune/train.py` | LoRA fine-tuning scaffold. Accepts JSONL dataset of `{video_path, view, ef, report_text}`. Uses `peft.LoraConfig` with target modules covering typical ViT+LLM layer names. Saves adapter weights only (not full model). |
| `app/finetune/__init__.py` | Exports `EchoReportDataset`, `EchoReportExample`, `build_lora_config`, `train`. |
| `app/fhir/diagnostic_report.py` | FHIR R4 DiagnosticReport builder. Maps EchoPrime outputs to: LOINC 26520-0 (Echo study), 12 measurement LOINC codes (EF, LV dimensions, valve velocities, diastolic indices), 11 view SNOMED codes. Includes contained Observations, conclusion text, media link to source DICOM ImagingStudy. |
| `app/api/main.py` | FastAPI app with `POST /analyze`, `POST /finetune/start` (async job queue stub), `GET /finetune/status/{id}`. |
| `Dockerfile` | Standard image with transformers/peft/accelerate. |
| `tests/test_fhir.py` | Tests for DiagnosticReport builder (12 measurement codes, 11 view codes, study UID linking, encounter reference). |
| `tests/test_finetune.py` | Tests for dataset loader and LoRA config builder (no GPU required). |

### Suggested improvement status (from your table)
✅ "Fine-tune on your institution's echo reports to auto-generate structured (FHIR-encoded) echo reports" — **scaffold implemented**. The fine-tuning itself requires (a) institution data, (b) GPU node, (c) adaptation of the loss function in `app/finetune/train.py` to the specific report schema.

---

## 3. PanEcho

**Upstream:** [CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho) (JAMA 2025)
**Location:** `services/panecho/upstream/`

### What upstream provides
- Multi-task model performing 39 echocardiography reporting tasks
- Loads via `torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho')`
- Returns dict of {task_name → prediction tensor}
- Pretrained weights (~150 MB) downloaded on first use

### What we add
| File | Purpose |
|------|---------|
| `app/fhir/mapping.py` | The crown jewel. Maps all 39 PanEcho tasks to FHIR R4 Observations: 13 regression tasks (with LOINC codes + units: EF, LVIDd, MVArea, etc.) and 26+ classification tasks (with class labels + probabilities). Builds a Bundle of Observations, plus a `build_preread_report()` function that produces a structured pre-read with summary, findings, and critical findings flags (e.g. EF < 35%, severe AS, tamponade). |
| `app/model.py` | Thin wrapper around `torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho')`. Lazy loading, CPU/GPU aware. |
| `app/cli.py` | "Pre-read" CLI: `python -m app.cli predict /path/to/echo.mp4 --patient-id P1 --output report.json`. Includes video loading, ImageNet normalization, clip subsampling to 16 frames, and dry-run mode for testing FHIR mapping without loading the model. |
| `app/api/main.py` | FastAPI app mirroring the CLI: `POST /predict`, `GET /tasks` (lists all 39 tasks), `GET /readyz`, `GET /healthz`. |
| `Dockerfile` | Standard image. Includes `git` (required for `torch.hub.load` from GitHub). |
| `tests/test_fhir_mapping.py` | Tests for: task count (≥39), regression Observation shape, classification Observation shape, bundle composition, critical EF detection, critical class detection, study UID linking. |

### Suggested improvement status (from your table)
✅ "Localize with clinical-Fhir output mapping so it can be a 'pre-read' tool in any echo lab" — **implemented**. The CLI + REST endpoint can be wired into any echo lab workflow.

---

## 4. Orchestrator (new)

**Location:** `services/orchestrator/`

Not in the original table — added to tie the three services together.

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app with `POST /v1/echo/full` that fans out to all three services in parallel via `httpx.AsyncClient` and returns a unified result. Supports `?skip=echonet,echoprime` to disable individual services. |

---

## 5. Cross-cutting

| File | Purpose |
|------|---------|
| `deploy/docker-compose.yml` | CPU deployment. 4 services (orchestrator + 3 models), health checks, named volumes for HF Hub + torch.hub caches. |
| `deploy/docker-compose.gpu.yml` | GPU deployment with NVIDIA Container Toolkit. Same services, CUDA base image for EchoNet-Dynamic, GPU reservations. |
| `NOTICE.md` | Attribution for all three upstream projects with citation, license pointer, and modification summary. |
| `LICENSE` | MIT for our additions only. Upstream licenses preserved in each `services/<name>/upstream/` directory. |
| `.github/workflows/ci.yml` | GitHub Actions CI: runs the non-GPU tests for all three services on every push. |

---

## What we did NOT change

- **Upstream model weights**: We never modify, redistribute, or re-train the upstream weights. They are always downloaded from the original source (`torch.hub` or HF Hub) on first use.
- **Upstream training data**: Not redistributed. Users must obtain datasets via the upstream project's data use agreement.
- **Upstream license**: Each upstream LICENSE file is preserved verbatim under `services/<name>/upstream/`.
- **Upstream model code**: Imported as-is. No patches to upstream `*.py` files.
