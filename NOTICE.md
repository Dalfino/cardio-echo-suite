# NOTICE — cardio-echo-suite

This repository incorporates code, models, and methodologies from the following upstream projects. Each upstream LICENSE is preserved in its respective subdirectory.

## 1. EchoNet-Dynamic
- Upstream: https://github.com/echonet/dynamic
- Citation: Ouyang D et al., *Video-based AI for beat-to-beat assessment of cardiac function*. Nature 2020.
- License: See `services/echonet-dynamic/upstream/LICENSE` (MIT)
- Modifications applied by cardio-echo-suite:
  - Added `app/ingestion/dicom_loader.py` — multi-vendor DICOM ingestion (GE/Philips/Siemens)
  - Added `app/api/main.py` — FastAPI endpoint `POST /predict/ef` returning EF + FHIR R4 Observation
  - Added `Dockerfile` (CPU + GPU variants)
  - No changes to upstream model code or weights

## 2. EchoPrime
- Upstream: https://huggingface.co/digital-echo/EchoPrime
- Citation: *EchoPrime: ...* (preprint 2024)
- License: See upstream HuggingFace model card (research use)
- Modifications applied by cardio-echo-suite:
  - Created `app/` — HF Hub model wrapper (loads weights at runtime via `huggingface_hub`)
  - Created `app/finetune/` — LoRA fine-tuning scaffold for institutional echo reports
  - Created `app/fhir/` — DiagnosticReport generator from model output
  - No changes to upstream model weights (downloaded on-demand)

## 3. PanEcho
- Upstream: https://github.com/CarDS-Yale/PanEcho
- Citation: Holste E et al., *Complete AI-Enabled Echocardiography Interpretation with Multitask Deep Learning*. JAMA 2024.
- License: See `services/panecho/upstream/LICENSE`
- Modifications applied by cardio-echo-suite:
  - Added `app/fhir/mapping.py` — maps 40+ PanEcho outputs to FHIR R4 Observation bundle
  - Added `app/cli.py` — "pre-read" CLI mode that emits a structured JSON report
  - Added `app/api/main.py` — FastAPI endpoint mirroring the CLI
  - No changes to upstream model code or weights

## General note
The model weights, training data, and core algorithms of all three upstreams are **unchanged**. cardio-echo-suite only adds:
- Ingestion glue (DICOM, HL7, FHIR)
- REST API surfaces
- Docker deployment recipes
- LoRA fine-tuning scaffolds (no fine-tuned weights shipped)

For clinical use, consult the original publications and your local IRB.
