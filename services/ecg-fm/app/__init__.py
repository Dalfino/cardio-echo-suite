"""ECG-FM service — ECG foundation model wrapper + FHIR output + LoRA fine-tune.

Wraps the upstream ECG-FM model (bman03/ECG-FM on HuggingFace) — a transformer
pretrained with contrastive + generative self-supervised learning on large ECG
corpora. Strong baseline for arrhythmia detection, EF estimation, and STEMI
detection.

Improvements over upstream:
- `app.ingestion.loader` — reads WFDB, DICOM waveform, MUSE XML, and generic
  CSV formats; resamples to 250 Hz; outputs (12, 1000) tensor.
- `app.fhir.observation` — FHIR R4 Observation builder for ECG findings
  (LOINC 8889-8 rhythm, 44967-8 PR interval, 44577-8 QRS duration, etc.)
- `app.finetune` — LoRA scaffold reusing the EchoPrime pattern.
- `app.api` — FastAPI endpoint with /predict/arrhythmia, /predict/intervals.

Upstream: https://huggingface.co/bman03/ECG-FM
"""

__version__ = "0.1.0"
