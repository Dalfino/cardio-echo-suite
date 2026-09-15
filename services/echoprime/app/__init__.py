"""EchoPrime service — HuggingFace Hub wrapper + LoRA fine-tuning scaffold.

EchoPrime is hosted on HuggingFace at `digital-echo/EchoPrime` (a model
repository, not a GitHub code repo). This package provides:

- `app.model.EchoPrimeModel` — loads weights from HF Hub on first use,
  supports CPU/GPU inference with caching.
- `app.finetune` — LoRA fine-tuning scaffold for institutional echo reports.
  Includes a dataset loader for JSONL report files and a PEFT/LoRA trainer.
- `app.fhir.diagnostic_report` — generates FHIR R4 DiagnosticReport from
  EchoPrime output (view classification, measurements, draft report text).

Upstream model card: https://huggingface.co/digital-echo/EchoPrime
"""

__version__ = "0.1.0"
