"""PanEcho service — FHIR output mapping + pre-read CLI + REST.

Wraps upstream PanEcho (CarDS-Yale/PanEcho, JAMA 2025) — a multi-task model
that performs 39 echocardiography reporting tasks from echo videos.

Improvements over upstream:
- `app.fhir.mapping` — maps PanEcho's 39 task outputs to a FHIR R4 bundle
  of Observation resources (one per task, with appropriate LOINC/SNOMED codes).
- `app.cli` — a "pre-read" CLI that emits a structured JSON report (useful
  for echo lab workflow integration).
- `app.api` — FastAPI endpoint mirroring the CLI.

The upstream model code and weights are unchanged. We use
`torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho')` as the entrypoint.
"""

__version__ = "0.1.0"
