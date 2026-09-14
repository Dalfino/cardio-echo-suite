"""EchoNet-Dynamic service — multi-vendor DICOM ingestion + REST API + FHIR output.

This package wraps the upstream `echonet` package (preserved under
`services/echonet-dynamic/upstream/`) and adds:

- `app.ingestion.dicom_loader` — multi-vendor DICOM video ingestion
  (GE Vivid, Philips IE33/EPIQ, Siemens Acuson, generic)
- `app.api.main` — FastAPI app with `POST /predict/ef` and `POST /predict/segment`
- `app.fhir.observation` — FHIR R4 Observation builder for EF

Upstream model code is unchanged. See NOTICE.md for attribution.
"""

__version__ = "0.1.0"
