"""DICOM C-STORE gateway — listens for incoming studies, routes to AI services.

A minimal DICOM Storage SCP (Service Class Provider) that:
1. Listens on port 11112 for incoming C-STORE associations from PACS
2. Saves each received DICOM file to /data/incoming/<StudyInstanceUID>/
3. When a study is complete (association closed), inspects the modality
   and routes the study to the appropriate AI service:
   - US (echo) -> orchestrator /v1/echo/full
   - ECG waveform -> orchestrator /v1/ecg/full
   - MR -> orchestrator /v1/imaging/full?modality=cmr
   - CT -> orchestrator /v1/imaging/full?modality=ct
4. Returns the AI result via DICOM-SR (in Phase A) or saves to /data/results/

Requires `pip install pynetdicom`.

Run:
    python -m app.main --port 11112 --ae-title CARDIO_ECHO_SUITE
"""

__version__ = "0.1.0"
