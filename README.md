# cardio-echo-suite

> Forked + improved cardiac echo AI suite. Three upstream projects are preserved as git subtrees (original LICENSE intact); improvements are layered on top. See `NOTICE.md` for attribution and `docs/MODIFICATIONS.md` for what changed per project.

## What's inside

| Service | Upstream | Improvement |
|---|---|---|
| `services/echonet-dynamic` | [echonet/dynamic](https://github.com/echonet/dynamic) | Multi-vendor DICOM ingestion + FastAPI `/predict/ef` endpoint + FHIR Observation output |
| `services/echoprime` | [digital-echo/EchoPrime](https://huggingface.co/digital-echo/EchoPrime) | HF Hub model wrapper + LoRA fine-tuning scaffold + FHIR DiagnosticReport generator |
| `services/panecho` | [CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho) | Clinical-FHIR output mapping (40+ outputs) + pre-read CLI mode + REST endpoint |

## Quickstart

```bash
# CPU
docker compose -f deploy/docker-compose.yml up

# GPU
docker compose -f deploy/docker-compose.gpu.yml up
```

Then open http://localhost:8080/docs for the orchestrator OpenAPI.

## Architecture

```
                 +-----------------------+
   DICOM/MP4 --> |  orchestrator (8080)  | -- /v1/echo/full
                 +-----------+-----------+
                             |
        +--------------------+--------------------+
        |                    |                    |
+-------v-------+   +--------v------+   +---------v----+
| echonet-dyn   |   |  echoprime    |   |   panecho    |
| :8001 /ef     |   |  :8002 /report|   |   :8003 /pre |
+---------------+   +---------------+   +---------------+
```

Each service exposes:
- `POST /predict` — accepts a file path or upload
- `GET /healthz` — liveness
- `GET /readyz` — model loaded

All responses include a FHIR R4 bundle in the `fhir` field.

## License & attribution

Each upstream project retains its original LICENSE under its subdirectory. Our additions are MIT-licensed (see `LICENSE`). See `NOTICE.md` for full attribution.

**Research use only. Not for clinical decision-making.**
