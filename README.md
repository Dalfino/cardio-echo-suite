# cardio-echo-suite

> 7-service cardiac AI suite: 6 model services + 1 signal-processing utility + 1 orchestrator. CPU-first, ONNX where it works, INT8 where it helps. Upstream licenses preserved; improvements layered on top.

## What's inside

### Model services (6)

| Service | Port | Upstream | Modality | What it does |
|---|---|---|---|---|
| `echonet-dynamic` | 8001 | [echonet/dynamic](https://github.com/echonet/dynamic) | Echo | EF + LV segmentation |
| `echoprime` | 8002 | [digital-echo/EchoPrime](https://huggingface.co/digital-echo/EchoPrime) | Echo | VLM: views + measurements + draft report |
| `panecho` | 8003 | [CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho) | Echo | 39-task pre-read |
| `ecg-fm` | 8004 | [bman03/ECG-FM](https://huggingface.co/bman03/ECG-FM) | ECG | Rhythm + intervals + STEMI |
| `medsam2` | 8005 | [Bowang-lab/MedSAM2](https://huggingface.co/Bowang-lab/MedSAM2) | CT/MR/Echo | Promptable 3D segmentation |
| `nnunet-cmr` | 8006 | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | CMR | Automatic LV/myo/RV segmentation |

### Utility services (1)

| Service | Port | Upstream | What it does |
|---|---|---|---|
| `neurokit` | 8007 | [neuropsychology/NeuroKit](https://github.com/neuropsychology/NeuroKit) | HRV + R-peak + signal quality (consumes ECG signal) |

### Orchestrator (1)

| Service | Port | What it does |
|---|---|---|
| `orchestrator` | 8080 | FHIR Composition tie-up: `/v1/echo/full`, `/v1/ecg/full`, `/v1/imaging/full` |

### Shared library

| Package | Path | What it provides |
|---|---|---|
| `cardio-echo-core` | `packages/cardio-echo-core/` | JWT auth, HIPAA audit log, FHIR R4 builders, DICOM utils, ONNX export with PyTorch fallback + parity test, GPTQ quantization with golden-test gate, dynamic model loader (LRU + idle eviction), model registry |

## Architecture

```
                +---------------------------+
                |  orchestrator (:8080)     |
                |  /v1/echo/full            |
                |  /v1/ecg/full             |
                |  /v1/imaging/full         |
                +-------------+-------------+
                              |
   +----------+----------+----+----+----------+----------+
   |          |          |         |          |          |
+--v---+  +---v---+  +---v---+ +--v---+  +---v---+  +---v---+  +-------+
|echo- |  |echo-  |  |pan-   | |ecg-  |  |med-   |  |nnunet|  |neuro- |
|net   |  |prime  |  |echo   | |fm    |  |sam2   |  |cmr   |  |kit   |
|:8001 |  |:8002  |  |:8003  | |:8004 |  |:8005  |  |:8006 |  |:8007 |
+------+  +-------+  +-------+ +------+  +-------+  +------+  +-------+
   |          |          |         |          |          |          |
   +----------+----------+---------+----------+----------+----------+
                              |
                  +-----------v-----------+
                  |  cardio-echo-core     |  (shared Python package)
                  |  - JWT auth           |
                  |  - HIPAA audit log    |
                  |  - FHIR R4 builders   |
                  |  - DICOM utils        |
                  |  - ONNX + fallback    |
                  |  - GPTQ + golden test |
                  |  - Dynamic loader     |
                  +-----------------------+
```

## Quickstart

```bash
# CPU deployment (all 7 services + orchestrator)
docker compose -f deploy/docker-compose.yml up

# GPU deployment
docker compose -f deploy/docker-compose.gpu.yml up
```

Then open http://localhost:8080/docs for the orchestrator OpenAPI.

## Efficiency strategy (B+ Lean)

The suite is designed to run on commodity CPU hardware (no GPU required for inference):

1. **Dynamic model loading** — at most 2 models resident at any time, idle eviction after 5 min. Peak memory: ~2 GB instead of ~6 GB.
2. **Shared `cardio-echo-core`** — auth, audit, FHIR, DICOM, ONNX, quantization all in one place. Change once, applies everywhere.
3. **CPU-first inference** — all 7 services run on CPU. GPU optional for fine-tune jobs.
4. **ONNX where it works** — automatic export with parity test. Falls back to PyTorch if export fails or numerical drift exceeds 1e-4.
5. **INT8 only for EchoPrime** — GPTQ with golden-test parity gate. Falls back to FP16 if MAE > 1.0 EF percentage points.

### Defense-in-depth for accuracy risks

| Risk | Defense |
|---|---|
| ONNX export fails on custom ops | Auto-fallback to PyTorch, parity test on 10 random inputs, max_diff < 1e-4 |
| INT8 quantization hurts accuracy | GPTQ (better than naive PTQ), golden test on N samples, MAE < 1.0 EF points, automatic FP16 fallback |
| Model version drift | `cardio_echo_core.model_registry` pins upstream git refs / HF revisions |

## Test coverage

64 tests across 8 packages:

```
echonet-dynamic    11 passed
echoprime           9 passed
panecho             8 passed
ecg-fm             15 passed
medsam2             2 passed
nnunet-cmr          4 passed
neurokit            4 passed
cardio-echo-core   11 passed
                  --------
TOTAL              64 passed
```

Run all tests:
```bash
for svc in echonet-dynamic echoprime panecho ecg-fm medsam2 nnunet-cmr neurokit; do
  cd services/$svc && PYTHONPATH=. pytest tests/ -q && cd -
done
cd packages/cardio-echo-core && pytest tests/ -q
```

## License & attribution

Each upstream project retains its original LICENSE under its respective subdirectory. Our additions (including `cardio-echo-core`) are MIT-licensed (see `LICENSE`). See `NOTICE.md` for full attribution.

**Research use only. Not for clinical decision-making.**
