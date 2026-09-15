# cardio-echo-core — shared library for cardio-echo-suite services

[![PyPI version](https://badge.fury.io/py/cardio-echo-core.svg)](https://badge.fury.io/py/cardio-echo-core)

Shared Python package used by all cardio-echo-suite services. Provides:

- **`cardio_echo_core.auth`** — JWT authentication middleware (FastAPI)
- **`cardio_echo_core.audit`** — HIPAA-compliant audit log middleware
- **`cardio_echo_core.fhir`** — FHIR R4 resource builders and helpers
- **`cardio_echo_core.dicom`** — DICOM loading and metadata extraction
- **`cardio_echo_core.onnx`** — ONNX export with automatic fallback to PyTorch
- **`cardio_echo_core.quantize`** — GPTQ quantization with golden-test parity gate
- **`cardio_echo_core.model_registry`** — per-model backend flags (onnx/pytorch/int8)
- **`cardio_echo_core.dynamic_loader`** — LRU model cache with idle eviction

## Installation

```bash
pip install -e packages/cardio-echo-core
```

## Usage

```python
from cardio_echo_core.auth import setup_auth, require_scopes
from cardio_echo_core.audit import AuditLogger
from cardio_echo_core.onnx import export_with_fallback
from cardio_echo_core.dynamic_loader import DynamicModelCache

app = FastAPI()
setup_auth(app, jwt_secret="...")
audit = AuditLogger(service_name="echonet-dynamic", log_dir="/var/log/cardio")

cache = DynamicModelCache(max_resident=2, idle_timeout_s=300)
model = await cache.get_or_load("echonet-dynamic", loader_fn)
```

## License

MIT (see `LICENSE` in repo root).
