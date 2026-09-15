"""cardio-echo-core — shared library for cardio-echo-suite services.

All cardio-echo-suite services depend on this package for:
- JWT authentication middleware
- HIPAA-compliant audit logging
- FHIR R4 resource builders
- DICOM loading and metadata extraction
- ONNX export with automatic PyTorch fallback (with parity tests)
- GPTQ quantization with golden-test parity gate
- Per-model backend flags (onnx/pytorch/int8)
- Dynamic model loader (LRU cache, idle eviction)
"""

__version__ = "0.1.0"
