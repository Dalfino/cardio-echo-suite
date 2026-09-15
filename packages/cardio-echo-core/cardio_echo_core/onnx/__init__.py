"""ONNX export with automatic fallback to PyTorch.

Defense-in-depth for the "ONNX export fails on custom ops" risk:

1. `export_with_fallback()` tries to export a model to ONNX.
2. If export fails, returns the original PyTorch model (no ONNX).
3. If export succeeds, runs a parity test on N random inputs.
4. If parity test fails, returns the original PyTorch model.
5. Otherwise returns an OnnxModelWrapper that runs via onnxruntime.

Usage:
    from cardio_echo_core.onnx import export_with_fallback, ModelBackend

    model = load_pytorch_model()
    model, backend = export_with_fallback(
        model,
        sample_input=torch.randn(1, 3, 16, 224, 224),
        opset=17,
        parity_threshold=1e-4,
        parity_n_samples=10,
    )
    # backend == ModelBackend.ONNX or ModelBackend.PYTORCH
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ModelBackend(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    INT8 = "int8"


@dataclass
class OnnxParityResult:
    backend: ModelBackend
    max_abs_diff: float
    mean_abs_diff: float
    n_samples: int
    fallback_reason: Optional[str] = None  # set if fell back to PyTorch


class OnnxModelWrapper(torch.nn.Module):
    """Wraps an ONNX Runtime session as a PyTorch-callable module."""

    def __init__(self, session, input_names: list, output_names: list):
        super().__init__()
        self._session = session
        self._input_names = input_names
        self._output_names = output_names

    def forward(self, *args, **kwargs):
        # Convert all positional args to numpy
        feed = {}
        for name, tensor in zip(self._input_names, args):
            feed[name] = tensor.detach().cpu().numpy()
        for name, tensor in kwargs.items():
            if name in self._input_names:
                feed[name] = tensor.detach().cpu().numpy()

        outputs = self._session.run(self._output_names, feed)
        if len(outputs) == 1:
            return torch.from_numpy(outputs[0]).to(args[0].device if args else "cpu")
        return tuple(torch.from_numpy(o).to(args[0].device if args else "cpu") for o in outputs)


def export_with_fallback(
    model: torch.nn.Module,
    sample_input: torch.Tensor | Tuple[torch.Tensor, ...],
    opset: int = 17,
    parity_threshold: float = 1e-4,
    parity_n_samples: int = 10,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[dict] = None,
) -> Tuple[torch.nn.Module, OnnxParityResult]:
    """Try to export a PyTorch model to ONNX. Fall back to PyTorch on any failure.

    Args:
        model: PyTorch model in eval mode.
        sample_input: a tensor or tuple of tensors matching the model's forward signature.
        opset: ONNX opset version (17 is widely supported).
        parity_threshold: max allowed |pytorch_output - onnx_output|. If exceeded, fall back.
        parity_n_samples: number of random inputs to test for parity.
        input_names, output_names, dynamic_axes: standard torch.onnx.export params.

    Returns:
        (model, OnnxParityResult) — model is either the original PyTorch model
        (if export/parity failed) or an OnnxModelWrapper (if successful).
    """
    if not isinstance(sample_input, tuple):
        sample_input = (sample_input,)
    input_names = input_names or [f"input_{i}" for i in range(len(sample_input))]
    output_names = output_names or ["output"]

    # Step 1: try to export
    buffer = io.BytesIO()
    try:
        torch.onnx.export(
            model,
            sample_input,
            buffer,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {n: {0: "batch"} for n in input_names + output_names},
        )
        buffer.seek(0)
    except Exception as e:
        logger.warning("ONNX export failed (%s); falling back to PyTorch.", e)
        return model, OnnxParityResult(
            backend=ModelBackend.PYTORCH,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            n_samples=0,
            fallback_reason=f"export_error: {type(e).__name__}: {str(e)[:200]}",
        )

    # Step 2: load ONNX Runtime session
    try:
        import onnxruntime as ort  # type: ignore
        sess = ort.InferenceSession(buffer.read(), providers=["CPUExecutionProvider"])
    except ImportError:
        logger.warning("onnxruntime not installed; falling back to PyTorch.")
        return model, OnnxParityResult(
            backend=ModelBackend.PYTORCH,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            n_samples=0,
            fallback_reason="onnxruntime_not_installed",
        )
    except Exception as e:
        logger.warning("ONNX session creation failed (%s); falling back.", e)
        return model, OnnxParityResult(
            backend=ModelBackend.PYTORCH,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            n_samples=0,
            fallback_reason=f"session_error: {type(e).__name__}: {str(e)[:200]}",
        )

    # Step 3: parity test
    max_diff = 0.0
    sum_diff = 0.0
    n_compared = 0
    try:
        with torch.inference_mode():
            for _ in range(parity_n_samples):
                # Generate random inputs of same shape/dtype as sample
                random_inputs = []
                for s in sample_input:
                    if s.dtype.is_floating_point:
                        random_inputs.append(torch.rand_like(s))
                    else:
                        random_inputs.append(torch.randint_like(s, 0, 100))
                pt_out = model(*random_inputs)
                if isinstance(pt_out, dict):
                    pt_out = list(pt_out.values())[0]
                if isinstance(pt_out, (tuple, list)):
                    pt_out = pt_out[0]
                pt_np = pt_out.detach().cpu().numpy()

                feed = {name: t.detach().cpu().numpy() for name, t in zip(input_names, random_inputs)}
                onnx_out = sess.run(output_names, feed)[0]

                diff = np.abs(pt_np - onnx_out)
                max_diff = max(max_diff, float(diff.max()))
                sum_diff += float(diff.mean())
                n_compared += 1
    except Exception as e:
        logger.warning("ONNX parity test errored (%s); falling back.", e)
        return model, OnnxParityResult(
            backend=ModelBackend.PYTORCH,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            n_samples=n_compared,
            fallback_reason=f"parity_error: {type(e).__name__}: {str(e)[:200]}",
        )

    mean_diff = sum_diff / max(n_compared, 1)

    # Step 4: parity check
    if max_diff > parity_threshold:
        logger.warning(
            "ONNX parity test failed (max_diff=%.2e > threshold=%.2e); falling back to PyTorch.",
            max_diff, parity_threshold,
        )
        return model, OnnxParityResult(
            backend=ModelBackend.PYTORCH,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
            n_samples=n_compared,
            fallback_reason=f"parity_failed: max_diff={max_diff:.2e}",
        )

    logger.info(
        "ONNX export + parity OK (max_diff=%.2e, mean_diff=%.2e, n=%d).",
        max_diff, mean_diff, n_compared,
    )
    wrapper = OnnxModelWrapper(sess, input_names, output_names)
    return wrapper, OnnxParityResult(
        backend=ModelBackend.ONNX,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        n_samples=n_compared,
    )
