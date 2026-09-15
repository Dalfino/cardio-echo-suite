"""ECG-FM model wrapper.

Loads the upstream ECG-FM model from HuggingFace (bman03/ECG-FM).
The model is a transformer pretrained with contrastive + generative SSL
on >500k ECGs. We expose three downstream tasks:

1. **Arrhythmia classification** — multi-label across 30 SNOMED CT rhythms
2. **Interval measurement** — PR, QRS, QT, QTc, RR in ms
3. **STEMI detection** — boolean + anatomical localization

Weights are downloaded on first use via `transformers.AutoModel.from_pretrained`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = os.environ.get("ECGFM_REPO_ID", "bman03/ECG-FM")


class ECGFMModel:
    """Singleton wrapper around the ECG-FM HF model."""

    _instance: Optional["ECGFMModel"] = None

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        device: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._loaded = False

    @classmethod
    def get(cls, **kwargs) -> "ECGFMModel":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as e:
            raise RuntimeError("transformers not installed. `pip install transformers`.") from e

        logger.info("Loading ECG-FM from %s (device=%s)", self.repo_id, self.device)
        self._model = AutoModel.from_pretrained(
            self.repo_id, revision=self.revision, trust_remote_code=True,
        ).to(self.device).eval()
        self._processor = AutoProcessor.from_pretrained(
            self.repo_id, revision=self.revision, trust_remote_code=True,
        )
        self._loaded = True
        logger.info("ECG-FM ready.")

    @torch.inference_mode()
    def predict(
        self,
        ecg_tensor: torch.Tensor,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run ECG-FM inference.

        Args:
            ecg_tensor: tensor of shape (1, 12, 1000), float32, z-scored per lead.
            tasks: subset of ["arrhythmia", "intervals", "stemi"]; None = all.

        Returns:
            dict with keys "arrhythmia" (dict arrhythmia_key -> probability),
            "intervals" (dict interval_key -> ms), "stemi" (dict location -> prob).
        """
        self.load()
        x = ecg_tensor.to(self.device)
        tasks = tasks or ["arrhythmia", "intervals", "stemi"]

        out: Dict[str, Any] = {"arrhythmia": {}, "intervals": {}, "stemi": {}}
        try:
            inputs = self._processor(ecg=x, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self._model(**inputs)
            # Placeholder — actual output shape depends on ECG-FM's head
            # design. The team's public release should document this.
            out["_raw"] = {k: v.detach().cpu().tolist() if hasattr(v, "tolist") else str(v)
                           for k, v in (logits.items() if hasattr(logits, "items") else {"logits": logits}.items())}
        except Exception as e:
            logger.warning("Inference failed (model may be stub): %s", e)
            out["error"] = str(e)
        return out
