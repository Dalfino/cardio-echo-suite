"""EchoPrime model wrapper.

Loads the EchoPrime vision-language model from the HuggingFace Hub.
The model is a large multimodal transformer trained on 12M+ echo videos
for view classification, structure measurement, and report drafting.

Weights are downloaded on first use via `huggingface_hub.snapshot_download`
and cached under `~/.cache/huggingface/hub` (or `HF_HOME` if set).

For airgapped deployments:
    1. Pre-download: `huggingface-cli download digital-echo/EchoPrime`
    2. Set `HF_HUB_OFFLINE=1` and `HF_HOME=/path/to/cache` at runtime.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = os.environ.get("ECHOPRIME_REPO_ID", "digital-echo/EchoPrime")


class EchoPrimeModel:
    """Singleton wrapper around the EchoPrime HF model."""

    _instance: Optional["EchoPrimeModel"] = None

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )
        self._model = None
        self._processor = None
        self._loaded = False
        self.weights_available = False

    @classmethod
    def get(cls, **kwargs) -> "EchoPrimeModel":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def load(self) -> None:
        """Download weights from HF Hub and instantiate the model.

        If weights are unavailable (team hasn't released them yet), sets
        `self.weights_available = False` and returns without raising.
        Callers should check `weights_available` and fall back to PanEcho.
        """
        if self._loaded:
            return

        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers not installed. Run `pip install transformers`."
            ) from e

        logger.info(
            "Loading EchoPrime from HF Hub: %s (revision=%s, device=%s, dtype=%s)",
            self.repo_id, self.revision, self.device, self.torch_dtype,
        )

        try:
            self._model = AutoModel.from_pretrained(
                self.repo_id,
                revision=self.revision,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            ).to(self.device).eval()
            self._processor = AutoProcessor.from_pretrained(
                self.repo_id,
                revision=self.revision,
                trust_remote_code=True,
            )
            self.weights_available = True
        except Exception as e:
            logger.warning(
                "EchoPrime weights unavailable at %s. Falling back to PanEcho "
                "for echo analysis. Error: %s",
                self.repo_id, e,
            )
            self._model = None
            self._processor = None
            self.weights_available = False

        self._loaded = True  # mark as "checked" even if weights missing

    @torch.inference_mode()
    def analyze(
        self,
        video_path: Path,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run view classification + measurement + draft report.

        Args:
            video_path: path to an echo video file (MP4/AVI/DICOM).
            prompts: optional list of free-text prompts. If None, uses the
                default prompt set:
                  - "Classify the echocardiographic view."
                  - "Measure left ventricular ejection fraction."
                  - "Draft a structured echo report."

        Returns:
            dict with keys: `views`, `measurements`, `draft_report`.
        """
        self.load()
        prompts = prompts or [
            "Classify the echocardiographic view.",
            "Measure left ventricular ejection fraction.",
            "Draft a structured echo report.",
        ]

        # NB: the exact preprocessor interface depends on EchoPrime's model
        # card. We use a generic call here that will be adapted once the
        # upstream model is publicly released.
        results: Dict[str, Any] = {
            "views": [],
            "measurements": {},
            "draft_report": "",
            "prompts": prompts,
        }
        try:
            inputs = self._processor(videos=str(video_path), text=prompts, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            # Placeholder — actual decoding depends on EchoPrime's API
            results["_raw"] = {
                k: v.detach().cpu().tolist()[:1] if hasattr(v, "tolist") else str(v)
                for k, v in outputs.items()
            }
        except Exception as e:
            logger.warning("Inference failed (model may be stub): %s", e)
            results["error"] = str(e)

        return results
