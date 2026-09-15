"""MedSAM2 model wrapper — loads from HF Hub, supports ONNX export with fallback."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from cardio_echo_core.onnx import ModelBackend, export_with_fallback

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = os.environ.get("MEDSAM2_REPO_ID", "Bowang-lab/MedSAM2")


class MedSAM2Model:
    _instance: Optional["MedSAM2Model"] = None

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        device: Optional[str] = None,
        prefer_onnx: bool = True,
    ):
        self.repo_id = repo_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prefer_onnx = prefer_onnx
        self._model = None
        self._backend = ModelBackend.PYTORCH
        self._loaded = False

    @classmethod
    def get(cls, **kwargs) -> "MedSAM2Model":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import Sam2Model, Sam2Processor  # type: ignore
        except ImportError as e:
            raise RuntimeError("transformers with SAM-2 support required") from e

        logger.info("Loading MedSAM2 from %s (device=%s)", self.repo_id, self.device)
        self._model = Sam2Model.from_pretrained(self.repo_id).to(self.device).eval()
        self._processor = Sam2Processor.from_pretrained(self.repo_id)

        if self.prefer_onnx:
            sample = torch.randn(1, 3, 1024, 1024)
            wrapped, result = export_with_fallback(
                self._model,
                sample,
                parity_n_samples=3,
                parity_threshold=1e-3,  # segmentation is more tolerant
            )
            self._model = wrapped
            self._backend = result.backend
            logger.info("MedSAM2 backend: %s (parity max_diff=%.2e)",
                        self._backend, result.max_abs_diff)

        self._loaded = True

    @torch.inference_mode()
    def predict_point(
        self,
        image: torch.Tensor,
        point: Tuple[int, int],
        label: int = 1,
    ) -> torch.Tensor:
        """Predict a segmentation mask from a single point prompt.

        Args:
            image: (1, 3, H, W) tensor, range [0, 1].
            point: (x, y) pixel coordinates.
            label: 1 = foreground, 0 = background.

        Returns:
            Binary mask (1, H, W) tensor.
        """
        self.load()
        inputs = self._processor(
            images=image,
            input_points=[[[list(point)]]],
            input_labels=[[[label]]],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        masks = outputs.pred_masks.cpu()
        # Pick highest-scoring mask
        mask = masks[0, 0] > 0.5
        return mask.float()

    @torch.inference_mode()
    def predict_bbox(
        self,
        image: torch.Tensor,
        bbox: Tuple[int, int, int, int],  # x1, y1, x2, y2
    ) -> torch.Tensor:
        """Predict a segmentation mask from a bounding box prompt."""
        self.load()
        x1, y1, x2, y2 = bbox
        # Convert bbox to 2-point prompt with labels
        inputs = self._processor(
            images=image,
            input_boxes=[[[[x1, y1], [x2, y2]]]],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        masks = outputs.pred_masks.cpu()
        return (masks[0, 0] > 0.5).float()

    @torch.inference_mode()
    def predict_auto(self, image: torch.Tensor) -> torch.Tensor:
        """Automatic segmentation — runs default grid of points."""
        self.load()
        # Run on center point as a default
        _, _, H, W = image.shape
        return self.predict_point(image, (W // 2, H // 2))

    @property
    def backend(self) -> str:
        return self._backend.value
