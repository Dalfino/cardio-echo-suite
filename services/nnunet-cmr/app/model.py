"""nnU-Net CMR model wrapper.

Loads a pretrained nnU-Net checkpoint for cardiac MRI short-axis segmentation.
Supports ONNX export with automatic PyTorch fallback.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from cardio_echo_core.onnx import ModelBackend, export_with_fallback

logger = logging.getLogger(__name__)

# nnU-Net checkpoints are hosted on Zenodo / nnUNet's official release page.
# Set NNUNET_CMR_CHECKPOINT to override.
DEFAULT_CHECKPOINT = os.environ.get(
    "NNUNET_CMR_CHECKPOINT",
    "https://zenodo.org/records/11643238/files/nnUNet_results.zip",
)

# Label conventions (ACDC)
LABELS = {
    1: "lv_blood_pool",
    2: "lv_myocardium",
    3: "rv_blood_pool",
}


class NNUNetCMRModel:
    _instance: Optional["NNUNetCMRModel"] = None

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
        prefer_onnx: bool = True,
    ):
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prefer_onnx = prefer_onnx
        self._model = None
        self._backend = ModelBackend.PYTORCH
        self._loaded = False

    @classmethod
    def get(cls, **kwargs) -> "NNUNetCMRModel":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        logger.info("Loading nnU-Net CMR (checkpoint=%s, device=%s)",
                    self.checkpoint, self.device)

        try:
            # nnU-Net's official API
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # type: ignore
            self._predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                device=torch.device(self.device),
                verbose=False,
                disable_tqdm=True,
            )
            self._predictor.initialize_from_trained_model_folder(
                self.checkpoint, use_folds=(0,),
                checkpoint_name="checkpoint_best.pth",
            )
            self._model = self._predictor.network
        except ImportError:
            logger.warning(
                "nnunetv2 not installed. Install with `pip install nnunetv2` "
                "and set NNUNET_CMR_CHECKPOINT to a trained model folder."
            )
            raise

        if self.prefer_onnx and self._model is not None:
            sample = torch.randn(1, 1, 32, 192, 192)
            wrapped, result = export_with_fallback(
                self._model, sample, parity_n_samples=2, parity_threshold=1e-3,
            )
            self._model = wrapped
            self._backend = result.backend
            logger.info("nnU-Net CMR backend: %s", self._backend)

        self._loaded = True

    @torch.inference_mode()
    def predict(self, volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run 3D segmentation on a CMR volume.

        Args:
            volume: (1, 1, D, H, W) tensor, range [0, 1].

        Returns:
            Dict mapping label name -> binary mask tensor of shape (D, H, W).
        """
        self.load()
        x = volume.to(self.device)
        if hasattr(self, "_predictor"):
            # Use nnU-Net's predictor
            logits = self._predictor.predict_sliding_window_return_logits(x)
            seg = logits.argmax(dim=1).cpu()
        else:
            logits = self._model(x)
            seg = logits.argmax(dim=1).cpu()

        # Split by label
        masks: Dict[str, torch.Tensor] = {}
        for label_id, name in LABELS.items():
            masks[name] = (seg[0] == label_id).float()
        return masks

    @property
    def backend(self) -> str:
        return self._backend.value
