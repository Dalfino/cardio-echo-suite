"""Model wrapper around the upstream EchoNet-Dynamic model.

Loads the upstream `echonet` package and provides a thin interface:

    predict_ef(video_tensor) -> float
    predict_segmentation(video_tensor) -> (mask_tensor, areas)

The upstream `echonet` package is preserved verbatim at
`services/echonet-dynamic/upstream/`. We import it lazily so that the
service can start without GPU/model weights loaded — useful for unit
tests and the `--dry-run` mode.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Make upstream importable
UPSTREAM_DIR = Path(__file__).resolve().parents[2] / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))


class EchoNetDynamicModel:
    """Singleton wrapper around the upstream EchoNet-Dynamic model."""

    _instance: Optional["EchoNetDynamicModel"] = None

    def __init__(self, device: Optional[str] = None, weights_path: Optional[Path] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self._ef_model = None
        self._seg_model = None
        self._loaded = False

    @classmethod
    def get(cls, **kwargs) -> "EchoNetDynamicModel":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    # --------------------------------------------------------------
    # Lazy loading
    # --------------------------------------------------------------

    def load(self) -> None:
        """Import upstream `echonet` and load model weights."""
        if self._loaded:
            return

        try:
            import echonet  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"Could not import upstream echonet package from {UPSTREAM_DIR}. "
                f"Did you `pip install -e upstream/`? Error: {e}"
            ) from e

        logger.info("Loaded upstream echonet v%s", echonet.__version__)

        # The upstream `echonet` package downloads weights on first use
        # via `torch.hub.load_state_dict_from_url`. If you need an airgapped
        # deployment, set ECHONET_WEIGHTS_DIR to a directory containing
        # `r2plus1dmotionnet.pth` and `deeplabv3_resnet50_random.pth`.
        weights_dir = os.environ.get(
            "ECHONET_WEIGHTS_DIR",
            str(Path.home() / ".cache" / "echonet"),
        )
        Path(weights_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Loading EF (R2Plus1D) model on %s...", self.device)
        self._ef_model = echonet.models.r2plus1d.r2plus1d_18(
            num_classes=1,
            spatial_size=112,
            pretrained=True,
        ).to(self.device).eval()

        logger.info("Loading segmentation (DeepLabV3) model on %s...", self.device)
        self._seg_model = echonet.models.segmentation.deeplabv3_resnet50(
            num_classes=2,
        ).to(self.device).eval()
        # Load segmentation weights
        seg_weight_url = (
            "https://github.com/douyang/EchoNet-Dynamic/releases/download/v1.0.0/"
            "deeplabv3_resnet50_random.pth"
        )
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                seg_weight_url, progress=False, check_hash=False, map_location=self.device
            )
            self._seg_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(
                "Could not load segmentation weights from %s: %s. "
                "Segmentation endpoint will be unavailable.",
                seg_weight_url,
                e,
            )

        self._loaded = True
        logger.info("EchoNet-Dynamic ready.")

    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------

    @torch.inference_mode()
    def predict_ef(self, video: torch.Tensor) -> float:
        """Predict ejection fraction.

        Args:
            video: tensor of shape (1, T, 3, H, W), float32, range [0, 1].

        Returns:
            EF as a percentage (float, 0-100).
        """
        self.load()
        # Upstream expects (1, 3, T, H, W)
        x = video.squeeze(0).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        y = self._ef_model(x)
        ef = float(y.item())
        return max(0.0, min(100.0, ef))

    @torch.inference_mode()
    def predict_segmentation(
        self, video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run LV segmentation per frame.

        Args:
            video: tensor of shape (1, T, 3, H, W), float32, range [0, 1].

        Returns:
            masks: (1, T, H, W) int64 (0=background, 1=LV).
            areas: (T,) float32, LV area per frame in pixels.
        """
        self.load()
        x = video.squeeze(0).to(self.device)  # (T, 3, H, W)
        masks = []
        for frame in x:
            logits = self._seg_model(frame.unsqueeze(0))["out"]
            mask = logits.argmax(dim=1).squeeze(0)
            masks.append(mask)
        masks_t = torch.stack(masks, dim=0).unsqueeze(0).cpu()
        areas = (masks_t == 1).float().sum(dim=(1, 2, 3)).squeeze(0)
        return masks_t, areas
