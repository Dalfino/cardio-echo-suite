"""PanEcho model wrapper.

Uses `torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho')` to load the upstream
pretrained model. Weights are downloaded on first use (~150 MB) and cached
under `~/.cache/torch/hub`.

For airgapped deployments, pre-populate the cache or set
`TORCH_HOME=/path/to/cache`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_REPO = os.environ.get("PANECHO_REPO", "CarDS-Yale/PanEcho")


class PanEchoModel:
    """Singleton wrapper around the upstream PanEcho model."""

    _instance: Optional["PanEchoModel"] = None

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        device: Optional[str] = None,
        clip_len: int = 16,
        tasks: str = "all",
    ):
        self.repo = repo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_len = clip_len
        self.tasks = tasks
        self._model = None
        self._loaded = False

    @classmethod
    def get(cls, **kwargs) -> "PanEchoModel":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        logger.info("Loading PanEcho from %s (clip_len=%d, tasks=%s, device=%s)",
                    self.repo, self.clip_len, self.tasks, self.device)
        self._model = torch.hub.load(
            self.repo, "PanEcho",
            source="github",
            pretrained=True,
            tasks=self.tasks,
            clip_len=self.clip_len,
            trust_repo=True,
        )
        self._model = self._model.to(self.device).eval()
        self._loaded = True
        logger.info("PanEcho ready.")

    @torch.inference_mode()
    def predict(self, video: torch.Tensor) -> Dict[str, Any]:
        """Run PanEcho inference on a video clip.

        Args:
            video: tensor of shape (1, 3, T, H, W), ImageNet-normalized,
                   224x224 resolution. T should equal `self.clip_len`.

        Returns:
            Dict mapping task name -> prediction. For classification tasks,
            the value is a dict with `probabilities` (per-class) and
            `predicted_class`. For regression tasks, the value is a dict
            with `value` (the scalar prediction).
        """
        self.load()
        x = video.to(self.device)
        if x.dim() == 4:
            x = x.unsqueeze(0)  # (T, 3, H, W) -> (1, T, 3, H, W)
        if x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)  # fix channel dim if needed
        out = self._model(x)
        return {k: v.detach().cpu() for k, v in out.items()}
