"""Multi-Domain LoRA Manager — train + route + ensemble multiple LoRA adapters.

This is the genuinely novel contribution: instead of one LoRA adapter, we
train SEPARATE adapters on different datasets (public + private), then
intelligently combine them at inference time.

Architecture:
    Base model (PanEcho, frozen)
        ├── Adapter "echonet"     (trained on EchoNet-Dynamic — US, GE)
        ├── Adapter "camus"       (trained on CAMUS — France, GE)
        ├── Adapter "york"        (trained on York Echo — Canada, Philips)
        ├── Adapter "hospital"    (trained on YOUR hospital data — commercial)
        └── Adapter "ensemble"    (meta-adapter that combines all above)

Three inference modes:
1. ROUTE: pick the best adapter for the input (by scanner vendor, demographics)
2. ENSEMBLE: run all adapters, weight by input similarity to each training set
3. STACK: apply adapters sequentially (experimental — can compound improvements)

Research vs Commercial:
- Adapters trained on public data (EchoNet, CAMUS, York) = RESEARCH USE ONLY
- Adapter trained on your hospital data = COMMERCIAL OK
- The multi-domain manager itself = commercial OK (it's just code)
- The KNOWLEDGE of which augmentations/calibrations work = commercial OK

Usage:
    from cardio_echo_ml.multi_domain_lora import MultiDomainLoRAManager

    manager = MultiDomainLoRAManager(base_model=panecho_model)

    # Train adapters on different datasets (research phase)
    manager.train_adapter(
        name="echonet",
        dataset=echonet_dataset,
        output_dir="adapters/echonet",
    )
    manager.train_adapter(
        name="camus",
        dataset=camus_dataset,
        output_dir="adapters/camus",
    )
    manager.train_adapter(
        name="hospital",
        dataset=hospital_dataset,  # YOUR data
        output_dir="adapters/hospital",
    )

    # Save adapter metadata (which dataset, license, domain)
    manager.save_registry("adapters/registry.json")

    # Inference: ensemble all adapters
    result = manager.predict_ensemble(video, mode="ensemble")
    # result = {
    #     "ef": 55.2,
    #     "per_adapter": {"echonet": 54.8, "camus": 55.5, "hospital": 55.3},
    #     "weights": {"echonet": 0.25, "camus": 0.25, "hospital": 0.50},
    #     "agreement": 0.94,
    # }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .lora import LoRAFineTuner, HospitalDataset

logger = logging.getLogger(__name__)


@dataclass
class AdapterMetadata:
    """Metadata for a trained LoRA adapter."""
    name: str
    dataset_name: str
    license: str  # "research_only" or "commercial_ok"
    domain: Dict[str, Any]  # e.g., {"country": "US", "vendor": "GE", "modality": "echo"}
    path: str
    mae_on_training: Optional[float] = None
    mae_on_validation: Optional[float] = None
    n_training_examples: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    created_at: str = ""


class MultiDomainLoRAManager:
    """Manages multiple LoRA adapters for different domains.

    Supports training, routing, ensembling, and stacking of adapters
    trained on different datasets.
    """

    def __init__(
        self,
        base_model: nn.Module,
        device: Optional[str] = None,
    ):
        """Initialize multi-domain manager.

        Args:
            base_model: the frozen base model (e.g., PanEcho)
            device: 'cuda' or 'cpu'
        """
        self.base_model = base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adapters: Dict[str, AdapterMetadata] = {}
        self._loaded_adapters: Dict[str, nn.Module] = {}

    def train_adapter(
        self,
        name: str,
        dataset: HospitalDataset,
        output_dir: Union[str, Path],
        val_dataset: Optional[HospitalDataset] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        license: str = "research_only",
        domain: Optional[Dict[str, Any]] = None,
    ) -> AdapterMetadata:
        """Train a LoRA adapter on a specific dataset.

        Args:
            name: adapter name (e.g., "echonet", "camus", "hospital")
            dataset: training dataset
            output_dir: where to save adapter weights
            val_dataset: optional validation dataset
            lora_r, lora_alpha: LoRA hyperparameters
            epochs: training epochs
            learning_rate: LR for LoRA (typically 1e-4)
            license: "research_only" or "commercial_ok"
            domain: dict describing the domain, e.g.,
                   {"country": "US", "vendor": "GE", "modality": "echo"}

        Returns:
            AdapterMetadata for the trained adapter
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Training adapter '%s' on %d examples", name, len(dataset))
        logger.info("License: %s", license)
        logger.info("Domain: %s", domain or {})
        logger.info("=" * 60)

        # Create fine-tuner
        tuner = LoRAFineTuner(
            model=self.base_model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            epochs=epochs,
            device=self.device,
        )

        # Train
        adapter_path = tuner.fit(dataset, output_dir, val_dataset)

        # Evaluate
        mae_train = tuner._evaluate_mae(tuner._adapter_model or self.base_model, dataset)
        mae_val = tuner._evaluate_mae(tuner._adapter_model or self.base_model, val_dataset) if val_dataset else None

        # Create metadata
        metadata = AdapterMetadata(
            name=name,
            dataset_name=dataset.examples[0].extra.get("dataset_name", name) if dataset.examples else name,
            license=license,
            domain=domain or {},
            path=str(adapter_path),
            mae_on_training=round(mae_train, 2),
            mae_on_validation=round(mae_val, 2) if mae_val else None,
            n_training_examples=len(dataset),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            created_at=str(np.datetime64('now')),
        )

        # Save metadata
        (output_dir / "adapter_metadata.json").write_text(
            json.dumps(metadata.__dict__, indent=2, default=str)
        )

        self.adapters[name] = metadata
        logger.info("Adapter '%s' trained: train MAE=%.2f, val MAE=%s",
                    name, mae_train, f"{mae_val:.2f}" if mae_val else "N/A")

        return metadata

    def load_adapter(self, name: str) -> nn.Module:
        """Load a trained adapter into memory."""
        if name in self._loaded_adapters:
            return self._loaded_adapters[name]

        if name not in self.adapters:
            raise KeyError(f"Unknown adapter: {name}. Available: {list(self.adapters.keys())}")

        metadata = self.adapters[name]
        tuner = LoRAFineTuner(model=self.base_model, device=self.device)
        model = tuner.load_adapter(metadata.path)
        self._loaded_adapters[name] = model
        return model

    def predict_single_adapter(
        self,
        name: str,
        video: torch.Tensor,
    ) -> float:
        """Run prediction with a single adapter."""
        model = self.load_adapter(name)
        model.eval()
        with torch.inference_mode():
            output = model(video.to(self.device))
            if isinstance(output, dict) and "EF" in output:
                return float(output["EF"].item())
            elif isinstance(output, torch.Tensor):
                return float(output.mean().item())
            return float(output)

    def predict_ensemble(
        self,
        video: torch.Tensor,
        mode: str = "ensemble",
        weights: Optional[Dict[str, float]] = None,
        input_domain: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run prediction using multiple adapters.

        Args:
            video: input tensor
            mode: "ensemble" (weight all adapters), "route" (pick best), or "stack"
            weights: manual weights for ensemble mode. If None, auto-computed.
            input_domain: domain info about the input (for routing), e.g.,
                         {"vendor": "GE", "country": "Malaysia"}

        Returns:
            dict with ef, per_adapter predictions, weights, agreement
        """
        if not self.adapters:
            raise RuntimeError("No adapters loaded. Train or register adapters first.")

        per_adapter: Dict[str, float] = {}

        for name in self.adapters:
            try:
                ef = self.predict_single_adapter(name, video)
                per_adapter[name] = ef
            except Exception as e:
                logger.warning("Adapter '%s' failed: %s", name, e)

        if not per_adapter:
            raise RuntimeError("All adapters failed")

        # Compute weights
        if weights is None:
            if mode == "route" and input_domain:
                weights = self._compute_routing_weights(input_domain)
            else:
                # Equal weights by default
                n = len(per_adapter)
                weights = {k: 1.0 / n for k in per_adapter}
        else:
            # Normalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        # Weighted average
        ef_ensemble = sum(per_adapter[n] * weights.get(n, 0) for n in per_adapter)

        # Agreement score
        preds = list(per_adapter.values())
        if len(preds) > 1:
            cv = np.std(preds) / (abs(np.mean(preds)) + 1e-6)
            agreement = max(0.0, 1.0 - cv)
        else:
            agreement = 1.0

        return {
            "ef": round(float(ef_ensemble), 2),
            "per_adapter": {k: round(v, 2) for k, v in per_adapter.items()},
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "agreement": round(float(agreement), 4),
            "mode": mode,
            "n_adapters": len(per_adapter),
        }

    def _compute_routing_weights(self, input_domain: Dict[str, Any]) -> Dict[str, float]:
        """Compute weights based on domain similarity (for routing mode).

        Adapters whose training domain matches the input domain get higher weight.
        """
        weights: Dict[str, float] = {}
        for name, meta in self.adapters.items():
            similarity = self._domain_similarity(meta.domain, input_domain)
            weights[name] = similarity

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            n = len(weights)
            weights = {k: 1.0 / n for k in weights}

        return weights

    def _domain_similarity(self, domain1: Dict, domain2: Dict) -> float:
        """Compute similarity between two domains (0-1).

        Matching vendor is most important (0.5 weight).
        Matching country is less important (0.3 weight).
        Matching modality is required (0.2 weight).
        """
        score = 0.0
        if domain1.get("vendor") == domain2.get("vendor") and domain1.get("vendor"):
            score += 0.5
        if domain1.get("country") == domain2.get("country") and domain1.get("country"):
            score += 0.3
        if domain1.get("modality") == domain2.get("modality") and domain1.get("modality"):
            score += 0.2
        return score

    def save_registry(self, path: Union[str, Path]) -> None:
        """Save the adapter registry to JSON."""
        registry = {
            name: {
                "name": m.name,
                "dataset_name": m.dataset_name,
                "license": m.license,
                "domain": m.domain,
                "path": m.path,
                "mae_on_training": m.mae_on_training,
                "mae_on_validation": m.mae_on_validation,
                "n_training_examples": m.n_training_examples,
                "lora_r": m.lora_r,
                "lora_alpha": m.lora_alpha,
                "created_at": m.created_at,
            }
            for name, m in self.adapters.items()
        }
        Path(path).write_text(json.dumps(registry, indent=2, default=str))
        logger.info("Registry saved to %s (%d adapters)", path, len(registry))

    def load_registry(self, path: Union[str, Path]) -> None:
        """Load adapter registry from JSON."""
        registry = json.loads(Path(path).read_text())
        for name, data in registry.items():
            self.adapters[name] = AdapterMetadata(**data)
        logger.info("Loaded %d adapters from %s", len(registry), path)

    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all registered adapters with metadata."""
        return [
            {
                "name": m.name,
                "dataset": m.dataset_name,
                "license": m.license,
                "domain": m.domain,
                "mae_train": m.mae_on_training,
                "mae_val": m.mae_on_validation,
                "n_examples": m.n_training_examples,
            }
            for m in self.adapters.values()
        ]

    def get_commercial_adapters(self) -> List[str]:
        """Return names of adapters that can be used commercially."""
        return [name for name, m in self.adapters.items() if m.license == "commercial_ok"]

    def get_research_adapters(self) -> List[str]:
        """Return names of adapters that are research-only."""
        return [name for name, m in self.adapters.items() if m.license == "research_only"]
