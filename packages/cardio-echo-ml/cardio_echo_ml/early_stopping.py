"""Early Stopping Callback — prevents overfitting in LoRA fine-tuning.

In medical AI, overfitting is dangerous: the model memorizes training data
instead of generalizing, leading to inflated accuracy on training set but
poor real-world performance. Early stopping monitors validation metrics and
halts training when improvement plateaus.

Three monitoring modes:
1. MAE-based (for regression tasks like EF)
   - Stop when validation MAE hasn't improved for N epochs
   - Restore best weights (lowest val MAE)

2. Loss-based (for classification tasks)
   - Stop when validation cross-entropy hasn't improved for N epochs
   - Restore best weights (lowest val loss)

3. Clinical metrics-based (for FN-critical tasks)
   - Stop when sensitivity drops below threshold
   - Or when specificity drops below threshold
   - Prioritize clinical safety over raw accuracy

Usage:
    from cardio_echo_ml.early_stopping import EarlyStopping

    # For EF regression (MAE-based)
    early_stop = EarlyStopping(
        monitor="val_mae",
        patience=5,
        min_delta=0.1,  # Need 0.1 EF% improvement to count
        mode="min",  # Lower is better
        restore_best=True,
    )

    for epoch in range(100):
        train_loss = train_one_epoch()
        val_mae = evaluate_validation()

        early_stop(epoch, val_mae=val_mae, model=model)
        if early_stop.should_stop:
            print(f"Stopped at epoch {epoch}")
            break

    # Best weights automatically restored if restore_best=True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    monitor: str = "val_loss"  # "val_loss", "val_mae", "val_sensitivity", "val_specificity"
    patience: int = 5  # Number of epochs without improvement before stopping
    min_delta: float = 0.001  # Minimum change to qualify as improvement
    mode: str = "min"  # "min" (lower is better) or "max" (higher is better)
    restore_best: bool = True  # Restore best weights after stopping
    # Clinical safety guards
    min_sensitivity: float = 0.0  # Stop if sensitivity drops below this
    min_specificity: float = 0.0  # Stop if specificity drops below this
    # Warmup: don't stop in first N epochs (let model learn)
    warmup_epochs: int = 2


class EarlyStopping:
    """Early stopping callback for medical AI training.

    Monitors validation metrics and stops training when:
    1. Monitored metric hasn't improved for `patience` epochs
    2. Clinical safety threshold violated (sensitivity/specificity too low)
    3. Training is clearly diverging (loss increasing)
    """

    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False
        self.stopped_epoch: int = 0
        self.best_state_dict: Optional[dict] = None
        self.history: list = []

    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        val_loss: Optional[float] = None,
        val_mae: Optional[float] = None,
        val_sensitivity: Optional[float] = None,
        val_specificity: Optional[float] = None,
        train_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check if training should stop.

        Args:
            epoch: current epoch number
            model: the model being trained (for best weight saving)
            val_loss: validation cross-entropy loss
            val_mae: validation MAE (for regression tasks)
            val_sensitivity: validation sensitivity (for classification)
            val_specificity: validation specificity (for classification)
            train_loss: training loss (for divergence detection)

        Returns:
            dict with status information
        """
        # Warmup: don't check in first N epochs
        if epoch < self.config.warmup_epochs:
            logger.info("Early stopping: warmup epoch %d/%d, skipping check",
                        epoch + 1, self.config.warmup_epochs)
            return {"should_stop": False, "reason": "warmup"}

        # Determine the metric to monitor
        metric_value: Optional[float] = None
        if self.config.monitor == "val_loss" and val_loss is not None:
            metric_value = val_loss
        elif self.config.monitor == "val_mae" and val_mae is not None:
            metric_value = val_mae
        elif self.config.monitor == "val_sensitivity" and val_sensitivity is not None:
            metric_value = val_sensitivity
        elif self.config.monitor == "val_specificity" and val_specificity is not None:
            metric_value = val_specificity

        # Record history
        record = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_sensitivity": val_sensitivity,
            "val_specificity": val_specificity,
            "train_loss": train_loss,
        }
        self.history.append(record)

        # Clinical safety check (override everything)
        if val_sensitivity is not None and self.config.min_sensitivity > 0:
            if val_sensitivity < self.config.min_sensitivity:
                logger.warning(
                    "⚠️  CLINICAL SAFETY: val_sensitivity=%.3f < min=%.3f — STOPPING",
                    val_sensitivity, self.config.min_sensitivity
                )
                self.should_stop = True
                self.stopped_epoch = epoch
                self._restore_best(model)
                return {"should_stop": True, "reason": "sensitivity_below_threshold"}

        if val_specificity is not None and self.config.min_specificity > 0:
            if val_specificity < self.config.min_specificity:
                logger.warning(
                    "⚠️  CLINICAL SAFETY: val_specificity=%.3f < min=%.3f — STOPPING",
                    val_specificity, self.config.min_specificity
                )
                self.should_stop = True
                self.stopped_epoch = epoch
                self._restore_best(model)
                return {"should_stop": True, "reason": "specificity_below_threshold"}

        # Divergence check (train loss increasing)
        if train_loss is not None and len(self.history) >= 3:
            recent_train_losses = [h["train_loss"] for h in self.history[-3:] if h["train_loss"] is not None]
            if len(recent_train_losses) == 3:
                if recent_train_losses[-1] > recent_train_losses[0] * 1.5:
                    logger.warning("⚠️  Training diverging: loss %.4f → %.4f — STOPPING",
                                  recent_train_losses[0], recent_train_losses[-1])
                    self.should_stop = True
                    self.stopped_epoch = epoch
                    self._restore_best(model)
                    return {"should_stop": True, "reason": "training_diverging"}

        # Standard early stopping (metric not improving)
        if metric_value is None:
            return {"should_stop": False, "reason": "no_metric"}

        # Check if this is an improvement
        is_improvement = False
        if self.best_score is None:
            is_improvement = True
        elif self.config.mode == "min":
            # Lower is better (e.g., MAE, loss)
            if metric_value < self.best_score - self.config.min_delta:
                is_improvement = True
        else:
            # Higher is better (e.g., sensitivity, specificity)
            if metric_value > self.best_score + self.config.min_delta:
                is_improvement = True

        if is_improvement:
            self.best_score = metric_value
            self.best_epoch = epoch
            self.counter = 0
            # Save best weights
            if self.config.restore_best:
                self.best_state_dict = {
                    k: v.clone() for k, v in model.state_dict().items()
                    if v.requires_grad or any(p.requires_grad for p in model.parameters())
                }
                # Simpler: just save all
                self.best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

            logger.info("✅ New best %s=%.4f at epoch %d",
                       self.config.monitor, metric_value, epoch + 1)
        else:
            self.counter += 1
            logger.info("No improvement in %s for %d/%d epochs (best=%.4f at epoch %d)",
                       self.config.monitor, self.counter, self.config.patience,
                       self.best_score or 0, self.best_epoch + 1)

            if self.counter >= self.config.patience:
                logger.info("⏹️  Early stopping triggered at epoch %d "
                           "(no improvement for %d epochs)",
                           epoch + 1, self.config.patience)
                self.should_stop = True
                self.stopped_epoch = epoch
                self._restore_best(model)
                return {"should_stop": True, "reason": "no_improvement"}

        return {
            "should_stop": False,
            "reason": "training",
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
        }

    def _restore_best(self, model: torch.nn.Module) -> None:
        """Restore best model weights."""
        if self.config.restore_best and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info("✅ Restored best weights from epoch %d (score=%.4f)",
                       self.best_epoch + 1, self.best_score or 0)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of early stopping history."""
        return {
            "monitor": self.config.monitor,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
            "patience": self.config.patience,
            "total_epochs_run": len(self.history),
            "should_stop": self.should_stop,
            "history": self.history,
        }


# Preset configurations for different task types
def get_preset_config(task_type: str) -> EarlyStoppingConfig:
    """Get preset early stopping config for different task types.

    Args:
        task_type: "ef_regression", "binary_classification", "fn_critical",
                   "multiclass_classification"
    """
    presets = {
        "ef_regression": EarlyStoppingConfig(
            monitor="val_mae",
            patience=5,
            min_delta=0.1,  # Need 0.1 EF% improvement
            mode="min",
            restore_best=True,
            warmup_epochs=2,
        ),
        "binary_classification": EarlyStoppingConfig(
            monitor="val_loss",
            patience=7,
            min_delta=0.001,
            mode="min",
            restore_best=True,
            warmup_epochs=2,
        ),
        "fn_critical": EarlyStoppingConfig(
            monitor="val_sensitivity",
            patience=5,
            min_delta=0.005,  # Need 0.5% sensitivity improvement
            mode="max",  # Higher sensitivity is better
            restore_best=True,
            min_sensitivity=0.85,  # Stop if sensitivity drops below 85%
            warmup_epochs=3,
        ),
        "multiclass_classification": EarlyStoppingConfig(
            monitor="val_loss",
            patience=7,
            min_delta=0.001,
            mode="min",
            restore_best=True,
            warmup_epochs=2,
        ),
    }
    return presets.get(task_type, presets["binary_classification"])
