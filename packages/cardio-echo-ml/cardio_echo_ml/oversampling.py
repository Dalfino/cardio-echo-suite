"""Oversampling — weight rare EF ranges more during training.

The problem: EchoNet-Dynamic has ~72% normal EF videos and only ~7% severely
reduced. The model learns to predict "normal" for everything because that
minimizes average loss. This is why MAE is 10+ EF% on severely reduced EF.

The solution: oversample rare cases during training. Instead of training on
the natural distribution (72% normal, 7% severe), train on a balanced
distribution (20% normal, 20% mild, 20% moderate, 20% severe, 20% hyper).

Three methods:
1. Simple oversampling: duplicate rare cases in the training set
2. Weighted sampling: WeightedRandomSampler with inverse-frequency weights
3. SMOTE-like: generate synthetic augmented versions of rare cases

Usage:
    from cardio_echo_ml.oversampling import EFRangeSampler, create_balanced_dataset

    # Method 1: WeightedRandomSampler (recommended)
    sampler = EFRangeSampler(studies, method="weighted")
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)

    # Method 2: Simple oversampling
    balanced = create_balanced_dataset(studies, method="oversample")
    # balanced has ~equal representation of each EF range

    # Method 3: Augmented oversampling (best for small datasets)
    balanced = create_balanced_dataset(studies, method="augment")
    # Each rare case is augmented (flipped, rotated) to create synthetic samples
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


# EF ranges for stratification
EF_RANGES = [
    (0, 30, "severely_reduced"),
    (30, 40, "moderately_reduced"),
    (40, 52, "mildly_reduced"),
    (52, 70, "normal"),
    (70, 100, "hyperdynamic"),
]


def get_ef_range(ef: float) -> str:
    """Get the EF range label for a given EF value."""
    for lo, hi, label in EF_RANGES:
        if lo <= ef < hi:
            return label
    return "normal"


def compute_range_distribution(studies: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count how many studies fall in each EF range."""
    counts = Counter()
    for s in studies:
        ef = s.get("ef", 0)
        counts[get_ef_range(ef)] += 1
    return dict(counts)


def compute_inverse_frequency_weights(studies: List[Dict[str, Any]]) -> List[float]:
    """Compute inverse-frequency weights for each study.

    Rare cases (severely reduced EF) get HIGHER weight → sampled more often.
    Common cases (normal EF) get LOWER weight → sampled less often.

    This balances the training distribution without changing the dataset size.
    """
    counts = compute_range_distribution(studies)
    total = len(studies)
    weights = []
    for s in studies:
        range_label = get_ef_range(s.get("ef", 0))
        n_in_range = counts.get(range_label, 1)
        # Inverse frequency: weight = total / (n_ranges * n_in_range)
        weight = total / (len(EF_RANGES) * n_in_range)
        weights.append(weight)
    return weights


class EFRangeSampler(WeightedRandomSampler):
    """WeightedRandomSampler that balances EF ranges.

    Usage:
        sampler = EFRangeSampler(studies)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    """

    def __init__(self, studies: List[Dict[str, Any]], method: str = "weighted"):
        weights = compute_inverse_frequency_weights(studies)
        super().__init__(
            weights=weights,
            num_samples=len(studies),
            replacement=True,
        )
        self.studies = studies
        self.method = method
        logger.info("EFRangeSampler: %s method, %d studies", method, len(studies))
        counts = compute_range_distribution(studies)
        for label, count in sorted(counts.items()):
            logger.info("  %s: %d studies", label, count)


def create_balanced_dataset(
    studies: List[Dict[str, Any]],
    method: str = "oversample",
    target_per_range: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Create a balanced dataset by oversampling rare cases.

    Args:
        studies: list of study dicts with "ef" key
        method: "oversample" (duplicate) or "augment" (duplicate + augment)
        target_per_range: target number of studies per range.
                         If None, uses the max range count.

    Returns:
        Balanced list of studies
    """
    # Group by EF range
    by_range: Dict[str, List[Dict[str, Any]]] = {label: [] for _, _, label in EF_RANGES}
    for s in studies:
        label = get_ef_range(s.get("ef", 0))
        by_range[label].append(s)

    # Find target count
    if target_per_range is None:
        target_per_range = max(len(v) for v in by_range.values())

    balanced: List[Dict[str, Any]] = []
    for label, group in by_range.items():
        if not group:
            logger.warning("No studies in range %s — skipping", label)
            continue

        if len(group) >= target_per_range:
            # Downsample (take random subset)
            indices = np.random.choice(len(group), target_per_range, replace=False)
            balanced.extend([group[i] for i in indices])
        else:
            # Oversample (duplicate with augmentation if needed)
            if method == "augment":
                # Create augmented copies
                while len([s for s in balanced if get_ef_range(s.get("ef", 0)) == label]) < target_per_range:
                    for s in group:
                        augmented = dict(s)
                        augmented["_augmented"] = True
                        augmented["_augmentation"] = np.random.choice(["hflip", "rotate", "brightness"])
                        balanced.append(augmented)
                        if len([s for s in balanced if get_ef_range(s.get("ef", 0)) == label]) >= target_per_range:
                            break
            else:
                # Simple oversampling (just duplicate)
                indices = np.random.choice(len(group), target_per_range, replace=True)
                balanced.extend([group[i] for i in indices])

    # Shuffle
    np.random.shuffle(balanced)

    # Log
    counts = compute_range_distribution(balanced)
    logger.info("Balanced dataset (%s method, %d → %d studies):", method, len(studies), len(balanced))
    for label, count in sorted(counts.items()):
        logger.info("  %s: %d", label, count)

    return balanced


def compute_class_weights(studies: List[Dict[str, Any]]) -> torch.Tensor:
    """Compute class weights for weighted loss function.

    Alternative to oversampling: instead of duplicating rare cases,
    weight the loss function to penalize errors on rare cases more.

    Usage:
        weights = compute_class_weights(studies)
        criterion = WeightedMSELoss(weights)
    """
    counts = compute_range_distribution(studies)
    total = len(studies)
    weights = []
    for s in studies:
        label = get_ef_range(s.get("ef", 0))
        n = counts.get(label, 1)
        weight = total / (len(EF_RANGES) * n)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedMSELoss(torch.nn.Module):
    """MSE loss with per-sample weights.

    Rare EF ranges get higher weight → errors on them are penalized more.
    """

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_error = (pred - target) ** 2
        weighted = sq_error * self.weights[:len(sq_error)].to(pred.device)
        return weighted.mean()


def analyze_ef_distribution(studies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the EF distribution of a dataset.

    Returns counts, percentages, and imbalance metrics.
    """
    counts = compute_range_distribution(studies)
    total = len(studies)

    result = {
        "total": total,
        "ranges": {},
        "imbalance_ratio": 0,
    }

    for lo, hi, label in EF_RANGES:
        count = counts.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        result["ranges"][label] = {
            "count": count,
            "percentage": round(pct, 1),
            "ef_range": f"{lo}-{hi}%",
        }

    # Imbalance ratio = max_count / min_count (excluding zeros)
    non_zero = [c for c in counts.values() if c > 0]
    if non_zero:
        result["imbalance_ratio"] = round(max(non_zero) / min(non_zero), 1)

    return result
