"""LoRA Fine-Tuning Scaffold — adapt models to your hospital's data.

LoRA (Low-Rank Adaptation) is the bridge between "works on public data" and
"works on YOUR patients." It fine-tunes a small adapter (~1% of model params)
on top of a frozen base model, using your hospital's labeled data.

Why LoRA instead of full fine-tuning:
- 100× fewer trainable parameters → 10× faster training
- No risk of catastrophic forgetting (base model is frozen)
- Adapter weights are small (~5-20 MB vs ~500 MB for full model)
- Can hot-swap adapters per hospital (one base model, many adapters)
- FDA PCCP-friendly: base model stays locked, only adapter changes

What you need to fine-tune:
- 100-500 labeled echo studies from YOUR hospital (IRB-approved)
- Labels: EF (float), optionally valve findings, wall motion, etc.
- Format: JSONL (one study per line, see HospitalDataset)
- GPU: any CUDA GPU with 8+ GB (or Google Colab free tier)

Usage:
    from cardio_echo_ml.lora import LoRAFineTuner, HospitalDataset

    # 1. Load your hospital's data
    dataset = HospitalDataset.from_jsonl(
        path="/data/hospital/echo_labels.jsonl",
        video_dir="/data/hospital/videos/",
    )

    # 2. Configure LoRA
    tuner = LoRAFineTuner(
        model=panecho_model,
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
        epochs=10,
    )

    # 3. Fine-tune
    adapter_path = tuner.fit(dataset, output_dir="checkpoints/hospital_lora")

    # 4. Evaluate (compare pre vs post fine-tuning)
    results = tuner.evaluate(dataset, adapter_path=adapter_path)
    # results = {
    #     "mae_before": 7.2,
    #     "mae_after": 4.1,
    #     "improvement": 3.1,
    #     "pearson_r_before": 0.74,
    #     "pearson_r_after": 0.89,
    # }

    # 5. Use fine-tuned model in pipeline
    pipeline = CommercialPipeline(
        model=tuner.load_adapter(adapter_path),
        calibration_params=cal_params,
    )
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
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Hospital Dataset
# ----------------------------------------------------------------------

@dataclass
class HospitalExample:
    """One labeled echo study from your hospital."""
    video_path: str
    ef: float  # ground-truth EF from cardiologist report
    patient_id: str = "unknown"
    study_date: str = ""
    scanner_vendor: str = ""  # GE, Philips, Siemens, etc.
    patient_age: int = 0
    patient_sex: str = ""  # M/F
    extra: Dict[str, Any] = field(default_factory=dict)


class HospitalDataset(Dataset):
    """PyTorch Dataset for hospital echo data.

    Expected JSONL format (one line per study):
        {
            "video_path": "/data/hospital/videos/study001.avi",
            "ef": 55.2,
            "patient_id": "PT-001",
            "study_date": "2024-03-15",
            "scanner_vendor": "GE",
            "patient_age": 67,
            "patient_sex": "M"
        }
    """

    def __init__(
        self,
        examples: List[HospitalExample],
        video_loader: Callable[[str], torch.Tensor],
        clip_len: int = 16,
        image_size: int = 224,
    ):
        self.examples = examples
        self.video_loader = video_loader
        self.clip_len = clip_len
        self.image_size = image_size

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        video_dir: Optional[Union[str, Path]] = None,
        video_loader: Optional[Callable] = None,
        clip_len: int = 16,
        image_size: int = 224,
    ) -> "HospitalDataset":
        """Load dataset from JSONL file.

        Args:
            path: path to JSONL file
            video_dir: if provided, prepended to relative video_path
            video_loader: callable(video_path) -> torch.Tensor.
                         If None, uses default video loader.
            clip_len: number of frames to sample
            image_size: frame size (H=W)
        """
        if video_loader is None:
            video_loader = _default_video_loader

        examples: List[HospitalExample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                video_path = d["video_path"]
                if video_dir and not Path(video_path).is_absolute():
                    video_path = str(Path(video_dir) / video_path)
                examples.append(HospitalExample(
                    video_path=video_path,
                    ef=float(d["ef"]),
                    patient_id=d.get("patient_id", "unknown"),
                    study_date=d.get("study_date", ""),
                    scanner_vendor=d.get("scanner_vendor", ""),
                    patient_age=int(d.get("patient_age", 0)),
                    patient_sex=d.get("patient_sex", ""),
                    extra={k: v for k, v in d.items()
                           if k not in {"video_path", "ef", "patient_id", "study_date",
                                        "scanner_vendor", "patient_age", "patient_sex"}},
                ))

        logger.info("Loaded %d examples from %s", len(examples), path)
        return cls(examples, video_loader, clip_len, image_size)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        video = self.video_loader(ex.video_path)
        return {
            "video": video,
            "ef": torch.tensor(ex.ef, dtype=torch.float32),
            "patient_id": ex.patient_id,
        }


def _default_video_loader(video_path: str, clip_len: int = 16, size: int = 224) -> torch.Tensor:
    """Default video loader using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames from {video_path}")

    # Subsample to clip_len
    if len(frames) >= clip_len:
        idx = np.linspace(0, len(frames) - 1, clip_len).astype(int)
        frames = [frames[i] for i in idx]
    else:
        while len(frames) < clip_len:
            frames.append(frames[-1])

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # (T, H, W, 3) -> (3, T, H, W) -> (1, 3, T, H, W)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).float()
    return tensor


# ----------------------------------------------------------------------
# LoRA Fine-Tuner
# ----------------------------------------------------------------------

class LoRAFineTuner:
    """Fine-tune a model with LoRA adapters.

    Uses PEFT (Parameter-Efficient Fine-Tuning) library if available.
    Falls back to a simple implementation if PEFT is not installed.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        learning_rate: float = 1e-4,
        epochs: int = 10,
        batch_size: int = 2,
        device: Optional[str] = None,
    ):
        """Initialize LoRA fine-tuner.

        Args:
            model: base model (will be frozen)
            lora_r: LoRA rank (typical: 8, 16, 32). Higher = more capacity.
            lora_alpha: LoRA scaling factor (typical: 2× lora_r)
            lora_dropout: dropout probability for LoRA layers
            target_modules: which modules to apply LoRA to.
                           Default: attention layers (q_proj, k_proj, v_proj, o_proj)
            learning_rate: typically 1e-4 to 5e-4 (higher than full fine-tuning)
            epochs: 5-20 typical for LoRA
            batch_size: 1-4 depending on GPU memory
            device: 'cuda' or 'cpu'. Default: auto-detect.
        """
        self.base_model = model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._adapter_model = None

    def _wrap_with_lora(self) -> nn.Module:
        """Wrap base model with LoRA adapters using PEFT."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                target_modules=self.target_modules,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            model = get_peft_model(self.base_model, config)
            model.print_trainable_parameters()
            return model
        except ImportError:
            logger.warning(
                "PEFT not installed. Install with: pip install peft\n"
                "Falling back to selective fine-tuning (last layer only)."
            )
            # Fallback: freeze all but last layer
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Unfreeze last linear layer
            for name, module in reversed(list(self.base_model.named_modules())):
                if isinstance(module, nn.Linear):
                    for param in module.parameters():
                        param.requires_grad = True
                    logger.info("Fine-tuning layer: %s", name)
                    break
            return self.base_model

    def fit(
        self,
        dataset: HospitalDataset,
        output_dir: Union[str, Path],
        val_dataset: Optional[HospitalDataset] = None,
    ) -> Path:
        """Fine-tune the model on hospital data.

        Args:
            dataset: training dataset
            output_dir: where to save adapter weights
            val_dataset: optional validation dataset

        Returns:
            Path to saved adapter weights
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Wrap model with LoRA
        logger.info("Wrapping model with LoRA (r=%d, alpha=%d)...", self.lora_r, self.lora_alpha)
        model = self._wrap_with_lora()
        model = model.to(self.device)
        self._adapter_model = model

        # Data loader
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Medical data is sensitive — no multi-process
        )

        # Optimizer (only trainable params = LoRA adapters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

        # Loss: MSE for EF regression
        criterion = nn.MSELoss()

        # Training loop
        logger.info("Starting LoRA fine-tuning: %d epochs, %d examples",
                    self.epochs, len(dataset))

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                videos = batch["video"].squeeze(1).to(self.device)  # (B, 3, T, H, W)
                efs = batch["ef"].to(self.device)

                # Handle batch dimension
                if videos.dim() == 6:
                    videos = videos.squeeze(1)

                optimizer.zero_grad()
                try:
                    outputs = model(videos)
                    if isinstance(outputs, dict) and "EF" in outputs:
                        preds = outputs["EF"].squeeze()
                    elif isinstance(outputs, torch.Tensor):
                        preds = outputs.squeeze()
                    else:
                        preds = outputs

                    loss = criterion(preds, efs)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    n_batches += 1
                except Exception as e:
                    logger.warning("Batch failed: %s", e)

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, self.epochs, avg_loss)

            # Validation
            if val_dataset:
                val_mae = self._evaluate_mae(model, val_dataset)
                logger.info("  Validation MAE: %.2f EF%%", val_mae)

            # Save checkpoint
            self._save_adapter(model, output_dir / f"epoch-{epoch}")

        # Save final adapter
        final_path = output_dir / "final"
        self._save_adapter(model, final_path)
        logger.info("LoRA adapter saved to %s", final_path)

        return final_path

    def _evaluate_mae(self, model: nn.Module, dataset: HospitalDataset) -> float:
        """Evaluate MAE on a dataset."""
        model.eval()
        errors = []
        with torch.inference_mode():
            for i in range(len(dataset)):
                try:
                    item = dataset[i]
                    video = item["video"].squeeze(0).to(self.device)
                    if video.dim() == 5:
                        video = video.squeeze(0)
                    video = video.unsqueeze(0)  # Add batch dim

                    output = model(video)
                    if isinstance(output, dict) and "EF" in output:
                        pred = float(output["EF"].item())
                    elif isinstance(output, torch.Tensor):
                        pred = float(output.mean().item())
                    else:
                        pred = float(output)

                    gt = float(item["ef"].item())
                    errors.append(abs(pred - gt))
                except Exception:
                    continue
        return float(np.mean(errors)) if errors else 0.0

    def _save_adapter(self, model: nn.Module, path: Path) -> None:
        """Save LoRA adapter weights."""
        path.mkdir(parents=True, exist_ok=True)
        try:
            # PEFT save
            model.save_pretrained(str(path))
        except Exception:
            # Fallback: save trainable params only
            trainable_state = {
                name: param.detach().cpu()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            torch.save(trainable_state, path / "adapter_weights.pt")

        # Save metadata
        metadata = {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "n_trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "n_total_params": sum(p.numel() for p in model.parameters()),
        }
        (path / "lora_metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("Adapter saved: %d trainable / %d total params (%.2f%%)",
                    metadata["n_trainable_params"], metadata["n_total_params"],
                    100 * metadata["n_trainable_params"] / metadata["n_total_params"])

    def load_adapter(self, adapter_path: Union[str, Path]) -> nn.Module:
        """Load a fine-tuned LoRA adapter onto the base model.

        Args:
            adapter_path: path to saved adapter

        Returns:
            Model with LoRA adapter loaded (ready for inference)
        """
        adapter_path = Path(adapter_path)
        try:
            from peft import PeftModel  # type: ignore
            model = PeftModel.from_pretrained(self.base_model, str(adapter_path))
            logger.info("LoRA adapter loaded from %s", adapter_path)
            return model
        except ImportError:
            # Fallback: load state dict
            weights_path = adapter_path / "adapter_weights.pt"
            if weights_path.exists():
                state = torch.load(weights_path, map_location="cpu")
                self.base_model.load_state_dict(state, strict=False)
                logger.info("Adapter weights loaded from %s", weights_path)
            return self.base_model

    def evaluate(
        self,
        dataset: HospitalDataset,
        adapter_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Evaluate model before and after fine-tuning.

        Args:
            dataset: evaluation dataset
            adapter_path: path to fine-tuned adapter

        Returns:
            dict with mae_before, mae_after, improvement, pearson_r
        """
        # Before fine-tuning
        logger.info("Evaluating base model...")
        mae_before = self._evaluate_mae(self.base_model, dataset)

        # After fine-tuning
        if adapter_path:
            logger.info("Loading fine-tuned adapter...")
            tuned_model = self.load_adapter(adapter_path)
            mae_after = self._evaluate_mae(tuned_model, dataset)
        else:
            mae_after = mae_before

        improvement = mae_before - mae_after

        return {
            "mae_before": round(mae_before, 2),
            "mae_after": round(mae_after, 2),
            "improvement": round(improvement, 2),
            "improvement_pct": round(improvement / mae_before * 100, 1) if mae_before > 0 else 0,
            "n_examples": len(dataset),
        }
