"""Pre-read CLI for PanEcho.

Usage:
    python -m app.cli predict /path/to/echo.mp4 \\
        --patient-id P1 \\
        --study-uid 1.2.840.113619.2.55.3.604688119.971 \\
        --output report.json

Outputs a structured JSON report (see `app.fhir.mapping.build_preread_report`).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

from .fhir import build_preread_report


def _load_video(path: Path, clip_len: int = 16, size: int = 224) -> torch.Tensor:
    """Load a video and prepare a (1, 3, T, H, W) tensor, ImageNet-normalized."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
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
        raise ValueError(f"No frames decoded from {path}")

    # Subsample / pad to clip_len
    if len(frames) >= clip_len:
        idx = torch.linspace(0, len(frames) - 1, clip_len).long().tolist()
        frames = [frames[i] for i in idx]
    else:
        # Pad by repeating last frame
        last = frames[-1]
        while len(frames) < clip_len:
            frames.append(last)

    arr = np.stack(frames, axis=0)  # (T, H, W, 3)
    arr = arr.astype(np.float32) / 255.0

    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # (T, H, W, 3) -> (1, 3, T, H, W)
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    return t


def predict(
    video_path: Path,
    patient_id: str = "unknown",
    encounter_id: str | None = None,
    study_uid: str | None = None,
    clip_len: int = 16,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a PanEcho pre-read on a video file."""
    from .model import PanEchoModel

    video = _load_video(video_path, clip_len=clip_len)

    if dry_run:
        # Don't load the model — return a stub with the EF regression + a few
        # classifications, useful for testing the FHIR mapping end-to-end.
        predictions = {
            "EF": torch.tensor([55.0]),
            "AVStenosis": torch.tensor([[0.85, 0.10, 0.04, 0.01]]),
            "MVRegurgitation": torch.tensor([[0.70, 0.20, 0.08, 0.02]]),
            "PericardialEffusion": torch.tensor([[0.92, 0.05, 0.02, 0.01]]),
            "TamponadePhysiology": torch.tensor([[0.97, 0.03]]),
            "LVSystolicFunction": torch.tensor([[0.65, 0.25, 0.07, 0.03]]),
        }
    else:
        m = PanEchoModel.get(clip_len=clip_len)
        m.load()
        predictions = m.predict(video)

    report = build_preread_report(
        predictions=predictions,
        patient_id=patient_id,
        encounter_id=encounter_id,
        study_instance_uid=study_uid,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="panecho-cli", description="PanEcho pre-read CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("predict", help="Run PanEcho pre-read on a video file")
    pp.add_argument("video", type=Path, help="Path to echo video (MP4/AVI)")
    pp.add_argument("--patient-id", default="unknown")
    pp.add_argument("--encounter-id", default=None)
    pp.add_argument("--study-uid", default=None, help="DICOM StudyInstanceUID")
    pp.add_argument("--clip-len", type=int, default=16)
    pp.add_argument("--output", "-o", type=Path, default=None, help="Write JSON to this path (default: stdout)")
    pp.add_argument("--dry-run", action="store_true",
                    help="Skip model loading; emit a stub report for testing")

    args = p.parse_args(argv)

    if args.cmd == "predict":
        if not args.video.exists():
            print(f"Error: video file not found: {args.video}", file=sys.stderr)
            return 2
        try:
            report = predict(
                args.video,
                patient_id=args.patient_id,
                encounter_id=args.encounter_id,
                study_uid=args.study_uid,
                clip_len=args.clip_len,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"Prediction failed: {e}", file=sys.stderr)
            return 1

        out = json.dumps(report, indent=2, default=str)
        if args.output:
            args.output.write_text(out)
            print(f"Report written to {args.output}", file=sys.stderr)
        else:
            print(out)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
