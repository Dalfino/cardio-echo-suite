"""MedSAM2 service — promptable 3D medical segmentation.

Wraps upstream MedSAM2 (Bowang-lab/MedSAM2 on HuggingFace) — a SAM-2 variant
fine-tuned on medical images. Excellent for cardiac chambers, aorta, lesions.

Unlike the other services (which are classification/regression), MedSAM2 is
*promptable*: the user provides point/bbox prompts and the model returns a mask.

Endpoints:
- POST /predict/point  — single-point prompt
- POST /predict/bbox   — bounding-box prompt
- POST /predict/auto   — automatic segmentation (no prompt, runs default points)
"""

__version__ = "0.1.0"
