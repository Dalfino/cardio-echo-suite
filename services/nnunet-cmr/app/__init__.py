"""nnU-Net CMR service — automatic cardiac MRI segmentation.

Wraps upstream nnU-Net (MIC-DKFZ) configured for cardiac MRI short-axis
ventricle/myocardium segmentation (ACDC, M&Ms challenges).

Unlike MedSAM2 (promptable), nnU-Net is fully automatic: give it a CMR stack
and it returns LV/blood pool/myocardium masks with no prompts.

Tasks:
- LV endocardium (label 1)
- LV myocardium (label 2)
- RV endocardium (label 3)
"""

__version__ = "0.1.0"
