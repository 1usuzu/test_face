"""
ai_config.py - Cấu hình tập trung cho AI Module
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

import torch

def _resolve_device() -> str:
    if os.environ.get("USE_GPU", "true").lower() != "true":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

_default_model_dir = str(Path(__file__).parent / "models")

class AISettings(BaseSettings):
    model_config = {"extra": "ignore"}

    MODEL_DIR: Path = Field(default=Path(__file__).parent / "models")
    DEVICE: str = Field(default_factory=_resolve_device)
    
    # C1 policy decision: default balanced operating threshold.
    DEFAULT_THRESHOLD: float = 0.65
    V1_WEIGHT: float = 0.4
    V2_WEIGHT: float = 0.6
    
    ENABLE_TTA: bool = False
    ENABLE_SIGNAL_ANALYSIS: bool = True
    # Face detection backend:
    # - "haar": lightweight OpenCV Haar cascade (recommended for low-memory hosts)
    # - "mtcnn": facenet-pytorch MTCNN
    # - "none": skip face detection and use center-crop fallback
    FACE_DETECTOR_BACKEND: str = "haar"
    # If true, detector returns NO_FACE / FACE_DETECTION_ERROR instead of fallback center-crop.
    FACE_DETECTION_STRICT: bool = False

    SIGNAL_LAPLACIAN_THRESHOLD: float = 100.0
    SIGNAL_HIGH_FREQ_THRESHOLD: float = 13.0
    SIGNAL_BOOST_STEP: float = 0.03

settings = AISettings()

if not settings.MODEL_DIR.exists():
    print(f"Warning: Model directory not found at {settings.MODEL_DIR}")