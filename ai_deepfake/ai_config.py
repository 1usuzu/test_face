"""
ai_config.py - Cấu hình tập trung cho AI Module
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings # Cần cài đặt: pip install pydantic-settings

class AISettings(BaseSettings):
    # Cấu hình Model
    MODEL_DIR: Path = Path(__file__).parent / "models"
    DEVICE: str = "cuda" if os.environ.get("USE_GPU", "true").lower() == "true" else "cpu"
    
    # Cấu hình Deepfake Detection
    DEFAULT_THRESHOLD: float = 0.50
    V1_WEIGHT: float = 0.4  # EfficientNet-B0
    V2_WEIGHT: float = 0.6  # EfficientNet-B4
    
    # Toggles
    ENABLE_TTA: bool = True     # Test Time Augmentation (Tăng độ chính xác, giảm tốc độ)
    ENABLE_SIGNAL_ANALYSIS: bool = True
    
    class Config:
        env_file = ".env"  # Có thể load từ file .env

settings = AISettings()

# Đảm bảo thư mục models tồn tại
if not settings.MODEL_DIR.exists():
    print(f"Warning: Model directory not found at {settings.MODEL_DIR}")