"""
ML Service Configuration
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Service info
    APP_NAME: str = "Hotel Cash Detector ML Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Models directory
    MODELS_DIR: str = "/app/models"

    # GPU settings
    USE_GPU: str = "auto"  # "auto", "True", "False"

    # Model names
    CASH_POSE_MODEL: str = "yolov8s-pose.pt"
    VIOLENCE_POSE_MODEL: str = "yolov8n-pose.pt"
    FIRE_YOLO_MODEL: str = "yolov8n.pt"
    FIRE_MODEL: str = "fire_smoke_yolov8.pt"

    # Gemini API
    GEMINI_API_KEY: str = ""

    # Detection defaults
    CASH_CONFIDENCE: float = 0.5
    VIOLENCE_CONFIDENCE: float = 0.6
    FIRE_CONFIDENCE: float = 0.5
    POSE_CONFIDENCE: float = 0.5

    # Timeouts
    DETECTION_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton settings instance
settings = Settings()


def get_device():
    """Determine the device to use for inference"""
    import torch

    use_gpu = settings.USE_GPU.lower()

    if use_gpu == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif use_gpu == "true":
        if torch.cuda.is_available():
            return "cuda"
        print("[WARNING] GPU requested but CUDA not available, using CPU")
        return "cpu"
    else:
        return "cpu"


def get_gpu_info() -> dict:
    """Get GPU information"""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "memory_used_mb": None,
        "memory_total_mb": None,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()

        if info["gpu_count"] > 0:
            info["gpu_name"] = torch.cuda.get_device_name(0)

            # Memory info
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory

            info["memory_used_mb"] = round(memory_allocated / (1024 * 1024), 2)
            info["memory_total_mb"] = round(memory_total / (1024 * 1024), 2)

    return info
