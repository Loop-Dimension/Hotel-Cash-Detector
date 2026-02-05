"""
Detector modules for ML Service.
Copied from main project's detectors/ folder.
"""
from .base_detector import BaseDetector, Detection
from .unified_detector import UnifiedDetector
from .cash_detector import CashTransactionDetector
from .violence_detector import ViolenceDetector
from .fire_detector import FireDetector
from .gemini_validator import GeminiValidator


def get_device(use_gpu_setting: str = 'auto') -> str:
    """
    Determine the device (cuda/cpu) based on configuration and availability.

    Args:
        use_gpu_setting: 'True', 'False', or 'auto'
    Returns:
        'cuda' or 'cpu'
    """
    import torch

    if isinstance(use_gpu_setting, bool):
        use_gpu = use_gpu_setting
    elif isinstance(use_gpu_setting, str):
        setting_lower = use_gpu_setting.lower().strip()
        if setting_lower == 'auto':
            use_gpu = None
        elif setting_lower in ('true', '1', 'yes', 'gpu', 'cuda'):
            use_gpu = True
        else:
            use_gpu = False
    else:
        use_gpu = None

    cuda_available = torch.cuda.is_available()

    if use_gpu is True:
        if cuda_available:
            return 'cuda'
        else:
            print("WARNING: GPU requested but CUDA is not available. Falling back to CPU.")
            return 'cpu'
    elif use_gpu is False:
        return 'cpu'
    else:
        return 'cuda' if cuda_available else 'cpu'


def get_device_info() -> dict:
    """Get detailed information about the current device configuration."""
    import torch

    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': None,
        'gpu_name': None,
        'gpu_count': 0,
        'device': 'cpu'
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['device'] = 'cuda'

    return info


__all__ = [
    'BaseDetector',
    'Detection',
    'UnifiedDetector',
    'CashTransactionDetector',
    'ViolenceDetector',
    'FireDetector',
    'GeminiValidator',
    'get_device',
    'get_device_info',
]
