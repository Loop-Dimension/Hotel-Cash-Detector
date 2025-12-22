"""
Detection modules for Hotel Cash Detector
"""
from .base_detector import BaseDetector
from .cash_detector import CashTransactionDetector
from .violence_detector import ViolenceDetector
from .fire_detector import FireDetector
from .unified_detector import UnifiedDetector


def get_device(use_gpu_setting: str = 'auto') -> str:
    """
    Determine the device (cuda/cpu) based on configuration and availability.
    
    Args:
        use_gpu_setting: 'True', 'False', or 'auto'
            - 'True' or True: Force GPU usage (requires CUDA)
            - 'False' or False: Force CPU usage
            - 'auto': Automatically detect GPU availability
    
    Returns:
        'cuda' or 'cpu'
    """
    import torch
    
    # Normalize the setting
    if isinstance(use_gpu_setting, bool):
        use_gpu = use_gpu_setting
    elif isinstance(use_gpu_setting, str):
        setting_lower = use_gpu_setting.lower().strip()
        if setting_lower == 'auto':
            use_gpu = None  # Will auto-detect
        elif setting_lower in ('true', '1', 'yes', 'gpu', 'cuda'):
            use_gpu = True
        else:
            use_gpu = False
    else:
        use_gpu = None  # Default to auto
    
    # Determine device
    cuda_available = torch.cuda.is_available()
    
    if use_gpu is True:
        if cuda_available:
            return 'cuda'
        else:
            print("⚠️ WARNING: GPU requested but CUDA is not available. Falling back to CPU.")
            return 'cpu'
    elif use_gpu is False:
        return 'cpu'
    else:
        # Auto-detect
        return 'cuda' if cuda_available else 'cpu'


def get_device_info() -> dict:
    """
    Get detailed information about the current device configuration.
    
    Returns:
        Dictionary with device information
    """
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
    'CashTransactionDetector',
    'ViolenceDetector',
    'FireDetector',
    'UnifiedDetector',
    'get_device',
    'get_device_info'
]
