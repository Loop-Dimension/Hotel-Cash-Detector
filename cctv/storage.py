"""
Storage utilities for Hotel CCTV Monitoring System

This module provides a centralized way to get the correct storage backend
(S3 or local filesystem) based on the USE_S3 setting.
"""
import os
from io import BytesIO
from pathlib import Path
from django.conf import settings
from django.core.files.base import ContentFile


# Cache storage instance
_storage_instance = None


def get_storage():
    """
    Get the appropriate storage backend based on settings.
    Returns S3Boto3Storage if USE_S3 is True, otherwise FileSystemStorage.
    
    Returns:
        Storage instance (S3Boto3Storage or FileSystemStorage)
    """
    global _storage_instance
    
    if _storage_instance is not None:
        return _storage_instance
    
    use_s3 = getattr(settings, 'USE_S3', False)
    
    if use_s3:
        try:
            from storages.backends.s3boto3 import S3Boto3Storage
            _storage_instance = S3Boto3Storage()
            print("[Storage] Using S3Boto3Storage")
        except ImportError:
            print("[Storage] WARNING: django-storages not installed, falling back to FileSystemStorage")
            from django.core.files.storage import FileSystemStorage
            _storage_instance = FileSystemStorage(location=settings.MEDIA_ROOT)
    else:
        from django.core.files.storage import FileSystemStorage
        _storage_instance = FileSystemStorage(location=settings.MEDIA_ROOT)
        print("[Storage] Using FileSystemStorage")
    
    return _storage_instance


def upload_file_to_storage(local_path: str, storage_path: str) -> str:
    """
    Upload a local file to the storage backend (S3 or local).
    
    Args:
        local_path: Absolute path to the local file
        storage_path: Relative path where the file should be stored (e.g., 'clips/video.mp4')
        
    Returns:
        URL to access the uploaded file
    """
    storage = get_storage()
    local_path = Path(local_path)
    
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    
    # Read the file content
    with open(local_path, 'rb') as f:
        content = f.read()
    
    # Delete if exists (for overwriting)
    if storage.exists(storage_path):
        storage.delete(storage_path)
    
    # Save to storage
    saved_path = storage.save(storage_path, ContentFile(content))
    
    # Get URL
    url = storage.url(saved_path)
    
    print(f"[Storage] Uploaded {local_path.name} -> {url}")
    
    return url


def upload_bytes_to_storage(data: bytes, storage_path: str, content_type: str = None) -> str:
    """
    Upload bytes data to the storage backend.
    
    Args:
        data: Bytes to upload
        storage_path: Relative path where the file should be stored
        content_type: Optional content type (e.g., 'image/jpeg', 'video/mp4')
        
    Returns:
        URL to access the uploaded file
    """
    storage = get_storage()
    
    # Delete if exists
    if storage.exists(storage_path):
        storage.delete(storage_path)
    
    # Save to storage
    content_file = ContentFile(data)
    saved_path = storage.save(storage_path, content_file)
    
    # Get URL
    url = storage.url(saved_path)
    
    print(f"[Storage] Uploaded bytes ({len(data)} bytes) -> {url}")
    
    return url


def save_image_to_storage(frame, storage_path: str, jpeg_quality: int = 90) -> str:
    """
    Encode an OpenCV frame as JPEG and upload to storage.
    
    Args:
        frame: OpenCV frame (numpy array)
        storage_path: Relative path where the image should be stored
        jpeg_quality: JPEG quality (0-100)
        
    Returns:
        URL to access the uploaded image
    """
    import cv2
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    image_bytes = buffer.tobytes()
    
    return upload_bytes_to_storage(image_bytes, storage_path)


def get_media_url(storage_path: str) -> str:
    """
    Get the URL for a file in storage.
    
    Args:
        storage_path: Relative path of the file in storage
        
    Returns:
        Full URL to access the file
    """
    storage = get_storage()
    use_s3 = getattr(settings, 'USE_S3', False)
    
    if use_s3:
        return storage.url(storage_path)
    else:
        return f'/media/{storage_path}'


def ensure_local_temp_dir() -> Path:
    """
    Ensure a local temp directory exists for ffmpeg processing.
    
    Returns:
        Path to temp directory
    """
    temp_dir = Path(settings.MEDIA_ROOT) / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def is_s3_enabled() -> bool:
    """Check if S3 storage is enabled"""
    return getattr(settings, 'USE_S3', False)
