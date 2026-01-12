"""
Shared utilities for Hotel CCTV Monitoring System

This module contains common functions used across the application to avoid code duplication.
All clip saving, event saving, RTSP capture, and validation utilities are centralized here.
"""
import os
import cv2
import json
import subprocess
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from django.conf import settings

# Import storage utilities
from cctv.storage import (
    get_storage, 
    upload_file_to_storage, 
    upload_bytes_to_storage,
    save_image_to_storage,
    get_media_url,
    is_s3_enabled
)


# ============================================================================
# GLOBAL LOCKS AND STATE
# ============================================================================

# FFmpeg lock to prevent concurrent encoding issues
_ffmpeg_lock = threading.Lock()


# ============================================================================
# RTSP CAPTURE UTILITIES
# ============================================================================

def create_rtsp_capture(rtsp_url: str) -> cv2.VideoCapture:
    """
    Create an optimized RTSP capture with proper settings.
    
    Args:
        rtsp_url: RTSP stream URL
        
    Returns:
        cv2.VideoCapture object configured for RTSP streaming
    """
    # Set FFMPEG options for RTSP - TCP transport, timeouts, low latency
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
        'rtsp_transport;tcp|'
        'stimeout;60000000|'
        'max_delay;1000000|'
        'fflags;nobuffer+discardcorrupt'
    )
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
    
    return cap


def test_rtsp_connection(rtsp_url: str, timeout: int = 10) -> Tuple[bool, Optional[Any]]:
    """
    Test if RTSP connection works and return a test frame.
    
    Args:
        rtsp_url: RTSP stream URL
        timeout: Connection timeout in seconds
        
    Returns:
        Tuple of (success, frame or None)
    """
    cap = create_rtsp_capture(rtsp_url)
    
    if not cap.isOpened():
        return False, None
    
    ret, frame = cap.read()
    cap.release()
    
    return ret and frame is not None, frame if ret else None


# ============================================================================
# VIDEO/CLIP SAVING UTILITIES
# ============================================================================

def save_clip(
    frames: List[Any],
    camera,
    event_type: str,
    fps: float = 15,
    add_overlay: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """
    Save video clip and thumbnail from frames.
    Uploads to S3 if USE_S3 is True, otherwise saves locally.
    
    Args:
        frames: List of OpenCV frames
        camera: Camera model instance
        event_type: Type of event (cash, violence, fire)
        fps: Frames per second for output video
        add_overlay: Whether to add detection label overlay
        
    Returns:
        Tuple of (clip_url, thumbnail_url) or (None, None) on failure
    """
    if not frames or len(frames) == 0:
        print(f"[Clip] No frames to save")
        return None, None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:6]
    
    # Always use local temp directory for ffmpeg processing
    temp_dir = Path(settings.MEDIA_ROOT) / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create local dirs (needed even for S3 since ffmpeg needs local files)
    clip_dir = Path(settings.MEDIA_ROOT) / 'clips'
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    thumb_dir = Path(settings.MEDIA_ROOT) / 'thumbnails'
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    temp_filename = f"{camera.camera_id}_{event_type}_{timestamp}_{unique_id}_temp.avi"
    final_filename = f"{camera.camera_id}_{event_type}_{timestamp}.mp4"
    thumb_filename = f"{camera.camera_id}_{event_type}_{timestamp}.jpg"
    
    temp_path = temp_dir / temp_filename
    final_path = temp_dir / final_filename  # Use temp_dir for S3 workflow
    
    height, width = frames[0].shape[:2]
    
    # Use MJPG codec for temp file (reliable, fast)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"[Clip] Failed to open video writer")
        return None, None
    
    # Color map for event types
    colors = {
        'cash': (0, 255, 0),        # Green
        'violence': (0, 0, 255),     # Red
        'fire': (0, 165, 255),       # Orange
        'potential_cash': (0, 255, 255)  # Yellow
    }
    color = colors.get(event_type, (255, 255, 255))
    
    # Write frames
    frame_count = 0
    for frame in frames:
        if frame is None:
            continue
        
        if add_overlay:
            # Add detection type label
            label = f"{event_type.upper()} DETECTED"
            cv2.rectangle(frame, (10, 10), (250, 45), (0, 0, 0), -1)
            cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        out.write(frame)
        frame_count += 1
    
    out.release()
    
    print(f"[Clip] Wrote {frame_count} frames to temp file: {temp_path}")
    print(f"[Clip] Temp file exists: {temp_path.exists()}, size: {temp_path.stat().st_size if temp_path.exists() else 0} bytes")
    
    # Convert to H.264 MP4 using ffmpeg
    ffmpeg_path = settings.DETECTION_CONFIG.get('FFMPEG_PATH', 'ffmpeg')
    
    with _ffmpeg_lock:
        try:
            print(f"[Clip] Running ffmpeg: {ffmpeg_path} -i {temp_path} -> {final_path}")
            result = subprocess.run([
                ffmpeg_path, '-y', '-i', str(temp_path),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-r', str(int(fps)),
                str(final_path)
            ], capture_output=True, timeout=180)
            
            print(f"[Clip] FFmpeg return code: {result.returncode}")
            
            # Clean up temp avi file
            safe_delete_file(temp_path)
            
            if result.returncode == 0 and final_path.exists():
                print(f"[Clip] Created: {final_path} ({final_path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"[Clip] FFmpeg failed! Return code: {result.returncode}")
                print(f"[Clip] FFmpeg stderr: {result.stderr.decode()[:500]}")
                return None, None
                
        except subprocess.TimeoutExpired:
            print(f"[Clip] FFmpeg timeout after 180s")
            safe_delete_file(temp_path)
            return None, None
        except FileNotFoundError:
            print(f"[Clip] FFmpeg not found at: {ffmpeg_path}")
            safe_delete_file(temp_path)
            return None, None
        except Exception as e:
            print(f"[Clip] FFmpeg exception: {e}")
            import traceback
            traceback.print_exc()
            safe_delete_file(temp_path)
            return None, None
    
    # Prepare thumbnail
    thumb_frame = frames[-1].copy()
    if add_overlay:
        label = f"{event_type.upper()}"
        cv2.rectangle(thumb_frame, (10, 10), (150, 45), (0, 0, 0), -1)
        cv2.putText(thumb_frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Upload to S3 or save locally
    if is_s3_enabled():
        try:
            # Upload video to S3
            clip_storage_path = f"clips/{final_filename}"
            clip_url = upload_file_to_storage(str(final_path), clip_storage_path)
            
            # Upload thumbnail to S3
            thumb_storage_path = f"thumbnails/{thumb_filename}"
            thumb_url = save_image_to_storage(thumb_frame, thumb_storage_path, jpeg_quality=90)
            
            # Clean up local temp file
            safe_delete_file(final_path)
            
            print(f"[Clip] Uploaded to S3: {clip_url}")
            print(f"[Clip] Thumbnail S3: {thumb_url}")
            
            # Return relative paths for database storage
            return f'/media/{clip_storage_path}', f'/media/{thumb_storage_path}'
            
        except Exception as e:
            print(f"[Clip] S3 upload failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to local storage
            safe_delete_file(final_path)
            return None, None
    else:
        # Save locally (move from temp to clips dir)
        local_clip_path = clip_dir / final_filename
        thumb_path = thumb_dir / thumb_filename
        
        # Move the video file
        if final_path != local_clip_path:
            import shutil
            shutil.move(str(final_path), str(local_clip_path))
        
        # Save thumbnail locally
        cv2.imwrite(str(thumb_path), thumb_frame)
        
        print(f"[Clip] Saved locally: {local_clip_path}")
        print(f"[Clip] Thumbnail: {thumb_path}")
        
        return f'/media/clips/{final_filename}', f'/media/thumbnails/{thumb_filename}'


def save_validation_clip(
    frames: List[Any],
    camera,
    event_type: str,
    duration_name: str = "3s"
) -> Optional[str]:
    """
    Save a short video clip for Gemini validation.
    Uploads to S3 if USE_S3 is True, otherwise saves locally.
    
    Args:
        frames: List of OpenCV frames (typically ~45 frames for 3 seconds)
        camera: Camera model instance
        event_type: Type of event
        duration_name: Name to include in filename (e.g., "3s")
        
    Returns:
        Path/URL to saved video file or None on failure
    """
    if not frames or len(frames) == 0:
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:6]
    
    # Use temp directory for processing
    temp_dir = Path(settings.MEDIA_ROOT) / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create validation_clips directory for local storage
    validation_dir = Path(settings.MEDIA_ROOT) / 'validation_clips'
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{camera.camera_id}_{event_type}_{timestamp}_{unique_id}.mp4"
    temp_avi_filename = f"temp_{unique_id}.avi"
    temp_avi_path = temp_dir / temp_avi_filename
    final_path = temp_dir / filename  # Use temp_dir for S3 workflow
    
    # Create video
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_avi_path), fourcc, 15, (w, h))
    
    for frame in frames:
        if frame is not None:
            out.write(frame)
    
    out.release()
    
    # Convert to H.264
    ffmpeg_path = settings.DETECTION_CONFIG.get('FFMPEG_PATH', 'ffmpeg')
    
    with _ffmpeg_lock:
        try:
            subprocess.run([
                ffmpeg_path, '-y', '-i', str(temp_avi_path),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
                '-pix_fmt', 'yuv420p', '-r', '15',
                str(final_path)
            ], capture_output=True, timeout=30, check=True)
            
            safe_delete_file(temp_avi_path)
            print(f"[Validation] Created {duration_name} clip for Gemini: {filename}")
            
        except Exception as e:
            print(f"[Validation] FFmpeg error creating validation clip: {e}")
            safe_delete_file(temp_avi_path)
            return None
    
    # Upload to S3 or return local path
    if is_s3_enabled():
        try:
            storage_path = f"validation_clips/{filename}"
            url = upload_file_to_storage(str(final_path), storage_path)
            
            # Clean up local temp file
            safe_delete_file(final_path)
            
            print(f"[Validation] Uploaded to S3: {url}")
            return url  # Return full S3 URL for Gemini to access
            
        except Exception as e:
            print(f"[Validation] S3 upload failed: {e}")
            import traceback
            traceback.print_exc()
            safe_delete_file(final_path)
            return None
    else:
        # Move to validation_clips directory
        local_path = validation_dir / filename
        if final_path != local_path:
            import shutil
            shutil.move(str(final_path), str(local_path))
        return str(local_path)


# ============================================================================
# EVENT SAVING UTILITIES
# ============================================================================

def save_event(
    camera,
    event_type: str,
    confidence: float,
    frame_number: int,
    bbox: Optional[List[int]] = None,
    clip_path: Optional[str] = None,
    thumbnail_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    gemini_validated: bool = True,
    gemini_confidence: float = 1.0,
    gemini_reason: str = ""
):
    """
    Save detection event to database with metadata as JSON file.
    Uploads JSON to S3 if USE_S3 is True, otherwise saves locally.
    
    Args:
        camera: Camera model instance
        event_type: Type of event (cash, violence, fire)
        confidence: Detection confidence score
        frame_number: Frame number when detection occurred
        bbox: Bounding box [x1, y1, x2, y2]
        clip_path: Path to video clip
        thumbnail_path: Path to thumbnail
        metadata: Additional metadata dictionary
        gemini_validated: Whether Gemini validated this event
        gemini_confidence: Gemini's confidence score
        gemini_reason: Gemini's validation reason
        
    Returns:
        Created Event instance or None on failure
    """
    try:
        from cctv.models import Event
        
        # Build metadata
        event_metadata = metadata or {}
        timestamp = datetime.now()
        
        event_metadata.update({
            'timestamp': timestamp.isoformat(),
            'frame_number': frame_number,
            'confidence': round(confidence, 3),
            'bbox': bbox,
            'camera_id': camera.camera_id,
            'camera_name': camera.name,
            'event_type': event_type,
            'clip_path': clip_path,
            'thumbnail_path': thumbnail_path,
            'gemini_validation': {
                'validated': gemini_validated,
                'confidence': round(gemini_confidence, 3),
                'reason': gemini_reason
            }
        })
        
        # Prepare JSON data
        json_filename = f"{event_type}_{camera.camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        storage_path = f"json/{json_filename}"
        
        # Convert numpy types to JSON-serializable
        event_metadata = convert_to_json_serializable(event_metadata)
        json_content = json.dumps(event_metadata, indent=2, ensure_ascii=False)
        
        # Save to S3 or locally
        if is_s3_enabled():
            try:
                url = upload_bytes_to_storage(
                    json_content.encode('utf-8'), 
                    storage_path,
                    content_type='application/json'
                )
                print(f"[JSON] Uploaded to S3: {storage_path}")
            except Exception as e:
                print(f"[JSON] S3 upload failed, saving locally: {e}")
                # Fallback to local
                json_dir = Path(settings.MEDIA_ROOT) / 'json'
                json_dir.mkdir(parents=True, exist_ok=True)
                json_path = json_dir / json_filename
                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(json_content)
        else:
            # Save locally
            json_dir = Path(settings.MEDIA_ROOT) / 'json'
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            print(f"[JSON] Saved locally: {storage_path}")
        
        json_relative_path = storage_path
        
        # Create database event
        event = Event.objects.create(
            branch=camera.branch,
            camera=camera,
            event_type=event_type,
            confidence=confidence,
            frame_number=frame_number,
            bbox_x1=bbox[0] if bbox else 0,
            bbox_y1=bbox[1] if bbox else 0,
            bbox_x2=bbox[2] if bbox else 0,
            bbox_y2=bbox[3] if bbox else 0,
            clip_path=clip_path,
            thumbnail_path=thumbnail_path,
            metadata=json_relative_path,
        )
        print(f"[DB] Saved event: {event_type} (id={event.id}) with JSON: {json_relative_path}")
        return event
        
    except Exception as e:
        print(f"[DB] Error saving event: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# GEMINI VALIDATION UTILITIES
# ============================================================================

def create_gemini_validator(camera_id: int):
    """
    Create a GeminiValidator instance with proper configuration.
    
    Args:
        camera_id: Camera ID for logging
        
    Returns:
        Configured GeminiValidator instance or None if not available
    """
    try:
        from detectors.gemini_validator import GeminiValidator
        from cctv.models import GeminiPrompts
        
        gemini_api_key = getattr(settings, 'GEMINI_API_KEY', '')
        if not gemini_api_key:
            return None
        
        # Create validator
        validator = GeminiValidator(api_key=gemini_api_key, camera_id=camera_id)
        
        # Load global prompts
        global_prompts = GeminiPrompts.get_prompts()
        if global_prompts.get('unified'):
            validator.set_custom_prompts({'unified': global_prompts['unified']})
        
        return validator
        
    except Exception as e:
        print(f"[Gemini] Failed to create validator: {e}")
        return None


def validate_detection(
    camera,
    event_type: str,
    frame=None,
    validation_frames: List = None,
    use_video: bool = False
) -> Tuple[bool, float, str, str]:
    """
    Validate a detection using Gemini AI.
    
    Args:
        camera: Camera model instance
        event_type: Type of event to validate
        frame: Single frame for image validation
        validation_frames: List of frames for video validation
        use_video: Whether to use video validation
        
    Returns:
        Tuple of (is_valid, confidence, reason, corrected_event_type)
    """
    if not settings.DETECTION_CONFIG.get('GEMINI_VALIDATION_ENABLED', True):
        return True, 1.0, "Validation disabled", event_type
    
    validator = create_gemini_validator(camera.id)
    if not validator:
        return True, 1.0, "Validator not available", event_type
    
    try:
        if use_video and validation_frames:
            # Create validation video
            video_path = save_validation_clip(validation_frames, camera, event_type)
            if video_path:
                result = validator.validate_event_video(video_path, event_type)
                print(f"[Detection] Gemini VIDEO validation: {event_type} = {result[0]}")
                return result
            # Fallback to image
            use_video = False
        
        if frame is not None:
            result = validator.validate_event(frame, event_type)
            print(f"[Detection] Gemini IMAGE validation: {event_type} = {result[0]}")
            return result
        
        return True, 1.0, "No frame provided", event_type
        
    except Exception as e:
        print(f"[Detection] Gemini validation error: {e}")
        return True, 1.0, f"Validation error: {e}", event_type


# ============================================================================
# FILE UTILITIES
# ============================================================================

def safe_delete_file(file_path: Path, retries: int = 3, delay: float = 0.5) -> bool:
    """
    Safely delete a file with retries for Windows file locking issues.
    
    Args:
        file_path: Path to file to delete
        retries: Number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        True if deleted, False otherwise
    """
    import time
    
    for attempt in range(retries):
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"[File] Note: Temp file will be cleaned up later: {file_path.name}")
                return False
        except Exception:
            return False
    return False


def cleanup_temp_files(directory: str = 'clips', pattern: str = '*_temp.avi'):
    """
    Clean up leftover temp files from previous runs.
    
    Args:
        directory: Subdirectory under MEDIA_ROOT
        pattern: Glob pattern for temp files
    """
    try:
        target_dir = Path(settings.MEDIA_ROOT) / directory
        if target_dir.exists():
            for temp_file in target_dir.glob(pattern):
                try:
                    temp_file.unlink()
                    print(f"[Cleanup] Removed old temp file: {temp_file.name}")
                except Exception:
                    pass
    except Exception:
        pass


def convert_to_json_serializable(obj):
    """
    Convert numpy types and other non-serializable objects to JSON-serializable types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


# ============================================================================
# CONSTANTS
# ============================================================================

# Default event cooldown in seconds (between same event type)
DEFAULT_EVENT_COOLDOWN = 60

# Default clip duration in seconds
DEFAULT_CLIP_DURATION = 30

# Default Gemini validation clip duration in seconds
DEFAULT_VALIDATION_CLIP_DURATION = 3

# Buffer FPS (typically half of stream FPS since we buffer every 2nd frame)
DEFAULT_BUFFER_FPS = 15

# Event type colors for overlays (BGR format)
EVENT_COLORS = {
    'cash': (0, 255, 0),         # Green
    'potential_cash': (0, 255, 255),  # Yellow
    'violence': (0, 0, 255),     # Red
    'fire': (0, 165, 255),       # Orange
}
