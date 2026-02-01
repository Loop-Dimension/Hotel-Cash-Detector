"""
ML Service Client for Django Backend

This module provides a client interface to the ML microservice,
replacing direct detector calls with HTTP requests.

Usage:
    # Get a detector proxy (works like UnifiedDetector)
    detector = get_detector_for_camera(camera)
    result = detector.process_frame(frame, draw_overlay=True)

    # Or use the client directly
    client = MLServiceClient()
    response = client.process_frame(camera_id, frame)
"""
import os
import base64
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import cv2
import numpy as np
from django.conf import settings


@dataclass
class MLServiceConfig:
    """Configuration for ML service connection"""
    base_url: str
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0


class MLServiceClient:
    """
    HTTP Client for communicating with the ML Detection Service.

    Handles:
    - Frame encoding/decoding
    - Retry logic with exponential backoff
    - Error handling and fallback
    """

    def __init__(self, config: MLServiceConfig = None):
        self.config = config or MLServiceConfig(
            base_url=getattr(settings, 'ML_SERVICE_URL', os.getenv('ML_SERVICE_URL', 'http://ml-service:8001')),
            timeout=getattr(settings, 'ML_SERVICE_TIMEOUT', int(os.getenv('ML_SERVICE_TIMEOUT', '30'))),
        )
        self._session = requests.Session()

    def _encode_frame(self, frame: np.ndarray, quality: int = 85) -> str:
        """Encode numpy array to base64 JPEG"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')

    def _decode_frame(self, image_base64: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make HTTP request with retry logic"""
        url = f"{self.config.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.config.timeout)

        last_error = None
        for attempt in range(self.config.retry_count):
            try:
                response = self._session.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise ConnectionError(f"ML Service unavailable after {self.config.retry_count} attempts: {last_error}")

    # ==================== Health & Status ====================

    def health_check(self) -> bool:
        """Check if ML service is healthy"""
        try:
            result = self._request('GET', '/health')
            return result.get('status') == 'healthy'
        except Exception:
            return False

    def get_status(self) -> dict:
        """Get ML service status"""
        return self._request('GET', '/status')

    # ==================== Detector Management ====================

    def initialize_detector(self, camera_id: int, config: dict) -> dict:
        """Initialize or update detector for a camera"""
        return self._request('POST', f'/detectors/{camera_id}/initialize', json=config)

    def remove_detector(self, camera_id: int) -> dict:
        """Remove detector for camera"""
        return self._request('DELETE', f'/detectors/{camera_id}')

    def list_detectors(self) -> dict:
        """List all active detectors"""
        return self._request('GET', '/detectors')

    # ==================== Frame Processing ====================

    def process_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        draw_overlay: bool = True,
        return_frame: bool = True,
        frame_number: int = None,
        config: dict = None
    ) -> dict:
        """
        Process a single frame through ML service.

        Returns dict compatible with current UnifiedDetector.process_frame():
        {
            'frame': numpy array with overlays (if return_frame),
            'detections': [{'label', 'confidence', 'bbox', ...}],
            'alerts': [...],
            'frame_number': int
        }
        """
        payload = {
            'camera_id': camera_id,
            'frame': {
                'image_base64': self._encode_frame(frame),
                'frame_number': frame_number,
            },
            'draw_overlay': draw_overlay,
            'return_frame': return_frame,
        }

        if config:
            payload['config'] = config

        result = self._request('POST', '/detect', json=payload)

        # Convert response to current format
        response = {
            'detections': self._convert_detections(result.get('detections', [])),
            'alerts': self._convert_detections(result.get('alerts', [])),
            'frame_number': result.get('frame_number', 0),
        }

        # Decode frame if returned
        if return_frame and result.get('processed_frame_base64'):
            response['frame'] = self._decode_frame(result['processed_frame_base64'])
        else:
            response['frame'] = frame

        return response

    def _convert_detections(self, detections: List[dict]) -> List[dict]:
        """Convert API response detections to current format"""
        result = []
        for det in detections:
            converted = {
                'label': det['label'],
                'confidence': det['confidence'],
                'timestamp': det.get('timestamp'),
                'frame_number': det.get('frame_number', 0),
                'metadata': det.get('metadata', {}),
            }
            if det.get('bbox'):
                bbox = det['bbox']
                converted['bbox'] = (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
            else:
                converted['bbox'] = None
            result.append(converted)
        return result

    # ==================== Gemini Validation ====================

    def validate_event(
        self,
        camera_id: int,
        frame: np.ndarray,
        event_type: str,
        custom_prompt: str = None
    ) -> Tuple[bool, float, str, str]:
        """
        Validate detection with Gemini AI.

        Returns: (is_valid, confidence, reason, corrected_event_type)
        """
        payload = {
            'camera_id': camera_id,
            'event_type': event_type,
            'frame': {
                'image_base64': self._encode_frame(frame),
            },
        }
        if custom_prompt:
            payload['custom_prompt'] = custom_prompt

        result = self._request('POST', '/validate', json=payload)

        return (
            result['is_valid'],
            result['confidence'],
            result['reason'],
            result['corrected_event_type']
        )

    def validate_event_video(
        self,
        camera_id: int,
        video_path: str,
        event_type: str,
        custom_prompt: str = None
    ) -> Tuple[bool, float, str, str]:
        """
        Validate detection with video clip.

        Returns: (is_valid, confidence, reason, corrected_event_type)
        """
        with open(video_path, 'rb') as f:
            video_data = f.read()

        payload = {
            'camera_id': camera_id,
            'event_type': event_type,
            'video_base64': base64.b64encode(video_data).decode('utf-8'),
        }
        if custom_prompt:
            payload['custom_prompt'] = custom_prompt

        result = self._request('POST', '/validate/video', json=payload)

        return (
            result['is_valid'],
            result['confidence'],
            result['reason'],
            result['corrected_event_type']
        )

    # ==================== Configuration ====================

    def update_zones(self, camera_id: int, zones: dict) -> dict:
        """Update zone polygons for detector"""
        return self._request('POST', f'/detectors/{camera_id}/zones', json=zones)


# ==================== Backward-Compatible Wrapper ====================

class MLDetectorProxy:
    """
    Proxy class that mimics UnifiedDetector interface but uses ML service.

    This allows minimal changes to existing code - just replace:
        detector = UnifiedDetector(config)
    with:
        detector = MLDetectorProxy(config, camera_id)
    """

    def __init__(self, config: Dict = None, camera_id: int = None):
        self.config = config or {}
        self.camera_id = camera_id
        self.client = MLServiceClient()
        self.is_initialized = False
        self.frame_count = 0

        # Detection toggles (for compatibility)
        self.detect_cash = self.config.get('detect_cash', True)
        self.detect_violence = self.config.get('detect_violence', True)
        self.detect_fire = self.config.get('detect_fire', True)

        # Initialize on ML service
        if camera_id:
            self._initialize_remote()

    def _initialize_remote(self):
        """Initialize detector on ML service"""
        detector_config = {
            'camera_id': self.camera_id,
            'detect_cash': self.config.get('detect_cash', True),
            'detect_violence': self.config.get('detect_violence', True),
            'detect_fire': self.config.get('detect_fire', True),
            'cash_confidence': self.config.get('cash_confidence', 0.5),
            'violence_confidence': self.config.get('violence_confidence', 0.6),
            'fire_confidence': self.config.get('fire_confidence', 0.5),
            'pose_confidence': self.config.get('pose_confidence', 0.5),
            'cashier_zone_polygon': self.config.get('cashier_zone_polygon'),
            'cash_drawer_zone_polygon': self.config.get('cash_drawer_zone_polygon'),
            'cashier_zone_enabled': self.config.get('cashier_zone_enabled', True),
            'cash_drawer_zone_enabled': self.config.get('cash_drawer_zone_enabled', True),
            'hand_touch_distance': self.config.get('hand_touch_distance', 100),
            'min_transaction_frames': self.config.get('min_transaction_frames', 1),
            'min_violence_frames': self.config.get('min_violence_frames', 10),
            'min_fire_frames': self.config.get('min_fire_frames', 3),
            'use_gpu': self.config.get('use_gpu', 'auto'),
        }

        try:
            self.client.initialize_detector(self.camera_id, detector_config)
            self.is_initialized = True
            print(f"[MLDetectorProxy] Initialized detector for camera {self.camera_id}")
        except Exception as e:
            print(f"[MLDetectorProxy] Failed to initialize: {e}")
            self.is_initialized = False

    def initialize(self) -> bool:
        """Initialize detector (compatibility method)"""
        if not self.is_initialized:
            self._initialize_remote()
        return self.is_initialized

    def process_frame(self, frame: np.ndarray, draw_overlay: bool = True) -> Dict:
        """
        Process frame through ML service.
        Returns same format as UnifiedDetector.process_frame()
        """
        self.frame_count += 1

        try:
            return self.client.process_frame(
                camera_id=self.camera_id,
                frame=frame,
                draw_overlay=draw_overlay,
                return_frame=draw_overlay,
                frame_number=self.frame_count
            )
        except Exception as e:
            print(f"[MLDetectorProxy] process_frame error: {e}")
            return {
                'frame': frame,
                'detections': [],
                'alerts': [],
                'frame_number': self.frame_count
            }

    def set_cashier_zone_polygon(self, polygon: List):
        """Update cashier zone"""
        try:
            self.client.update_zones(self.camera_id, {
                'cashier_zone_polygon': polygon
            })
        except Exception as e:
            print(f"[MLDetectorProxy] set_cashier_zone_polygon error: {e}")

    def set_cash_drawer_zone_polygon(self, polygon: List):
        """Update cash drawer zone"""
        try:
            self.client.update_zones(self.camera_id, {
                'cash_drawer_zone_polygon': polygon
            })
        except Exception as e:
            print(f"[MLDetectorProxy] set_cash_drawer_zone_polygon error: {e}")

    def reset(self):
        """Reset detector state"""
        self.frame_count = 0


# ==================== Factory Function ====================

# Cache for detectors
_detector_cache: Dict[int, Any] = {}
_detector_cache_keys: Dict[int, str] = {}


def get_detector_for_camera(camera, use_ml_service: bool = None) -> Any:
    """
    Factory function to get detector for camera.

    Supports both ML service mode and local detector mode with automatic fallback.

    Args:
        camera: Camera model instance
        use_ml_service: Force ML service usage (None = auto-detect from settings)

    Returns:
        Either MLDetectorProxy (microservice) or UnifiedDetector (local)
    """
    global _detector_cache, _detector_cache_keys

    # Determine whether to use ML service
    if use_ml_service is None:
        use_ml_service = getattr(settings, 'USE_ML_SERVICE', False)
        if isinstance(use_ml_service, str):
            use_ml_service = use_ml_service.lower() in ('true', '1', 'yes')

    # Build config from camera settings
    config = {
        'models_dir': str(getattr(settings, 'BASE_DIR', '.') / 'models'),
        'use_gpu': getattr(settings, 'DETECTION_CONFIG', {}).get('USE_GPU', 'auto'),
        'detect_cash': camera.detect_cash,
        'detect_violence': camera.detect_violence,
        'detect_fire': camera.detect_fire,
        'cash_confidence': camera.cash_confidence,
        'violence_confidence': camera.violence_confidence,
        'fire_confidence': camera.fire_confidence,
        'pose_confidence': getattr(settings, 'DETECTION_CONFIG', {}).get('POSE_CONFIDENCE', 0.5),
        'cashier_zone_polygon': camera.get_cashier_zone_polygon_points(),
        'cashier_zone_enabled': camera.cashier_zone_enabled,
        'cash_drawer_zone_polygon': camera.get_cash_drawer_zone_polygon_points(),
        'cash_drawer_zone_enabled': camera.cash_drawer_zone_enabled,
        'hand_touch_distance': getattr(camera, 'hand_touch_distance', 100),
        'min_transaction_frames': getattr(settings, 'DETECTION_CONFIG', {}).get('MIN_TRANSACTION_FRAMES', 1),
        'min_violence_frames': getattr(settings, 'DETECTION_CONFIG', {}).get('MIN_VIOLENCE_FRAMES', 10),
        'min_fire_frames': getattr(settings, 'DETECTION_CONFIG', {}).get('MIN_FIRE_FRAMES', 3),
    }

    # Cache key for invalidation
    cache_key = f"{camera.id}_{camera.updated_at.timestamp() if hasattr(camera, 'updated_at') else 0}_{use_ml_service}"

    # Check cache
    if camera.id in _detector_cache and _detector_cache_keys.get(camera.id) == cache_key:
        return _detector_cache[camera.id]

    # Try ML service first if enabled
    if use_ml_service:
        try:
            client = MLServiceClient()
            if client.health_check():
                detector = MLDetectorProxy(config=config, camera_id=camera.id)
                _detector_cache[camera.id] = detector
                _detector_cache_keys[camera.id] = cache_key
                return detector
            else:
                print(f"[get_detector_for_camera] ML Service unhealthy, falling back to local detector")
        except Exception as e:
            print(f"[get_detector_for_camera] ML Service error: {e}, falling back to local detector")

    # Fallback to local detector
    try:
        from detectors import UnifiedDetector

        detector = UnifiedDetector(config)
        detector.initialize()
        _detector_cache[camera.id] = detector
        _detector_cache_keys[camera.id] = cache_key
        return detector
    except ImportError as e:
        print(f"[get_detector_for_camera] Local detectors not available: {e}")
        raise


def clear_detector_cache(camera_id: int = None):
    """Clear detector cache for a camera or all cameras"""
    global _detector_cache, _detector_cache_keys

    if camera_id:
        _detector_cache.pop(camera_id, None)
        _detector_cache_keys.pop(camera_id, None)
    else:
        _detector_cache.clear()
        _detector_cache_keys.clear()
