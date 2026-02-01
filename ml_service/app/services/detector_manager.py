"""
Detector Manager Service

Manages UnifiedDetector instances for multiple cameras.
Handles lifecycle, caching, and configuration updates.
"""
import threading
from typing import Dict, Optional, List
from datetime import datetime
import sys
import os

# Add detectors to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.config import settings, get_gpu_info
from app.models.schemas import DetectorConfig, DetectorStatus


class DetectorManager:
    """
    Singleton manager for UnifiedDetector instances.

    Provides:
    - Lazy initialization of detectors per camera
    - Configuration updates
    - Lifecycle management
    - Thread-safe operations
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the manager"""
        self._detectors: Dict[int, object] = {}  # camera_id -> UnifiedDetector
        self._configs: Dict[int, DetectorConfig] = {}  # camera_id -> config
        self._last_detection_time: Dict[int, datetime] = {}
        self._start_time = datetime.now()
        self._detector_class = None

    def _get_detector_class(self):
        """Lazy load the UnifiedDetector class"""
        if self._detector_class is None:
            from detectors import UnifiedDetector
            self._detector_class = UnifiedDetector
        return self._detector_class

    def get_or_create_detector(self, config: DetectorConfig) -> object:
        """
        Get existing detector or create new one for camera.

        Args:
            config: DetectorConfig with camera settings

        Returns:
            UnifiedDetector instance
        """
        camera_id = config.camera_id

        with self._lock:
            # Check if detector exists and config matches
            if camera_id in self._detectors:
                existing_config = self._configs.get(camera_id)
                if existing_config and self._configs_equal(existing_config, config):
                    return self._detectors[camera_id]
                # Config changed, update detector
                self._update_detector_config(camera_id, config)
                return self._detectors[camera_id]

            # Create new detector
            detector = self._create_detector(config)
            self._detectors[camera_id] = detector
            self._configs[camera_id] = config
            return detector

    def _configs_equal(self, config1: DetectorConfig, config2: DetectorConfig) -> bool:
        """Check if two configs are equal (for cache invalidation)"""
        # Compare key fields that affect detection
        return (
            config1.detect_cash == config2.detect_cash and
            config1.detect_violence == config2.detect_violence and
            config1.detect_fire == config2.detect_fire and
            config1.cash_confidence == config2.cash_confidence and
            config1.violence_confidence == config2.violence_confidence and
            config1.fire_confidence == config2.fire_confidence and
            config1.cashier_zone_polygon == config2.cashier_zone_polygon and
            config1.cash_drawer_zone_polygon == config2.cash_drawer_zone_polygon
        )

    def _create_detector(self, config: DetectorConfig) -> object:
        """Create a new UnifiedDetector with config"""
        UnifiedDetector = self._get_detector_class()

        detector_config = {
            'models_dir': settings.MODELS_DIR,
            'use_gpu': config.use_gpu or settings.USE_GPU,

            # Model names
            'cash_pose_model': settings.CASH_POSE_MODEL,
            'violence_pose_model': settings.VIOLENCE_POSE_MODEL,
            'fire_yolo_model': settings.FIRE_YOLO_MODEL,
            'fire_model': settings.FIRE_MODEL,

            # Detection toggles
            'detect_cash': config.detect_cash,
            'detect_violence': config.detect_violence,
            'detect_fire': config.detect_fire,

            # Zones
            'cashier_zone_polygon': config.cashier_zone_polygon,
            'cash_drawer_zone_polygon': config.cash_drawer_zone_polygon,
            'cashier_zone_enabled': config.cashier_zone_enabled,
            'cash_drawer_zone_enabled': config.cash_drawer_zone_enabled,

            # Thresholds
            'cash_confidence': config.cash_confidence,
            'violence_confidence': config.violence_confidence,
            'fire_confidence': config.fire_confidence,
            'pose_confidence': config.pose_confidence,

            # Detection parameters
            'hand_touch_distance': config.hand_touch_distance,
            'min_transaction_frames': config.min_transaction_frames,
            'min_violence_frames': config.min_violence_frames,
            'min_fire_frames': config.min_fire_frames,
            'min_fire_area': config.min_fire_area,
        }

        detector = UnifiedDetector(detector_config)
        detector.initialize()

        print(f"[DetectorManager] Created detector for camera {config.camera_id}")
        return detector

    def _update_detector_config(self, camera_id: int, config: DetectorConfig):
        """Update existing detector configuration"""
        detector = self._detectors.get(camera_id)
        if not detector:
            return

        # Update detection toggles
        detector.detect_cash = config.detect_cash
        detector.detect_violence = config.detect_violence
        detector.detect_fire = config.detect_fire

        # Update zone polygons
        if config.cashier_zone_polygon is not None:
            detector.set_cashier_zone_polygon(config.cashier_zone_polygon)
        if config.cash_drawer_zone_polygon is not None:
            detector.set_cash_drawer_zone_polygon(config.cash_drawer_zone_polygon)

        self._configs[camera_id] = config
        print(f"[DetectorManager] Updated config for camera {camera_id}")

    def get_detector(self, camera_id: int) -> Optional[object]:
        """Get detector by camera ID"""
        return self._detectors.get(camera_id)

    def remove_detector(self, camera_id: int) -> bool:
        """Remove detector for camera"""
        with self._lock:
            if camera_id in self._detectors:
                del self._detectors[camera_id]
                if camera_id in self._configs:
                    del self._configs[camera_id]
                if camera_id in self._last_detection_time:
                    del self._last_detection_time[camera_id]
                print(f"[DetectorManager] Removed detector for camera {camera_id}")
                return True
            return False

    def update_zones(self, camera_id: int, cashier_zone: List = None,
                     cash_drawer_zone: List = None) -> bool:
        """Update zone polygons for a detector"""
        detector = self._detectors.get(camera_id)
        if not detector:
            return False

        if cashier_zone is not None:
            detector.set_cashier_zone_polygon(cashier_zone)
        if cash_drawer_zone is not None:
            detector.set_cash_drawer_zone_polygon(cash_drawer_zone)

        return True

    def record_detection(self, camera_id: int):
        """Record that a detection occurred"""
        self._last_detection_time[camera_id] = datetime.now()

    def list_detectors(self) -> Dict[int, dict]:
        """List all active detectors with status"""
        result = {}
        for camera_id, detector in self._detectors.items():
            result[camera_id] = {
                'is_initialized': getattr(detector, 'is_initialized', False),
                'frame_count': getattr(detector, 'frame_count', 0),
            }
        return result

    def get_detector_status(self, camera_id: int) -> Optional[DetectorStatus]:
        """Get detailed status of a detector"""
        detector = self._detectors.get(camera_id)
        if not detector:
            return None

        detection_types = []
        if getattr(detector, 'detect_cash', False):
            detection_types.append("cash")
        if getattr(detector, 'detect_violence', False):
            detection_types.append("violence")
        if getattr(detector, 'detect_fire', False):
            detection_types.append("fire")

        last_time = self._last_detection_time.get(camera_id)

        return DetectorStatus(
            camera_id=camera_id,
            is_initialized=getattr(detector, 'is_initialized', False),
            frame_count=getattr(detector, 'frame_count', 0),
            detection_types=detection_types,
            last_detection_time=last_time.isoformat() if last_time else None,
        )

    def get_status(self) -> dict:
        """Get overall service status"""
        return {
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'active_detectors': len(self._detectors),
            'gpu_info': get_gpu_info(),
        }

    def cleanup_all(self):
        """Cleanup all detectors"""
        with self._lock:
            camera_ids = list(self._detectors.keys())
            for camera_id in camera_ids:
                self.remove_detector(camera_id)
            print("[DetectorManager] Cleaned up all detectors")


# Singleton instance
detector_manager = DetectorManager()
