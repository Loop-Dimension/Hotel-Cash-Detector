"""
Fire/Smoke Detector

Detects fire and smoke using:
- Custom trained YOLO fire/smoke model (primary)
- General YOLO with fire class fallback
"""

import numpy as np
from typing import List, Dict

from .base_detector import BaseDetector, Detection


class FireDetector(BaseDetector):
    """
    Fire and Smoke Detection

    Uses custom fire_smoke_yolov8.pt model for detection.
    Falls back to general YOLO model if custom model not available.
    """

    # Model priority for fire detection
    FIRE_MODEL_PRIORITY = [
        "fire_smoke_yolov8.pt",
        "fire_yolov8.pt"
    ]

    # Fallback general models
    GENERAL_MODEL_PRIORITY = [
        "yolo26n.pt",
        "yolov8n.pt",
        "yolov8s.pt"
    ]

    # Fire model class mapping
    FIRE_CLASSES = {
        0: 'fire',
        1: 'default',
        2: 'smoke'
    }

    def __init__(self, config: Dict = None):
        """
        Initialize Fire Detector

        Config options:
            - fire_confidence: Minimum confidence for fire (default: 0.4)
            - cooldown: Frames between detections (default: 300)
            - use_fallback: Try general model if fire model unavailable (default: True)
        """
        super().__init__(config)

        # Detection parameters
        self.fire_confidence = self.config.get('fire_confidence', 0.4)

        # Cooldown
        self.last_detection_frame = -300
        self.cooldown = self.config.get('cooldown', 300)  # 10s at 30fps

        # Fallback option
        self.use_fallback = self.config.get('use_fallback', True)
        self.using_fire_model = False

        # Logging
        self.enable_logs = self.config.get('enable_logs', True)

    def initialize(self) -> bool:
        """Load YOLO Fire model"""
        self._setup_device()

        # Try fire-specific model first
        self.model = self._load_yolo_model(self.FIRE_MODEL_PRIORITY)

        if self.model:
            self.using_fire_model = True
            self.is_initialized = True
            print(f"[FireDetector] Initialized with fire-specific model")
            return True

        # Fallback to general model
        if self.use_fallback:
            self.model = self._load_yolo_model(self.GENERAL_MODEL_PRIORITY)
            if self.model:
                self.using_fire_model = False
                self.is_initialized = True
                print(f"[FireDetector] Initialized with general YOLO model (fallback)")
                return True

        print(f"[FireDetector] Failed to load any model")
        return False

    def _log(self, message: str):
        """Log message if logging enabled"""
        if self.enable_logs:
            print(message)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run fire/smoke detection

        Returns list of Detection objects with event_type='fire'
        """
        detections = []

        if not self.is_initialized:
            return detections

        self.frame_count += 1

        # Check cooldown
        if self.frame_count - self.last_detection_frame < self.cooldown:
            return detections

        try:
            # Run detection
            results = self.model(frame, verbose=False, conf=self.fire_confidence)

            if not results or len(results) == 0:
                return detections

            result = results[0]

            if result.boxes is None:
                return detections

            # Process detections
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                # Get class name
                if self.using_fire_model:
                    class_name = self.FIRE_CLASSES.get(cls, 'unknown').lower()
                else:
                    class_name = self.model.names.get(cls, 'unknown').lower()

                # Check if fire or smoke
                is_fire = 'fire' in class_name
                is_smoke = 'smoke' in class_name

                if not (is_fire or is_smoke):
                    continue

                # Create detection
                label = "FIRE" if is_fire else "SMOKE"

                self._log(f"\n  [{label}] DETECTED!")
                self._log(f"    - Class: {class_name}")
                self._log(f"    - Confidence: {conf:.2f}")

                detection = Detection(
                    label=label,
                    confidence=conf,
                    bbox=tuple(map(int, xyxy)),
                    metadata={
                        'class': class_name,
                        'class_id': cls,
                        'using_fire_model': self.using_fire_model
                    }
                )
                detection.event_type = 'fire'
                detections.append(detection)

                self.last_detection_frame = self.frame_count

        except Exception as e:
            if self.enable_logs:
                print(f"[FireDetector] Error: {e}")

        return detections
