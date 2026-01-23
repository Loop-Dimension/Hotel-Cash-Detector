"""
Violence Detector

Detects potential violence based on:
- Bounding box overlap between people
- High overlap ratio indicates close physical contact
"""

import numpy as np
from typing import List, Dict, Tuple

from .base_detector import BaseDetector, Detection


class ViolenceDetector(BaseDetector):
    """
    Violence Detection using pose estimation

    Detection criteria:
    - At least 2 people detected
    - Significant bounding box overlap (>35%)
    - Optional: exclude cashier zone (normal transaction area)
    """

    # Model priority for pose detection
    POSE_MODEL_PRIORITY = [
        "yolo26n-pose.pt",
        "yolov8n-pose.pt",
        "yolov8s-pose.pt"
    ]

    def __init__(self, config: Dict = None):
        """
        Initialize Violence Detector

        Config options:
            - overlap_threshold: Minimum overlap ratio (default: 0.35)
            - exclude_cashier_zone: Exclude detections in cashier area (default: True)
            - cashier_zone: Polygon for exclusion
            - cooldown: Frames between detections (default: 300)
            - min_confidence: Minimum detection confidence (default: 0.5)
        """
        super().__init__(config)

        # Detection parameters
        self.overlap_threshold = self.config.get('overlap_threshold', 0.35)
        self.exclude_cashier_zone = self.config.get('exclude_cashier_zone', True)
        self.cashier_zone = self.config.get('cashier_zone', None)

        # Cooldown
        self.last_detection_frame = -300
        self.cooldown = self.config.get('cooldown', 300)  # 10s at 30fps

        # Confidence
        self.min_confidence = self.config.get('min_confidence', 0.5)

        # Logging
        self.enable_logs = self.config.get('enable_logs', True)

    def initialize(self) -> bool:
        """Load YOLO Pose model"""
        self._setup_device()

        self.model = self._load_yolo_model(self.POSE_MODEL_PRIORITY, task='pose')

        if self.model is None:
            print(f"[ViolenceDetector] Failed to load pose model")
            return False

        self.is_initialized = True
        print(f"[ViolenceDetector] Initialized")
        print(f"  - Overlap threshold: {self.overlap_threshold}")
        print(f"  - Cooldown: {self.cooldown} frames")

        return True

    def set_zones(self, cashier_zone: List = None):
        """Update cashier zone for exclusion"""
        if cashier_zone is not None:
            self.cashier_zone = cashier_zone

    def _log(self, message: str):
        """Log message if logging enabled"""
        if self.enable_logs:
            print(message)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run violence detection

        Returns list of Detection objects with event_type='violence'
        """
        detections = []

        if not self.is_initialized:
            return detections

        self.frame_count += 1

        # Check cooldown
        if self.frame_count - self.last_detection_frame < self.cooldown:
            return detections

        # Run pose estimation
        results = self.model(frame, verbose=False, conf=self.pose_confidence)

        if not results or len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()

        # Need at least 2 people
        if len(boxes) < 2:
            return detections

        # Check all pairs for overlap
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                detection = self._check_overlap(frame, boxes[i], boxes[j], i, j)
                if detection:
                    detections.append(detection)
                    self.last_detection_frame = self.frame_count

        return detections

    def _check_overlap(self, frame: np.ndarray,
                       box1: np.ndarray, box2: np.ndarray,
                       idx1: int, idx2: int) -> Detection:
        """
        Check if two bounding boxes overlap significantly

        Returns Detection if violence detected, None otherwise
        """
        # Calculate overlap region
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # No overlap
        if x1 >= x2 or y1 >= y2:
            return None

        # Calculate overlap ratio
        overlap_area = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        min_area = min(box1_area, box2_area)

        if min_area <= 0:
            return None

        overlap_ratio = overlap_area / min_area

        # Check threshold
        if overlap_ratio < self.overlap_threshold:
            return None

        # Calculate combined bounding box
        combined_bbox = (
            int(min(box1[0], box2[0])),
            int(min(box1[1], box2[1])),
            int(max(box1[2], box2[2])),
            int(max(box1[3], box2[3]))
        )

        # Check if in cashier zone (exclude normal transactions)
        if self.exclude_cashier_zone and self.cashier_zone:
            center_x = (combined_bbox[0] + combined_bbox[2]) // 2
            center_y = (combined_bbox[1] + combined_bbox[3]) // 2
            if self.point_in_polygon((center_x, center_y), self.cashier_zone):
                return None

        # Calculate confidence
        confidence = min(0.95, self.min_confidence + overlap_ratio * 0.5)

        self._log(f"\n  [Violence] DETECTED!")
        self._log(f"    - Overlap ratio: {overlap_ratio:.2f}")
        self._log(f"    - Persons: {idx1+1} <-> {idx2+1}")
        self._log(f"    - Confidence: {confidence:.2f}")

        detection = Detection(
            label="VIOLENCE",
            confidence=confidence,
            bbox=combined_bbox,
            metadata={
                'overlap_ratio': round(overlap_ratio, 3),
                'person1_idx': idx1,
                'person2_idx': idx2,
                'box1': list(map(int, box1)),
                'box2': list(map(int, box2))
            }
        )
        detection.event_type = 'violence'

        return detection
