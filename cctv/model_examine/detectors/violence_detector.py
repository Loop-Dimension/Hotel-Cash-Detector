"""
Violence Detector

Detects violent actions using:
1. Physical contact between TWO or more people
2. Rapid aggressive movements while in close proximity
3. Sustained over multiple frames

STRICT RULES:
- Single person actions are NEVER violence
- Requires TWO people very close together
- Requires sustained aggressive motion between them
- Normal walking, waving, reaching = NOT violence
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque

from .base_detector import BaseDetector, Detection


class ViolenceDetector(BaseDetector):
    """
    Violence Detection using pose estimation and motion analysis.

    STRICT DETECTION CRITERIA:
    1. Minimum 2 people required
    2. People must be in close contact (overlapping bboxes >= 20%)
    3. HIGH motion from BOTH people simultaneously
    4. Sustained over many frames (min_violence_frames)

    WHAT IS NOT VIOLENCE:
    - One person raising arms (waving, stretching)
    - One person moving fast (running, exercising)
    - People standing close but not moving aggressively
    - Normal customer-cashier interactions
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
            - overlap_threshold: Minimum overlap ratio (default: 0.20)
            - motion_threshold: Minimum motion for BOTH people (default: 100)
            - min_violence_frames: Consecutive frames needed (default: 15)
            - violence_confidence: Final confidence threshold (default: 0.70)
            - exclude_cashier_zone: Exclude detections in cashier area (default: True)
            - cashier_zone: Polygon for exclusion
            - cooldown: Frames between detections (default: 150)
        """
        super().__init__(config)

        # Detection parameters - strict to reduce false positives
        self.overlap_threshold = self.config.get('overlap_threshold', 0.20)
        self.motion_threshold = self.config.get('motion_threshold', 100)
        self.min_violence_frames = self.config.get('min_violence_frames', 15)
        self.violence_confidence = self.config.get('violence_confidence', 0.70)

        self.exclude_cashier_zone = self.config.get('exclude_cashier_zone', True)
        self.cashier_zone = self.config.get('cashier_zone', None)

        # Cooldown
        self.last_detection_frame = -150
        self.cooldown = self.config.get('cooldown', 150)

        # Tracking state
        self.previous_keypoints = {}
        self.previous_positions = {}
        self.consecutive_violence = 0
        self.person_motion_history = {}  # person_id -> deque of motion values

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
        print(f"[ViolenceDetector] Initialized (STRICT MODE)")
        print(f"  - Overlap threshold: {self.overlap_threshold}")
        print(f"  - Motion threshold: {self.motion_threshold}")
        print(f"  - Min violence frames: {self.min_violence_frames}")
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

    def calculate_motion(self, current_kpts: np.ndarray, previous_kpts: np.ndarray) -> float:
        """Calculate average motion between keypoint sets"""
        if current_kpts is None or previous_kpts is None:
            return 0.0

        if len(current_kpts) != len(previous_kpts):
            return 0.0

        total_motion = 0.0
        valid_points = 0

        for curr, prev in zip(current_kpts, previous_kpts):
            if len(curr) >= 3 and len(prev) >= 3:
                if curr[2] > 0.3 and prev[2] > 0.3:  # Both points visible
                    motion = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                    total_motion += motion
                    valid_points += 1

        return total_motion / valid_points if valid_points > 0 else 0.0

    def check_bbox_overlap(self, box1: Tuple, box2: Tuple) -> float:
        """Check how much two bounding boxes overlap (0-1 ratio)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        min_area = min(area1, area2)
        if min_area <= 0:
            return 0.0

        return intersection / min_area

    def is_in_cashier_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if bounding box center is inside cashier zone"""
        if self.cashier_zone is None:
            return False

        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        return self.point_in_polygon((center_x, center_y), self.cashier_zone)

    def detect_physical_altercation(self, people: List[Dict]) -> List[Dict]:
        """
        Detect physical fighting between two people

        STRICT REQUIREMENTS:
        1. Two people with OVERLAPPING bounding boxes (physically close)
        2. BOTH people have high motion (not just one person moving)
        3. Not in cashier zone (avoid false positives from transactions)
        """
        altercations = []

        if len(people) < 2:
            return altercations

        for i, person1 in enumerate(people):
            for j, person2 in enumerate(people):
                if i >= j:
                    continue

                # Skip if either person is in cashier zone
                if person1.get('in_cashier_zone') or person2.get('in_cashier_zone'):
                    continue

                box1 = person1['bbox']
                box2 = person2['bbox']

                # Check for significant overlap (people physically touching)
                overlap = self.check_bbox_overlap(box1, box2)

                # Require minimum overlap
                if overlap < self.overlap_threshold:
                    continue

                # Get motion for both people
                motion1 = person1.get('avg_motion', 0)
                motion2 = person2.get('avg_motion', 0)

                # BOTH must be moving aggressively (not just one attacking)
                if motion1 < self.motion_threshold or motion2 < self.motion_threshold:
                    continue

                # Calculate violence confidence based on overlap and motion
                motion_score = min(1.0, (motion1 + motion2) / (self.motion_threshold * 4))
                overlap_score = min(1.0, overlap * 2)

                confidence = (motion_score * 0.6) + (overlap_score * 0.4)

                if confidence >= self.violence_confidence:
                    combined_bbox = (
                        int(min(box1[0], box2[0])),
                        int(min(box1[1], box2[1])),
                        int(max(box1[2], box2[2])),
                        int(max(box1[3], box2[3]))
                    )

                    altercations.append({
                        'person1': i,
                        'person2': j,
                        'overlap': overlap,
                        'motion1': motion1,
                        'motion2': motion2,
                        'confidence': confidence,
                        'bbox': combined_bbox
                    })

        return altercations

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect violence in the frame

        STRICT LOGIC - Only triggers on actual physical fighting:
        1. Must have 2+ people with overlapping bounding boxes
        2. BOTH people must have high motion (fighting involves both parties)
        3. Must be sustained over many consecutive frames
        4. Excludes cashier zone interactions
        """
        detections = []

        if not self.is_initialized:
            return detections

        self.frame_count += 1

        try:
            # Run pose estimation
            results = self.model(frame, verbose=False, conf=self.pose_confidence)

            if not results or len(results) == 0:
                self.consecutive_violence = max(0, self.consecutive_violence - 2)
                return detections

            result = results[0]
            people = []

            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()

                for idx, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                    bbox = tuple(map(int, box))
                    person_id = f"person_{idx}"

                    # Calculate motion from previous frame
                    current_motion = 0.0
                    if person_id in self.previous_keypoints:
                        current_motion = self.calculate_motion(kpts, self.previous_keypoints[person_id])
                    self.previous_keypoints[person_id] = kpts.copy()

                    # Track motion history for this person (average over last 5 frames)
                    if person_id not in self.person_motion_history:
                        self.person_motion_history[person_id] = deque(maxlen=5)
                    self.person_motion_history[person_id].append(current_motion)

                    avg_motion = np.mean(self.person_motion_history[person_id]) if self.person_motion_history[person_id] else 0

                    # Check if in cashier zone
                    in_cashier = self.is_in_cashier_zone(bbox)

                    person_info = {
                        'idx': idx,
                        'bbox': bbox,
                        'keypoints': kpts,
                        'current_motion': current_motion,
                        'avg_motion': avg_motion,
                        'in_cashier_zone': in_cashier
                    }
                    people.append(person_info)

            # Clean up old person tracking
            current_ids = {f"person_{p['idx']}" for p in people}
            old_ids = set(self.person_motion_history.keys()) - current_ids
            for old_id in old_ids:
                del self.person_motion_history[old_id]
                if old_id in self.previous_keypoints:
                    del self.previous_keypoints[old_id]

            # Detect physical altercations (two people fighting)
            altercations = self.detect_physical_altercation(people)

            # Update consecutive violence counter
            if altercations:
                self.consecutive_violence += 1
            else:
                # Decay faster when no violence detected
                self.consecutive_violence = max(0, self.consecutive_violence - 2)

            # Generate detection only after sustained violence over many frames
            if (self.consecutive_violence >= self.min_violence_frames and
                self.frame_count - self.last_detection_frame > self.cooldown and
                len(altercations) > 0):

                # Find the highest confidence altercation
                best = max(altercations, key=lambda x: x['confidence'])

                self._log(f"\n  [Violence] DETECTED! (sustained {self.consecutive_violence} frames)")
                self._log(f"    - Overlap: {best['overlap']:.2f}")
                self._log(f"    - Motion1: {best['motion1']:.1f}, Motion2: {best['motion2']:.1f}")
                self._log(f"    - Confidence: {best['confidence']:.2f}")

                detection = Detection(
                    label="VIOLENCE",
                    confidence=best['confidence'],
                    bbox=best['bbox'],
                    metadata={
                        'type': 'physical_altercation',
                        'overlap': round(best['overlap'], 3),
                        'motion1': round(best['motion1'], 1),
                        'motion2': round(best['motion2'], 1),
                        'people_count': len(people),
                        'consecutive_frames': self.consecutive_violence
                    }
                )
                detection.event_type = 'violence'
                detections.append(detection)

                self.last_detection_frame = self.frame_count
                self.consecutive_violence = 0

        except Exception as e:
            self._log(f"[Violence] Detection error: {e}")

        return detections
