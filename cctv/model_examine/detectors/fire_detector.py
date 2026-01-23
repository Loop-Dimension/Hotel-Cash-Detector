"""
Fire/Smoke Detector

Detects fire and smoke using:
1. Custom trained YOLO fire/smoke model (primary - most accurate)
2. Color-based fire detection with skin exclusion (fallback)
3. Flickering detection (temporal variation - key fire characteristic)
4. Smoke detection via background subtraction

STRICT RULES:
- Skin color regions are EXCLUDED (prevents detecting people as fire)
- Flickering is REQUIRED for color-based detection
- Must sustain over multiple frames (min_fire_frames)
- Higher confidence threshold (0.70)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from collections import deque

from .base_detector import BaseDetector, Detection


class FireDetector(BaseDetector):
    """
    Fire and Smoke Detection with strict false-positive prevention.

    Detection Methods:
    1. YOLO fire/smoke model (primary - most accurate)
    2. Color-based with skin exclusion (fallback)
    3. Flickering analysis (fire characteristic)

    WHAT IS NOT FIRE:
    - Skin-colored regions (people's faces, hands)
    - Static red/orange objects (signs, products)
    - Small fire-colored regions (LEDs, indicators)
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
            - fire_confidence: Minimum confidence (default: 0.70)
            - min_fire_frames: Consecutive frames needed (default: 10)
            - min_fire_area: Minimum pixel area for color detection (default: 3000)
            - cooldown: Frames between detections (default: 60)
            - use_fallback: Try color-based if YOLO unavailable (default: True)
        """
        super().__init__(config)

        # Detection parameters - STRICT to reduce false positives
        self.fire_confidence = self.config.get('fire_confidence', 0.70)
        self.min_fire_frames = self.config.get('min_fire_frames', 10)
        self.min_fire_area = self.config.get('min_fire_area', 3000)

        # Cooldown
        self.last_detection_frame = -60
        self.cooldown = self.config.get('cooldown', 60)  # 2s at 30fps

        # Fallback option
        self.use_fallback = self.config.get('use_fallback', True)
        self.using_fire_model = False

        # Tracking state
        self.consecutive_fire = 0
        self.consecutive_smoke = 0

        # Frame history for flickering detection
        self.fire_mask_history = deque(maxlen=10)
        self._last_flicker = 0.0  # For debug display

        # Color ranges for fire detection (HSV) - STRICT ranges
        # Fire is very bright orange/yellow, not just red
        self.fire_lower1 = np.array([5, 150, 200])     # Bright orange-yellow
        self.fire_upper1 = np.array([25, 255, 255])

        self.fire_lower2 = np.array([0, 200, 220])     # Very bright red
        self.fire_upper2 = np.array([5, 255, 255])

        # Skin color ranges to EXCLUDE (HSV) - prevents detecting people as fire
        self.skin_lower1 = np.array([0, 20, 70])
        self.skin_upper1 = np.array([20, 150, 255])
        self.skin_lower2 = np.array([0, 30, 100])
        self.skin_upper2 = np.array([25, 170, 200])

        # Smoke detection (gray/white)
        self.smoke_lower = np.array([0, 0, 150])
        self.smoke_upper = np.array([180, 30, 255])

        # Background subtractor for smoke detection
        self.bg_subtractor = None

        # Logging
        self.enable_logs = self.config.get('enable_logs', True)

    def initialize(self) -> bool:
        """Load YOLO Fire model"""
        self._setup_device()

        # Try fire-specific model first
        self.model = self._load_yolo_model(self.FIRE_MODEL_PRIORITY)

        if self.model:
            self.using_fire_model = True
            # Get class names from the model
            self.fire_classes = self.model.names
            self.is_initialized = True
            print(f"[FireDetector] Initialized with fire-specific model (STRICT MODE)")
            print(f"  - Confidence threshold: {self.fire_confidence}")
            print(f"  - Min fire frames: {self.min_fire_frames}")
            print(f"  - Cooldown: {self.cooldown} frames")
            return True

        # Fallback to general model
        if self.use_fallback:
            self.model = self._load_yolo_model(self.GENERAL_MODEL_PRIORITY)
            if self.model:
                self.using_fire_model = False
                self.is_initialized = True
                print(f"[FireDetector] Initialized with general YOLO + color fallback (STRICT MODE)")
                return True

        # Initialize background subtractor for color-based fallback
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.is_initialized = True
        print(f"[FireDetector] Initialized with color-based detection only (STRICT MODE)")
        return True

    def _log(self, message: str):
        """Log message if logging enabled"""
        if self.enable_logs:
            print(message)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run fire/smoke detection

        STRICT LOGIC:
        1. YOLO detection with high confidence threshold
        2. Color-based fallback with skin exclusion
        3. Flickering required for color-based
        4. Must sustain over min_fire_frames

        Returns list of Detection objects with event_type='fire'
        """
        detections = []

        if not self.is_initialized:
            return detections

        self.frame_count += 1

        try:
            # Use YOLO fire/smoke model if available (PRIMARY - most accurate)
            if self.model is not None and self.using_fire_model:
                return self._detect_with_yolo(frame)

            # Fallback to color-based detection
            if self.use_fallback:
                return self._detect_with_color(frame)

        except Exception as e:
            if self.enable_logs:
                print(f"[FireDetector] Error: {e}")

        return detections

    def _detect_with_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Detect fire and smoke using trained YOLO model"""
        detections = []

        try:
            # Run YOLO inference with lower initial threshold
            results = self.model(frame, verbose=False, conf=0.25)

            if not results or len(results) == 0:
                self.consecutive_fire = max(0, self.consecutive_fire - 1)
                self._last_flicker = 0.0
                return detections

            result = results[0]
            fire_detected = False
            best_detection = None
            best_confidence = 0.0

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confidences, classes):
                    if self.using_fire_model:
                        class_name = self.fire_classes.get(int(cls), "unknown").lower()
                    else:
                        class_name = self.model.names.get(int(cls), "unknown").lower()

                    # Check if it's fire or smoke
                    if "fire" in class_name or "smoke" in class_name:
                        if conf > best_confidence:
                            best_confidence = float(conf)
                            best_detection = {
                                'bbox': tuple(map(int, box)),
                                'confidence': float(conf),
                                'class': class_name
                            }
                            fire_detected = True

            # Store for debug display
            self._last_flicker = best_confidence if fire_detected else 0.0

            # Track fire detection - only count if meets confidence threshold
            if fire_detected and best_confidence >= self.fire_confidence:
                self.consecutive_fire += 1
            else:
                self.consecutive_fire = max(0, self.consecutive_fire - 1)

            # Generate fire alert after consecutive detections
            # IMPORTANT: Also check that current frame meets confidence threshold
            if (self.consecutive_fire >= self.min_fire_frames and
                self.frame_count - self.last_detection_frame > self.cooldown and
                best_detection is not None and
                best_detection['confidence'] >= self.fire_confidence):

                label = "FIRE" if "fire" in best_detection['class'] else "SMOKE"

                self._log(f"\n  [{label}] DETECTED!")
                self._log(f"    - Class: {best_detection['class']}")
                self._log(f"    - Confidence: {best_detection['confidence']:.2f}")
                self._log(f"    - Consecutive frames: {self.consecutive_fire}")

                detection = Detection(
                    label=label,
                    confidence=best_detection['confidence'],
                    bbox=best_detection['bbox'],
                    metadata={
                        'type': best_detection['class'],
                        'method': 'yolo',
                        'consecutive_frames': self.consecutive_fire
                    }
                )
                detection.event_type = 'fire'
                detections.append(detection)

                self.last_detection_frame = self.frame_count
                self.consecutive_fire = 0

        except Exception as e:
            if self.enable_logs:
                print(f"[FireDetector] YOLO error: {e}")

        return detections

    def _detect_with_color(self, frame: np.ndarray) -> List[Detection]:
        """
        Fallback color-based fire detection with strict filtering

        STRICT REQUIREMENTS:
        1. Fire-colored region must be large enough
        2. Skin-colored regions are EXCLUDED
        3. Flickering is REQUIRED (fire flickers, static objects don't)
        4. Must be bright and saturated
        """
        detections = []

        try:
            # Detect fire-colored regions (excluding skin)
            fire_mask, fire_regions = self._detect_fire_color(frame)
            self.fire_mask_history.append(fire_mask)

            # Calculate flickering score - KEY differentiator
            flicker_score = self._detect_flickering(fire_mask)
            self._last_flicker = flicker_score

            # Analyze fire detections
            fire_detected = False
            best_fire_region = None
            fire_confidence = 0.0

            if fire_regions:
                # Sort by area (larger = more likely to be real fire)
                fire_regions.sort(key=lambda x: x['area'], reverse=True)
                best_fire_region = fire_regions[0]

                # Calculate confidence
                area_score = min(1.0, best_fire_region['area'] / 15000)
                brightness_score = min(1.0, best_fire_region['brightness'] / 255)
                saturation_score = min(1.0, best_fire_region.get('saturation', 100) / 255)

                # FLICKERING IS REQUIRED for color-based detection
                # This is the key differentiator from static red objects
                if flicker_score < 0.15:
                    fire_confidence = 0.0  # No flickering = not fire
                else:
                    fire_confidence = (
                        area_score * 0.15 +
                        flicker_score * 0.60 +  # Flickering is most important
                        brightness_score * 0.15 +
                        saturation_score * 0.10
                    )

                if fire_confidence >= self.fire_confidence:
                    fire_detected = True

            # Track fire detection
            if fire_detected:
                self.consecutive_fire += 1
            else:
                self.consecutive_fire = max(0, self.consecutive_fire - 1)

            # Generate fire alert
            if (self.consecutive_fire >= self.min_fire_frames and
                self.frame_count - self.last_detection_frame > self.cooldown and
                best_fire_region is not None and
                fire_confidence >= self.fire_confidence):

                self._log(f"\n  [FIRE] DETECTED (color-based)!")
                self._log(f"    - Area: {best_fire_region['area']}")
                self._log(f"    - Flicker score: {flicker_score:.2f}")
                self._log(f"    - Confidence: {fire_confidence:.2f}")

                detection = Detection(
                    label="FIRE",
                    confidence=fire_confidence,
                    bbox=best_fire_region['bbox'],
                    metadata={
                        'type': 'fire',
                        'method': 'color',
                        'area': best_fire_region['area'],
                        'flicker_score': round(flicker_score, 2),
                        'consecutive_frames': self.consecutive_fire
                    }
                )
                detection.event_type = 'fire'
                detections.append(detection)

                self.last_detection_frame = self.frame_count
                self.consecutive_fire = 0

            # Also check for smoke
            smoke_detections = self._detect_smoke(frame)
            detections.extend(smoke_detections)

        except Exception as e:
            if self.enable_logs:
                print(f"[FireDetector] Color detection error: {e}")

        return detections

    def _detect_fire_color(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect fire-colored regions in the frame
        EXCLUDES skin-colored regions to avoid false positives

        Returns:
            fire_mask: Binary mask of fire-colored regions
            fire_regions: List of detected fire region info
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create fire color mask (combining orange and bright red ranges)
        mask1 = cv2.inRange(hsv, self.fire_lower1, self.fire_upper1)
        mask2 = cv2.inRange(hsv, self.fire_lower2, self.fire_upper2)
        fire_mask = cv2.bitwise_or(mask1, mask2)

        # Create skin color mask to EXCLUDE
        skin_mask1 = cv2.inRange(hsv, self.skin_lower1, self.skin_upper1)
        skin_mask2 = cv2.inRange(hsv, self.skin_lower2, self.skin_upper2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)

        # Dilate skin mask to be more aggressive in excluding skin
        kernel_skin = np.ones((15, 15), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel_skin, iterations=2)

        # Remove skin regions from fire mask
        fire_mask = cv2.bitwise_and(fire_mask, cv2.bitwise_not(skin_mask))

        # Apply morphological operations to clean up mask
        kernel = np.ones((7, 7), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fire_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_fire_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate region properties
                aspect_ratio = w / h if h > 0 else 0

                # Fire typically has irregular shape, not too flat or too tall
                # Exclude very horizontal regions (could be shelves, signs)
                if 0.3 < aspect_ratio < 4:
                    # Calculate mean color values in the region
                    roi = frame[y:y+h, x:x+w]
                    roi_hsv = hsv[y:y+h, x:x+w]
                    mean_brightness = np.mean(roi)
                    mean_saturation = np.mean(roi_hsv[:, :, 1])

                    # Fire is typically VERY bright and saturated
                    # This helps exclude dim red objects
                    if mean_brightness > 150 and mean_saturation > 100:
                        fire_regions.append({
                            'bbox': (x, y, x + w, y + h),
                            'area': area,
                            'center': (x + w // 2, y + h // 2),
                            'brightness': mean_brightness,
                            'saturation': mean_saturation
                        })

        return fire_mask, fire_regions

    def _detect_flickering(self, current_mask: np.ndarray) -> float:
        """
        Detect flickering patterns typical of fire

        Fire flickers - the bright regions change rapidly
        This is a KEY differentiator from static red/orange objects
        """
        if len(self.fire_mask_history) < 5:
            return 0.0

        # Compare current mask with previous masks
        total_diff = 0
        count = 0
        current_h, current_w = current_mask.shape[:2]

        for prev_mask in list(self.fire_mask_history)[-5:]:
            # Ensure masks are the same size before comparing
            prev_h, prev_w = prev_mask.shape[:2]
            if prev_h != current_h or prev_w != current_w:
                prev_mask = cv2.resize(prev_mask, (current_w, current_h))

            diff = cv2.absdiff(current_mask, prev_mask)
            total_diff += np.sum(diff) / (current_h * current_w)
            count += 1

        # Normalize flickering score
        flicker_score = total_diff / count / 255.0 if count > 0 else 0.0

        # Fire needs SIGNIFICANT flickering - be stricter
        return min(1.0, flicker_score * 3)

    def _detect_smoke(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect smoke using background subtraction and color analysis
        """
        detections = []

        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Look for gray/white moving regions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        smoke_color_mask = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)

        # Combine motion and color
        smoke_mask = cv2.bitwise_and(fg_mask, smoke_color_mask)

        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Smoke regions tend to be larger
            if area > self.min_fire_area * 2:
                x, y, w, h = cv2.boundingRect(contour)

                # Smoke confidence based on area
                smoke_confidence = min(1.0, area / 20000) * 0.7

                # Only alert for very confident smoke detection
                if smoke_confidence > 0.6 and self.frame_count - self.last_detection_frame > self.cooldown:
                    self._log(f"\n  [SMOKE] DETECTED!")
                    self._log(f"    - Area: {area}")
                    self._log(f"    - Confidence: {smoke_confidence:.2f}")

                    detection = Detection(
                        label="SMOKE",
                        confidence=smoke_confidence,
                        bbox=(x, y, x + w, y + h),
                        metadata={
                            'type': 'smoke',
                            'method': 'background_subtraction',
                            'area': area
                        }
                    )
                    detection.event_type = 'fire'
                    detections.append(detection)

                    self.last_detection_frame = self.frame_count
                    break  # Only one smoke detection per frame

        return detections

    def get_flicker_score(self) -> float:
        """Get last flicker score for debug display"""
        return self._last_flicker
