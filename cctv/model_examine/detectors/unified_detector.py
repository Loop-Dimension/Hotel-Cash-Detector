"""
Unified Detector

Combines all detectors (Cash, Violence, Fire) into a single interface.
Handles:
- Detector initialization
- Zone management
- Detection routing
- Overlay drawing
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .base_detector import Detection, BaseDetector
from .cash_detector import CashDetector
from .violence_detector import ViolenceDetector
from .fire_detector import FireDetector
try:
    from .gemini_validator import GeminiValidator
except Exception:
    GeminiValidator = None


class UnifiedDetector:
    """
    Unified Detection Interface

    Combines CashDetector, ViolenceDetector, and FireDetector
    into a single easy-to-use interface.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Unified Detector

        Config options:
            - models_dir: Path to models directory
            - device: 'cuda', 'cpu', or 'auto'

            - detect_cash: Enable cash detection (default: True)
            - detect_violence: Enable violence detection (default: True)
            - detect_fire: Enable fire detection (default: True)

            - cashier_zone: Polygon for cashier area
            - cash_drawer_zone: Polygon for cash drawer

            - hand_touch_distance: Pixel threshold for hand proximity
            - detection_mode: 'simple' or 'strict'

            - enable_logs: Enable detection logs (default: True)
        """
        self.config = config or {}

        # Detection enable flags
        self.detect_cash = self.config.get('detect_cash', True)
        self.detect_violence = self.config.get('detect_violence', True)
        self.detect_fire = self.config.get('detect_fire', True)

        # Zone configurations
        self.cashier_zone = self.config.get('cashier_zone', None)
        self.cash_drawer_zone = self.config.get('cash_drawer_zone', None)

        # Detectors
        self.cash_detector: Optional[CashDetector] = None
        self.violence_detector: Optional[ViolenceDetector] = None
        self.fire_detector: Optional[FireDetector] = None

        # Gemini validation (optional)
        self.gemini_validator = self.config.get('gemini_validator')
        if not self.gemini_validator and self.config.get('gemini_enabled') and GeminiValidator:
            try:
                self.gemini_validator = GeminiValidator(api_key=self.config.get('gemini_api_key', ''))
                custom_prompts = self.config.get('gemini_prompts')
                if custom_prompts:
                    self.gemini_validator.set_custom_prompts(custom_prompts)
            except Exception:
                self.gemini_validator = None

        # Initialization state
        self.is_initialized = False

        # Frame counter
        self.frame_count = 0

        # Last pose result for overlay drawing
        self._last_pose_result = None
        self._last_people = []

    def initialize(self) -> bool:
        """Initialize all enabled detectors"""
        print(f"\n{'='*50}")
        print("Unified Detector Initialization")
        print(f"{'='*50}")

        success = True

        # Initialize Cash Detector
        if self.detect_cash:
            cash_config = {
                'models_dir': self.config.get('models_dir', './models'),
                'device': self.config.get('device', 'auto'),
                'cashier_zone': self.cashier_zone,
                'cash_drawer_zone': self.cash_drawer_zone,
                'hand_touch_distance': self.config.get('hand_touch_distance', 100),
                'detection_mode': self.config.get('detection_mode', 'strict'),
                'hand_tracking_duration': self.config.get('hand_tracking_duration', 600),
                'enable_logs': self.config.get('enable_logs', True)
            }
            if self.config.get('potential_cash_cooldown') is not None:
                cash_config['potential_cash_cooldown'] = self.config.get('potential_cash_cooldown')
            if self.config.get('transaction_cooldown') is not None:
                cash_config['transaction_cooldown'] = self.config.get('transaction_cooldown')
            if self.config.get('cash_object_cooldown') is not None:
                cash_config['cash_object_cooldown'] = self.config.get('cash_object_cooldown')
            if self.config.get('cash_object_confidence') is not None:
                cash_config['cash_object_confidence'] = self.config.get('cash_object_confidence')
            self.cash_detector = CashDetector(cash_config)
            if not self.cash_detector.initialize():
                print("[UnifiedDetector] Cash detector failed to initialize")
                success = False

        # Initialize Violence Detector
        if self.detect_violence:
            violence_config = {
                'models_dir': self.config.get('models_dir', './models'),
                'device': self.config.get('device', 'auto'),
                'cashier_zone': self.cashier_zone,
                'exclude_cashier_zone': True,
                'enable_logs': self.config.get('enable_logs', True)
            }
            if self.config.get('violence_cooldown') is not None:
                violence_config['cooldown'] = self.config.get('violence_cooldown')
            self.violence_detector = ViolenceDetector(violence_config)
            if not self.violence_detector.initialize():
                print("[UnifiedDetector] Violence detector failed to initialize")
                success = False

        # Initialize Fire Detector
        if self.detect_fire:
            fire_config = {
                'models_dir': self.config.get('models_dir', './models'),
                'device': self.config.get('device', 'auto'),
                'enable_logs': self.config.get('enable_logs', True)
            }
            if self.config.get('fire_cooldown') is not None:
                fire_config['cooldown'] = self.config.get('fire_cooldown')
            self.fire_detector = FireDetector(fire_config)
            if not self.fire_detector.initialize():
                print("[UnifiedDetector] Fire detector failed to initialize")
                success = False

        self.is_initialized = success or (
            (self.cash_detector and self.cash_detector.is_initialized) or
            (self.violence_detector and self.violence_detector.is_initialized) or
            (self.fire_detector and self.fire_detector.is_initialized)
        )

        print(f"{'='*50}")
        print(f"Initialization: {'SUCCESS' if self.is_initialized else 'FAILED'}")
        print(f"  - Cash: {self.cash_detector.is_initialized if self.cash_detector else 'disabled'}")
        print(f"  - Violence: {self.violence_detector.is_initialized if self.violence_detector else 'disabled'}")
        print(f"  - Fire: {self.fire_detector.is_initialized if self.fire_detector else 'disabled'}")
        print(f"{'='*50}\n")

        return self.is_initialized

    def set_zones(self, cashier_zone: List = None, cash_drawer_zone: List = None):
        """Update zone configurations for all detectors"""
        if cashier_zone is not None:
            self.cashier_zone = cashier_zone
        if cash_drawer_zone is not None:
            self.cash_drawer_zone = cash_drawer_zone

        # Update detectors
        if self.cash_detector:
            self.cash_detector.set_zones(cashier_zone, cash_drawer_zone)
        if self.violence_detector:
            self.violence_detector.set_zones(cashier_zone)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run all enabled detections on frame

        Returns combined list of all detections
        """
        detections = []

        if not self.is_initialized:
            return detections

        self.frame_count += 1

        # Cash detection
        if self.detect_cash and self.cash_detector and self.cash_detector.is_initialized:
            cash_dets = self.cash_detector.detect(frame)
            detections.extend(cash_dets)

            # Store pose result for overlay
            if self.cash_detector.model:
                try:
                    results = self.cash_detector.model(frame, verbose=False, conf=0.3)
                    if results and len(results) > 0:
                        self._last_pose_result = results[0]
                        self._last_people = self.cash_detector.extract_people(
                            results[0], self.cashier_zone
                        )
                except:
                    pass

        # Violence detection
        if self.detect_violence and self.violence_detector and self.violence_detector.is_initialized:
            violence_dets = self.violence_detector.detect(frame)
            detections.extend(violence_dets)

        # Fire detection
        if self.detect_fire and self.fire_detector and self.fire_detector.is_initialized:
            fire_dets = self.fire_detector.detect(frame)
            detections.extend(fire_dets)

        return detections

    def draw_overlay(self, frame: np.ndarray, detections: List[Detection] = None,
                     overlay_mode: str = "full") -> np.ndarray:
        """
        Draw detection overlay on frame

        Args:
            frame: Input frame
            detections: Optional list of detections to draw
            overlay_mode: "full" or "minimal"

        Returns:
            Frame with overlay drawn
        """
        overlay = frame.copy()

        # Draw Cashier Zone
        if self.cashier_zone and len(self.cashier_zone) >= 3:
            pts = np.array(self.cashier_zone, dtype=np.int32).reshape((-1, 1, 2))
            temp = overlay.copy()
            cv2.fillPoly(temp, [pts], (255, 200, 0))
            cv2.addWeighted(temp, 0.15, overlay, 0.85, 0, overlay)
            cv2.polylines(overlay, [pts], True, (255, 200, 0), 2)
            if overlay_mode != "minimal":
                zx, zy = self.cashier_zone[0]
                cv2.putText(overlay, "Cashier Zone", (int(zx), int(zy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        # Draw Cash Drawer Zone
        if self.cash_drawer_zone and len(self.cash_drawer_zone) >= 3:
            pts = np.array(self.cash_drawer_zone, dtype=np.int32).reshape((-1, 1, 2))
            temp = overlay.copy()
            cv2.fillPoly(temp, [pts], (0, 255, 0))
            cv2.addWeighted(temp, 0.2, overlay, 0.8, 0, overlay)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            if overlay_mode != "minimal":
                dx, dy = self.cash_drawer_zone[0]
                cv2.putText(overlay, "Cash Drawer", (int(dx), int(dy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw skeleton and people
        if overlay_mode != "minimal":
            self._draw_skeletons(overlay)
        else:
            self._draw_hand_positions(overlay)

        # Draw tracking status
        if overlay_mode != "minimal" and self.cash_detector:
            status = self.cash_detector.get_tracking_status()
            if status['tracking']:
                text = f"Track2: {status['frames_since_touch']}/{status['max_frames']}"
                cv2.putText(overlay, text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw detections
        if overlay_mode != "minimal" and detections:
            colors = {
                'CASH': (0, 255, 255),      # Yellow
                'VIOLENCE': (0, 0, 255),     # Red
                'FIRE': (0, 165, 255),       # Orange
                'SMOKE': (128, 128, 128)     # Gray
            }

            for det in detections:
                x1, y1, x2, y2 = det.bbox
                color = colors.get(det.label, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

                # Draw label
                event_type = det.event_type or det.label.lower()
                label = f"{det.label} ({event_type}) {det.confidence:.2f}"

                # Label background
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                cv2.putText(overlay, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw low-confidence cash object candidates (debug overlay)
        if overlay_mode != "minimal" and self.cash_detector:
            debug_dets = self.cash_detector.get_debug_detections()
            if debug_dets:
                threshold = getattr(self.cash_detector, 'cash_object_confidence', 0.7)
                for det in debug_dets:
                    conf = det.get('confidence', 0.0)
                    if conf < 0.01:
                        continue
                    x1, y1, x2, y2 = det.get('bbox', (0, 0, 0, 0))
                    color = (255, 255, 0) if conf < threshold else (0, 255, 255)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
                    label = f"CASH_OBJ {conf*100:.1f}%"
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(overlay, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), color, -1)
                    cv2.putText(overlay, label, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return overlay

    def validate_frame(self, frame: np.ndarray, event_type: str):
        """Validate a single frame with Gemini (if enabled)."""
        if not self.gemini_validator:
            return True, 1.0, "Validation disabled", event_type
        return self.gemini_validator.validate_event(frame, event_type)

    def _draw_skeletons(self, overlay: np.ndarray):
        """Draw keypoints and role labels for all detected people"""
        if not self._last_people:
            return

        # Colors: Cashier = Cyan, Customer = Magenta
        CASHIER_COLOR = (255, 255, 0)  # Cyan (BGR)
        CUSTOMER_COLOR = (255, 0, 255)  # Magenta (BGR)
        HAND_COLOR = (0, 255, 0)  # Green for hands

        for person in self._last_people:
            keypoints = person.get('keypoints')
            bbox = person.get('bbox')
            in_zone = person.get('in_cashier_zone')

            # Determine color based on role
            if in_zone is True:
                color = CASHIER_COLOR
                role_text = "CASHIER"
            elif in_zone is False:
                color = CUSTOMER_COLOR
                role_text = "CUSTOMER"
            else:
                color = (200, 200, 200)  # Gray for unknown
                role_text = "PERSON"

            # Draw keypoints only (no skeleton lines)
            if keypoints is not None and len(keypoints) >= 17:
                for i, kp in enumerate(keypoints):
                    if len(kp) >= 3 and kp[2] > 0.3:
                        x, y = int(kp[0]), int(kp[1])
                        # Hands (wrists) - larger green circles
                        if i in [9, 10]:  # LEFT_WRIST, RIGHT_WRIST
                            cv2.circle(overlay, (x, y), 8, HAND_COLOR, -1)
                            cv2.circle(overlay, (x, y), 10, (255, 255, 255), 2)
                        else:
                            # Other keypoints - small dots
                            cv2.circle(overlay, (x, y), 3, color, -1)

            # Draw bounding box with role
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                # Role label
                label = f"{role_text} #{person.get('idx', 0)+1}"
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(overlay, (x1, y2), (x1 + text_w + 6, y2 + text_h + 6), color, -1)
                cv2.putText(overlay, label, (x1 + 3, y2 + text_h + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw hand position labels (L/R)
            hands = person.get('hands', {})
            for hand_name, hand_pos in hands.items():
                hx, hy = int(hand_pos[0]), int(hand_pos[1])
                hand_label = "L" if hand_name == 'left' else "R"
                cv2.putText(overlay, hand_label, (hx + 12, hy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_hand_positions(self, overlay: np.ndarray):
        """Draw hand positions only (no skeletons/bboxes)."""
        if not self._last_people:
            return

        HAND_COLOR = (0, 255, 0)  # Green for hands (BGR)

        for person in self._last_people:
            hands = person.get('hands', {})
            for hand_name, hand_pos in hands.items():
                hx, hy = int(hand_pos[0]), int(hand_pos[1])
                cv2.circle(overlay, (hx, hy), 6, HAND_COLOR, -1)
                cv2.circle(overlay, (hx, hy), 8, (255, 255, 255), 1)
                hand_label = "L" if hand_name == 'left' else "R"
                cv2.putText(overlay, hand_label, (hx + 10, hy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def get_status(self) -> Dict:
        """Get current detector status"""
        return {
            'initialized': self.is_initialized,
            'frame_count': self.frame_count,
            'gemini_enabled': bool(self.gemini_validator),
            'cash': {
                'enabled': self.detect_cash,
                'initialized': self.cash_detector.is_initialized if self.cash_detector else False,
                'tracking': self.cash_detector.get_tracking_status() if self.cash_detector else None
            },
            'violence': {
                'enabled': self.detect_violence,
                'initialized': self.violence_detector.is_initialized if self.violence_detector else False
            },
            'fire': {
                'enabled': self.detect_fire,
                'initialized': self.fire_detector.is_initialized if self.fire_detector else False
            },
            'zones': {
                'cashier_zone': len(self.cashier_zone) if self.cashier_zone else 0,
                'cash_drawer_zone': len(self.cash_drawer_zone) if self.cash_drawer_zone else 0
            }
        }

    def toggle_detection(self, detection_type: str, enabled: bool = None):
        """Toggle detection type on/off"""
        if detection_type == 'cash':
            self.detect_cash = enabled if enabled is not None else not self.detect_cash
            return self.detect_cash
        elif detection_type == 'violence':
            self.detect_violence = enabled if enabled is not None else not self.detect_violence
            return self.detect_violence
        elif detection_type == 'fire':
            self.detect_fire = enabled if enabled is not None else not self.detect_fire
            return self.detect_fire
        return None
